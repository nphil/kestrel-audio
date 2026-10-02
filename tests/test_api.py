"""HTTP contract: auth, job submission, preview and info states, Range, delete, the LAN-only key."""
import pytest
from fastapi.testclient import TestClient

from kestrel_audio import config
from kestrel_audio.server import create_app, is_lan, job_info, load_or_create_key
from kestrel_audio.store import Store

KEY = "k" * 64
H = {"X-Kestrel-Audio-Key": KEY}


class FakeManager:
    def __init__(self):
        self.kicked = 0

    def kick(self):
        self.kicked += 1

    def stats(self):
        return {"service": "kestrel-audio", "queue": {"depth": 0}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_AUDIO_DATA", str(tmp_path))
    monkeypatch.setenv("KESTREL_AUDIO_ASSETS", str(tmp_path / "assets"))
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "icon-512.png").write_bytes(b"\x89PNGfake")
    cfg = config.load()
    store = Store(cfg.data_dir, cfg.inbox_dir, cfg.previews_dir, cfg.db_path)
    mgr = FakeManager()
    client = TestClient(create_app(cfg, store, mgr, KEY))
    yield client, store, mgr
    store.close()


def post(client, det=1, **extra):
    q = {"detection_id": det, "species": "Blue Jay", "scientific": "Cyanocitta cristata", **extra}
    return client.post("/v1/jobs", params=q, content=b"OggS-clip", headers={**H, "Content-Type": "audio/ogg"})


def finish(store, det, text=b"m4a-bytes" * 500):
    store.claim_next()
    store.preview_path(det).write_bytes(text)
    store.finish(det, {"segment": {"start": 3.0, "end": 9.0, "source": "perch"}, "method": "gate", "cleaned": True,
                       "scores": {"original": 0.5, "preview": 0.6}, "loudnessLufs": -16.1, "durationS": 6.0, "variant": "G20",
                       "device": "cuda", "notes": []}, nbytes=len(text), cleaned=True, took_s=2.5, device="cuda")


def test_open_endpoints_need_no_key(env):
    client, _, _ = env
    assert client.get("/healthz").json()["ok"] is True
    assert client.get("/icon-512.png").status_code == 200
    assert client.get("/api/status").status_code == 200


def test_everything_under_v1_needs_the_key(env):
    client, _, _ = env
    for method, url in (("get", "/v1/stats"), ("post", "/v1/jobs?detection_id=1&species=x"), ("get", "/v1/previews/1"),
                        ("get", "/v1/previews/1/info"), ("delete", "/v1/previews/1")):
        assert getattr(client, method)(url).status_code == 401
        assert getattr(client, method)(url, headers={"X-Kestrel-Audio-Key": "wrong"}).status_code == 401


def test_submit_validates_and_queues(env):
    client, _, mgr = env
    assert client.post("/v1/jobs?species=x", content=b"a", headers=H).status_code == 400
    assert client.post("/v1/jobs?detection_id=1", content=b"a", headers=H).status_code == 400
    assert client.post("/v1/jobs?detection_id=1&species=x", content=b"", headers=H).status_code == 400
    r = post(env[0], 5)
    assert r.status_code == 202
    body = r.json()
    assert body["detectionId"] == 5 and body["state"] == "pending" and body["queuePosition"] == 1
    assert mgr.kicked == 1


def test_posting_again_is_a_no_op_and_ready_jobs_answer_200(env):
    client, store, mgr = env
    post(client, 5)
    assert post(client, 5).status_code == 202 and mgr.kicked == 1
    finish(store, 5)
    r = post(client, 5)
    assert r.status_code == 200 and r.json()["state"] == "ready"
    assert post(client, 5, force="1").status_code == 202


def test_body_size_limit(env, monkeypatch):
    client, _, _ = env
    big = b"x" * (21 * 1024 * 1024)
    assert client.post("/v1/jobs?detection_id=1&species=x", content=big, headers=H).status_code == 413


def test_preview_states(env):
    client, store, _ = env
    assert client.get("/v1/previews/9", headers=H).status_code == 404
    assert client.get("/v1/previews/9", headers=H).json() == {"state": "none"}
    post(client, 9)
    r = client.get("/v1/previews/9", headers=H)
    assert r.status_code == 202 and r.json() == {"state": "pending"} and r.headers["retry-after"] == "5"
    store.claim_next()
    store.fail(9, "could not decode audio")
    r = client.get("/v1/previews/9", headers=H)
    assert r.status_code == 404 and r.json()["state"] == "failed" and "decode" in r.json()["error"]


def test_ready_preview_is_audio_with_range_and_caching(env):
    client, store, _ = env
    post(client, 3)
    finish(store, 3)
    r = client.get("/v1/previews/3", headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mp4"
    assert "immutable" in r.headers["cache-control"] and r.headers["accept-ranges"] == "bytes" and r.headers.get("etag")
    part = client.get("/v1/previews/3", headers={**H, "Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.headers["content-range"].startswith("bytes 0-99/")
    assert client.get("/v1/previews/3", headers={**H, "If-None-Match": r.headers["etag"]}).status_code == 304


def test_info_always_has_every_field(env):
    client, store, _ = env
    assert client.get("/v1/previews/4/info", headers=H).status_code == 404
    post(client, 4)
    pending = client.get("/v1/previews/4/info", headers=H).json()
    fields = {"detectionId", "state", "segment", "method", "cleaned", "scores", "loudnessLufs", "durationS", "createdAt", "readyAt",
              "queuePosition", "error"}
    assert fields <= set(pending)
    assert pending["state"] == "pending" and pending["segment"] is None and pending["cleaned"] is False and pending["error"] is None
    finish(store, 4)
    ready = client.get("/v1/previews/4/info", headers=H).json()
    assert ready["state"] == "ready" and ready["segment"]["source"] == "perch" and ready["method"] == "gate" and ready["cleaned"] is True
    assert ready["scores"] == {"original": 0.5, "preview": 0.6} and ready["queuePosition"] is None and ready["readyAt"]
    post(client, 6)
    store.claim_next()
    store.fail(6, "the clip is silent")
    failed = client.get("/v1/previews/6/info", headers=H).json()
    assert failed["state"] == "failed" and failed["error"] == "the clip is silent" and failed["segment"] is None


def test_running_jobs_report_as_pending_with_position_zero(env):
    client, store, _ = env
    post(client, 8)
    store.claim_next()
    info = client.get("/v1/previews/8/info", headers=H).json()
    assert info["state"] == "pending" and info["queuePosition"] == 0


def test_delete_is_idempotent(env):
    client, store, _ = env
    post(client, 2)
    finish(store, 2)
    assert client.delete("/v1/previews/2", headers=H).json() == {"deleted": True}
    assert client.delete("/v1/previews/2", headers=H).json() == {"deleted": False}
    assert client.get("/v1/previews/2", headers=H).status_code == 404


def test_the_key_is_only_shown_to_the_home_network(env):
    client, _, _ = env                                  # the test client's address is not an IP: treated as outside
    assert client.get("/api/key").status_code == 403
    assert is_lan("192.168.1.50") and is_lan("10.0.0.2") and is_lan("127.0.0.1") and is_lan("100.101.102.103")
    assert not is_lan("8.8.8.8") and not is_lan("testclient") and not is_lan(None)


def test_key_is_created_once_with_private_permissions(tmp_path):
    p = tmp_path / "sub" / "key"
    k1 = load_or_create_key(p)
    assert len(k1) == 64 and load_or_create_key(p) == k1
    assert oct(p.stat().st_mode & 0o777) == "0o600"
