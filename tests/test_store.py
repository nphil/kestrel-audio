"""Queue order, idempotence, restart recovery and retention."""
import pytest

from kestrel_audio.store import FAILED, PENDING, READY, RUNNING, Store

DAY = 86400.0


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path, tmp_path / "inbox", tmp_path / "previews", tmp_path / "db.sqlite")
    yield s
    s.close()


def put(store, det, now, **kw):
    return store.submit(det, b"clip", species="Blue Jay", scientific="Cyanocitta cristata", camera="Back", now=now, **kw)


def done(store, det, now, nbytes=1000, cleaned=False):
    store.claim_next(now)
    store.preview_path(det).write_bytes(b"x" * nbytes)
    store.finish(det, {"cleaned": cleaned}, nbytes=nbytes, cleaned=cleaned, took_s=2.0, device="cuda", now=now)


def test_newest_job_runs_first_and_backfill_waits_for_everything_else(store):
    put(store, 1, now=100)
    put(store, 2, now=200)
    put(store, 3, now=150, low_priority=True)
    put(store, 4, now=300, low_priority=True)
    order = [store.claim_next(now=400).detection_id for _ in range(4)]
    assert order == [2, 1, 4, 3]
    assert store.claim_next() is None


def test_queue_position_follows_the_same_order(store):
    for det, t in ((1, 100), (2, 200), (3, 300)):
        put(store, det, now=t)
    put(store, 9, now=50, low_priority=True)
    assert [store.queue_position(store.get(d)) for d in (3, 2, 1, 9)] == [1, 2, 3, 4]
    store.claim_next()
    assert store.queue_position(store.get(3)) is None           # running, not waiting


def test_posting_the_same_detection_twice_changes_nothing_unless_forced(store):
    job, created = put(store, 7, now=100)
    assert created and job.state == PENDING
    again, created = put(store, 7, now=999)
    assert not created and again.created_at == 100
    done(store, 7, now=120)
    again, created = put(store, 7, now=130)
    assert not created and again.state == READY
    forced, created = put(store, 7, now=140, force=True)
    assert created and forced.state == PENDING and not store.preview_path(7).exists()


def test_a_running_job_cannot_be_forced_over(store):
    put(store, 7, now=100)
    store.claim_next()
    again, created = put(store, 7, now=110, force=True)
    assert not created and again.state == RUNNING


def test_failed_jobs_are_not_retried_by_a_plain_repost(store):
    put(store, 5, now=100)
    store.claim_next()
    store.fail(5, "could not decode audio")
    again, created = put(store, 5, now=200)
    assert not created and again.state == FAILED and again.error == "could not decode audio"
    assert not store.clip_path(5).exists()                     # the clip is not kept once the job is over


def test_after_a_restart_running_jobs_requeue_if_their_clip_survived(store):
    put(store, 1, now=100)
    put(store, 2, now=110)
    store.claim_next()
    store.claim_next()
    store.clip_path(1).unlink()
    assert store.recover() == 2
    assert store.get(1).state == FAILED and "restart" in store.get(1).error
    assert store.get(2).state == PENDING


def test_retention_removes_old_previews_by_age(store):
    for det, t in ((1, 0.0), (2, 20 * DAY), (3, 40 * DAY)):
        put(store, det, now=t)
        done(store, det, now=t)
    gone = store.purge(max_age_days=30, cap_bytes=10**9, now=45 * DAY)       # ages: 45, 25 and 5 days
    assert gone == [1] and store.get(1) is None and not store.preview_path(1).exists()
    assert store.get(2) and store.get(3)


def test_retention_cap_drops_the_oldest_first(store):
    for det in (1, 2, 3, 4):
        put(store, det, now=det)
        done(store, det, now=det * 10.0, nbytes=1000)
    gone = store.purge(max_age_days=30, cap_bytes=2500, now=100.0)
    assert gone == [1, 2]
    assert store.storage() == {"previews": 2, "bytes": 2000}


def test_waiting_jobs_are_never_purged(store):
    put(store, 1, now=0.0)
    assert store.purge(max_age_days=1, cap_bytes=0, now=100 * DAY) == [] and store.get(1)


def test_counts_today_and_timing(store):
    for det, took in ((1, 1.0), (2, 3.0), (3, 2.0)):
        put(store, det, now=det)
        store.claim_next()
        store.preview_path(det).write_bytes(b"x")
        store.finish(det, {}, nbytes=1, cleaned=(det == 2), took_s=took, device="cuda", now=1000.0 + det)
    put(store, 4, now=5)
    store.claim_next()
    store.fail(4, "boom", now=1004.0)
    c = store.counts()
    assert (c["ready"], c["failed"], c["cleaned"]) == (3, 1, 1)
    assert store.today(since=1000.0) == {"processed": 4, "cleaned": 1, "failed": 1}
    assert store.today(since=1003.0) == {"processed": 2, "cleaned": 0, "failed": 1}
    assert store.timing()["medianS"] == 2.0
    assert store.last_errors()[0]["detectionId"] == 4
