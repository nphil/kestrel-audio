"""HTTP API. Open without a key: `/`, `/healthz`, `/api/status`, `/icon-512.png`, favicons. Everything under `/v1`
needs the header `X-Kestrel-Audio-Key`. `/api/key` shows the key, to clients on the home network only."""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .config import DEFAULT_LOCAL_SPECIES, PACKAGE_DIR, Config
from .manager import Manager
from .perch import LABELS_FILE
from .species import Labels, LocalList, LocalSpecies, local_list_json, parse_local_list
from .store import FAILED, PENDING, READY, RUNNING, Job, Store, iso

HOME_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",      # private IPv4 (RFC 1918)
    "127.0.0.0/8", "169.254.0.0/16", "100.64.0.0/10",     # loopback, link-local, Tailscale's CGNAT range
    "::1/128", "fc00::/7", "fe80::/10"))                  # IPv6 loopback, unique-local, link-local

LOCAL_SPECIES_MAX_BYTES = 1024 * 1024


def load_or_create_key(path: Path) -> str:
    """The shared API key: 32 random bytes as hex, created on first start, mode 0600."""
    if path.exists():
        key = path.read_text().strip()
        if key:
            return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    return key


def _parse_ip(text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """'1.2.3.4', '1.2.3.4:5678', '[::1]:80', '::ffff:10.0.0.1' -> an address, or None."""
    if not text:
        return None
    t = text.strip().strip('"')
    if t.startswith("["):
        t = t[1:].split("]")[0]
    elif t.count(":") == 1:
        t = t.split(":")[0]
    try:
        ip = ipaddress.ip_address(t)
    except ValueError:
        return None
    return ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip


def is_lan(host: str | None) -> bool:
    """Home network, loopback or tailnet (Tailscale's 100.64.0.0/10)."""
    ip = _parse_ip(host)
    return ip is not None and any(ip in net for net in HOME_NETWORKS if net.version == ip.version)


_FOR_RE = re.compile(r'for="?\[?([^;,"\]]+)', re.I)


def caller_is_lan(peer: str | None, headers: Mapping[str, str]) -> bool:
    """May this caller see the API key? The TCP peer must be on the home network, and so must the ORIGINAL caller when the
    request came through a reverse proxy or a tunnel (whose own address is always private). A request that announces a
    forwarding hop we cannot read (Cloudflare's `cf-ray` without `cf-connecting-ip`) is refused."""
    if not is_lan(peer):
        return False
    hops: list[str] = []
    for name in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
        if headers.get(name):
            hops.append(headers[name])
    if headers.get("x-forwarded-for"):
        hops += headers["x-forwarded-for"].split(",")
    if headers.get("forwarded"):
        hops += _FOR_RE.findall(headers["forwarded"])
    if headers.get("cf-ray") and not headers.get("cf-connecting-ip"):
        return False
    return all(is_lan(h) for h in hops)


def job_info(job: Job, store: Store) -> dict[str, Any]:
    """The /info body: every field always present, null when unknown."""
    state = PENDING if job.state in (PENDING, RUNNING) else job.state
    body: dict[str, Any] = {
        "detectionId": job.detection_id, "state": state, "segment": None, "method": None, "cleaned": False, "scores": None,
        "loudnessLufs": None, "durationS": None, "createdAt": iso(job.created_at), "readyAt": None,
        "queuePosition": 0 if job.state == RUNNING else store.queue_position(job), "error": None,
        "alternatives": None, "announced": None,
    }
    if job.state == READY and job.info:
        i = job.info
        body.update({k: i.get(k) for k in ("segment", "method", "cleaned", "scores", "loudnessLufs", "durationS", "alternatives", "announced")})
        body["cleaned"] = bool(i.get("cleaned"))
        body["readyAt"] = iso(job.finished_at)
        body["variant"] = i.get("variant")
        body["device"] = i.get("device")
        body["tookS"] = None if job.took_s is None else round(job.took_s, 2)
        body["notes"] = i.get("notes") or []
    if job.state == FAILED:
        body["error"] = job.error or "failed"
        body["readyAt"] = iso(job.finished_at)
    return body


def create_app(cfg: Config, store: Store, manager: Manager, key: str, labels: Labels | None = None) -> FastAPI:
    app = FastAPI(title="Kestrel Audio", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    static = PACKAGE_DIR / "static"

    def auth(x_kestrel_audio_key: str | None = Header(default=None)) -> None:
        if not x_kestrel_audio_key or not hmac.compare_digest(x_kestrel_audio_key.encode(), key.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong X-Kestrel-Audio-Key")

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    # ------------------------------------------------------------------ open endpoints
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/")
    async def index() -> Response:
        page = static / "index.html"
        if not page.exists():
            raise HTTPException(404, "status page missing")
        return FileResponse(page, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})

    for route, name, mime in (("/icon-512.png", "icon-512.png", "image/png"), ("/favicon-32.png", "favicon-32.png", "image/png"),
                              ("/favicon.ico", "favicon.ico", "image/x-icon")):
        def _asset(name: str = name, mime: str = mime) -> Response:
            p = cfg.assets_dir / name
            if not p.exists():
                raise HTTPException(404, "not found")
            return FileResponse(p, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})
        app.add_api_route(route, _asset, methods=["GET"], include_in_schema=False)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return manager.stats()

    @app.get("/api/key")
    async def show_key(request: Request) -> Response:
        if not caller_is_lan(request.client.host if request.client else None, request.headers):
            return JSONResponse({"error": "lan_only"}, status_code=403)
        return JSONResponse({"key": key}, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ /v1
    @app.get("/v1/stats", dependencies=[Depends(auth)])
    async def stats() -> dict[str, Any]:
        return manager.stats()

    # The species that can be real where the microphone is (BirdNET-Go's range-filter list, kept up to date by Home Assistant): what
    # Perch's scores are measured against, so a 'could also be' list is made of birds that live here, not of 14,000 species.
    def perch_labels() -> Labels | None:
        nonlocal labels
        if labels is None:
            try:
                labels = Labels.from_file(cfg.models_dir / LABELS_FILE)
            except (OSError, ValueError):
                return None
        return labels

    def local_state(local: LocalList, custom: bool) -> dict[str, Any]:
        known = perch_labels()
        return {"source": local.source, "updatedAt": local.updated_at, "count": len(local.entries), "custom": custom,
                "matched": None if known is None else LocalSpecies.build(known, local).matched}

    @app.get("/v1/local-species", dependencies=[Depends(auth)])
    async def local_species() -> dict[str, Any]:
        custom = cfg.local_species_path.exists()
        local = parse_local_list(json.loads((cfg.local_species_path if custom else DEFAULT_LOCAL_SPECIES).read_text(encoding="utf-8")))
        return local_state(local, custom)

    @app.put("/v1/local-species", dependencies=[Depends(auth)])
    async def put_local_species(request: Request) -> dict[str, Any]:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > LOCAL_SPECIES_MAX_BYTES:
                raise HTTPException(413, "the species list is larger than 1 MiB")
        try:
            local = parse_local_list(json.loads(bytes(body)))
        except (ValueError, RecursionError) as exc:     # includes a body that is not JSON
            raise HTTPException(400, f"not a species list: {exc}")
        text = json.dumps(local_list_json(local), ensure_ascii=False)
        path = cfg.local_species_path
        changed = not path.exists() or path.read_text(encoding="utf-8") != text
        if changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)                           # the worker reads the file between jobs: never half a list
        return {**local_state(local, True), "changed": changed}

    @app.delete("/v1/local-species", dependencies=[Depends(auth)])
    async def delete_local_species() -> dict[str, bool]:
        existed = cfg.local_species_path.exists()
        cfg.local_species_path.unlink(missing_ok=True)
        return {"deleted": existed}

    @app.post("/v1/jobs", dependencies=[Depends(auth)])
    async def submit(request: Request, detection_id: int | None = None, species: str | None = None, scientific: str | None = None,
                     camera: str | None = None, force: str | None = None, priority: str | None = None) -> Response:
        if detection_id is None or detection_id < 0:
            raise HTTPException(400, "detection_id (integer) is required")
        if not species or not species.strip():
            raise HTTPException(400, "species is required")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cfg.max_body_bytes:
            raise HTTPException(413, f"the clip is larger than {cfg.max_body_bytes // 2**20} MiB")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > cfg.max_body_bytes:
                raise HTTPException(413, f"the clip is larger than {cfg.max_body_bytes // 2**20} MiB")
        if not body:
            raise HTTPException(400, "the request body must be the audio clip")
        job, created = store.submit(
            detection_id, bytes(body), species=species.strip()[:120], scientific=(scientific or "").strip()[:120] or None,
            camera=(camera or "").strip()[:120] or None, low_priority=(priority or "").lower() == "low",
            force=(force or "").lower() in ("1", "true", "yes"))
        if created:
            manager.kick()
        info = job_info(job, store)
        return JSONResponse(info, status_code=202 if job.state in (PENDING, RUNNING) else 200)

    @app.get("/v1/previews/{detection_id}/info", dependencies=[Depends(auth)])
    async def info(detection_id: int) -> Response:
        job = store.get(detection_id)
        if job is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return JSONResponse(job_info(job, store))

    @app.get("/v1/previews/{detection_id}", dependencies=[Depends(auth)])
    async def preview(request: Request, detection_id: int) -> Response:
        job = store.get(detection_id)
        if job is None:
            return JSONResponse({"state": "none"}, status_code=404)
        if job.state in (PENDING, RUNNING):
            return JSONResponse({"state": "pending"}, status_code=202, headers={"Retry-After": "5"})
        if job.state == FAILED:
            return JSONResponse({"state": "failed", "error": job.error}, status_code=404)
        path = store.preview_path(detection_id)
        if not path.exists():
            return JSONResponse({"state": "none"}, status_code=404)
        etag = f'"{detection_id}-{int(job.finished_at or 0)}-{job.nbytes}"'
        headers = {"Cache-Control": "public, max-age=31536000, immutable", "ETag": etag}
        sent = [t.strip().removeprefix("W/") for t in request.headers.get("if-none-match", "").split(",")]
        if etag in sent or "*" in sent:
            return Response(status_code=304, headers=headers)
        return FileResponse(path, media_type="audio/mp4", headers=headers)

    @app.delete("/v1/previews/{detection_id}", dependencies=[Depends(auth)])
    async def delete(detection_id: int) -> dict[str, bool]:
        return {"deleted": store.delete(detection_id)}

    return app
