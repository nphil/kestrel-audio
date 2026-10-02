# Kestrel Audio API

Base URL: `http://<unraid-ip>:8787`. Everything under `/v1` needs the header `X-Kestrel-Audio-Key: <key>`.
The key is a text file named `key` in the app data folder (`/mnt/nvme/appdata/kestrel-audio/key` on the home server);
it is created on first start and can also be revealed on the status page from the home network.

Open without a key: `GET /` (status page), `GET /healthz`, `GET /api/status`, `GET /icon-512.png`, `GET /favicon.ico`,
`GET /api/key` (answers 403 unless the caller is on the home network or the tailnet).

Errors are JSON: `{"error": "..."}` with 400 (bad parameters), 401 (missing or wrong key), 404, 413 (clip over 20 MiB).

## Submit a clip

```
POST /v1/jobs?detection_id=<int>&species=<common name>&scientific=<latin name>[&camera=<name>][&force=1][&priority=low]
Content-Type: audio/ogg            (any audio the server's ffmpeg can read; clips are ~100 KB)
body = the original clip
```

* `scientific` is how the animal is found in Perch. Always pass it when you have it (BirdNET-Go's `scientificName`).
  Without a name Perch knows, the preview is BirdNET-Go's own detection window, made loud and never cleaned.
* One job per `detection_id`. A repeat POST is a no-op that returns the current state. `force=1` re-runs it (and replaces a
  finished preview). Failed jobs are not retried unless `force=1`.
* `priority=low` queues behind everything else (use it for back-filling old detections). Within a priority the NEWEST
  job runs first.
* **Success is 200 or 202**; read `state` in the body. `202` = queued or running, `200` = already finished (`ready` or `failed`).

## Read the result

```
GET /v1/previews/<detection_id>/info     -> 200 {...} | 404 {"error":"not_found"}
GET /v1/previews/<detection_id>          -> 200 audio/mp4 | 202 {"state":"pending"} | 404 {"state":"none"|"failed",...}
DELETE /v1/previews/<detection_id>       -> 200 {"deleted": true|false}       (idempotent)
```

`/info` always returns every field below (`null` when unknown):

| field | meaning |
|---|---|
| `detectionId` | the id you posted |
| `state` | `pending` (queued or running), `ready`, or `failed` |
| `segment` | `{start, end, source}`: where the preview sits **inside the original clip**, in seconds. `source` is `perch` (Perch found the moment), `fallback` (BirdNET-Go's own window: the species is unknown to Perch, or Perch is barely sure of it, best match under 0.10, so there is no real moment to find and nothing is cleaned) or `whole` (clip no longer than one Perch window) |
| `method` | `trim` (the moment, made loud), `gate` (hiss reduction), `separate` (AI separation), `separate+gate` |
| `cleaned` | `true` only when a clean-up passed the Perch re-check and was shipped |
| `scores` | `{original, preview}`: how sure Perch is of the species on the untouched moment and on the shipped preview, both measured at the preview's loudness |
| `loudnessLufs`, `durationS` | measured on the shipped file |
| `createdAt`, `readyAt` | ISO 8601 UTC |
| `queuePosition` | 1 = next; `0` while running; `null` otherwise |
| `error` | why a `failed` job failed |

Extra (not part of the contract, may grow): `variant` (candidate id such as `G20`), `device` (`cuda` or `cpu`), `tookS`, `notes`.
The stored `<id>.json` next to the audio also holds the full decision trace (every candidate, its clarity gain and Perch scores).

`GET /v1/previews/<id>` serves `audio/mp4` (AAC-LC, 96 kb/s, mono, 48 kHz) with `Accept-Ranges`, `ETag` and
`Cache-Control: public, max-age=31536000, immutable`; `Range` requests answer `206`, `If-None-Match` answers `304`.

## Status

`GET /v1/stats` (key) and `GET /api/status` (no key) return the same JSON: queue depth (`queue.depth` = waiting + running),
the worker (`running`, `device`, `idleS`, `vramMiB`), the GPU (`freeMiB`, `totalMiB`), counts, today's totals, storage,
timing and the last errors. `GET /healthz` is the Docker health check.

## Configuration (environment variables)

| variable | default | meaning |
|---|---|---|
| `KESTREL_AUDIO_DEVICE` | `auto` | `auto` (GPU when enough memory is free, else CPU), `cpu`, or `cuda` |
| `KESTREL_AUDIO_MIN_FREE_VRAM_MIB` | `2500` | free GPU memory needed before the GPU is used |
| `KESTREL_AUDIO_GPU_CAP_MIB` | `1500` | the worker's own GPU budget |
| `KESTREL_AUDIO_IDLE_UNLOAD_S` | `120` | idle seconds before the worker (and its GPU memory) exits |
| `KESTREL_AUDIO_CPU_THREADS` | `4` | threads for CPU work (max 4) |
| `KESTREL_AUDIO_CPU_CLEANUP` | `1` | `0` = on the CPU path only make the moment loud, skip the clean-up search |
| `KESTREL_AUDIO_RETENTION_DAYS` | `30` | previews older than this are deleted |
| `KESTREL_AUDIO_CAP_MB` | `2048` | storage cap; oldest previews are deleted first |
| `KESTREL_AUDIO_MAX_BODY_MB` | `20` | largest accepted clip |
| `KESTREL_AUDIO_MAX_CLIP_S` | `60` | longest accepted clip |
| `KESTREL_AUDIO_PORT` | `8787` | listen port |
