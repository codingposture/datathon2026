<!-- # Local DRF backend -->

Upload a `.mov`, poll its job, and receive the existing `hybrid_v2` model's verdict,
four broad categories, timestamps and 1–3 JPEG evidence frames per event. All inference
runs on this computer's CPU using local weights. This backend performs no training.

No Django models, migrations, database, Redis, Celery, accounts or history are used.
Temporary videos/evidence are removed after success, failure or timeout. Only JSON/base64
results stay in RAM for up to 15 minutes, with at most eight retained jobs. Oldest results
are evicted if capacity is reached; restart clears every job. A force-killed/crashed process
can leave OS temporary files; normal completion and graceful shutdown clean them up.

## Start this workspace

The private backend key is configured as `API_KEY` in `backend/.env` (ignored by Git).
Users enter this key in the frontend password field; it is kept only in page memory.
Run these commands from the project root in separate terminals:

```bash
.venv/bin/python backend/manage.py runserver 127.0.0.1:8000 --noreload
```

```bash
cd fortebank-frontend
npm run dev
```

Open **http://127.0.0.1:4173/**. Local AI mode is the default. Enter `API_KEY`, select a
`.mov`, then click **Upload & analyze**. Django checks the entered key on upload and polling.
The proxy passes it unchanged; no frontend environment key is injected. Demo mode remains
an explicit option for fictional results. Restart Vite after editing its backend origin.

Use **one Django process with `--noreload`** because job state lives in that process.
One video is analyzed at a time; another submission receives HTTP 429 rather than waiting
in a queue. Polling stays responsive while inference runs in an isolated subprocess.
Reset in the UI stops polling; it does not cancel the server job.

## Fresh setup

Requires Python 3.12, ffmpeg/ffprobe, Node supported by the frontend's dependencies
(Node 24.15+ LTS recommended), and `artifacts/hybrid_v2/` with its complete local weights.
The existing workspace `.venv` and frontend dependencies are already configured.
For another machine:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
.venv/bin/python -m pip install -r requirements-backend.txt
.venv/bin/python scripts/configure_local_backend.py
cd fortebank-frontend
npm install
```

`configure_local_backend.py` is idempotent and does not print secrets or replace existing
settings. It configures the backend key and frontend origin; enter the backend key in the UI.
No `VITE_` secret is used.
There is no `migrate` command to run. The model folder is read-only during inference;
the API calls the detector with a temporary evidence directory instead of the model helper
that normally saves inference outputs beside the weights.

Backend `.env` options:

| Variable | Default / purpose |
|---|---|
| `API_KEY` | Private generated jury/API key; required on both endpoints |
| `DJANGO_SECRET_KEY` | Separate private generated Django secret |
| `MODEL_ARTIFACT_DIR` | `artifacts/hybrid_v2`, resolved against the project root |
| `INFERENCE_THREADS` | `2`, CPU threads for the inference subprocess |
| `ANALYSIS_TIMEOUT_SECONDS` | `295` maximum, including backend upload/validation time |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost,[::1]` |

The subprocess is stopped at the deadline, including child processes, and the job becomes
`failed`. The margin before 300s allows the frontend's next poll. CPU speed depends on the
machine and video; the previous T4 benchmark does not establish CPU performance.
Uploads are limited to 64 MiB, five minutes of video, one video stream and up to 4K pixels
(portrait supported). Actual MOV-family contents are probed; file extension/MIME alone is
insufficient. Audio tracks may exist but are never analyzed. Invalid/corrupt inputs fail
validation or analysis; they are not reported as `no_violation`.

## API

Both routes require `X-API-Key`. Missing/incorrect keys return 403. Use exact paths without
trailing slashes:

```http
POST /api/v1/analyze
X-API-Key: <your-key>
Content-Type: multipart/form-data

file=<video.mov>
```

Returns HTTP 202 with `{"job_id":"..."}`. Poll `GET /api/v1/jobs/<job_id>`:

```json
{ "job_id": "...", "status": "processing" }
```

```json
{
  "job_id": "...",
  "status": "done",
  "verdict": "violation",
  "violations": [{
    "type": "Acceptance step skipped",
    "category_id": "ACCEPTANCE_STEP_SKIPPED",
    "timecode_sec": 12.25,
    "description": "The model flagged a possible skipped acceptance step near this timestamp. Review the surrounding video to check identification, counting or banknote inspection.",
    "evidence_frames": [{ "timestamp_sec": 12.27, "image_base64": "/9j/..." }],
    "rule_reference": null,
    "score": 0.9
  }],
  "score": 0.9,
  "score_kind": "uncalibrated_hybrid_detection_score",
  "experimental": true
}
```

Example times/scores above are illustrative. Evidence contains raw JPEG base64 without a
data-URL prefix or filesystem paths. `no_violation` has an empty `violations` array.
Failed jobs return `{"job_id":"...","status":"failed","error":"..."}`.
Expired/unknown IDs return 404. Malformed uploads return 400, oversized uploads 413,
unsupported/invalid video 415, and busy submission 429. Results use `Cache-Control: no-store`.

Descriptions are cautious category templates, **not generated accounts of what actually
happened**. The model does not produce those explanations. Legal research remains deferred:
`rule_reference` is null, and the existing frontend shows its missing-source notice.
The narrative case's verified-rule requirement therefore remains unfinished.

The model is experimental: its reused eight-video diagnostic test had 62.5% balanced
accuracy, 14.3% category macro-F1 and 25% typed-event recall within five seconds. See
[the ML report](../artifacts/HYBRID_V2_REPORT.md). This API does not improve model accuracy
or identify faces. Public hosting is a separate deployment step; keep the Vite development server on loopback.

## Verification

```bash
.venv/bin/python backend/manage.py check
.venv/bin/python -m pytest -q backend/tests
node --env-file=backend/.env scripts/smoke_backend.mjs open/videos/X-2.mov
```

The automated tests use synthetic videos and mocked model execution, plus real subprocess
timeout/cleanup checks. The optional smoke command calls the **real CPU model** through the
frontend's proxy/adapter, checks JPEG evidence and temporary-file removal, and saves no
video, frames or result. It requires both servers running. The frontend's existing tests
and build run with `npm test` and `npm run build` in its directory.

Completed checks and the measured 172.5s CPU result for the longest supplied video are recorded
in [VALIDATION.md](VALIDATION.md). This is a hardware-specific timing result, not an accuracy claim.
