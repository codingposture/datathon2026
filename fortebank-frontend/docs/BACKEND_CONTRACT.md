# Django REST Framework integration contract

The local DRF implementation now lives in [../../backend/](../../backend/README.md).
It uses the existing `hybrid_v2` weights, temporary files, and RAM-only job results.
Run `python scripts/configure_local_backend.py` from the project root to configure
private backend credentials and the frontend origin. Enter the backend API_KEY in the UI. No database or migrations are needed. Verified legal references
remain deferred; the backend returns `rule_reference: null` and cautious category descriptions.

The paths, multipart field, header, status names, verdicts, and evidence schema below come from section 10 of the provided case. The React UI is compatible with that wire contract. The Django backend and local model are separate deliverables.

## Submission

```http
POST /api/v1/analyze
X-API-Key: <team-key>
Content-Type: multipart/form-data; boundary=<set-by-client>
```

Multipart field: `file`, containing a `.mov` video. Response: HTTP 200 or 202 with JSON:

```json
{ "job_id": "3f2a9c1e" }
```

The frontend lets the browser construct multipart boundaries. Do not require a JSON body, a `video` field, or a trailing-slash redirect. The API is asynchronous; return the ID promptly and run the local model in a backend worker/process. Store job state on the backend rather than relying on one browser session.

## Job status

```http
GET /api/v1/jobs/3f2a9c1e
X-API-Key: <team-key>
```

While processing:

```json
{ "job_id": "3f2a9c1e", "status": "processing" }
```

Completed with possible violations:

```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "A category defined by your team",
      "timecode_sec": 44.2,
      "description": "Observable action and why it may indicate a violation.",
      "evidence_frames": [
        { "timestamp_sec": 44.2, "image_base64": "<raw JPEG base64, without a data URL prefix>" }
      ]
    }
  ]
}
```

Completed with no violation:

```json
{ "job_id": "3f2a9c1e", "status": "done", "verdict": "no_violation", "violations": [] }
```

Failed:

```json
{ "job_id": "3f2a9c1e", "status": "failed", "error": "A useful, non-sensitive explanation." }
```

`error` is a proposed failure-message convention; the case fixes the `failed` status but does not define its error body. `detail` is also accepted for DRF-style messages.

Job IDs must be non-empty strings and match between submission and polling. Timestamps must be non-negative JSON numbers; fractional seconds are preserved internally and displayed as whole `MM:SS` or `HH:MM:SS`. Each finding normally contains one to three JPEG frames. Missing or invalid image content falls back to a visible unavailable-image message rather than hiding the event. More than three frames and contradictory verdicts are treated as contract errors.

The frontend polls every two seconds, permits a maximum of five minutes including submission, and applies a 30-second timeout to individual HTTP requests. It never automatically resubmits a video. Timeout/network messages explain that a previously submitted server job may still be running. Reset aborts browser requests and prevents stale updates; the case defines no server cancellation endpoint, so none is invented here.

## Optional source extension

The narrative needs a source, but the supplied JSON has no source field. Add this optional object to a violation after agreeing on it with the frontend team:

```json
{
  "rule_reference": {
    "document_title": "Title of a source researched by your team",
    "section": "Relevant section or clause, if available",
    "url": "https://adilet.zan.kz/..."
  }
}
```

The frontend shows document/section text and an optional HTTP(S) link opened only on user action. It displays a missing-source notice if the field is absent; it does not fabricate a rule or fetch remote citations. The team's research must establish applicability. Opening an external source link can require internet even though analysis does not.

## DRF implementation notes

- Define the exact no-trailing-slash routes `api/v1/analyze` and `api/v1/jobs/<str:job_id>`. The adapter does not follow redirects, so incorrect routes fail visibly instead of potentially forwarding a video/key elsewhere.
- Use DRF `MultiPartParser` for submission and a serializer with a `file` field. Validate actual file contents, supported format, size, and processing constraints server-side; do not trust filename extensions.
- Require the correct `X-API-Key` for both routes and reject missing/incorrect keys. The frontend does not create or own the team's jury key.
- Return JSON responses and appropriate HTTP errors. The adapter handles 401/403, 413, 415, 429, 5xx, unreadable responses, and common DRF `detail`/`file` validation errors.
- Run all inference locally. No cloud AI calls, face identification, or audio processing should be added by the frontend; model enforcement is the backend team's responsibility.
- The loopback Vite proxy makes browser requests same-origin, so frontend development does not require permissive CORS settings. If the team deliberately uses session/cookie authentication instead, implement its CSRF flow separately; this frontend integration uses the case's API-key contract.

Official references: [DRF multipart parsing](https://www.django-rest-framework.org/api-guide/parsers/#multipartparser), [Vite development proxy](https://vite.dev/config/server-options#server-proxy), [React with a build tool](https://react.dev/learn/build-a-react-app-from-scratch).

## Production handoff

`pnpm build` produces the `dist/` directory. Serve it with the team's production server; Vite's development proxy is not part of the built frontend. Serve local assets and base64 evidence with a policy that permits same-origin scripts/styles/API requests and `data:` images. The built HTML includes such a policy; development-only inline Fast Refresh permission is not included in the production script policy.

Both API routes validate the user-entered `X-API-Key`. The frontend requires a nonempty key
before video selection and sends it on upload and polling. It keeps the key in memory only;
Django is the authority on whether it is valid. The local proxy forwards this header unchanged.
For a production build, configure your same-origin server to forward `/api/` to Django and
preserve `X-API-Key`. The Vite development proxy is not included in `dist/`.

This frontend's automated adapter checks mock HTTP and use synthetic File objects. The local
backend's separate real-model smoke command is documented in `backend/README.md`. Production
gateway authentication and public hosting remain separate deployment work.
