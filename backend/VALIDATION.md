# Backend validation

Verified in this workspace with Python 3.12.11, Django 5.2.18, DRF 3.16.1,
PyTorch 2.8.0+cpu and the existing `artifacts/hybrid_v2` weights. Inference used two
CPU threads on an Intel Core i5-1135G7 @ 2.40 GHz. No model training was performed.

- Django system checks: passed; no database or migrations required.
- Backend tests: **19 passed**. Covers both-route authentication, actual multipart/container
  validation, partial/multiple-upload cleanup, async polling, base64 evidence, failure handling,
  busy admission, real subprocess termination, shutdown cleanup, bounded RAM and expiry.
- Existing frontend tests: **27 passed**.
- Frontend production build: passed. The generated assets do not contain the private API key.
- Real HTTP integration used the frontend's actual service adapter and Vite proxy, including
  server-side API-key injection, then the CPU model and JSON result normalization.

| Integration example | Total upload-to-result time | Model candidates | JPEG frames |
|---|---:|---:|---:|
| Short supplied positive recording | 28.2 s | 1 | 3 |
| Short supplied negative recording | 18.3 s | 0 | 0 |
| Longest supplied recording, 142.533 s | 172.5 s | 6 | 18 |

These are execution checks, not new model-quality estimates. The longest recording is labeled
negative in the dataset, so its six candidates are false alarms. The model remains experimental;
the backend faithfully returns its outputs and does not retune it or invent explanations.
Other inputs/hardware can take longer. The server stops a job at 295 seconds and reports
`failed`; the frontend has the case's 300-second deadline.

Every real smoke check confirmed temporary job-file removal and no inference output written
into the model artifact directory. No video, frame or complete analysis response was saved by
the smoke script. Completed results remain only in server RAM until expiry/eviction/restart.
The worker loads local-only model files under the existing Python socket-denial guard and
offline Hugging Face flags. There are no cloud AI calls, audio analysis or face identification.

Legal citations remain deferred (`rule_reference: null`). Public hosting is not configured in
this local CPU iteration. Start commands and the exact API are in [README.md](README.md).
