# Case analysis

Source: the user-provided `Кейс_ForteBank.docx` in Downloads, read as reference material. Its project requirements informed this implementation; instructions about datasets, submission, deployment, and organization were not treated as permission to access videos, contact organizers, publish the project, or change the user's requested scope.

## What the task actually asks for

The case concerns visually observable **cash-discipline events at a bank cashier**, rather than unrestricted general surveillance detection. The service should report possible issues, with observable evidence, without inferring bookkeeping correctness or other invisible system facts. Findings aid a human reviewer.

Sections 4–7 require a video upload, processing status, one of two final verdicts, multiple possible events, event types, timestamps, descriptions, evidence frames, and related rules/normative sources. Models must run locally with no cloud AI processing, and the deployed system should process videos without internet after model weights are available. Audio and face identification are excluded by the FAQ.

## Mapping to this frontend

| Requirement | Frontend behavior |
| --- | --- |
| MOV input, approximately 29 MB in the dataset | Case-insensitive `.mov` selection; no invented fixed 29 MB cap |
| Two-step analysis API | Exact section 10 upload/status routes and multipart field `file` |
| API key | Required user-entered key; `X-API-Key` forwarded unchanged and checked by Django |
| `processing`, `done`, `failed` | Normalized status display and useful error messages |
| `violation` or `no_violation` | Explicit verdict using cautious human-review wording |
| One or more events | Selectable event cards, without a hard-coded backend category list |
| Event type, timecode, description | Displayed directly from the validated backend response |
| One to three JPEG evidence frames | Base64 JPEG rendering and per-frame controls/timestamps |
| Rule and normative source | Optional `rule_reference` extension; visible missing-source notice otherwise |
| Five-minute evaluation limit | Whole-job browser deadline, including upload, with local cancellation |
| Local processing | No external runtime assets or cloud inference calls; development proxy accepts only loopback origins |

## Specification gap for the backend team

The narrative requires the rule and normative source, but section 10's sample JSON omits a corresponding property. The frontend supports an optional `rule_reference` object per violation with `document_title`, `section`, and `url`. This is our proposed extension, not a field claimed to exist in the case. Backend teammates must agree to populate it with researched sources. No legal citations were invented or externally researched for this conversion.

The existing three demo categories and schematic images are retained to preserve the requested design/solution. They are fictional UI examples, not a definitive cash-discipline taxonomy. The React results renderer accepts the team's actual backend event types.

## Remaining team responsibilities

Implement the Django API and local inference, validate video content, document model limitations, research relevant Kazakhstan normative sources, authenticate the jury API, and deliver the required public URL. The FAQ's statement that other banks' open documents may be accepted should be reconciled by the team with the normative-source requirements. The frontend supplies no definitive interpretation of those legal sources.

The frontend cannot verify model locality, detection accuracy, or the validity of a rule reference.
Its original frontend-only checks used mocks without accessing footage. The sibling backend now
has real local CPU integration checks through this frontend's proxy/adapter; see
[backend validation](../../backend/VALIDATION.md). No legal citations are fabricated.
