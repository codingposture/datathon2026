# Typed hybrid video detector

The local React-to-model API is implemented in [backend/](backend/README.md), with no database
or saved analysis history. Start Django with `.venv/bin/python backend/manage.py runserver
127.0.0.1:8000 --noreload`, start the frontend with `npm run dev` in `fortebank-frontend/`,
then enter your backend `API_KEY` in the frontend at http://127.0.0.1:4173/. Local AI is the default. Private backend credentials
are configured locally. Uploads/evidence are temporary; results expire from RAM after 15 minutes.

[case.ipynb](case.ipynb) implements the experimental `hybrid_v2` workflow. It compares
zero-shot X-CLIP with its video-specific prompt generator, a certainty-weighted five-class
logistic classifier, two temporal contexts, score blends, and optional RT-DETR person tracking.
Outputs are possible visual violations for human review, with category IDs, timestamps and JPEG
evidence. Scores are uncalibrated. Legal conclusions, subtypes and web development are deferred.

The Colab T4 experiment is complete. The selected four-second zero-shot model is in
`artifacts/hybrid_v2/` and `artifacts/hybrid_v2.zip`; see
[the new evaluation report](artifacts/HYBRID_V2_REPORT.md). On the reused eight-video diagnostic
test, balanced accuracy is 62.5%, recall 75%, category macro-F1 14.3%, and typed event recall
within five seconds 25%. Typing and localization remain weak; this is experimental.
Longest-video offline inference took 52–55s on T4. Archive checksums, fresh-process cache
recovery and local offline CPU weight loading passed. Training ran only on Colab.

The previous binary model remains in `artifacts/trained_model_provisional/`. Its historical
results are in [artifacts/MODEL_REPORT.md](artifacts/MODEL_REPORT.md). Those results do not
describe the new workflow. New trained results exist only after a successful experiment run.

## Data

The supplied annotated workbook is copied to `open/forte_labels_annotated.xlsx`.
Named sheets and column headers are read as inert data; formulas are never evaluated.
Every source column is retained as provenance. Hashes consolidate 42 files into 40 recordings.
The identical X-4/X-47 recording has a cash event at an uncertain 53–57s interval and a separate
acceptance event at 68s. Untimed secondary annotations supply vocabulary, not supervised targets.
HIGH/MEDIUM/LOW annotations weigh 1/.5/.25; original negatives weigh 1.

`review/legacy_split.json` preserves the original 32-development/8-test partition.
New folds group exact duplicates and the confirmed X-44/X-45 scenario pair. Other possible
scene overlap remains unverified. The reused eight-video test is **diagnostic**, not untouched;
it has no cash/workstation-security positives. Its recall for that category is unavailable.
Only one personal-items recording is available for development. Labels remain provisional.
Do not mark the old manual-review table completed without watching the recordings.

## Run on Colab T4

1. Select and start the Colab GPU kernel for `case.ipynb` in VS Code. The installed local
   bridge uses the Jupyter API and Colab filesystem provider; it does not extract credentials.
2. Run `.venv/bin/python scripts/launch_colab_training.py --launch`. It uploads source and
   supplied data, installs dependencies **before** starting a fresh training kernel, and runs
   the notebook. It saves cell progress, caches and intermediate results under `runs/hybrid_v2`.
3. Check `.venv/bin/python scripts/launch_colab_training.py --status`.
4. Retrieve `.venv/bin/python scripts/launch_colab_training.py --download` after completion.
   This verifies archive checksums, preserves prior exports, and writes `artifacts/hybrid_v2/`
   plus its archive and executed notebook, then verifies offline CPU loading in a fresh process.

For private Drive persistence, use `python scripts/package_colab.py --include-data`, extract
privately into `MyDrive/Datathon`, and open the notebook there. Install requirements in a
disposable setup kernel, restart, then execute the notebook. The notebook stages video on runtime
storage and writes caches/results to the project in Drive. The VS Code launcher's default is
runtime storage; mount/configure Drive before running if persistence beyond VM lifetime is needed.
Model inference never calls cloud AI services.

Restart with the same code, data, dependencies and run name to reuse caches and saved test
outputs. Source hashes lock the run; deliberate changes require a new experiment directory.
The runner skips interactive camera controls. To test cashier filtering, use the notebook's
ROI preview and confirm development-only cashier/departure examples before selection.
Unknown cameras, ambiguous role assignment and uncertain tracks disable filtering.
Departures alone never create a violation; pretrained RT-DETR does not recognize cash or keys.

## Selection and reports

Compare 4s/12s contexts with eight letterboxed frames per window and 1s scan steps.
Search C=.001/.01/.1 and supervised blend weights 0/.25/.5/.75/1. Standardization, learned
heads and prompt descriptions use training-fold data only. Select threshold and components
using OOF binary balanced accuracy, then category macro-F1, then fewer false-positive videos.
Missing classes are explicitly recorded; intermediate blends can use zero-shot fallback.

Events require two consecutive positive scans, quarter-second refinement and within-category
merging. The selected configuration is evaluated once on the reused diagnostic test. Reports
include supports, confusion matrices, baselines, mistakes, high-confidence subset metrics,
2s/5s typed event matching, timing error and false events per negative video. The exported
model report remains explicitly experimental. A one-day compact experiment cannot establish
performance on unseen camera angles or violation types.

## Local offline inference

The export includes safetensors, tokenizer, JSON classifier/scaler parameters, prompts, taxonomy,
camera settings, pinned revisions, exact dependencies, licenses, evaluation and SHA256 checksums.
Provision compatible dependencies and ffmpeg/ffprobe before disconnecting. Run from the export
directory, or with this project's package installed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m datathon.cli \
  open/videos/X-2.mov --artifacts artifacts/hybrid_v2 --offline-check \
  --report artifacts/hybrid_v2_cpu_check.json
```

```python
from datathon.inference import analyze_video
result = analyze_video('new.mov', 'artifacts/hybrid_v2', camera_profile=None)
```

The API preserves verdict/score/events, adding each event's stable spreadsheet `category_id`.
Events contain `timecode_sec`, `score` and up to three `evidence_frames` with actual
`timestamp_sec` and `image_path`. Evidence and result JSON are written beside the artifact.
Version-one artifacts still load through the same API. A camera profile can be a confirmed
JSON file/dictionary or a bundled profile name; an unknown name falls back to X-CLIP.

Two fresh offline processes benchmark the longest video on the training hardware without
feature caches. The Python socket guard checks application-level network independence, not OS
isolation. Five-minute acceptance results are hardware-specific; verify the local deployment
machine separately. Local execution tests are not trained accuracy measurements.

## Development

Use Python 3.12 and a compatible torch/torchvision pair (local CPU: 2.8/0.23).

```bash
.venv/bin/python -m pytest -q
python scripts/build_notebook.py
```

Tests cover workbook provenance/formulas, timestamps and bounds, exact duplicate multi-events,
uncertain intervals and weights, grouped splitting, ambiguous windows, unavailable classes,
prompt isolation, normal departures, uncertain tracking, PTS evidence, legacy loading and v2
fresh-process offline reload. Small random model weights are test fixtures only.
The pre-hybrid notebook is preserved at `artifacts/case.before_hybrid_v2.ipynb`.
