"""Regenerate the reviewable notebook from ordinary, syntax-checkable cell sources."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': dedent(source).strip().splitlines(keepends=True)})


def code(source, tag=None):
    cells.append({'cell_type': 'code', 'metadata': {'tags': [tag]} if tag else {},
                  'execution_count': None, 'outputs': [], 'source': dedent(source).strip().splitlines(keepends=True)})


md('''
# Bank surveillance: reviewed, grouped ML baseline

This notebook audits the supplied videos, collects your review, trains a binary event
model on a frozen X-CLIP encoder, evaluates held-out recordings, and exports local inference.
Outputs are possible visual violations for human review. Scores are **uncalibrated**.
Types, legal explanations, the UI, and the jury API are a later phase.

**Colab setup:** privately copy this project (including `datathon/`, `scripts/`,
`requirements-colab.txt`, `review/`, and `open/`) to `MyDrive/Datathon`.
You can create a private upload archive locally with
`python scripts/package_colab.py --include-data`, then extract it into MyDrive.
Open `case.ipynb` using the VS Code Colab extension and select a GPU runtime (T4 target).
The notebook expects an existing compatible PyTorch/torchvision pair.
If your extension cannot complete Drive authentication, run the mount cell in Colab's
browser UI, or configure an already-mounted private path below.

This run defaults to the user-authorized **provisional baseline** using supplied labels.
Manual review is optional for this provisional run and remains incomplete.
For the reviewed experiment, set ANNOTATION_MODE to reviewed and complete every review first.
''')
code('''
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    IN_COLAB = importlib.util.find_spec('google.colab') is not None
except ModuleNotFoundError:
    IN_COLAB = False

PROJECT_DIR = Path(os.environ.get('DATATHON_PROJECT_DIR', '/content/drive/MyDrive/Datathon' if IN_COLAB else '.')).resolve()
RUNTIME_DIR = Path('/content/datathon_runtime') if IN_COLAB else PROJECT_DIR
RUN_NAME = 'provisional_v1'  # Use a new name for a new experiment; preserve evaluated runs.
ANNOTATION_MODE = 'provisional'  # User-authorized spreadsheet baseline; use 'reviewed' after manual review.
INSTALL_DEPENDENCIES = IN_COLAB and os.environ.get('DATATHON_DEPS_READY') != '1'
REQUIRE_GPU = IN_COLAB
REBUILD_MANIFEST = False  # Only after dataset changes; re-review changed footage.

if IN_COLAB and str(PROJECT_DIR).startswith('/content/drive/') and not Path('/content/drive/MyDrive').exists():
    from google.colab import drive
    drive.mount('/content/drive')
assert (PROJECT_DIR / 'open/open_labels.xlsx').is_file(), f'Copy the complete project to {PROJECT_DIR}'
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
if PROJECT_DIR != RUNTIME_DIR:
    for directory in ('datathon', 'scripts', 'tests'):
        shutil.copytree(PROJECT_DIR / directory, RUNTIME_DIR / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copy2(PROJECT_DIR / 'requirements-colab.txt', RUNTIME_DIR)
    # Stage private footage on runtime storage; hashes are audited below.
    shutil.copytree(PROJECT_DIR / 'open', RUNTIME_DIR / 'open', dirs_exist_ok=True, copy_function=shutil.copy2)
os.chdir(RUNTIME_DIR)
sys.path.insert(0, str(RUNTIME_DIR))
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
    print('Dependencies installed. If prompted by Colab, restart the kernel and rerun from this cell.')
print('Project:', PROJECT_DIR, '| Runtime:', RUNTIME_DIR)
''', 'setup')
code('''
import json
import random
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from datathon.data import (build_manifest, create_review_table, fingerprint, load_reviewed,
                           read_labels, save_json, sha256_file, provisional_records)
from datathon.inference import Detector, export_artifacts, hardware_info
from datathon.training import LinearHead, evaluation_report, make_split, tune
from datathon.video import MODEL_ID, MODEL_REVISION, VideoConfig, XClipEncoder, cached_features

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError('Select a Colab GPU runtime before training. Set REQUIRE_GPU=False only for CPU diagnostics.')
for command in ('ffmpeg', 'ffprobe'):
    assert shutil.which(command), f'{command} must be installed'

DATASET_DIR = RUNTIME_DIR / 'open'
RUN_DIR = PROJECT_DIR / 'runs' / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = RUN_DIR / 'manifest.json'
REVIEW_PATH = RUN_DIR / 'annotations.csv'
CONFIG = VideoConfig()
save_json(RUN_DIR / 'runtime_hardware.json', hardware_info())
display(hardware_info())
''', 'setup')
md('''
## 1. Audit original data and review every unique recording

The initial audit fully decodes each unique recording. Subsequent runs verify the labels
and every source-file hash before reusing it. Duplicate annotations retain their provenance.
The supplied data should contain 42 files, 40 unique recordings, and 21 positive/19 negative
unique recordings. `Х-4`/`Х-47` is identical footage with annotations at 57 and 68 seconds.

**Scene groups:** assign the same group to recordings from the same staged scenario,
continuous session, or closely related takes. A shared camera angle alone need not imply
a shared group. Do not split related scenes to obtain better-looking validation results.
For a genuinely independent recording, use a unique group such as `scene_01`.
Use comma-separated event times in seconds; leave this field empty for negatives.
Document all corrections in Notes. Mark reviewed only after watching the entire recording.
''')
code('''
if MANIFEST_PATH.exists() and not REBUILD_MANIFEST:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest['source_label_sha256'] == sha256_file(DATASET_DIR / 'open_labels.xlsx'), 'Labels changed; re-audit deliberately'
    expected_names = set()
    for recording in manifest['recordings']:
        for alias in recording['aliases']:
            expected_names.add(alias)
            assert sha256_file(DATASET_DIR / 'videos' / alias) == recording['sha256'], f'Changed footage: {alias}'
    actual_names = {p.name for p in (DATASET_DIR / 'videos').iterdir() if p.suffix.lower() == '.mov'}
    assert expected_names == actual_names, 'Video inventory changed; re-audit deliberately'
else:
    manifest = build_manifest(DATASET_DIR, MANIFEST_PATH)

# Reuse an existing local review on the first Drive run, without overwriting edits.
seed_review = PROJECT_DIR / 'review/annotations.csv'
seed_manifest = PROJECT_DIR / 'review/manifest.json'
if not REVIEW_PATH.exists() and seed_review.exists() and seed_manifest.exists():
    old = json.loads(seed_manifest.read_text())
    if old == manifest:
        shutil.copy2(seed_review, REVIEW_PATH)
create_review_table(manifest, REVIEW_PATH)
summary = pd.DataFrame([{'aliases': '|'.join(r['aliases']), 'label': r['label'],
                        'duration_sec': r['duration_sec'], 'decode_status': r['decode_status'],
                        'source_events': [a['timestamp_sec'] for a in r['source_annotations'] if a['timestamp_sec'] is not None]}
                       for r in manifest['recordings']])
print(f"{manifest['source_video_count']} files -> {len(summary)} unique recordings")
print('Unique class counts:', summary.label.value_counts().to_dict())
display(summary)
print('Persistent review table:', REVIEW_PATH)
''', 'audit')
code('''
from datathon.review import review_widget
if IN_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()
display(review_widget(manifest, DATASET_DIR, REVIEW_PATH, RUNTIME_DIR / 'review_previews'))
''', 'review')
md('''
## 2. Annotation mode and immutable grouped split

The user has authorized a **provisional** run using the original spreadsheet labels.
This mode leaves manual review flags unchanged and groups exact duplicates only;
scene/session separation is unverified. Set `ANNOTATION_MODE='reviewed'` and a new
RUN_NAME for the reviewed experiment. In reviewed mode, finish and save all reviews above. CSV editing in Drive/VS Code is
also supported: `event_times_sec` is a JSON array, `reviewed` is `true`, and `group_id` is required.
The split reserves about 20% for final testing. The rest has four grouped folds.
Every fold must contain both classes. If the actual scene groups cannot support this design,
the notebook stops instead of weakening the leakage controls.
''')
code('''
if ANNOTATION_MODE == 'provisional':
    records = provisional_records(manifest)
    print('PROVISIONAL BASELINE: unreviewed labels; exact duplicate grouping only; scene separation unverified.')
elif ANNOTATION_MODE == 'reviewed':
    records = load_reviewed(manifest, REVIEW_PATH)
else:
    raise ValueError('ANNOTATION_MODE must be provisional or reviewed')
split = make_split(records, RUN_DIR / 'split.json')
lookup = {r['recording_id']: r for r in records}
print('Development:', len(split['development']), '| Test:', len(split['test']))
print('Reviewed-data fingerprint:', fingerprint(records))
display(pd.DataFrame([{'fold': i, 'train': len(f['train']), 'validation': len(f['validation'])}
                      for i, f in enumerate(split['folds'])]))
''', 'review-gate')
md('''
## 3. Frozen video features, persisted after each recording

Only model files and public metadata are downloaded; video content is processed inside this
runtime. Use full-frame letterboxing, eight frames/four seconds, and one-second scan steps.
Quarter-second PTS samples support later refinement. The cache identity includes the video
hash, immutable model revision, and preprocessing configuration; partial files are recomputed.
No event-type or face-recognition model is used.
''')
code('''
from dataclasses import asdict
from importlib import metadata as package_metadata
from huggingface_hub import hf_hub_download
import urllib.request

# Reopen the pinned online checkpoint if an earlier offline check set these flags.
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
encoder = XClipEncoder(config=CONFIG)
NOTICES_DIR = RUN_DIR / 'model_notices'
NOTICES_DIR.mkdir(exist_ok=True)
model_card = hf_hub_download(MODEL_ID, 'README.md', revision=MODEL_REVISION)
shutil.copy2(model_card, NOTICES_DIR / 'MODEL_CARD.md')
metadata_url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}'
with urllib.request.urlopen(metadata_url, timeout=30) as response:
    model_metadata = json.load(response)
assert model_metadata.get('cardData', {}).get('license') == 'mit', 'Recheck model licensing before use'
save_json(NOTICES_DIR / 'MODEL_METADATA.json', model_metadata)
license_url = 'https://raw.githubusercontent.com/microsoft/VideoX/master/LICENSE'
with urllib.request.urlopen(license_url, timeout=30) as response:
    (NOTICES_DIR / 'XCLIP_LICENSE').write_bytes(response.read())
save_json(NOTICES_DIR / 'SOURCES.json', {'model_metadata': metadata_url,
    'model_card': f'https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/README.md',
    'upstream_repository_license': license_url})

experiment_identity = {'dataset_fingerprint': fingerprint(records), 'annotation_mode': ANNOTATION_MODE, 'model_revision': MODEL_REVISION,
                       'video_config': asdict(CONFIG),
                       'precision': 'float16' if encoder.device.startswith('cuda') else 'float32',
                       'dependency_versions': {name: package_metadata.version(name)
                                               for name in ('torch', 'transformers', 'Pillow', 'av', 'numpy', 'scikit-learn')},
                       'source_hashes': {str(p.relative_to(RUNTIME_DIR)): sha256_file(p)
                                         for p in sorted((RUNTIME_DIR / 'datathon').glob('*.py'))}}
identity_path = RUN_DIR / 'experiment_identity.json'
if identity_path.exists():
    assert json.loads(identity_path.read_text()) == experiment_identity, 'Code/config/review changed: use a new RUN_NAME'
else:
    save_json(identity_path, experiment_identity)
features = {}
for index, recording_id in enumerate(split['development'], 1):
    features[recording_id] = cached_features(lookup[recording_id], DATASET_DIR, RUN_DIR / 'feature_cache', encoder)
    print(f"Development features {index}/{len(split['development'])}: {lookup[recording_id]['aliases']}")
''', 'features')
md('''
## 4. Train classifier; choose C and threshold using development OOF predictions

Positive training windows are centered within one second of reviewed events. Other windows
in positive recordings are ignored. Reviewed negative recordings provide negative windows.
Each recording receives equal total weight within its class; both classes receive equal total
weight. Standardization is fit only on each training fold. Select C from 0.01/0.1/1 and maximize
video-level F2, breaking ties by precision and then stronger regularization.
OOF scores are used for model selection; the untouched test set estimates final performance.
''')
code('''
selection_path = RUN_DIR / 'model_selection.json'
if selection_path.exists() and (RUN_DIR / 'head.json').exists():
    selection = json.loads(selection_path.read_text())
    assert selection['dataset_fingerprint'] == fingerprint(records)
    head = LinearHead.load(RUN_DIR / 'head.json')
    print('Resumed saved classifier and model selection')
else:
    head, selection = tune(records, features, split, RUN_DIR)
display(pd.DataFrame([{'C': c['C'], 'threshold': c['threshold'], **c['oof_metrics']}
                      for c in selection['candidates']]).drop(columns='confusion_matrix_labels_0_1'))
print('Selected C:', selection['selected_C'], '| Threshold:', selection['threshold'])
''', 'training')
md('''
## 5. Final held-out evaluation (no retuning afterward)

This evaluates the selected model once and saves the results for restart recovery.
Metrics include constant baselines, video mistakes, one-to-one event matching at two/five
seconds, and false events per negative video. These tolerances are not jury tolerances.
Evidence is extracted from original frames with actual PTS, not attention maps.
Cached-feature evaluation timing is **not** a deployment benchmark.
''')
code('''
EVALUATION_PATH = RUN_DIR / 'evaluation.json'
detector = Detector(encoder, head, selection['threshold'])
test_records = [lookup[i] for i in split['test']]
if EVALUATION_PATH.exists():
    evaluation = json.loads(EVALUATION_PATH.read_text())
    assert evaluation['experiment_identity'] == experiment_identity
    assert evaluation['classifier_sha256'] == sha256_file(RUN_DIR / 'head.json')
    assert evaluation['threshold'] == selection['threshold']
    print('Loaded saved held-out evaluation; do not tune against these results')
else:
    outputs = {}
    for recording in test_records:
        item = cached_features(recording, DATASET_DIR, RUN_DIR / 'feature_cache', encoder)
        output_path = RUN_DIR / 'test_outputs' / recording['recording_id'] / 'result.json'
        if output_path.exists():
            saved = json.loads(output_path.read_text())
            assert saved['experiment_identity'] == experiment_identity
            assert saved['classifier_sha256'] == sha256_file(RUN_DIR / 'head.json')
            assert saved['threshold'] == selection['threshold']
            result = saved['result']
        else:
            result = detector.predict(DATASET_DIR / recording['relative_path'], output_path.parent, cached=item)
            save_json(output_path, {'experiment_identity': experiment_identity,
                                   'classifier_sha256': sha256_file(RUN_DIR / 'head.json'),
                                   'threshold': selection['threshold'], 'result': result})
        outputs[recording['recording_id']] = result
    evaluation = evaluation_report(test_records, outputs, selection['threshold'])
    if ANNOTATION_MODE == 'provisional':
        evaluation['limitations'].append('PROVISIONAL: spreadsheet labels unreviewed; scene/session separation unverified.')
    evaluation.update(experiment_identity=experiment_identity, annotation_mode=ANNOTATION_MODE, threshold=selection['threshold'],
                      classifier_sha256=sha256_file(RUN_DIR / 'head.json'), test_only=True)
    save_json(EVALUATION_PATH, evaluation)
display(pd.DataFrame({'model': evaluation['model'], **evaluation['baselines']}).drop(index='confusion_matrix_labels_0_1'))
display(evaluation['events'])
display(pd.DataFrame(evaluation['mistakes']))
''', 'evaluation')
code('''
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
predictions = evaluation['predictions']
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(np.array(evaluation['model']['confusion_matrix_labels_0_1']),
                       display_labels=['No violation', 'Violation']).plot(ax=axes[0], colorbar=False)
PrecisionRecallDisplay.from_predictions([r['label'] for r in predictions], [r['score'] for r in predictions], ax=axes[1])
fig.tight_layout()
fig.savefig(RUN_DIR / 'heldout_metrics.png', dpi=160)
plt.show()
''', 'evaluation')
md('''
## 6. Export offline bundle and verify in a fresh process

The bundle includes the pinned encoder, JSON classifier parameters (no pickle), preprocessing,
threshold, source package, exact installed dependency versions, split/provenance, evaluation,
third-party notices, and file checksums. It keeps the development-trained model; test recordings
are not folded back into training. Install dependencies and ffmpeg before disconnecting.

The verification process blocks Python socket connections before loading the model and uses
local-only model loading. This is an application-level check, not an OS network isolation claim.
It does not use feature caches. It records cold-load/inference time and hardware on the longest
recording, then repeats in a second fresh process and checks output consistency.
''')
code('''
ARTIFACT_DIR = RUN_DIR / 'exported_model'
if not ARTIFACT_DIR.exists():
    export_artifacts(encoder, head, selection, split, evaluation, ARTIFACT_DIR, RUN_DIR, NOTICES_DIR)
else:
    from datathon.inference import verify_export
    verify_export(ARTIFACT_DIR)
    exported_config = json.loads((ARTIFACT_DIR / 'inference_config.json').read_text())
    assert exported_config['dataset_fingerprint'] == fingerprint(records)
    assert exported_config['threshold'] == selection['threshold']
    assert sha256_file(ARTIFACT_DIR / 'head.json') == sha256_file(RUN_DIR / 'head.json')

# Benchmark runtime-local files so network-backed Drive I/O does not define inference latency.
LOCAL_EXPORT = RUNTIME_DIR / 'offline_bundle'
if LOCAL_EXPORT.resolve() != ARTIFACT_DIR.resolve():
    shutil.copytree(ARTIFACT_DIR, LOCAL_EXPORT, dirs_exist_ok=True)
longest = max(records, key=lambda r: r['duration_sec'])
video_path = (DATASET_DIR / longest['relative_path']).resolve()
# Release GPU encoder before loading another model in the verification subprocess.
del detector, encoder
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
benchmark_reports = []
for attempt in range(2):
    report_path = RUN_DIR / f'offline_check_{attempt+1}.json'
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_HOME': str(RUNTIME_DIR / 'empty_hf_cache'), 'PYTHONPATH': str(LOCAL_EXPORT)}
    completed = subprocess.run([sys.executable, '-m', 'datathon.cli', str(video_path),
                                '--artifacts', str(LOCAL_EXPORT), '--offline-check', '--report', str(report_path)],
                               cwd=LOCAL_EXPORT, env=environment, capture_output=True, text=True, timeout=600)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-6000:])
    benchmark_reports.append(json.loads(report_path.read_text()))
first, second = benchmark_reports
assert first['verdict'] == second['verdict']
np.testing.assert_allclose(first['score'], second['score'], atol=1e-5)
np.testing.assert_allclose([e['timecode_sec'] for e in first['events']],
                           [e['timecode_sec'] for e in second['events']], atol=0.25)
display(pd.DataFrame([{'attempt': i+1, 'cold_load_and_inference_sec': r['cold_load_and_inference_sec'],
                      'within_5_minute_budget': r['within_5_minute_budget'], 'hardware': r['hardware']}
                     for i, r in enumerate(benchmark_reports)]))
if not all(r['within_5_minute_budget'] for r in benchmark_reports):
    print('Five-minute latency acceptance FAILED on this hardware; optimize and evaluate as a new experiment.')
print('Export:', ARTIFACT_DIR)
''', 'export')
md('''
## 7. Use the exported model on a new video

Run `analyze_video(video_path, artifact_dir)` from the exported bundle directory, or use
`python -m datathon.cli VIDEO.mov --artifacts EXPORTED_MODEL --offline-check`.
It returns a binary verdict, uncalibrated score, and an `events` list containing
`timecode_sec`, scores, and timestamped JPEG paths. It writes `result.json` alongside evidence.

For another local video, edit `NEW_VIDEO` below. Leave it unset to avoid extra inference.
The model may miss unfamiliar actions and angles; evidence still needs human review.
''')
code('''
NEW_VIDEO = None  # Example: Path('/content/new_video.mov')
if NEW_VIDEO is not None:
    from datathon.inference import analyze_video
    result = analyze_video(NEW_VIDEO, LOCAL_EXPORT)
    display(result)
''', 'inference')

notebook = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                         'colab': {'provenance': []}}, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:03d}'
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell-{i}', 'exec')
Path(__file__).resolve().parents[1].joinpath('case.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print(f'Wrote {len(cells)} cells')
