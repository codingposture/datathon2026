"""Generate the restartable typed hybrid notebook."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': dedent(source).strip().splitlines(True)})


def code(source, tag):
    source = dedent(source).strip()
    compile(source, tag, 'exec')
    cells.append({'cell_type': 'code', 'metadata': {'tags': [tag]}, 'source': source.splitlines(True),
                  'execution_count': None, 'outputs': []})


md('''
# Typed hybrid detector — experimental v2

Compare full-forward zero-shot X-CLIP, a frozen-feature five-class classifier and person-assisted
decisions. Return four broad violation categories, timestamps and original JPEG evidence.
No legal conclusions, subtype classification or face identification. Scores are uncalibrated.

**Before starting the training kernel:** install requirements-colab.txt in a separate process.
The supplied scripts/run_colab_notebook.py handles this automatically. In browser Colab,
install in a disposable setup kernel, restart, then run this notebook. Keep Colab's compatible
CUDA torch/torchvision pair. Use a T4 GPU and private Drive storage for persistent caches.
Package the complete project with python scripts/package_colab.py --include-data.
The VS Code launcher also supports runtime storage and retrieves the final bundle locally.

Preserve the previous model. Labels remain provisional. The reused eight-video test is
diagnostic, not untouched, and has no cash/workstation-security positives. Only one
personal-items recording is available for development. Other scene overlap remains unverified.
''')
code('''
import importlib.util
import os
from pathlib import Path
import shutil
import sys
try:
    IN_COLAB = importlib.util.find_spec('google.colab') is not None
except ModuleNotFoundError:
    IN_COLAB = False
PROJECT_DIR = Path(os.environ.get('DATATHON_PROJECT_DIR', '/content/drive/MyDrive/Datathon' if IN_COLAB else '.')).resolve()
RUN_NAME = 'hybrid_v2'
if IN_COLAB and str(PROJECT_DIR).startswith('/content/drive/') and not Path('/content/drive/MyDrive').exists():
    from google.colab import drive
    drive.mount('/content/drive')
assert (PROJECT_DIR / 'open/forte_labels_annotated.xlsx').is_file(), 'Stage the annotated workbook'
assert (PROJECT_DIR / 'review/legacy_split.json').is_file(), 'Preserved partition required'
RUNTIME_DIR = Path('/content/datathon_hybrid_runtime') if IN_COLAB else PROJECT_DIR
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
if RUNTIME_DIR != PROJECT_DIR:
    for directory in ('datathon', 'open'):
        shutil.copytree(PROJECT_DIR / directory, RUNTIME_DIR / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
sys.path.insert(0, str(RUNTIME_DIR))
os.chdir(RUNTIME_DIR)
print('Persistent project:', PROJECT_DIR, '| Runtime footage:', RUNTIME_DIR / 'open')
''', 'setup')
md('''
## Workbook audit and preserved split

Import named sheets/headers as inert data; retain all other columns as provenance. Verify every
video hash and decoding audit. Consolidate 42 files into 40 recordings. X-4/X-47 has a 53–57s
uncertain cash event and a separate 68s count-skipped event. HIGH/MEDIUM/LOW weights are
1/.5/.25; original negatives weigh 1. Untimed secondary events are prompt vocabulary only.
Keep the original 32/8 partition; rebuild four development folds grouping X-44/X-45 together.
''')
code('''
import json
import pandas as pd
from IPython.display import display
from datathon.hybrid_experiment import Experiment
experiment = Experiment(PROJECT_DIR, RUNTIME_DIR / 'open', RUN_NAME, require_gpu=IN_COLAB)
display(pd.DataFrame([{'aliases': '|'.join(r['aliases']), 'label': r['label'],
                      'duration_sec': r['duration_sec'], 'events': r['events'], 'group': r['group_id']}
                     for r in experiment.records]))
print(len(experiment.records), 'unique recordings; development/test:',
      len(experiment.split['development']), len(experiment.split['test']))
display(pd.DataFrame([{'fold': i, 'train': len(f['train']), 'validation': len(f['validation']),
                      'missing_training_categories': f['missing_training_categories']}
                     for i, f in enumerate(experiment.split['folds'])]))
''', 'review-gate')
md('''
## Optional camera controls (local, human confirmed)

Watch recordings in your local player or datathon.review.review_widget. Mark the workstation
ROI in fractions of original image width/height. Confirm cashier-present and departure examples
only after watching them. No defaults are confirmed. The departure-category checkbox applies
only where all security candidates being filtered require departure. RT-DETR detects people,
not keys, cash or cabinet state. Unknown/unconfirmed camera profiles disable filtering.
Complete these controls before model selection; headless execution skips this cell.
''')
code('''
from datathon.tracking import camera_profile_widget
display(camera_profile_widget(PROJECT_DIR / 'review/camera_profile.json',
                              [experiment.lookup[k] for k in experiment.split['development']], experiment.dataset))
''', 'review')
md('''
## Frozen feature caches

Compare 4s and 12s contexts, eight full-frame letterboxed samples, one-second scan steps.
Use presentation timestamps, including quarter-second samples for refinement. Each cache key
contains video hash, pinned revision, preprocessing and dependency versions. Restarting verifies
the immutable run identity and resumes completed caches; changed code/data needs a new run name.
''')
code('experiment.development_features()', 'features')
md('''
## Compare standalone branches and blends using development OOF predictions only

Zero-shot uses XCLIPModel's complete text/video forward with video-specific prompting.
Prompts contain fixed taxonomy text plus training-fold-only descriptions and compliant contrasts.
Validation/test descriptions, filenames and annotated timestamps are never inference inputs.
The supervised branch learns normal plus four categories from frozen features. Train positive
centers within 1s of timed events/intervals; ignore other positive footage and incompatible
overlaps. Balance recordings, retain certainty discounts, fit scaling on training windows only.
Missing supervised classes produce no learned score; intermediate blends use zero-shot fallback.

Search C=.001/.01/.1 and supervised blend weights 0/.25/.5/.75/1. Select global threshold and
components by binary balanced accuracy, then category macro-F1, then fewer false-positive videos.
Two consecutive positive scan windows are required. These OOF metrics are selection results.
''')
code('''
selection = experiment.standalone_and_blends()
comparison = json.loads((experiment.run / 'standalone_comparison.json').read_text())
display(pd.DataFrame([{**{k: c[k] for k in ('branch', 'window_sec', 'selected_C', 'supervised_weight', 'threshold')},
                       **{k: c['metrics'][k] for k in ('balanced_accuracy', 'category_macro_f1', 'false_positive_videos')}}
                      for c in comparison['candidates']]))
display(selection)
''', 'training')
md('''
## Detector experiment after standalone comparison

Apache-2.0 RT-DETR-R18 person detection with conservative IoU tracking measures stable departures.
Missing tracks, competing matches, multiple ROI occupants or uncertain roles disable filtering.
Departure cannot create an event. Retain assistance only if development balanced accuracy improves
without decreasing category macro-F1. Without a confirmed camera, person observations are measured
but filtering remains disabled; this does not establish a detector benefit.
''')
code('''
detector_results = experiment.detector_experiment()
display({k: v for k, v in detector_results.items() if k != 'observations'})
''', 'detector')
md('''
## Evaluate fixed configuration once on reused diagnostic test

Refine candidates at .25s centers within ±2s, merge nearby events within the same category only,
and save up to three original JPEG evidence frames with actual PTS. Report binary/category
confusion matrices, balanced accuracy, precision/recall/F1/F2, average precision, macro-F1,
constant baselines, mistakes and a higher-confidence subset. Typed one-to-one temporal matching
uses 2s/5s diagnostic tolerances and distance to uncertain intervals. Untimed metadata is excluded.
No cash/workstation-security positives occur in this reused test; its recall is unavailable.
''')
code('''
evaluation = experiment.fit_final_and_evaluate()
display(evaluation['model'])
display(evaluation['higher_confidence_subset'])
display(evaluation['events_2_sec'])
display(evaluation['events_5_sec'])
display(pd.DataFrame(evaluation['mistakes']))
''', 'evaluation')
md('''
## Separate hybrid_v2 export and fresh-process offline verification

Save weights/tokenizer, classifier/scaler arrays, prompts, taxonomy, camera config, dependencies,
licenses, reports and checksums. Keep the old model. Two fresh processes with local-only loads,
empty HF caches and Python socket denial analyze the longest recording without feature caches.
This is an application-level network check, not OS isolation. Record hardware, consistency and
the five-minute latency result. A T4 result does not establish speed on another machine.
The launcher retrieves the archive and checks SHA256; a separate local CPU smoke check is required.
Results remain explicitly experimental, without a minimum accuracy or readiness claim.
''')
code('''
ARTIFACT_DIR = experiment.export_and_verify()
display(json.loads((experiment.run / 'latency_acceptance.json').read_text()))
print('Experimental bundle:', ARTIFACT_DIR)
''', 'export')
code('''
NEW_VIDEO = None
if NEW_VIDEO is not None:
    from datathon.inference import analyze_video
    display(analyze_video(NEW_VIDEO, ARTIFACT_DIR, camera_profile=None))
''', 'inference')

for index, cell in enumerate(cells):
    cell['id'] = f'hybrid-{index:03d}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'provenance': []}}, 'cells': cells}
Path(__file__).resolve().parents[1].joinpath('case.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
print('Wrote', len(cells), 'cells')
