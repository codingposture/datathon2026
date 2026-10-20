"""Add full experiment/recovery reports and portable diagnostic evidence to the bundle.

No fitting, inference or selection is performed here. Run after the notebook completes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/content/Datathon'))
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from datathon.data import save_json, sha256_file
    from datathon.inference import verify_export
    run = root / 'runs/hybrid_v2'
    progress = json.loads((run / 'progress.json').read_text())
    if progress['status'] != 'done':
        raise ValueError('Complete notebook execution before finalizing the bundle')
    bundle = run / 'exported_model'
    verify_export(bundle)
    recovery = json.loads((run / 'cache_recovery_check.json').read_text())
    if not recovery['passed']:
        raise ValueError('Fresh-process cache recovery verification failed')
    reports = ('standalone_comparison.json', 'development_evaluation.json', 'fold_training_audit.json',
               'final_training_audit.json', 'camera_profile_identity.json', 'cache_recovery_check.json')
    for name in reports:
        shutil.copy2(run / name, bundle / name)
    comparison = json.loads((run / 'standalone_comparison.json').read_text())
    development = json.loads((run / 'development_evaluation.json').read_text())
    hardware = json.loads((run / 'runtime_hardware.json').read_text())
    latency = json.loads((run / 'latency_acceptance.json').read_text())
    notes = ['\n## Best development results by branch\n',
             '| Branch | Context | Balanced accuracy | Category macro-F1 |', '|---|---:|---:|---:|']
    for branch, candidate in comparison['standalone_best'].items():
        notes.append(f"| {branch} | {candidate['window_sec']}s | {candidate['metrics']['balanced_accuracy']:.4f} | "
                     f"{candidate['metrics']['category_macro_f1']:.4f} |")
    notes.extend(['', 'These are OOF selection results, not independent performance estimates.',
                  '', '## Development category results', '',
                  '| Category | Positive videos | Recall | F1 |', '|---|---:|---:|---:|'])
    for category, metrics in development['model']['category_results'].items():
        recall = 'unavailable' if metrics['recall'] is None else f"{metrics['recall']:.4f}"
        notes.append(f"| {category} | {metrics['positive_support']} | {recall} | {metrics['f1']:.4f} |")
    notes.extend(['', '## Execution verification', '',
                  f"Training/experiments ran on Google Colab, {hardware['gpu']}, PyTorch {hardware['torch']}.",
                  'The X-CLIP encoder remained frozen. The selected branch is recorded above; '
                  'classifier weights are included for reproducibility even when the selected path is zero-shot.',
                  f"Longest-video cold-load/inference times: {', '.join(format(c['cold_load_and_inference_sec'], '.2f') + 's' for c in latency['checks'])}. "
                  f"Five-minute acceptance on this hardware: {latency['passed']}.",
                  f"Fresh-process offline cache recovery: {recovery['feature_caches_loaded']} feature caches and "
                  f"{recovery['zero_shot_caches_loaded']} zero-shot caches loaded with recomputation forbidden.",
                  'This run used Colab runtime storage and local checkpoint backups; Drive was not mounted.'
                  if not Path('/content/drive/MyDrive').is_dir() else 'Drive was mounted during finalization.',
                  'The local download receives a separate CPU load-only check; no local training is performed.', ''])
    (bundle / 'MODEL_REPORT.md').write_text((run / 'MODEL_REPORT.md').read_text() + '\n'.join(notes))
    experiment_files = bundle / 'experiment'
    experiment_files.mkdir(exist_ok=True)
    shutil.copy2(root / 'case.ipynb', experiment_files / 'case.ipynb')
    shutil.copy2(run / 'case.executed.ipynb', experiment_files / 'case.executed.ipynb')
    shutil.copy2(root / 'requirements-colab.txt', experiment_files / 'requirements-colab.txt')
    folds = bundle / 'development_folds'
    folds.mkdir(exist_ok=True)
    for path in run.glob('prompts_*s_fold*.json'):
        shutil.copy2(path, folds)
    shutil.copytree(run / 'heads', folds / 'heads', dirs_exist_ok=True)
    evidence = bundle / 'diagnostic_outputs'
    shutil.copytree(run / 'test_outputs', evidence, dirs_exist_ok=True)
    for path in evidence.glob('*/result.json'):
        saved = json.loads(path.read_text())
        for event in saved['result']['events']:
            for frame in event['evidence_frames']:
                frame['image_path'] = str(Path('diagnostic_outputs') / path.parent.name / Path(frame['image_path']).name)
        saved['evidence_path_base'] = 'artifact_root'
        save_json(path, saved)
    scripts = bundle / 'scripts'
    scripts.mkdir(exist_ok=True)
    for name in ('finalize_colab_bundle.py', 'check_colab_recovery.py'):
        shutil.copy2(root / 'scripts' / name, scripts / name)
    checksums = {str(p.relative_to(bundle)): sha256_file(p) for p in bundle.rglob('*')
                 if p.is_file() and p.name != 'COMPLETE.json'}
    save_json(bundle / 'COMPLETE.json', {'schema_version': 2, 'sha256': checksums})
    verified = verify_export(bundle)
    archive = root / 'hybrid_v2.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(bundle))
    parts = root / 'hybrid_v2_download_parts'
    parts.mkdir(exist_ok=True)
    descriptors = []
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(64 * 1024 * 1024):
            path = parts / f'model.part{index:03d}'
            path.write_bytes(block)
            descriptors.append({'path': str(path), 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
            index += 1
    progress.update(archive=str(archive), archive_sha256=sha256_file(archive), archive_bytes=archive.stat().st_size,
                    download_parts=descriptors, full_reports_included=True, verified_artifact_files=verified)
    save_json(run / 'progress.json', progress)
    print(json.dumps({'finalized': True, 'verified_files': verified, 'archive_bytes': archive.stat().st_size,
                      'archive_sha256': progress['archive_sha256']}))


if __name__ == '__main__':
    main()
