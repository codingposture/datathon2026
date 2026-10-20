"""Execute case.ipynb on a fresh kernel inside an already provisioned Colab VM.

Launch remotely with: python scripts/run_colab_notebook.py --project /content/Datathon
Progress, completed notebook, logs, and the trained bundle stay under runs/provisional_v1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    run = root / 'runs/provisional_v1'
    run.mkdir(parents=True, exist_ok=True)
    os.environ['DATATHON_PROJECT_DIR'] = str(root)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PIP_PROGRESS_BAR'] = 'off'
    # Colab preloads NumPy: install before starting the notebook kernel, never mid-kernel.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                    str(root / 'requirements-colab.txt')], check=True)
    os.environ['DATATHON_DEPS_READY'] = '1'
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(root / 'case.ipynb', as_version=4)
    # The CSV remains unreviewed; the explicit provisional mode is selected in the notebook.
    notebook.cells = [cell for cell in notebook.cells if 'review' not in cell.metadata.get('tags', [])]
    progress = run / 'progress.json'
    started = time.time()
    def save(state):
        temporary = progress.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(progress)
    def on_start(cell, cell_index, **kwargs):
        state = {'status': 'running', 'cell_index': cell_index,
                 'tags': cell.metadata.get('tags', []), 'elapsed_sec': time.time() - started}
        save(state)
        print(json.dumps(state), flush=True)
    def on_complete(cell, cell_index, **kwargs):
        nbformat.write(notebook, run / 'case.executed.ipynb')
    save({'status': 'starting', 'pid': os.getpid()})
    try:
        client = NotebookClient(notebook, timeout=7200, kernel_name='datathon-training',
                                resources={'metadata': {'path': str(root)}},
                                on_cell_start=on_start, on_cell_executed=on_complete)
        client.execute()
        nbformat.write(notebook, run / 'case.executed.ipynb')
        save({'status': 'packaging', 'elapsed_sec': time.time() - started})
        import shutil
        archive = Path(shutil.make_archive(str(root / 'trained_model_provisional'), 'zip', run / 'exported_model'))
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        parts_dir = root / 'model_download_parts'
        parts_dir.mkdir(exist_ok=True)
        parts = []
        with archive.open('rb') as stream:
            index = 0
            while block := stream.read(64 * 1024 * 1024):
                part = parts_dir / f'model.part{index:03d}'
                part.write_bytes(block)
                parts.append({'path': str(part), 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
                index += 1
        save({'status': 'done', 'elapsed_sec': time.time() - started, 'archive': str(archive),
              'archive_sha256': digest, 'archive_bytes': archive.stat().st_size, 'download_parts': parts})
        print('TRAINING_COMPLETE', digest, str(archive), flush=True)
    except Exception as error:
        nbformat.write(notebook, run / 'case.executed.ipynb')
        save({'status': 'failed', 'elapsed_sec': time.time() - started,
              'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
