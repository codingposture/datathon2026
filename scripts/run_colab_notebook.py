"""Execute case.ipynb on a fresh kernel inside an already provisioned Colab VM.

Launch remotely with: python scripts/run_colab_notebook.py --project /content/Datathon
Progress, completed notebook, logs, and the trained bundle stay under runs/hybrid_v2.
"""
import argparse
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
    run = root / 'runs/hybrid_v2'
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
        save({'status': 'verifying_cache_recovery', 'elapsed_sec': time.time() - started})
        verification_env = {**os.environ, 'USE_TF': '0'}
        subprocess.run([sys.executable, str(root / 'scripts/check_colab_recovery.py'), '--project', str(root)],
                       cwd=root, env=verification_env, check=True)
        # Finalization assembles full reports, portable evidence and checksum-verified parts.
        # Download requires full_reports_included, so this intermediate state is not downloadable.
        save({'status': 'done', 'elapsed_sec': time.time() - started})
        subprocess.run([sys.executable, str(root / 'scripts/finalize_colab_bundle.py'), '--project', str(root)],
                       cwd=root, env=verification_env, check=True)
        final = json.loads(progress.read_text())
        print('TRAINING_COMPLETE', final['archive_sha256'], final['archive'], flush=True)
    except Exception as error:
        nbformat.write(notebook, run / 'case.executed.ipynb')
        save({'status': 'failed', 'elapsed_sec': time.time() - started,
              'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
