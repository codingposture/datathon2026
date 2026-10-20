import ast
import json
from pathlib import Path


def test_notebook_schema_and_code():
    import nbformat
    path = Path(__file__).resolve().parents[1] / 'case.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert not cell.outputs
    assert any('review-gate' in cell.metadata.get('tags', []) for cell in notebook.cells)


def test_fresh_notebook_audits_hybrid_and_resumes(tmp_path):
    """Execute actual notebook setup/audit twice in fresh Python kernels."""
    import os
    import sys
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / 'project'
    project.mkdir()
    for name in ('open', 'review', 'datathon', 'scripts', 'tests', 'requirements-colab.txt'):
        (project / name).symlink_to(root / name, target_is_directory=(root / name).is_dir())
    kernels = tmp_path / 'kernels'
    spec = kernels / 'datathon-test'
    spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Datathon test', 'language': 'python',
        'env': {'DATATHON_PROJECT_DIR': str(project), 'MPLBACKEND': 'Agg'}}))
    manager = KernelSpecManager(kernel_dirs=[str(kernels)])
    original = nbformat.read(root / 'case.ipynb', as_version=4)
    selected = []
    for cell in original.cells:
        tags = cell.metadata.get('tags', [])
        if 'review' not in tags:
            cell = cell.copy()
            selected.append(cell)
        if 'review-gate' in tags:
            break
    for attempt in range(2):
        notebook = nbformat.v4.new_notebook(cells=selected)
        km = KernelManager(kernel_name='datathon-test', kernel_spec_manager=manager)
        client = NotebookClient(notebook, km=km, timeout=120, allow_errors=True,
                                resources={'metadata': {'path': str(project)}})
        try:
            executed = client.execute()
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        errors = [output for cell in executed.cells if cell.cell_type == 'code'
                  for output in cell.outputs if output.output_type == 'error']
        assert errors == [], errors
        saved = project / 'runs/hybrid_v2'
        assert len(json.loads((saved / 'typed_manifest.json').read_text())['recordings']) == 40
        split = json.loads((saved / 'split.json').read_text())
        assert len(split['development']) == 32 and len(split['test']) == 8
        assert (saved / 'experiment_identity.json').exists()
