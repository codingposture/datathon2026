"""Upload private data, execute case.ipynb on the selected Colab GPU, retrieve exports.

Uses only the local VS Code bridge and its user-authorized Jupyter/Colab providers.
No Google login tokens are read. Run with --launch, --status, or --download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import time
from urllib.parse import quote
import uuid
import zipfile

from colab_control import ROOT, request

RUN = ROOT / 'runs/colab_remote'
REMOTE = '/content/Datathon'
STATE = RUN / 'connection.json'


def execute(code, timeout=90):
    return request({'action': 'execute', 'code': code}, timeout)


def parse_payload(reply, prefix):
    text = ''.join(item['text'] for item in reply['outputs'] if 'stdout' in item['mime'])
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError(f'Missing {prefix} payload')


def known_colab_handles():
    config = Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config')))
    database = config / 'Code/User/globalStorage/state.vscdb'
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
        raw = connection.execute("SELECT value FROM ItemTable WHERE key='ms-toolsai.jupyter'").fetchone()
    state = json.loads(raw[0])
    providers = next(v for k, v in state.items() if 'STORING_USED_JUPYTER_PROVIDERS' in k)
    entries = sorted(providers, key=lambda item: item.get('time', 0), reverse=True)
    return list(dict.fromkeys(item['serverHandle']['handle'] for item in entries
                              if item['serverHandle']['extensionId'] == 'google.colab'))


def remote_uri(handle, remote_path):
    return 'colab://' + handle + quote(remote_path, safe='/')


def transfer(handle, remote_path, local_path, direction, timeout=180):
    return request({'action': 'transfer', 'direction': direction,
                    'remoteUri': remote_uri(handle, remote_path),
                    'localPath': str(Path(local_path).resolve().relative_to(ROOT))}, timeout)


def launch():
    RUN.mkdir(parents=True, exist_ok=True)
    marker = 'datathon_' + uuid.uuid4().hex
    code = f'''import json, shutil, subprocess
from pathlib import Path
assert shutil.which('nvidia-smi'), 'Selected Colab kernel is still CPU-only'
print('DATATHON_GPU_INFO ' + json.dumps({{'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True).strip()}}))
Path('/content/{marker}').write_text('{marker}')
'''
    gpu = parse_payload(execute(code), 'DATATHON_GPU_INFO ')
    selected = None
    for handle in known_colab_handles():
        destination = RUN / ('marker_' + marker)
        try:
            transfer(handle, '/content/' + marker, destination, 'download', 45)
            if destination.read_text() == marker:
                selected = handle
                break
        except RuntimeError:
            continue
    if not selected:
        raise RuntimeError('Could not match the active GPU kernel to a Colab filesystem provider')
    state = {'handle': selected, 'gpu': gpu, 'remote_project': REMOTE}
    STATE.write_text(json.dumps(state, indent=2))
    STATE.chmod(0o600)
    subprocess.run([sys.executable, str(ROOT / 'scripts/package_colab.py'),
                    '--output', str(ROOT / 'Datathon-colab-source.zip')], check=True)
    transfer(selected, '/content/datathon_payload.zip', ROOT / 'Datathon-colab-source.zip', 'upload')
    execute("""from pathlib import Path
import zipfile
with zipfile.ZipFile('/content/datathon_payload.zip') as archive:
    assert all(name.startswith('Datathon/') and '..' not in Path(name).parts for name in archive.namelist())
    archive.extractall('/content')
Path('/content/Datathon/open/videos').mkdir(parents=True, exist_ok=True)
print('Source staged')
""")
    files = sorted((ROOT / 'open').glob('*.xlsx')) + sorted((ROOT / 'open/videos').glob('*.mov'))
    inventory = parse_payload(execute("""from pathlib import Path
import hashlib,json
root=Path('/content/Datathon')
result={}
for p in (root/'open').rglob('*'):
 if p.is_file():
  with p.open('rb') as stream: result[p.relative_to(root).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
print('DATATHON_INVENTORY '+json.dumps(result))
"""), 'DATATHON_INVENTORY ')
    for index, local in enumerate(files, 1):
        relative = local.relative_to(ROOT).as_posix()
        local_digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if inventory.get(relative) == local_digest:
            print(f'RESUMED {index}/{len(files)} {local.name}', flush=True)
            continue
        for attempt in range(3):
            try:
                transferred = transfer(selected, REMOTE + '/' + relative, local, 'upload')
                assert transferred['sha256'] == local_digest
                break
            except (RuntimeError, TimeoutError) as error:
                if attempt == 2:
                    raise
                print(f'Retrying {local.name}: {error}', flush=True)
        print(f'UPLOADED {index}/{len(files)} {local.name}', flush=True)
    # Provision notebook execution support in a separate process, leaving the user's kernel intact.
    execute("""import subprocess, sys
from pathlib import Path
bootstrap = Path('/content/Datathon/runs/remote_bootstrap.py')
bootstrap.parent.mkdir(parents=True, exist_ok=True)
bootstrap.write_text('''import subprocess,sys,os
subprocess.run([sys.executable,'-m','pip','install','-q','nbclient==0.10.2','nbformat==5.10.4','ipykernel==6.30.1'],check=True)
from ipykernel.kernelspec import install
install(user=True,kernel_name='datathon-training',display_name='Datathon training')
os.execv(sys.executable,[sys.executable,'/content/Datathon/scripts/run_colab_notebook.py','--project','/content/Datathon'])
''')
log = open('/content/Datathon/runs/hybrid_v2_training.log','a',buffering=1)
process = subprocess.Popen([sys.executable,str(bootstrap)],cwd='/content/Datathon',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('DATATHON_JOB ' + __import__('json').dumps({'pid':process.pid,'log':'/content/Datathon/runs/hybrid_v2_training.log'}))
""")
    print('Training launched on', gpu, flush=True)


def status():
    reply = execute("""from pathlib import Path
p=Path('/content/Datathon/runs/hybrid_v2/progress.json')
print('DATATHON_PROGRESS ' + (p.read_text() if p.exists() else '{"status":"bootstrapping"}').replace('\\n',' '))
log=Path('/content/Datathon/runs/hybrid_v2_training.log')
print(log.read_text(errors='replace')[-2500:] if log.exists() else 'No log yet')
""")
    return parse_payload(reply, 'DATATHON_PROGRESS ')


def download():
    state = json.loads(STATE.read_text())
    progress = status()
    if progress['status'] != 'done':
        raise RuntimeError('Training is not complete; refusing to download an incomplete model')
    if not progress.get('full_reports_included'):
        raise RuntimeError('Model reports and cache verification are still being finalized')
    archive = ROOT / 'artifacts/hybrid_v2.zip'
    if progress.get('download_parts'):
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open('wb') as destination:
            for index, part in enumerate(progress['download_parts'], 1):
                local_part = ROOT / 'artifacts/hybrid_v2_download_parts' / Path(part['path']).name
                if local_part.exists():
                    with local_part.open('rb') as stream:
                        valid = hashlib.file_digest(stream, 'sha256').hexdigest() == part['sha256']
                else:
                    valid = False
                if not valid:
                    for attempt in range(3):
                        try:
                            reply = transfer(state['handle'], part['path'], local_part, 'download', 300)
                            assert reply['sha256'] == part['sha256']
                            break
                        except (RuntimeError, TimeoutError):
                            if attempt == 2:
                                raise
                            print(f'Retrying model part {index}', flush=True)
                with local_part.open('rb') as stream:
                    shutil.copyfileobj(stream, destination, 4 * 1024 * 1024)
                print(f"Downloaded model part {index}/{len(progress['download_parts'])}", flush=True)
    else:
        reply = transfer(state['handle'], progress['archive'], archive, 'download', 600)
        assert reply['sha256'] == progress['archive_sha256']
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == progress['archive_sha256']
    archive.with_suffix('.zip.sha256').write_text(progress['archive_sha256'] + '  ' + archive.name + '\n')
    output = ROOT / 'artifacts/hybrid_v2'
    if output.exists():
        raise FileExistsError('Local model folder already exists; preserve or archive it before downloading again')
    with zipfile.ZipFile(archive) as bundle:
        assert all(not name.startswith('/') and '..' not in Path(name).parts for name in bundle.namelist())
        bundle.extractall(output)
    for remote, local in [('runs/hybrid_v2/case.executed.ipynb', 'case.hybrid_v2.colab.executed.ipynb'),
                          ('runs/hybrid_v2_training.log', 'hybrid_v2_training.log'),
                          ('runs/hybrid_v2/offline_check_1.json', 'hybrid_v2_offline_check_1.json'),
                          ('runs/hybrid_v2/offline_check_2.json', 'hybrid_v2_offline_check_2.json'),
                          ('runs/hybrid_v2/latency_acceptance.json', 'hybrid_v2_latency_acceptance.json')]:
        transfer(state['handle'], REMOTE + '/' + remote, ROOT / 'artifacts' / local, 'download')
    sys.path.insert(0, str(ROOT))
    from datathon.inference import verify_export
    print('Verified local artifact files:', verify_export(output))
    # Local load-only verification. Training and video benchmarks remain on Colab.
    cpu_report = ROOT / 'artifacts/hybrid_v2_cpu_check.json'
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(output),
                   'HF_HOME': str(ROOT / 'runs/hybrid_cpu_empty_cache'), 'USE_TF': '0'}
    load_code = '''import sys,time
from pathlib import Path
from datathon.data import save_json
from datathon.inference import Detector,deny_network,hardware_info,verify_export
root=Path(sys.argv[1])
with deny_network():
 verified=verify_export(root)
 started=time.perf_counter()
 detector=Detector.from_artifacts(root,device='cpu')
 elapsed=time.perf_counter()-started
 assert next(detector.encoder.model.parameters()).device.type=='cpu'
 report={'passed':True,'check':'fresh_process_offline_cpu_loading_only','load_sec':elapsed,
         'device':'cpu','python_socket_guard_enabled':True,'verified_artifact_files':verified,
         'selected_branch':detector.selection['branch'],'hardware':hardware_info(),
         'training_run':False,'inference_run':False}
save_json(sys.argv[2],report)
print(report)
'''
    check = subprocess.run([sys.executable, '-c', load_code, str(output), str(cpu_report)],
                           cwd=output, env=environment, text=True, capture_output=True, timeout=1200)
    if check.returncode:
        raise RuntimeError('Model downloaded but local CPU check failed: ' + check.stderr[-4000:])
    print('LOCAL_CPU_OFFLINE_CHECK', cpu_report, flush=True)
    print('LOCAL_MODEL', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.status:
        status()
    else:
        download()
