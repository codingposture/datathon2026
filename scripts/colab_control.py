"""Control the user-started Colab kernel through the local VS Code bridge.

No account credentials are extracted. The Jupyter extension enforces kernel access.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SPOOL = ROOT / 'runs/colab_bridge'


def request(payload, timeout=90):
    SPOOL.mkdir(parents=True, exist_ok=True)
    identifier = 'request_' + uuid.uuid4().hex
    path = SPOOL / (identifier + '.request.json')
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    subprocess.run(['code', '--open-url', f'vscode://datathon-local.datathon-colab-bridge/request?id={identifier}'],
                   check=True, timeout=30, stdout=subprocess.DEVNULL)
    reply = SPOOL / (identifier + '.reply.json')
    started = time.monotonic()
    printed = 0
    while time.monotonic() - started < timeout:
        if reply.exists():
            data = json.loads(reply.read_text())
            for output in data.get('outputs', [])[printed:]:
                text = output['text']
                if 'error' in output['mime']:
                    try:
                        error = json.loads(text)
                        text = f"{error.get('name')}: {error.get('message')}"
                    except ValueError:
                        pass
                print(text, end='' if text.endswith('\n') else '\n', flush=True)
            printed = len(data.get('outputs', []))
            if data['status'] in ('done', 'failed'):
                if data['status'] == 'failed':
                    raise RuntimeError(data.get('error', 'Remote request failed; inspect ' + str(reply)))
                return data
        time.sleep(1)
    raise TimeoutError(f'Request still pending: {identifier}; inspect {reply}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-file', type=Path)
    parser.add_argument('--json-file', type=Path)
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()
    payload = json.loads(args.json_file.read_text()) if args.json_file else \
        {'action': 'execute', 'code': args.code_file.read_text()} if args.code_file else {'action': 'inspect'}
    result = request(payload, args.timeout)
    print(json.dumps({k: v for k, v in result.items() if k != 'outputs'}, indent=2))
