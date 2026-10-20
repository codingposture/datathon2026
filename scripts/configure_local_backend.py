"""Configure private backend credentials and the local frontend origin without printing secrets."""
from pathlib import Path
import secrets
from dotenv import dotenv_values, set_key

root = Path(__file__).resolve().parents[1]
backend = root / 'backend/.env'
frontend = root / 'fortebank-frontend/.env.local'
server = dotenv_values(backend) if backend.exists() else {}
key = server.get('API_KEY') or secrets.token_urlsafe(32)
for path, values in ((backend, {'API_KEY': key, 'DJANGO_SECRET_KEY': server.get('DJANGO_SECRET_KEY') or secrets.token_urlsafe(48)}),
                     (frontend, {'DJANGO_ORIGIN': 'http://127.0.0.1:8000'})):
    path.touch(mode=0o600, exist_ok=True)
    existing = dotenv_values(path)
    for name, value in values.items():
        if not existing.get(name):
            set_key(path, name, value)
    path.chmod(0o600)
print('Local backend and frontend proxy configured. Enter the backend API_KEY in the frontend when analyzing a video.')
