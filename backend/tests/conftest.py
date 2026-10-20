import os
from pathlib import Path
import subprocess
import threading
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
os.environ['API_KEY'] = 'backend-unit-test-key'
os.environ['DJANGO_SECRET_KEY'] = 'backend-unit-test-secret'

import django
django.setup()
from django.test import override_settings
import pytest
from rest_framework.test import APIClient

from backend.jobs import JobManager


@pytest.fixture
def client():
    client = APIClient(HTTP_HOST='127.0.0.1')
    client.credentials(HTTP_X_API_KEY='backend-unit-test-key')
    return client


@pytest.fixture
def manager(monkeypatch):
    with override_settings(JOB_TTL_SECONDS=10, MAX_RETAINED_JOBS=2):
        manager = JobManager()
    monkeypatch.setattr('backend.views.get_manager', lambda: manager)
    yield manager
    manager.close()


@pytest.fixture(scope='session')
def mov_bytes(tmp_path_factory):
    path = tmp_path_factory.mktemp('mov') / 'synthetic.mov'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x96:r=4',
                    '-t', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path.read_bytes()


def wait_done(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job and job['status'] != 'processing':
            return job
        time.sleep(.01)
    raise AssertionError('Test worker did not finish')
