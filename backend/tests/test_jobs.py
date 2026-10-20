import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .conftest import wait_done


def test_real_subprocess_timeout_and_cleanup(manager, monkeypatch):
    original = subprocess.Popen
    processes = []
    def sleeping_worker(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
        assert command[2] == 'backend.worker'
        process = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr('backend.jobs.subprocess.Popen', sleeping_worker)
    manager.timeout = .1
    temporary = tempfile.TemporaryDirectory(prefix='fortebank-test-')
    path = Path(temporary.name)
    manager.reserve()
    job_id = manager.start(temporary, time.monotonic())
    job = wait_done(manager, job_id)
    assert job['status'] == 'failed' and 'time limit' in job['error']
    assert processes[0].poll() is not None and not path.exists()


def test_results_expire_without_polling_and_store_is_bounded(manager, monkeypatch):
    monkeypatch.setattr(manager, '_execute', lambda *a: {'verdict': 'no_violation', 'violations': []})
    manager.ttl = .1
    ids = []
    for index in range(3):
        manager.reserve()
        job_id = manager.start(tempfile.TemporaryDirectory(), time.monotonic())
        wait_done(manager, job_id)
        ids.append(job_id)
    assert len(manager.jobs) <= 2 and manager.get(ids[0]) is None
    time.sleep(.15)
    assert manager.jobs == {} and not manager.timers


def test_shutdown_stops_worker_and_removes_files(manager, monkeypatch):
    original = subprocess.Popen
    processes = []
    def sleeping_worker(command, **kwargs):
        process = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr('backend.jobs.subprocess.Popen', sleeping_worker)
    temporary = tempfile.TemporaryDirectory(prefix='fortebank-test-')
    path = Path(temporary.name)
    manager.reserve()
    job_id = manager.start(temporary, time.monotonic())
    deadline = time.monotonic() + 2
    while not processes and time.monotonic() < deadline:
        time.sleep(.01)
    manager.close()
    assert processes[0].poll() is not None
    assert not path.exists() and manager.get(job_id) is None
