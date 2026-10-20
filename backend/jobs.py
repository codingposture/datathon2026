"""One CPU job at a time, temporary files, bounded RAM results; no database/queue service."""
import atexit
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from django.conf import settings
from rest_framework.exceptions import APIException

logger = logging.getLogger(__name__)


class Busy(APIException):
    status_code = 429
    default_detail = 'A video is already being analyzed. Please try again after it finishes.'


class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.slot = threading.BoundedSemaphore(1)
        self.timers = {}
        self.process = None
        self.thread = None
        self.closed = False
        self.ttl = settings.JOB_TTL_SECONDS
        self.capacity = settings.MAX_RETAINED_JOBS
        self.timeout = settings.ANALYSIS_TIMEOUT_SECONDS
        self.artifacts = settings.MODEL_ARTIFACT_DIR

    def reserve(self):
        if self.closed or not self.slot.acquire(blocking=False):
            raise Busy()

    def start(self, temporary, started):
        job_id = uuid.uuid4().hex
        with self.lock:
            while len(self.jobs) >= self.capacity:
                oldest = next(iter(self.jobs))
                self.jobs.pop(oldest)
                timer = self.timers.pop(oldest, None)
                if timer:
                    timer.cancel()
            self.jobs[job_id] = {'job_id': job_id, 'status': 'processing'}
        self.thread = threading.Thread(target=self._work, args=(job_id, temporary, started), daemon=True)
        try:
            self.thread.start()
        except Exception:
            with self.lock:
                self.jobs.pop(job_id, None)
            raise
        return job_id

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job and time.monotonic() >= job.get('_expires', float('inf')):
                self.jobs.pop(job_id, None)
                timer = self.timers.pop(job_id, None)
                if timer:
                    timer.cancel()
                job = None
            if job is None:
                return None
            return {k: v for k, v in job.items() if k != '_expires'}

    @staticmethod
    def _stop(process):
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _execute(self, directory, remaining):
        if remaining <= 0:
            raise subprocess.TimeoutExpired('analysis', self.timeout)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'USE_TF': '0',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
            **{key: str(settings.INFERENCE_THREADS) for key in
               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
        command = [sys.executable, '-m', 'backend.worker', '--video', str(directory / 'input.mov'),
                   '--evidence', str(directory / 'evidence'), '--artifacts', str(self.artifacts),
                   '--max-result-bytes', str(settings.MAX_RESULT_BYTES)]
        with self.lock:
            if self.closed:
                raise RuntimeError('Server stopping')
            process = subprocess.Popen(command, cwd=settings.BASE_DIR, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
            self.process = process
        try:
            output, _ = process.communicate(timeout=remaining)
            if process.returncode or len(output) > settings.MAX_RESULT_BYTES:
                raise RuntimeError('Worker failed or response was too large')
            return json.loads(output)
        finally:
            self._stop(process)
            process.communicate()
            with self.lock:
                self.process = None

    def _work(self, job_id, temporary, started):
        state = {'job_id': job_id, 'status': 'failed', 'error': 'Analysis was interrupted.'}
        try:
            result = self._execute(Path(temporary.name), self.timeout - (time.monotonic() - started))
            state = {'job_id': job_id, 'status': 'done', **result}
        except subprocess.TimeoutExpired:
            state = {'job_id': job_id, 'status': 'failed', 'error': 'Analysis exceeded the processing time limit.'}
        except Exception:
            logger.warning('Analysis failed for job %s', job_id)
            state = {'job_id': job_id, 'status': 'failed',
                     'error': 'The local model could not analyze this video. Check the model files or try a different video.'}
        finally:
            try:
                temporary.cleanup()
            except OSError:
                logger.error('Temporary directory cleanup failed for job %s', job_id)
            with self.lock:
                if not self.closed:
                    state['_expires'] = time.monotonic() + self.ttl
                    self.jobs[job_id] = state
                    timer = threading.Timer(self.ttl, self._expire, args=(job_id,))
                    timer.daemon = True
                    self.timers[job_id] = timer
                    timer.start()
            self.slot.release()

    def _expire(self, job_id):
        with self.lock:
            self.jobs.pop(job_id, None)
            self.timers.pop(job_id, None)

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
            for timer in self.timers.values():
                timer.cancel()
            self.timers.clear()
            self.jobs.clear()
        self._stop(process)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)


@lru_cache(maxsize=1)
def get_manager():
    manager = JobManager()
    atexit.register(manager.close)
    return manager
