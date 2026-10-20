from pathlib import Path
import base64
import json
import tempfile
import threading
import time

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from PIL import Image
import pytest

from backend.worker import api_result
from .conftest import wait_done


def upload(data, name='synthetic.mov'):
    return SimpleUploadedFile(name, data, content_type='video/quicktime')


@pytest.mark.parametrize('method,url', [('post', '/api/v1/analyze'), ('get', '/api/v1/jobs/unknown')])
@pytest.mark.parametrize('key', ['', 'incorrect', 'неверный'])
def test_api_key_required_on_both_routes(client, manager, method, url, key):
    client.credentials(HTTP_X_API_KEY=key)
    response = getattr(client, method)(url)
    assert response.status_code == 403
    assert response['Content-Type'] == 'application/json'
    assert not manager.jobs


def test_async_positive_wire_contract_and_cleanup(client, manager, mov_bytes, monkeypatch):
    release = threading.Event()
    temporary = []
    def execute(directory, remaining):
        temporary.append(directory)
        assert (directory / 'input.mov').read_bytes() == mov_bytes
        release.wait(3)
        evidence = directory / 'evidence'
        evidence.mkdir()
        image = evidence / 'frame.jpg'
        Image.new('RGB', (64, 96), 'blue').save(image, 'JPEG')
        return api_result({'verdict': 'violation', 'duration_sec': 1., 'score': .9,
            'score_kind': 'uncalibrated_hybrid_detection_score', 'events': [
                {'category_id': 'ACCEPTANCE_STEP_SKIPPED', 'timecode_sec': .5, 'score': .9,
                 'evidence_frames': [{'timestamp_sec': .5, 'image_path': str(image)}]}]}, evidence, 100000)
    monkeypatch.setattr(manager, '_execute', execute)
    response = client.post('/api/v1/analyze', {'file': upload(mov_bytes, 'synthetic.MOV')}, format='multipart')
    assert response.status_code == 202
    job_id = response.json()['job_id']
    processing = client.get(f'/api/v1/jobs/{job_id}')
    assert processing.json() == {'job_id': job_id, 'status': 'processing'}
    assert processing['Cache-Control'] == 'no-store'
    assert client.post('/api/v1/analyze', {'file': upload(mov_bytes)}, format='multipart').status_code == 429
    release.set()
    wait_done(manager, job_id)
    done = client.get(f'/api/v1/jobs/{job_id}').json()
    assert done['status'] == 'done' and done['verdict'] == 'violation'
    event = done['violations'][0]
    assert event['type'] == 'Acceptance step skipped' and event['timecode_sec'] == .5
    assert event['category_id'] == 'ACCEPTANCE_STEP_SKIPPED' and event['rule_reference'] is None
    assert event['evidence_frames'][0]['image_base64'].startswith('/9j/')
    assert base64.b64decode(event['evidence_frames'][0]['image_base64']).startswith(b'\xff\xd8\xff')
    assert '/tmp/' not in json.dumps(done) and 'image_path' not in json.dumps(done)
    assert not temporary[0].exists()


def test_negative_and_failed_jobs_are_distinct(client, manager, mov_bytes, monkeypatch):
    monkeypatch.setattr(manager, '_execute', lambda *args: {'verdict': 'no_violation', 'violations': []})
    response = client.post('/api/v1/analyze', {'file': upload(mov_bytes)}, format='multipart')
    first = wait_done(manager, response.json()['job_id'])
    assert first['status'] == 'done' and first['verdict'] == 'no_violation' and first['violations'] == []
    directories = []
    def fail(directory, remaining):
        directories.append(directory)
        raise ValueError('Sensitive internal path must not leak')
    monkeypatch.setattr(manager, '_execute', fail)
    response = client.post('/api/v1/analyze', {'file': upload(mov_bytes)}, format='multipart')
    failed = wait_done(manager, response.json()['job_id'])
    assert failed['status'] == 'failed' and 'verdict' not in failed
    assert 'Sensitive' not in failed['error'] and not directories[0].exists()


@pytest.mark.parametrize('data,name,status', [(b'invalid', 'video.mov', 415),
    (b'invalid', 'video.mp4', 415), (b'', 'empty.mov', 400)])
def test_rejects_invalid_uploads_and_releases_capacity(client, manager, data, name, status):
    response = client.post('/api/v1/analyze', {'file': upload(data, name)}, format='multipart')
    assert response.status_code == status and not manager.jobs
    manager.reserve()
    manager.slot.release()


def test_missing_file_wrong_media_type_and_unknown_job(client, manager):
    assert client.post('/api/v1/analyze', {}, format='multipart').status_code == 400
    assert client.post('/api/v1/analyze', {}, format='json').status_code == 415
    assert client.get('/api/v1/jobs/unknown').status_code == 404
    assert client.post('/api/v1/analyze/', {}, format='multipart').status_code == 404


def test_streaming_limit_discards_partial_upload(client, manager, tmp_path):
    with override_settings(MAX_UPLOAD_BYTES=16, FILE_UPLOAD_TEMP_DIR=str(tmp_path)):
        response = client.post('/api/v1/analyze', {'file': upload(b'x' * 1000)}, format='multipart')
    assert response.status_code == 413
    assert not list(tmp_path.iterdir()) and not manager.jobs


def test_multiple_files_rejected_as_json_and_cleaned(client, manager, tmp_path, mov_bytes):
    with override_settings(FILE_UPLOAD_TEMP_DIR=str(tmp_path)):
        response = client.post('/api/v1/analyze', {'file': upload(mov_bytes), 'extra': upload(mov_bytes)}, format='multipart')
    assert response.status_code == 400 and response['Content-Type'] == 'application/json'
    assert not list(tmp_path.iterdir()) and not manager.jobs


def test_rejects_excessive_duration(client, manager, mov_bytes, monkeypatch):
    with override_settings(MAX_VIDEO_SECONDS=.5):
        response = client.post('/api/v1/analyze', {'file': upload(mov_bytes)}, format='multipart')
    assert response.status_code == 415 and not manager.jobs


def test_evidence_paths_bounds_and_response_limit(tmp_path):
    evidence = tmp_path / 'frames'
    evidence.mkdir()
    image = evidence / 'frame.jpg'
    Image.new('RGB', (64, 96)).save(image, 'JPEG')
    source = {'verdict': 'violation', 'score': .9, 'score_kind': 'uncalibrated', 'duration_sec': 1,
              'events': [{'category_id': 'PERSONAL_ITEMS_IN_CASH_AREA', 'score': .9, 'timecode_sec': .2,
                          'evidence_frames': [{'timestamp_sec': .25, 'image_path': str(image)}]}]}
    assert api_result(source, evidence, 100000)['violations'][0]['rule_reference'] is None
    with pytest.raises(ValueError, match='large'):
        api_result(source, evidence, 8)
    source['events'][0]['evidence_frames'][0]['image_path'] = str(tmp_path / 'outside.jpg')
    with pytest.raises(ValueError, match='Invalid evidence'):
        api_result(source, evidence, 100000)
    source['events'][0]['timecode_sec'] = float('nan')
    with pytest.raises(ValueError, match='Invalid event'):
        api_result(source, evidence, 100000)
