"""Offline inference and a self-contained artifact export."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from .data import save_json, sha256_file
from .training import LinearHead
from .video import MODEL_ID, MODEL_REVISION, SampledVideo, VideoConfig, XClipEncoder, requested_frames


@contextmanager
def deny_network():
    """Application-level socket guard; not an OS network namespace."""
    def blocked(*args, **kwargs):
        raise RuntimeError('Network access blocked during offline inference check')
    with patch.object(socket.socket, 'connect', blocked), \
         patch.object(socket.socket, 'connect_ex', blocked), \
         patch.object(socket, 'create_connection', blocked), \
         patch.object(socket, 'getaddrinfo', blocked):
        yield


def hardware_info():
    import torch
    info = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return info


def coarse_events(centers, scores, threshold):
    positive = np.flatnonzero(np.asarray(scores) >= threshold)
    if not len(positive):
        return []
    runs = np.split(positive, np.flatnonzero(np.diff(positive) > 1) + 1)
    return [{'timecode_sec': float(centers[run[np.argmax(scores[run])]]),
             'score': float(np.max(scores[run]))} for run in runs]


def merge_events(events, separation):
    # Greedy highest-score nonmaximum suppression; chains cannot swallow distant events.
    kept = []
    for event in sorted(events, key=lambda e: (-e['score'], e['timecode_sec'])):
        if all(abs(event['timecode_sec'] - other['timecode_sec']) >= separation for other in kept):
            kept.append(event)
    return sorted(kept, key=lambda e: e['timecode_sec'])


class Detector:
    def __init__(self, encoder, head, threshold):
        self.encoder = encoder
        self.head = head
        self.threshold = float(threshold)
        self.config = encoder.config

    @classmethod
    def from_artifacts(cls, artifact_dir, device=None):
        artifact_dir = Path(artifact_dir).resolve()
        spec = json.loads((artifact_dir / 'inference_config.json').read_text())
        if spec.get('schema_version') == 2:
            from .hybrid_inference import HybridDetector
            return HybridDetector.from_artifacts(artifact_dir, device=device)
        if not (artifact_dir / 'COMPLETE.json').exists():
            raise ValueError('Incomplete export: COMPLETE.json missing')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        encoder = XClipEncoder(artifact_dir / 'encoder', revision=spec['model_revision'],
                               device=device, local_only=True, config=VideoConfig(**spec['video']))
        return cls(encoder, LinearHead.load(artifact_dir / 'head.json'), spec['threshold'])

    def predict(self, video_path, evidence_dir, cached=None):
        import torch
        if self.encoder.device.startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        video = SampledVideo(video_path, self.config)
        centers = video.centers() if cached is None else cached['centers']
        features = self.encoder.encode(video, centers) if cached is None else cached['features']
        scores = self.head.predict(features)
        candidates = coarse_events(centers, scores, self.threshold)
        refined = []
        for event in candidates:
            center = event['timecode_sec']
            times = np.arange(center - self.config.refine_radius_sec,
                              center + self.config.refine_radius_sec + self.config.refine_step_sec / 2,
                              self.config.refine_step_sec)
            times = np.unique(np.clip(times, 0, max(0, video.duration - 0.001)))
            local_scores = self.head.predict(self.encoder.encode(video, times))
            index = int(np.argmax(local_scores))
            refined.append({'timecode_sec': float(times[index]), 'score': float(local_scores[index])})
        events = merge_events(refined, self.config.merge_sec)
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for index, event in enumerate(events):
            targets = sorted({float(np.clip(event['timecode_sec'] + offset, 0, video.timestamps[-1])) for offset in (-1, 0, 1)})
            frames, seen = [], set()
            for image, timestamp in requested_frames(video_path, targets):
                if timestamp in seen:
                    continue
                seen.add(timestamp)
                path = evidence_dir / f'event_{index:03d}_{len(frames):02d}.jpg'
                image.convert('RGB').save(path, format='JPEG', quality=90)
                frames.append({'timestamp_sec': float(timestamp), 'image_path': str(path.resolve())})
            event['evidence_frames'] = frames
        if self.encoder.device.startswith('cuda'):
            torch.cuda.synchronize()
        return {'verdict': 'violation' if events else 'no_violation',
                'score': max([float(scores.max())] + [e['score'] for e in events]),
                'score_kind': 'uncalibrated_logistic_score', 'threshold': self.threshold,
                'duration_sec': video.duration, 'events': events,
                'runtime_sec': time.perf_counter() - started,
                'feature_cache_used': cached is not None}


_DETECTORS = {}


def analyze_video(video_path, artifact_dir, *, camera_profile=None):
    """Analyze one video using local exported weights; write JSON/JPEG beside artifacts."""
    artifact_dir = Path(artifact_dir).resolve()
    identity = (str(artifact_dir), (artifact_dir / 'COMPLETE.json').stat().st_mtime_ns)
    if identity not in _DETECTORS:
        _DETECTORS[identity] = Detector.from_artifacts(artifact_dir)
    digest = sha256_file(video_path)
    output_dir = artifact_dir / 'inference_outputs' / digest
    detector = _DETECTORS[identity]
    if hasattr(detector, 'selection'):
        result = detector.predict(video_path, output_dir, camera_profile=camera_profile)
    else:
        result = detector.predict(video_path, output_dir)
    save_json(output_dir / 'result.json', result)
    return result


def export_artifacts(encoder, head, selection, split, evaluation, output_dir, run_dir, notices_dir):
    """Export weights, source, exact installed versions, provenance, and third-party notices."""
    output_dir, run_dir, notices_dir = Path(output_dir), Path(run_dir), Path(notices_dir)
    if not all((notices_dir / name).is_file() for name in ('MODEL_CARD.md', 'XCLIP_LICENSE', 'MODEL_METADATA.json')):
        raise ValueError('Download and preserve upstream model card, license, and metadata first')
    temporary = output_dir.with_name(output_dir.name + '.building')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    encoder.save(temporary / 'encoder')
    head.save(temporary / 'head.json')
    save_json(temporary / 'inference_config.json', {
        'schema_version': 1, 'model_id': MODEL_ID, 'model_revision': encoder.source_revision,
        'threshold': selection['threshold'], 'selected_C': selection['selected_C'],
        'video': asdict(encoder.config), 'score_kind': 'uncalibrated_logistic_score',
        'training_subset': 'development_only', 'annotation_mode': evaluation.get('annotation_mode', 'reviewed'),
        'dataset_fingerprint': split['dataset_fingerprint']})
    save_json(temporary / 'split.json', split)
    save_json(temporary / 'model_selection.json', selection)
    save_json(temporary / 'evaluation.json', evaluation)
    save_json(temporary / 'training_hardware.json', hardware_info())
    shutil.copytree(Path(__file__).parent, temporary / 'datathon', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(notices_dir, temporary / 'licenses')
    packages = ('torch', 'torchvision', 'transformers', 'huggingface-hub', 'safetensors',
                'numpy', 'scipy', 'scikit-learn', 'Pillow', 'av')
    versions, license_metadata = {}, {}
    for name in packages:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        versions[name] = distribution.version
        license_metadata[name] = {'version': distribution.version,
                                  'license': distribution.metadata.get('License'),
                                  'license_expression': distribution.metadata.get('License-Expression'),
                                  'home_page': distribution.metadata.get('Home-page')}
        # Preserve installed packages' license/NOTICE files when distributed in wheel metadata.
        for file in distribution.files or []:
            if '.dist-info/' in str(file) and any(word in file.name.lower() for word in ('license', 'notice', 'copying')):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    destination = temporary / 'licenses' / 'dependencies' / name / str(file).split('.dist-info/', 1)[1]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    save_json(temporary / 'dependency_versions.json', versions)
    save_json(temporary / 'licenses' / 'dependency_metadata.json', license_metadata)
    (temporary / 'requirements-runtime.txt').write_text(''.join(f'{k}=={v}\n' for k, v in versions.items()))
    (temporary / 'OFFLINE_USAGE.md').write_text(
        'Provision Python dependencies and ffmpeg/ffprobe before disconnecting.\n'
        'Run from this directory: python -m datathon.cli VIDEO.mov --artifacts . --offline-check\n'
        'This is a binary detector; types, rules, and the jury API are deferred.\n'
        'Scores are uncalibrated. Review evidence; outputs are possible visual violations.\n')
    checksums = {str(p.relative_to(temporary)): sha256_file(p) for p in temporary.rglob('*') if p.is_file()}
    save_json(temporary / 'COMPLETE.json', {'schema_version': 1, 'sha256': checksums})
    # Never destroy a prior usable artifact directory on accidental rerun.
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}; choose a new RUN_NAME or archive it explicitly')
    temporary.replace(output_dir)
    return output_dir


def verify_export(artifact_dir):
    root = Path(artifact_dir)
    manifest = json.loads((root / 'COMPLETE.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Artifact integrity failure: {name}')
    return len(manifest['sha256'])
