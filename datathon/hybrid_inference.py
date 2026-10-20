"""Version-two offline inference; leaves version-one artifact loading intact."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np

from .data import save_json, sha256_file
from .hybrid_data import CATEGORIES, TAXONOMY
from .hybrid import (CategoryHead, PromptedEncoder, blend_scores, consecutive_scores,
                     merge_typed_events, typed_events)
from .video import SampledVideo, VideoConfig, requested_frames
from .tracking import PersonDetector, DEFAULT_PROFILE, load_profile, valid_profile, departure_observations, assist_events


class HybridDetector:
    def __init__(self, encoder, head, prompts, selection, detector=None, artifact_dir=None):
        self.encoder, self.head, self.prompts, self.selection = encoder, head, prompts, selection
        self.config = encoder.config
        self.threshold = selection['threshold']
        self.person_detector, self.artifact_dir = detector, artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir, device=None):
        root = Path(artifact_dir)
        if not (root / 'COMPLETE.json').is_file():
            raise ValueError('Incomplete hybrid export')
        spec = json.loads((root / 'inference_config.json').read_text())
        encoder = PromptedEncoder(root / 'encoder', revision=spec['model_revision'], device=device,
                                 local_only=True, config=VideoConfig(**spec['video']))
        detector = PersonDetector(root / 'person_detector', local_only=True, device=device) \
            if spec['selection'].get('detector_enabled') else None
        return cls(encoder, CategoryHead.load(root / 'head.json'),
                   json.loads((root / 'prompts.json').read_text()), spec['selection'], detector, root)

    def score(self, video, centers, cached=None):
        weight = self.selection['supervised_weight']
        supervised = np.zeros((len(centers), 4))
        zero = np.zeros_like(supervised)
        if weight > 0:
            features = self.encoder.encode(video, centers) if cached is None else cached['features']
            supervised = self.head.predict(features)
        if weight < 1:
            zero = self.encoder.zero_shot(video, centers, self.prompts)
        return blend_scores(supervised, zero, weight, self.head.classes)

    def predict(self, video_path, evidence_dir, cached=None, *, camera_profile=None):
        torch = self.encoder.torch
        if self.encoder.device.startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        video = SampledVideo(video_path, self.config)
        centers = video.centers() if cached is None else cached['centers']
        scores = self.score(video, centers, cached)
        events = typed_events(centers, scores, self.threshold, self.config.merge_sec, self.config.stride_sec)
        refined = []
        for event in events:
            peak = event['timecode_sec']
            times = np.unique(np.clip(np.arange(peak - 2, peak + 2.001, .25), 0, video.duration - .001))
            j = CATEGORIES.index(event['category_id'])
            local = self.score(video, times)[:, j]
            index = int(np.argmax(local))
            refined.append({**event, 'timecode_sec': float(times[index]), 'score': float(local[index]),
                            'coarse_timecode_sec': peak})
        events = merge_typed_events(refined, self.config.merge_sec)
        profile = load_profile(camera_profile, self.artifact_dir)
        tracking_status = 'disabled_by_development_selection'
        if self.selection.get('detector_enabled'):
            tracking_status = 'disabled_unconfirmed_camera'
            if valid_profile(profile) and self.person_detector is not None:
                observations = self.person_detector.observe(video_path, video.duration)
                tracking = departure_observations(observations, profile)
                events = assist_events(events, tracking, profile)
                tracking_status = tracking['status']
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for index, event in enumerate(events):
            targets = sorted(set(float(np.clip(event['timecode_sec'] + offset, 0, video.timestamps[-1]))
                                 for offset in (-1, 0, 1)))
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
        category_scores = consecutive_scores(centers, scores, self.config.stride_sec)
        return {'verdict': 'violation' if events else 'no_violation',
                'score': float(category_scores.max()), 'score_kind': 'uncalibrated_hybrid_detection_score',
                'category_scores': dict(zip(CATEGORIES, map(float, category_scores))),
                'threshold': float(self.threshold), 'events': events, 'duration_sec': video.duration,
                'runtime_sec': time.perf_counter() - started, 'tracking_status': tracking_status,
                'experimental': True, 'feature_cache_used': cached is not None}


def export_hybrid(encoder, head, prompts, selection, split, evaluation, output_dir, run_dir,
                  notices_dir, person_detector=None, profile=None):
    from .inference import export_artifacts
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f'Preserve existing bundle: {root}')
    staging = root.with_name(root.name + '.assembling')
    if staging.exists():
        raise FileExistsError(f'Interrupted export exists: inspect {staging}')
    export_artifacts(encoder, head, selection, split, evaluation, staging, run_dir, notices_dir)
    (staging / 'COMPLETE.json').unlink()
    spec = json.loads((staging / 'inference_config.json').read_text())
    spec.update(schema_version=2, selection=selection, score_kind='uncalibrated_hybrid_detection_score',
                experimental=True, annotation_mode='provisional_typed_workbook')
    save_json(staging / 'inference_config.json', spec)
    save_json(staging / 'prompts.json', prompts)
    save_json(staging / 'taxonomy.json', TAXONOMY)
    save_json(staging / 'camera_profiles.json', {'supplied_camera': profile or DEFAULT_PROFILE})
    if selection.get('detector_enabled'):
        if person_detector is None:
            raise ValueError('Selected detector weights missing')
        person_detector.save(staging / 'person_detector')
    for name in ('typed_manifest.json', 'experiment_identity.json', 'detector_experiment.json', 'MODEL_REPORT.md'):
        source = Path(run_dir) / name
        if source.exists():
            import shutil
            shutil.copy2(source, staging / name)
    (staging / 'OFFLINE_USAGE.md').write_text(
        'Experimental typed hybrid detector. Scores are uncalibrated; review evidence.\n'
        'Install requirements-runtime.txt and ffmpeg before disconnecting.\n'
        'python -m datathon.cli VIDEO.mov --artifacts . --offline-check\n'
        'Python: from datathon.inference import analyze_video\n'
        'analyze_video(video_path, artifact_dir, camera_profile=None)\n'
        'Unknown/unconfirmed cameras use the selected X-CLIP path.\n'
        'Reused eight-video test is diagnostic; it has no workstation-security positives.\n')
    checksums = {str(p.relative_to(staging)): sha256_file(p) for p in staging.rglob('*') if p.is_file()}
    save_json(staging / 'COMPLETE.json', {'schema_version': 2, 'sha256': checksums})
    staging.replace(root)
    return root
