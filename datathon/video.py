"""PTS-based video sampling and a frozen X-CLIP encoder. No hosted inference."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import zipfile
from importlib import metadata

import numpy as np
from PIL import Image, ImageOps

from .data import probe_video

MODEL_ID = 'microsoft/xclip-base-patch32'
# Immutable upstream revision, rather than a moving main branch.
MODEL_REVISION = 'a2e27a78a2b5d802e894b8a1ef14f3a8ce490963'


@dataclass(frozen=True)
class VideoConfig:
    window_sec: float = 4.0
    stride_sec: float = 1.0
    num_frames: int = 8
    sample_fps: float = 4.0  # Includes quarter-second samples for local refinement.
    image_size: int = 224
    batch_size: int = 8
    refine_radius_sec: float = 2.0
    refine_step_sec: float = 0.25
    merge_sec: float = 4.0

    def __post_init__(self):
        if self.num_frames != 8 or self.image_size != 224:
            raise ValueError('This checkpoint requires eight 224-pixel frames')
        if min(self.window_sec, self.stride_sec, self.sample_fps, self.batch_size,
               self.refine_step_sec, self.merge_sec) <= 0 or self.refine_radius_sec < 0:
            raise ValueError('Invalid video configuration')


def letterbox(image, size=224):
    image = image.convert('RGB')
    return ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def requested_frames(path, targets):
    """Decode once, returning nearest original frames and actual normalized PTS.

    Targets must be sorted and nonnegative. The first displayed frame defines t=0.
    Decode errors and missing/nonmonotonic timestamps are explicit failures.
    """
    import av
    targets = list(map(float, targets))
    if not targets or targets[0] < 0 or not np.isfinite(targets).all() or targets != sorted(targets):
        raise ValueError('Frame targets must be finite, sorted, nonnegative, and nonempty')
    index = 0
    previous = None
    previous_time = None
    origin = None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Expected one video stream')
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError(f'Missing presentation timestamp: {path}')
            raw_time = float(frame.pts * frame.time_base)
            if origin is None:
                origin = raw_time
            timestamp = raw_time - origin
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError(f'Nonmonotonic presentation timestamps: {path}')
            while index < len(targets) and targets[index] <= timestamp:
                if previous is not None and targets[index] - previous_time <= timestamp - targets[index]:
                    chosen, actual = previous, previous_time
                else:
                    chosen, actual = frame, timestamp
                yield chosen.to_image(), actual
                index += 1
            if index == len(targets):
                return
            previous, previous_time = frame, timestamp
    if previous is None:
        raise ValueError(f'Video contains no decoded frames: {path}')
    while index < len(targets):
        yield previous.to_image(), previous_time
        index += 1


class SampledVideo:
    def __init__(self, path, config=VideoConfig()):
        self.path = Path(path)
        self.config = config
        self.duration = probe_video(path, full_decode=False)['duration_sec']
        targets = np.arange(0, self.duration, 1 / config.sample_fps)
        images, timestamps = [], []
        for image, timestamp in requested_frames(path, targets):
            images.append(np.asarray(letterbox(image, config.image_size)))
            timestamps.append(timestamp)
        self.frames = np.stack(images)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)

    def clips(self, centers):
        centers = np.asarray(centers, dtype=np.float64)
        offsets = ((np.arange(self.config.num_frames) + 0.5) / self.config.num_frames - 0.5) * self.config.window_sec
        wanted = np.clip(centers[:, None] + offsets, 0, self.timestamps[-1])
        right = np.searchsorted(self.timestamps, wanted).clip(0, len(self.timestamps) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(np.abs(self.timestamps[left] - wanted) <= np.abs(self.timestamps[right] - wanted), left, right)
        return self.frames[indices]

    def centers(self):
        return np.arange(0, self.duration, self.config.stride_sec, dtype=np.float64)


class XClipEncoder:
    def __init__(self, source=MODEL_ID, revision=MODEL_REVISION, device=None, local_only=False, config=VideoConfig()):
        import torch
        from transformers import VideoMAEImageProcessor, XCLIPModel
        self.torch = torch
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        source = str(source)
        kwargs = {'local_files_only': local_only}
        if not Path(source).is_dir():
            kwargs['revision'] = revision
        self.model = XCLIPModel.from_pretrained(source, use_safetensors=True, **kwargs).to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = VideoMAEImageProcessor.from_pretrained(source, **kwargs)
        # Spatial geometry is handled by letterbox(), preserving the entire frame.
        self.processor.do_resize = False
        self.processor.do_center_crop = False
        self.source_revision = revision

    def encode(self, video, centers):
        vectors = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(centers), self.config.batch_size):
                clips = video.clips(centers[start:start + self.config.batch_size])
                inputs = self.processor([list(clip) for clip in clips], return_tensors='pt')
                pixels = inputs['pixel_values'].to(self.device)
                if pixels.ndim != 5 or pixels.shape[1:3] != (self.config.num_frames, 3):
                    raise ValueError(f'Unexpected X-CLIP input shape: {tuple(pixels.shape)}')
                context = torch.autocast(device_type='cuda', dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
                with context:
                    features = self.model.get_video_features(pixel_values=pixels)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
                vectors.append(features.cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)

    def save(self, path):
        path = Path(path)
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)


def cache_identity(recording, encoder):
    return {'schema_version': 1, 'sha256': recording['sha256'], 'model': MODEL_ID,
            'revision': encoder.source_revision, 'video_config': asdict(encoder.config),
            'precision': 'float16' if encoder.device.startswith('cuda') else 'float32',
            'dependency_versions': {name: metadata.version(name) for name in ('torch', 'transformers', 'Pillow', 'av', 'numpy')}}


def cached_features(recording, dataset_dir, cache_dir, encoder):
    identity = cache_identity(recording, encoder)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f'{key}.npz'
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                centers, features = data['centers'], data['features']
                if json.loads(str(data['identity'])) == identity and centers.ndim == 1 and len(centers) \
                        and features.shape == (len(centers), 512) and np.isfinite(centers).all() \
                        and (np.diff(centers) > 0).all() and np.isfinite(features).all():
                    return {'centers': centers.copy(), 'features': features.copy()}
        except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile):
            pass  # Interrupted/corrupt caches are safely recomputed.
    video = SampledVideo(Path(dataset_dir) / recording['relative_path'], encoder.config)
    centers = video.centers()
    features = encoder.encode(video, centers)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, identity=json.dumps(identity, sort_keys=True), centers=centers, features=features)
    os.replace(temporary, path)
    return {'centers': centers, 'features': features}
