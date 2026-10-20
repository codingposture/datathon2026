from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import av
import numpy as np
from PIL import Image
import pytest

from datathon.data import save_json, sha256_file
from datathon.inference import (Detector, coarse_events, deny_network, export_artifacts,
                                merge_events, verify_export)
from datathon.training import LinearHead
from datathon.video import SampledVideo, VideoConfig, XClipEncoder, cached_features, letterbox, requested_frames


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / 'fixture.mov'
    # Variable PTS deliberately exercise timing rather than frame_index / FPS.
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('libx264', rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 96, 'yuv420p'
        stream.time_base = Fraction(1, 8000)
        stream.codec_context.time_base = Fraction(1, 8000)
        for i in range(32):
            image = np.zeros((96, 64, 3), dtype=np.uint8)
            image[:, :, i % 3] = 180
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts = i * 1000 + (250 if i >= 16 else 0)
            frame.time_base = Fraction(1, 8000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_letterbox_and_nearest_pts(synthetic_video):
    image = Image.new('RGB', (64, 96), color='white')
    padded = np.asarray(letterbox(image))
    assert padded.shape == (224, 224, 3)
    assert padded[:, 0].sum() == 0
    assert padded[:, 112].min() == 255
    desired = [0, 0.25, 1.99, 2.1, 3.8]
    frames = list(requested_frames(synthetic_video, desired))
    assert len(frames) == len(desired)
    times = [t for _, t in frames]
    assert all(abs(a - b) <= 0.09 for a, b in zip(times, desired))
    assert any(abs(t * 8 - round(t * 8)) > 0.01 for t in times)
    video = SampledVideo(synthetic_video)
    clips = video.clips(np.array([0, 2, video.duration - 0.01]))
    assert clips.shape == (3, 8, 224, 224, 3)
    assert np.isfinite(clips).all()


class FakeEncoder:
    config = VideoConfig()
    source_revision = 'test-only'
    device = 'cpu'
    def __init__(self):
        self.calls = 0
    def encode(self, video, centers):
        self.calls += 1
        return np.tile(np.linspace(0, 1, 512, dtype=np.float32), (len(centers), 1))


def test_cache_restart_corruption_and_preprocess_identity(synthetic_video, tmp_path):
    encoder = FakeEncoder()
    record = {'sha256': sha256_file(synthetic_video), 'relative_path': synthetic_video.name}
    cache = tmp_path / 'cache'
    first = cached_features(record, tmp_path, cache, encoder)
    restarted = FakeEncoder()
    second = cached_features(record, tmp_path, cache, restarted)
    assert restarted.calls == 0
    np.testing.assert_equal(first['features'], second['features'])
    next(cache.glob('*.npz')).write_bytes(b'interrupted partial cache')
    cached_features(record, tmp_path, cache, restarted)
    assert restarted.calls == 1
    restarted.config = VideoConfig(stride_sec=2)
    cached_features(record, tmp_path, cache, restarted)
    assert restarted.calls == 2


def test_event_runs_merging_and_evidence(synthetic_video, tmp_path):
    events = coarse_events(np.arange(7), np.array([0.1, 0.9, 0.7, 0.1, 0.95, 0.8, 0.1]), 0.5)
    assert [e['timecode_sec'] for e in events] == [1, 4]
    assert len(merge_events(events, 4)) == 1
    encoder = FakeEncoder()
    head = LinearHead(np.zeros(512), np.ones(512), np.zeros(512), 2)
    result = Detector(encoder, head, 0.5).predict(synthetic_video, tmp_path / 'evidence')
    assert result['verdict'] == 'violation'
    assert not result['feature_cache_used']
    for event in result['events']:
        assert 0 <= event['timecode_sec'] < result['duration_sec']
        assert 1 <= len(event['evidence_frames']) <= 3
        for frame in event['evidence_frames']:
            assert 0 <= frame['timestamp_sec'] < result['duration_sec']
            with Image.open(frame['image_path']) as image:
                assert image.format == 'JPEG'
                assert image.size == (64, 96)
    negative = LinearHead(np.zeros(512), np.ones(512), np.zeros(512), -2)
    result = Detector(encoder, negative, 0.5).predict(synthetic_video, tmp_path / 'negative')
    assert result['verdict'] == 'no_violation'
    assert result['events'] == []


def test_network_guard_blocks_socket():
    with deny_network(), pytest.raises(RuntimeError, match='blocked'):
        socket.create_connection(('example.com', 443))


@pytest.fixture
def tiny_encoder(tmp_path):
    from transformers import (VideoMAEImageProcessor, XCLIPConfig, XCLIPModel,
                              XCLIPTextConfig, XCLIPVisionConfig)
    import torch
    torch.manual_seed(42)
    torch.set_num_threads(2)
    text = XCLIPTextConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=16)
    vision = XCLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_attention_heads=4, mit_hidden_size=16, mit_intermediate_size=32,
                               mit_num_hidden_layers=1, mit_num_attention_heads=4,
                               image_size=224, patch_size=32, num_frames=8)
    config = XCLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                         projection_dim=16, prompt_layers=1, prompt_num_attention_heads=4)
    source = tmp_path / 'tiny-model'
    XCLIPModel(config).save_pretrained(source)
    VideoMAEImageProcessor().save_pretrained(source)
    return XClipEncoder(source, revision='random-test-weights', device='cpu', local_only=True)


def test_real_xclip_shape_freeze_and_offline_roundtrip(tiny_encoder, synthetic_video, tmp_path):
    video = SampledVideo(synthetic_video)
    encoded = tiny_encoder.encode(video, np.array([0, 1]))
    assert encoded.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-6)
    assert all(not parameter.requires_grad for parameter in tiny_encoder.model.parameters())
    assert not tiny_encoder.model.training
    head = LinearHead(np.zeros(16), np.ones(16), np.zeros(16), 2)
    notices = tmp_path / 'notices'
    notices.mkdir()
    for name in ('MODEL_CARD.md', 'XCLIP_LICENSE', 'MODEL_METADATA.json'):
        (notices / name).write_text('SYNTHETIC TEST FIXTURE; NOT PRETRAINED WEIGHTS')
    output = tmp_path / 'exported'
    selection = {'threshold': 0.5, 'selected_C': 0.1}
    split = {'dataset_fingerprint': 'synthetic'}
    export_artifacts(tiny_encoder, head, selection, split, {}, output, tmp_path, notices)
    assert verify_export(output) > 10
    with deny_network():
        loaded = Detector.from_artifacts(output, device='cpu')
        np.testing.assert_allclose(loaded.encoder.encode(video, np.array([0, 1])), encoded, atol=1e-6)
        result = loaded.predict(synthetic_video, tmp_path / 'offline_evidence')
    assert result['verdict'] == 'violation'
    assert result['events'][0]['evidence_frames']
    # A fresh interpreter must also load only exported files under the socket guard.
    import os
    import sys
    environment = {**os.environ, 'PYTHONPATH': str(output), 'HF_HOME': str(tmp_path / 'empty-cache'),
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    report = tmp_path / 'cli-report.json'
    process = subprocess.run([sys.executable, '-m', 'datathon.cli', str(synthetic_video),
                               '--artifacts', str(output), '--offline-check', '--report', str(report)],
                              cwd=output, env=environment, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    saved = json.loads(report.read_text())
    assert saved['verdict'] == 'violation'
    assert saved['python_socket_guard_enabled']
    assert saved['hardware']['gpu'] is None
    (output / 'head.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        verify_export(output)


def test_typed_full_forward_prompt_generator_and_fresh_offline_bundle(synthetic_video, tmp_path):
    """Real tiny X-CLIP catches processor, forward, tokenizer and v2 loader failures."""
    from transformers import (XCLIPModel, XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig,
                              CLIPTokenizerFast, VideoMAEImageProcessor)
    from datathon.hybrid import PromptedEncoder, CategoryHead, prompt_bank
    from datathon.hybrid_inference import HybridDetector, export_hybrid
    from datathon.inference import analyze_video
    import torch
    import os
    import sys
    torch.set_num_threads(2)
    torch.manual_seed(42)
    text = XCLIPTextConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77,
                          bos_token_id=0, eos_token_id=1)
    vision = XCLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, mit_hidden_size=16, mit_intermediate_size=32,
        mit_num_hidden_layers=1, mit_num_attention_heads=4, image_size=224, patch_size=32, num_frames=8)
    config = XCLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                        projection_dim=16, prompt_layers=1, prompt_num_attention_heads=4)
    source = tmp_path / 'prompted-tiny'
    XCLIPModel(config).save_pretrained(source)
    VideoMAEImageProcessor().save_pretrained(source)
    (source / 'vocab.json').write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1, 'a</w>': 2}))
    (source / 'merges.txt').write_text('#version: 0.2\n')
    CLIPTokenizerFast(vocab_file=str(source / 'vocab.json'), merges_file=str(source / 'merges.txt')).save_pretrained(source)
    encoder = PromptedEncoder(source, revision='random-test-weights', local_only=True, device='cpu')
    bank = prompt_bank([])
    calls = []
    hook = encoder.model.prompts_generator.register_forward_hook(lambda *args: calls.append(1))
    with deny_network():
        zero = encoder.zero_shot(SampledVideo(synthetic_video), np.array([0., 1.]), bank)
    hook.remove()
    assert calls and zero.shape == (2, 4) and np.isfinite(zero).all()
    head = CategoryHead(np.zeros(16), np.ones(16), np.zeros((1, 16)), [2.], [0, 1], .01)
    selection = {'threshold': .4, 'selected_C': .01, 'supervised_weight': .5,
                 'window_sec': 4, 'detector_enabled': False}
    notices = tmp_path / 'typed-notices'
    notices.mkdir()
    for name in ('MODEL_CARD.md', 'XCLIP_LICENSE', 'MODEL_METADATA.json'):
        (notices / name).write_text('RANDOM TEST WEIGHTS — NOT A TRAINED MODEL')
    bundle = tmp_path / 'hybrid_v2'
    export_hybrid(encoder, head, bank, selection, {'dataset_fingerprint': 'synthetic'}, {},
                  bundle, tmp_path, notices)
    assert verify_export(bundle) > 10
    with deny_network():
        loaded = Detector.from_artifacts(bundle, device='cpu')
        assert isinstance(loaded, HybridDetector)
        result = analyze_video(synthetic_video, bundle, camera_profile='unknown_camera')
    assert result['events'] and result['tracking_status'] == 'disabled_by_development_selection'
    for event in result['events']:
        assert event['category_id']
        assert 1 <= len(event['evidence_frames']) <= 3
        assert all(0 <= f['timestamp_sec'] < result['duration_sec'] for f in event['evidence_frames'])
    environment = {**os.environ, 'PYTHONPATH': str(bundle), 'HF_HOME': str(tmp_path / 'empty-hf'),
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    process = subprocess.run([sys.executable, '-m', 'datathon.cli', str(synthetic_video),
                              '--artifacts', str(bundle), '--offline-check'], cwd=bundle, env=environment,
                             capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    saved = json.loads(process.stdout)
    assert saved['python_socket_guard_enabled'] and saved['events']
