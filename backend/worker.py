"""CPU inference only. Temporary evidence is consumed as base64, never persisted to the model."""
import argparse
import base64
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys

DESCRIPTIONS = {
    'CASH_OR_WORKPLACE_NOT_SECURED': 'The model flagged a possible cash or workstation security issue near this timestamp. Review the frames and surrounding video to check whether cash, keys, storage or equipment were left unsecured.',
    'ACCEPTANCE_STEP_SKIPPED': 'The model flagged a possible skipped acceptance step near this timestamp. Review the surrounding video to check identification, counting or banknote inspection.',
    'CASH_AREA_ACCESS_UNCONTROLLED': 'The model flagged possible uncontrolled cash-area access near this timestamp. Review who entered and whether access was controlled.',
    'PERSONAL_ITEMS_IN_CASH_AREA': 'The model flagged a possible personal item in the cash area near this timestamp. Review the evidence to confirm the object and its location.',
}


def api_result(result, evidence_dir, max_bytes):
    from datathon.hybrid_data import TAXONOMY
    root = Path(evidence_dir).resolve()
    violations, total_bytes = [], 0
    duration = float(result['duration_sec'])
    for event in result['events']:
        category = event['category_id']
        if category not in DESCRIPTIONS:
            raise ValueError('Unknown model category')
        time = float(event['timecode_sec'])
        if not math.isfinite(time) or not 0 <= time < duration:
            raise ValueError('Invalid event timestamp')
        source_frames = event['evidence_frames']
        if not 1 <= len(source_frames) <= 3:
            raise ValueError('Expected one to three evidence frames')
        frames = []
        for frame in source_frames:
            path = Path(frame['image_path']).resolve()
            timestamp = float(frame['timestamp_sec'])
            if not path.is_relative_to(root) or not math.isfinite(timestamp) or not 0 <= timestamp < duration:
                raise ValueError('Invalid evidence frame')
            if path.stat().st_size > max_bytes:
                raise ValueError('Evidence is too large')
            image = path.read_bytes()
            if not image.startswith(b'\xff\xd8\xff') or not image.endswith(b'\xff\xd9'):
                raise ValueError('Evidence is not JPEG')
            encoded = base64.b64encode(image).decode('ascii')
            total_bytes += len(encoded)
            if total_bytes > max_bytes:
                raise ValueError('Evidence response is too large')
            frames.append({'timestamp_sec': timestamp, 'image_base64': encoded})
        violations.append({'type': TAXONOMY[category]['name'], 'category_id': category,
            'timecode_sec': time, 'description': DESCRIPTIONS[category],
            'evidence_frames': frames, 'rule_reference': None, 'score': float(event['score'])})
    verdict = 'violation' if violations else 'no_violation'
    if result['verdict'] != verdict:
        raise ValueError('Model verdict contradicts events')
    payload = {'verdict': verdict, 'violations': violations, 'score': float(result['score']),
               'score_kind': result['score_kind'], 'experimental': True}
    if len(json.dumps(payload, allow_nan=False).encode()) > max_bytes:
        raise ValueError('Analysis response is too large')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--max-result-bytes', type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        import torch
        from datathon.inference import Detector, deny_network, verify_export
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
        torch.set_num_interop_threads(1)
        with deny_network():
            verify_export(args.artifacts)
            detector = Detector.from_artifacts(args.artifacts, device='cpu')
            result = detector.predict(args.video, args.evidence)
            payload = api_result(result, args.evidence, args.max_result_bytes)
    print(json.dumps(payload, allow_nan=False))


if __name__ == '__main__':
    main()
