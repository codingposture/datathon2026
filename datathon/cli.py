"""python -m datathon.cli video.mov --artifacts exported_model --offline-check"""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import time

from .data import save_json
from .inference import analyze_video, deny_network, hardware_info, verify_export


def main():
    parser = argparse.ArgumentParser(description='Local binary video detector')
    parser.add_argument('video', type=Path)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--offline-check', action='store_true', help='Block Python socket connections during load and inference')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--camera-profile', help='Confirmed local profile JSON or bundled profile name')
    args = parser.parse_args()
    with deny_network() if args.offline_check else nullcontext():
        verified = verify_export(args.artifacts)
        started = time.perf_counter()
        result = analyze_video(args.video, args.artifacts, camera_profile=args.camera_profile)
        result['cold_load_and_inference_sec'] = time.perf_counter() - started
        result['hardware'] = hardware_info()
        result['python_socket_guard_enabled'] = args.offline_check
        result['verified_artifact_files'] = verified
        result['within_5_minute_budget'] = result['cold_load_and_inference_sec'] <= 300
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
