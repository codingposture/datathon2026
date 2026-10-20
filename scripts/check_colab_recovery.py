"""Verify real hybrid caches in a fresh Colab GPU process, without recomputation."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/content/Datathon'))
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from datathon.data import save_json
    from datathon.hybrid_experiment import Experiment
    from datathon.hybrid import PromptedEncoder, cached_zero_shot
    from datathon.inference import deny_network
    from datathon.video import VideoConfig, cached_features
    Experiment.require_colab_training()
    run = root / 'runs/hybrid_v2'
    bundle = run / 'exported_model'
    records = {r['recording_id']: r for r in json.loads((run / 'typed_manifest.json').read_text())['recordings']}
    split = json.loads((run / 'split.json').read_text())
    def forbidden(*args, **kwargs):
        raise RuntimeError('Recovery check attempted to recompute a cached model output')
    count_features, count_zero_shot = 0, 0
    with deny_network():
        encoder = PromptedEncoder(bundle / 'encoder', device='cuda', local_only=True)
        encoder.encode = forbidden
        encoder.zero_shot = forbidden
        for context in (4, 12):
            encoder.config = VideoConfig(window_sec=float(context))
            for key in split['development']:
                cached_features(records[key], root / 'open', run / 'feature_cache', encoder)
                count_features += 1
            for index, fold in enumerate(split['folds']):
                bank = json.loads((run / f'prompts_{context}s_fold{index}.json').read_text())
                for key in fold['validation']:
                    cached_zero_shot(records[key], root / 'open', run / 'zero_shot_cache', encoder, bank)
                    count_zero_shot += 1
    result = {'passed': True, 'fresh_process': True, 'python_socket_guard_enabled': True,
              'model_recomputation_forbidden': True, 'feature_caches_loaded': count_features,
              'zero_shot_caches_loaded': count_zero_shot, 'device': 'Colab GPU'}
    save_json(run / 'cache_recovery_check.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
