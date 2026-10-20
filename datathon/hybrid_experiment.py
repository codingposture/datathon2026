"""Restartable, one-day hybrid experiment orchestration used by case.ipynb."""
from __future__ import annotations

from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

from .data import save_json, sha256_file
from .hybrid_data import build_typed_manifest, typed_split, typed_fingerprint, apply_scene_groups
from .hybrid import (PromptedEncoder, CategoryHead, prompt_bank, cached_zero_shot, blend_scores,
                     consecutive_scores, typed_events, select_threshold, metric_rank)
from .hybrid_evaluation import report, markdown_report
from .hybrid_inference import HybridDetector, export_hybrid
from .inference import hardware_info, verify_export
from .tracking import (PersonDetector, DETECTOR_ID, DETECTOR_REVISION, DEFAULT_PROFILE, load_profile,
                       valid_profile, departure_observations, assist_events)
from .video import MODEL_ID, MODEL_REVISION, VideoConfig, cached_features, cache_identity


class Experiment:
    def __init__(self, project_dir, dataset_dir=None, run_name='hybrid_v2', require_gpu=True):
        self.root = Path(project_dir).resolve()
        self.dataset = Path(dataset_dir or self.root / 'open').resolve()
        self.run = self.root / 'runs' / run_name
        self.run.mkdir(parents=True, exist_ok=True)
        import torch
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError('Connect a Colab GPU before running model experiments')
        torch.manual_seed(42); np.random.seed(42)
        previous = json.loads((self.root / 'review/manifest.json').read_text())
        self.manifest = build_typed_manifest(self.dataset, self.dataset / 'forte_labels_annotated.xlsx',
                                             previous, self.run / 'typed_manifest.json')
        self.records = apply_scene_groups(self.manifest['recordings'], self.root / 'review/annotations.csv')
        self.manifest['recordings'] = self.records
        save_json(self.run / 'typed_manifest.json', self.manifest)
        self.lookup = {r['recording_id']: r for r in self.records}
        self.split = typed_split(self.records, json.loads((self.root / 'review/legacy_split.json').read_text()),
                                 self.run / 'split.json')
        self.identity = {'schema_version': 2, 'dataset_fingerprint': typed_fingerprint(self.records),
            'model_revision': MODEL_REVISION, 'detector_revision': DETECTOR_REVISION,
            'contexts_sec': [4, 12], 'regularization_C': [.001, .01, .1],
            'blend_weights': [0, .25, .5, .75, 1], 'seed': 42,
            'source_hashes': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
            'versions': {k: metadata.version(k) for k in ('torch', 'transformers', 'numpy', 'av', 'scikit-learn')},
            'precision': 'float16' if torch.cuda.is_available() else 'float32'}
        identity_path = self.run / 'experiment_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != self.identity:
            raise ValueError('Code/data/dependencies changed: preserve this run and choose a new run name')
        save_json(identity_path, self.identity)
        save_json(self.run / 'runtime_hardware.json', hardware_info())
        self.encoder = None
        self.features = {}
        self.oof = {}

    @staticmethod
    def require_colab_training():
        """User constraint: production experiments run only on a Colab GPU."""
        import torch
        try:
            in_colab = importlib.util.find_spec('google.colab') is not None
        except ModuleNotFoundError:
            in_colab = False
        if not in_colab or not torch.cuda.is_available():
            raise RuntimeError('Training and model experiments are restricted to a Google Colab GPU runtime')

    def get_encoder(self, context):
        self.require_colab_training()
        config = VideoConfig(window_sec=float(context), batch_size=8)
        if self.encoder is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            self.encoder = PromptedEncoder(config=config)
        self.encoder.config = config
        return self.encoder

    def development_features(self):
        for context in (4, 12):
            encoder = self.get_encoder(context)
            items = {}
            for index, key in enumerate(self.split['development'], 1):
                items[key] = cached_features(self.lookup[key], self.dataset, self.run / 'feature_cache', encoder)
                print(f'features {context}s {index}/32 {self.lookup[key]["aliases"]}', flush=True)
            self.features[context] = items

    def standalone_and_blends(self):
        self.require_colab_training()
        if not self.features:
            self.development_features()
        dev = [self.lookup[k] for k in self.split['development']]
        candidates, folds_audit = [], []
        for context in (4, 12):
            encoder = self.get_encoder(context)
            oof_zero, available = {}, {}
            oof_supervised = {c: {} for c in (.001, .01, .1)}
            for fold_index, fold in enumerate(self.split['folds']):
                train = [self.lookup[k] for k in fold['train']]
                bank = prompt_bank(train)
                if set(bank['source_training_ids']) & set(fold['validation'] + self.split['test']):
                    raise AssertionError('Prompt leakage')
                save_json(self.run / f'prompts_{context}s_fold{fold_index}.json', bank)
                heads = {}
                for c in (.001, .01, .1):
                    head, audit = CategoryHead.fit(train, self.features[context], c, context)
                    heads[c] = head
                    head.save(self.run / f'heads/{context}s_fold{fold_index}_C{c}.json')
                    folds_audit.append({'context_sec': context, 'fold': fold_index, 'C': c, **audit})
                for key in fold['validation']:
                    record = self.lookup[key]
                    item = cached_zero_shot(record, self.dataset, self.run / 'zero_shot_cache', encoder, bank)
                    np.testing.assert_allclose(item['centers'], self.features[context][key]['centers'])
                    oof_zero[key] = item['scores']
                    for c, head in heads.items():
                        oof_supervised[c][key] = head.predict(self.features[context][key]['features'])
                    available[key] = heads[.001].classes
                    print(f'zero-shot {context}s fold {fold_index + 1}/4 {record["aliases"]}', flush=True)
            save_json(self.run / 'fold_training_audit.json', folds_audit)
            for c in (.001, .01, .1):
                for weight in (0, .25, .5, .75, 1):
                    if weight == 0 and c != .001:
                        continue
                    window_scores = {k: blend_scores(oof_supervised[c][k], oof_zero[k], weight, available[k])
                                     for k in self.split['development']}
                    category_scores = [consecutive_scores(self.features[context][k]['centers'], window_scores[k])
                                       for k in self.split['development']]
                    candidate = select_threshold(dev, category_scores)
                    candidate.update(window_sec=context, selected_C=c, supervised_weight=weight,
                                     branch='zero_shot' if weight == 0 else 'supervised' if weight == 1 else 'blend')
                    candidates.append(candidate)
                    self.oof[(context, c, weight)] = window_scores
                    print('selection', context, c, weight, candidate['metrics']['balanced_accuracy'], flush=True)
            save_json(self.run / 'comparison_partial.json', candidates)
        best = max(candidates, key=lambda item: metric_rank(item['metrics']))
        self.selection = {k: best[k] for k in ('window_sec', 'selected_C', 'supervised_weight', 'threshold', 'branch')}
        self.selection['detector_enabled'] = False
        self.selection['selection_criterion'] = ['binary_balanced_accuracy', 'category_macro_f1', 'fewer_false_positive_videos']
        self.selection['missing_class_policy'] = 'No supervised score; intermediate blends use zero-shot fallback'
        self.selected_oof = self.oof[(best['window_sec'], best['selected_C'], best['supervised_weight'])]
        self.development_outputs = {}
        for key in self.split['development']:
            centers = self.features[best['window_sec']][key]['centers']
            scores = self.selected_oof[key]
            self.development_outputs[key] = {'events': typed_events(centers, scores, best['threshold']),
                                             'score': float(consecutive_scores(centers, scores).max())}
        self.development_report = report(dev, self.development_outputs, best['threshold'])
        self.development_report['temporal_stage'] = 'one_second_coarse_scans_for_OOF_selection; final inference refines quarter-second'
        save_json(self.run / 'standalone_comparison.json', {'candidates': candidates, 'selected': self.selection,
            'standalone_best': {branch: max([c for c in candidates if c['branch'] == branch],
                                           key=lambda x: metric_rank(x['metrics']))
                                for branch in ('zero_shot', 'supervised', 'blend')}})
        save_json(self.run / 'development_evaluation.json', self.development_report)
        save_json(self.run / 'model_selection.json', self.selection)
        return self.selection

    def detector_experiment(self):
        """Run only after standalone selection; never use diagnostic test to tune."""
        self.require_colab_training()
        detector = PersonDetector()
        profile_path = self.root / 'review/camera_profile.json'
        self.profile = load_profile(profile_path) if profile_path.exists() else DEFAULT_PROFILE.copy()
        if valid_profile(self.profile):
            if any(e['video_sha256'] not in self.split['development'] for e in self.profile['role_examples']):
                raise ValueError('Camera role examples must come from development recordings only')
        profile_identity = self.run / 'camera_profile_identity.json'
        if profile_identity.exists() and json.loads(profile_identity.read_text()) != self.profile:
            raise ValueError('Camera profile changed: preserve this experiment and choose a new run name')
        save_json(profile_identity, self.profile)
        outputs, statuses, inventory = {}, [], []
        cache_dir = self.run / 'person_cache'
        cache_dir.mkdir(exist_ok=True)
        for index, key in enumerate(self.split['development'], 1):
            record = self.lookup[key]
            cache = cache_dir / f'{key}_{DETECTOR_REVISION}.json'
            if cache.exists():
                observations = json.loads(cache.read_text())
            else:
                observations = detector.observe(self.dataset / record['relative_path'], record['duration_sec'])
                save_json(cache, observations)
            tracking = departure_observations(observations, self.profile)
            statuses.append(tracking['status'])
            events = assist_events(self.development_outputs[key]['events'], tracking, self.profile)
            outputs[key] = {**self.development_outputs[key], 'events': events}
            inventory.append({'recording_id': key, 'scans': len(observations),
                              'person_boxes': sum(len(o['boxes']) for o in observations), **tracking})
            print(f'person tracking {index}/32: {tracking["status"]}', flush=True)
        dev = [self.lookup[k] for k in self.split['development']]
        assisted = report(dev, outputs, self.selection['threshold'])
        before, after = self.development_report['model'], assisted['model']
        retained = (valid_profile(self.profile) and after['balanced_accuracy'] > before['balanced_accuracy']
                    and after['category_macro_f1'] >= before['category_macro_f1'])
        self.selection['detector_enabled'] = bool(retained)
        if retained:
            self.person_detector = detector
            self.development_report = assisted
        else:
            del detector
            gc.collect()
        self.detector_results = {'status': 'retained' if retained else 'not_retained',
             'reason': 'Requires confirmed camera and strictly higher OOF balanced accuracy without macro-F1 loss',
             'camera_profile_valid': valid_profile(self.profile), 'observations': inventory,
             'before': before, 'after': after, 'model_id': DETECTOR_ID, 'revision': DETECTOR_REVISION,
             'license': 'apache-2.0', 'person_only': True, 'role_statuses': sorted(set(statuses))}
        save_json(self.run / 'detector_experiment.json', self.detector_results)
        save_json(self.run / 'model_selection.json', self.selection)
        save_json(self.run / 'development_evaluation.json', self.development_report)
        return self.detector_results

    def fit_final_and_evaluate(self):
        self.require_colab_training()
        dev = [self.lookup[k] for k in self.split['development']]
        context = self.selection['window_sec']
        self.head, audit = CategoryHead.fit(dev, self.features[context], self.selection['selected_C'], context)
        self.head.save(self.run / 'head.json')
        self.prompts = prompt_bank(dev)
        save_json(self.run / 'prompts.json', self.prompts)
        save_json(self.run / 'final_training_audit.json', audit)
        encoder = self.get_encoder(context)
        detector = HybridDetector(encoder, self.head, self.prompts, self.selection,
                                  getattr(self, 'person_detector', None))
        path = self.run / 'evaluation.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['experiment_identity'] != self.identity or saved['selection'] != self.selection:
                raise ValueError('Evaluation already exists for a different configuration')
            self.evaluation = saved
            return saved
        outputs = {}
        for key in self.split['test']:
            record = self.lookup[key]
            output_path = self.run / 'test_outputs' / key / 'result.json'
            if output_path.exists():
                saved = json.loads(output_path.read_text())
                if saved['experiment_identity'] != self.identity or saved['selection'] != self.selection:
                    raise ValueError('Partial test output has a different configuration')
                result = saved['result']
            else:
                # Test processed only after configuration is fixed. No descriptions supplied.
                result = detector.predict(self.dataset / record['relative_path'], output_path.parent,
                                           camera_profile=self.profile)
                save_json(output_path, {'experiment_identity': self.identity,
                                       'selection': self.selection, 'result': result})
            outputs[key] = result
            print('diagnostic test:', record['aliases'], result['verdict'], flush=True)
        self.evaluation = report([self.lookup[k] for k in self.split['test']], outputs,
                                  self.selection['threshold'], diagnostic=True)
        self.evaluation.update(experiment_identity=self.identity, selection=self.selection,
                               annotation_mode='provisional_typed_workbook', threshold=self.selection['threshold'])
        save_json(path, self.evaluation)
        return self.evaluation

    def notices(self):
        from huggingface_hub import hf_hub_download
        notices = self.run / 'model_notices'
        notices.mkdir(exist_ok=True)
        for model, revision, prefix, license_id in ((MODEL_ID, MODEL_REVISION, 'XCLIP', 'mit'),
                                                    (DETECTOR_ID, DETECTOR_REVISION, 'RTDETR', 'apache-2.0')):
            url = f'https://huggingface.co/api/models/{model}/revision/{revision}'
            with urllib.request.urlopen(url, timeout=60) as response:
                info = json.load(response)
            if info.get('cardData', {}).get('license') != license_id:
                raise ValueError(f'Unexpected license: {model}')
            card = hf_hub_download(model, 'README.md', revision=revision)
            shutil.copy2(card, notices / ('MODEL_CARD.md' if prefix == 'XCLIP' else 'RTDETR_MODEL_CARD.md'))
            save_json(notices / ('MODEL_METADATA.json' if prefix == 'XCLIP' else 'RTDETR_MODEL_METADATA.json'), info)
        sources = {'XCLIP_LICENSE': 'https://raw.githubusercontent.com/microsoft/VideoX/master/LICENSE',
                   'RTDETR_LICENSE': 'https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/LICENSE'}
        for filename, url in sources.items():
            with urllib.request.urlopen(url, timeout=60) as response:
                (notices / filename).write_bytes(response.read())
        save_json(notices / 'SOURCES.json', sources)
        return notices

    def export_and_verify(self):
        text = markdown_report(self.selection, self.development_report, self.evaluation, self.detector_results)
        (self.run / 'MODEL_REPORT.md').write_text(text)
        bundle = self.run / 'exported_model'
        if not bundle.exists():
            export_hybrid(self.get_encoder(self.selection['window_sec']), self.head, self.prompts,
                           self.selection, self.split, self.evaluation, bundle, self.run, self.notices(),
                           getattr(self, 'person_detector', None), self.profile)
        verify_export(bundle)
        # Fresh processes, no feature caches, empty HF cache and socket denial.
        local = Path('/content/hybrid_v2_offline_bundle') if Path('/content').is_dir() else self.run / 'offline_bundle'
        shutil.copytree(bundle, local, dirs_exist_ok=True)
        self.encoder = None
        if hasattr(self, 'person_detector'):
            del self.person_detector
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        longest = max(self.records, key=lambda r: r['duration_sec'])
        benchmarks = []
        for index in (1, 2):
            report_path = self.run / f'offline_check_{index}.json'
            environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                           'HF_HOME': str(self.run / 'empty_cache'), 'PYTHONPATH': str(local),
                           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
            command = [sys.executable, '-m', 'datathon.cli', str(self.dataset / longest['relative_path']),
                       '--artifacts', str(local), '--offline-check', '--report', str(report_path)]
            process = subprocess.run(command, cwd=local, env=environment, text=True, capture_output=True, timeout=1200)
            if process.returncode:
                raise RuntimeError(process.stderr[-6000:])
            benchmarks.append(json.loads(report_path.read_text()))
        assert benchmarks[0]['verdict'] == benchmarks[1]['verdict']
        np.testing.assert_allclose(benchmarks[0]['score'], benchmarks[1]['score'], atol=1e-5)
        np.testing.assert_allclose([e['timecode_sec'] for e in benchmarks[0]['events']],
                                   [e['timecode_sec'] for e in benchmarks[1]['events']], atol=.25)
        acceptance = {'longest_video': longest['aliases'],
            'duration_sec': longest['duration_sec'], 'passed': all(r['within_5_minute_budget'] for r in benchmarks),
            'hardware_specific': True, 'checks': benchmarks}
        save_json(self.run / 'latency_acceptance.json', acceptance)
        # Include acceptance evidence in the final checksum-protected downloadable archive.
        save_json(bundle / 'latency_acceptance.json', acceptance)
        for index in (1, 2):
            shutil.copy2(self.run / f'offline_check_{index}.json', bundle)
        checksums = {str(p.relative_to(bundle)): sha256_file(p) for p in bundle.rglob('*')
                     if p.is_file() and p.name != 'COMPLETE.json' and 'inference_outputs' not in p.parts}
        save_json(bundle / 'COMPLETE.json', {'schema_version': 2, 'sha256': checksums})
        verify_export(bundle)
        print('EXPORTED', bundle, 'five-minute limit:', all(r['within_5_minute_budget'] for r in benchmarks), flush=True)
        return bundle
