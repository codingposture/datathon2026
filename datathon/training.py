"""Group-isolated splitting, video-balanced training, and held-out metrics."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .data import fingerprint, save_json

SEED = 42


def validate_split(records, split):
    if split['dataset_fingerprint'] != fingerprint(records):
        raise ValueError('Annotations/groups changed. Archive previous run and create a new split deliberately.')
    lookup = {r['recording_id']: r for r in records}
    dev, test = set(split['development']), set(split['test'])
    if dev & test or dev | test != set(lookup):
        raise ValueError('Split must partition every unique recording exactly once')
    def groups(ids):
        return {lookup[i]['group_id'] for i in ids}
    def both_labels(ids):
        return {lookup[i]['label'] for i in ids} == {0, 1}
    if groups(dev) & groups(test) or not both_labels(dev) or not both_labels(test):
        raise ValueError('Test/development must be group-disjoint and each contain both classes')
    seen = []
    if len(split['folds']) != 4:
        raise ValueError('Exactly four development folds required')
    for fold in split['folds']:
        train, valid = set(fold['train']), set(fold['validation'])
        if train & valid or train | valid != dev or groups(train) & groups(valid):
            raise ValueError('Cross-validation leakage or incomplete fold')
        if not both_labels(train) or not both_labels(valid):
            raise ValueError('Every training/validation fold must contain both classes')
        seen.extend(valid)
    if len(seen) != len(dev) or set(seen) != dev:
        raise ValueError('Every development recording must be validated exactly once')


def make_split(records, path, seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold
    path = Path(path)
    if path.exists():
        split = json.loads(path.read_text())
        validate_split(records, split)
        return split
    ids = np.array([r['recording_id'] for r in records])
    y = np.array([r['label'] for r in records])
    groups = np.array([r['group_id'] for r in records])
    if len(set(groups)) < 5 or min(np.bincount(y, minlength=2)) < 5:
        raise ValueError('Too few independent groups/classes for a 20% holdout plus four-fold CV. Do not split related scenes to bypass this.')
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    for development, test in outer.split(ids, y, groups):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        split = {'schema_version': 1, 'seed': seed, 'dataset_fingerprint': fingerprint(records),
                 'development': ids[development].tolist(), 'test': ids[test].tolist(), 'folds': []}
        for train, valid in inner.split(ids[development], y[development], groups[development]):
            split['folds'].append({'train': ids[development[train]].tolist(),
                                   'validation': ids[development[valid]].tolist()})
        try:
            validate_split(records, split)
        except ValueError:
            continue
        distance = abs(len(test) / len(ids) - 0.2) + abs(y[test].mean() - y.mean())
        candidates.append((distance, split))
    if not candidates:
        raise ValueError('Reviewed scene groups do not support class-balanced holdout and four-fold CV. Revisit evaluation design, not true scene grouping.')
    split = min(candidates, key=lambda item: item[0])[1]
    save_json(path, split)
    return split


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict(self, features):
        logits = ((np.asarray(features, dtype=np.float64) - self.mean) / self.scale) @ self.coefficient + self.intercept
        # Stable sigmoid, clipped only beyond double-precision practical range.
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))

    def save(self, path):
        save_json(path, {'schema_version': 1, 'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                         'coefficient': self.coefficient.tolist(), 'intercept': self.intercept,
                         'score_kind': 'uncalibrated_logistic_score'})

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        mean, scale, coefficient = [np.asarray(data[k], dtype=np.float64) for k in ('mean', 'scale', 'coefficient')]
        if not (mean.ndim == 1 and mean.shape == scale.shape == coefficient.shape
                and np.isfinite([mean, scale, coefficient]).all() and (scale > 0).all()
                and np.isfinite(data['intercept'])):
            raise ValueError('Malformed classifier parameters')
        return cls(mean, scale, coefficient, float(data['intercept']))


def training_rows(records, features, ids):
    lookup = {r['recording_id']: r for r in records}
    xs, ys, ws = [], [], []
    counts = {label: sum(lookup[i]['label'] == label for i in ids) for label in (0, 1)}
    if not all(counts.values()):
        raise ValueError('Training requires both classes')
    for recording_id in ids:
        recording = lookup[recording_id]
        item = features[recording_id]
        if recording['label']:
            events = np.asarray(recording['event_times_sec'])
            selected = (np.abs(item['centers'][:, None] - events[None, :]) <= 1.0).any(axis=1)
        else:
            selected = np.ones(len(item['centers']), dtype=bool)
        x = item['features'][selected]
        if not len(x):
            raise ValueError(f'No labeled training windows: {recording_id}')
        xs.append(x)
        ys.extend([recording['label']] * len(x))
        # Equal total weight per recording within each class; equal class totals.
        ws.extend([1 / (len(x) * counts[recording['label']])] * len(x))
    weights = np.array(ws, dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return np.concatenate(xs), np.asarray(ys), weights


def fit_head(records, features, ids, regularization):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x, y, weights = training_rows(records, features, ids)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    classifier = LogisticRegression(C=regularization, solver='lbfgs', max_iter=3000, random_state=SEED)
    classifier.fit(scaler.transform(x), y, sample_weight=weights)
    if classifier.n_iter_.max() >= classifier.max_iter:
        raise RuntimeError('Classifier did not converge; investigate features before evaluating test data')
    return LinearHead(scaler.mean_, scaler.scale_, classifier.coef_[0], float(classifier.intercept_[0]))


def binary_metrics(labels, scores, threshold):
    from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                                 precision_score, recall_score)
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = (scores >= threshold).astype(int)
    return {'precision': float(precision_score(labels, predicted, zero_division=0)),
            'recall': float(recall_score(labels, predicted, zero_division=0)),
            'f1': float(fbeta_score(labels, predicted, beta=1, zero_division=0)),
            'f2': float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
            'pr_auc_average_precision': float(average_precision_score(labels, scores)),
            'confusion_matrix_labels_0_1': confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            'n_videos': len(labels)}


def select_threshold(labels, scores):
    # Evaluating observed scores enumerates every attainable nonempty positive set.
    thresholds = np.unique(np.r_[scores, np.nextafter(np.max(scores), np.inf)])
    options = [(binary_metrics(labels, scores, float(t)), float(t)) for t in thresholds]
    metrics, threshold = max(options, key=lambda option: (option[0]['f2'], option[0]['precision'], option[1]))
    return threshold, metrics


def tune(records, features, split, output_dir):
    validate_split(records, split)
    lookup = {r['recording_id']: r for r in records}
    candidates = []
    for regularization in (0.01, 0.1, 1.0):
        rows = []
        for fold_index, fold in enumerate(split['folds']):
            head = fit_head(records, features, fold['train'], regularization)
            for recording_id in fold['validation']:
                score = float(head.predict(features[recording_id]['features']).max())
                rows.append({'recording_id': recording_id, 'label': lookup[recording_id]['label'],
                             'score': score, 'fold': fold_index})
        threshold, metrics = select_threshold([r['label'] for r in rows], [r['score'] for r in rows])
        candidates.append({'C': regularization, 'threshold': threshold, 'oof_metrics': metrics, 'oof_predictions': rows})
    selected = max(candidates, key=lambda item: (item['oof_metrics']['f2'], item['oof_metrics']['precision'], -item['C']))
    selection = {'schema_version': 1, 'dataset_fingerprint': split['dataset_fingerprint'],
                 'selection_data': 'development_out_of_fold_only', 'selected_C': selected['C'],
                 'threshold': selected['threshold'], 'candidates': candidates}
    save_json(Path(output_dir) / 'model_selection.json', selection)
    head = fit_head(records, features, split['development'], selected['C'])
    head.save(Path(output_dir) / 'head.json')
    return head, selection


def matched_errors(truth, predictions, tolerance):
    """Maximum-cardinality one-to-one matching, minimizing error among matches."""
    from scipy.optimize import linear_sum_assignment
    if not truth or not predictions:
        return []
    distance = np.abs(np.asarray(truth)[:, None] - np.asarray(predictions)[None, :])
    penalty = (max(distance.shape) + 1) * (distance.max() + tolerance + 1)
    rows, cols = linear_sum_assignment(np.where(distance <= tolerance, distance, penalty))
    return [float(distance[r, c]) for r, c in zip(rows, cols) if distance[r, c] <= tolerance]


def evaluation_report(records, outputs, threshold):
    labels = [r['label'] for r in records]
    scores = [outputs[r['recording_id']]['score'] for r in records]
    result = {'model': binary_metrics(labels, scores, threshold),
              'baselines': {'always_negative': binary_metrics(labels, np.zeros(len(labels)), 0.5),
                            'always_positive': binary_metrics(labels, np.ones(len(labels)), 0.5)},
              'predictions': [], 'mistakes': [], 'events': {}}
    for recording in records:
        output = outputs[recording['recording_id']]
        predicted = int(output['verdict'] == 'violation')
        row = {'recording_id': recording['recording_id'], 'aliases': recording['aliases'],
               'label': recording['label'], 'prediction': predicted, 'score': output['score'],
               'duration_sec': recording['duration_sec'],
               'truth_events': recording['event_times_sec'],
               'predicted_events': [e['timecode_sec'] for e in output['events']]}
        result['predictions'].append(row)
        if predicted != recording['label']:
            result['mistakes'].append(row)
    event_count = sum(len(r['event_times_sec']) for r in records)
    negative = [r for r in records if not r['label']]
    false_events = sum(len(outputs[r['recording_id']]['events']) for r in negative)
    for tolerance in (2, 5):
        errors = []
        for recording in records:
            predicted = [e['timecode_sec'] for e in outputs[recording['recording_id']]['events']]
            errors.extend(matched_errors(recording['event_times_sec'], predicted, tolerance))
        result['events'][f'within_{tolerance}_sec'] = {
            'matched': len(errors), 'annotated': event_count,
            'recall': len(errors) / event_count if event_count else None,
            'mean_matched_error_sec': float(np.mean(errors)) if errors else None,
            'median_matched_error_sec': float(np.median(errors)) if errors else None}
    result['events']['false_events_per_negative_video'] = false_events / len(negative) if negative else None
    result['limitations'] = ['Small held-out sample; no minimum quality guaranteed.',
                            'Two/five-second tolerances are diagnostic, not official jury tolerances.',
                            'Unseen violation types and camera angles remain unverified.',
                            'Scores are uncalibrated; candidate events need human review.']
    return result
