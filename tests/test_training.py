import copy
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from datathon.training import (LinearHead, binary_metrics, evaluation_report, fit_head,
                               make_split, matched_errors, select_threshold, training_rows, tune, validate_split)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(42)
    records, features = [], {}
    for i in range(40):
        target = i % 2
        recording_id = f'recording-{i}'
        records.append({'recording_id': recording_id, 'group_id': f'session-{i // 2}',
                        'label': target, 'event_times_sec': [5.0] if target else [],
                        'aliases': [f'X-{i}.mov'], 'duration_sec': 10})
        x = rng.normal(0, 0.1, size=(10, 12))
        if target:
            x[4:7, 0] += 3
        features[recording_id] = {'centers': np.arange(10, dtype=float), 'features': x}
    return records, features


def test_grouped_split_resume_and_drift(toy_data, tmp_path):
    records, features = toy_data
    path = tmp_path / 'split.json'
    split = make_split(records, path)
    assert split == make_split(records, path)
    validate_split(records, split)
    assert len(split['test']) == 8
    changed = copy.deepcopy(records)
    changed[0]['group_id'] = 'other'
    with pytest.raises(ValueError, match='changed'):
        make_split(changed, path)
    leaking = copy.deepcopy(split)
    leaking['folds'][0]['train'].append(leaking['folds'][0]['validation'][0])
    with pytest.raises(ValueError):
        validate_split(records, leaking)


def test_training_window_masks_and_weights(toy_data):
    records, features = toy_data
    ids = [r['recording_id'] for r in records[:4]]
    x, y, weights = training_rows(records, features, ids)
    assert len(x) == 26  # 2 normal videos * 10 + 2 positive videos * 3.
    assert weights[y == 0].sum() == pytest.approx(weights[y == 1].sum())
    assert len(y[y == 1]) == 6
    # Positive windows outside the reviewed neighborhood never enter training.
    for r in records[:4]:
        if r['label']:
            features[r['recording_id']]['features'][[0, 1, 2, 3, 7, 8, 9]] = 999
    new_x, _, _ = training_rows(records, features, ids)
    np.testing.assert_equal(new_x, x)


def test_head_export_matches_sklearn(toy_data, tmp_path):
    records, features = toy_data
    ids = [r['recording_id'] for r in records]
    x, y, weights = training_rows(records, features, ids)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    reference = LogisticRegression(C=0.1, max_iter=3000, random_state=42).fit(scaler.transform(x), y, sample_weight=weights)
    head = fit_head(records, features, ids, 0.1)
    np.testing.assert_allclose(head.predict(x), reference.predict_proba(scaler.transform(x))[:, 1])
    head.save(tmp_path / 'head.json')
    np.testing.assert_equal(LinearHead.load(tmp_path / 'head.json').predict(x), head.predict(x))


def test_tuning_cannot_access_test_features(toy_data, tmp_path):
    records, features = toy_data
    split = make_split(records, tmp_path / 'split.json')
    development_only = {key: features[key] for key in split['development']}
    head, selection = tune(records, development_only, split, tmp_path)
    assert selection['selection_data'] == 'development_out_of_fold_only'
    assert len(selection['candidates']) == 3
    for candidate in selection['candidates']:
        ids = {row['recording_id'] for row in candidate['oof_predictions']}
        assert ids == set(split['development'])
        assert not ids.intersection(split['test'])


def test_threshold_f2_and_constant_baselines():
    threshold, metrics = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.3, 0.8])
    assert threshold == 0.3
    assert metrics['recall'] == 1
    assert binary_metrics([0, 1], [0, 0], 0.5)['confusion_matrix_labels_0_1'] == [[1, 0], [1, 0]]


def test_one_to_one_event_matching():
    assert len(matched_errors([0, 3], [1.5, 4.5], 2)) == 2
    assert len(matched_errors([0, 1], [0.5], 2)) == 1
    assert matched_errors([0], [20], 5) == []
    assert matched_errors([], [0], 5) == []


def test_report_retains_missed_events_and_false_events(toy_data):
    records, _ = toy_data
    records = records[:2]
    outputs = {records[0]['recording_id']: {'verdict': 'violation', 'score': 0.9, 'events': [{'timecode_sec': 1}]},
               records[1]['recording_id']: {'verdict': 'no_violation', 'score': 0.1, 'events': []}}
    report = evaluation_report(records, outputs, 0.5)
    assert len(report['mistakes']) == 2
    assert report['events']['within_2_sec']['recall'] == 0
    assert report['events']['false_events_per_negative_video'] == 1
