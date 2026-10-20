from copy import deepcopy
from pathlib import Path
import json
import zipfile

import numpy as np
import pytest

from datathon.hybrid_data import (CATEGORIES, build_typed_manifest, read_typed_workbook, typed_split, apply_scene_groups,
                                 event_distance, workbook_sheets)
from datathon.hybrid import (CategoryHead, blend_scores, consecutive_scores, prompt_bank,
                            training_arrays, typed_events, select_threshold)
from datathon.hybrid_evaluation import temporal_report
from datathon.tracking import (DEFAULT_PROFILE, valid_profile, departure_observations, assist_events)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def dataset():
    return build_typed_manifest(ROOT / 'open', ROOT / 'open/forte_labels_annotated.xlsx',
                                json.loads((ROOT / 'review/manifest.json').read_text()))


def test_named_workbook_duplicate_events_and_partition(dataset):
    rows, sheets = read_typed_workbook(ROOT / 'open/forte_labels_annotated.xlsx')
    assert len(rows) == 42 and {'labels', 'types', 'sequences'}.issubset(sheets)
    records = dataset['recordings']
    assert len(records) == 40 and sum(r['label'] for r in records) == 21
    cash = next(r for r in records if len(r['events']) == 2)
    assert len(cash['aliases']) == 2
    assert [(e['start_sec'], e['end_sec']) for e in cash['events']] == [(53, 57), (68, 68)]
    assert [e['category_id'] for e in cash['events']] == list(CATEGORIES[:2])
    assert any(r['untimed_events'] for r in records)
    split = typed_split(records, json.loads((ROOT / 'review/legacy_split.json').read_text()))
    lookup = {r['recording_id']: r for r in records}
    assert len(split['development']) == 32 and len(split['test']) == 8
    assert not any(e['category_id'] == CATEGORIES[0] for k in split['test'] for e in lookup[k]['events'])
    assert any(CATEGORIES[3] in f['missing_training_categories'] for f in split['folds'])
    for fold in split['folds']:
        assert not {lookup[k]['group_id'] for k in fold['train']} & {lookup[k]['group_id'] for k in fold['validation']}


def test_formula_is_inert_and_rejected_as_target(tmp_path):
    source = ROOT / 'open/forte_labels_annotated.xlsx'
    target = tmp_path / 'formula.xlsx'
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, 'w') as out:
        for name in src.namelist():
            data = src.read(name)
            if name == 'xl/worksheets/sheet2.xml':
                text = data.decode()
                import re
                text, count = re.subn(r'(<c r="F2"[^>]*>).*?</c>', r'\1<f>1+0</f><v>1</v></c>', text, count=1)
                assert count == 1
                data = text.encode()
            out.writestr(name, data)
    assert isinstance(workbook_sheets(target)['labels'][0]['label_binary'], dict)
    with pytest.raises(ValueError, match='Formulas'):
        read_typed_workbook(target)


def test_new_scene_overlap_is_flagged_without_changing_test_partition(dataset, tmp_path):
    import csv
    old = json.loads((ROOT / 'review/legacy_split.json').read_text())
    groups = tmp_path / 'groups.csv'
    with groups.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['recording_id', 'group_id'])
        writer.writeheader()
        for key in (old['development'][0], old['test'][0]):
            writer.writerow({'recording_id': key, 'group_id': 'newly_confirmed_scene'})
    records = apply_scene_groups(dataset['recordings'], groups)
    with pytest.raises(ValueError, match='scene overlap'):
        typed_split(records, old)


def record(key, label, events):
    return {'recording_id': key, 'label': label, 'events': events, 'untimed_events': []}


def event(category=CATEGORIES[0], start=3., end=3., weight=1.):
    return {'category_id': category, 'start_sec': start, 'end_sec': end, 'weight': weight,
            'certainty': 'HIGH', 'description': 'cash left on the table', 'normal_contrast': 'cash stored away'}


def test_interval_certainty_weights_ambiguity_and_missing_classes(tmp_path):
    records = [record('normal', 0, []), record('positive', 1, [event(start=3, end=5, weight=.25)])]
    items = {k: {'centers': np.arange(9), 'features': np.random.default_rng(42).normal(size=(9, 6))}
             for k in ('normal', 'positive')}
    x, y, w, audit = training_arrays(records, items)
    assert (y == 1).sum() == 5  # centers 2,3,4,5,6 around interval
    assert np.isclose(w[y == 1].sum(), .25) and np.isclose(w[y == 0].sum(), 1.)
    head, audit = CategoryHead.fit(records, items, .01)
    assert len(audit['missing_classes']) == 3
    predicted = head.predict(items['normal']['features'])
    assert predicted.shape == (9, 4) and (predicted[:, 1:] == 0).all()
    head.save(tmp_path / 'head.json')
    np.testing.assert_allclose(predicted, CategoryHead.load(tmp_path / 'head.json').predict(items['normal']['features']))
    zero = np.ones_like(predicted) * .8
    blended = blend_scores(predicted, zero, .5, head.classes)
    np.testing.assert_allclose(blended[:, 1:], .8)
    records[1]['events'].append(event(CATEGORIES[1], start=4, end=4))
    _, y, _, audit = training_arrays(records, items)
    assert audit['ambiguous_windows_skipped'] == 5 and (y == 1).sum() == 0


def test_two_scan_requirement_missing_class_and_category_preservation():
    scores = np.zeros((7, 4))
    scores[0, 0] = .9  # isolated, excluded
    scores[2:4, :2] = .8
    scores[5:7, 0] = .7
    events = typed_events(np.arange(7), scores, .5)
    assert len(events) == 2 and {e['category_id'] for e in events} == set(CATEGORIES[:2])
    assert consecutive_scores(np.arange(7), scores)[0] == .8
    assert typed_events([0, 2], np.ones((2, 4)), .5) == []
    assert typed_events([0, 1], np.zeros((2, 4)), 0) == []


def test_prompt_bank_does_not_include_heldout_annotations():
    train = record('train', 1, [event()])
    heldout = record('secret_test_filename', 1, [dict(event(), description='UNIQUE TEST CAPTION')])
    bank = prompt_bank([train])
    text = json.dumps(bank['categories'])
    assert 'UNIQUE TEST CAPTION' not in text and heldout['recording_id'] not in text
    assert bank['source_training_ids'] == ['train']


def test_balanced_accuracy_selection_and_temporal_matching():
    records = [record('n', 0, []), record('p', 1, [event(start=53, end=57)])]
    choice = select_threshold(records, [[.4, 0, 0, 0], [.8, 0, 0, 0]])
    assert choice['metrics']['balanced_accuracy'] == 1
    output = {'n': {'events': []}, 'p': {'events': [
        {'category_id': CATEGORIES[0], 'timecode_sec': 56, 'score': .8},
        {'category_id': CATEGORIES[0], 'timecode_sec': 58, 'score': .7}]}}
    result = temporal_report(records, output, 2)
    assert result['matched'] == 1 and result['mean_matched_timing_error_sec'] == 0
    assert result['categories'][CATEGORIES[0]]['unmatched_predictions'] == 1
    assert event_distance(58, records[1]['events'][0]) == 1


def profile():
    return {'confirmed': True, 'workstation_roi': [.1, .1, .5, .9],
            'departure_categories': [CATEGORIES[0]], 'role_examples': [
                {'label': label, 'confirmed': True, 'video_sha256': 'hash', 'timestamp_sec': 0.}
                for label in ('cashier_present', 'cashier_departure')]}


def test_normal_departure_never_creates_violation_and_uncertainty_falls_back():
    p = profile()
    assert valid_profile(p) and not valid_profile(DEFAULT_PROFILE)
    inside = [.2, .2, .4, .8]
    # Gradual observed movement preserves IoU while crossing the ROI boundary.
    boxes = [inside] * 3 + [[.3,.2,.5,.8], [.4,.2,.6,.8], [.5,.2,.7,.8], [.5,.2,.7,.8], [.5,.2,.7,.8]]
    observations = [{'timestamp_sec': i, 'boxes': [box]} for i, box in enumerate(boxes)]
    result = departure_observations(observations, p)
    assert result['status'] == 'stable' and result['departures']
    assert assist_events([], result, p) == []
    uncertain = departure_observations([{'timestamp_sec': 0, 'boxes': [inside, inside]}], p)
    assert uncertain['status'] == 'disabled_ambiguous_role'
    candidates = [{'category_id': CATEGORIES[0], 'timecode_sec': 4, 'score': .8}]
    assert assist_events(candidates, uncertain, p) == candidates
    lost = departure_observations(observations[:3] + [{'timestamp_sec': 3, 'boxes': []}], p)
    assert lost['status'] == 'disabled_lost_cashier_track'


def test_legacy_artifact_dispatch_remains_compatible(monkeypatch):
    from datathon.inference import Detector
    from datathon.training import LinearHead
    monkeypatch.setattr('datathon.inference.XClipEncoder', lambda *a, **k: type('Encoder', (), {'config': None})())
    monkeypatch.setattr(LinearHead, 'load', lambda *a: object())
    loaded = Detector.from_artifacts(ROOT / 'artifacts/trained_model_provisional')
    assert isinstance(loaded, Detector)
