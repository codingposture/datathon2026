import csv
import json
from pathlib import Path
import subprocess

import pytest

from datathon.data import (build_manifest, create_review_table, load_reviewed,
                           normalize_video_id, parse_timestamp, probe_video, read_labels)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('raw,expected', [('X-1.mov', 'X-1'), ('Х-047.MOV', 'X-47'), ('х–3', 'X-3'), (' x-2 ', 'X-2')])
def test_normalize(raw, expected):
    assert normalize_video_id(raw) == expected


@pytest.mark.parametrize('raw,expected', [('22 секунда', 22), ('1минута 8 секунда', 68), ('2 минуты 4 секунды', 124),
                                         ('01:08.5', 68.5), ('1,25', 1.25), (0, 0), ('', None), (None, None)])
def test_timestamps(raw, expected):
    assert parse_timestamp(raw) == expected


@pytest.mark.parametrize('raw', ['x', '-1', '1:70', float('nan'), float('inf')])
def test_bad_timestamps(raw):
    with pytest.raises(ValueError):
        parse_timestamp(raw)


def test_supplied_labels_and_audit():
    labels = read_labels(ROOT / 'open/open_labels.xlsx')
    assert len(labels) == 42
    assert sum(r['label'] for r in labels) == 22
    if (ROOT / 'review/manifest.json').exists():
        manifest = json.loads((ROOT / 'review/manifest.json').read_text())
        assert len(manifest['recordings']) == 40
        assert sum(r['label'] for r in manifest['recordings']) == 21
        conflicts = [r for r in manifest['recordings'] if len(r['source_annotations']) == 2 and r['label']]
        assert len(conflicts) == 1
        assert {a['timestamp_sec'] for a in conflicts[0]['source_annotations']} == {57, 68}
        assert all(r['decode_status'] == 'ok' for r in manifest['recordings'])


@pytest.fixture
def simple_manifest():
    return {'recordings': [{'recording_id': 'abc', 'label': 1, 'duration_sec': 10,
                            'aliases': ['X-1.mov'], 'source_annotations': [{'timestamp_sec': 5}]}]}


def test_review_gate_and_no_overwrite(simple_manifest, tmp_path):
    path = tmp_path / 'annotations.csv'
    create_review_table(simple_manifest, path)
    before = path.read_bytes()
    create_review_table(simple_manifest, path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Review incomplete'):
        load_reviewed(simple_manifest, path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(reviewed='true', group_id='scene_1')
    def write():
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write()
    assert load_reviewed(simple_manifest, path)[0]['event_times_sec'] == [5.0]
    for invalid in ('[10]', '[-1]', '[NaN]', '[true]', '"five"'):
        rows[0]['event_times_sec'] = invalid
        write()
        with pytest.raises(ValueError):
            load_reviewed(simple_manifest, path)
    rows[0]['event_times_sec'] = '[6]'
    write()
    with pytest.raises(ValueError, match='corrections'):
        load_reviewed(simple_manifest, path)
    rows[0]['notes'] = 'Corrected event after full review'
    write()
    assert load_reviewed(simple_manifest, path)[0]['event_times_sec'] == [6.0]


def test_bad_video_fails(tmp_path):
    path = tmp_path / 'corrupt.mov'
    path.write_bytes(b'not a video')
    with pytest.raises(ValueError, match='ffprobe failed'):
        probe_video(path)


def test_provisional_mode_never_marks_review_complete(simple_manifest, tmp_path):
    from datathon.data import provisional_records
    simple_manifest['recordings'][0]['sha256'] = 'abc'
    path = tmp_path / 'annotations.csv'
    create_review_table(simple_manifest, path)
    before = path.read_bytes()
    records = provisional_records(simple_manifest)
    assert records[0]['annotation_status'] == 'provisional_unreviewed'
    assert records[0]['group_id'] == 'exact_duplicate_abc'
    assert records[0]['event_times_sec'] == [5.0]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='Review incomplete'):
        load_reviewed(simple_manifest, path)
