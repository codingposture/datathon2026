"""Typed workbook ingestion. Workbook cells are inert data, including formulas."""
from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

from .data import normalize_video_id, parse_timestamp, probe_video, save_json, sha256_file

CATEGORIES = ('CASH_OR_WORKPLACE_NOT_SECURED', 'ACCEPTANCE_STEP_SKIPPED',
              'CASH_AREA_ACCESS_UNCONTROLLED', 'PERSONAL_ITEMS_IN_CASH_AREA')
CLASSES = ('NONE',) + CATEGORIES
CERTAINTY = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.25}
TAXONOMY = {
    CATEGORIES[0]: {'name': 'Cash/workstation not secured',
        'positive': 'A cashier leaves cash, a key, open storage, or working equipment unsecured at the workstation.',
        'normal': 'A cashier puts cash away, removes the key, closes storage and secures the workstation before leaving.'},
    CATEGORIES[1]: {'name': 'Acceptance step skipped',
        'positive': 'A cashier accepts cash without checking identification, counting in front of the client or inspecting flagged notes.',
        'normal': 'A cashier checks identification, counts cash in front of the client and inspects flagged banknotes.'},
    CATEGORIES[2]: {'name': 'Cash-area access uncontrolled',
        'positive': 'Another person enters the cash area during a transaction or a cashier leaves the access door open.',
        'normal': 'Only the cashier is in the cash area during the transaction and the access door is closed.'},
    CATEGORIES[3]: {'name': 'Personal items in the cash area',
        'positive': 'A personal bag or another personal item is placed on the cash work table.',
        'normal': 'The cash work table is clear of bags and personal belongings.'}}


def workbook_sheets(path):
    """Read named OOXML sheets and named headers without evaluating any formula.

    Formula cells are retained as provenance dictionaries, never accepted as labels.
    Also retains values in columns without headers and the README as provenance.
    """
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in e.findall('.//s:t', ns))
                       for e in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        rels = {e.get('Id'): e.get('Target') for e in
                ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        result = {}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet', ns):
            target = rels[sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
            member = target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl') / target)
            if '..' in PurePosixPath(member).parts:
                raise ValueError('Unsafe workbook relationship')
            rows = []
            for row in ET.fromstring(archive.read(member)).findall('s:sheetData/s:row', ns):
                cells = {}
                for cell in row.findall('s:c', ns):
                    column = re.sub(r'\d+', '', cell.get('r'))
                    value = cell.findtext('s:v', default='', namespaces=ns)
                    if cell.get('t') == 's':
                        value = strings[int(value)]
                    elif cell.get('t') == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.findall('.//s:t', ns))
                    formula = cell.find('s:f', ns)
                    cells[column] = {'formula': formula.text, 'cached_value': value} if formula is not None else value
                rows.append((int(row.get('r')), cells))
            if not rows:
                result[sheet.get('name')] = []
                continue
            headers = rows[0][1]
            if any(isinstance(v, dict) for v in headers.values()):
                raise ValueError('Formula in workbook header')
            names = [str(v).strip() for v in headers.values() if str(v).strip()]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate workbook headers')
            result[sheet.get('name')] = [dict({'_excel_row': number}, **{
                str(headers.get(col) or f'_column_{col}').strip(): value for col, value in values.items()})
                for number, values in rows[1:]]
        return result


def read_typed_workbook(path):
    sheets = workbook_sheets(path)
    if not {'labels', 'types', 'sequences'}.issubset(sheets):
        raise ValueError('Workbook must contain labels, types and sequences sheets')
    required = {'video_id', 'label_binary', 'violation_sec', 'violation_type',
                'what_happened', 'normal_behavior', 'certainty'}
    rows, seen = [], set()
    for source in sheets['labels']:
        if not source.get('video_id'):
            continue
        if not required.issubset(source):
            raise ValueError(f'Missing label headers: {required - source.keys()}')
        if any(isinstance(source[k], dict) for k in required):
            raise ValueError('Formulas are not allowed in annotation fields')
        video_id = normalize_video_id(source['video_id'])
        if source.get('наименование файла') and normalize_video_id(source['наименование файла']) != video_id:
            raise ValueError(f'Conflicting filename identifiers: {video_id}')
        if video_id in seen:
            raise ValueError(f'Duplicate video row: {video_id}')
        seen.add(video_id)
        if source['label_binary'] not in ('0', '1', 0, 1):
            raise ValueError(f'Invalid binary label: {video_id}')
        label = int(source['label_binary'])
        russian_label = source.get('нарушение / без нарушения', '').strip().lower()
        if russian_label and russian_label != ('нарушение' if label else 'без нарушения'):
            raise ValueError(f'Conflicting binary labels: {video_id}')
        category = source['violation_type'].strip() or 'NONE'
        if category not in CLASSES or label != int(category != 'NONE'):
            raise ValueError(f'Invalid category/label: {video_id}')
        certainty = source['certainty'].strip().upper()
        if label and certainty not in CERTAINTY:
            raise ValueError(f'Unknown certainty: {video_id}')
        time = parse_timestamp(source['violation_sec'])
        russian = parse_timestamp(source.get('таймстемп'))
        if russian is not None and time is not None and abs(russian - time) > .01:
            raise ValueError(f'Conflicting timestamps: {video_id}')
        if label and time is None:
            raise ValueError(f'Positive primary annotation needs timing: {video_id}')
        if not label and time is not None:
            raise ValueError(f'Negative annotation has timing: {video_id}')
        row = {'video_id': video_id, 'label': label, 'category_id': category, 'time_sec': time,
               'certainty': certainty if label else 'HIGH', 'weight': CERTAINTY[certainty] if label else 1.0,
               'description': source['what_happened'], 'normal_contrast': source['normal_behavior'],
               'subtype': source.get('violation_subtype', ''), 'provenance': source}
        secondary = source.get('secondary_type', '').strip()
        row['secondary'] = None
        if secondary:
            category2, _, subtype = secondary.partition('/')
            if category2 not in CATEGORIES:
                raise ValueError(f'Unknown secondary type: {category2}')
            row['secondary'] = {'category_id': category2, 'subtype': subtype,
                'time_sec': parse_timestamp(source.get('secondary_event_sec')), 'certainty': row['certainty'],
                'weight': row['weight'], 'description': '', 'normal_contrast': '', 'source_id': video_id}
        rows.append(row)
    if not rows:
        raise ValueError('Empty label sheet')
    return rows, sheets


def event_distance(time, event):
    return max(event['start_sec'] - time, 0, time - event['end_sec'])


def build_typed_manifest(dataset_dir, workbook, previous_manifest=None, output=None):
    dataset_dir, workbook = Path(dataset_dir), Path(workbook)
    labels, sheets = read_typed_workbook(workbook)
    files = {}
    for p in (dataset_dir / 'videos').iterdir():
        if p.suffix.lower() == '.mov':
            key = normalize_video_id(p.name)
            if key in files:
                raise ValueError('Filename normalization collision')
            files[key] = p
    if set(files) != {r['video_id'] for r in labels}:
        raise ValueError('Workbook/video inventory mismatch')
    previous = {r['sha256']: r for r in (previous_manifest or {}).get('recordings', [])}
    records = {}
    for row in labels:
        path = files[row['video_id']]
        digest = sha256_file(path)
        if digest not in records:
            old = previous.get(digest)
            probe = ({k: old[k] for k in ('duration_sec', 'decode_status', 'width', 'height', 'codec')}
                     if old and old['decode_status'] == 'ok' else probe_video(path))
            records[digest] = {'recording_id': digest, 'sha256': digest,
                'relative_path': str(path.relative_to(dataset_dir)), 'aliases': [], 'events': [],
                'untimed_events': [], 'source_annotations': [], 'label': row['label'],
                'group_id': digest, 'annotation_status': 'provisional_workbook', **probe}
        record = records[digest]
        if record['label'] != row['label']:
            raise ValueError('Conflicting duplicate labels')
        record['aliases'].append(path.name)
        record['source_annotations'].append(row)
        if not row['label']:
            continue
        event = {k: row[k] for k in ('category_id', 'subtype', 'certainty', 'weight', 'description', 'normal_contrast')}
        event.update(start_sec=row['time_sec'], end_sec=row['time_sec'], source_ids=[row['video_id']])
        record['events'].append(event)
        if row['secondary']:
            event2 = deepcopy(row['secondary'])
            time = event2.pop('time_sec')
            event2['source_ids'] = [event2.pop('source_id')]
            if time is None:
                record['untimed_events'].append(event2)
            else:
                event2.update(start_sec=time, end_sec=time)
                record['events'].append(event2)
    for record in records.values():
        ids = {normalize_video_id(n) for n in record['aliases']}
        if {'X-4', 'X-47'}.issubset(ids):
            # Explicit user-approved reconciliation; retain every source row unchanged.
            cash = next(e.copy() for e in record['events'] if e['category_id'] == CATEGORIES[0])
            cash.update(start_sec=53.0, end_sec=57.0, source_ids=['X-4', 'X-47'],
                        timing_status='uncertain_interval', correction='Approved reconciliation: 53–57 seconds')
            count = next(e for e in record['events'] if e['category_id'] == CATEGORIES[1])
            record['events'] = [cash, count]
        # Collapse identical secondary/primary events but retain annotation provenance.
        unique = {}
        for event in record['events']:
            if not 0 <= event['start_sec'] <= event['end_sec'] < record['duration_sec']:
                raise ValueError(f'Annotation outside video: {record["aliases"]}')
            key = (event['category_id'], event['start_sec'], event['end_sec'])
            if key not in unique:
                unique[key] = event
            else:
                unique[key]['source_ids'] = sorted(set(unique[key]['source_ids'] + event['source_ids']))
        record['events'] = sorted(unique.values(), key=lambda e: e['start_sec'])
        record['event_times_sec'] = [(e['start_sec'] + e['end_sec']) / 2 for e in record['events']]
        if ids.intersection({'X-44', 'X-45'}):
            record['group_id'] = 'confirmed_scenario_X44_X45'
    result = {'schema_version': 2, 'source_workbook_sha256': sha256_file(workbook),
              'source_workbook_name': workbook.name, 'source_video_count': len(files),
              'sheet_provenance': sheets, 'recordings': sorted(records.values(), key=lambda r: r['relative_path']),
              'limitations': ['Annotations remain provisional; only exact duplicates and X44/X45 scene overlap confirmed.',
                             'Workbook rules and inferred normal contrasts are unverified metadata.']}
    if output:
        save_json(output, result)
    return result


def typed_fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def apply_scene_groups(records, review_csv):
    """Union subsequently confirmed CSV groups with the known scenario relationships.

    Reads group_id only; never replaces workbook event labels with the old binary CSV.
    Transitive unions preserve X44/X45 even if just one is linked to another recording.
    """
    path = Path(review_csv)
    if not path.exists():
        return records
    result = deepcopy(records)
    lookup = {r['recording_id']: r for r in result}
    parents = {key: key for key in lookup}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def union(a, b):
        ra, rb = root(a), root(b)
        parents[max(ra, rb)] = min(ra, rb)
    groups = {}
    for r in result:
        group = r['group_id']
        if group in groups:
            union(r['recording_id'], groups[group])
        groups[group] = r['recording_id']
    manual = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            group = row.get('group_id', '').strip()
            if not group:
                continue
            key = row['recording_id']
            if key not in lookup:
                raise ValueError('Scene group references an unknown recording')
            if group in manual:
                union(key, manual[group])
            manual[group] = key
            lookup[key]['scene_group_provenance'] = {'source': path.name, 'group_id': group}
    counts = {}
    for key in lookup:
        counts[root(key)] = counts.get(root(key), 0) + 1
    for key, r in lookup.items():
        if counts[root(key)] > 1:
            r['group_id'] = 'confirmed_scene_' + root(key)
    return result


def typed_split(records, old_split, output=None):
    from sklearn.model_selection import StratifiedGroupKFold
    import numpy as np
    lookup = {r['recording_id']: r for r in records}
    dev, test = old_split['development'], old_split['test']
    if set(dev) & set(test) or set(dev + test) != set(lookup):
        raise ValueError('Preserved partition does not match dataset')
    if {lookup[k]['group_id'] for k in dev} & {lookup[k]['group_id'] for k in test}:
        raise ValueError('Confirmed scene overlap across preserved test boundary; explicitly resolve before training')
    folds = []
    for train, val in StratifiedGroupKFold(4, shuffle=True, random_state=42).split(
            np.zeros(len(dev)), [lookup[k]['label'] for k in dev], [lookup[k]['group_id'] for k in dev]):
        train_ids, val_ids = [dev[i] for i in train], [dev[i] for i in val]
        available = {e['category_id'] for k in train_ids for e in lookup[k]['events']}
        folds.append({'train': train_ids, 'validation': val_ids,
                      'missing_training_categories': sorted(set(CATEGORIES) - available)})
    result = {'schema_version': 2, 'seed': 42, 'development': dev, 'test': test, 'folds': folds,
              'dataset_fingerprint': typed_fingerprint(records), 'test_status': 'reused_diagnostic_not_untouched',
              'scene_overlap': ['X44/X45 grouped together; additional confirmed CSV groups imported; remaining relationships unverified'],
              'source_partition_sha256': hashlib.sha256(json.dumps(old_split, sort_keys=True).encode()).hexdigest()}
    if output:
        save_json(output, result)
    return result
