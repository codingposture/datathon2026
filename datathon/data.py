"""Dataset audit and human review. Importable without ML dependencies."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 1


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_video_id(value):
    stem = Path(unicodedata.normalize('NFKC', str(value)).strip()).stem
    match = re.fullmatch(r'[xх]\s*[-–—]\s*0*(\d+)', stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f'Unsupported video identifier: {value!r}')
    return f'X-{int(match.group(1))}'


def parse_timestamp(value):
    """Parse supplied Russian seconds/minutes, mm:ss, or numeric seconds."""
    if value is None or str(value).strip() == '':
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().lower().replace(',', '.')
        if re.fullmatch(r'\d+(?:\.\d+)?', text):
            result = float(text)
        elif re.fullmatch(r'\d+:\d{1,2}(?:\.\d+)?', text):
            minute, second = map(float, text.split(':'))
            if second >= 60:
                raise ValueError(f'Invalid timestamp: {value!r}')
            result = minute * 60 + second
        else:
            match = re.fullmatch(
                r'\s*(?:(\d+(?:\.\d+)?)\s*(?:минут[аы]?|мин\.?|minutes?|m)\s*)?'
                r'(?:(\d+(?:\.\d+)?)\s*(?:секунд[аы]?|сек\.?|seconds?|s)\s*)?', text)
            if not match or not any(match.groups()):
                raise ValueError(f'Invalid timestamp: {value!r}')
            result = float(match.group(1) or 0) * 60 + float(match.group(2) or 0)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'Invalid timestamp: {value!r}')
    return result


def read_labels(path):
    """Read the first XLSX sheet without requiring pandas/openpyxl."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//s:t', ns))
                       for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        root = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        labels = []
        seen = set()
        for row in root.findall('.//s:sheetData/s:row', ns)[1:]:
            cells = {}
            for cell in row.findall('s:c', ns):
                node = cell.find('s:v', ns)
                value = node.text if node is not None else ''
                if cell.get('t') == 's':
                    value = strings[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//s:t', ns))
                cells[re.sub(r'\d+', '', cell.get('r'))] = value
            if not cells.get('A'):
                continue
            video_id = normalize_video_id(cells['A'])
            if video_id in seen:
                raise ValueError(f'Duplicate label identifier: {video_id}')
            seen.add(video_id)
            raw_label = cells.get('C', '').strip().lower()
            if raw_label not in {'нарушение', 'без нарушения'}:
                raise ValueError(f'Unknown label for {video_id}: {raw_label!r}')
            timestamp = parse_timestamp(cells.get('B'))
            target = int(raw_label == 'нарушение')
            if target != int(timestamp is not None):
                raise ValueError(f'Label/timestamp inconsistency: {video_id}')
            labels.append({'video_id': video_id, 'original_label_id': cells['A'],
                           'label': target, 'timestamp_sec': timestamp})
    if not labels:
        raise ValueError('No labels found')
    return labels


def probe_video(path, full_decode=True):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries',
         'format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,duration',
         '-of', 'json', str(path)], capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f'ffprobe failed for {Path(path).name}: {result.stderr[-1000:]}')
    metadata = json.loads(result.stdout)
    streams = [s for s in metadata.get('streams', []) if s['codec_type'] == 'video']
    if len(streams) != 1:
        raise ValueError(f'Expected one video stream in {path}')
    video = streams[0]
    duration = float(video.get('duration', metadata.get('format', {}).get('duration', 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid duration: {path}')
    if full_decode:
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                                  '-map', '0:v:0', '-an', '-f', 'null', '-'],
                                 capture_output=True, text=True, check=False)
        if decoded.returncode:
            raise ValueError(f'Decoding failed for {Path(path).name}: {decoded.stderr[-1000:]}')
    return {'duration_sec': duration, 'width': video['width'], 'height': video['height'],
            'codec': video['codec_name'], 'fps': video.get('avg_frame_rate'),
            'audio_streams': sum(s['codec_type'] == 'audio' for s in metadata['streams']),
            'decode_status': 'ok' if full_decode else 'not_checked'}


def build_manifest(dataset_dir, output_path=None):
    dataset_dir = Path(dataset_dir)
    labels = read_labels(dataset_dir / 'open_labels.xlsx')
    video_files = {}
    for path in (dataset_dir / 'videos').iterdir():
        if path.suffix.lower() != '.mov':
            continue
        key = normalize_video_id(path.name)
        if key in video_files:
            raise ValueError(f'Filename normalization collision: {key}')
        video_files[key] = path
    expected = {r['video_id'] for r in labels}
    if expected != set(video_files):
        raise ValueError(f'Video/label mismatch: missing={expected-set(video_files)}, extra={set(video_files)-expected}')
    recordings = {}
    for label in labels:
        path = video_files[label['video_id']]
        digest = sha256_file(path)
        if digest not in recordings:
            recordings[digest] = {'recording_id': digest, 'sha256': digest,
                                  'relative_path': str(path.relative_to(dataset_dir)),
                                  'size_bytes': path.stat().st_size, 'aliases': [],
                                  'source_annotations': [], 'label': label['label'],
                                  **probe_video(path)}
        recording = recordings[digest]
        if recording['label'] != label['label']:
            raise ValueError(f'Conflicting binary labels for identical footage: {path}')
        timestamp = label['timestamp_sec']
        if timestamp is not None and not 0 <= timestamp < recording['duration_sec']:
            raise ValueError(f'Timestamp outside video: {path.name}, {timestamp}')
        recording['aliases'].append(path.name)
        recording['source_annotations'].append(label)
    result = {'schema_version': SCHEMA_VERSION, 'source_label_sha256': sha256_file(dataset_dir / 'open_labels.xlsx'),
              'source_video_count': len(labels), 'recordings': sorted(recordings.values(), key=lambda r: r['relative_path'])}
    if output_path:
        save_json(output_path, result)
    return result


REVIEW_COLUMNS = ['recording_id', 'aliases', 'label', 'event_times_sec', 'group_id', 'reviewed', 'notes']


def create_review_table(manifest, path):
    """Create once; never overwrite a user's review."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for recording in manifest['recordings']:
            events = sorted({a['timestamp_sec'] for a in recording['source_annotations'] if a['timestamp_sec'] is not None})
            writer.writerow({'recording_id': recording['recording_id'], 'aliases': '|'.join(recording['aliases']),
                             'label': recording['label'], 'event_times_sec': json.dumps(events),
                             'group_id': '', 'reviewed': 'false',
                             'notes': 'Review duplicate event timestamps' if len(events) > 1 else ''})


def load_reviewed(manifest, review_path):
    with Path(review_path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    lookup = {r['recording_id']: r for r in manifest['recordings']}
    if len(rows) != len(lookup) or {r['recording_id'] for r in rows} != set(lookup):
        raise ValueError('Review table must contain exactly one row per unique recording')
    result = []
    for row in rows:
        source = lookup[row['recording_id']]
        if row['reviewed'].strip().lower() != 'true' or not row['group_id'].strip():
            raise ValueError(f"Review incomplete: {row['aliases']}. Set group_id and reviewed=true after playback.")
        if row['label'].strip() not in {'0', '1'}:
            raise ValueError('Reviewed label must be 0 or 1')
        events = json.loads(row['event_times_sec'])
        if not isinstance(events, list) or any(isinstance(t, bool) or not isinstance(t, (float, int))
                or not math.isfinite(t) or not 0 <= t < source['duration_sec'] for t in events):
            raise ValueError(f"Invalid reviewed timestamps: {row['aliases']}")
        label = int(row['label'])
        if label != int(bool(events)):
            raise ValueError(f"Label must agree with reviewed events: {row['aliases']}")
        if (label != source['label'] or sorted(set(events)) != sorted({a['timestamp_sec'] for a in source['source_annotations']
                if a['timestamp_sec'] is not None})) and not row['notes'].strip():
            raise ValueError('Document annotation corrections in notes')
        result.append({**source, 'label': label, 'event_times_sec': sorted(set(map(float, events))),
                       'group_id': row['group_id'].strip(), 'review_notes': row['notes']})
    return result


def fingerprint(records):
    fields = [{k: r[k] for k in ('recording_id', 'label', 'event_times_sec', 'group_id')} for r in records]
    return hashlib.sha256(json.dumps(sorted(fields, key=lambda x: x['recording_id']), sort_keys=True).encode()).hexdigest()


def provisional_records(manifest):
    """Explicit opt-in to unreviewed spreadsheet labels; never marks review rows complete.

    Only exact duplicate footage is grouped. Scene/session independence is unknown.
    Multiple source timestamps on identical footage are retained as candidate events.
    """
    return [{**recording,
             'event_times_sec': sorted({float(a['timestamp_sec']) for a in recording['source_annotations']
                                       if a['timestamp_sec'] is not None}),
             'group_id': 'exact_duplicate_' + recording['sha256'],
             'annotation_status': 'provisional_unreviewed',
             'review_notes': 'Unreviewed spreadsheet labels; only exact duplicates grouped; scene independence unverified.'}
            for recording in manifest['recordings']]
