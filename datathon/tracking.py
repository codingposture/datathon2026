"""Optional RT-DETR person observations. Unconfirmed cameras always fall back."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .data import save_json
from .video import requested_frames

DETECTOR_ID = 'PekingU/rtdetr_r18vd'
DETECTOR_REVISION = 'ac77a11ff0170a41b771c03264987f8ce2b0d753'
DEFAULT_PROFILE = {'schema_version': 1, 'profile_id': 'unconfigured', 'confirmed': False,
                   'workstation_roi': None, 'role_examples': [], 'departure_categories': [],
                   'note': 'Filtering disabled until ROI and cashier/departure examples are confirmed.'}


def valid_profile(profile):
    if not isinstance(profile, dict) or not profile.get('confirmed'):
        return False
    roi = profile.get('workstation_roi')
    if not isinstance(roi, list) or len(roi) != 4:
        return False
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in roi):
        return False
    if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        return False
    examples = profile.get('role_examples', [])
    return all(any(e.get('label') == label and e.get('confirmed') and e.get('video_sha256')
                   and isinstance(e.get('timestamp_sec'), (float, int)) and e['timestamp_sec'] >= 0
                   for e in examples) for label in ('cashier_present', 'cashier_departure'))


def load_profile(value, artifact_dir=None):
    if isinstance(value, dict):
        return value if valid_profile(value) else DEFAULT_PROFILE.copy()
    if value is not None:
        path = Path(value)
        if path.is_file():
            profile = json.loads(path.read_text())
            return profile if valid_profile(profile) else DEFAULT_PROFILE.copy()
        if artifact_dir:
            path = Path(artifact_dir) / 'camera_profiles.json'
            profiles = json.loads(path.read_text()) if path.exists() else {}
            profile = profiles.get(str(value), {})
            return profile if valid_profile(profile) else DEFAULT_PROFILE.copy()
    return DEFAULT_PROFILE.copy()


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def departure_observations(observations, profile):
    """Conservative IoU tracking; only an observed inside→outside stable track qualifies.

    Missing detections, competing matches and multiple ROI occupants invalidate role
    assignment. Disappearance alone cannot establish that the cashier departed.
    """
    if not valid_profile(profile):
        return {'status': 'disabled_unconfirmed_camera', 'departures': []}
    roi = profile['workstation_roi']
    tracks, next_id, role, departures = {}, 0, None, []
    inside_streak, outside_streak = 0, 0
    for observation in observations:
        time = observation['timestamp_sec']
        boxes = observation['boxes']
        current, matched = {}, set()
        for box in boxes:
            candidates = [(iou(box, t['box']), key) for key, t in tracks.items()
                          if time - t['time'] <= 1.5 and iou(box, t['box']) > .2]
            if len(candidates) > 1:
                return {'status': 'disabled_ambiguous_tracking', 'departures': []}
            if candidates:
                key = candidates[0][1]
                if key in matched:
                    return {'status': 'disabled_ambiguous_tracking', 'departures': []}
            else:
                key = next_id; next_id += 1
            matched.add(key)
            current[key] = {'box': box, 'time': time}
        def inside(box):
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]
        occupants = [k for k, t in current.items() if inside(t['box'])]
        if len(occupants) > 1:
            return {'status': 'disabled_ambiguous_role', 'departures': []}
        if role is None:
            if len(occupants) == 1:
                candidate = occupants[0]
                inside_streak = inside_streak + 1 if candidate in tracks else 1
                if inside_streak >= 3:
                    role = candidate
            else:
                inside_streak = 0
        elif role not in current:
            return {'status': 'disabled_lost_cashier_track', 'departures': []}
        elif inside(current[role]['box']):
            outside_streak = 0
        else:
            outside_streak += 1
            if outside_streak == 3:
                departures.append(float(time - 2))
        tracks = current
    return {'status': 'stable' if role is not None else 'disabled_no_confirmed_role', 'departures': departures}


def assist_events(events, observation_result, profile):
    """Only existing, explicitly departure-related candidates can be changed."""
    if not valid_profile(profile) or observation_result['status'] != 'stable':
        return events
    output = []
    for event in events:
        if event['category_id'] not in profile.get('departure_categories', []):
            output.append(event)
            continue
        nearby = [t for t in observation_result['departures'] if abs(t - event['timecode_sec']) <= 6]
        if nearby:
            best = min(nearby, key=lambda t: abs(t - event['timecode_sec']))
            output.append({**event, 'timecode_sec': best, 'departure_confirmed': True})
    return output


class PersonDetector:
    def __init__(self, source=DETECTOR_ID, local_only=False, device=None):
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        options = {'local_files_only': local_only}
        if not Path(source).is_dir():
            options['revision'] = DETECTOR_REVISION
        self.model = RTDetrForObjectDetection.from_pretrained(str(source), **options).eval().to(self.device)
        self.model.requires_grad_(False)
        self.processor = RTDetrImageProcessor.from_pretrained(str(source), **options)

    def observe(self, video_path, duration):
        observations = []
        torch = self.torch
        # One-second person scans on original full frames; X-CLIP uses its own cache.
        with torch.inference_mode():
            for image, time in requested_frames(video_path, np.arange(0, duration, 1.0)):
                inputs = self.processor(images=image, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                result = self.processor.post_process_object_detection(outputs, threshold=.6,
                            target_sizes=torch.tensor([[image.height, image.width]], device=self.device))[0]
                boxes = [box.cpu().numpy() / [image.width, image.height, image.width, image.height]
                         for box, label in zip(result['boxes'], result['labels']) if int(label) == 0]
                observations.append({'timestamp_sec': float(time), 'boxes': [b.tolist() for b in boxes]})
        return observations

    def save(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)


def camera_profile_widget(output_path, records, dataset_dir=None):
    """Local controls. The user must watch examples before confirming role labels."""
    import ipywidgets as w
    from IPython.display import display
    roi = [w.FloatSlider(value=v, min=0, max=1, step=.01, description=d)
           for v, d in zip((.1, .1, .9, .9), ('left', 'top', 'right', 'bottom'))]
    options = [('|'.join(r['aliases']), r['sha256']) for r in records]
    examples = []
    controls = []
    for label in ('cashier_present', 'cashier_departure'):
        video = w.Dropdown(options=options, description=label)
        time = w.FloatText(value=0, description='time (sec)')
        confirmed = w.Checkbox(value=False, description='Watched and confirmed')
        examples.append((label, video, time, confirmed))
        controls.extend([video, time, confirmed])
    overall = w.Checkbox(value=False, description='Confirm ROI and cashier role')
    departure = w.Checkbox(value=False, description='Security candidates in this camera require departure')
    button, status = w.Button(description='Save camera profile'), w.Output()
    preview, preview_output = w.Button(description='Preview ROI on example'), w.Output()
    def show(_):
        with preview_output:
            preview_output.clear_output(wait=True)
            if dataset_dir is None:
                print('Pass dataset_dir to preview an original frame.')
                return
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            _, video, time, _ = examples[0]
            record = next(r for r in records if r['sha256'] == video.value)
            if not 0 <= time.value < record['duration_sec']:
                print('Example timestamp must lie inside the video.')
                return
            image, actual = next(requested_frames(Path(dataset_dir) / record['relative_path'], [time.value]))
            fig, ax = plt.subplots(figsize=(5, 8))
            ax.imshow(image)
            left, top, right, bottom = [r.value for r in roi]
            ax.add_patch(Rectangle((left * image.width, top * image.height),
                         (right-left)*image.width, (bottom-top)*image.height, fill=False, edgecolor='red', linewidth=2))
            ax.set_title(f'Actual PTS: {actual:.3f}s'); ax.axis('off')
            plt.show()
    preview.on_click(show)
    def save(_):
        with status:
            status.clear_output()
            profile = {'schema_version': 1, 'profile_id': 'supplied_camera', 'confirmed': overall.value,
                       'workstation_roi': [r.value for r in roi],
                       'departure_categories': ['CASH_OR_WORKPLACE_NOT_SECURED'] if departure.value else [],
                       'role_examples': [{'label': label, 'video_sha256': video.value,
                                          'timestamp_sec': time.value, 'confirmed': checked.value}
                                         for label, video, time, checked in examples]}
            if not valid_profile(profile):
                print('Confirm a valid ROI and both watched role examples before saving.')
                return
            save_json(output_path, profile)
            print('Saved', output_path, 'Only load this profile for the confirmed camera.')
    button.on_click(save)
    return w.VBox(roi + controls + [preview, preview_output, overall, departure, button, status])
