"""Private, notebook-local playback and editable CSV annotation controls."""
import csv
import json
from pathlib import Path
import subprocess

from .data import REVIEW_COLUMNS


def review_widget(manifest, dataset_dir, review_path, preview_dir):
    import ipywidgets as widgets
    from IPython.display import Video, clear_output, display
    review_path, preview_dir = Path(review_path), Path(preview_dir)
    recordings = {r['recording_id']: r for r in manifest['recordings']}
    def read_rows():
        with review_path.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    rows = read_rows()
    selector = widgets.Dropdown(options=[(f"{i+1}/{len(rows)}: {r['aliases']}", r['recording_id']) for i, r in enumerate(rows)],
                                description='Recording:', layout=widgets.Layout(width='95%'))
    target = widgets.Dropdown(options=[('No violation', '0'), ('Possible violation', '1')], description='Label:')
    events = widgets.Text(description='Events (s):', layout=widgets.Layout(width='95%'))
    group = widgets.Text(description='Scene group:', layout=widgets.Layout(width='95%'))
    notes = widgets.Textarea(description='Notes:', layout=widgets.Layout(width='95%'))
    reviewed = widgets.Checkbox(value=False, description='I reviewed the full recording')
    save = widgets.Button(description='Save review', button_style='success')
    play = widgets.Button(description='Load video')
    status, player = widgets.Output(), widgets.Output()
    def populate(change=None):
        row = next(r for r in read_rows() if r['recording_id'] == selector.value)
        target.value = row['label']
        events.value = ', '.join(map(str, json.loads(row['event_times_sec'])))
        group.value = row['group_id']
        notes.value = row['notes']
        reviewed.value = row['reviewed'].lower() == 'true'
        with player:
            clear_output(wait=False)
        with status:
            clear_output(wait=False)
            source = recordings[selector.value]
            print(f"Duration: {source['duration_sec']:.2f}s; original annotations: {source['source_annotations']}")
    def save_row(button):
        with status:
            clear_output(wait=False)
            try:
                times = sorted(set(float(t.strip()) for t in events.value.split(',') if t.strip()))
                duration = recordings[selector.value]['duration_sec']
                if any(not 0 <= t < duration for t in times):
                    raise ValueError('Event timestamps must fall within the recording')
                if int(target.value) != int(bool(times)):
                    raise ValueError('A positive label needs event times; a negative label must have none')
                if reviewed.value and not group.value.strip():
                    raise ValueError('Set a scene group before marking a recording reviewed')
                updated = read_rows()
                for row in updated:
                    if row['recording_id'] == selector.value:
                        row.update(label=target.value, event_times_sec=json.dumps(times), group_id=group.value.strip(),
                                   reviewed=str(reviewed.value).lower(), notes=notes.value)
                temporary = review_path.with_suffix('.csv.tmp')
                with temporary.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
                    writer.writeheader()
                    writer.writerows(updated)
                temporary.replace(review_path)
                print(f"Saved. Reviewed: {sum(r['reviewed'] == 'true' for r in updated)}/{len(updated)}")
            except (ValueError, OSError) as exc:
                print(f'Not saved: {exc}')
    def play_row(button):
        with player:
            clear_output(wait=True)
            recording = recordings[selector.value]
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = preview_dir / f"{selector.value}.mp4"
            if not preview.exists():
                temporary = preview.with_suffix('.tmp.mp4')
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(Path(dataset_dir) / recording['relative_path']),
                                '-an', '-vf', 'scale=-2:640', '-c:v', 'libx264', '-crf', '28',
                                '-preset', 'veryfast', '-movflags', '+faststart', str(temporary)], check=True)
                temporary.replace(preview)
            display(Video(str(preview), embed=True, width=320))
    selector.observe(populate, names='value')
    save.on_click(save_row)
    play.on_click(play_row)
    populate()
    return widgets.VBox([selector, widgets.HBox([play, save]), target, events, group, notes, reviewed, status, player])
