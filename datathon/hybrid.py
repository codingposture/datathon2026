"""Frozen X-CLIP branches, typed event decisions and development-only selection."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, fbeta_score, average_precision_score)

from .data import save_json
from .hybrid_data import CATEGORIES, CLASSES, TAXONOMY, event_distance
from .video import XClipEncoder, cache_identity


def prompt_bank(training_records):
    """Only caller-supplied training rows may contribute text. No IDs/times in text."""
    bank = {k: {'positive': [TAXONOMY[k]['positive']], 'normal': [TAXONOMY[k]['normal']]}
            for k in CATEGORIES}
    for record in training_records:
        for event in record['events'] + record.get('untimed_events', []):
            k = event['category_id']
            for field, target in (('description', 'positive'), ('normal_contrast', 'normal')):
                value = event.get(field, '')
                if value and value.strip() not in bank[k][target]:
                    bank[k][target].append(value.strip())
            # Untimed secondary types supply vocabulary without creating targets.
            if not event.get('description') and event.get('subtype'):
                text = 'A video showing ' + event['subtype'].replace('_', ' ').lower() + '.'
                if text not in bank[k]['positive']:
                    bank[k]['positive'].append(text)
    return {'categories': bank, 'source_training_ids': sorted(r['recording_id'] for r in training_records),
            'method': 'full_xclip_video_specific_prompting_mean_logit_contrast',
            'template_version': 1}


class PromptedEncoder(XClipEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from transformers import CLIPTokenizerFast
        options = {'local_files_only': kwargs.get('local_only', False)}
        source = args[0] if args else kwargs.get('source', self.model.name_or_path)
        if not Path(source).is_dir():
            options['revision'] = self.source_revision
        self.tokenizer = CLIPTokenizerFast.from_pretrained(str(source), **options)

    def zero_shot(self, video, centers, bank):
        texts, spans = [], []
        for k in CATEGORIES:
            sides = []
            for side in ('positive', 'normal'):
                start = len(texts)
                texts.extend(bank['categories'][k][side])
                sides.append((start, len(texts)))
            spans.append(sides)
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        torch = self.torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(centers), self.config.batch_size):
                clips = video.clips(centers[start:start + self.config.batch_size])
                pixels = self.processor([list(clip) for clip in clips], return_tensors='pt')['pixel_values'].to(self.device)
                context = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
                with context:
                    # Full forward is essential: includes the video-specific prompt generator.
                    logits = self.model(pixel_values=pixels, **tokens).logits_per_video.float().cpu().numpy()
                batches.append(np.stack([expit(logits[:, a:b].mean(1) - logits[:, c:d].mean(1))
                                         for (a, b), (c, d) in spans], axis=1))
        return np.concatenate(batches).astype(np.float32)

    def save(self, path):
        super().save(path)
        self.tokenizer.save_pretrained(path)


def cached_zero_shot(record, dataset_dir, cache_dir, encoder, bank):
    from .video import SampledVideo
    identity = {**cache_identity(record, encoder), 'prompt_bank': bank}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / (key + '.npz')
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as item:
                centers, scores = item['centers'], item['scores']
                if (json.loads(str(item['identity'])) == identity and scores.shape == (len(centers), 4)
                        and len(centers) and np.isfinite(scores).all() and np.isfinite(centers).all()
                        and (np.diff(centers) > 0).all() and ((scores >= 0) & (scores <= 1)).all()):
                    return {'centers': centers.copy(), 'scores': scores.copy()}
        except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile):
            pass
    video = SampledVideo(Path(dataset_dir) / record['relative_path'], encoder.config)
    centers = video.centers()
    scores = encoder.zero_shot(video, centers, bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, identity=json.dumps(identity, sort_keys=True), centers=centers, scores=scores)
    tmp.replace(path)
    return {'centers': centers, 'scores': scores}


def training_arrays(records, features, context_sec=4):
    xs, ys, ws, audit = [], [], [], {'ambiguous_windows_skipped': 0, 'recordings': []}
    for record in records:
        item = features[record['recording_id']]
        targets, positions, certainty = [], [], []
        for j, time in enumerate(item['centers']):
            events = [e for e in record['events'] if event_distance(float(time), e) <= 1]
            types = {e['category_id'] for e in events}
            context_types = {e['category_id'] for e in record['events']
                             if event_distance(float(time), e) <= context_sec / 2}
            if not record['label']:
                positions.append(j); targets.append(0); certainty.append(1.0)
            elif types and len(context_types) > 1:
                audit['ambiguous_windows_skipped'] += 1
            elif len(types) == 1:
                positions.append(j); targets.append(CLASSES.index(next(iter(types))))
                certainty.append(max(e['weight'] for e in events))
            elif len(types) > 1:
                audit['ambiguous_windows_skipped'] += 1
        if positions:
            xs.append(item['features'][positions]); ys.extend(targets)
            # Equal base total per recording; certainty discounts remain effective.
            ws.extend(np.asarray(certainty) / len(positions))
        audit['recordings'].append({'recording_id': record['recording_id'], 'windows': len(positions)})
    if not xs:
        raise ValueError('No unambiguous timed training windows')
    return np.concatenate(xs), np.asarray(ys), np.asarray(ws), audit


class CategoryHead:
    def __init__(self, mean, scale, coef, intercept, classes, c):
        self.mean, self.scale, self.coef, self.intercept = map(np.asarray, (mean, scale, coef, intercept))
        self.classes, self.c = np.asarray(classes, dtype=int), c

    @classmethod
    def fit(cls, records, features, c, context_sec=4):
        x, y, weights, audit = training_arrays(records, features, context_sec)
        if len(np.unique(y)) < 2:
            raise ValueError('Need at least two supervised classes')
        scaler = StandardScaler().fit(x, sample_weight=weights)
        # Keep average sample weight at one, preserving recording/certainty ratios.
        weights = weights * len(weights) / weights.sum()
        fit = LogisticRegression(C=c, solver='lbfgs', max_iter=3000, random_state=42).fit(
            scaler.transform(x), y, sample_weight=weights)
        head = cls(scaler.mean_, scaler.scale_, fit.coef_, fit.intercept_, fit.classes_, c)
        audit['missing_classes'] = [CLASSES[i] for i in range(5) if i not in head.classes]
        return head, audit

    def predict(self, features):
        logits = ((features - self.mean) / self.scale) @ self.coef.T + self.intercept
        probabilities = np.stack([1 - expit(logits[:, 0]), expit(logits[:, 0])], 1) \
            if len(self.classes) == 2 else softmax(logits, axis=1)
        output = np.zeros((len(features), 5), dtype=np.float64)
        output[:, self.classes] = probabilities
        return output[:, 1:]

    def save(self, path):
        save_json(path, {'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                        'coef': self.coef.tolist(), 'intercept': self.intercept.tolist(),
                        'classes': self.classes.tolist(), 'c': self.c, 'schema_version': 2})

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(*(d[k] for k in ('mean', 'scale', 'coef', 'intercept', 'classes', 'c')))


def blend_scores(supervised, zero_shot, weight, available_classes):
    result = weight * supervised + (1 - weight) * zero_shot
    if 0 < weight < 1:
        for j in range(4):
            if j + 1 not in available_classes:
                result[:, j] = zero_shot[:, j]
    return result


def consecutive_scores(centers, scores, stride=1.0):
    """Maximum supported score per class, requiring adjacent one-second scans."""
    if len(centers) < 2:
        return np.zeros(4)
    paired = np.minimum(scores[:-1], scores[1:]).copy()
    paired[~np.isclose(np.diff(centers), stride, atol=1e-4)] = 0
    return paired.max(0)


def typed_events(centers, scores, threshold, merge_sec=4.0, stride=1.0):
    events = []
    for j, category in enumerate(CATEGORIES):
        positive = np.flatnonzero((scores[:, j] >= threshold) & (scores[:, j] > 0))
        if not len(positive):
            continue
        boundaries = (np.diff(positive) > 1) | ~np.isclose(np.diff(np.asarray(centers)[positive]), stride, atol=1e-4)
        for run in np.split(positive, np.flatnonzero(boundaries) + 1):
            if len(run) < 2:
                continue
            peak = run[np.argmax(scores[run, j])]
            events.append({'category_id': category, 'timecode_sec': float(centers[peak]), 'score': float(scores[peak, j])})
    return merge_typed_events(events, merge_sec)


def merge_typed_events(events, separation=4.0):
    kept = []
    for event in sorted(events, key=lambda e: (-e['score'], e['timecode_sec'])):
        if all(e['category_id'] != event['category_id'] or
               abs(e['timecode_sec'] - event['timecode_sec']) >= separation for e in kept):
            kept.append(event)
    return sorted(kept, key=lambda e: (e['timecode_sec'], e['category_id']))


def video_metrics(records, category_scores, threshold):
    target = np.asarray([[any(e['category_id'] == k for e in r['events']) for k in CATEGORIES] for r in records], dtype=int)
    predicted = (np.asarray(category_scores) >= threshold) & (np.asarray(category_scores) > 0)
    y, p = np.asarray([r['label'] for r in records]), predicted.any(1).astype(int)
    return {'balanced_accuracy': float(balanced_accuracy_score(y, p)),
            'precision': float(precision_score(y, p, zero_division=0)),
            'recall': float(recall_score(y, p, zero_division=0)),
            'f1': float(f1_score(y, p, zero_division=0)), 'f2': float(fbeta_score(y, p, beta=2, zero_division=0)),
            'pr_auc_average_precision': float(average_precision_score(y, np.asarray(category_scores).max(1))),
            'confusion_matrix_labels_0_1': confusion_matrix(y, p, labels=[0, 1]).tolist(),
            'category_macro_f1': float(f1_score(target, predicted, average='macro', zero_division=0)),
            'false_positive_videos': int(((p == 1) & (y == 0)).sum()),
            'category_results': {k: {'positive_support': int(target[:, j].sum()),
                'precision': float(precision_score(target[:, j], predicted[:, j], zero_division=0)),
                'recall': float(recall_score(target[:, j], predicted[:, j], zero_division=0)) if target[:, j].any() else None,
                'f1': float(f1_score(target[:, j], predicted[:, j], zero_division=0)),
                'confusion_matrix_labels_0_1': confusion_matrix(target[:, j], predicted[:, j], labels=[0, 1]).tolist()}
                for j, k in enumerate(CATEGORIES)}}


def metric_rank(metrics):
    return (metrics['balanced_accuracy'], metrics['category_macro_f1'], -metrics['false_positive_videos'])


def select_threshold(records, category_scores):
    scores = np.asarray(category_scores)
    # Threshold candidates cover every distinct video decision, plus constant-negative.
    thresholds = np.unique(np.r_[.0, scores.ravel(), np.nextafter(float(scores.max()), np.inf)])
    predictions = (scores[None, :, :] >= thresholds[:, None, None]) & (scores[None, :, :] > 0)
    targets = np.asarray([[any(e['category_id'] == k for e in r['events']) for k in CATEGORIES] for r in records])
    y = np.asarray([r['label'] for r in records], dtype=bool)
    binary = predictions.any(2)
    ba = (binary[:, y].mean(1) + (~binary[:, ~y]).mean(1)) / 2
    tp = (predictions & targets).sum(1)
    fp = (predictions & ~targets).sum(1)
    fn = (~predictions & targets).sum(1)
    denominator = 2 * tp + fp + fn
    macro = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0).mean(1)
    false_positives = binary[:, ~y].sum(1)
    index = max(range(len(thresholds)), key=lambda i: (ba[i], macro[i], -false_positives[i]))
    threshold = float(thresholds[index])
    return {'threshold': threshold, 'metrics': video_metrics(records, scores, threshold)}
