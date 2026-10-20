"""Recording and one-to-one typed temporal metrics with interval-aware matching."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

from .hybrid_data import CATEGORIES, event_distance
from .hybrid import video_metrics


def temporal_report(records, outputs, tolerance):
    results, all_errors = {}, []
    for category in CATEGORIES:
        support, predicted, matched, errors = 0, 0, 0, []
        for record in records:
            targets = [e for e in record['events'] if e['category_id'] == category]
            events = [e for e in outputs[record['recording_id']]['events'] if e['category_id'] == category]
            support += len(targets); predicted += len(events)
            if targets and events:
                distances = np.asarray([[event_distance(p['timecode_sec'], t) for p in events] for t in targets])
                # Large penalty ensures maximum cardinality before minimum timing error.
                row, col = linear_sum_assignment(np.where(distances <= tolerance, distances, 1e6))
                valid = distances[row, col] <= tolerance
                errors.extend(distances[row[valid], col[valid]].tolist())
                matched += int(valid.sum())
        all_errors.extend(errors)
        results[category] = {'support': support, 'predicted': predicted, 'matched': matched,
                             'recall': matched / support if support else None,
                             'unmatched_predictions': predicted - matched,
                             'mean_matched_timing_error_sec': float(np.mean(errors)) if errors else None}
    support = sum(v['support'] for v in results.values())
    matched = sum(v['matched'] for v in results.values())
    return {'tolerance_sec': tolerance, 'support': support, 'matched': matched,
            'recall': matched / support if support else None, 'categories': results,
            'mean_matched_timing_error_sec': float(np.mean(all_errors)) if all_errors else None,
            'median_matched_timing_error_sec': float(np.median(all_errors)) if all_errors else None}


def report(records, outputs, threshold, *, diagnostic=False):
    # Evaluate actual retained events, including the two-scan requirement and optional filter.
    category_scores = [[max([e['score'] for e in outputs[r['recording_id']]['events']
                                if e['category_id'] == k] or [0.0]) for k in CATEGORIES] for r in records]
    metrics = video_metrics(records, category_scores, threshold)
    metrics['pr_auc_average_precision'] = float(average_precision_score(
        [r['label'] for r in records], [outputs[r['recording_id']]['score'] for r in records]))
    negative = [r for r in records if not r['label']]
    false_events = sum(len(outputs[r['recording_id']]['events']) for r in negative)
    rows = []
    for r in records:
        output = outputs[r['recording_id']]
        truth = sorted({e['category_id'] for e in r['events']})
        predicted = sorted({e['category_id'] for e in output['events']})
        rows.append({'recording_id': r['recording_id'], 'aliases': r['aliases'], 'label': r['label'],
                     'predicted_label': int(bool(predicted)), 'score': output['score'],
                     'true_categories': truth, 'predicted_categories': predicted,
                     'binary_correct': r['label'] == int(bool(predicted)), 'categories_correct': truth == predicted})
    high = [r for r in records if not r['label'] or all(e['certainty'] == 'HIGH' for e in r['events'])]
    high_scores = [category_scores[i] for i, r in enumerate(records) if r in high]
    return {'status': 'experimental', 'test_status': 'reused_diagnostic_not_untouched' if diagnostic else 'development_oof',
            'model': metrics, 'predictions': rows,
            'mistakes': [r for r in rows if not r['binary_correct'] or not r['categories_correct']],
            'higher_confidence_subset': {'definition': 'Original negatives plus videos whose timed annotations are all HIGH',
                'recordings': len(high), 'model': video_metrics(high, high_scores, threshold) if high else None},
            'events_2_sec': temporal_report(records, outputs, 2),
            'events_5_sec': temporal_report(records, outputs, 5),
            'false_events_on_negative_videos': false_events,
            'negative_video_count': len(negative),
            'false_events_per_negative_video': false_events / len(negative) if negative else None,
            'baselines': {'constant_normal': video_metrics(records, np.zeros((len(records), 4)), .5),
                          'constant_all_categories': video_metrics(records, np.ones((len(records), 4)), .5)},
            'limitations': ['Unreviewed workbook labels; unknown additional scene overlap.',
                'Diagnostic temporal tolerances are not official jury tolerances.',
                'Untimed secondary events excluded from training targets and evaluation.',
                'Zero-shot contrast scores and logistic scores are uncalibrated.',
                'No workstation-security positives in reused diagnostic test; its recall is unavailable.'
                if diagnostic else 'Thresholds/components selected on these OOF predictions; these are selection results, not independent performance.']}


def markdown_report(selection, development, evaluation, detector_experiment):
    lines = ['# Hybrid v2 — experimental', '',
             'Four broad visual categories; no legal conclusions or subtype predictions.', '',
             f"Selected: {selection['window_sec']}s context, C={selection['selected_C']}, "
             f"supervised weight={selection['supervised_weight']}, threshold={selection['threshold']:.6f}.",
             f"Person filtering selected: {selection.get('detector_enabled', False)}.", '',
             '| Metric | Development OOF (selected) | Reused diagnostic test |',
             '|---|---:|---:|']
    for name in ('balanced_accuracy', 'precision', 'recall', 'f1', 'f2', 'category_macro_f1'):
        lines.append(f"| {name} | {development['model'][name]:.4f} | {evaluation['model'][name]:.4f} |")
    lines += ['', 'The eight-video test was reused from the previous experiment. It has no cash/workstation-security positives; '
              'that category’s test recall is unavailable. One personal-items development recording makes fold estimates unstable.',
              '', '## Test category support', '', '| Category | Positive videos | Recall |', '|---|---:|---:|']
    for k, row in evaluation['model']['category_results'].items():
        lines.append(f"| {k} | {row['positive_support']} | {row['recall']} |")
    lines += ['', f"Detector experiment: {detector_experiment['status']}.", '',
              'No readiness or minimum-accuracy claim is made. Inspect the JSON reports for mistakes, confusion matrices, '
              'constant baselines, high-confidence subset results and temporal matching. '
              'Cash/key/storage states are not detected by RT-DETR; it detects people only.']
    return '\n'.join(lines) + '\n'
