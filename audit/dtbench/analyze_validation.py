"""Compare live Inspect accuracy with the raw CRI score and item uncertainty."""
import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log

NAMES = {'openai/gpt-4o-mini': 'GPT-4o mini', 'openai/gpt-4.1-nano': 'GPT-4.1 nano',
         'openai/gpt-4.1-mini': 'GPT-4.1 mini'}
RATES = {'openai/gpt-4o-mini': (0.15, 0.6), 'openai/gpt-4.1-nano': (0.1, 0.4),
         'openai/gpt-4.1-mini': (0.4, 1.6)}


def analyze(root):
    published = {r['model']: r for r in csv.DictReader((root / 'cri-scores.csv').open())}
    reports = []
    for model, name in NAMES.items():
        summary_file = root / (model.split('/')[-1] + '.json')
        if not summary_file.exists():
            continue
        summary = json.loads(summary_file.read_text())
        log = read_eval_log(summary['log'])
        values, trials, stops = collections.defaultdict(list), collections.Counter(), collections.Counter()
        errors = invalid = disagreements = 0
        from bench.task.dtbench import grade_completion
        for s in log.samples:
            trials[str(s.id)] += 1
            if s.error:
                errors += 1
                continue
            stops[s.output.stop_reason] += 1
            score = s.scores['dtbench_score'].value
            fresh = grade_completion(s.output.completion, s.choices, list(s.target))
            if not (score == float(fresh) if fresh is not None else math.isnan(score)):
                disagreements += 1
            if math.isfinite(score):
                values[str(s.id)].append(float(score))
            else:
                invalid += 1
        means = [statistics.mean(v) for v in values.values()]
        accuracy = statistics.mean(means)
        # Item-sampling uncertainty, matching the published CI's interpretation.
        se = statistics.stdev(means) / math.sqrt(len(means))
        row = published[name]
        target = float(row['dtbench_accuracy'])
        halfwidth = float(row['dtbench_ci']) * 0.6 / 100
        combined_halfwidth = math.sqrt((1.96 * se)**2 + halfwidth**2)
        # Repeat noise conditional on the fixed item population is separate.
        repeat_se = math.sqrt(sum(statistics.variance(v) / len(v) for v in values.values() if len(v) > 1)) / len(means)
        usage = next(iter(log.stats.model_usage.values()))
        rates = RATES[model]
        upper_cost = (usage.input_tokens * rates[0] + usage.output_tokens * rates[1]) / 1e6
        reports.append({'model': model, 'status': log.status, 'items': len(values),
                        'attempts': len(log.samples), 'trial_counts': dict(collections.Counter(trials.values())),
                        'errors': errors, 'invalid': invalid, 'stop_reasons': dict(stops),
                        'regrading_disagreements': disagreements, 'accuracy': accuracy,
                        'published_accuracy': target, 'difference': accuracy-target,
                        'item_sampling_95_halfwidth': 1.96*se,
                        'fixed_item_repeat_95_halfwidth': 1.96*repeat_se,
                        'published_95_halfwidth_raw': halfwidth,
                        'combined_difference_95_halfwidth': combined_halfwidth,
                        'within_combined_sampling_uncertainty': abs(accuracy-target) <= combined_halfwidth,
                        'cost_upper_bound_usd': upper_cost})
    result = {'models': reports,
              'caveat': 'Public CRI does not identify its evaluated item subset or publish current per-item traces. Aggregate agreement is a consistency check, not proof of identical population or protocol.',
              'cost_upper_bound_usd': sum(r['cost_upper_bound_usd'] for r in reports)}
    (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory)
