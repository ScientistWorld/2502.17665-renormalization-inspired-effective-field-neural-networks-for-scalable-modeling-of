#!/usr/bin/env python3
"""Report actual test metrics and explicit cohort-dependent normalized ranks."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
EXPERIMENT = 'three_spin_binary'


def main():
    spec = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments'][EXPERIMENT]
    metrics = spec['metrics']
    scores = {}
    statuses = {}
    for status_path in sorted(AUDIT.glob('*/status.json')):
        name = status_path.parent.name
        statuses[name] = json.loads(status_path.read_text())
        score_path = status_path.parent / 'scores_test.json'
        if statuses[name].get('test_exit') == 0 and score_path.exists():
            scores[name] = json.loads(score_path.read_text())['experiments'][EXPERIMENT]['results']['efnn']
    finite = {name: row for name, row in scores.items()
              if all(k in row and math.isfinite(row[k]) for k in metrics)}
    ref = scores['reference_saved']
    honest = [n for n in finite if not n.startswith('shortcut_')]
    without_catastrophic = [n for n in finite if n != 'shortcut_catastrophic_output']
    cohorts = {'honest_variants': honest, 'finite_audit_attempts': list(finite),
               'without_catastrophic_output': without_catastrophic}
    rank_data = {}
    for cohort, names in cohorts.items():
        sd = {k: statistics.pstdev(finite[n][k] for n in names) for k in metrics}
        ranks = {}
        for name, row in finite.items():
            terms = {}
            for k, definition in metrics.items():
                delta = row[k] - ref[k]
                direction = 1 if definition['higher_is_better'] else -1
                terms[k] = (spec['weight'] * abs(definition['coefficient']) * direction * delta / sd[k]
                            if sd[k] else None)
            # Zero-spread metrics with zero difference contribute zero.
            valid = all(v is not None or row[k] == ref[k] for k, v in terms.items())
            ranks[name] = {'rank_delta': sum(v or 0 for v in terms.values()) if valid else None,
                           'terms': terms}
        rank_data[cohort] = {'members': names, 'population_std': sd, 'ranks': ranks}
    seeds = [n for n in scores if n.startswith('reference_seed_')]
    spread = {k: {'min': min(scores[n][k] for n in seeds),
                  'max': max(scores[n][k] for n in seeds),
                  'range': max(scores[n][k] for n in seeds) - min(scores[n][k] for n in seeds),
                  'population_std': statistics.pstdev(scores[n][k] for n in seeds)}
              for k in metrics} if seeds else {}
    report = {'note': 'Positive rank_delta beats the saved reference. All coefficients use absolute magnitude. '
                      'Normalization depends on future attempts; these named cohorts are explicit audit examples. '
                      'Only binary is active in official scores; continuous has no official test reference.',
              'reference': ref, 'scores': scores, 'statuses': statuses,
              'seed_spread': spread, 'cohorts': rank_data}
    (AUDIT / 'rank_summary.json').write_text(json.dumps(report, indent=2))
    for name, row in scores.items():
        rank = rank_data['finite_audit_attempts']['ranks'].get(name, {}).get('rank_delta')
        summary = {'metrics': row, 'rank_delta_finite_audit_cohort': rank}
        (AUDIT / name / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f'{name}: RMSE={row["rmse"]:.8g}, rel={row["relative_error"]:.8g}, '
              f'width={row["neuron_count"]:g}, rank={rank}', flush=True)
    print('SEED_SPREAD', json.dumps(spread), flush=True)


if __name__ == '__main__':
    main()
