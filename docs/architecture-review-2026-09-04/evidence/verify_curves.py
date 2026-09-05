"""Recheck archived paired periodic evaluations; never train or rewrite results."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent


def read_arm(suite, arm):
    runs = {}
    for path in sorted((ROOT / 'nanolab/out' / suite).glob('*/metrics.jsonl')):
        name = path.parent.name.split('_', 1)[1].rsplit('_s', 1)[0]
        if name != arm:
            continue
        config = json.loads((path.parent / 'config.json').read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert sum(row.get('event') == 'start' for row in rows) == 1, path
        seed = config['seed']
        assert seed not in runs, (suite, arm, seed)
        evaluations = [row for row in rows if row.get('event') == 'eval']
        curve = {row['tokens']: row['val_loss'] for row in evaluations}
        assert len(curve) == len(evaluations), path
        assert all(math.isfinite(value) for value in curve.values()), path
        runs[seed] = curve
    assert set(runs) == {42, 100, 777, 1337, 2026}, (suite, arm, sorted(runs))
    return runs


def main():
    saved = json.loads((EVIDENCE / 'memo-curve-check.json').read_text())
    attention = read_arm('crossover50m', 'attention')
    mingru = read_arm('crossover50m', 'mingru')
    hybrid = read_arm('crossover50m_ratioplace32', 'hybrid_mingru8_attn4')
    periodic = read_arm('crossover50m_ratioplace32', 'hybrid_mingru_periodic')
    comparisons = [
        ('hybrid8_4-minus-attention', hybrid, attention),
        ('hybrid8_4-minus-mingru', hybrid, mingru),
        ('hybrid9_3-minus-attention', periodic, attention),
        ('replication-loop-minus-old-attention', read_arm('crossover50m_loop32', 'attention'), attention),
        ('replication-moe-minus-old-attention', read_arm('crossover50m_moe32b', 'attention'), attention),
    ]
    results = {}
    for name, first, second in comparisons:
        seeds = sorted(first)
        markers = sorted(set.intersection(*(set(c) for c in [*first.values(), *second.values()])))
        archived = saved[name]
        assert seeds == archived['seeds']
        assert markers == [row['tokens'] for row in archived['rows']]
        for tokens, row in zip(markers, archived['rows']):
            deltas = [first[seed][tokens] - second[seed][tokens] for seed in seeds]
            mean = statistics.mean(deltas)
            half = 2.776445105 * statistics.stdev(deltas) / math.sqrt(5)
            assert deltas == row['deltas'], (name, tokens)
            assert abs(mean - row['mean']) < 1e-12
            assert all(abs(a - b) < 1e-12 for a, b in zip([mean - half, mean + half], row['ci95']))
            assert sum(value < 0 for value in deltas) == row['wins']
        if name.startswith('hybrid8_4'):
            assert len(markers[1:]) == 60
            assert all(row['wins'] == 5 for row in archived['rows'][1:])
        results[name] = {'markers': len(markers), 'seeds': len(seeds),
                         'last_periodic_delta': archived['rows'][-1]['mean']}
    print(json.dumps({'status': 'passed', 'comparisons': results,
                      'scope': 'Saved periodic eval.val_loss, not final_val or independent training replication.'}, indent=2))


if __name__ == '__main__':
    main()
