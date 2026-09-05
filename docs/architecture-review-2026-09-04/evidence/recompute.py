"""Read-only analysis of the local MLSystemsLab experiment snapshot.

Writes a fresh snapshot to a required new --output-dir. Does not launch training,
modify source experiments, deserialize model checkpoints, or access providers.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True, help='New output directory; archived evidence is never overwritten.')
args = parser.parse_args()
DEST = args.output_dir.resolve()
if DEST.exists():
    raise SystemExit('Refusing an existing output directory: ' + str(DEST))
DEST.mkdir(parents=True)
T95 = {1: 12.706204736, 2: 4.30265273, 3: 3.182446305,
       4: 2.776445105, 9: 2.262157163, 14: 2.144786688}


def estimate(values):
    n = len(values)
    if not n:
        return {'n': 0, 'mean': None, 'ci95': None}
    mean = statistics.fmean(values)
    interval = None
    if n - 1 in T95:
        half = T95[n - 1] * statistics.stdev(values) / math.sqrt(n)
        interval = [mean - half, mean + half]
    return {'n': n, 'mean': mean, 'ci95': interval,
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def load_lines(path):
    rows, problems = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            problems.append({'line': line_no, 'error': str(exc)})
            continue
        if not isinstance(row, dict):
            problems.append({'line': line_no, 'error': 'non-object row'})
            continue
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                problems.append({'line': line_no, 'error': 'nonfinite ' + k})
        rows.append((line_no, row))
    return rows, problems


def pair(a, b):
    shared = sorted(a.keys() & b.keys())
    vals = [a[s] - b[s] for s in shared]
    return {'meaning': 'first arm minus second; negative favors first',
            'shared_seeds': shared, 'delta': estimate(vals),
            'first_wins': sum(x < 0 for x in vals)}


def main():
    result_roots = ['nanolab/out', 'out',
                    'Rust_MLKit/arch_02_value_resid/metal-native/out',
                    'Rust_MLKit/reference/ablation_results', 'research',
                    'Rust_MLKit/crates/tessl/bench/results',
                    'Rust_MLKit/gemma-metal/bench/results']
    inventory, metric_inventory = [], []
    for rel in result_roots:
        for path in sorted((ROOT / rel).rglob('*')):
            if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.csv'):
                continue
            data = path.read_bytes()
            entry = {'path': str(path.relative_to(ROOT)), 'root': rel,
                     'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            inventory.append(entry)
            if path.name != 'metrics.jsonl':
                continue
            rows, problems = load_lines(path)
            counts = collections.Counter(str(r.get('event', '<absent>')) for _, r in rows)
            terminal = [(i, r) for i, r in rows
                        if r.get('event') in ('done', 'FINAL', 'final')]
            metric_inventory.append({'path': entry['path'], 'rows': len(rows),
                                     'events': dict(counts), 'problems': problems,
                                     'terminal': terminal[-1] if terminal else None})
    suites = {}
    for root in sorted((ROOT / 'nanolab/out').glob('crossover*')):
        if not root.is_dir():
            continue
        recipe_path, queue_path = root / 'recipe.json', root / 'queue.json'
        recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else None
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else None
        runs, arms, per_seed = [], collections.defaultdict(list), collections.defaultdict(dict)
        for path in sorted(root.glob('*/metrics.jsonl')):
            rows, problems = load_lines(path)
            cfg_path = path.parent / 'config.json'
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            arm = path.parent.name.split('_', 1)[1].rsplit('_s', 1)[0]
            starts = [(i, r) for i, r in rows if r.get('event') == 'start']
            last_start = starts[-1][0] if starts else 0
            done = [(i, r) for i, r in rows if i > last_start and r.get('event') == 'done']
            final = done[-1] if done else None
            entry = {'path': str(path.relative_to(ROOT)), 'arm': arm,
                     'seed': cfg.get('seed'), 'config': cfg,
                     'start': starts[-1] if starts else None,
                     'start_count': len(starts), 'done': final, 'problems': problems}
            runs.append(entry)
            if final is not None and isinstance(final[1].get('final_val'), (int, float)):
                value = final[1]['final_val']
                if not math.isfinite(value) or problems:
                    continue
                seed = str(cfg.get('seed'))
                if seed in per_seed[arm]:
                    raise RuntimeError(f'duplicate seed in {root.name}/{arm}: {seed}')
                per_seed[arm][seed] = value
                arms[arm].append(entry)
        board = {}
        for arm, entries in arms.items():
            completed = [e['done'][1] for e in entries]
            board[arm] = {'final_ce': estimate(list(per_seed[arm].values())),
                          'per_seed': per_seed[arm],
                          'params': sorted({e['start'][1].get('params') for e in entries}),
                          'tokens': sorted({r.get('tokens') for r in completed}),
                          'elapsed_s': estimate([r['elapsed_s'] for r in completed if r.get('elapsed_s')]),
                          'sources': [e['path'] + ':' + str(e['done'][0]) for e in entries]}
        pairs = {}
        for a, b in [('swa_w512', 'attention'), ('swa_w256', 'attention'),
                     ('swa_w64', 'swa_w64_nosink'), ('looped_attn6x2', 'attention'),
                     ('looped_attn6x2', 'attn6'), ('looped_attn3x4', 'attn3'),
                     ('moe_e1k1', 'attention'),
                     ('hybrid_mingru_periodic', 'hybrid_mingru10_attn2'),
                     ('hybrid_mingru10_attn2', 'hybrid_mingru_bookend')]:
            if a in per_seed and b in per_seed:
                pairs[a + ' minus ' + b] = pair(per_seed[a], per_seed[b])
        for width in (384, 768, 1152):
            a, b = f'w{width}_attention_lr80', f'w{width}_mingru_lr40'
            if a in per_seed and b in per_seed:
                pairs[a + ' minus ' + b] = pair(per_seed[a], per_seed[b])
        suites[root.name] = {'recipe': recipe,
                            'queue_statuses': dict(collections.Counter(j['status'] for j in queue['jobs'])) if queue else None,
                            'runs': runs, 'board': board, 'paired': pairs}
    recall = {}
    for root in sorted((ROOT / 'nanolab/out').glob('mqar*')):
        for path in root.glob('runs.jsonl'):
            rows, problems = load_lines(path)
            groups = collections.defaultdict(list)
            for i, r in rows:
                family = r.get('layer_mixers') or r['arm']
                key = (family, r.get('n_pairs'), r.get('steps'),
                       r.get('batch_size'), r.get('block_size'))
                groups[key].append((i, r))
            cells = []
            for key, vals in sorted(groups.items(), key=lambda x: str(x[0])):
                seeds = [r['seed'] for _, r in vals]
                if len(set(seeds)) != len(seeds):
                    raise RuntimeError(f'duplicate MQAR cell seed {root}/{key}')
                cells.append({'family_pairs_steps_batch_context': key, 'n': len(vals),
                              'solved': sum(r['solved'] for _, r in vals),
                              'recall': estimate([r['recall'] for _, r in vals]),
                              'lines': [i for i, _ in vals]})
            recall[root.name] = {'path': str(path.relative_to(ROOT)), 'records': len(rows),
                                 'problems': problems, 'cells': cells}
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    report = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT), 'git_head': head,
              'limitations': ['Archived measurements were recomputed; training was not rerun.',
                              'Inventoried artifacts may duplicate experiments across directories.',
                              'Only explicit final_val values enter final-CE boards; no best_val substitution.',
                              'Recipe matching and causal interpretation require the accompanying review.',
                              'Intervals are descriptive Student-t seed intervals, not corrected multi-comparison tests.'],
              'inventory_counts': dict(collections.Counter(x['root'] for x in inventory)),
              'inventory_files': len(inventory), 'unique_content_hashes': len({x['sha256'] for x in inventory}),
              'metrics_files': len(metric_inventory), 'metrics': metric_inventory,
              'crossover_suites': suites, 'mqar': recall}
    (DEST / 'artifact-manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (DEST / 'recomputed-evidence.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Recomputed architecture evidence', '',
             f'Snapshot: {report["snapshot_utc"]}; Git HEAD: `{head}`.', '',
             'Local snapshot only. Final CE is used; lower is better. Intervals describe seed variation.', '',
             f'{len(inventory)} result artifacts inventoried; {len(metric_inventory)} metrics files parsed.', '',
             '| Suite | Arm | Finished n | Final CE | 95% seed interval | Parameters | Mean elapsed s |',
             '|---|---|---:|---:|---|---:|---:|']
    for name, suite in suites.items():
        for arm, row in suite['board'].items():
            stat = row['final_ce']; ci = stat['ci95']; tm = row['elapsed_s']['mean']
            ci_text = f'[{ci[0]:.5f}, {ci[1]:.5f}]' if ci else 'not estimated'
            time_text = f'{tm:.1f}' if tm else 'missing'
            lines.append(f'| {name} | {arm} | {stat["n"]} | {stat["mean"]:.6f} | {ci_text} | {row["params"]} | {time_text} |')
    lines += ['', '## Recall cells', '',
              'A hybrid is identified by its full layer layout; grouping only by the base mixer would combine different architectures.', '',
              '| Suite | Family / pairs / steps / batch / context | Solved | Median recall |', '|---|---|---:|---:|']
    for name, suite in recall.items():
        for cell in suite['cells']:
            lines.append(f'| {name} | {cell["family_pairs_steps_batch_context"]} | {cell["solved"]}/{cell["n"]} | {cell["recall"]["median"]:.6f} |')
    (DEST / 'evidence-tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'inventory_files': len(inventory), 'unique_hashes': report['unique_content_hashes'],
                      'metrics_files': len(metric_inventory),
                      'metrics_with_parse_or_nonfinite_issues': sum(bool(x['problems']) for x in metric_inventory),
                      'crossover_suites': len(suites), 'crossover_metrics': sum(len(x['runs']) for x in suites.values()),
                      'final_ce_runs': sum(r['final_ce']['n'] for x in suites.values() for r in x['board'].values()),
                      'mqar_records': sum(x['records'] for x in recall.values()), 'counts': report['inventory_counts']}, indent=2))


if __name__ == '__main__':
    main()
