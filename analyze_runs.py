#!/usr/bin/env python3
"""Aggregate finished attack runs into a per-run CSV + per-config summary.

Every run is identified from the `run_metadata` block the runner writes into
each comprehensive_log_iter_FINAL_*.json (algorithm + full hyperparameters),
NOT from the directory name -- so runs from any batch/layout are picked up and
different configurations are never pooled together.

Metrics (MAP-Elites framing, so beam -- which truncates each cell to
tree_width survivors -- stays comparable to MCTS, which keeps every tree node):
  AQD    mean per-cell BEST judge score (1-10), never-scored migrants excluded
  ge7    cells whose best score reaches >= 7
  filled cells with at least one scored instance
  target/attack/eval  query counts (target is the cost currency)

Usage:
  python analyze_runs.py                      # scan the default /tmp run roots
  python analyze_runs.py /tmp/conf /tmp/long  # specific roots
  python analyze_runs.py --csv results/runs.csv
  python analyze_runs.py --dataset harmbench --budget 1500
"""
import argparse, csv, glob, json, os, statistics
from collections import defaultdict

DEFAULT_ROOTS = ['/tmp/conf', '/tmp/long', '/tmp/curve', '/tmp/peak',
                 '/tmp/alloc', '/tmp/sweep', '/tmp/full', '/tmp/data', '/tmp/var']
METHOD = {'gap-beam-search': 'beam', 'gap-mcts': 'mcts', 'rainbowplus-baseline': 'baseline'}


def config_sig(method, hp):
    """Configuration signature -- runs only pool together if this matches."""
    if method == 'beam':
        return f"d{hp.get('tree_depth')}"
    if method == 'mcts':
        return (f"t{hp.get('total')}w{hp.get('tree_width')}"
                f"c{hp.get('uct_c') if hp.get('uct_c') is not None else 'def'}"
                f"T{hp.get('terminal_score')}")
    return 'base'


def read_log(path):
    with open(path) as f:
        d = json.load(f)
    meta = d.get('run_metadata') or {}
    hp = meta.get('hyperparameters') or {}
    method = METHOD.get(meta.get('algorithm'), meta.get('algorithm') or '?')
    archive = d.get('map_elites_archive', {})
    cell_best = []
    for items in archive.values():
        s = [it.get('score') or 0 for it in items]
        s = [x for x in s if x > 0]
        if s:
            cell_best.append(max(s))
    if not cell_best:
        return None
    logs = d.get('llm_call_logs', {})
    q = lambda r: (logs.get(r, {}) or {}).get('total_queries')
    return dict(
        method=method,
        dataset=os.path.basename(str(hp.get('dataset', '?'))).replace('formatted_', '').replace('.json', ''),
        budget=hp.get('target_query_budget'),
        config=config_sig(method, hp),
        seed=hp.get('seed'),
        aqd=sum(cell_best) / len(cell_best),
        ge5=sum(1 for b in cell_best if b >= 5),
        ge7=sum(1 for b in cell_best if b >= 7),
        maxq=max(cell_best),
        filled=len(cell_best),
        target=q('target_llm'), attack=q('attack_llm'), eval=q('eval_llm'),
        path=path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='*')
    ap.add_argument('--csv')
    ap.add_argument('--dataset')
    ap.add_argument('--budget', type=int)
    args = ap.parse_args()

    roots = args.roots or [r for r in DEFAULT_ROOTS if os.path.isdir(r)]
    rows = []
    for root in roots:
        for p in glob.glob(os.path.join(root, '**', 'comprehensive_log_iter_FINAL_*.json'), recursive=True):
            try:
                r = read_log(p)
            except Exception:
                continue
            if r:
                rows.append(r)
    if args.dataset:
        rows = [r for r in rows if r['dataset'] == args.dataset]
    if args.budget:
        rows = [r for r in rows if r['budget'] == args.budget]
    if not rows:
        print('no finished runs found under: ' + ', '.join(roots))
        return

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        cols = ['method', 'config', 'dataset', 'budget', 'seed', 'aqd', 'ge5', 'ge7',
                'maxq', 'filled', 'target', 'attack', 'eval', 'path']
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r['dataset'], str(r['budget']), r['method'], r['config'])))
        print(f'per-run CSV -> {args.csv}  ({len(rows)} runs)\n')

    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['budget'], r['method'], r['config'])].append(r)

    hdr = f"{'dataset':20s} {'bud':>5s} {'method':>6s} {'config':>16s} {'n':>3s} | {'AQD':>14s} {'ge7':>6s} {'filled':>6s} {'tgtQ':>6s}"
    print(hdr); print('-' * len(hdr))
    for k in sorted(groups, key=lambda k: (k[0], k[1] or 0, k[2], k[3])):
        ds, bud, meth, cfg = k
        g = groups[k]
        a = [x['aqd'] for x in g]
        sd = statistics.stdev(a) if len(a) > 1 else 0.0
        print(f"{ds:20s} {str(bud):>5s} {meth:>6s} {cfg:>16s} {len(g):>3d} | "
              f"{statistics.mean(a):7.2f}±{sd:<6.2f} {statistics.mean(x['ge7'] for x in g):6.1f} "
              f"{statistics.mean(x['filled'] for x in g):6.1f} {statistics.mean((x['target'] or 0) for x in g):6.0f}")
    print("\nAQD = mean per-cell best judge score (1-10). Runs pool only when dataset+budget+method+config all match.")


if __name__ == '__main__':
    main()
