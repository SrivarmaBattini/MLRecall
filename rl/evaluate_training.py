"""Optional *actual training-seed* learning curves / sample efficiency for QWI.

Trains SEPARATE ephemeral QWI agents and evaluates each after a series of
training-transition checkpoints; does not overwrite artifacts/index_table.npz.

Example (small sanity experiment only):
  python -m rl.evaluate_training --train-seeds 3 --eval-seeds 5 \
      --steps 8192 --checkpoints 4 --days 30 --K 5

For report-quality experiments increase steps toward your real training budget,
use >=5 independent training seeds, and explicitly predeclare a target mean
retention if you want to report steps-to-target. The target is not necessarily
reached, which will be reported as missing rather than fabricated.

X-axis is TRAINING transitions PER HALF-LIFE BUCKET (not full environment days).
Outputs are simulated and cannot establish gains for real learners.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import AGENT, ROOT
from rl.evaluation_metrics import mean_ci95
from rl.evaluate_baselines import make_scenario, topic_betas, evaluate_policy
from rl.qwi import QWI


def run_training_experiment(train_seeds=3, eval_seeds=5, steps=8192,
                            checkpoints=4, days=30, K=5, target=None,
                            world='exp', batch=64):
    if min(train_seeds, eval_seeds, steps, checkpoints, days, K) < 1:
        raise ValueError('All counts must be positive')
    if checkpoints > steps // batch:
        raise ValueError('Too few training steps for distinct batch-sized checkpoints')
    if target is not None and not 0 < target <= 1:
        raise ValueError('target mean retention must be in (0,1]')
    # Train/test seeds independent by construction; evaluation scenarios fixed
    # across all checkpoints so changes are attributable to training.
    betas = topic_betas()
    scenarios = [make_scenario(10_000 + i, days, betas) for i in range(eval_seeds)]
    # Keep batch-aligned checkpoints; train() uses steps//batch updates.
    total_batches = steps // batch
    batch_marks = np.linspace(1, total_batches, checkpoints).astype(int)
    batch_marks = np.unique(batch_marks)
    if len(batch_marks) != checkpoints:
        raise ValueError('Not enough unique training checkpoints')
    rows = []
    for seed in range(train_seeds):
        model = QWI(seed=seed, batch=batch)
        prior_mark = 0
        cumulative_train_time = 0.0
        for mark in batch_marks:
            num_steps = int(mark - prior_mark) * batch
            start = time.perf_counter()
            model.train(steps=num_steps, verbose=False)
            cumulative_train_time += time.perf_counter() - start
            prior_mark = mark
            W = model.index_table()
            vals = [evaluate_policy('QWI (online speed)', scenario, world, K, W)
                    for scenario in scenarios]
            rows.append({'train_seed': seed, 'world': world,
                         'steps_per_bucket': int(mark * batch),
                         'total_bucket_transitions': int(mark * batch * model.B),
                         'eval_seeds': eval_seeds, 'eval_days': days,
                         'mean_eval_episodic_return': float(np.mean([v['episodic_return'] for v in vals])),
                         'mean_eval_daily_retention': float(np.mean([v['mean_retention'] for v in vals])),
                         'mean_eval_final_ARP': float(np.mean([v['final_ARP'] for v in vals])),
                         'cumulative_train_seconds': cumulative_train_time})
        print(f'Training seed {seed + 1}/{train_seeds} complete', flush=True)
    per_seed = pd.DataFrame(rows)
    summary = []
    for n, g in per_seed.groupby('steps_per_bucket'):
        m, sd, lo, hi = mean_ci95(g['mean_eval_episodic_return'])
        daily = mean_ci95(g['mean_eval_daily_retention'])
        summary.append({'steps_per_bucket': n,
                        'total_bucket_transitions': int(g['total_bucket_transitions'].iloc[0]),
                        'n_training_seeds': train_seeds,
                        'mean_eval_episodic_return': m,
                        'std_across_training_seeds': sd,
                        'return_ci_low': lo, 'return_ci_high': hi,
                        'mean_eval_daily_retention': daily[0],
                        'daily_retention_ci_low': daily[2],
                        'daily_retention_ci_high': daily[3],
                        'mean_final_ARP': float(g['mean_eval_final_ARP'].mean()),
                        'mean_cumulative_train_seconds': float(g['cumulative_train_seconds'].mean())})
    summary = pd.DataFrame(summary).sort_values('steps_per_bucket')
    # Steps-to-target after *two consecutive checkpoints*, not a single spike.
    target_rows = []
    if target is not None:
        for seed, g in per_seed.groupby('train_seed'):
            g = g.sort_values('steps_per_bucket')
            good = (g['mean_eval_daily_retention'].to_numpy() >= target)
            idx = np.flatnonzero(good[:-1] & good[1:]) if len(good) >= 2 else []
            target_rows.append({'train_seed': seed, 'target_mean_retention': target,
                                'steps_to_target_per_bucket':
                                int(g.iloc[idx[0]]['steps_per_bucket']) if len(idx) else np.nan,
                                'target_reached': int(bool(len(idx)))})
    return per_seed, summary, pd.DataFrame(target_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-seeds', type=int, default=3)
    p.add_argument('--eval-seeds', type=int, default=5)
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--checkpoints', type=int, default=4)
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--K', type=int, default=AGENT.default_K)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--world', choices=['exp', 'power'], default='exp')
    p.add_argument('--target', type=float, default=None,
                   help='predeclared mean daily retention target, two consecutive checkpoints')
    a = p.parse_args()
    per_seed, summary, target = run_training_experiment(
        a.train_seeds, a.eval_seeds, a.steps, a.checkpoints,
        a.days, a.K, a.target, a.world, a.batch)
    out = Path(ROOT) / 'results'
    out.mkdir(exist_ok=True)
    per_seed.to_csv(out / 'training_seed_checkpoints.csv', index=False)
    summary.to_csv(out / 'training_learning_curve.csv', index=False)
    if not target.empty:
        target.to_csv(out / 'training_steps_to_target.csv', index=False)
    print('\nSimulated EVALUATION RETURN vs actual QWI training steps:')
    print(summary.round(4).to_string(index=False))
    if not target.empty:
        print('\nTarget reached for', int(target['target_reached'].sum()),
              'of', len(target), 'training seeds')
    print('\nNote: training steps are simulated transitions PER BUCKET;'
          ' all results are simulation-only. The saved app model was not changed.')


if __name__ == '__main__':
    main()
