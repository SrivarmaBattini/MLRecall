"""Fairer, reproducible SIMULATED SmartRecall scheduler evaluation.

Example (run from project root after replacing this file):
    python -m rl.evaluate_baselines --world both --seeds 30 --days 120 --K 5
    python -m rl.evaluate_baselines --world exp --seeds 3 --days 20 --include-dp

Primary *deployable* QWI uses an ONLINE speed MAP estimate from prior QUIZES;
it cannot inspect hidden simulated learner speed.  "QWI (oracle speed)" and
"DP Whittle (oracle)" are diagnostic nondeployable reference policies.

For paired comparison, every policy/seed/world uses the same initial quiz
uniforms, latent speed, topic difficulty and day/topic/question random uniforms.
Different state histories result in different quiz pass probabilities, as they
should. All policies use apply_quiz() for the memory-state transition.

No actual human subjects, real-life retention, training convergence curves,
DRL-SRS reproduction, or optimal-policy regret are measured here.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import AGENT, RULES, ROOT, CONTENT_DIR, INDEX_TABLE_PATH
from rl.evaluation_metrics import (mean_ci95, success_ci95,
                                  quiz_prediction_metrics, empirical_hindsight_gap)
from rl.memory_model import (apply_quiz, estimate_speed, nearest_bucket,
                             question_correct_prob, recall_probability)
from rl.simulator import true_recall

G = np.asarray(RULES.level_multipliers, dtype=float)
DEPLOYABLE = ('QWI (online speed)', 'QWI (prior speed)', 'Random',
              'Fixed rotation', 'Most overdue', 'Lowest last score',
              'Forgetting risk (online)', 'Leitner-style', 'SM-2-style')
ORACLE = ('QWI (oracle speed)', 'DP Whittle (oracle)')


def topic_betas():
    """Read difficulties from original lesson frontmatter without importing the web API."""
    files = sorted(Path(CONTENT_DIR).glob('*.md'))
    if not files:
        raise FileNotFoundError('No content/topics/*.md lessons found')
    out = []
    for path in files:
        txt = path.read_text(encoding='utf-8')
        frontmatter = txt.split('---', 2)[1] if txt.startswith('---') else ''
        m = re.search(r'^difficulty:\s*(\d+)\s*$', frontmatter, re.M)
        if not m:
            raise ValueError(f'Missing difficulty frontmatter in {path.name}')
        out.append(float(RULES.base_half_life[int(m.group(1))]))
    return np.asarray(out, dtype=float)


def make_scenario(seed: int, days: int, betas: np.ndarray):
    """Fixed exogenous randomness shared among all policies in one scenario."""
    rng = np.random.default_rng(seed)
    n = len(betas)
    # Same distribution as the project, but do not reveal this variable to
    # deployable policies. Topic order is fixed across experiments.
    speed = float(np.clip(np.exp(rng.normal(0, RULES.speed_prior_sigma)),
                          *RULES.speed_bounds))
    comprehension = rng.beta(6, 2, size=n)
    u_first = rng.random((n, RULES.n_questions))
    q_first = question_correct_prob(comprehension)
    first_scores = (u_first < q_first[:, None]).sum(axis=1).astype(int)
    first_levels = np.array([apply_quiz(None, 0, int(sc))[0]['k'] for sc in first_scores])
    return {'seed': seed, 'speed': speed, 'betas': betas,
            'initial_levels': first_levels, 'initial_scores': first_scores,
            'quiz_uniforms': rng.random((days, n, RULES.n_questions)),
            'choice_uniforms': rng.random((days, n))}


def _choose(name, day, K, state, W, dp_W=None):
    """Policy only receives observable state, except explicitly named oracles."""
    n = len(state['k'])
    k = state['k']
    age = state['age']
    d_clip = np.minimum(age, AGENT.D)
    if K <= 0:
        return np.array([], dtype=int)
    if name.startswith('QWI') or name == 'DP Whittle (oracle)':
        if name == 'QWI (online speed)':
            s = state['online_speed']
        elif name == 'QWI (prior speed)':
            s = 1.0
        else:
            s = state['oracle_speed']
        bucket = np.array([nearest_bucket(x) for x in state['beta'] * s])
        idx = dp_W if name == 'DP Whittle (oracle)' else W
        score = idx[bucket, k, d_clip]
        return np.argsort(-score, kind='stable')[:K]
    if name == 'Random':
        return np.argsort(state['rand'], kind='stable')[:K]
    if name == 'Fixed rotation':
        start = ((day - 1) * K) % n
        return np.array([(start + x) % n for x in range(min(K, n))], int)
    if name == 'Most overdue':
        return np.argsort(-age + 1e-7 * state['rand'], kind='stable')[:K]
    if name == 'Lowest last score':
        return np.argsort(state['score'] + 1e-7 * state['rand'], kind='stable')[:K]
    if name == 'Forgetting risk (online)':
        p_est = recall_probability(age, state['beta'] * state['online_speed'] * G[k])
        return np.argsort(p_est + 1e-7 * state['rand'], kind='stable')[:K]
    if name in ('SM-2-style', 'Leitner-style'):
        due = state['due']
        available = np.flatnonzero(due <= day)
        if available.size == 0:
            return np.array([], dtype=int)
        # larger overdue gap receives higher priority
        return available[np.argsort(-(day - due[available]), kind='stable')][:K]
    raise ValueError(f'Unknown policy {name}')


def _update_due(name, chosen, scores, day, state):
    if name == 'Leitner-style':
        for i, count in zip(chosen, scores):
            # Level derived from the *observed* shared quiz update.
            state['due'][i] = day + max(1, int(2 ** int(state['k'][i])))
    if name == 'SM-2-style':
        for i, count in zip(chosen, scores):
            if count < RULES.review_success_min_correct:
                state['reps'][i] = 0
                state['interval'][i] = 1
            else:
                state['reps'][i] += 1
                if state['reps'][i] == 1:
                    interval = 1
                elif state['reps'][i] == 2:
                    interval = 6
                else:
                    interval = max(1, int(round(state['interval'][i] * state['ease'][i])))
                state['interval'][i] = interval
            # An openly declared simplified SM-2 approximation, not Anki or FSRS.
            q = int(count)
            state['ease'][i] = max(1.3, state['ease'][i] + (.1 - (5 - q) * (.08 + (5 - q) * .02)))
            state['due'][i] = day + int(state['interval'][i])


def evaluate_policy(name, scenario, world, K, W, threshold=.5,
                    success_share=.8, dp_W=None):
    """One matched simulated learner; daily reward is mean PRE-review recall."""
    betas = scenario['betas']
    n = len(betas)
    days = len(scenario['quiz_uniforms'])
    k = scenario['initial_levels'].copy()
    last = np.zeros(n, dtype=int)      # first quiz on day zero
    score = scenario['initial_scores'].copy()
    n_reviews = np.zeros(n, dtype=int)
    state = {'k': k, 'beta': betas, 'score': score,
             'due': np.ones(n, dtype=int), 'interval': np.ones(n, dtype=int),
             'ease': np.full(n, 2.5), 'reps': np.zeros(n, dtype=int)}
    reviews = []
    measured_q, measured_correct = [], []
    rewards, final_good_share, budget_violations, review_count = [], np.nan, 0, 0
    selection_sec = 0.0
    scheduler_sec = 0.0
    for t in range(days):
        day = t + 1
        age = day - last  # TRUE elapsed days, not clipped by the agent's D=40
        p_true = np.asarray(true_recall(age, betas * scenario['speed'] * G[k], world), float)
        rewards.append(float(p_true.mean()))
        scheduler_start = time.perf_counter()
        estimated = estimate_speed(reviews) if reviews else 1.0
        state.update({'age': age.copy(), 'online_speed': estimated,
                      'rand': scenario['choice_uniforms'][t]})
        if name in ORACLE:
            state['oracle_speed'] = scenario['speed']  # explicitly nondeployable
        tic = time.perf_counter()
        chosen = np.asarray(_choose(name, day, min(K, n), state, W, dp_W), dtype=int)
        selection_sec += time.perf_counter() - tic
        scheduler_sec += time.perf_counter() - scheduler_start
        if len(set(chosen.tolist())) != len(chosen) or np.any((chosen < 0) | (chosen >= n)):
            raise RuntimeError(f'{name} selected invalid/duplicate topics')
        if len(chosen) > K:
            budget_violations += 1
        review_count += len(chosen)
        quiz_scores = []
        for i in chosen:
            q_true = float(question_correct_prob(p_true[i]))
            right = int((scenario['quiz_uniforms'][t, i] < q_true).sum())
            # Prediction must use the *pre-quiz* online estimate, even for
            # oracle scheduler comparisons: prediction quality is deployable.
            p_hat = float(recall_probability(age[i], betas[i] * estimated * G[k[i]]))
            measured_q.append(float(question_correct_prob(p_hat)))
            measured_correct.append(right)
            old = {'k': int(k[i]), 'last_day': int(last[i]),
                   'n_reviews': int(n_reviews[i])}
            next_state, row, kind = apply_quiz(old, day, right, RULES.n_questions)
            if kind != 'review' or row is None:
                raise AssertionError('Evaluation review must match shared app rules')
            reviews.append((betas[i],) + row)
            k[i] = next_state['k']
            n_reviews[i] = next_state['n_reviews']
            last[i] = day
            score[i] = right
            quiz_scores.append(right)
        _update_due(name, chosen, quiz_scores, day, state)
    # End-of-horizon recall at the next day, NOT the preceding pre-review reward.
    final_age = (days + 1) - last
    p_final = np.asarray(true_recall(final_age, betas * scenario['speed'] * G[k], world), float)
    final_good_share = float(np.mean(p_final >= threshold))
    qm = quiz_prediction_metrics(measured_q, measured_correct,
                                 [RULES.n_questions] * len(measured_correct))
    return {'seed': int(scenario['seed']), 'world': world, 'policy': name,
            'speed_true': scenario['speed'],
            'mean_retention': float(np.mean(rewards)),
            'episodic_return': float(sum(rewards)),
            'final_ARP': float(p_final.mean()),
            'final_share_above_threshold': final_good_share,
            'success': int(final_good_share >= success_share),
            'reviews_total': review_count, 'reviews_per_day': review_count / days,
            'budget_violations': budget_violations,
            'budget_respected': int(budget_violations == 0),
            'policy_selection_ms_per_day': selection_sec * 1000 / days,
            'scheduler_compute_ms_per_day': scheduler_sec * 1000 / days,
            **qm, 'rewards': np.asarray(rewards)}


def _reference_table(W, grid=61):
    """DP *single-arm index*, not an optimal N-arm budget-constrained scheduler."""
    from rl.whittle_reference import whittle_index_dp
    from rl.memory_model import bucket_grid
    result = np.zeros_like(W)
    for i, b in enumerate(bucket_grid()):
        idx, report = whittle_index_dp(float(b), n_grid=grid)
        print(f'  DP bucket {i + 1}/{len(bucket_grid())}: '
              f'{report["violating_states"]} indexability violations', flush=True)
        result[i] = idx.reshape(RULES.L, AGENT.D + 1)
    return result


def run_benchmark(seeds=30, days=120, K=None, world='both', include_dp=False,
                  threshold=.5, success_share=.8, verbose=True):
    K = AGENT.default_K if K is None else int(K)
    if seeds < 1 or days < 1 or K < 1:
        raise ValueError('seeds, days and K must all be positive')
    if not (0 < threshold <= 1 and 0 < success_share <= 1):
        raise ValueError('Threshold and success share must be within (0, 1]')
    if not INDEX_TABLE_PATH.exists():
        raise FileNotFoundError(f'Missing {INDEX_TABLE_PATH}; run python -m rl.train_agent')
    with np.load(INDEX_TABLE_PATH) as z:
        W = np.asarray(z['W'])
        if W.shape != (AGENT.n_buckets, RULES.L, AGENT.D + 1):
            raise ValueError(f'Index table shape {W.shape} mismatches current config')
        if 'rules_version' in z and str(z['rules_version']) != RULES.version:
            raise ValueError('Trained index table memory_rules version is different')
    betas = topic_betas()
    names = list(DEPLOYABLE) + ['QWI (oracle speed)']
    dp_W = _reference_table(W) if include_dp else None
    if include_dp:
        names += ['DP Whittle (oracle)']
    worlds = ('exp', 'power') if world == 'both' else (world,)
    all_runs, all_curves = [], []
    for w in worlds:
        for seed in range(seeds):
            scenario = make_scenario(seed, days, betas)
            run_for_seed = [evaluate_policy(name, scenario, w, K, W,
                           threshold, success_share, dp_W) for name in names]
            # Ex-post best measured *deployable* policy within this seed.
            hindsight = empirical_hindsight_gap({r['policy']: r['episodic_return']
                                                 for r in run_for_seed
                                                 if r['policy'] in DEPLOYABLE})
            for row in run_for_seed:
                row['hindsight_gap_to_best_deployable'] = hindsight.get(row['policy'], np.nan)
                for d, reward in enumerate(row.pop('rewards'), 1):
                    all_curves.append({'world': w, 'seed': seed, 'policy': row['policy'],
                                       'day': d, 'mean_recall_before_review': float(reward)})
                all_runs.append(row)
            if verbose and (seed + 1) % max(1, seeds // 5) == 0:
                print(f'{w}: {seed + 1}/{seeds} simulated learners', flush=True)
    episodes = pd.DataFrame(all_runs)
    curves = pd.DataFrame(all_curves)
    summary = []
    for (w, name), frame in episodes.groupby(['world', 'policy'], sort=False):
        retention = mean_ci95(frame['mean_retention'])
        arp = mean_ci95(frame['final_ARP'])
        success, success_lo, success_hi = success_ci95(frame['success'])
        entry = {'world': w, 'policy': name,
                 'diagnostic_oracle': int(name in ORACLE), 'n_seeds': len(frame),
                 'mean_retention': retention[0], 'sd_retention': retention[1],
                 'retention_ci_low': retention[2], 'retention_ci_high': retention[3],
                 'mean_episodic_return': float(frame['episodic_return'].mean()),
                 'mean_final_ARP': arp[0], 'final_ARP_ci_low': arp[2],
                 'final_ARP_ci_high': arp[3],
                 'success_rate': success, 'success_ci_low': success_lo,
                 'success_ci_high': success_hi,
                 'mean_final_share_above_threshold': float(frame['final_share_above_threshold'].mean()),
                 'mean_reviews_per_day': float(frame['reviews_per_day'].mean()),
                 'any_budget_violations': int(frame['budget_violations'].sum()),
                 'quiz_mae': float(frame['quiz_mae'].mean()),
                 'quiz_brier': float(frame['quiz_brier'].mean()),
                 'quiz_log_loss': float(frame['quiz_log_loss'].mean()),
                 'quiz_pass_auc': float(frame['quiz_pass_auc'].mean()),
                 'quiz_pass_ece': float(frame['quiz_pass_ece'].mean()),
                 'mean_selection_ms_per_day': float(frame['policy_selection_ms_per_day'].mean()),
                 'mean_scheduler_compute_ms_per_day': float(frame['scheduler_compute_ms_per_day'].mean()),
                 'mean_hindsight_gap': float(frame['hindsight_gap_to_best_deployable'].mean())
                 if name in DEPLOYABLE else float('nan'),
                 'index_table_cells': int(W.size) if name.startswith('QWI') else 0,
                 'index_table_bytes': int(W.nbytes) if name.startswith('QWI') else 0}
        summary.append(entry)
    summary = pd.DataFrame(summary)
    # Proper paired differences to QWI-online, *not* independent two-sample CIs.
    anchor = episodes[episodes.policy == 'QWI (online speed)'].set_index(['world', 'seed'])
    paired = []
    for name in names:
        if name == 'QWI (online speed)':
            continue
        challenger = episodes[episodes.policy == name].set_index(['world', 'seed'])
        for w in worlds:
            a = anchor.loc[w]
            b = challenger.loc[w]
            diff = a['mean_retention'].to_numpy() - b['mean_retention'].to_numpy()
            m, sd, lo, hi = mean_ci95(diff)
            paired.append({'world': w, 'comparison': 'QWI (online speed) minus ' + name,
                           'n_seeds': seeds, 'paired_mean_difference': m,
                           'paired_sd': sd, 'paired_ci_low': lo, 'paired_ci_high': hi,
                           'qwi_wins_seeds': int((diff > 0).sum())})
    paired = pd.DataFrame(paired)
    generalization = []
    if len(worlds) == 2:
        for name in names:
            exp = episodes[(episodes.world == 'exp') & (episodes.policy == name)].sort_values('seed')
            power = episodes[(episodes.world == 'power') & (episodes.policy == name)].sort_values('seed')
            gap = exp['mean_retention'].to_numpy() - power['mean_retention'].to_numpy()
            mean, sd, lo, hi = mean_ci95(gap)
            generalization.append({'policy': name, 'n_seeds': seeds,
                                   'exp_minus_power_mean_retention': mean,
                                   'gap_ci_low': lo, 'gap_ci_high': hi})
    return episodes, curves, summary, paired, pd.DataFrame(generalization)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--days', type=int, default=120)
    ap.add_argument('--K', type=int, default=AGENT.default_K)
    ap.add_argument('--world', choices=['exp', 'power', 'both'], default='both')
    ap.add_argument('--include-dp', action='store_true',
                    help='slow optional known-dynamics single-arm Whittle reference')
    ap.add_argument('--recall-threshold', type=float, default=.5)
    ap.add_argument('--success-share', type=float, default=.8)
    args = ap.parse_args()
    episodes, curves, summary, paired, generalization = run_benchmark(
        args.seeds, args.days, args.K, args.world, args.include_dp,
        args.recall_threshold, args.success_share)
    out = Path(ROOT) / 'results'
    out.mkdir(exist_ok=True)
    for filename, frame in [('evaluation_episodes.csv', episodes),
                            ('evaluation_retention_trajectories.csv', curves),
                            ('evaluation_summary.csv', summary),
                            ('evaluation_paired.csv', paired),
                            ('evaluation_generalization.csv', generalization)]:
        frame.to_csv(out / filename, index=False)
    spec = {'simulated': True, 'world': args.world, 'seeds': args.seeds,
            'days': args.days, 'K': args.K,
            'success_definition': f'at end, at least {args.success_share:.0%} topics have '
                                  f'true model recall >= {args.recall_threshold}',
            'episodic_return_definition': 'sum of daily mean pre-review true recall',
            'final_ARP_definition': 'mean next-day post-horizon true recall over all topics',
            'hindsight_gap_definition': 'best observed deployable policy episodic return '
                                        'in same simulated seed minus policy return (not optimal-policy regret)',
            'evaluation_not_measured': ['QWI training convergence', 'training sample efficiency',
                                        'DRL-SRS or other neural RL baseline',
                                        'human retention outcomes'],
            'indexability_warning': 'Whittle index may be non-indexable for some states; '
                                    'DP reference is not multi-arm optimal'}
    (out / 'evaluation_protocol.json').write_text(json.dumps(spec, indent=2))
    print('\nSIMULATED evaluation (not real learners)\n')
    cols = ['world', 'policy', 'mean_retention', 'mean_final_ARP',
            'success_rate', 'mean_reviews_per_day', 'quiz_mae']
    print(summary[cols].round(4).to_string(index=False))
    print('\nAll CSVs and evaluation_protocol.json saved in', out)
    print('WARNING: Oracle policies are nondeployable diagnostics. Training convergence'
          ' and sample efficiency were NOT measured. Existing baseline CSVs are not overwritten.')


if __name__ == '__main__':
    main()
