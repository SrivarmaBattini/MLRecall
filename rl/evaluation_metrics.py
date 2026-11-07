"""Metrics for the SmartRecall *simulated* scheduling benchmark.

The memory-prediction metrics are based on quiz *observations*, not the hidden
simulator recall probability.  Scheduling metrics use hidden recall truth and
are therefore simulation-only.  Never apply the latter to real learners.

Literature connection:
 - Settles & Meeder (2016): MAE and held-out prediction evaluation;
 - Xiao & Wang (2024): final average recall probability (ARP);
 - Tabibian et al. (2019): forgetting/retention under alternative schedules;
 - RMAB: cumulative reward, resource constraint, policy comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import binomtest, binom, rankdata


def mean_ci95(values):
    """Mean, sample std, 95% Student-t interval across independent learners."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float('nan'),) * 4
    m = float(a.mean())
    if a.size == 1:
        return m, float('nan'), float('nan'), float('nan')
    sd = float(a.std(ddof=1))
    half = float(student_t.ppf(.975, a.size - 1) * sd / np.sqrt(a.size))
    return m, sd, m - half, m + half


def success_ci95(successes):
    """Observed success rate and Wilson 95% interval."""
    vals = np.asarray(successes, dtype=int)
    if vals.size == 0:
        return (float('nan'),) * 3
    count = int(vals.sum())
    conf = binomtest(count, len(vals)).proportion_ci(confidence_level=.95, method='wilson')
    return count / len(vals), float(conf.low), float(conf.high)


def quiz_prediction_metrics(predicted_q, observed_correct, n_questions):
    """Proper scoring for 5-question binomial quizzes (per-question Brier/log-loss).

    Predictions refer to P(correct on one MCQ), not the latent probability
    of recalling the topic.  The observed quiz fraction is noisy, so MAE is
    an observational score rather than an estimate of true recall error.
    """
    q = np.clip(np.asarray(predicted_q, dtype=float), 1e-8, 1 - 1e-8)
    right = np.asarray(observed_correct, dtype=float)
    n = np.asarray(n_questions, dtype=float)
    if q.size == 0:
        return {'quiz_mae': float('nan'), 'quiz_brier': float('nan'),
                'quiz_log_loss': float('nan'), 'quiz_pass_auc': float('nan'),
                'quiz_pass_ece': float('nan'), 'n_quizzes': 0}
    if not (q.shape == right.shape == n.shape):
        raise ValueError('Predictions and quiz counts must have matching shape')
    if np.any((n <= 0) | (right < 0) | (right > n)):
        raise ValueError('Invalid quiz count')
    frac = right / n
    brier = (right * (1 - q) ** 2 + (n - right) * q ** 2) / n
    logloss = -(right * np.log(q) + (n - right) * np.log(1 - q)) / n
    # The threshold is fixed by the app's 4-of-5 review-success rule.
    # This AUC is for the *observed pass/fail quiz*, not for hidden recall.
    predicted_pass = binom.sf(3, n.astype(int), q)
    passed = (right >= 4).astype(int)
    positives = int(passed.sum())
    negatives = len(passed) - positives
    if positives and negatives:
        ranks = rankdata(predicted_pass, method='average')
        auc = float((ranks[passed == 1].sum() - positives * (positives + 1) / 2)
                    / (positives * negatives))
    else:
        auc = float('nan')
    bins = np.minimum((predicted_pass * 10).astype(int), 9)
    ece = float(sum((bins == b).mean() *
                    abs(passed[bins == b].mean() - predicted_pass[bins == b].mean())
                    for b in range(10) if np.any(bins == b)))
    return {'quiz_mae': float(np.mean(np.abs(frac - q))),
            'quiz_brier': float(np.mean(brier)),
            'quiz_log_loss': float(np.mean(logloss)),
            'quiz_pass_auc': auc, 'quiz_pass_ece': ece,
            'n_quizzes': int(q.size)}


def empirical_hindsight_gap(episodes):
    """Nonnegative regret *relative only to best measured deployable policy*.

    episodes: mapping from policy-name to episode-total reward for one matched
    learner seed. This is NOT regret to an optimal policy or a known oracle.
    """
    if not episodes:
        return {}
    best = max(float(x) for x in episodes.values())
    return {name: float(max(0.0, best - val)) for name, val in episodes.items()}
