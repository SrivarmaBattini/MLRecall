"""Tests for fair evaluation and the quiz-to-memory-state alignment."""
import numpy as np

from rl.evaluation_metrics import (quiz_prediction_metrics, mean_ci95,
                                   empirical_hindsight_gap, success_ci95)
from rl.evaluate_baselines import (make_scenario, evaluate_policy, topic_betas,
                                   run_benchmark)
from rl.memory_model import apply_quiz


def test_brier_matches_binary_per_question_formula():
    q = [0.8, 0.25]
    right = [4, 0]
    result = quiz_prediction_metrics(q, right, [5, 5])
    expected = ((4 * .2 ** 2 + 1 * .8 ** 2) / 5 + .25 ** 2) / 2
    assert abs(result['quiz_brier'] - expected) < 1e-10
    assert result['n_quizzes'] == 2


def test_hindsight_gap_nonnegative_and_zero_for_best():
    gaps = empirical_hindsight_gap({'a': 6.1, 'b': 7.5, 'c': 3.0})
    assert abs(gaps['a'] - 1.4) < 1e-12
    assert gaps['b'] == 0
    assert gaps['c'] > gaps['a']


def test_mean_ci_and_success_ci_refuse_fake_precision():
    assert np.isnan(mean_ci95([1])[2])
    assert success_ci95([1, 1, 0])[1] < 2 / 3 < success_ci95([1, 1, 0])[2]


def test_scenario_first_quiz_matches_shared_memory_rule():
    scenario = make_scenario(3, 5, topic_betas())
    for level, right in zip(scenario['initial_levels'], scenario['initial_scores']):
        assert int(level) == apply_quiz(None, 0, int(right))[0]['k']


def test_no_oracle_speed_in_primary_qwi(monkeypatch):
    import rl.evaluate_baselines as evaluator
    from config import AGENT, RULES
    scenario = make_scenario(7, 4, topic_betas())
    # Use a fake, known priority index so no dependence on saved model.
    W = np.random.default_rng(0).random((AGENT.n_buckets, RULES.L, AGENT.D + 1))
    first = evaluate_policy('QWI (online speed)', scenario, 'exp', 3, W)
    changed = dict(scenario)
    # With changing hidden speed, outcomes/retention CAN change, so instead
    # check selected policy's state sees prior (1.0) speed until reviews exist.
    observed = []
    original = evaluator._choose
    def spying(name, day, K, state, W, dp_W=None):
        if day == 1 and name == 'QWI (online speed)':
            observed.append(state['online_speed'])
        return original(name, day, K, state, W, dp_W)
    monkeypatch.setattr(evaluator, '_choose', spying)
    second = evaluate_policy('QWI (online speed)', changed, 'exp', 3, W)
    assert observed == [1.0]
    assert first['budget_respected'] == second['budget_respected'] == 1


def test_small_benchmark_paired_and_no_budget_overflow():
    episodes, curves, summary, paired, general = run_benchmark(seeds=2, days=4,
                                                                 K=3, world='both',
                                                                 verbose=False)
    assert not episodes.empty and not curves.empty
    assert episodes['budget_violations'].sum() == 0
    assert (episodes['reviews_per_day'] <= 3).all()
    assert summary['mean_retention'].between(0, 1).all()
    assert (episodes['hindsight_gap_to_best_deployable'].dropna() >= 0).all()
    assert not paired.empty and not general.empty
