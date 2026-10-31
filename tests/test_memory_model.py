import numpy as np

from config import RULES
from rl.memory_model import (apply_quiz, estimate_speed, first_quiz_level, nearest_bucket,
                             review_update, bucket_grid, half_life, recall_probability,
                             question_correct_prob, success_probability)


def test_first_quiz_levels_follow_the_documented_bands():
    assert [first_quiz_level(n) for n in range(6)] == [0, 0, 1, 1, 2, 3]


def test_review_update_moves_level_and_clamps():
    assert review_update(1, 4) == (2, True)
    assert review_update(3, 5) == (3, True)
    assert review_update(2, 3) == (1, False)
    assert review_update(0, 0) == (0, False)


def test_apply_quiz_first_review_and_same_day_retake():
    state, row, kind = apply_quiz(None, 10, 4)
    assert kind == "first" and row is None and state["k"] == 2
    same, row, kind = apply_quiz(state, 10, 1)
    assert kind == "retake" and row is None and same == state       # unchanged
    new, row, kind = apply_quiz(state, 13, 5)
    assert kind == "review" and row == (2, 3, 5, 5) and new["k"] == 3 and new["n_reviews"] == 1


def test_half_life_and_recall():
    assert half_life(4.0, 1.0, 0) == 4.0
    assert half_life(4.0, 0.5, 2) == 8.0                              # 4 * 0.5 * g_2(=4)
    assert abs(float(recall_probability(4.0, 4.0)) - 0.5) < 1e-12


def test_no_reviews_gives_the_prior_mean():
    assert estimate_speed([]) == 1.0


def test_speed_estimate_separates_fast_and_slow_forgetters():
    rng = np.random.default_rng(0)
    for true_s, expect_above in ((2.0, 1.0), (0.5, None)):
        rows = []
        for _ in range(120):
            k, d, beta = int(rng.integers(0, 4)), int(rng.integers(1, 15)), 4.0
            p = 2.0 ** (-d / (beta * true_s * RULES.level_multipliers[k]))
            n_right = int(rng.binomial(5, float(question_correct_prob(p))))
            rows.append((beta, k, d, n_right, 5))
        est = estimate_speed(rows)
        if expect_above:
            assert est > 1.3
        else:
            assert est < 0.8


def test_bucket_mapping_is_log_nearest():
    grid = bucket_grid()
    assert nearest_bucket(grid[3]) == 3
    assert nearest_bucket(1e-3) == 0 and nearest_bucket(1e4) == len(grid) - 1


def test_measurement_model_is_monotone_and_bounded():
    p = np.linspace(0, 1, 11)
    q, s = question_correct_prob(p), success_probability(p)
    assert (np.diff(q) > 0).all() and (np.diff(s) > 0).all()
    assert abs(q[0] - RULES.quiz_guess) < 1e-12 and 0 < s[0] < s[-1] < 1
    assert s[-1] < 1.0     # even perfect recall can fail a 4-of-5 quiz through slips
