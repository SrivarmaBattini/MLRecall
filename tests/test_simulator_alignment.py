"""The synthetic learner must obey the same quiz-to-level rules as the API."""
import numpy as np
import pytest

from config import RULES
from rl.memory_model import apply_quiz
from rl.simulator import SimLearner


@pytest.mark.parametrize("n_correct,expected_level", [
    (0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 3),
])
def test_first_quiz_sets_same_level_as_application(monkeypatch, n_correct, expected_level):
    monkeypatch.setattr("rl.simulator.simulate_quiz", lambda *args, **kwargs: n_correct)
    sim = SimLearner(speed=1.0, betas=[4.0], rng=np.random.default_rng(1))
    sim.study(0, day=10)
    assert sim.first_quiz(0) == n_correct
    assert sim.k[0] == expected_level
    assert sim.n_reviews[0] == 0
    assert sim.last[0] == 10


def test_review_and_same_day_retake_match_application(monkeypatch):
    score = [4, 3, 5, 0]
    monkeypatch.setattr("rl.simulator.simulate_quiz", lambda *args, **kwargs: score.pop(0))
    sim = SimLearner(speed=1.0, betas=[4.0], rng=np.random.default_rng(1))
    sim.study(0, day=10)
    sim.first_quiz(0)  # 4 correct -> level 2
    assert int(sim.k[0]) == 2

    # 3 correct on day 13: level 2 -> 1
    sim.quiz(0, day=13)
    assert (int(sim.k[0]), int(sim.last[0]), int(sim.n_reviews[0])) == (1, 13, 1)

    # 5 correct on SAME day: practice only, no state change
    sim.quiz(0, day=13)
    assert (int(sim.k[0]), int(sim.last[0]), int(sim.n_reviews[0])) == (1, 13, 1)

    # 0 correct on day 15: level 1 -> 0
    sim.quiz(0, day=15)
    assert (int(sim.k[0]), int(sim.last[0]), int(sim.n_reviews[0])) == (0, 15, 2)


def test_first_quiz_required_before_review():
    sim = SimLearner(speed=1.0, betas=[4.0], rng=np.random.default_rng(0))
    sim.study(0, day=0)
    with pytest.raises(ValueError, match="first quiz"):
        sim.quiz(0, day=1)


@pytest.mark.parametrize("invalid_n,invalid_correct", [(4, 4), (6, 5), (5, -1), (5, 6)])
def test_apply_quiz_rejects_invalid_counts(invalid_n, invalid_correct):
    with pytest.raises(ValueError):
        apply_quiz(None, 0, invalid_correct, invalid_n)
