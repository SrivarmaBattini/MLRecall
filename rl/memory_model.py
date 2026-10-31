"""
memory_model.py - the memory model and the quiz-to-state rules (version 1).

Per learner j and topic i:
    half-life   h = beta_i * s_j * g_k        beta_i: topic base (from difficulty)
                                              s_j   : ONE speed factor per learner
                                              g_k   : fixed level multiplier
    recall      p = 2 ** (-d / h)             d: days since the last quiz

Who is allowed to change what (so the same learning effect is never counted twice):
    * the level k changes ONLY through the quiz rules below (deterministic);
    * the speed s_j changes ONLY through the likelihood in estimate_speed();
    * beta_i and g_k are fixed.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binom

from config import RULES, AGENT


def base_half_life(difficulty: int) -> float:
    return float(RULES.base_half_life[int(difficulty)])


def half_life(beta_i, speed, k):
    return beta_i * speed * np.asarray(RULES.level_multipliers, dtype=float)[np.asarray(k, dtype=int)]


def recall_probability(d, h):
    return np.power(2.0, -np.asarray(d, dtype=float) / h)


def question_correct_prob(p):
    """Measurement model: chance that ONE question is answered correctly when the learner's
    true recall of the topic is p (slips on known items, lucky guesses on unknown ones)."""
    p = np.asarray(p, dtype=float)
    return p * (1.0 - RULES.quiz_slip) + (1.0 - p) * RULES.quiz_guess


def success_probability(p, n_questions=None, min_correct=None):
    """Chance that a quiz counts as a review SUCCESS (>= min_correct right out of n).
    This is the probability the app's own rules use to move a topic up a level, so the agent
    is trained on the same transition probability."""
    n = RULES.n_questions if n_questions is None else n_questions
    t = RULES.review_success_min_correct if min_correct is None else min_correct
    return binom.sf(t - 1, n, question_correct_prob(p))


def first_quiz_level(n_correct: int) -> int:
    """Starting level from the first quiz after reading (assumption, see config)."""
    level = 0
    for k, m in enumerate(RULES.first_quiz_min_correct):
        if n_correct >= m:
            level = k
    return level


def review_update(k: int, n_correct: int):
    """Level update after a review quiz. Returns (new_level, success)."""
    success = n_correct >= RULES.review_success_min_correct
    if success:
        return min(k + 1, RULES.L - 1), True
    return max(k - 1, 0), False


def bucket_grid() -> np.ndarray:
    """Log-spaced base half-life buckets the agent is trained on."""
    return np.geomspace(AGENT.b_min, AGENT.b_max, AGENT.n_buckets)


def nearest_bucket(b) -> int:
    grid = np.log(bucket_grid())
    return int(np.argmin(np.abs(grid - np.log(max(float(b), 1e-6)))))


def estimate_speed(reviews) -> float:
    """
    MAP estimate of the learner speed s_j from the quiz COUNTS.

    reviews: iterable of (beta_i, k_before, d, n_correct, n_questions) - one row per REVIEW
    quiz with d >= 1. k_before is the level BEFORE that review, used as a known covariate.
    The first quiz on a topic and same-day retakes are not passed in.

        p = 2 ** (-d / (beta_i * s * g_k))               true recall
        q = p*(1-slip) + (1-p)*guess                     per-question chance of a right answer
        n_correct ~ Binomial(n_questions, q)
        maximise  sum[ n_correct*log q + (n - n_correct)*log(1-q) ]  -  (log s)^2 / (2 sigma^2)

    Using the counts (not a pass/fail threshold) keeps the likelihood consistent with how the
    data are produced, so the estimate does not drift as reviews accumulate.
    """
    rows = list(reviews)
    if not rows:
        return 1.0
    beta = np.array([r[0] for r in rows], dtype=float)
    k = np.array([r[1] for r in rows], dtype=int)
    d = np.array([r[2] for r in rows], dtype=float)
    right = np.array([r[3] for r in rows], dtype=float)
    n = np.array([r[4] for r in rows], dtype=float)
    g = np.asarray(RULES.level_multipliers, dtype=float)[k]
    sigma = RULES.speed_prior_sigma
    lo, hi = np.log(RULES.speed_bounds[0]), np.log(RULES.speed_bounds[1])

    def nll(log_s):
        p = np.power(2.0, -d / (beta * np.exp(log_s) * g))
        q = np.clip(question_correct_prob(p), 1e-6, 1 - 1e-6)
        return -np.sum(right * np.log(q) + (n - right) * np.log(1 - q)) + log_s ** 2 / (2 * sigma ** 2)

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def apply_quiz(state, day: int, n_correct: int, n_questions: int = None):
    """
    The ONE place the quiz rules are applied (used by the app and by every script).

    state: None before the first quiz on a topic, else dict(k, last_day, n_reviews).
    Returns (new_state, review_row, kind):
      kind 'first'  - sets the starting level; no review row
      kind 'retake' - a same-day repeat: stored by the caller, state unchanged
      kind 'review' - level update; review_row = (k_before, d, n_correct, n_questions) feeds estimate_speed
    """
    if state is None:
        return {"k": first_quiz_level(n_correct), "last_day": day, "n_reviews": 0}, None, "first"
    d = day - state["last_day"]
    if d <= 0:
        return state, None, "retake"
    k2, success = review_update(state["k"], n_correct)
    new_state = {"k": k2, "last_day": day, "n_reviews": state["n_reviews"] + 1}
    nq = RULES.n_questions if n_questions is None else n_questions
    return new_state, (state["k"], d, int(n_correct), int(nq)), "review"
