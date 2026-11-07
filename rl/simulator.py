"""
simulator.py - the environment the agent is trained in, plus a simulated-learner
world used to generate the dataset and to validate the rules.

Two separate things live here:
  1. single-arm dynamics (transitions) used by QWI and the DP reference;
  2. SimLearner: simulated learners with hidden speed, noisy MCQ quizzes and skipped days.
Everything produced from this module is SIMULATED and must be labelled as such.
"""
import numpy as np

from config import RULES, AGENT
from rl.memory_model import apply_quiz, success_probability


# ---------- 1. single-arm dynamics (state = level k and days-since-quiz d) ----------
def n_states() -> int:
    return RULES.L * (AGENT.D + 1)


def state_index(k, d):
    return np.asarray(k) * (AGENT.D + 1) + np.asarray(d)


def state_kd(s):
    s = np.asarray(s)
    return s // (AGENT.D + 1), s % (AGENT.D + 1)


def transitions(b: float):

    """For base half-life b: recall p[s] (the reward), the probability psucc[s] that 
    a review quiz counts as a success under the app's rules (it drives level-ups), 
    and next-state tables."""
    
    L, D = RULES.L, AGENT.D
    g = np.asarray(RULES.level_multipliers, dtype=float)
    s = np.arange(n_states())
    k, d = state_kd(s)
    p = np.power(2.0, -d / (b * g[k]))
    psucc = success_probability(p)
    s_pass = state_index(k, np.minimum(d + 1, D))
    s_succ = state_index(np.minimum(k + 1, L - 1), 1)
    s_fail = state_index(np.maximum(k - 1, 0), 1)
    return p, psucc, s_pass, s_succ, s_fail


# ---------- 2. simulated learners ----------
def true_recall(d, h, world="exp", omega=0.5):
    
    """Probability of remembering. 'exp' matches the agent's model; 'power' is a
    deliberately different forgetting curve used for robustness checks."""

    d = np.asarray(d, dtype=float)
    if world == "exp":
        return np.power(2.0, -d / h)
    return np.power(1.0 + (d / h) * (2.0 ** (1.0 / omega) - 1.0), -omega)


def simulate_quiz(p, n_questions, rng, slip=0.1, guess=0.25):
    """Noisy MCQ quiz: each question is right with prob p*(1-slip) + (1-p)*guess."""
    q = p * (1.0 - slip) + (1.0 - p) * guess
    return int(rng.binomial(n_questions, np.clip(q, 0.0, 1.0)))


class SimLearner:
    """One simulated learner with hidden speed and quiz-derived per-topic memory levels."""

    def __init__(self, speed, betas, rng, world="exp", slip=0.1, guess=0.25):
        self.speed, self.betas, self.rng = float(speed), np.asarray(betas, float), rng
        self.world, self.slip, self.guess = world, slip, guess
        n = len(self.betas)
        self.k = np.zeros(n, dtype=int)          # quiz-derived memory level
        self.last = np.full(n, -1, dtype=int)    # most recent first/review quiz day
        self.read = np.zeros(n, dtype=bool)
        self.quizzed = np.zeros(n, dtype=bool)   # first quiz has established a level
        self.n_reviews = np.zeros(n, dtype=int)
        self.comp = np.zeros(n)                  # hidden comprehension after reading

    def study(self, i, day):
        """Read a topic; comprehension generates the first quiz, not the memory level."""
        c = float(self.rng.beta(6, 2))
        self.comp[i] = c
        self.k[i] = 0                            # placeholder until the first quiz
        self.last[i] = day
        self.read[i] = True
        self.quizzed[i] = False
        self.n_reviews[i] = 0

    def first_quiz(self, i):
        """Sample the initial quiz and initialize mastery using the application's rules."""
        if not self.read[i]:
            raise ValueError("Read the topic before taking its first quiz.")
        if self.quizzed[i]:
            raise ValueError("The first quiz has already been taken for this topic.")
        n_correct = simulate_quiz(self.comp[i], RULES.n_questions, self.rng,
                                  self.slip, self.guess)
        state, _, kind = apply_quiz(None, int(self.last[i]), n_correct,
                                    RULES.n_questions)
        assert kind == "first"
        self.k[i] = state["k"]
        self.n_reviews[i] = state["n_reviews"]
        self.quizzed[i] = True
        return n_correct

    def recall(self, i, day):
        if not self.quizzed[i]:
            raise ValueError("Take the first quiz before measuring review recall.")
        h = self.betas[i] * self.speed * RULES.level_multipliers[self.k[i]]
        return float(true_recall(day - self.last[i], h, self.world))

    def quiz(self, i, day):
        """Sample the review score; let apply_quiz decide the state change."""
        if not self.quizzed[i]:
            raise ValueError("Take the first quiz before reviewing a topic.")
        p = self.recall(i, day)
        n_correct = simulate_quiz(p, RULES.n_questions, self.rng,
                                  self.slip, self.guess)
        state = {"k": int(self.k[i]), "last_day": int(self.last[i]),
                 "n_reviews": int(self.n_reviews[i])}
        next_state, _, _ = apply_quiz(state, day, n_correct, RULES.n_questions)
        self.k[i] = next_state["k"]
        self.last[i] = next_state["last_day"]
        self.n_reviews[i] = next_state["n_reviews"]
        return n_correct, p
