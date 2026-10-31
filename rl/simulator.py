"""
simulator.py - the environment the agent is trained in, plus a simulated-learner
world used to generate the dataset and to validate the rules.

Two separate things live here:
  1. single-arm dynamics (transitions) used by QWI and the DP reference;
  2. SimWorld: simulated learners with hidden truth, noisy MCQ quizzes and skipped days.
Everything produced from this module is SIMULATED and must be labelled as such.
"""
import numpy as np

from config import RULES, AGENT
from rl.memory_model import success_probability


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
    """One simulated learner with hidden speed and hidden per-topic latent level."""

    def __init__(self, speed, betas, rng, world="exp", slip=0.1, guess=0.25):
        self.speed, self.betas, self.rng = float(speed), np.asarray(betas, float), rng
        self.world, self.slip, self.guess = world, slip, guess
        n = len(self.betas)
        self.k = np.zeros(n, dtype=int)          # latent level
        self.last = np.full(n, -1, dtype=int)    # last day the topic was refreshed
        self.read = np.zeros(n, dtype=bool)
        self.comp = np.zeros(n)                  # hidden comprehension after reading

    def study(self, i, day):
        """Reading a topic. Hidden comprehension c ~ Beta(6,2) sets the starting level."""
        c = float(self.rng.beta(6, 2))
        self.comp[i] = c
        self.k[i] = 0 if c < 0.45 else 1 if c < 0.70 else 2 if c < 0.90 else 3
        self.last[i] = day
        self.read[i] = True

    def first_quiz(self, i):
        """Quiz taken right after reading: it measures comprehension, not long-term recall."""
        return simulate_quiz(self.comp[i], RULES.n_questions, self.rng, self.slip, self.guess)

    def recall(self, i, day):
        h = self.betas[i] * self.speed * RULES.level_multipliers[self.k[i]]
        return float(true_recall(day - self.last[i], h, self.world))

    def quiz(self, i, day):
        """Take a quiz. Returns (n_correct, true_recall_before). Updates latent state."""
        p = self.recall(i, day)
        n_correct = simulate_quiz(p, RULES.n_questions, self.rng, self.slip, self.guess)
        ok = self.rng.random() < p                       # latent recall event
        self.k[i] = min(self.k[i] + 1, RULES.L - 1) if ok else max(self.k[i] - 1, 0)
        self.last[i] = day
        return n_correct, p
