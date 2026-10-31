"""
qwi.py - the project's single RL model: QWI, Q-learning for the Whittle index
(Robledo, Borkar, Ayesta, Avrachenkov), tabular, discounted criterion.

Idea. The Whittle index of state j is the passive subsidy lam_j at which "review" and
"don't review" are equally good in j. QWI learns it WITHOUT knowing the dynamics, with
two coupled timescales:
    fast:  Q^j(s,a) <- Q^j(s,a) + alpha * [ r + lam_j*(1-a) + beta*max_b Q^j(s',b) - Q^j(s,a) ]
    slow:  lam_j    <- lam_j    + c     * [ Q^j(j,1) - Q^j(j,0) ]
(one Q table per target state j, all updated from the same simulated transition).

Training runs inside the simulator (the "generative model" setting): at every step a
state-action pair is drawn uniformly, the simulator returns the next state and reward.
All half-life buckets are trained in parallel. The output is an index table
W[bucket, level k, days d] saved to artifacts/index_table.npz.

Convergence to the true index is a TARGET, not a guarantee - rl/verify_agent.py measures
how close the learned index is to the dynamic-programming reference.
"""
import time
import numpy as np

from config import AGENT, RULES, INDEX_TABLE_PATH, POLICY_VERSION
from rl.memory_model import bucket_grid
from rl.simulator import transitions, n_states


class QWI:
    def __init__(self, seed=0, alpha_exp=0.6, c0=3.0, c_exp=0.8, batch=64):
        self.buckets = bucket_grid()
        self.B, self.S = len(self.buckets), n_states()
        self.rng = np.random.default_rng(seed)
        self.alpha_exp, self.c0, self.c_exp = alpha_exp, c0, c_exp
        self.batch = min(batch, 2 * self.S)
        tr = [transitions(b) for b in self.buckets]
        self.P = np.array([t[0] for t in tr])           # (B,S) recall = reward
        self.PS = np.array([t[1] for t in tr])          # (B,S) probability a review succeeds
        self.SP = np.array([t[2] for t in tr])
        self.SS = np.array([t[3] for t in tr])
        self.SF = np.array([t[4] for t in tr])
        # Q[bucket, state s, action a, target state j]
        self.Q = np.zeros((self.B, self.S, 2, self.S))
        self.lam = np.full((self.B, self.S), 0.25 * 0.5 * (AGENT.qwi_lambda_lo + AGENT.qwi_lambda_hi))
        self.n_pair = np.zeros((self.B, self.S, 2))
        self.n_state = np.zeros((self.B, self.S))

    def train(self, steps=None, verbose=True):
        """steps = simulated transitions PER BUCKET (split into mini-batches of distinct
        state-action pairs, so no two samples in a batch overwrite each other)."""
        steps = steps or AGENT.qwi_steps
        B, S, M, beta = self.B, self.S, self.batch, AGENT.beta
        bi = np.arange(B)[:, None]
        lo, hi = AGENT.qwi_lambda_lo, AGENT.qwi_lambda_hi
        iters = max(1, steps // M)
        t0 = time.time()
        for n in range(iters):
            keys = self.rng.random((B, 2 * S))
            sel = np.argpartition(keys, M - 1, axis=1)[:, :M]           # distinct (s,a) per bucket
            s, a = sel // 2, sel % 2
            p = self.P[bi, s]
            ok = self.rng.random((B, M)) < self.PS[bi, s]
            s2 = np.where(a == 0, self.SP[bi, s], np.where(ok, self.SS[bi, s], self.SF[bi, s]))
            # fast timescale: each target state j learns with its own subsidy lam_j
            self.n_pair[bi, s, a] += 1
            alpha = (1.0 + self.n_pair[bi, s, a]) ** (-self.alpha_exp)          # (B,M)
            q_next = self.Q[bi, s2].max(axis=2)                                  # (B,M,S)
            target = p[..., None] + self.lam[:, None, :] * (1 - a)[..., None] + beta * q_next
            cur = self.Q[bi, s, a]                                               # (B,M,S)
            self.Q[bi, s, a] = cur + alpha[..., None] * (target - cur)
            # slow timescale: move lam_s toward indifference at state s
            self.n_state[bi, s] += 1
            c = self.c0 * self.n_state[bi, s] ** (-self.c_exp)
            gap = self.Q[bi, s, 1, s] - self.Q[bi, s, 0, s]
            self.lam[bi, s] = np.clip(self.lam[bi, s] + c * gap, lo, hi)
            if verbose and (n + 1) % max(1, iters // 10) == 0:
                print(f"  QWI {100 * (n + 1) // iters:3d}%  ({time.time() - t0:.0f}s)")
        return self

    def index_table(self):
        """W[bucket, k, d]"""
        return self.lam.reshape(self.B, RULES.L, AGENT.D + 1).copy()

    def save(self, path=INDEX_TABLE_PATH, meta=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.index_table(), buckets=self.buckets,
                 policy_version=POLICY_VERSION, rules_version=RULES.version,
                 D=AGENT.D, L=RULES.L)
        return path
