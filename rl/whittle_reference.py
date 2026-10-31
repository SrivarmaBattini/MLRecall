"""
whittle_reference.py - dynamic-programming Whittle index for one arm.

This is a VERIFICATION tool, not a second model: QWI claims to learn this index
without knowing the dynamics, so this is what rl/verify_agent.py checks it against.

For a passive subsidy lam, solve the single-arm MDP by value iteration; the index of
state s is the lam at which review and no-review are equally good in s.
The Whittle index is only meaningful if the arm is indexable (the set of states where
'no review' is optimal grows with lam). indexability_report() measures violations
numerically - it does not assume them away.
"""
import numpy as np

from config import AGENT
from rl.simulator import transitions


def _solve(p, ps, s_pass, s_succ, s_fail, lam, V0=None, tol=1e-9, max_iter=3000):
    beta = AGENT.beta
    V = np.zeros_like(p) if V0 is None else V0.copy()
    for _ in range(max_iter):
        q0 = p + lam + beta * V[s_pass]
        q1 = p + beta * (ps * V[s_succ] + (1 - ps) * V[s_fail])
        Vn = np.maximum(q0, q1)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    q0 = p + lam + beta * V[s_pass]
    q1 = p + beta * (ps * V[s_succ] + (1 - ps) * V[s_fail])
    return V, q1 - q0


def whittle_index_dp(b, n_grid=161, lo=None, hi=None):
    """Return (index per state, indexability report) for base half-life b."""
    lo = AGENT.qwi_lambda_lo if lo is None else lo
    hi = AGENT.qwi_lambda_hi if hi is None else hi
    p, ps, sp, ss, sf = transitions(b)
    grid = np.linspace(lo, hi, n_grid)
    diffs, V = [], None
    for lam in grid:
        V, diff = _solve(p, ps, sp, ss, sf, lam, V)
        diffs.append(diff)
    diffs = np.array(diffs)                       # (G, S): >0 means review preferred
    S = diffs.shape[1]
    W = np.empty(S)
    for s in range(S):
        col = diffs[:, s]
        idx = np.where(col <= 0)[0]
        if len(idx) == 0:
            W[s] = hi                             # index above the grid
        elif idx[0] == 0:
            W[s] = lo                             # index below the grid
        else:
            g0, g1 = idx[0] - 1, idx[0]
            W[s] = grid[g0] + (grid[g1] - grid[g0]) * col[g0] / (col[g0] - col[g1])
    passive = diffs <= 1e-9
    violating = int(np.sum([np.any(passive[:-1, s] & ~passive[1:, s]) for s in range(S)]))
    report = {"states": S, "violating_states": violating,
              "violation_share": violating / S,
              "at_upper_bound": int(np.sum(W >= hi - 1e-9)),
              "at_lower_bound": int(np.sum(W <= lo + 1e-9))}
    return W, report
