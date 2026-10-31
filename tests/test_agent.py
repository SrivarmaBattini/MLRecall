import numpy as np

from config import AGENT, RULES
from rl.qwi import QWI
from rl.simulator import transitions, n_states
from rl.whittle_reference import whittle_index_dp


def test_transitions_are_consistent():
    p, ps, sp, ss, sf = transitions(4.0)
    S = n_states()
    assert p.shape == (S,) and ((p > 0) & (p <= 1)).all()
    assert ((ps > 0) & (ps < 1)).all() and (ps <= p + 0.2).all()
    assert sp.max() < S and ss.max() < S and sf.max() < S
    assert (ss % (AGENT.D + 1) == 1).all() and (sf % (AGENT.D + 1) == 1).all()   # reviews reset d


def test_dp_reference_is_indexable_and_in_range():
    W, rep = whittle_index_dp(4.0, n_grid=81)
    assert rep["violation_share"] <= 0.02
    assert np.isfinite(W).all() and W.min() >= AGENT.qwi_lambda_lo


def test_qwi_smoke_output_shape_and_finite():
    q = QWI(seed=0, batch=32).train(steps=3000, verbose=False)
    W = q.index_table()
    assert W.shape == (AGENT.n_buckets, RULES.L, AGENT.D + 1)
    assert np.isfinite(W).all()
