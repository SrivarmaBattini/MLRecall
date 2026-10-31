"""
Does the speed estimator recover the hidden learner speed without drifting?

Simulated learners (hidden speed) take quizzes; every attempt is replayed through the
SAME rules the app uses (apply_quiz). After n review quizzes we estimate speed and compare
with the truth. 'bias' is the mean of log(estimate/truth): it should stay near 0 as the
number of reviews grows. Run with --world power to test a mis-specified forgetting curve.

    python -m rl.validate_estimator
    python -m rl.validate_estimator --world power
"""
import argparse
import numpy as np
import pandas as pd

from config import RULES
from rl.memory_model import apply_quiz, estimate_speed
from rl.simulate_dataset import generate

CHECKPOINTS = (5, 10, 20, 40, 80)


def replay(attempts, betas):
    """Return per-learner list of review rows (beta, k_before, d, success) in time order."""
    rows = {}
    for (lid, tid), g in attempts.sort_values("day").groupby(["learner_id", "topic_id"]):
        state = None
        for r in g.itertuples():
            state, row, kind = apply_quiz(state, int(r.day), int(r.n_correct))
            if row is not None:
                rows.setdefault(lid, []).append((r.day, (betas[tid],) + row))
    return {lid: [x[1] for x in sorted(v, key=lambda t: t[0])] for lid, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["exp", "power"], default="exp")
    ap.add_argument("--learners", type=int, default=300)
    a = ap.parse_args()
    learners, topics, _, attempts, _ = generate(a.learners, days=150, world=a.world, seed=11)
    betas = dict(zip(topics["topic_id"], topics["base_half_life_days"]))
    truth = dict(zip(learners["learner_id"], learners["true_speed"]))
    reviews = replay(attempts, betas)
    out = []
    for n in CHECKPOINTS:
        err = [np.log(estimate_speed(rows[:n]) / truth[lid]) for lid, rows in reviews.items() if len(rows) >= n]
        if err:
            out.append({"reviews_used": n, "learners": len(err), "bias_log": np.mean(err),
                        "rmse_log": float(np.sqrt(np.mean(np.square(err))))})
    df = pd.DataFrame(out)
    print(f"Estimator validation (world={a.world}); prior sigma={RULES.speed_prior_sigma}")
    print(df.round(3).to_string(index=False))
    from config import ROOT
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"estimator_validation_{a.world}.csv", index=False)


if __name__ == "__main__":
    main()
