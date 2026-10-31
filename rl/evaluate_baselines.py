"""
Compare the QWI scheduler with the simple baselines required by the project brief (section 10).

    python -m rl.evaluate_baselines                 # 30 simulated learners, exponential forgetting
    python -m rl.evaluate_baselines --world power   # real forgetting follows a power law instead
    python -m rl.evaluate_baselines --seeds 50
Everything here is SIMULATED. It shows behaviour under the model, not results for real students.
"""
import argparse
import numpy as np
import pandas as pd

from config import AGENT, RULES, ROOT, INDEX_TABLE_PATH
from backend.content import load_topics
from rl.memory_model import nearest_bucket, question_correct_prob
from rl.simulator import true_recall

N, HORIZON = 30, 120
G = np.asarray(RULES.level_multipliers, float)


def make_policies(W, K):
    def qwi(s):
        return np.argsort(-W[s["bucket"], s["k"], s["d"]])[:K]

    def random(s):
        return s["rng"].choice(N, K, replace=False)

    def fixed(s):                       # rotation: the topic quizzed longest ago goes first
        return np.argsort(-s["d"] + 1e-3 * s["rng"].random(N))[:K]

    def lowest_score(s):
        return np.argsort(s["score"] + 1e-3 * s["rng"].random(N))[:K]

    def forgetting_risk(s):             # the app's own estimate: exponential model, true half-life
        return np.argsort(2.0 ** (-s["d"] / s["h"]))[:K]

    return {"QWI (RL agent)": qwi, "Random": random, "Fixed schedule": fixed,
            "Lowest score first": lowest_score, "Highest forgetting risk first": forgetting_risk}


def run_episode(policy, seed, world, K):
    rng = np.random.default_rng(seed)
    diffs = [t["difficulty"] for t in load_topics()]
    beta = np.array([RULES.base_half_life[int(rng.choice(diffs))] for _ in range(N)])
    speed = float(np.clip(np.exp(rng.normal(0, RULES.speed_prior_sigma)), *RULES.speed_bounds))
    b = beta * speed
    bucket = np.array([nearest_bucket(x) for x in b])
    k, d = np.zeros(N, int), np.ones(N, int)
    score = rng.binomial(RULES.n_questions, question_correct_prob(true_recall(d, b * G[k], world)))
    ret, used, within_budget = [], 0, True
    for _ in range(HORIZON):
        h = b * G[k]
        p = true_recall(d, h, world)
        ret.append(p.mean())
        chosen = policy({"rng": rng, "k": k, "d": d, "h": h, "score": score, "bucket": bucket})
        within_budget &= len(chosen) <= K
        used += len(chosen)
        rev = np.zeros(N, bool)
        rev[chosen] = True
        right = rng.binomial(RULES.n_questions, question_correct_prob(p))
        up = rev & (right >= RULES.review_success_min_correct)
        down = rev & ~up
        k[up] = np.minimum(k[up] + 1, RULES.L - 1)
        k[down] = np.maximum(k[down] - 1, 0)
        score = np.where(rev, right, score)
        d = np.where(rev, 1, np.minimum(d + 1, AGENT.D))
    ret = np.array(ret)
    final_p = true_recall(d, b * G[k], world)
    return {"mean_retention": ret.mean(), "final_retention": ret[-1], "share_below_half": float((final_p < 0.5).mean()),
            "reviews_per_day": used / HORIZON, "within_budget": within_budget}


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--world", choices=["exp", "power"], default="exp")
    ap.add_argument("--K", type=int, default=AGENT.default_K)
    a = ap.parse_args()
    if not INDEX_TABLE_PATH.exists():
        raise SystemExit("Run first: python -m rl.train_agent")
    W = np.load(INDEX_TABLE_PATH, allow_pickle=True)["W"]
    pols = make_policies(W, a.K)
    runs = {name: [run_episode(f, s, a.world, a.K) for s in range(a.seeds)] for name, f in pols.items()}

    print(f"SIMULATED comparison: world={a.world}, N={N}, K={a.K}, {HORIZON} days, {a.seeds} learners (same for every policy)\n")
    rows = []
    qwi_mean = np.array([r["mean_retention"] for r in runs["QWI (RL agent)"]])
    for name, rs in runs.items():
        m = np.array([r["mean_retention"] for r in rs])
        diff = qwi_mean - m
        is_qwi = name == "QWI (RL agent)"
        rows.append({"policy": name, "mean_retention": m.mean(), "ci95": ci95(m),
                     "final_retention": np.mean([r["final_retention"] for r in rs]),
                     "share_below_half": np.mean([r["share_below_half"] for r in rs]),
                     "reviews_per_day": np.mean([r["reviews_per_day"] for r in rs]),
                     "budget_respected": all(r["within_budget"] for r in rs),
                     "qwi_minus_this_paired": np.nan if is_qwi else diff.mean(),
                     "paired_ci95": np.nan if is_qwi else ci95(diff),
                     "qwi_better_in_seeds": a.seeds if is_qwi else int((diff > 0).sum())})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"baseline_comparison_{a.world}.csv", index=False)
    print(f"\nSaved results/baseline_comparison_{a.world}.csv  (simulated data; not evidence about real learners)")


if __name__ == "__main__":
    main()