"""
Verify the trained QWI agent. Run after `python -m rl.train_agent`.

    python -m rl.verify_agent                 # checks 1-4
    python -m rl.verify_agent --seed-check    # also trains a 2nd agent to test stability (slow)

1. Indexability   - is the per-arm problem indexable? (measured, not assumed)
2. Index accuracy - learned index vs the dynamic-programming (DP) Whittle index.
                    This tests the IMPLEMENTATION; the DP index uses the true dynamics, so
                    it is a reference, not necessarily the globally optimal policy.
3. Decisions      - in random 30-topic situations, how often does the learned index pick the
                    same K topics as the DP index, and how much DP-index value does it keep?
4. Retention      - absolute metrics when the learned agent schedules simulated learners.
Closeness to the DP index is a target, not a guarantee; if a number is poor, diagnose
(training length, step sizes, indexability, discretisation) before changing anything.
Reports are written to results/.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import AGENT, RULES, ROOT, INDEX_TABLE_PATH
from rl.memory_model import bucket_grid, nearest_bucket, success_probability
from rl.whittle_reference import whittle_index_dp

RES = ROOT / "results"


def load_table(path=INDEX_TABLE_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m rl.train_agent")
    z = np.load(path, allow_pickle=True)
    return z["W"], z["buckets"]


def dp_tables(buckets):
    tabs, reps = [], []
    for b in buckets:
        W, rep = whittle_index_dp(b)
        tabs.append(W.reshape(RULES.L, AGENT.D + 1))
        reps.append(rep)
    return np.array(tabs), reps


def accuracy_table(W, ref, buckets):
    rows = []
    for i, b in enumerate(buckets):
        a, r = W[i].ravel(), ref[i].ravel()
        rows.append({"bucket_b_days": round(float(b), 2), "mean_abs_error": float(np.mean(np.abs(a - r))),
                     "max_abs_error": float(np.max(np.abs(a - r))),
                     "spearman_all_states": float(spearmanr(a, r)[0])})
    return pd.DataFrame(rows)


def decision_agreement(W, ref, n_situations=2000, n_topics=30, seed=0):
    rng = np.random.default_rng(seed)
    K = AGENT.default_K
    overlap, kept = [], []
    for _ in range(n_situations):
        b = rng.integers(0, len(W), n_topics)
        k = rng.integers(0, RULES.L, n_topics)
        d = rng.integers(1, AGENT.D + 1, n_topics)
        learned, dp = W[b, k, d], ref[b, k, d]
        top_l, top_d = np.argsort(-learned)[:K], np.argsort(-dp)[:K]
        overlap.append(len(set(top_l) & set(top_d)) / K)
        kept.append(dp[top_l].sum() / max(dp[top_d].sum(), 1e-9))
    return {"top_k_overlap": float(np.mean(overlap)), "dp_index_value_kept": float(np.mean(kept))}


def retention_episode(W, seed, N=30, horizon=120, K=AGENT.default_K):

    """Schedule simulated learners with the learned index. The simulator knows the true
    state; a review succeeds with the same probability the agent was trained on."""

    from backend.content import load_topics
    rng = np.random.default_rng(seed)
    diffs = [t["difficulty"] for t in load_topics()]
    beta = np.array([RULES.base_half_life[int(rng.choice(diffs))] for _ in range(N)])
    speed = float(np.clip(np.exp(rng.normal(0, RULES.speed_prior_sigma)), *RULES.speed_bounds))
    b = beta * speed
    bucket = np.array([nearest_bucket(x) for x in b])
    g = np.asarray(RULES.level_multipliers, float)
    k, d = np.zeros(N, int), np.ones(N, int)
    ret, ok_budget = [], True
    for _ in range(horizon):
        p = np.power(2.0, -d / (b * g[k]))
        ret.append(p.mean())
        chosen = np.argsort(-W[bucket, k, d])[:K]
        ok_budget &= len(chosen) <= K
        rev = np.zeros(N, bool)
        rev[chosen] = True
        success = rev & (rng.random(N) < success_probability(p))
        failure = rev & ~success
        k[success] = np.minimum(k[success] + 1, RULES.L - 1)
        k[failure] = np.maximum(k[failure] - 1, 0)
        d = np.where(rev, 1, np.minimum(d + 1, AGENT.D))
    ret = np.array(ret)
    return {"mean": ret.mean(), "final": ret[-1], "below_half": float(np.mean(
        np.power(2.0, -d / (b * g[k])) < 0.5)), "budget_ok": ok_budget, "curve": ret}


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed-check", action="store_true")
    ap.add_argument("--seed-check-steps", type=int, default=AGENT.qwi_steps)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    (RES / "plots").mkdir(exist_ok=True)
    W, buckets = load_table()
    print("Building the DP reference index (a few seconds)...")
    ref, reps = dp_tables(buckets)
    report = {}

    print("\n[1] Indexability (share of states where the passive set is not monotone in the subsidy)")
    rep_df = pd.DataFrame(reps)
    rep_df.insert(0, "bucket_b_days", np.round(buckets, 2))
    print(rep_df.round(3).to_string(index=False))
    report["indexability"] = reps

    print("\n[2] Learned index vs DP index")
    acc = accuracy_table(W, ref, buckets)
    print(acc.round(3).to_string(index=False))
    report["index_accuracy"] = acc.to_dict("records")
    acc.to_csv(RES / "verify_index_accuracy.csv", index=False)

    print("\n[3] Decision agreement in random 30-topic situations")
    dec = decision_agreement(W, ref)
    print({k: round(v, 3) for k, v in dec.items()})
    report["decisions"] = dec

    print(f"\n[4] Retention when the learned agent schedules simulated learners ({a.episodes} episodes)")
    eps = [retention_episode(W, s) for s in range(a.episodes)]
    m, h = ci95([e["mean"] for e in eps])
    f, fh = ci95([e["final"] for e in eps])
    bl, blh = ci95([e["below_half"] for e in eps])
    print(f"  mean retention        {m:.3f} +/- {h:.3f}")
    print(f"  final-day retention   {f:.3f} +/- {fh:.3f}")
    print(f"  share of topics < 50% {bl:.3f} +/- {blh:.3f}")
    print(f"  budget respected      {all(e['budget_ok'] for e in eps)}")
    report["retention"] = {"mean": [m, h], "final": [f, fh], "below_half": [bl, blh],
                           "budget_ok": all(e["budget_ok"] for e in eps)}

    if a.seed_check:
        from rl.qwi import QWI
        print("\n[5] Seed stability: training a second agent...")
        W2 = QWI(seed=123).train(a.seed_check_steps, verbose=False).index_table()
        stab = decision_agreement(W, W2)
        print("  top-K overlap between two training seeds:", round(stab["top_k_overlap"], 3))
        report["seed_stability"] = stab

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(ref.ravel(), W.ravel(), s=4, alpha=0.35, color="#2F6FA8")
    top = float(max(ref.max(), W.max()))
    ax[0].plot([0, top], [0, top], color="#3F8F5F")
    ax[0].set(xlabel="DP Whittle index", ylabel="QWI learned index", title="Learned vs reference index")
    curves = np.array([e["curve"] for e in eps])
    mu, se = curves.mean(0), 1.96 * curves.std(0, ddof=1) / np.sqrt(len(curves))
    ax[1].plot(mu, color="#2F6FA8")
    ax[1].fill_between(range(len(mu)), mu - se, mu + se, color="#2F6FA8", alpha=0.2)
    ax[1].set(xlabel="Day", ylabel="Mean recall", ylim=(0, 1.02), title="Retention under the learned agent (simulated)")
    fig.tight_layout()
    fig.savefig(RES / "plots" / "verify_agent.png", dpi=140)
    (RES / "verify_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved report and plot to {RES}")


if __name__ == "__main__":
    main()
