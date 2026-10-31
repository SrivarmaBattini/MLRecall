"""
Test the quiz-to-state rules (they are version-1 heuristics, not facts about memory).

Part 1  P(quiz counts as 'success') as a function of the learner's true recall p, for
        several pass marks. Shows how noisy a 5-question MCQ quiz is (slips, 25% guessing).
Part 2  How the first-quiz level bands relate to hidden comprehension.
Part 3  Calibration: the app's predicted recall (before each review quiz, using only the
        past) pushed through the measurement model, against what the quiz then showed,
        in a simulated world. 'expected_*' columns are what the model predicts the quiz
        will show; 'observed_*' are what it showed.

    python -m rl.validate_rules
    python -m rl.validate_rules --world power     # forgetting curve the model does NOT assume
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binom

from config import RULES, ROOT
from rl.memory_model import (apply_quiz, estimate_speed, half_life, recall_probability, first_quiz_level,
                             question_correct_prob, success_probability)
from rl.simulate_dataset import generate

SLIP, GUESS = 0.1, 0.25


def part1():
    n = RULES.n_questions
    rows = []
    for p in np.round(np.arange(0.1, 1.01, 0.1), 1):
        q = p * (1 - SLIP) + (1 - p) * GUESS
        row = {"true_recall": p}
        for t in (3, 4, 5):
            row[f"P(>={t}/{n})"] = 1 - binom.cdf(t - 1, n, q)
        rows.append(row)
    return pd.DataFrame(rows)


def part2():
    n = RULES.n_questions
    rows = []
    for c in (0.2, 0.4, 0.6, 0.8, 0.95):
        q = c * (1 - SLIP) + (1 - c) * GUESS
        pm = binom.pmf(np.arange(n + 1), n, q)
        lv = np.zeros(RULES.L)
        for k in range(n + 1):
            lv[first_quiz_level(k)] += pm[k]
        rows.append({"comprehension": c, **{f"level_{i}": lv[i] for i in range(RULES.L)}})
    return pd.DataFrame(rows)


def part3(world):
    learners, topics, _, attempts, _ = generate(200, days=150, world=world, seed=21)
    betas = dict(zip(topics["topic_id"], topics["base_half_life_days"]))
    recs = []
    for lid, g in attempts.sort_values("day").groupby("learner_id"):
        states, reviews = {}, []
        for r in g.itertuples():
            st = states.get(r.topic_id)
            if st is not None and r.day - st["last_day"] >= 1:
                s_est = estimate_speed(reviews[-60:])
                p_hat = float(recall_probability(r.day - st["last_day"], half_life(betas[r.topic_id], s_est, st["k"])))
                recs.append({"p_hat": p_hat, "success": int(r.n_correct >= RULES.review_success_min_correct),
                             "accuracy": r.n_correct / r.n_questions,
                             "exp_accuracy": float(question_correct_prob(p_hat)),
                             "exp_success": float(success_probability(p_hat))})
            new, row, kind = apply_quiz(st, int(r.day), int(r.n_correct))
            states[r.topic_id] = new
            if row is not None:
                reviews.append((betas[r.topic_id],) + row)
    df = pd.DataFrame(recs)
    df["bin"] = pd.cut(df["p_hat"], [0, .2, .4, .6, .8, 1.0], include_lowest=True)
    return df.groupby("bin", observed=True).agg(n=("p_hat", "size"), mean_predicted_recall=("p_hat", "mean"),
                                                expected_accuracy=("exp_accuracy", "mean"),
                                                observed_accuracy=("accuracy", "mean"),
                                                expected_success=("exp_success", "mean"),
                                                observed_success=("success", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["exp", "power"], default="exp")
    a = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    for name, df in (("part1_pass_mark_vs_true_recall", part1()), ("part2_first_quiz_level_bands", part2()),
                     (f"part3_calibration_{a.world}", part3(a.world))):
        print(f"\n== {name} ==")
        print(df.round(3).to_string(index=False))
        df.to_csv(ROOT / "results" / f"rules_{name}.csv", index=False)
    print("\nRead part 3 as: do the expected_* columns match the observed_* columns? A gap means the "
          "rule/threshold/measurement assumptions need revisiting (see docs/ASSUMPTIONS.md).")


if __name__ == "__main__":
    main()
