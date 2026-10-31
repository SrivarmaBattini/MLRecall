"""
Generate the SIMULATED learner dataset (ML topics only).

    python -m rl.simulate_dataset                      # 200 learners, 120 days
    python -m rl.simulate_dataset --learners 500 --days 180 --world power

Learners follow the curriculum, read a topic, take its first quiz, and do random
review quizzes on skipped-or-attended days. The behaviour policy does NOT use the agent.
Hidden ground truth (true speed, true recall) is written to separate files that are only
meant for validating the estimator - never feed them to the agent.
Every file carries 'simulated' in its name. Never present this data as real students.
"""
import argparse
import numpy as np
import pandas as pd

from config import DATA_DIR, RULES, AGENT
from backend.content import load_topics
from rl.memory_model import base_half_life
from rl.simulator import SimLearner


def generate(n_learners=200, days=120, K=AGENT.default_K, participation=0.7,
             new_topic_prob=0.4, world="exp", seed=7):
    rng = np.random.default_rng(seed)
    topics = load_topics()
    betas = [base_half_life(t["difficulty"]) for t in topics]
    reading, attempts, truth_attempts, learners = [], [], [], []
    for j in range(n_learners):
        speed = float(np.clip(np.exp(rng.normal(0.0, RULES.speed_prior_sigma)), *RULES.speed_bounds))
        learners.append({"learner_id": j, "simulated": 1, "world": world, "true_speed": speed})
        sim = SimLearner(speed, betas, rng, world=world)
        next_topic = 0
        for day in range(days):
            if rng.random() > participation:
                continue                                    # skipped day
            if next_topic < len(topics) and rng.random() < new_topic_prob:
                i = next_topic
                next_topic += 1
                sim.study(i, day)
                reading.append({"learner_id": j, "topic_id": topics[i]["id"], "day": day})
                attempts.append({"learner_id": j, "topic_id": topics[i]["id"], "day": day,
                                 "kind": "first", "n_correct": sim.first_quiz(i),
                                 "n_questions": RULES.n_questions})
            pool = [i for i in range(next_topic) if sim.last[i] < day]
            if pool:
                for i in rng.choice(pool, size=min(K, len(pool)), replace=False):
                    n_correct, p = sim.quiz(int(i), day)
                    attempts.append({"learner_id": j, "topic_id": topics[i]["id"], "day": day,
                                     "kind": "review", "n_correct": n_correct,
                                     "n_questions": RULES.n_questions})
                    truth_attempts.append({"learner_id": j, "topic_id": topics[i]["id"],
                                           "day": day, "true_recall_before": p})
    topic_rows = [{"topic_id": t["id"], "title": t["title"], "group": t["group"],
                   "difficulty": t["difficulty"], "base_half_life_days": b}
                  for t, b in zip(topics, betas)]
    return (pd.DataFrame(learners), pd.DataFrame(topic_rows), pd.DataFrame(reading),
            pd.DataFrame(attempts), pd.DataFrame(truth_attempts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", type=int, default=200)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--world", choices=["exp", "power"], default="exp")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = DATA_DIR / "simulated"
    out.mkdir(parents=True, exist_ok=True)
    learners, topics, reading, attempts, truth = generate(a.learners, a.days, world=a.world, seed=a.seed)
    learners.drop(columns=["true_speed"]).to_csv(out / "simulated_learners.csv", index=False)
    topics.to_csv(out / "topics.csv", index=False)
    reading.to_csv(out / "simulated_reading_events.csv", index=False)
    attempts.to_csv(out / "simulated_quiz_attempts.csv", index=False)
    learners[["learner_id", "true_speed"]].to_csv(out / "HIDDEN_truth_learner_speed.csv", index=False)
    truth.to_csv(out / "HIDDEN_truth_recall_before_quiz.csv", index=False)
    print(f"Wrote {len(learners)} simulated learners, {len(attempts)} quiz attempts to {out}")


if __name__ == "__main__":
    main()
