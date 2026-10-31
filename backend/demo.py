"""Seed a SIMULATED demo learner so Today's Recall has something to show on day one."""
from datetime import date

import numpy as np

from backend import service
from backend.content import load_topics
from rl.memory_model import base_half_life
from rl.simulator import SimLearner

DEMO_NAME = "demo_learner"


def seed_demo_learner(conn, topics_to_study=14, history_days=35, seed=2024):
    old = conn.execute("SELECT id FROM users WHERE username=?", (DEMO_NAME,)).fetchone()
    if old:
        for tbl in ("reading_events", "quiz_attempts", "memory_state", "review_events", "recommendations"):
            conn.execute(f"DELETE FROM {tbl} WHERE user_id=?", (old["id"],))
        conn.execute("DELETE FROM users WHERE id=?", (old["id"],))
    conn.execute("INSERT INTO users(username, K, is_simulated) VALUES (?,?,1)", (DEMO_NAME, 5))
    uid = conn.execute("SELECT id FROM users WHERE username=?", (DEMO_NAME,)).fetchone()["id"]

    rng = np.random.default_rng(seed)
    topics = load_topics()
    sim = SimLearner(0.8, [base_half_life(t["difficulty"]) for t in topics], rng)
    today = date.today().toordinal()
    start, nxt = today - history_days, 0
    for day in range(start, today):
        if rng.random() > 0.75:
            continue                                                      # a skipped day
        rel = day - start
        if nxt < topics_to_study and rng.random() < 0.45:
            t = topics[nxt]
            service.record_reading(conn, uid, t["id"], day)
            sim.study(nxt, rel)
            qids = [q["id"] for q in service.pick_questions(conn, uid, t)]
            service.record_quiz(conn, uid, t["id"], day, sim.first_quiz(nxt), qids)
            nxt += 1
        pool = [i for i in range(nxt) if sim.last[i] < rel]
        for i in rng.choice(pool, size=min(3, len(pool)), replace=False) if pool else []:
            t = topics[int(i)]
            n_correct, _ = sim.quiz(int(i), rel)
            qids = [q["id"] for q in service.pick_questions(conn, uid, t)]
            service.record_quiz(conn, uid, t["id"], day, n_correct, qids)
    return conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
