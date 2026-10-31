"""Core app logic shared by the API and the demo seeder: record reading, record quizzes."""
import json
import random

from config import RULES
from backend import db
from backend.content import topics_by_id
from rl.memory_model import apply_quiz, estimate_speed, base_half_life


def record_reading(conn, user_id, topic_id, day):
    conn.execute("INSERT INTO reading_events(user_id, topic_id, day) VALUES (?,?,?)", (user_id, topic_id, day))


def has_read(conn, user_id, topic_id) -> bool:
    return conn.execute("SELECT 1 FROM reading_events WHERE user_id=? AND topic_id=? LIMIT 1",
                        (user_id, topic_id)).fetchone() is not None


def get_state(conn, user_id, topic_id):
    r = conn.execute("SELECT * FROM memory_state WHERE user_id=? AND topic_id=?", (user_id, topic_id)).fetchone()
    return None if r is None else {"k": r["k"], "last_day": r["last_quiz_day"], "n_reviews": r["n_reviews"]}


def pick_questions(conn, user_id, topic, n=RULES.n_questions):
    """Least-recently-used first, so repeated reviews do not repeat the same questions."""
    used = {}
    for r in conn.execute("SELECT question_ids FROM quiz_attempts WHERE user_id=? AND topic_id=? ORDER BY id",
                          (user_id, topic["id"])):
        for qid in json.loads(r["question_ids"] or "[]"):
            used[qid] = used.get(qid, 0) + 1
    pool = list(topic["questions"])
    random.shuffle(pool)
    pool.sort(key=lambda q: used.get(q["id"], 0))
    chosen = pool[:n]
    random.shuffle(chosen)
    return chosen


def record_quiz(conn, user_id, topic_id, day, n_correct, question_ids, answers=None):
    """Apply the quiz rules, store the attempt, update memory state and the speed estimate."""
    topic = topics_by_id()[topic_id]
    state = get_state(conn, user_id, topic_id)
    new_state, review_row, kind = apply_quiz(state, day, n_correct, len(question_ids))
    conn.execute(
        "INSERT INTO quiz_attempts(user_id,topic_id,day,kind,n_correct,n_questions,question_ids,answers,"
        "k_before,k_after,d_since_last) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, topic_id, day, kind, n_correct, len(question_ids), json.dumps(question_ids),
         json.dumps(answers) if answers is not None else None,
         None if state is None else state["k"], new_state["k"],
         None if state is None else day - state["last_day"]))
    if kind != "retake":
        first_day = day if state is None else conn.execute(
            "SELECT first_quiz_day FROM memory_state WHERE user_id=? AND topic_id=?", (user_id, topic_id)).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO memory_state(user_id,topic_id,k,last_quiz_day,first_quiz_day,n_reviews,"
            "last_n_correct,last_n_questions) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, topic_id, new_state["k"], day, first_day, new_state["n_reviews"], n_correct, len(question_ids)))
    if review_row is not None:
        k_before, d, right, n_q = review_row
        conn.execute("INSERT INTO review_events(user_id,topic_id,day,beta,k_before,d,n_correct,n_questions) "
                     "VALUES (?,?,?,?,?,?,?,?)",
                     (user_id, topic_id, day, base_half_life(topic["difficulty"]), k_before, d, right, n_q))
        rows = [(r["beta"], r["k_before"], r["d"], r["n_correct"], r["n_questions"]) for r in conn.execute(
            "SELECT beta,k_before,d,n_correct,n_questions FROM review_events WHERE user_id=? ORDER BY id", (user_id,))]
        conn.execute("UPDATE users SET speed_est=? WHERE id=?", (estimate_speed(rows), user_id))
    conn.execute("UPDATE recommendations SET completed=1, status='completed', finished_at=CURRENT_TIMESTAMP "
                 "WHERE user_id=? AND day=? AND topic_id=?", (user_id, day, topic_id))
    return {"kind": kind, "k_before": None if state is None else state["k"], "k_after": new_state["k"],
            "success": None if review_row is None else bool(n_correct >= RULES.review_success_min_correct)}

def start_review(conn, user_id, topic_id, day):
    """Open a focused revision session: store its start time and mark today's recommendation as opened."""
    cur = conn.execute("INSERT INTO review_attempts(user_id, topic_id, day) VALUES (?,?,?)", (user_id, topic_id, day))
    conn.execute("UPDATE recommendations SET status='opened', opened_at=CURRENT_TIMESTAMP "
                 "WHERE user_id=? AND day=? AND topic_id=? AND status='recommended'", (user_id, day, topic_id))
    return cur.lastrowid


def finish_review(conn, attempt_id, user_id, topic_id, n_correct, n_questions, kind, success):
    """Close a revision session once its quiz has been submitted. Returns False if the id is not valid."""
    row = conn.execute("SELECT id FROM review_attempts WHERE id=? AND user_id=? AND topic_id=? AND completed_at IS NULL",
                       (attempt_id, user_id, topic_id)).fetchone()
    if row is None:
        return False
    outcome = ("success" if success else "failure") if kind == "review" else "practice"
    conn.execute("UPDATE review_attempts SET completed_at=CURRENT_TIMESTAMP, n_correct=?, n_questions=?, outcome=? WHERE id=?",
                 (n_correct, n_questions, outcome, attempt_id))
    return True