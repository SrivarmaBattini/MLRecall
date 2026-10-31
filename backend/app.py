"""
MLRecall API (FastAPI). Run with:  python run.py
Local single-machine tool: users are identified by a username only (no passwords).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import ROOT, RULES, AGENT, POLICY_VERSION
from backend import auth, db, service
from backend.content import load_topics, topics_by_id, grouped, revision_summary_html
from backend.scheduler import scheduler
from rl.memory_model import base_half_life

FRONTEND = ROOT / "frontend"


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    load_topics()          # fail early if a content file is malformed
    yield


app = FastAPI(title="MLRecall", lifespan=lifespan)


# ---------- helpers ----------
def get_user(conn, authorization):
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "Log in first.")
    u = conn.execute("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token=?", (token,)).fetchone()
    if u is None:
        raise HTTPException(401, "Session expired. Log in again.")
    return u


def user_payload(u):
    day = db.app_day(u)
    return {"id": u["id"], "username": u["username"], "K": u["K"], "today": db.day_label(day),
            "day_offset": u["day_offset"], "is_simulated": bool(u["is_simulated"]),
            "speed_estimate": round(u["speed_est"], 2)}


def topic_status(conn, u, t):
    day = db.app_day(u)
    st = conn.execute("SELECT * FROM memory_state WHERE user_id=? AND topic_id=?", (u["id"], t["id"])).fetchone()
    read = service.has_read(conn, u["id"], t["id"])
    out = {"id": t["id"], "title": t["title"], "group": t["group"], "difficulty": t["difficulty"],
           "summary": t["summary"], "status": "not_started", "level": None, "last_score": None,
           "last_quiz": None, "days_since_quiz": None, "est_recall": None, "due_today": False,
           "n_reviews": 0}
    if st is not None:
        d = day - st["last_quiz_day"]
        out.update(status="quizzed", level=st["k"], last_score=f'{st["last_n_correct"]}/{st["last_n_questions"]}',
                   last_quiz=db.day_label(st["last_quiz_day"]), days_since_quiz=d,
                   n_reviews=st["n_reviews"],
                   est_recall=round(scheduler.est_recall(base_half_life(t["difficulty"]), u["speed_est"], st["k"], d), 3))
    elif read:
        out["status"] = "read"
    rec = conn.execute("SELECT status FROM recommendations WHERE user_id=? AND day=? AND topic_id=?",
                       (u["id"], day, t["id"])).fetchone()
    out["due_today"] = bool(rec is not None and rec["status"] in ("recommended", "opened"))
    if out["due_today"]:
        out["label"] = "Needs revision"
    elif out["n_reviews"] > 0 and out["days_since_quiz"] <= 2:
        out["label"] = "Recently reviewed"
    else:
        out["label"] = {"not_started": "Not started", "read": "Read", "quizzed": "Quiz attempted"}[out["status"]]
    return out


def reason_text(d, n_correct, n_q, est, score):
    plural = "" if d == 1 else "s"
    return (f"Last quiz {d} day{plural} ago: {n_correct}/{n_q} correct. "
            f"Estimated recall about {round(est * 100)}% (a model estimate). Priority {score:.1f}.")


def ensure_today(conn, u, force=False):
    """Create (or top up) today's review list. Completed items are kept."""
    day, K = db.app_day(u), u["K"]
    rows = conn.execute("SELECT * FROM recommendations WHERE user_id=? AND day=? ORDER BY rank", (u["id"], day)).fetchall()
    if rows and not force:
        return rows
    if not scheduler.available:
        raise HTTPException(503, "The agent has not been trained yet. Run: python -m rl.train_agent")
    keep = [r for r in rows if r["status"] in ("completed", "skipped")]       # skipped topics keep their slot
    conn.execute("DELETE FROM recommendations WHERE user_id=? AND day=? AND status IN ('recommended','opened')",
                 (u["id"], day))
    already = {r["topic_id"] for r in keep}
    items = []
    for st in conn.execute("SELECT * FROM memory_state WHERE user_id=?", (u["id"],)):
        d = day - st["last_quiz_day"]
        if d >= 1 and st["topic_id"] not in already:
            t = topics_by_id()[st["topic_id"]]
            items.append({"topic_id": st["topic_id"], "beta": base_half_life(t["difficulty"]), "speed": u["speed_est"],
                          "k": st["k"], "d": d, "n_correct": st["last_n_correct"], "n_q": st["last_n_questions"]})
    ranked = scheduler.rank(items, max(0, K - len(keep)))
    for i, it in enumerate(ranked, start=len(keep) + 1):
        conn.execute(
            "INSERT INTO recommendations(user_id,day,topic_id,rank,index_score,est_recall,reason,policy_version) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (u["id"], day, it["topic_id"], i, it["index_score"], it["est_recall"],
             reason_text(it["d"], it["n_correct"], it["n_q"], it["est_recall"], it["index_score"]), POLICY_VERSION))
    conn.commit()
    return conn.execute("SELECT * FROM recommendations WHERE user_id=? AND day=? ORDER BY rank", (u["id"], day)).fetchall()


# ---------- schemas ----------
class LoginIn(BaseModel):
    username: str
    password: str


class QuizIn(BaseModel):
    question_ids: list[str]
    answers: list[int | None]
    attempt_id: int | None = None          # the focused revision session this quiz completes


class SettingsIn(BaseModel):
    K: int


class AdvanceIn(BaseModel):
    days: int = 1


# ---------- auth / settings ----------
@app.post("/api/login")
def login(body: LoginIn):
    """Create the account on first use, otherwise check the password. Returns a session token."""
    name = body.username.strip()[:40]
    if not name or len(body.password) < 6:
        raise HTTPException(400, "Enter a username and a password of at least 6 characters.")
    with db.connect() as conn:
        u = conn.execute("SELECT * FROM users WHERE username=?", (name,)).fetchone()
        if u is None:
            salt, digest = auth.hash_password(body.password)
            conn.execute("INSERT INTO users(username, K, password_salt, password_hash) VALUES (?,?,?,?)",
                         (name, AGENT.default_K, salt, digest))
            u = conn.execute("SELECT * FROM users WHERE username=?", (name,)).fetchone()
        elif u["password_hash"] is None or not auth.verify_password(body.password, u["password_salt"], u["password_hash"]):
            raise HTTPException(401, "Wrong username or password.")
        token = auth.new_token()
        conn.execute("INSERT INTO sessions(token, user_id) VALUES (?,?)", (token, u["id"]))
        return {**user_payload(u), "token": token}


@app.post("/api/logout")
def logout(authorization: str | None = Header(None)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    with db.connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    return {"ok": True}


@app.get("/api/me")
def me(authorization: str | None = Header(None)):
    with db.connect() as conn:
        return user_payload(get_user(conn, authorization))


@app.post("/api/settings")
def settings(body: SettingsIn, authorization: str | None = Header(None)):
    if not 1 <= body.K <= 15:
        raise HTTPException(400, "K must be between 1 and 15.")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        conn.execute("UPDATE users SET K=? WHERE id=?", (body.K, u["id"]))
        u = get_user(conn, authorization)
        if scheduler.available:
            ensure_today(conn, u, force=True)
        return user_payload(u)


# ---------- content ----------
@app.get("/api/topics")
def list_topics(authorization: str | None = Header(None)):
    with db.connect() as conn:
        u = get_user(conn, authorization)
        return [{"group": g["group"], "topics": [topic_status(conn, u, t) for t in g["topics"]]} for g in grouped()]


@app.get("/api/topics/{topic_id}")
def get_topic(topic_id: str, authorization: str | None = Header(None)):
    t = topics_by_id().get(topic_id)
    if t is None:
        raise HTTPException(404, "Unknown topic.")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        info = topic_status(conn, u, t)
        info.update(lesson_html=t["lesson_html"], n_questions_in_bank=len(t["questions"]),
                    prerequisites=[{"id": p, "title": topics_by_id()[p]["title"]} for p in t["prerequisites"]])
        return info


@app.post("/api/topics/{topic_id}/read")
def mark_read(topic_id: str, authorization: str | None = Header(None)):
    if topic_id not in topics_by_id():
        raise HTTPException(404, "Unknown topic.")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        if not service.has_read(conn, u["id"], topic_id):
            service.record_reading(conn, u["id"], topic_id, db.app_day(u))
        return topic_status(conn, u, topics_by_id()[topic_id])


# ---------- quizzes ----------
@app.get("/api/topics/{topic_id}/quiz")
def start_quiz(topic_id: str, authorization: str | None = Header(None)):
    t = topics_by_id().get(topic_id)
    if t is None:
        raise HTTPException(404, "Unknown topic.")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        if not service.has_read(conn, u["id"], topic_id):
            raise HTTPException(400, "Read the lesson and press 'Mark as read' before taking the quiz.")
        state = service.get_state(conn, u["id"], topic_id)
        kind = "first" if state is None else ("retake" if db.app_day(u) - state["last_day"] <= 0 else "review")
        qs = service.pick_questions(conn, u["id"], t)
        return {"kind": kind, "topic": t["title"],
                "questions": [{"id": q["id"], "stem": q["stem"], "options": q["options"]} for q in qs]}


@app.post("/api/topics/{topic_id}/quiz")
def submit_quiz(topic_id: str, body: QuizIn, authorization: str | None = Header(None)):
    t = topics_by_id().get(topic_id)
    if t is None:
        raise HTTPException(404, "Unknown topic.")
    bank = {q["id"]: q for q in t["questions"]}
    if (len(body.question_ids) != len(body.answers) or len(set(body.question_ids)) != len(body.question_ids)
            or not body.question_ids or any(q not in bank for q in body.question_ids)):
        raise HTTPException(400, "Invalid quiz submission.")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        if not service.has_read(conn, u["id"], topic_id):
            raise HTTPException(400, "Mark the lesson as read first.")
        review, n_correct = [], 0
        for qid, ans in zip(body.question_ids, body.answers):
            q = bank[qid]
            ok = ans == q["answer"]
            n_correct += int(ok)
            review.append({"id": qid, "stem": q["stem"], "options": q["options"], "your_answer": ans,
                           "correct_answer": q["answer"], "correct": ok, "why": q["why"]})
        info = service.record_quiz(conn, u["id"], topic_id, db.app_day(u), n_correct, body.question_ids, body.answers)
        if body.attempt_id is not None:
            service.finish_review(conn, body.attempt_id, u["id"], topic_id, n_correct, len(review), info["kind"], info["success"])
        u = get_user(conn, authorization)
        return {"n_correct": n_correct, "n_questions": len(review), "review": review,
                "kind": info["kind"], "level_before": info["k_before"], "level_after": info["k_after"],
                "topic": topic_status(conn, u, t)}


# ---------- the agent's daily list ----------
@app.get("/api/recall/today")
def recall_today(authorization: str | None = Header(None)):
    with db.connect() as conn:
        u = get_user(conn, authorization)
        n_studied = conn.execute("SELECT COUNT(*) FROM memory_state WHERE user_id=?", (u["id"],)).fetchone()[0]
        if n_studied == 0:
            return {"today": db.day_label(db.app_day(u)), "K": u["K"], "items": [], "agent_ready": scheduler.available,
                    "message": "Read a topic and take its first quiz. Then the agent can plan your reviews."}
        rows = ensure_today(conn, u)
        items = []
        for r in rows:
            t = topics_by_id()[r["topic_id"]]
            items.append({"topic_id": r["topic_id"], "title": t["title"], "group": t["group"], "rank": r["rank"],
                          "priority": round(r["index_score"], 2), "est_recall": round(r["est_recall"], 3),
                          "reason": r["reason"], "completed": bool(r["completed"]), "status": r["status"]})
        msg = "" if items else "No topic needs a review yet. Topics you quizzed today become due from tomorrow."
        reviewed = sum(i["status"] == "completed" for i in items)
        return {"today": db.day_label(db.app_day(u)), "K": u["K"], "items": items, "agent_ready": True,
                "remaining": max(0, u["K"] - reviewed), "due_count": sum(i["status"] in ("recommended", "opened") for i in items),
                "policy_version": POLICY_VERSION, "message": msg}


# ---------- focused revision and skipping ----------
@app.post("/api/topics/{topic_id}/revision/start")
def revision_start(topic_id: str, authorization: str | None = Header(None)):
    t = topics_by_id().get(topic_id)
    if t is None:
        raise HTTPException(404, "Unknown topic.")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        if service.get_state(conn, u["id"], topic_id) is None:
            raise HTTPException(400, "Take the first quiz on this topic before revising it.")
        attempt_id = service.start_review(conn, u["id"], topic_id, db.app_day(u))
        return {"attempt_id": attempt_id, "title": t["title"], "summary_html": revision_summary_html(t)}


@app.post("/api/recall/{topic_id}/skip")
def skip_recommendation(topic_id: str, authorization: str | None = Header(None)):
    with db.connect() as conn:
        u = get_user(conn, authorization)
        cur = conn.execute("UPDATE recommendations SET status='skipped', finished_at=CURRENT_TIMESTAMP "
                           "WHERE user_id=? AND day=? AND topic_id=? AND status IN ('recommended','opened')",
                           (u["id"], db.app_day(u), topic_id))
        if cur.rowcount == 0:
            raise HTTPException(404, "That topic is not waiting on today's list.")
        return {"ok": True}


# ---------- progress ----------
@app.get("/api/progress")
def progress(authorization: str | None = Header(None)):
    with db.connect() as conn:
        u = get_user(conn, authorization)
        topics = [topic_status(conn, u, t) for t in load_topics()]
        quizzed = [t for t in topics if t["status"] == "quizzed"]
        recs = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(completed),0) done FROM recommendations WHERE user_id=?",
                            (u["id"],)).fetchone()
        today = db.app_day(u)
        revision = [{"day": db.day_label(r["day"]), "topic": topics_by_id()[r["topic_id"]]["title"],
                     "status": "overdue" if r["day"] < today and r["status"] in ("recommended", "opened") else r["status"]}
                    for r in conn.execute("SELECT * FROM recommendations WHERE user_id=? ORDER BY day DESC, rank LIMIT 40", (u["id"],))]
        due_today = conn.execute("SELECT COUNT(*) FROM recommendations WHERE user_id=? AND day=? AND status IN ('recommended','opened')",
                                 (u["id"], today)).fetchone()[0]
        history = [{"day": db.day_label(r["day"]), "at": r["created_at"], "topic": topics_by_id()[r["topic_id"]]["title"],
                    "kind": r["kind"], "score": f'{r["n_correct"]}/{r["n_questions"]}'}
                   for r in conn.execute("SELECT * FROM quiz_attempts WHERE user_id=? ORDER BY id DESC LIMIT 40", (u["id"],))]
        obs = [int(t["last_score"].split("/")[0]) / int(t["last_score"].split("/")[1]) for t in quizzed]
        est = [t["est_recall"] for t in quizzed]
        return {"total_topics": len(topics), "read": sum(t["status"] != "not_started" for t in topics),
                "quizzed": len(quizzed),
                "avg_last_score": round(sum(obs) / len(obs), 3) if obs else None,           # observed
                "avg_est_recall": round(sum(est) / len(est), 3) if est else None,           # model estimate
                "recommended_total": recs["n"], "recommended_done": recs["done"],
                "due_today": due_today, "K": u["K"], "revision_history": revision,
                "topics": topics, "history": history, "user": user_payload(u)}


# ---------- demo / development tools ----------
@app.post("/api/dev/advance_day")
def advance_day(body: AdvanceIn, authorization: str | None = Header(None)):
    if not 1 <= body.days <= 60:
        raise HTTPException(400, "days must be between 1 and 60")
    with db.connect() as conn:
        u = get_user(conn, authorization)
        conn.execute("UPDATE users SET day_offset=day_offset+? WHERE id=?", (body.days, u["id"]))
        return user_payload(get_user(conn, authorization))


@app.post("/api/dev/seed_demo")
def seed_demo():
    if os.environ.get("MLRECALL_DEV") != "1":
        raise HTTPException(404, "Not found.")
    from backend.demo import seed_demo_learner
    with db.connect() as conn:
        u = seed_demo_learner(conn)
        token = auth.new_token()
        conn.execute("INSERT INTO sessions(token, user_id) VALUES (?,?)", (token, u["id"]))
        return {**user_payload(u), "token": token}


# ---------- frontend ----------
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")