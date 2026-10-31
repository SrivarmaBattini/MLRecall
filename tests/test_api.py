"""End-to-end API test with a temporary database and a tiny trained index table."""
import pytest
from fastapi.testclient import TestClient

from rl.qwi import QWI


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import backend.db as db
    import backend.scheduler as sched
    table = tmp_path / "index_table.npz"
    QWI(seed=1, batch=32).train(steps=4000, verbose=False).save(table)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("MLRECALL_DEV", "1")
    monkeypatch.setattr(sched, "INDEX_TABLE_PATH", table)
    sched.scheduler._W = None
    from backend.app import app
    with TestClient(app) as c:
        yield c


def hdr(u):
    return {"Authorization": f"Bearer {u['token']}"}


def login(c, name, password="secret1"):
    r = c.post("/api/login", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


def take_quiz(c, u, topic, correct):
    q = c.get(f"/api/topics/{topic}/quiz", headers=hdr(u)).json()
    from backend.content import topics_by_id
    bank = {x["id"]: x for x in topics_by_id()[topic]["questions"]}
    answers = []
    for i, item in enumerate(q["questions"]):
        right = bank[item["id"]]["answer"]
        answers.append(right if i < correct else (right + 1) % 4)
    r = c.post(f"/api/topics/{topic}/quiz", headers=hdr(u),
               json={"question_ids": [x["id"] for x in q["questions"]], "answers": answers})
    assert r.status_code == 200, r.text
    return r.json()


def take_quiz_with_attempt(c, u, topic, correct, attempt_id):
    q = c.get(f"/api/topics/{topic}/quiz", headers=hdr(u)).json()
    from backend.content import topics_by_id
    bank = {x["id"]: x for x in topics_by_id()[topic]["questions"]}
    answers = [bank[it["id"]]["answer"] if i < correct else (bank[it["id"]]["answer"] + 1) % 4
               for i, it in enumerate(q["questions"])]
    r = c.post(f"/api/topics/{topic}/quiz", headers=hdr(u),
               json={"question_ids": [x["id"] for x in q["questions"]], "answers": answers, "attempt_id": attempt_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_full_learning_loop(client):
    u = login(client, "tester")
    assert client.get("/api/recall/today", headers=hdr(u)).json()["items"] == []

    # quiz is locked until the lesson is read
    assert client.get("/api/topics/linear-regression/quiz", headers=hdr(u)).status_code == 400
    assert client.post("/api/topics/linear-regression/read", headers=hdr(u)).json()["status"] == "read"

    first = take_quiz(client, u, "linear-regression", 4)
    assert first["kind"] == "first" and first["level_after"] == 2

    same_day = take_quiz(client, u, "linear-regression", 1)
    assert same_day["kind"] == "retake" and same_day["topic"]["level"] == 2   # unchanged

    client.post("/api/dev/advance_day", headers=hdr(u), json={"days": 3})
    today = client.get("/api/recall/today", headers=hdr(u)).json()
    assert [i["topic_id"] for i in today["items"]] == ["linear-regression"]
    assert "Estimated recall" in today["items"][0]["reason"]

    review = take_quiz(client, u, "linear-regression", 5)
    assert review["kind"] == "review" and review["level_after"] == 3
    done = client.get("/api/recall/today", headers=hdr(u)).json()["items"][0]
    assert done["completed"] is True

    prog = client.get("/api/progress", headers=hdr(u)).json()
    assert prog["quizzed"] == 1 and prog["recommended_done"] == 1


def test_budget_is_respected(client):
    u = login(client, "budget")
    topics = [t["id"] for g in client.get("/api/topics", headers=hdr(u)).json() for t in g["topics"]][:8]
    for t in topics:
        client.post(f"/api/topics/{t}/read", headers=hdr(u))
        take_quiz(client, u, t, 3)
    client.post("/api/settings", headers=hdr(u), json={"K": 3})
    client.post("/api/dev/advance_day", headers=hdr(u), json={"days": 5})
    items = client.get("/api/recall/today", headers=hdr(u)).json()["items"]
    assert len(items) == 3


def test_demo_learner_has_history_and_recommendations(client):
    d = client.post("/api/dev/seed_demo").json()
    assert d["is_simulated"] is True
    items = client.get("/api/recall/today", headers=hdr(d)).json()["items"]
    assert 1 <= len(items) <= d["K"]


def test_other_students_data_is_not_visible(client):
    a, b = login(client, "alice"), login(client, "bob")
    client.post("/api/topics/linear-regression/read", headers=hdr(a))
    take_quiz(client, a, "linear-regression", 4)
    assert client.get("/api/progress", headers=hdr(a)).json()["quizzed"] == 1
    assert client.get("/api/progress", headers=hdr(b)).json()["quizzed"] == 0
    # a guessed numeric id or a made-up token gives nothing
    assert client.get("/api/me", headers={"Authorization": "Bearer 1"}).status_code == 401
    assert client.get("/api/me").status_code == 401
    # wrong password is refused, and so is the passwordless demo account
    assert client.post("/api/login", json={"username": "alice", "password": "wrong-pass"}).status_code == 401
    d = client.post("/api/dev/seed_demo").json()
    assert client.post("/api/login", json={"username": d["username"], "password": "anything1"}).status_code == 401


def test_logout_ends_the_session(client):
    u = login(client, "leaver")
    assert client.get("/api/me", headers=hdr(u)).status_code == 200
    client.post("/api/logout", headers=hdr(u))
    assert client.get("/api/me", headers=hdr(u)).status_code == 401


def test_skip_open_and_overdue_are_recorded(client):
    u = login(client, "skipper")
    for t in ("linear-regression", "logistic-regression", "decision-trees"):
        client.post(f"/api/topics/{t}/read", headers=hdr(u))
        take_quiz(client, u, t, 4)
    client.post("/api/dev/advance_day", headers=hdr(u), json={"days": 3})
    items = client.get("/api/recall/today", headers=hdr(u)).json()["items"]
    assert len(items) == 3 and all(i["status"] == "recommended" for i in items)
    first, second = items[0]["topic_id"], items[1]["topic_id"]

    assert client.post(f"/api/recall/{first}/skip", headers=hdr(u)).status_code == 200
    assert client.post(f"/api/recall/{first}/skip", headers=hdr(u)).status_code == 404     # already closed
    start = client.post(f"/api/topics/{second}/revision/start", headers=hdr(u)).json()
    assert "Key ideas" in start["summary_html"] and start["attempt_id"] > 0
    now = {i["topic_id"]: i["status"] for i in client.get("/api/recall/today", headers=hdr(u)).json()["items"]}
    assert now[first] == "skipped" and now[second] == "opened"

    # the list is stable: skipping does not add a replacement topic, so the day stays within K
    today = client.get("/api/recall/today", headers=hdr(u)).json()
    assert len(today["items"]) == 3 and today["remaining"] == today["K"]      # a skip uses no review capacity

    # finishing the revision quiz completes the session and the recommendation
    take_quiz_with_attempt(client, u, second, 5, start["attempt_id"])
    today = client.get("/api/recall/today", headers=hdr(u)).json()
    assert {i["topic_id"]: i["status"] for i in today["items"]}[second] == "completed"
    assert today["remaining"] == today["K"] - 1

    # the next day, what was left unfinished is overdue
    client.post("/api/dev/advance_day", headers=hdr(u), json={"days": 1})
    client.get("/api/recall/today", headers=hdr(u))
    hist = client.get("/api/progress", headers=hdr(u)).json()["revision_history"]
    assert {h["status"] for h in hist if h["topic"] != "x"} >= {"skipped", "completed", "overdue"}


def test_revision_requires_a_first_quiz_and_a_valid_attempt(client):
    u = login(client, "newbie")
    assert client.post("/api/topics/linear-regression/revision/start", headers=hdr(u)).status_code == 400
    client.post("/api/topics/linear-regression/read", headers=hdr(u))
    take_quiz(client, u, "linear-regression", 4)
    # an attempt id that belongs to nobody is ignored, not trusted
    r = take_quiz_with_attempt(client, u, "linear-regression", 3, 999)
    assert r["n_correct"] == 3


def test_untrained_agent_gives_a_clear_message(tmp_path, monkeypatch):
    import backend.db as db
    import backend.scheduler as sched
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t2.db")
    monkeypatch.setattr(sched, "INDEX_TABLE_PATH", tmp_path / "missing.npz")
    sched.scheduler._W = None
    from backend.app import app
    with TestClient(app) as c:
        u = login(c, "x")
        c.post("/api/topics/linear-regression/read", headers=hdr(u))
        take_quiz(c, u, "linear-regression", 4)
        c.post("/api/dev/advance_day", headers=hdr(u), json={"days": 2})
        r = c.get("/api/recall/today", headers=hdr(u))
        assert r.status_code == 503 and "train_agent" in r.json()["detail"]