from backend.content import load_topics, GROUP_ORDER


def test_thirty_ml_topics_with_complete_banks():
    topics = load_topics()
    assert len(topics) == 30
    ids = [t["id"] for t in topics]
    assert len(set(ids)) == 30
    for t in topics:
        assert t["group"] in GROUP_ORDER
        assert t["difficulty"] in (1, 2, 3)
        assert len(t["questions"]) >= 10
        for p in t["prerequisites"]:
            assert p in ids
        for q in t["questions"]:
            assert 0 <= q["answer"] < 4 and len(set(q["options"])) == 4


def test_math_is_left_for_katex_and_code_is_kept():
    t = {x["id"]: x for x in load_topics()}["linear-regression"]
    assert "$$" in t["lesson_html"] and "<code" in t["lesson_html"]
    assert "MATHTOKEN" not in t["lesson_html"]


def test_question_ids_are_unique_across_the_bank():
    qids = [q["id"] for t in load_topics() for q in t["questions"]]
    assert len(qids) == len(set(qids))
