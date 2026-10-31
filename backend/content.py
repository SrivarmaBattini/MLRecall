"""
content.py - loads the lessons and question bank from content/topics/*.md

Each topic file = front matter + lesson sections + a '## Questions' block.
Edit the .md files to change what learners see; no code changes needed.
Run `python scripts/validate_content.py` after editing.
"""
import re
from functools import lru_cache

import markdown

from config import CONTENT_DIR

GROUP_ORDER = [
    "Supervised Learning", "Unsupervised Learning", "Deep Learning",
    "Transformers and Language", "Reinforcement Learning", "Transfer and Few-shot Learning",
]
_FRONT = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
_MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
_MATH_INLINE = re.compile(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)")


def _render(md_text: str) -> str:
    """Markdown -> HTML while leaving $...$ and $$...$$ untouched for KaTeX."""
    stash = []

    def keep(m):
        stash.append(m.group(0))
        return f"MATHTOKEN{len(stash) - 1}ENDTOKEN"

    text = _MATH_BLOCK.sub(keep, md_text)
    text = _MATH_INLINE.sub(keep, text)
    out = markdown.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
    for i, raw in enumerate(stash):
        safe = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out = out.replace(f"MATHTOKEN{i}ENDTOKEN", safe)
    return out


REVISION_SECTIONS = ("Key ideas", "The math", "Summary")


def _split_sections(html: str) -> dict:
    """Map each <h2> heading to its HTML (heading included)."""
    out = {}
    for part in re.split(r"(?=<h2>)", html):
        m = re.match(r"<h2>(.*?)</h2>", part)
        if m:
            out[m.group(1)] = part
    return out


def revision_summary_html(topic: dict) -> str:
    """Concise revision view: key ideas, formulas and the summary of a lesson."""
    return "\n".join(topic["sections"][h] for h in REVISION_SECTIONS if h in topic["sections"])

    

def _parse_questions(block: str, topic_id: str):
    questions = []
    for n, chunk in enumerate(re.split(r"\n(?=Q:)", block.strip()), start=1):
        chunk = chunk.strip()
        if not chunk.startswith("Q:"):
            continue
        m = re.match(r"Q:(.*?)\nA\)(.*?)\nB\)(.*?)\nC\)(.*?)\nD\)(.*?)\nANSWER:\s*([ABCD])\s*\nWHY:(.*)$",
                     chunk, re.S)
        if not m:
            raise ValueError(f"Malformed question #{n} in topic '{topic_id}':\n{chunk[:200]}")
        stem, a, b, c, d, ans, why = (x.strip() for x in m.groups())
        questions.append({"id": f"{topic_id}-q{n}", "stem": stem, "options": [a, b, c, d],
                          "answer": "ABCD".index(ans), "why": why})
    return questions


def _parse_topic(path):
    raw = path.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise ValueError(f"{path.name}: missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    body = m.group(2)
    lesson_md, _, q_block = body.partition("\n## Questions")
    topic_id = meta["id"]
    prereq = [p.strip() for p in meta.get("prerequisites", "").split(",") if p.strip()]
    topic = {
        "id": topic_id, "title": meta["title"], "group": meta["group"],
        "difficulty": int(meta["difficulty"]), "order": int(meta["order"]),
        "summary": meta.get("summary", ""), "prerequisites": prereq,
        "lesson_html": _render(lesson_md.strip()),
        "questions": _parse_questions(q_block, topic_id),
    }
    topic["sections"] = _split_sections(topic["lesson_html"])
    return topic


@lru_cache(maxsize=1)
def load_topics():
    topics = [_parse_topic(p) for p in sorted(CONTENT_DIR.glob("*.md"))]
    topics.sort(key=lambda t: t["order"])
    return topics


def topics_by_id():
    return {t["id"]: t for t in load_topics()}


def grouped():
    out = []
    for g in GROUP_ORDER:
        items = [t for t in load_topics() if t["group"] == g]
        if items:
            out.append({"group": g, "topics": items})
    return out
