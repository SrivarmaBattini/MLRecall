"""Check every topic file: front matter, sections, question format, prerequisites."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.content import load_topics, GROUP_ORDER

REQUIRED_SECTIONS = ["What it is and why it matters", "Intuition", "Key ideas", "The math",
                     "Worked example", "Code", "Common mistakes", "Summary"]
MIN_QUESTIONS = 10


def main():
    topics = load_topics()
    ids = {t["id"] for t in topics}
    problems = []
    orders = [t["order"] for t in topics]
    if len(set(orders)) != len(orders):
        problems.append("duplicate 'order' values")
    for t in topics:
        if t["group"] not in GROUP_ORDER:
            problems.append(f"{t['id']}: unknown group '{t['group']}'")
        if t["difficulty"] not in (1, 2, 3):
            problems.append(f"{t['id']}: difficulty must be 1, 2 or 3")
        for p in t["prerequisites"]:
            if p not in ids:
                problems.append(f"{t['id']}: unknown prerequisite '{p}'")
        for s in REQUIRED_SECTIONS:
            if f"<h2>{s}</h2>" not in t["lesson_html"]:
                problems.append(f"{t['id']}: missing section '{s}'")
        if len(t["questions"]) < MIN_QUESTIONS:
            problems.append(f"{t['id']}: only {len(t['questions'])} questions (need {MIN_QUESTIONS}+)")
        for q in t["questions"]:
            if len(set(q["options"])) != 4:
                problems.append(f"{q['id']}: options must be 4 distinct strings")
    print(f"{len(topics)} topics, {sum(len(t['questions']) for t in topics)} questions checked.")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("Content OK.")


if __name__ == "__main__":
    main()
