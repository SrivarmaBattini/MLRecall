# MLRecall

Learn machine learning topic by topic, quiz yourself, and let a reinforcement-learning agent
tell you which topics to revisit **today**, within a daily budget **K**.

- **30 machine-learning topics** with full lessons: key ideas, formulas, a worked example, code,
  common mistakes, summary, and a bank of quiz questions (300 in total).
- **Quizzes decide everything.** Reading a lesson never counts as learning. Each quiz result sets
  how well you remember a topic and when you last tested it.
- **One RL model: QWI** (Q-learning for the Whittle index). It is trained in a simulator and
  produces a priority index. The app looks up the index for each studied topic and shows the top K.

## Quick start

You need Python 3.10 or newer. Run every command from the project folder.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m rl.train_agent           # trains the agent, a few minutes
python run.py                      # then open http://127.0.0.1:8000
```

On the login screen, enter a name to start fresh, or open **the simulated demo learner** to see a
learner with 35 days of history. In **Settings**, the simulated clock moves your account forward in
days so you can watch the agent's list change without waiting.

An internet connection is used only for fonts and math rendering (KaTeX). Without it the app still
works, with plain fonts and math shown as raw text.

## What to run, and what each step produces

| Command | Purpose | Output |
|---|---|---|
| `python scripts/validate_content.py` | Checks every lesson and question | console |
| `python -m pytest -q` | Unit and API tests | console |
| `python -m rl.train_agent` | Trains QWI in the simulator | `artifacts/index_table.npz` |
| `python -m rl.verify_agent` | Indexability, learned vs reference index, decision agreement, retention in simulation | `results/` |
| `python -m rl.simulate_dataset` | Simulated learner dataset (ML topics only) | `data/simulated/` |
| `python -m rl.validate_estimator` | Does the learner-speed estimate recover the truth without drifting? | `results/` |
| `python -m rl.validate_rules` | How noisy are the quiz rules? Is the model calibrated? | `results/` |
| `python scripts/run_pipeline.py` | All of the above in order | all of the above |

Add `--world power` to the two validators to test a forgetting curve the model does **not** assume.
Nothing in this repository contains results: you generate every number yourself.

## How it works

```
OFFLINE (simulation)
  simulated learners -> QWI training -> index table W[half-life bucket, level k, days d]

ONLINE (the app)
  lesson -> "Mark as read" -> quiz -> memory state per topic (level k, days since quiz d)
  learner speed s  <- estimated from the learner's own review quizzes (MAP)
  today's list     <- look up W for each studied topic, take the top K
```

**Memory model.** For learner *j* and topic *i*: half-life `h = beta_i * s_j * g_k` and recall
`p = 2^(-d / h)`.
`beta_i` comes from the topic's difficulty, `g_k` is a fixed level multiplier (1, 2, 4, 8), `d` is
days since the last quiz. The level `k` changes only through the quiz rules; the speed `s_j`
changes only through the likelihood of the quiz counts. Nothing else touches the half-life, so a
learning effect is never counted twice.

**Quiz rules (version 1).** First quiz after reading sets the starting level (0-1 right: level 0,
2-3: 1, 4: 2, 5: 3). A later review quiz with 4 or more of 5 right moves the level up, otherwise
down. A same-day repeat is stored as practice and changes nothing.

**The agent.** Each topic is one "arm" of a restless bandit. Reviewing a topic resets its clock and
may raise its level; not reviewing lets it decay. QWI learns, for every (half-life bucket, level,
days) state, the subsidy at which reviewing and not reviewing are equally good. That number is the
priority shown next to each recommendation. The app takes the K largest, so the budget is
respected by construction.

All assumptions (quiz thresholds, difficulty half-lives, slip and guess rates, and so on) are
listed in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) with the script that tests each one.

## Editing the lessons

Each topic is one Markdown file in `content/topics/`:

```
---
id: linear-regression
title: Linear Regression
group: Supervised Learning
difficulty: 1            # 1 easy, 2 medium, 3 hard to retain; sets the base half-life
order: 1
summary: One line shown under the title.
prerequisites: other-topic-id, another-id
---
## What it is and why it matters ... (eight sections, in this order)
## Questions
Q: Question text
A) ...
B) ...
C) ...
D) ...
ANSWER: B
WHY: One-sentence explanation shown after the quiz.
```

Math uses `$...$` and `$$...$$`. Run `python scripts/validate_content.py` after editing. Please read
every explanation and answer yourself before relying on them.

## Project layout

```
config.py              every setting and assumption in one place
rl/                    memory model, simulator, QWI agent, DP reference, validators
backend/               FastAPI app, SQLite storage, scheduler, lesson loader
frontend/              plain HTML, CSS and JavaScript (no build step)
content/topics/        the 30 lessons and question banks
scripts/               content validator and pipeline runner
tests/                 unit and API tests
docs/ASSUMPTIONS.md    assumptions register and known limits
```

## Troubleshooting

- **"The agent has not been trained yet"**: run `python -m rl.train_agent`, then refresh.
- **Port 8000 is busy**: edit the port in `run.py`.
- **Reset everything**: stop the app and delete `data/mlrecall.db`.
- **Training is slow**: lower `qwi_steps` in `config.py`; the index will be less accurate, and
  `python -m rl.verify_agent` will show by how much.

## Honest scope

The agent is trained and evaluated in simulation, under the forgetting model described above.
Simulated results show how the method behaves under that model. They do not show that real learners
remember more. The local app has no passwords: users are identified by name only.
