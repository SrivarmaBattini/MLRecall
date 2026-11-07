# MLRecall — RL-Powered Spaced Repetition for ML Concepts

MLRecall is a full-stack spaced-repetition web application that uses **Reinforcement Learning** to intelligently schedule daily quiz reviews for 30 machine-learning topics. The core agent is trained offline via **QWI (Q-learning for the Whittle Index)** — a model-free algorithm that solves the restless multi-armed bandit problem without knowing the environment dynamics. At runtime, the agent reads the learned index table and picks which topics each learner should review today.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [RL Formulation](#rl-formulation)
- [Memory Model](#memory-model)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Training the RL Agent](#training-the-rl-agent)
- [Evaluation & Verification](#evaluation--verification)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [API Reference](#api-reference)
- [Content Topics](#content-topics)

---

## Project Overview

Standard spaced-repetition systems (Anki, SM-2, Leitner) use fixed heuristic intervals. MLRecall replaces those heuristics with a **learned policy** derived from the Whittle index — a theoretically principled priority score for restless bandits.

Key design goals:
- **No ground truth retention data needed** — the agent is trained entirely in a simulator.
- **Personalized** — each learner's quiz performance updates a MAP estimate of their memory "speed", which shifts the agent's scheduling priorities.
- **Deployable** — the trained policy is a small NumPy array (`artifacts/index_table.npz`). No neural network runs at serving time.

---

## System Architecture

```
┌───────────────────┐      HTTP/JSON      ┌──────────────────────┐
│   Browser UI      │ ◄─────────────────► │  FastAPI Backend      │
│  (Vanilla JS)     │                     │  backend/app.py       │
└───────────────────┘                     └──────────┬───────────┘
                                                     │
                              ┌──────────────────────┼────────────────────────┐
                              │                      │                        │
                     ┌────────▼──────┐    ┌──────────▼──────┐    ┌───────────▼──────┐
                     │   SQLite DB   │    │   Scheduler      │    │  Content Parser  │
                     │  backend/db   │    │  backend/sched.  │    │  backend/content │
                     └───────────────┘    └──────────┬───────┘    └──────────────────┘
                                                     │ loads
                                          ┌──────────▼───────────┐
                                          │  Index Table          │
                                          │  artifacts/           │
                                          │  index_table.npz      │
                                          └──────────┬────────────┘
                                                     │ produced by
                                          ┌──────────▼───────────┐
                                          │  QWI Agent (offline) │
                                          │  rl/qwi.py            │
                                          └──────────────────────┘
```

---

## RL Formulation

### Problem: Restless Multi-Armed Bandit (RMAB)

Each topic is one **arm**. On every day, the learner has a budget of **K reviews**. Each arm evolves whether or not it is pulled:
- **Active (reviewed)**: topic level may go up or down based on quiz score.
- **Passive (not reviewed)**: days since last quiz increases, recall probability decays.

Solving a RMAB exactly is PSPACE-hard. MLRecall uses the **Whittle index**, which decomposes the problem into one single-arm MDP per arm and assigns each state a *subsidy* λ at which reviewing and not reviewing are equally valuable.

### State Space

Each topic's state is `(k, d)` where:
- `k ∈ {0, 1, 2, 3}` — memory level (affects half-life via multipliers `[1×, 2×, 4×, 8×]`)
- `d ∈ {0, …, D}` — days since last quiz (capped at `D = 40`)

Total states per bucket: `4 × 41 = 164`.

### Reward

Instantaneous reward at state `(k, d)` is the **estimated recall probability**:

```
p = 2^(-d / h)    where  h = β_i × s_j × g_k
```

- `β_i` — topic base half-life (by difficulty: easy=8d, medium=4d, hard=2d)
- `s_j` — learner speed estimate (MAP-estimated from quiz history)
- `g_k` — level multiplier

### Algorithm: QWI

QWI uses **two coupled timescales** to learn the Whittle index without a model:

```
Fast:  Q^j(s,a) ← Q^j(s,a) + α[r + λ_j·(1-a) + β·max_b Q^j(s',b) − Q^j(s,a)]
Slow:  λ_j      ← λ_j      + c·[Q^j(j,1) − Q^j(j,0)]
```

- One Q-table per *target state* `j`, all updated from every simulated transition.
- The slow timescale drives `λ_j` toward the indifference point (Whittle index definition).
- All `n_buckets = 12` base half-life buckets are trained in parallel.
- Output: index table `W[bucket, k, d]` — shape `(12, 4, 41)`.

Training runs entirely inside the simulator (generative model setting): `4,000,000` transitions per bucket by default (~few minutes on CPU).

---

## Memory Model

The memory model lives in [`rl/memory_model.py`](rl/memory_model.py) and [`config.py`](config.py).

| Component | Description |
|-----------|-------------|
| **Forgetting curve** | Exponential: `p = 2^(-d/h)` |
| **Half-life** | `h = base_half_life(difficulty) × learner_speed × level_multiplier` |
| **Level transitions** | Discrete (0–3). First quiz sets the starting level; review quiz shifts ±1 |
| **Measurement model** | Noisy MCQ: slip probability 0.10, guess probability 0.25 (4-option) |
| **Speed estimation** | MAP estimate of `s_j` via log-normal prior + binomial quiz likelihood, updated after every review quiz |

All assumptions are documented in [`config.py`](config.py) and validated by `rl/validate_rules.py` and `rl/validate_estimator.py`.

---

## Project Structure

```
mlrecall/
├── run.py                       # Entry point: starts the FastAPI server
├── config.py                    # All hyperparameters and paths (single source of truth)
├── requirements.txt             # Python dependencies
│
├── backend/
│   ├── app.py                   # FastAPI routes (login, quiz, recall, progress, dev)
│   ├── db.py                    # SQLite schema and helpers
│   ├── service.py               # Core quiz logic, speed estimation, state updates
│   ├── scheduler.py             # Loads index table, ranks topics for today's review
│   ├── content.py               # Markdown lesson loader and quiz parser
│   ├── auth.py                  # Password hashing and session tokens
│   └── demo.py                  # Simulated demo learner seeder
│
├── rl/
│   ├── qwi.py                   # QWI algorithm — trains the Whittle index table
│   ├── train_agent.py           # CLI wrapper for training
│   ├── verify_agent.py          # 4-part verification: indexability, accuracy, decisions, retention
│   ├── memory_model.py          # Memory model, recall/quiz probability, speed estimator
│   ├── simulator.py             # Single-arm dynamics + SimLearner (synthetic learners)
│   ├── evaluate_baselines.py    # Compares QWI vs SM-2, Leitner, Most-Overdue, Random, etc.
│   ├── evaluate_training.py     # Training curve analysis
│   ├── evaluation_metrics.py    # Shared evaluation metrics (CI, hindsight gap, calibration)
│   ├── plot_evaluation.py       # Generates result figures
│   ├── simulate_dataset.py      # Generates a synthetic review history dataset
│   ├── validate_estimator.py    # Validates the speed MAP estimator
│   ├── validate_rules.py        # Validates quiz → level rules
│   └── whittle_reference.py     # DP reference Whittle index (for verification)
│
├── frontend/
│   ├── index.html               # Single-page app shell
│   ├── app.js                   # Vanilla JS: topics, quiz, today's review list, progress
│   └── styles.css               # Styling
│
├── content/
│   └── topics/                  # 30 Markdown lesson files with embedded quiz questions
│       ├── 01-linear-regression.md
│       ├── ...
│       └── 30-few-shot-learning.md
│
├── scripts/
│   ├── run_pipeline.py          # Runs the full offline pipeline end-to-end
│   └── validate_content.py      # Validates all content files
│
├── tests/
│   ├── test_agent.py
│   ├── test_api.py
│   ├── test_content.py
│   ├── test_evaluation_metrics.py
│   ├── test_memory_model.py
│   ├── test_simulator_alignment.py
│   └── conftest.py
│
├── artifacts/
│   └── index_table.npz          # Trained QWI index table (gitignored after generation)
│
└── data/
    └── mlrecall.db              # SQLite database (gitignored, created on first run)
```

---

## Dependencies

Python ≥ 3.11 is recommended. All dependencies are pure Python / NumPy-based — no GPU required.

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Index table, simulator, all numerical ops |
| `scipy` | ≥ 1.10 | Speed MAP estimation (`minimize_scalar`), binomial distribution |
| `pandas` | ≥ 2.0 | Evaluation result tables and CSV exports |
| `matplotlib` | ≥ 3.7 | Evaluation and verification plots |
| `fastapi` | ≥ 0.110 | REST API server |
| `uvicorn` | ≥ 0.27 | ASGI server for FastAPI |
| `markdown` | ≥ 3.5 | Renders lesson markdown to HTML for the frontend |
| `pytest` | ≥ 8.0 | Unit and integration tests |
| `httpx` | ≥ 0.27 | Async HTTP client used in API tests |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/SrivarmaBattini/MLRecall.git
cd MLRecall
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

The app requires the trained index table. You can either train it yourself (see below) or use a pre-committed table if present.

### Start the server

```bash
python run.py
```

Open **http://127.0.0.1:8000** in your browser.

- **First visit**: Create an account with any username and a password ≥ 6 characters.
- **Demo mode**: In dev mode (`MLRECALL_DEV=1`, enabled by default via `run.py`), a simulated learner can be seeded via `POST /api/dev/seed_demo`.
- **Day simulation**: For testing, use `POST /api/dev/advance_day` to fast-forward the internal clock.

### Workflow

1. **Read a topic** — Browse topics grouped by category, open a lesson, mark it as read.
2. **First quiz** — Take the 5-question MCQ quiz; your starting memory level (0–3) is set from your score.
3. **Daily review** — `GET /api/recall/today` returns up to K topics ranked by the QWI index score.
4. **Review quiz** — Complete each review quiz; the agent updates your level and re-estimates your speed.
5. **Progress** — `GET /api/progress` shows your overall recall statistics and history.

---

## Training the RL Agent

Training runs entirely offline in the simulator and produces `artifacts/index_table.npz`.

### Default training (~few minutes on CPU)

```bash
python -m rl.train_agent
```

### Options

```bash
python -m rl.train_agent --steps 4000000   # transitions per bucket (default)
python -m rl.train_agent --steps 1500000   # faster, slightly less accurate
python -m rl.train_agent --seed 1          # different random seed
python -m rl.train_agent --out path/to/custom.npz
```

The index table shape is `(n_buckets=12, L=4, D+1=41)` — a 12 × 4 × 41 float array.

---

## Evaluation & Verification

### Verify the trained agent

```bash
python -m rl.verify_agent
```

Runs 4 checks and writes results to `results/`:

| Check | What it measures |
|-------|-----------------|
| **Indexability** | Whether the passive set is monotone in the subsidy for each bucket |
| **Index accuracy** | Mean absolute error and Spearman correlation vs. DP Whittle reference |
| **Decision agreement** | Top-K overlap with the DP index across 2000 random 30-topic situations |
| **Retention** | Mean recall over 120 simulated days under the learned policy |

```bash
# Also train a second agent to check training stability
python -m rl.verify_agent --seed-check
```

### Compare against baselines

```bash
# Matched forgetting curve (exponential, same as the agent's model)
python -m rl.evaluate_baselines

# Mismatched curve (power-law, robustness test)
python -m rl.evaluate_baselines --world power

# Full options
python -m rl.evaluate_baselines --world both --seeds 30 --days 120 --K 5
```

Baselines evaluated:
- **QWI (online speed)** — the deployable policy with MAP-estimated speed
- **QWI (prior speed)** — QWI assuming speed = 1.0
- **QWI (oracle speed)** — diagnostic: uses hidden true speed
- **DP Whittle (oracle)** — diagnostic: exact solution with true dynamics
- **SM-2-style** — SuperMemo 2 scheduling heuristic
- **Leitner-style** — Leitner box system
- **Most overdue** — greedy: review the most overdue topic first
- **Lowest last score** — greedy: review by worst recent quiz score
- **Forgetting risk** — greedy: review by lowest estimated recall
- **Fixed rotation** — round-robin
- **Random** — random selection

### Validate the memory model

```bash
python -m rl.validate_rules        # quiz-to-level rule calibration
python -m rl.validate_estimator    # speed MAP estimator accuracy
```

### Unit tests

```bash
pytest -q
```

---

## Running the Full Pipeline

To reproduce all offline results from scratch:

```bash
python scripts/run_pipeline.py
```

Steps executed in order:
1. Validate lesson content
2. Run unit tests
3. Train the QWI agent
4. Verify the agent
5. Compare with baselines (matched + mismatched forgetting curve)
6. Simulate a learner dataset
7. Validate the speed estimator (both worlds)
8. Validate the quiz rules (both worlds)

```bash
# Skip retraining if you already have an index table
python scripts/run_pipeline.py --skip-train
```

All outputs land in `results/` (CSVs, PNGs, JSON reports).

---

## API Reference

All endpoints return JSON. Auth uses a Bearer token obtained from `/api/login`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/login` | Create account / log in. Returns `token`. |
| `POST` | `/api/logout` | Invalidate the session token. |
| `GET` | `/api/me` | Current user info and speed estimate. |
| `POST` | `/api/settings` | Set daily review budget `K` (1–15). |
| `GET` | `/api/topics` | All topics with per-user status and estimated recall. |
| `GET` | `/api/topics/{id}` | Full topic: lesson HTML, questions, prerequisites. |
| `POST` | `/api/topics/{id}/read` | Mark a lesson as read. |
| `GET` | `/api/topics/{id}/quiz` | Fetch a 5-question quiz (LRU question selection). |
| `POST` | `/api/topics/{id}/quiz` | Submit quiz answers; updates memory state and speed. |
| `POST` | `/api/topics/{id}/revision/start` | Open a focused revision session. |
| `GET` | `/api/recall/today` | Today's review list ranked by QWI index score. |
| `POST` | `/api/recall/{id}/skip` | Skip a topic from today's list. |
| `GET` | `/api/progress` | Full learning progress and quiz history. |
| `POST` | `/api/dev/advance_day` | (Dev) Fast-forward the internal clock by N days. |
| `POST` | `/api/dev/seed_demo` | (Dev) Create a simulated demo learner. |

---

## Content Topics

30 ML topics across 6 groups, each with a Markdown lesson and ≥5 MCQ questions:

| Group | Topics |
|-------|--------|
| **Supervised Learning** | Linear Regression, Logistic Regression, Decision Trees, Random Forests, SVMs, Gradient Boosting |
| **Unsupervised Learning** | K-Means, PCA |
| **Deep Learning** | Neural Networks Intro, Backpropagation, CNNs, RNNs, LSTMs, Autoencoders, GANs |
| **NLP** | Word Embeddings, Attention Mechanism, Self-Attention, Transformers, BERT, NER, Machine Translation |
| **Reinforcement Learning** | MDPs, Q-Learning, DQN, Policy Gradients, Actor-Critic, PPO |
| **Transfer Learning** | Transfer Learning, Few-Shot Learning |

Each `.md` file uses YAML frontmatter (`title`, `difficulty`, `group`, `summary`, `prerequisites`) followed by lesson content and a fenced `questions` block.
