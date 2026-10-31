"""SQLite storage. One file: data/mlrecall.db (created on first run)."""
import sqlite3
from datetime import date

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password_salt TEXT, password_hash TEXT,       -- NULL for the simulated demo learner
  K INTEGER NOT NULL DEFAULT 5,
  day_offset INTEGER NOT NULL DEFAULT 0,        -- simulated clock: days added to today
  speed_est REAL NOT NULL DEFAULT 1.0,          -- learner speed estimate s_j
  is_simulated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
  token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS reading_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, topic_id TEXT NOT NULL, day INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS quiz_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, topic_id TEXT NOT NULL,
  day INTEGER NOT NULL, kind TEXT NOT NULL,                 -- first | review | retake
  n_correct INTEGER NOT NULL, n_questions INTEGER NOT NULL,
  question_ids TEXT, answers TEXT, k_before INTEGER, k_after INTEGER, d_since_last INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS memory_state (
  user_id INTEGER NOT NULL, topic_id TEXT NOT NULL,
  k INTEGER NOT NULL, last_quiz_day INTEGER NOT NULL, first_quiz_day INTEGER NOT NULL,
  n_reviews INTEGER NOT NULL DEFAULT 0, last_n_correct INTEGER NOT NULL, last_n_questions INTEGER NOT NULL,
  PRIMARY KEY (user_id, topic_id)
);
CREATE TABLE IF NOT EXISTS review_events (      -- rows used to estimate learner speed
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, topic_id TEXT NOT NULL,
  day INTEGER NOT NULL, beta REAL NOT NULL, k_before INTEGER NOT NULL, d INTEGER NOT NULL,
  n_correct INTEGER NOT NULL, n_questions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, day INTEGER NOT NULL, topic_id TEXT NOT NULL,
  rank INTEGER NOT NULL, index_score REAL NOT NULL, est_recall REAL NOT NULL, reason TEXT NOT NULL,
  policy_version TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'recommended',               -- recommended | opened | completed | skipped
  opened_at TEXT, finished_at TEXT,
  UNIQUE (user_id, day, topic_id)
);
CREATE TABLE IF NOT EXISTS review_attempts (    -- one focused revision session
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, topic_id TEXT NOT NULL, day INTEGER NOT NULL,
  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, completed_at TEXT,
  n_correct INTEGER, n_questions INTEGER, outcome TEXT      -- success | failure | practice
);
"""


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


def app_day(user) -> int:
    """Today's day number for this user (real date plus the simulated-clock offset)."""
    return date.today().toordinal() + int(user["day_offset"])


def day_label(day: int) -> str:
    return date.fromordinal(day).isoformat()
