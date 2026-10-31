"""
config.py - the single place for every number that defines MLRecall.

Everything marked ASSUMPTION is a version-1 heuristic, not a validated fact about
human memory. They are documented in docs/ASSUMPTIONS.md and tested by
rl/validate_rules.py and rl/validate_estimator.py. Change them here only.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / "content" / "topics"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
INDEX_TABLE_PATH = ARTIFACT_DIR / "index_table.npz"
DB_PATH = DATA_DIR / "mlrecall.db"


@dataclass(frozen=True)
class MemoryRules:
    """Quiz-result -> memory-state rules (version 1)."""
    version: str = "memory_rules_v1"
    n_questions: int = 5                      # ASSUMPTION: questions per quiz
    # ASSUMPTION: minimum correct answers (out of n_questions) that sets the
    # starting level after the FIRST quiz on a topic: level 0,1,2,3
    first_quiz_min_correct: tuple = (0, 2, 4, 5)
    # ASSUMPTION: a review quiz counts as a success at >= this many correct
    review_success_min_correct: int = 4
    # Level multipliers g_k. Fixed, never estimated.
    level_multipliers: tuple = (1.0, 2.0, 4.0, 8.0)
    # ASSUMPTION: topic base half-life (days, at level 0, speed 1) by difficulty
    base_half_life: dict = field(default_factory=lambda: {1: 8.0, 2: 4.0, 3: 2.0})
    # Measurement model linking true recall p to what a 5-question MCQ quiz shows.
    quiz_slip: float = 0.10                   # ASSUMPTION: chance of a wrong answer to a known item
    quiz_guess: float = 0.25                  # chance of a right guess (4-option MCQ)
    # Learner speed s_j: log-normal prior, MAP estimate, clamped
    speed_prior_sigma: float = 0.5            # ASSUMPTION
    speed_bounds: tuple = (0.25, 4.0)

    @property
    def L(self) -> int:
        return len(self.level_multipliers)


@dataclass(frozen=True)
class AgentConfig:
    """Scheduler / RL settings."""
    D: int = 40                # days-since-quiz is capped at D
    beta: float = 0.95         # discount factor
    n_buckets: int = 12        # half-life buckets (log-spaced)
    b_min: float = 0.5         # smallest bucket base half-life (days)
    b_max: float = 40.0        # largest bucket base half-life (days)
    default_K: int = 5         # daily review budget
    # QWI training (see rl/qwi.py)
    qwi_steps: int = 4_000_000   # transitions per bucket (a few minutes on a laptop)
    qwi_lambda_lo: float = -1.0
    qwi_lambda_hi: float = 15.0


RULES = MemoryRules()
AGENT = AgentConfig()
POLICY_VERSION = "qwi_v1"
