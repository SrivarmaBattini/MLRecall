"""Turns memory states into a ranked 'review today' list using the trained QWI index table."""
import numpy as np

from config import AGENT, RULES, INDEX_TABLE_PATH, POLICY_VERSION
from rl.memory_model import nearest_bucket, half_life, recall_probability


class Scheduler:
    def __init__(self):
        self._W = None
        self._mtime = None

    def _load(self):
        if not INDEX_TABLE_PATH.exists():
            self._W = None
            return
        mtime = INDEX_TABLE_PATH.stat().st_mtime
        if self._W is None or mtime != self._mtime:
            self._W = np.load(INDEX_TABLE_PATH, allow_pickle=True)["W"]
            self._mtime = mtime

    @property
    def available(self) -> bool:
        self._load()
        return self._W is not None

    def score(self, beta, speed, k, d):
        self._load()
        b = beta * speed
        d_c = int(min(max(d, 0), AGENT.D))
        return float(self._W[nearest_bucket(b), int(k), d_c])

    def est_recall(self, beta, speed, k, d) -> float:
        return float(recall_probability(d, half_life(beta, speed, k)))

    def rank(self, items, K):
        """items: dicts with topic_id, beta, k, d (d >= 1). Returns the top-K by learned index."""
        self._load()
        if self._W is None:
            raise RuntimeError("index table missing")
        scored = []
        for it in items:
            w = self.score(it["beta"], it["speed"], it["k"], it["d"])
            scored.append({**it, "index_score": w,
                           "est_recall": self.est_recall(it["beta"], it["speed"], it["k"], it["d"])})
        scored.sort(key=lambda x: -x["index_score"])
        return scored[:K]


scheduler = Scheduler()
policy_version = POLICY_VERSION
