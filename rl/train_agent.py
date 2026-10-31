"""
Train the QWI agent in the simulator and save the index table.

    python -m rl.train_agent                 # default budget
    python -m rl.train_agent --steps 1500000 --seed 1
"""
import argparse
import time

from config import AGENT, INDEX_TABLE_PATH
from rl.qwi import QWI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=AGENT.qwi_steps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(INDEX_TABLE_PATH))
    args = ap.parse_args()
    t0 = time.time()
    agent = QWI(seed=args.seed).train(args.steps)
    from pathlib import Path
    path = agent.save(Path(args.out))
    print(f"Saved index table to {path}  (trained in {time.time() - t0:.0f}s)")
    print("Next: python -m rl.verify_agent")


if __name__ == "__main__":
    main()
