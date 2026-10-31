"""
Run the whole offline pipeline in order (from the project root):

    python scripts/run_pipeline.py              # everything
    python scripts/run_pipeline.py --skip-train # reuse an already trained agent

Steps: validate content -> unit tests -> train QWI -> verify agent -> simulate dataset
       -> validate estimator (both worlds) -> validate rules (both worlds)
Outputs land in artifacts/, data/simulated/ and results/ (all generated on your machine).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(title, *cmd):
    print(f"\n=== {title} ===", flush=True)
    code = subprocess.call([PY, *cmd], cwd=ROOT)
    if code != 0:
        sys.exit(f"Step failed: {title}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--seed-check", action="store_true", help="also train a second agent (slow)")
    a = ap.parse_args()
    run("Validate lesson content", "scripts/validate_content.py")
    run("Unit tests", "-m", "pytest", "-q")
    if not a.skip_train:
        run("Train the QWI agent", "-m", "rl.train_agent")
    verify = ["-m", "rl.verify_agent"] + (["--seed-check"] if a.seed_check else [])
    run("Verify the agent", *verify)
    run("Compare with baselines (matched forgetting curve)", "-m", "rl.evaluate_baselines")
    run("Compare with baselines (mismatched forgetting curve)", "-m", "rl.evaluate_baselines", "--world", "power")
    run("Simulate the learner dataset", "-m", "rl.simulate_dataset")
    run("Validate the speed estimator (matched forgetting curve)", "-m", "rl.validate_estimator")
    run("Validate the speed estimator (mismatched forgetting curve)", "-m", "rl.validate_estimator", "--world", "power")
    run("Validate the quiz rules (matched)", "-m", "rl.validate_rules")
    run("Validate the quiz rules (mismatched)", "-m", "rl.validate_rules", "--world", "power")
    print("\nDone. Reports are in results/. Start the app with: python run.py")


if __name__ == "__main__":
    main()
