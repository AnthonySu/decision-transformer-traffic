#!/usr/bin/env python3
"""Master script: generate data, train all methods, evaluate, and produce results."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]):
    """Run a subprocess step with logging."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def main(config: str = "configs/default.yaml", skip_data: bool = False):
    python = sys.executable

    # Step 1: Generate offline dataset
    if not skip_data:
        run_step(
            "Generate Offline Dataset",
            [python, "scripts/generate_dataset.py", "--config", config],
        )

    # Step 2: Train Decision Transformer
    run_step(
        "Train Decision Transformer",
        [python, "scripts/train_dt.py", "--config", config],
    )

    # Step 3: Train Multi-Agent Decision Transformer
    run_step(
        "Train MADT",
        [python, "scripts/train_madt.py", "--config", config],
    )

    # Step 4: Train PPO baseline
    run_step(
        "Train PPO Baseline",
        [python, "scripts/train_baselines.py", "--config", config, "--method", "ppo"],
    )

    # Step 5: Train DQN baseline
    run_step(
        "Train DQN Baseline",
        [python, "scripts/train_baselines.py", "--config", config, "--method", "dqn"],
    )

    # Step 6: Evaluate all methods
    run_step(
        "Evaluate All Methods",
        [python, "scripts/evaluate.py", "--config", config],
    )

    print(f"\n{'='*60}")
    print("  ALL STEPS COMPLETE")
    print(f"{'='*60}")
    print("Results: logs/evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full EV-DT pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-data", action="store_true", help="Skip data generation")
    args = parser.parse_args()
    main(args.config, args.skip_data)
