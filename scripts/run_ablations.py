#!/usr/bin/env python3
"""Run ablation studies for the EV-DT project.

Each ablation removes or modifies one component and measures impact on
EV travel time, background delay, and other metrics.  Results are saved
to logs/ablation_results.json.

Ablations
---------
1. no_gat          -- MADT without graph attention (agents don't communicate)
2. no_rtg          -- DT trained without return-to-go conditioning
3. short_context   -- DT with context_length=5 vs default 30
4. no_mixed_data   -- DT trained on expert-only data (no suboptimal)
5. small_model     -- DT with 2 layers instead of 4

Usage::

    python scripts/run_ablations.py
    python scripts/run_ablations.py --config configs/default.yaml --device cuda
    python scripts/run_ablations.py --ablations no_gat short_context
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Write a config dict to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_step(name: str, cmd: List[str], timeout: Optional[int] = None) -> int:
    """Run a subprocess step with logging."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, timeout=timeout)
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return result.returncode


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

def _make_no_gat(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation: remove graph attention from MADT."""
    cfg = copy.deepcopy(cfg)
    cfg["madt"]["gat_layers"] = 0
    cfg["madt"]["gat_heads"] = 0
    return cfg


def _make_no_rtg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation: disable return-to-go conditioning."""
    cfg = copy.deepcopy(cfg)
    cfg["dt"]["use_return_conditioning"] = False
    cfg["dt"]["target_returns"] = [0.0]
    cfg["madt"]["use_return_conditioning"] = False
    cfg["madt"]["target_returns"] = [0.0]
    return cfg


def _make_short_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation: short context window (5 instead of 30 / 20)."""
    cfg = copy.deepcopy(cfg)
    cfg["dt"]["context_length"] = 5
    cfg["madt"]["context_length"] = 5
    return cfg


def _make_no_mixed_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation: expert-only data (no suboptimal trajectories)."""
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["mixed_quality"] = False
    cfg["dataset"]["suboptimal_ratio"] = 0.0
    # Use a separate dataset file so we don't overwrite the main one
    original_path = cfg["dataset"].get("save_path", "data/offline_dataset.h5")
    cfg["dataset"]["save_path"] = original_path.replace(".h5", "_expert_only.h5")
    return cfg


def _make_small_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ablation: small model (2 layers instead of 4)."""
    cfg = copy.deepcopy(cfg)
    cfg["dt"]["n_layers"] = 2
    cfg["dt"]["n_heads"] = 2
    cfg["madt"]["n_layers"] = 2
    cfg["madt"]["n_heads"] = 2
    return cfg


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_gat": {
        "description": "MADT without graph attention (no inter-agent communication)",
        "modify_fn": _make_no_gat,
        "train_methods": ["madt"],
    },
    "no_rtg": {
        "description": "DT/MADT without return-to-go conditioning",
        "modify_fn": _make_no_rtg,
        "train_methods": ["dt", "madt"],
    },
    "short_context": {
        "description": "DT/MADT with context_length=5 (vs 30/20)",
        "modify_fn": _make_short_context,
        "train_methods": ["dt", "madt"],
    },
    "no_mixed_data": {
        "description": "DT trained on expert-only data (no suboptimal episodes)",
        "modify_fn": _make_no_mixed_data,
        "train_methods": ["dt"],
        "regenerate_data": True,
    },
    "small_model": {
        "description": "DT/MADT with 2 layers instead of 4/3",
        "modify_fn": _make_small_model,
        "train_methods": ["dt", "madt"],
    },
}


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_single_ablation(
    ablation_name: str,
    base_config: Dict[str, Any],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a single ablation experiment end-to-end.

    Returns a dict with the ablation name, description, config diff,
    and evaluation metrics.
    """
    ablation_spec = ABLATIONS[ablation_name]
    modify_fn = ablation_spec["modify_fn"]
    train_methods = ablation_spec["train_methods"]
    regenerate_data = ablation_spec.get("regenerate_data", False)

    print(f"\n{'#'*70}")
    print(f"  ABLATION: {ablation_name}")
    print(f"  {ablation_spec['description']}")
    print(f"{'#'*70}")

    # Build modified config
    ablated_cfg = modify_fn(base_config)

    # Write temporary config file
    tmp_dir = Path(_PROJECT_ROOT) / "configs" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_config_path = str(tmp_dir / f"ablation_{ablation_name}.yaml")
    save_config(ablated_cfg, tmp_config_path)

    python = sys.executable
    t_start = time.time()

    # Step 1: Regenerate data if needed (e.g., expert-only ablation)
    if regenerate_data:
        run_step(
            f"[{ablation_name}] Generate Dataset",
            [python, "scripts/generate_dataset.py",
             "--config", tmp_config_path,
             "--seed", str(seed),
             "--skip-multi-agent"],
        )

    # Step 2: Train ablated methods
    for method in train_methods:
        if method == "dt":
            run_step(
                f"[{ablation_name}] Train DT",
                [python, "scripts/train_dt.py",
                 "--config", tmp_config_path,
                 "--device", device],
            )
        elif method == "madt":
            run_step(
                f"[{ablation_name}] Train MADT",
                [python, "scripts/train_madt.py",
                 "--config", tmp_config_path,
                 "--device", device],
            )

    # Step 3: Evaluate
    run_step(
        f"[{ablation_name}] Evaluate",
        [python, "scripts/evaluate.py",
         "--config", tmp_config_path],
    )

    training_time = time.time() - t_start

    # Step 4: Load evaluation results
    eval_results_path = Path(_PROJECT_ROOT) / "logs" / "evaluation_results.json"
    eval_results: Dict[str, Any] = {}
    if eval_results_path.exists():
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)

    return {
        "ablation": ablation_name,
        "description": ablation_spec["description"],
        "train_methods": train_methods,
        "training_time_sec": training_time,
        "config_path": tmp_config_path,
        "evaluation": eval_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation studies for EV-DT project."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base configuration YAML.",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        choices=list(ABLATIONS.keys()),
        help="Which ablations to run. Default: all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/ablation_results.json",
        help="Output path for results JSON.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    ablation_names = args.ablations or list(ABLATIONS.keys())

    print("=" * 70)
    print("  EV-DT Ablation Studies")
    print("=" * 70)
    print(f"  Base config : {args.config}")
    print(f"  Ablations   : {ablation_names}")
    print(f"  Device      : {args.device}")
    print(f"  Seed        : {args.seed}")
    print("=" * 70)

    all_results: Dict[str, Any] = {
        "base_config": args.config,
        "seed": args.seed,
        "ablations": {},
    }

    for name in ablation_names:
        result = run_single_ablation(
            ablation_name=name,
            base_config=base_config,
            device=args.device,
            seed=args.seed,
        )
        all_results["ablations"][name] = result

    # Save results
    output_path = Path(_PROJECT_ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("  ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    for name, result in all_results["ablations"].items():
        print(f"\n  {name}: {result['description']}")
        print(f"    Training time: {result['training_time_sec']:.1f}s")
        eval_data = result.get("evaluation", {})
        for scenario, methods in eval_data.items():
            for method, metrics in methods.items():
                ev_time = metrics.get("mean_ev_travel_time", "N/A")
                bg_delay = metrics.get("background_delay_mean", "N/A")
                print(f"    [{scenario}] {method}: EV time={ev_time}, bg delay={bg_delay}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
