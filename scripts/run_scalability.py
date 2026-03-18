#!/usr/bin/env python3
"""Test scalability of EV-DT methods across different network sizes.

Grid sizes tested:
    - 3x3  (9 intersections)
    - 4x4  (16 intersections)
    - 6x6  (36 intersections)
    - 8x8  (64 intersections)

For each grid size, we:
    1. Generate a dataset
    2. Train DT, MADT, and baselines
    3. Evaluate all methods
    4. Record: EV travel time, training time, inference time, memory usage

Results are saved to logs/scalability_results.json.

Usage::

    python scripts/run_scalability.py
    python scripts/run_scalability.py --device cuda --grid-sizes 3 4 6
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_step(name: str, cmd: List[str], timeout: Optional[int] = None) -> int:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, timeout=timeout)
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return result.returncode


# ---------------------------------------------------------------------------
# Grid-size config builder
# ---------------------------------------------------------------------------

GRID_CONFIGS = [
    {"rows": 3, "cols": 3, "label": "3x3", "n_intersections": 9},
    {"rows": 4, "cols": 4, "label": "4x4", "n_intersections": 16},
    {"rows": 6, "cols": 6, "label": "6x6", "n_intersections": 36},
    {"rows": 8, "cols": 8, "label": "8x8", "n_intersections": 64},
]


def build_grid_config(
    base_cfg: Dict[str, Any],
    rows: int,
    cols: int,
    label: str,
) -> Dict[str, Any]:
    """Build a config for a specific grid size, scaling hyperparameters."""
    cfg = copy.deepcopy(base_cfg)
    n_intersections = rows * cols

    # Environment
    cfg["env"]["network"] = f"grid-{label}-v0"

    # Scale episode length with network size
    base_steps = cfg["env"].get("max_episode_steps", 300)
    scale_factor = n_intersections / 16.0  # 4x4 is the baseline
    cfg["env"]["max_episode_steps"] = int(base_steps * max(1.0, scale_factor ** 0.5))

    # Dataset -- scale episodes with network complexity
    base_episodes = cfg["dataset"].get("num_episodes", 5000)
    cfg["dataset"]["num_episodes"] = int(base_episodes * max(1.0, scale_factor ** 0.5))
    cfg["dataset"]["save_path"] = f"data/offline_dataset_{label}.h5"

    # DT -- scale context and max_ep_len
    cfg["dt"]["max_ep_len"] = cfg["env"]["max_episode_steps"]
    # Keep context_length reasonable but scale slightly for bigger networks
    if n_intersections > 36:
        cfg["dt"]["context_length"] = min(40, cfg["dt"]["context_length"] + 10)

    # MADT -- adjust for more agents
    cfg["madt"]["max_ep_len"] = cfg["env"]["max_episode_steps"]
    if n_intersections > 36:
        cfg["madt"]["context_length"] = min(30, cfg["madt"]["context_length"] + 5)
        # For large networks, reduce batch size to fit in memory
        cfg["madt"]["batch_size"] = max(16, cfg["madt"]["batch_size"] // 2)

    # Evaluation scenarios
    cfg["eval"]["scenarios"] = [f"grid-{label}-v0"]

    # Logging
    cfg["logging"]["project"] = f"ev-dt-scalability-{label}"

    return cfg


# ---------------------------------------------------------------------------
# Timing and memory measurement
# ---------------------------------------------------------------------------

def measure_inference_time(
    config_path: str,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Measure per-step inference time for each method.

    Returns a dict mapping method name to mean inference time per step (seconds).
    """
    import torch
    import numpy as np

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_times: Dict[str, float] = {}

    # DT inference time
    dt_path = Path(_PROJECT_ROOT) / "models" / "dt_best.pt"
    if dt_path.exists():
        from src.models.decision_transformer import DecisionTransformer

        ckpt = torch.load(dt_path, map_location=device, weights_only=False)
        dt_model = DecisionTransformer(
            state_dim=ckpt["state_dim"],
            act_dim=ckpt["act_dim"],
            hidden_dim=ckpt["config"]["embed_dim"],
            n_layers=ckpt["config"]["n_layers"],
            n_heads=ckpt["config"]["n_heads"],
            max_length=ckpt["config"]["context_length"],
        ).to(device)
        dt_model.load_state_dict(ckpt["model_state_dict"])
        dt_model.eval()

        # Warm up
        dummy_s = torch.randn(1, ckpt["config"]["context_length"],
                              ckpt["state_dim"], device=device)
        dummy_a = torch.zeros(1, ckpt["config"]["context_length"],
                              dtype=torch.long, device=device)
        dummy_r = torch.zeros(1, ckpt["config"]["context_length"], 1,
                              device=device)
        dummy_t = torch.arange(ckpt["config"]["context_length"],
                               device=device).unsqueeze(0)

        for _ in range(5):
            dt_model.get_action(dummy_s, dummy_a, dummy_r, dummy_t)

        # Measure
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            dt_model.get_action(dummy_s, dummy_a, dummy_r, dummy_t)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        inference_times["DT"] = float(np.mean(times))

    # MADT inference time
    madt_path = Path(_PROJECT_ROOT) / "models" / "madt_best.pt"
    if madt_path.exists():
        from src.models.madt import MultiAgentDecisionTransformer

        ckpt = torch.load(madt_path, map_location=device, weights_only=False)
        madt_model = MultiAgentDecisionTransformer(
            state_dim=ckpt["state_dim"],
            act_dim=ckpt["act_dim"],
            n_agents=ckpt["n_agents"],
            adj_matrix=ckpt["adj_matrix"],
            hidden_dim=ckpt["config"]["embed_dim"],
            n_layers=ckpt["config"]["n_layers"],
            n_heads=ckpt["config"]["n_heads"],
            gat_heads=ckpt["config"]["gat_heads"],
            gat_layers=ckpt["config"]["gat_layers"],
            max_length=ckpt["config"]["context_length"],
        ).to(device)
        madt_model.load_state_dict(ckpt["model_state_dict"])
        madt_model.eval()

        n_agents = ckpt["n_agents"]
        ctx = ckpt["config"]["context_length"]
        dummy_s = torch.randn(1, n_agents, ctx, ckpt["state_dim"], device=device)
        dummy_a = torch.zeros(1, n_agents, ctx, dtype=torch.long, device=device)
        dummy_r = torch.zeros(1, n_agents, ctx, 1, device=device)
        dummy_t = torch.arange(ctx, device=device).unsqueeze(0).unsqueeze(0).expand(
            1, n_agents, ctx
        )

        for _ in range(5):
            madt_model.get_action(dummy_s, dummy_a, dummy_r, dummy_t, agent_idx=0)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            madt_model.get_action(dummy_s, dummy_a, dummy_r, dummy_t, agent_idx=0)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        inference_times["MADT"] = float(np.mean(times))

    return inference_times


def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB, or 0.0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scalability_experiment(
    grid_spec: Dict[str, Any],
    base_config: Dict[str, Any],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Run the full pipeline for a single grid size."""
    rows = grid_spec["rows"]
    cols = grid_spec["cols"]
    label = grid_spec["label"]
    n_intersections = grid_spec["n_intersections"]

    print(f"\n{'#'*70}")
    print(f"  SCALABILITY TEST: {label} grid ({n_intersections} intersections)")
    print(f"{'#'*70}")

    # Build config for this grid size
    cfg = build_grid_config(base_config, rows, cols, label)

    tmp_dir = Path(_PROJECT_ROOT) / "configs" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(tmp_dir / f"scalability_{label}.yaml")
    save_config(cfg, config_path)

    python = sys.executable
    result: Dict[str, Any] = {
        "grid_size": label,
        "n_intersections": n_intersections,
        "config_path": config_path,
        "methods": {},
    }

    # 1. Generate dataset
    t0 = time.time()
    run_step(
        f"[{label}] Generate Dataset",
        [python, "scripts/generate_dataset.py",
         "--config", config_path,
         "--seed", str(seed)],
    )
    result["data_generation_time_sec"] = time.time() - t0

    # 2. Train DT
    t0 = time.time()
    rc = run_step(
        f"[{label}] Train DT",
        [python, "scripts/train_dt.py",
         "--config", config_path,
         "--device", device],
    )
    dt_train_time = time.time() - t0
    result["methods"]["DT"] = {
        "training_time_sec": dt_train_time,
        "training_success": rc == 0,
    }

    # 3. Train MADT
    t0 = time.time()
    rc = run_step(
        f"[{label}] Train MADT",
        [python, "scripts/train_madt.py",
         "--config", config_path,
         "--device", device],
    )
    madt_train_time = time.time() - t0
    result["methods"]["MADT"] = {
        "training_time_sec": madt_train_time,
        "training_success": rc == 0,
    }

    # 4. Train baselines
    for baseline in ["ppo", "dqn"]:
        t0 = time.time()
        rc = run_step(
            f"[{label}] Train {baseline.upper()}",
            [python, "scripts/train_baselines.py",
             "--config", config_path,
             "--method", baseline],
        )
        result["methods"][baseline.upper()] = {
            "training_time_sec": time.time() - t0,
            "training_success": rc == 0,
        }

    # 5. Evaluate all
    run_step(
        f"[{label}] Evaluate",
        [python, "scripts/evaluate.py",
         "--config", config_path,
         "--scenarios", f"grid-{label}-v0"],
    )

    # Load evaluation results
    eval_path = Path(_PROJECT_ROOT) / "logs" / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        result["evaluation"] = eval_data
    else:
        result["evaluation"] = {}

    # 6. Measure inference times
    try:
        inference_times = measure_inference_time(config_path)
        for method, inf_time in inference_times.items():
            if method in result["methods"]:
                result["methods"][method]["inference_time_per_step_sec"] = inf_time
    except Exception as exc:
        print(f"  WARNING: could not measure inference time: {exc}")
        traceback.print_exc()

    # 7. GPU memory
    result["peak_gpu_memory_mb"] = get_gpu_memory_mb()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test scalability across network sizes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base configuration YAML.",
    )
    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Grid sizes to test (e.g., 3 4 6 8). Default: all.",
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
        default="logs/scalability_results.json",
        help="Output path for results JSON.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)

    # Filter grid configs if specific sizes requested
    if args.grid_sizes is not None:
        grids = [g for g in GRID_CONFIGS if g["rows"] in args.grid_sizes]
    else:
        grids = GRID_CONFIGS

    print("=" * 70)
    print("  EV-DT Scalability Study")
    print("=" * 70)
    print(f"  Base config : {args.config}")
    print(f"  Grid sizes  : {[g['label'] for g in grids]}")
    print(f"  Device      : {args.device}")
    print(f"  Seed        : {args.seed}")
    print("=" * 70)

    all_results: Dict[str, Any] = {
        "base_config": args.config,
        "seed": args.seed,
        "grid_experiments": {},
    }

    for grid_spec in grids:
        label = grid_spec["label"]
        try:
            result = run_scalability_experiment(
                grid_spec=grid_spec,
                base_config=base_config,
                device=args.device,
                seed=args.seed,
            )
            all_results["grid_experiments"][label] = result
        except Exception as exc:
            print(f"\n  ERROR running {label}: {exc}")
            traceback.print_exc()
            all_results["grid_experiments"][label] = {
                "grid_size": label,
                "error": str(exc),
            }

    # Save results
    output_path = Path(_PROJECT_ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("  SCALABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Grid':<8} {'N':<6} {'DT Train(s)':<14} {'MADT Train(s)':<14} "
          f"{'DT Inf(ms)':<12} {'MADT Inf(ms)':<12} {'GPU(MB)':<10}")
    print(f"  {'-'*8} {'-'*6} {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")

    for grid_spec in grids:
        label = grid_spec["label"]
        exp = all_results["grid_experiments"].get(label, {})
        methods = exp.get("methods", {})

        dt_time = methods.get("DT", {}).get("training_time_sec", -1)
        madt_time = methods.get("MADT", {}).get("training_time_sec", -1)
        dt_inf = methods.get("DT", {}).get("inference_time_per_step_sec", -1)
        madt_inf = methods.get("MADT", {}).get("inference_time_per_step_sec", -1)
        gpu_mem = exp.get("peak_gpu_memory_mb", 0)

        dt_inf_ms = f"{dt_inf*1000:.2f}" if dt_inf >= 0 else "N/A"
        madt_inf_ms = f"{madt_inf*1000:.2f}" if madt_inf >= 0 else "N/A"

        print(f"  {label:<8} {grid_spec['n_intersections']:<6} "
              f"{dt_time:<14.1f} {madt_time:<14.1f} "
              f"{dt_inf_ms:<12} {madt_inf_ms:<12} {gpu_mem:<10.1f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
