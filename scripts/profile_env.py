#!/usr/bin/env python3
"""Quick environment profiling script for EV-DT.

Measures:
  - Steps/second for different grid sizes
  - Observation and action space sizes
  - Approximate memory usage per environment
  - Comparison of mock CTM vs lightsim backend (if available)

Usage::

    python scripts/profile_env.py
    python scripts/profile_env.py --grids 3 4 6 8 --steps 500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np

from src.envs.ev_corridor_env import EVCorridorEnv
from src.envs.lightsim_adapter import is_lightsim_available
from src.utils.timer import Timer


def _measure_memory_bytes(env: EVCorridorEnv) -> int:
    """Rough RSS-based memory estimate for an environment instance."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    except ImportError:
        return -1


def profile_grid(
    rows: int,
    cols: int,
    num_steps: int,
    use_lightsim: bool = False,
) -> dict:
    """Profile a single grid configuration and return metrics."""
    backend = "lightsim" if use_lightsim else "mock-CTM"
    label = f"{rows}x{cols} ({backend})"

    env = EVCorridorEnv(rows=rows, cols=cols, use_lightsim=use_lightsim, seed=42)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape if hasattr(env.action_space, "shape") else None
    act_nvec = (
        env.action_space.nvec.tolist()
        if hasattr(env.action_space, "nvec")
        else None
    )

    mem_before = _measure_memory_bytes(env)

    # --- Benchmark stepping ---
    obs, info = env.reset(seed=0)
    total_steps = 0
    with Timer(f"stepping {label}") as t:
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            if terminated or truncated:
                obs, info = env.reset()

    steps_per_sec = total_steps / t.elapsed if t.elapsed > 0 else float("inf")
    mem_after = _measure_memory_bytes(env)
    env.close()

    return {
        "label": label,
        "rows": rows,
        "cols": cols,
        "backend": backend,
        "obs_shape": obs_shape,
        "obs_size": int(np.prod(obs_shape)),
        "act_nvec": act_nvec,
        "act_dims": int(np.prod(act_shape)) if act_shape else None,
        "total_steps": total_steps,
        "elapsed_s": round(t.elapsed, 4),
        "steps_per_sec": round(steps_per_sec, 1),
        "mem_rss_mb": round(mem_after / 1024 / 1024, 1) if mem_after > 0 else "N/A",
    }


def print_report(results: list[dict]) -> None:
    """Pretty-print profiling results as a table."""
    print()
    print("=" * 78)
    print("EV-DT Environment Profile")
    print("=" * 78)

    # Header
    header = f"{'Config':<22} {'Obs size':>10} {'Act dims':>10} {'Steps/s':>10} {'Mem (MB)':>10}"
    print(header)
    print("-" * 78)

    for r in results:
        act_str = str(r["act_dims"]) if r["act_dims"] else "multi"
        mem_str = str(r["mem_rss_mb"])
        print(
            f"{r['label']:<22} {r['obs_size']:>10d} {act_str:>10} "
            f"{r['steps_per_sec']:>10.1f} {mem_str:>10}"
        )

    print("-" * 78)
    print()

    # Detail per config
    for r in results:
        print(f"  {r['label']}:")
        print(f"    observation_space.shape = {r['obs_shape']}")
        if r["act_nvec"]:
            nvec = r["act_nvec"]
            print(f"    action_space.nvec length = {len(nvec)}, each = {nvec[0]}")
        print(f"    {r['total_steps']} steps in {r['elapsed_s']}s")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EV corridor environments")
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[3, 4, 6, 8],
        help="Grid sizes to profile (rows=cols)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps per grid size",
    )
    args = parser.parse_args()

    results: list[dict] = []

    # Profile mock CTM backend
    for size in args.grids:
        print(f"Profiling {size}x{size} mock-CTM ...")
        results.append(profile_grid(size, size, args.steps, use_lightsim=False))

    # Profile lightsim backend if available
    if is_lightsim_available():
        print()
        print("lightsim is available -- profiling lightsim backend ...")
        for size in args.grids:
            print(f"Profiling {size}x{size} lightsim ...")
            try:
                results.append(profile_grid(size, size, args.steps, use_lightsim=True))
            except Exception as exc:
                print(f"  lightsim {size}x{size} failed: {exc}")
    else:
        print()
        print("lightsim is NOT installed -- skipping lightsim comparison.")

    print_report(results)


if __name__ == "__main__":
    main()
