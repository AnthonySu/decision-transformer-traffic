#!/usr/bin/env python3
"""Quick demo: sanity-check that the EV corridor environment behaves correctly.

Runs two episodes side-by-side:
  1. Random actions  -- the EV stumbles through the network
  2. Greedy preemption -- the EV gets green lights along its route

Prints step-by-step progress and compares key metrics at the end.
Optionally renders the environment at key moments (EV start, midpoint, end).

Usage::

    python scripts/quick_demo.py
    python scripts/quick_demo.py --rows 4 --cols 4 --max-steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.baselines.fixed_time_evp import FixedTimeEVP
from src.baselines.greedy_preempt import GreedyPreemptPolicy
from src.envs.ev_corridor_env import EVCorridorEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIV = "-" * 60


def _run_episode(env: EVCorridorEnv, policy_fn, policy_name: str,
                 verbose: bool = True) -> dict:
    """Run one episode, printing step-by-step progress if verbose."""
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    render_snapshots: list[str] = []

    route_len = info.get("route_length", 1)

    if verbose:
        print(f"\n{_DIV}")
        print(f"  Episode: {policy_name}")
        print(f"  Route length: {route_len} links")
        print(_DIV)

    # Capture initial render
    render_text = env.render()
    if render_text:
        render_snapshots.append(("start", render_text))

    while not done:
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if verbose and (step <= 5 or step % 10 == 0 or done):
            ev_idx = info.get("ev_link_idx", 0)
            ev_prog = info.get("ev_progress", 0.0)
            arrived = info.get("ev_arrived", False)
            queue = info.get("total_queue", 0.0)
            print(f"    Step {step:3d} | reward={reward:+7.2f} | "
                  f"EV link={ev_idx}/{route_len-1} prog={ev_prog:.2f} | "
                  f"arrived={arrived} | queue={queue:.1f}")

    # Capture final render
    render_text = env.render()
    if render_text:
        render_snapshots.append(("end", render_text))

    ev_arrived = info.get("ev_arrived", False)
    ev_travel_time = step if ev_arrived else -1

    if verbose:
        print(f"  {'--- Episode Summary ---':^40}")
        print(f"    Total reward:    {total_reward:.2f}")
        print(f"    Steps:           {step}")
        print(f"    EV arrived:      {ev_arrived}")
        print(f"    EV travel time:  {ev_travel_time if ev_travel_time > 0 else 'N/A'}")

    return {
        "policy": policy_name,
        "total_reward": total_reward,
        "steps": step,
        "ev_arrived": ev_arrived,
        "ev_travel_time": ev_travel_time,
        "renders": render_snapshots,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick demo of EV corridor environment behaviour"
    )
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=3, help="Grid cols")
    parser.add_argument("--max-steps", type=int, default=80, help="Max episode steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    np.random.seed(args.seed)
    verbose = not args.quiet

    print("=" * 60)
    print("  EV-DT Quick Demo")
    print(f"  Grid: {args.rows}x{args.cols}  |  Max steps: {args.max_steps}  |  Seed: {args.seed}")
    print("=" * 60)

    # Create shared environment (same seed for fair comparison)
    env = EVCorridorEnv(
        rows=args.rows,
        cols=args.cols,
        max_steps=args.max_steps,
        seed=args.seed,
        render_mode="ansi",
    )

    # ---- Episode 1: Random ----
    result_random = _run_episode(
        env,
        policy_fn=lambda obs, info: env.action_space.sample(),
        policy_name="Random Actions",
        verbose=verbose,
    )

    # Reset with same seed for fair comparison
    env = EVCorridorEnv(
        rows=args.rows,
        cols=args.cols,
        max_steps=args.max_steps,
        seed=args.seed,
        render_mode="ansi",
    )

    # ---- Episode 2: Greedy Preemption ----
    # Need to reset once to get the route, then build the policy
    obs, info = env.reset()
    greedy = GreedyPreemptPolicy(network=env._network, route=env._route_intersections)

    # Re-reset with same conditions
    env_greedy = EVCorridorEnv(
        rows=args.rows,
        cols=args.cols,
        max_steps=args.max_steps,
        seed=args.seed,
        render_mode="ansi",
    )
    greedy_env_obs, greedy_env_info = env_greedy.reset()
    greedy2 = GreedyPreemptPolicy(
        network=env_greedy._network, route=env_greedy._route_intersections
    )
    result_greedy = _run_episode(
        env_greedy,
        policy_fn=lambda obs, info: greedy2.select_action(obs, info),
        policy_name="Greedy Preemption",
        verbose=verbose,
    )

    # ---- Comparison ----
    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25s} {'Random':>12s} {'Greedy':>12s}")
    print(f"  {'-'*50}")

    for key in ["total_reward", "steps", "ev_arrived", "ev_travel_time"]:
        rv = result_random[key]
        gv = result_greedy[key]
        if isinstance(rv, float):
            print(f"  {key:<25s} {rv:>12.2f} {gv:>12.2f}")
        elif isinstance(rv, bool):
            print(f"  {key:<25s} {str(rv):>12s} {str(gv):>12s}")
        else:
            print(f"  {key:<25s} {str(rv):>12s} {str(gv):>12s}")

    # Show render snapshots at key moments
    print(f"\n{'=' * 60}")
    print("  ENVIRONMENT RENDERS")
    print(f"{'=' * 60}")

    for result in [result_random, result_greedy]:
        policy = result["policy"]
        for label, text in result["renders"]:
            print(f"\n  [{policy} - {label}]")
            for line in text.split("\n"):
                print(f"    {line}")

    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
