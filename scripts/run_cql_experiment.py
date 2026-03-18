#!/usr/bin/env python
"""Run CQL baseline experiment on the 3x3 EV corridor grid.

Trains Conservative Q-Learning on the same offline dataset used by DT,
evaluates it, and saves results alongside the other baselines.

Usage::

    python scripts/run_cql_experiment.py
    python scripts/run_cql_experiment.py --data data/hp_sweep.h5 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.cql_baseline import CQLAgent, OfflineRLDataset  # noqa: E402
from src.envs.ev_corridor_env import EVCorridorEnv  # noqa: E402

# ------------------------------------------------------------------
# Dataset generation (fallback when no HDF5 exists)
# ------------------------------------------------------------------

def generate_dataset_if_missing(data_path: str, rows: int = 3, cols: int = 3) -> str:
    """Generate a small offline dataset if the requested file does not exist.

    Collects episodes with a mix of greedy-preempt expert and random policies
    using the DataCollector, then saves to HDF5.

    Parameters
    ----------
    data_path : str
        Desired path for the HDF5 file.
    rows, cols : int
        Grid dimensions.

    Returns
    -------
    str
        The path to the (possibly newly created) dataset.
    """
    if os.path.exists(data_path):
        return data_path

    print(f"Dataset not found at {data_path}. Generating a small one ...")
    from src.baselines.greedy_preempt import GreedyPreemptPolicy
    from src.utils.data_collector import DataCollector

    env = EVCorridorEnv(rows=rows, cols=cols, max_steps=80, seed=42)
    expert = GreedyPreemptPolicy(network=env.network, route=env.ev_route)
    collector = DataCollector(env=env, save_path=data_path)
    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=60,
        num_random=30,
        num_suboptimal=10,
    )
    collector.save_dataset()
    print(f"  Saved {len(collector._episodes)} episodes to {data_path}")
    return data_path


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_cql(
    agent: CQLAgent,
    rows: int = 3,
    cols: int = 3,
    max_steps: int = 80,
    n_episodes: int = 30,
) -> dict[str, float | list[float]]:
    """Evaluate a trained CQL agent on the EV corridor environment.

    Parameters
    ----------
    agent : CQLAgent
        Trained CQL agent.
    rows, cols : int
        Grid dimensions.
    max_steps : int
        Episode time limit.
    n_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    dict
        Aggregated evaluation metrics.
    """
    returns: list[float] = []
    ev_times: list[float] = []
    bg_delays: list[float] = []
    throughputs: list[float] = []
    phase_changes_list: list[int] = []

    for ep in range(n_episodes):
        env = EVCorridorEnv(rows=rows, cols=cols, max_steps=max_steps, seed=123 + ep)
        obs, info = env.reset()
        done = False
        total_r = 0.0
        ep_throughput = 0.0
        ep_phase_changes = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            total_r += r
            ep_throughput += info.get("throughput", 0.0)
            if info.get("phase_changed_for_ev", False):
                ep_phase_changes += 1

        returns.append(total_r)
        ev_times.append(info.get("ev_travel_time", -1))
        bg_delays.append(info.get("background_delay", 0.0))
        throughputs.append(ep_throughput)
        phase_changes_list.append(ep_phase_changes)

    arrived = [t for t in ev_times if t > 0]
    return {
        "avg_return": round(float(np.mean(returns)), 2),
        "std_return": round(float(np.std(returns)), 2),
        "avg_ev_travel_time": round(float(np.mean(arrived)), 2) if arrived else -1.0,
        "arrival_rate": round(len(arrived) / n_episodes, 3),
        "avg_background_delay": round(float(np.mean(bg_delays)), 2),
        "avg_throughput": round(float(np.mean(throughputs)), 2),
        "avg_phase_changes_for_ev": round(float(np.mean(phase_changes_list)), 2),
        "num_episodes": n_episodes,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CQL baseline experiment")
    parser.add_argument(
        "--data", type=str, default="data/hp_sweep.h5",
        help="Path to HDF5 offline dataset.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Q-net hidden dim.")
    parser.add_argument("--alpha", type=float, default=1.0, help="CQL penalty weight.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows.")
    parser.add_argument("--cols", type=int, default=3, help="Grid cols.")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Eval episodes.")
    parser.add_argument(
        "--output", type=str, default="results/cql_results.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CQL Baseline Experiment — EV Corridor")
    print("=" * 60)

    # Ensure dataset exists
    data_path = os.path.join(str(_PROJECT_ROOT), args.data)
    data_path = generate_dataset_if_missing(data_path, rows=args.rows, cols=args.cols)

    # Load offline dataset
    print(f"\nLoading dataset from {data_path} ...")
    dataset = OfflineRLDataset(data_path)
    print(f"  Transitions: {len(dataset)}")
    print(f"  State dim:   {dataset.states.shape[1]}")
    act_dim = max(int(dataset.actions.max()) + 1, 4)
    print(f"  Action dim:  {act_dim}")

    # Create CQL agent
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    agent = CQLAgent(
        state_dim=dataset.states.shape[1],
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        device=device,
    )
    print(f"  Device:      {device}")
    print(f"  Alpha (CQL): {args.alpha}")

    # Train
    print(f"\nTraining CQL for {args.epochs} epochs ...")
    t0 = time.time()
    history = agent.train_offline(
        dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        log_interval=max(1, args.epochs // 10),
    )
    train_time = time.time() - t0
    print(f"Training complete in {train_time:.1f}s")

    # Save model
    model_path = os.path.join(str(_PROJECT_ROOT), "models", "cql_3x3.pt")
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    print(f"\nEvaluating over {args.eval_episodes} episodes ...")
    eval_results = evaluate_cql(
        agent,
        rows=args.rows,
        cols=args.cols,
        max_steps=80,
        n_episodes=args.eval_episodes,
    )

    print(f"  Avg return:       {eval_results['avg_return']}")
    print(f"  Avg EV time:      {eval_results['avg_ev_travel_time']}")
    print(f"  Arrival rate:     {eval_results['arrival_rate']}")
    print(f"  Avg bg delay:     {eval_results['avg_background_delay']}")
    print(f"  Avg throughput:   {eval_results['avg_throughput']}")

    # Combine into output
    output = {
        "CQL": eval_results,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "lr": args.lr,
            "gamma": args.gamma,
            "hidden_dim": args.hidden_dim,
            "train_time_s": round(train_time, 2),
            "final_bellman_loss": round(history["bellman_loss"][-1], 6),
            "final_cql_loss": round(history["cql_loss"][-1], 6),
            "final_total_loss": round(history["total_loss"][-1], 6),
        },
    }

    # Save results
    out_path = os.path.join(str(_PROJECT_ROOT), args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
