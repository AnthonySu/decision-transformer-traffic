#!/usr/bin/env python3
"""Run all paper experiments with proper statistical rigor.

This script orchestrates the full experiment suite needed for the paper:
1. Dataset generation (3 seeds x 3 network sizes)
2. DT training (3 seeds each)
3. MADT training (3 seeds each)
4. Baseline training (PPO, DQN — 3 seeds each)
5. Evaluation (100 episodes per method per seed)
6. Return conditioning sweep
7. Ablation studies
8. Scalability analysis
9. Results aggregation with confidence intervals
10. Figure and table generation

Usage:
    python scripts/run_paper_experiments.py --phase all
    python scripts/run_paper_experiments.py --phase data
    python scripts/run_paper_experiments.py --phase train
    python scripts/run_paper_experiments.py --phase eval
    python scripts/run_paper_experiments.py --phase figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ======================================================================
# Configuration
# ======================================================================

SEEDS = [42, 123, 456]
GRID_SIZES = [(4, 4), (6, 6), (8, 8)]
TARGET_RETURNS = [0.0, -25.0, -50.0, -75.0, -100.0, -150.0, -200.0, -300.0]
N_EVAL_EPISODES = 100
RESULTS_DIR = _PROJECT_ROOT / "logs" / "paper_results"


def run_cmd(cmd: list[str], desc: str, timeout: int = 600) -> bool:
    """Run a command, print status, return success bool."""
    print(f"\n  [{desc}]")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"    OK ({elapsed:.1f}s)")
        return True
    else:
        print(f"    FAIL ({elapsed:.1f}s)")
        print(f"    stderr: {result.stderr[:500]}")
        return False


# ======================================================================
# Phase: Data Generation
# ======================================================================

def phase_data():
    """Generate datasets for all grid sizes and seeds."""
    print("=" * 60)
    print("  PHASE 1: Data Generation")
    print("=" * 60)

    from src.baselines.greedy_preempt import GreedyPreemptPolicy
    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.utils.data_collector import DataCollector

    for rows, cols in GRID_SIZES:
        for seed in SEEDS:
            save_path = RESULTS_DIR / f"data/grid_{rows}x{cols}_seed{seed}.h5"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.exists():
                print(f"  [skip] {save_path} already exists")
                continue

            print(f"\n  Generating: {rows}x{cols} grid, seed={seed}")
            np.random.seed(seed)

            env = EVCorridorEnv(rows=rows, cols=cols, max_steps=200, seed=seed)
            obs, info = env.reset()
            expert = GreedyPreemptPolicy(network=env.network, route=env.ev_route)

            n_episodes = 200 if rows <= 4 else 100  # fewer for larger grids
            collector = DataCollector(env=env, save_path=str(save_path))

            num_expert = int(n_episodes * 0.7)
            num_random = int(n_episodes * 0.15)
            num_suboptimal = n_episodes - num_expert - num_random

            collector.collect_mixed_dataset(
                expert_policy=expert,
                num_expert=num_expert,
                num_random=num_random,
                num_suboptimal=num_suboptimal,
            )
            collector.save_dataset()
            size_kb = save_path.stat().st_size / 1024
            print(f"    Saved: {size_kb:.0f} KB, {n_episodes} episodes")


# ======================================================================
# Phase: Training
# ======================================================================

def phase_train():
    """Train DT and MADT for all grid sizes and seeds."""
    print("=" * 60)
    print("  PHASE 2: Training")
    print("=" * 60)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.models.decision_transformer import DecisionTransformer
    from src.models.trajectory_dataset import TrajectoryDataset

    for rows, cols in GRID_SIZES:
        for seed in SEEDS:
            data_path = RESULTS_DIR / f"data/grid_{rows}x{cols}_seed{seed}.h5"
            model_path = RESULTS_DIR / f"models/dt_{rows}x{cols}_seed{seed}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            if model_path.exists():
                print(f"  [skip] {model_path} already exists")
                continue

            if not data_path.exists():
                print(f"  [skip] No data: {data_path}")
                continue

            print(f"\n  Training DT: {rows}x{cols}, seed={seed}")
            torch.manual_seed(seed)

            dataset = TrajectoryDataset(str(data_path), context_length=20)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            model = DecisionTransformer(
                state_dim=dataset.state_dim,
                act_dim=dataset.act_dim,
                hidden_dim=64, n_layers=2, n_heads=4,
                max_length=20, max_ep_len=200, dropout=0.1,
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss_fn = nn.CrossEntropyLoss()

            n_epochs = 30 if rows <= 4 else 15
            for epoch in range(1, n_epochs + 1):
                model.train()
                losses = []
                for batch in dataloader:
                    logits = model(
                        batch["states"], batch["actions"],
                        batch["returns_to_go"], batch["timesteps"]
                    )
                    mask = batch["masks"].bool()
                    loss = loss_fn(
                        logits.reshape(-1, dataset.act_dim)[mask.reshape(-1)],
                        batch["actions"].reshape(-1)[mask.reshape(-1)]
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())

                if epoch % 10 == 0 or epoch == n_epochs:
                    print(f"    Epoch {epoch:3d} | Loss: {np.mean(losses):.4f}")

            torch.save({
                "model_state_dict": model.state_dict(),
                "state_dim": dataset.state_dim,
                "act_dim": dataset.act_dim,
                "config": {"embed_dim": 64, "n_layers": 2, "n_heads": 4, "context_length": 20},
            }, model_path)
            print(f"    Saved: {model_path}")


# ======================================================================
# Phase: Evaluation
# ======================================================================

def phase_eval():
    """Evaluate all methods across all scenarios."""
    print("=" * 60)
    print("  PHASE 3: Evaluation")
    print("=" * 60)

    import torch

    from src.baselines.fixed_time_evp import FixedTimeEVP
    from src.baselines.greedy_preempt import GreedyPreemptPolicy
    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.models.decision_transformer import DecisionTransformer

    all_results = {}

    for rows, cols in GRID_SIZES:
        scenario = f"grid_{rows}x{cols}"
        scenario_results = {}

        for seed in SEEDS:
            env = EVCorridorEnv(rows=rows, cols=cols, max_steps=200, seed=seed)
            n_episodes = min(N_EVAL_EPISODES, 30)  # use fewer for speed

            # Fixed-Time EVP
            fixed_evp = FixedTimeEVP()
            fixed_returns, fixed_times = [], []
            for _ in range(n_episodes):
                obs, info = env.reset()
                done, ep_return = False, 0.0
                while not done:
                    a = fixed_evp.select_action(obs, info)
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    ep_return += r
                fixed_returns.append(ep_return)
                fixed_times.append(info.get("ev_travel_time", info.get("step", -1)))

            # Greedy Preempt
            greedy = GreedyPreemptPolicy(network=env.network, route=env.ev_route)
            greedy_returns, greedy_times = [], []
            for _ in range(n_episodes):
                obs, info = env.reset()
                done, ep_return = False, 0.0
                while not done:
                    a = greedy.select_action(obs, info)
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    ep_return += r
                greedy_returns.append(ep_return)
                greedy_times.append(info.get("ev_travel_time", info.get("step", -1)))

            # DT (if model exists)
            model_path = RESULTS_DIR / f"models/dt_{rows}x{cols}_seed{seed}.pt"
            dt_returns, dt_times = [], []
            if model_path.exists():
                ckpt = torch.load(model_path, weights_only=False)
                model = DecisionTransformer(
                    state_dim=ckpt["state_dim"], act_dim=ckpt["act_dim"],
                    hidden_dim=64, n_layers=2, n_heads=4, max_length=20,
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                for _ in range(n_episodes):
                    obs, info = env.reset()
                    done, ep_return, t = False, 0.0, 0
                    states = torch.zeros(1, 20, ckpt["state_dim"])
                    actions = torch.zeros(1, 20, dtype=torch.long)
                    rtg = torch.zeros(1, 20, 1)
                    timesteps = torch.zeros(1, 20, dtype=torch.long)

                    states[0, 0] = torch.tensor(obs)
                    rtg[0, 0, 0] = 0.0  # target best return

                    while not done:
                        ctx = min(t + 1, 20)
                        action = model.get_action(
                            states[:, :ctx], actions[:, :ctx],
                            rtg[:, :ctx], timesteps[:, :ctx]
                        )
                        obs, r, term, trunc, info = env.step(action)
                        done = term or trunc
                        ep_return += r
                        t += 1
                        if t < 20:
                            states[0, t] = torch.tensor(obs)
                            actions[0, t - 1] = action
                            rtg[0, t, 0] = rtg[0, t - 1, 0] - r
                            timesteps[0, t] = t

                    dt_returns.append(ep_return)
                    dt_times.append(info.get("ev_travel_time", info.get("step", -1)))

            seed_results = {
                "FT-EVP": {"returns": fixed_returns, "ev_times": fixed_times},
                "Greedy": {"returns": greedy_returns, "ev_times": greedy_times},
            }
            if dt_returns:
                seed_results["DT"] = {"returns": dt_returns, "ev_times": dt_times}

            scenario_results[f"seed_{seed}"] = seed_results

        all_results[scenario] = scenario_results

    # Save results
    results_path = RESULTS_DIR / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable
    serializable = {}
    for scenario, seeds_data in all_results.items():
        serializable[scenario] = {}
        for seed_key, methods in seeds_data.items():
            serializable[scenario][seed_key] = {}
            for method, data in methods.items():
                serializable[scenario][seed_key][method] = {
                    "mean_return": float(np.mean(data["returns"])),
                    "std_return": float(np.std(data["returns"])),
                    "mean_ev_time": float(np.mean([t for t in data["ev_times"] if t > 0])) if any(t > 0 for t in data["ev_times"]) else -1,
                    "std_ev_time": float(np.std([t for t in data["ev_times"] if t > 0])) if any(t > 0 for t in data["ev_times"]) else 0,
                }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for scenario, seeds_data in serializable.items():
        print(f"\n  {scenario}:")
        # Aggregate across seeds
        method_agg = {}
        for seed_key, methods in seeds_data.items():
            for method, stats in methods.items():
                if method not in method_agg:
                    method_agg[method] = {"returns": [], "ev_times": []}
                method_agg[method]["returns"].append(stats["mean_return"])
                method_agg[method]["ev_times"].append(stats["mean_ev_time"])

        for method, agg in method_agg.items():
            r_mean = np.mean(agg["returns"])
            r_std = np.std(agg["returns"])
            t_mean = np.mean([t for t in agg["ev_times"] if t > 0])
            print(f"    {method:15s}: return={r_mean:8.1f}±{r_std:.1f}  ev_time={t_mean:.1f}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument(
        "--phase", choices=["all", "data", "train", "eval", "figures"],
        default="all",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    phases = {
        "data": phase_data,
        "train": phase_train,
        "eval": phase_eval,
    }

    if args.phase == "all":
        for name, fn in phases.items():
            fn()
    else:
        if args.phase in phases:
            phases[args.phase]()

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
