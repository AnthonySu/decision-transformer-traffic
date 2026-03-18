#!/usr/bin/env python3
"""Run hyperparameter sensitivity analysis for the DT model.

Sweeps over key hyperparameters to understand their impact:
- Hidden dimension: [32, 64, 128, 256]
- Number of layers: [1, 2, 3, 4]
- Context length: [5, 10, 20, 30]
- Dataset size: [50, 100, 200, 500]

Usage:
    python scripts/run_hyperparameter_sweep.py [--param hidden_dim]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.baselines.greedy_preempt import GreedyPreemptPolicy
from src.envs.ev_corridor_env import EVCorridorEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import TrajectoryDataset
from src.utils.data_collector import DataCollector


def train_and_eval(
    env, dataset_path, hidden_dim=64, n_layers=2, n_heads=4,
    context_length=15, n_epochs=20, n_eval=30,
):
    """Train DT and evaluate, return metrics dict."""
    dataset = TrajectoryDataset(dataset_path, context_length=context_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DecisionTransformer(
        state_dim=dataset.state_dim, act_dim=dataset.act_dim,
        hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
        max_length=context_length, max_ep_len=200,
    )
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    final_loss = 0.0
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for batch in dataloader:
            logits = model(
                batch["states"], batch["actions"],
                batch["returns_to_go"], batch["timesteps"],
            )
            mask = batch["masks"].bool()
            loss = loss_fn(
                logits.reshape(-1, dataset.act_dim)[mask.reshape(-1)],
                batch["actions"].reshape(-1)[mask.reshape(-1)],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        final_loss = np.mean(losses)
    train_time = time.time() - t0

    # Evaluate
    model.eval()
    returns, ev_times = [], []
    for _ in range(n_eval):
        obs, info = env.reset()
        done, r_sum, t = False, 0.0, 0
        states = torch.zeros(1, context_length, dataset.state_dim)
        actions = torch.zeros(1, context_length, dtype=torch.long)
        rtg = torch.zeros(1, context_length, 1)
        timesteps = torch.zeros(1, context_length, dtype=torch.long)
        states[0, 0] = torch.tensor(obs)
        rtg[0, 0, 0] = 0.0

        while not done:
            ctx = min(t + 1, context_length)
            a = model.get_action(
                states[:, :ctx], actions[:, :ctx],
                rtg[:, :ctx], timesteps[:, :ctx],
            )
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            r_sum += r
            t += 1
            if t < context_length:
                states[0, t] = torch.tensor(obs)
                actions[0, t - 1] = a
                rtg[0, t, 0] = rtg[0, t - 1, 0] - r
                timesteps[0, t] = t

        returns.append(r_sum)
        ev_times.append(info.get("ev_travel_time", -1))

    valid_times = [x for x in ev_times if x > 0]
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ev_time": float(np.mean(valid_times)) if valid_times else -1,
        "final_loss": float(final_loss),
        "train_time": float(train_time),
        "n_params": n_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="all",
                        choices=["hidden_dim", "n_layers", "context", "dataset_size", "all"])
    args = parser.parse_args()

    env = EVCorridorEnv(rows=3, cols=3, max_steps=80, seed=42,
                        origin="n0_0", destination="n2_2")

    # Generate base dataset
    data_path = "data/hp_sweep.h5"
    if not Path(data_path).exists():
        obs, info = env.reset()
        expert = GreedyPreemptPolicy(network=env.network, route=env.ev_route)
        collector = DataCollector(env=env, save_path=data_path)
        collector.collect_mixed_dataset(
            expert_policy=expert, num_expert=140,
            num_random=30, num_suboptimal=30,
        )
        collector.save_dataset()

    results = {}

    sweeps = {
        "hidden_dim": [32, 64, 128],
        "n_layers": [1, 2, 3],
        "context": [5, 10, 20],
    }

    if args.param in ("hidden_dim", "all"):
        print("=== Hidden Dimension Sweep ===")
        for hd in sweeps["hidden_dim"]:
            print(f"  hidden_dim={hd}...")
            r = train_and_eval(env, data_path, hidden_dim=hd, n_epochs=15)
            results[f"hidden_{hd}"] = {**r, "hidden_dim": hd}
            print(f"    return={r['mean_return']:.1f}, ev_time={r['mean_ev_time']:.1f}, "
                  f"params={r['n_params']:,}")

    if args.param in ("n_layers", "all"):
        print("=== Layer Count Sweep ===")
        for nl in sweeps["n_layers"]:
            print(f"  n_layers={nl}...")
            r = train_and_eval(env, data_path, n_layers=nl, n_epochs=15)
            results[f"layers_{nl}"] = {**r, "n_layers": nl}
            print(f"    return={r['mean_return']:.1f}, ev_time={r['mean_ev_time']:.1f}")

    if args.param in ("context", "all"):
        print("=== Context Length Sweep ===")
        for cl in sweeps["context"]:
            print(f"  context={cl}...")
            r = train_and_eval(env, data_path, context_length=cl, n_epochs=15)
            results[f"context_{cl}"] = {**r, "context_length": cl}
            print(f"    return={r['mean_return']:.1f}, ev_time={r['mean_ev_time']:.1f}")

    # Save
    out_path = _ROOT / "results" / "hyperparameter_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
