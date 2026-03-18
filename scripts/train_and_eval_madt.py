#!/usr/bin/env python3
"""Train and evaluate MADT on multi-agent EV corridor environment.

Standalone script that:
  1. Loads an MA dataset (HDF5) via MultiAgentTrajectoryDataset
  2. Trains MADT for a configurable number of epochs
  3. Evaluates against baselines (greedy, fixed-time, random) in the MA env
  4. Prints a comparison table
  5. Saves model checkpoint and results JSON

Usage::

    python scripts/train_and_eval_madt.py \\
        --data data/ma_dataset_3x3.h5 \\
        --rows 3 --cols 3 --epochs 30 --save-results results/madt_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.models.madt import MultiAgentDecisionTransformer
from src.models.trajectory_dataset import MultiAgentTrajectoryDataset

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_madt(
    data_path: str,
    n_agents: int,
    rows: int = 3,
    cols: int = 3,
    epochs: int = 30,
    batch_size: int = 32,
    context_length: int = 20,
    hidden_dim: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    gat_heads: int = 2,
    gat_layers: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_steps: int = 100,
    device: str = "cpu",
) -> tuple[MultiAgentDecisionTransformer, torch.Tensor, dict]:
    """Train MADT on the MA dataset.

    Returns (model, adj_matrix, training_info).
    """
    dataset = MultiAgentTrajectoryDataset(
        data_path=data_path,
        n_agents=n_agents,
        context_length=context_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    state_dim = dataset.episodes[0]["states"].shape[-1]
    act_dim = 4  # Discrete(4) per agent

    # Build adjacency: chain along the route + self-loops
    adj = torch.eye(n_agents, dtype=torch.float32)
    for i in range(n_agents - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    model = MultiAgentDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_agents=n_agents,
        adj_matrix=adj,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        gat_heads=gat_heads,
        gat_layers=gat_layers,
        max_length=context_length,
        max_ep_len=max_steps + 10,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MADT: {n_params:,} params, {len(dataset)} segments, {n_agents} agents")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup + cosine schedule
    total_steps = epochs * max(len(dataloader), 1)
    warmup_steps = min(100, total_steps // 5)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 0.01)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses: list[float] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in dataloader:
            states = batch["states"].to(device)          # [B, N, K, D]
            actions = batch["actions"].to(device)        # [B, N, K]
            rtg = batch["returns_to_go"].to(device)      # [B, N, K, 1]
            timesteps = batch["timesteps"].to(device)    # [B, N, K]
            mask = batch["attention_mask"].to(device)     # [B, K]

            logits = model(states, actions, rtg, timesteps)  # [B, N, K, act_dim]

            logits_flat = logits.reshape(-1, act_dim)
            targets_flat = actions.reshape(-1).clamp(0, act_dim - 1)
            # Expand mask: [B, K] -> [B, N, K] -> flat
            mask_expanded = mask.unsqueeze(1).expand(-1, n_agents, -1).reshape(-1).bool()

            if mask_expanded.any():
                loss = loss_fn(logits_flat[mask_expanded], targets_flat[mask_expanded])
            else:
                loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(avg_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

    training_info = {
        "epochs": epochs,
        "final_loss": losses[-1] if losses else float("nan"),
        "losses": losses,
        "n_params": n_params,
        "train_time_s": time.time() - t0,
    }

    return model, adj, training_info


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_madt_in_ma_env(
    model: MultiAgentDecisionTransformer,
    rows: int,
    cols: int,
    max_steps: int,
    n_agents: int,
    n_episodes: int = 20,
    target_return: float = 50.0,
    device: str = "cpu",
) -> dict:
    """Evaluate trained MADT in the MA env."""
    origin = "n0_0"
    destination = f"n{rows - 1}_{cols - 1}"

    env = EVCorridorMAEnv(
        rows=rows, cols=cols, max_steps=max_steps,
        origin=origin, destination=destination,
    )

    model.eval()
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    ev_arrived_count = 0

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        if len(env.agents) != n_agents:
            continue

        agents_snapshot = list(env.agents)
        state_dim = model.state_dim
        K = model.max_length
        N = n_agents

        # Buffers for the rolling context window
        states_buf = torch.zeros(1, N, K, state_dim, device=device)
        actions_buf = torch.zeros(1, N, K, dtype=torch.long, device=device)
        rtg_buf = torch.zeros(1, N, K, 1, device=device)
        ts_buf = torch.zeros(1, N, K, dtype=torch.long, device=device)

        # Initialize first timestep
        for i, a in enumerate(agents_snapshot):
            states_buf[0, i, 0] = torch.tensor(obs_dict[a], dtype=torch.float32, device=device)
            rtg_buf[0, i, 0, 0] = target_return / N

        ep_return = 0.0
        t = 0

        for _ in range(max_steps):
            if not env.agents:
                break

            ctx = min(t + 1, K)
            action_dict = {}

            for i, agent_name in enumerate(agents_snapshot):
                if agent_name not in env.agents:
                    action_dict[agent_name] = 0
                    continue
                action = model.get_action(
                    states_buf[:, :, :ctx],
                    actions_buf[:, :, :ctx],
                    rtg_buf[:, :, :ctx],
                    ts_buf[:, :, :ctx],
                    agent_idx=i,
                )
                action_dict[agent_name] = action

            # Only pass actions for active agents
            active_actions = {a: action_dict[a] for a in env.agents if a in action_dict}
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(active_actions)

            step_reward = sum(rew_dict.values())
            ep_return += step_reward

            t += 1
            if t < K and env.agents:
                for i, agent_name in enumerate(agents_snapshot):
                    if agent_name in obs_dict:
                        states_buf[0, i, t] = torch.tensor(
                            obs_dict[agent_name], dtype=torch.float32, device=device
                        )
                    actions_buf[0, i, t - 1] = action_dict.get(agent_name, 0)
                    agent_rew = rew_dict.get(agent_name, 0.0)
                    rtg_buf[0, i, t, 0] = rtg_buf[0, i, t - 1, 0] - agent_rew
                    ts_buf[0, i, t] = min(t, max_steps)

            if not env.agents:
                # Check if EV arrived (episode ended via termination, not truncation)
                if any(term_dict.values()) and not any(trunc_dict.values()):
                    ev_arrived_count += 1
                break

        episode_returns.append(ep_return)
        episode_lengths.append(t)

    return {
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "ev_arrival_rate": ev_arrived_count / max(len(episode_returns), 1),
        "n_episodes": len(episode_returns),
        "target_return": target_return,
    }


def evaluate_baselines_in_ma_env(
    rows: int, cols: int, max_steps: int,
    n_agents: int, n_episodes: int = 20,
) -> dict[str, dict]:
    """Evaluate baseline policies in the MA env."""
    origin = "n0_0"
    destination = f"n{rows - 1}_{cols - 1}"

    results: dict[str, dict] = {}

    # Policies to evaluate
    policy_configs = {
        "Greedy Preempt (Expert)": "greedy",
        "Fixed-Time Cycling": "fixed",
        "Random": "random",
    }

    for name, ptype in policy_configs.items():
        env = EVCorridorMAEnv(
            rows=rows, cols=cols, max_steps=max_steps,
            origin=origin, destination=destination,
        )
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ev_arrived = 0

        for _ in range(n_episodes):
            obs_dict, _ = env.reset()
            if len(env.agents) != n_agents:
                continue

            agents_snapshot = list(env.agents)
            ep_return = 0.0
            t = 0

            for _ in range(max_steps):
                if not env.agents:
                    break

                if ptype == "greedy":
                    # Expert: green for EV approach, max-pressure otherwise
                    action_dict = _greedy_actions(obs_dict, env, agents_snapshot)
                elif ptype == "fixed":
                    # Fixed-time: cycle phases based on step count
                    phase = (t // 10) % 4
                    action_dict = {a: phase for a in env.agents}
                else:
                    # Random
                    action_dict = {a: env.action_space(a).sample() for a in env.agents}

                obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(action_dict)
                ep_return += sum(rew_dict.values())
                t += 1

                if not env.agents:
                    if any(term_dict.values()) and not any(trunc_dict.values()):
                        ev_arrived += 1
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(t)

        results[name] = {
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "std_return": float(np.std(ep_returns)) if ep_returns else 0.0,
            "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "ev_arrival_rate": ev_arrived / max(len(ep_returns), 1),
            "n_episodes": len(ep_returns),
        }

    return results


def _greedy_actions(
    obs_dict: dict[str, np.ndarray],
    env: EVCorridorMAEnv,
    agents_snapshot: list[str],
) -> dict[str, int]:
    """Greedy preemption for MA env (same logic as MAGreedyPreemptPolicy)."""
    actions: dict[str, int] = {}
    for agent_name in env.agents:
        idx = int(agent_name.split("_")[1])
        node_id = env._route_intersections[idx]

        if not env._ev_arrived and env._ev_link_idx < len(env._route) - 1:
            _, lid = env._route[env._ev_link_idx]
            if lid is not None:
                downstream = env._network["links"][lid]["target"]
                if downstream == node_id:
                    phase_idx = env._network["links"][lid]["phase_index"]
                    actions[agent_name] = phase_idx
                    continue

        # Max-pressure fallback
        node = env._network["nodes"][node_id]
        phase_pressures = np.zeros(4)
        for lid in node["incoming_links"][:4]:
            lk = env._network["links"][lid]
            phase_pressures[lk["phase_index"]] += lk["density"] * lk["length"]
        actions[agent_name] = int(np.argmax(phase_pressures))

    return actions


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate MADT")
    parser.add_argument("--data", type=str, default="data/ma_dataset_3x3.h5")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--target-return", type=float, default=50.0)
    parser.add_argument("--save-results", type=str, default="results/madt_results.json")
    parser.add_argument("--save-model", type=str, default="models/madt_trained.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  MADT Training & Evaluation Pipeline")
    print("=" * 60)

    # Determine n_agents from the env
    origin = "n0_0"
    destination = f"n{args.rows - 1}_{args.cols - 1}"
    tmp_env = EVCorridorMAEnv(
        rows=args.rows, cols=args.cols, max_steps=args.max_steps,
        origin=origin, destination=destination,
    )
    tmp_env.reset()
    n_agents = len(tmp_env.agents)
    print(f"Grid: {args.rows}x{args.cols}, n_agents={n_agents}, "
          f"data={args.data}, epochs={args.epochs}")

    # --- Train ---
    print("\n--- Training MADT ---")
    model, adj, train_info = train_madt(
        data_path=args.data,
        n_agents=n_agents,
        rows=args.rows,
        cols=args.cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        max_steps=args.max_steps,
        device=args.device,
    )

    # Save model
    model_dir = Path(args.save_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_agents": n_agents,
        "state_dim": model.state_dim,
        "act_dim": model.act_dim,
        "adj_matrix": adj,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "context_length": args.context_length,
        "max_steps": args.max_steps,
    }, args.save_model)
    print(f"Model saved to {args.save_model}")

    # --- Evaluate MADT ---
    print("\n--- Evaluating MADT ---")
    madt_results = evaluate_madt_in_ma_env(
        model, args.rows, args.cols, args.max_steps,
        n_agents, args.eval_episodes, args.target_return, args.device,
    )

    # --- Evaluate baselines ---
    print("\n--- Evaluating Baselines ---")
    baseline_results = evaluate_baselines_in_ma_env(
        args.rows, args.cols, args.max_steps, n_agents, args.eval_episodes,
    )

    # --- Comparison table ---
    all_results = {"MADT": madt_results, **baseline_results}

    print("\n" + "=" * 70)
    print(f"  {'Method':<25s} {'Return':>10s} {'Std':>8s} "
          f"{'Length':>8s} {'EV Arr%':>8s}")
    print("  " + "-" * 62)
    for name, m in all_results.items():
        print(f"  {name:<25s} {m['mean_return']:>10.1f} {m.get('std_return', 0):>8.1f} "
              f"{m['mean_length']:>8.1f} {m['ev_arrival_rate'] * 100:>7.1f}%")
    print("=" * 70)

    # --- Save results ---
    results_dir = Path(args.save_results).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "rows": args.rows,
            "cols": args.cols,
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "context_length": args.context_length,
            "lr": args.lr,
            "n_agents": n_agents,
            "eval_episodes": args.eval_episodes,
            "target_return": args.target_return,
        },
        "training": {
            "final_loss": train_info["final_loss"],
            "n_params": train_info["n_params"],
            "train_time_s": train_info["train_time_s"],
            "epochs": train_info["epochs"],
        },
        "results": {},
    }
    for name, m in all_results.items():
        output["results"][name] = {
            k: v for k, v in m.items()
            if isinstance(v, (int, float, str, bool))
        }

    with open(args.save_results, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
