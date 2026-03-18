#!/usr/bin/env python3
"""Train improved MADT addressing credit assignment via centralized expert + shared rewards.

End-to-end pipeline:
  1. Generate improved MA dataset with centralized expert policy and higher shared_reward_frac
  2. Train MADT on the improved data
  3. Evaluate against baselines
  4. Save results to results/madt_improved.json

Key improvements over the original MADT pipeline:
  - Centralized expert: pre-clears the entire corridor ahead of the EV, giving
    consistent high-quality trajectories across ALL agents simultaneously.
  - Higher shared_reward_frac (0.5): every agent gets a direct signal when the
    EV advances, solving the credit assignment problem where off-route agents
    had no learning signal.
  - EV progress delta reward: step-wise shared bonus when the EV moves forward,
    giving all agents immediate feedback for coordinated corridor clearing.
  - Episode mix tuned: 70% expert, 15% noisy, 15% random (more expert data to
    give clearer behavioral signal).

Usage::

    python scripts/train_improved_madt.py
    python scripts/train_improved_madt.py --epochs 50 --shared-reward-frac 0.5
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

import numpy as np  # noqa: I001
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.generate_ma_dataset import generate_dataset
from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.models.madt import MultiAgentDecisionTransformer
from src.models.trajectory_dataset import MultiAgentTrajectoryDataset


# ------------------------------------------------------------------
# Training (same as train_and_eval_madt but factored for reuse)
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
    """Train MADT on the MA dataset. Returns (model, adj_matrix, training_info)."""
    dataset = MultiAgentTrajectoryDataset(
        data_path=data_path,
        n_agents=n_agents,
        context_length=context_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    state_dim = dataset.episodes[0]["states"].shape[-1]
    act_dim = 4

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
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(states, actions, rtg, timesteps)

            logits_flat = logits.reshape(-1, act_dim)
            targets_flat = actions.reshape(-1).clamp(0, act_dim - 1)
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
    shared_reward_frac: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Evaluate trained MADT in the MA env."""
    origin = "n0_0"
    destination = f"n{rows - 1}_{cols - 1}"

    env = EVCorridorMAEnv(
        rows=rows, cols=cols, max_steps=max_steps,
        origin=origin, destination=destination,
        shared_reward_frac=shared_reward_frac,
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

        states_buf = torch.zeros(1, N, K, state_dim, device=device)
        actions_buf = torch.zeros(1, N, K, dtype=torch.long, device=device)
        rtg_buf = torch.zeros(1, N, K, 1, device=device)
        ts_buf = torch.zeros(1, N, K, dtype=torch.long, device=device)

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
    shared_reward_frac: float = 0.5,
) -> dict[str, dict]:
    """Evaluate baseline policies in the MA env."""
    from scripts.generate_ma_dataset import MACentralizedExpertPolicy, MAGreedyPreemptPolicy

    origin = "n0_0"
    destination = f"n{rows - 1}_{cols - 1}"

    results: dict[str, dict] = {}

    policy_configs = {
        "Greedy Preempt": "greedy",
        "Centralized Expert": "centralized",
        "Fixed-Time Cycling": "fixed",
        "Random": "random",
    }

    for name, ptype in policy_configs.items():
        env = EVCorridorMAEnv(
            rows=rows, cols=cols, max_steps=max_steps,
            origin=origin, destination=destination,
            shared_reward_frac=shared_reward_frac,
        )

        if ptype == "greedy":
            policy = MAGreedyPreemptPolicy()
        elif ptype == "centralized":
            policy = MACentralizedExpertPolicy()
        else:
            policy = None

        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ev_arrived = 0

        for _ in range(n_episodes):
            obs_dict, _ = env.reset()
            if len(env.agents) != n_agents:
                continue

            ep_return = 0.0
            t = 0

            for step in range(max_steps):
                if not env.agents:
                    break

                if ptype == "fixed":
                    phase = (step // 10) % 4
                    action_dict = {a: phase for a in env.agents}
                elif ptype == "random":
                    action_dict = {a: env.action_space(a).sample() for a in env.agents}
                else:
                    action_dict = policy.select_actions(obs_dict, env)

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


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved MADT (credit assignment fix)")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--target-return", type=float, default=50.0)
    parser.add_argument("--shared-reward-frac", type=float, default=0.5,
                        help="Shared reward fraction (higher = better credit assignment)")
    parser.add_argument("--save-results", type=str, default="results/madt_improved.json")
    parser.add_argument("--save-model", type=str, default="models/madt_improved.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  Improved MADT Pipeline (Credit Assignment Fix)")
    print("=" * 70)

    # --- Step 1: Generate improved dataset ---
    data_path = f"data/ma_dataset_improved_{args.rows}x{args.cols}.h5"
    n_expert = max(1, int(args.episodes * 0.70))  # 70% expert (up from 50%)
    n_random = max(1, int(args.episodes * 0.15))
    n_noisy = args.episodes - n_expert - n_random

    print("\n--- Step 1: Generate improved dataset ---")
    print(f"  Centralized expert: {n_expert} episodes")
    print(f"  Random: {n_random} episodes")
    print(f"  Noisy expert: {n_noisy} episodes")
    print(f"  shared_reward_frac: {args.shared_reward_frac}")

    generate_dataset(
        rows=args.rows,
        cols=args.cols,
        max_steps=args.max_steps,
        num_expert=n_expert,
        num_random=n_random,
        num_noisy=n_noisy,
        save_path=data_path,
        seed=args.seed,
        use_centralized_expert=True,
        shared_reward_frac=args.shared_reward_frac,
    )

    # --- Step 2: Determine n_agents ---
    origin = "n0_0"
    destination = f"n{args.rows - 1}_{args.cols - 1}"
    tmp_env = EVCorridorMAEnv(
        rows=args.rows, cols=args.cols, max_steps=args.max_steps,
        origin=origin, destination=destination,
    )
    tmp_env.reset()
    n_agents = len(tmp_env.agents)
    print(f"\nGrid: {args.rows}x{args.cols}, n_agents={n_agents}")

    # --- Step 3: Train MADT ---
    print("\n--- Step 2: Training improved MADT ---")
    model, adj, train_info = train_madt(
        data_path=data_path,
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

    # --- Step 4: Evaluate ---
    print("\n--- Step 3: Evaluating improved MADT ---")
    madt_results = evaluate_madt_in_ma_env(
        model, args.rows, args.cols, args.max_steps,
        n_agents, args.eval_episodes, args.target_return,
        shared_reward_frac=args.shared_reward_frac,
        device=args.device,
    )

    print("\n--- Evaluating Baselines ---")
    baseline_results = evaluate_baselines_in_ma_env(
        args.rows, args.cols, args.max_steps, n_agents,
        args.eval_episodes, shared_reward_frac=args.shared_reward_frac,
    )

    # --- Comparison table ---
    all_results = {"MADT (improved)": madt_results, **baseline_results}

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
            "shared_reward_frac": args.shared_reward_frac,
            "use_centralized_expert": True,
            "episode_mix": {
                "expert": n_expert,
                "random": n_random,
                "noisy": n_noisy,
            },
        },
        "improvements": [
            "Centralized expert policy that pre-clears corridor ahead of EV",
            f"Higher shared_reward_frac ({args.shared_reward_frac} vs 0.3 original)",
            "EV progress delta reward for step-wise credit assignment",
            "70% expert episodes (up from 50%) for stronger behavioral signal",
        ],
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
