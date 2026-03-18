#!/usr/bin/env python3
"""Quick ablation study for the EV-DT project.

Compares four DT variants on a 3x3 grid environment:
    1. full_dt        -- Full DT (context=15, 2 layers)
    2. no_rtg         -- DT without return conditioning (R_t always 0)
    3. short_context  -- DT with short context (context=3)
    4. expert_only    -- DT with expert-only data (no random/suboptimal)

Collects 80 training episodes, trains for 10 epochs, evaluates 20 episodes each.
Results saved to results/ablation_results.json.

Usage::

    python scripts/run_quick_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.envs.ev_corridor_env import EVCorridorEnv
from src.models.decision_transformer import DecisionTransformer


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

GRID_ROWS = 3
GRID_COLS = 3
MAX_STEPS = 50
NUM_TRAIN_EPISODES = 80
NUM_TRAIN_EPOCHS = 10
NUM_EVAL_EPISODES = 20
DEVICE = "cpu"
SEED = 42


# -----------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------

def collect_episodes(
    env: EVCorridorEnv,
    n_episodes: int,
    policy: str = "mixed",
    expert_ratio: float = 0.7,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Collect episodes using greedy/random policies.

    Args:
        env: The environment.
        n_episodes: Number of episodes to collect.
        policy: "mixed" (expert + random), "expert" (greedy only), or "random".
        expert_ratio: Fraction of expert episodes when policy="mixed".
        seed: Random seed.

    Returns:
        List of episode dicts with keys: states, actions, rewards, returns_to_go.
    """
    rng = np.random.default_rng(seed)
    episodes = []

    for i in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 100000)))
        states_list = [obs.copy()]
        actions_list = []
        rewards_list = []
        done = False

        # Decide policy for this episode
        if policy == "expert":
            use_expert = True
        elif policy == "random":
            use_expert = False
        else:  # mixed
            use_expert = (i / n_episodes) < expert_ratio

        while not done:
            if use_expert:
                # Greedy heuristic: pick action 0 (keep green for EV direction)
                # This is a simplified expert that always preempts for EV
                action = 0
            else:
                action = env.action_space.sample()
                if hasattr(action, '__len__'):
                    action = int(action[0])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states_list.append(obs.copy())
            actions_list.append(action if isinstance(action, int) else int(action))
            rewards_list.append(float(reward))

        T = len(rewards_list)
        if T == 0:
            continue

        states_arr = np.array(states_list[:T], dtype=np.float32)
        actions_arr = np.array(actions_list, dtype=np.int64)
        rewards_arr = np.array(rewards_list, dtype=np.float32)

        # Compute returns-to-go (undiscounted)
        rtg = np.zeros(T, dtype=np.float32)
        rtg[-1] = rewards_arr[-1]
        for t in range(T - 2, -1, -1):
            rtg[t] = rewards_arr[t] + rtg[t + 1]

        episodes.append({
            "states": states_arr,
            "actions": actions_arr,
            "rewards": rewards_arr,
            "returns_to_go": rtg,
            "episode_return": float(rewards_arr.sum()),
            "policy": "expert" if use_expert else "random",
        })

    return episodes


def episodes_to_dataset(
    episodes: List[Dict[str, Any]],
    context_length: int,
    zero_rtg: bool = False,
) -> TensorDataset:
    """Convert episode list to a PyTorch TensorDataset of context windows.

    Args:
        episodes: List of episode dicts.
        context_length: Window length for each training sample.
        zero_rtg: If True, set all returns-to-go to 0 (ablation).

    Returns:
        TensorDataset of (states, actions, returns_to_go, timesteps, masks).
    """
    all_states = []
    all_actions = []
    all_rtg = []
    all_timesteps = []
    all_masks = []

    state_dim = episodes[0]["states"].shape[1]

    for ep in episodes:
        T = len(ep["actions"])
        for start in range(0, max(1, T - context_length + 1), max(1, context_length // 2)):
            end = min(start + context_length, T)
            seg_len = end - start

            # Pad to context_length
            states = np.zeros((context_length, state_dim), dtype=np.float32)
            actions = np.zeros(context_length, dtype=np.int64)
            rtg = np.zeros((context_length, 1), dtype=np.float32)
            timesteps = np.zeros(context_length, dtype=np.int64)
            mask = np.zeros(context_length, dtype=np.float32)

            states[:seg_len] = ep["states"][start:end]
            actions[:seg_len] = ep["actions"][start:end]
            if zero_rtg:
                rtg[:seg_len, 0] = 0.0
            else:
                rtg[:seg_len, 0] = ep["returns_to_go"][start:end]
            timesteps[:seg_len] = np.arange(start, end)
            mask[:seg_len] = 1.0

            all_states.append(states)
            all_actions.append(actions)
            all_rtg.append(rtg)
            all_timesteps.append(timesteps)
            all_masks.append(mask)

    return TensorDataset(
        torch.tensor(np.array(all_states)),
        torch.tensor(np.array(all_actions)),
        torch.tensor(np.array(all_rtg)),
        torch.tensor(np.array(all_timesteps)),
        torch.tensor(np.array(all_masks)),
    )


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_dt(
    model: DecisionTransformer,
    dataset: TensorDataset,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Train a Decision Transformer on the given dataset.

    Returns:
        List of per-epoch average losses.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        for states, actions, rtg, timesteps, masks in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            rtg = rtg.to(device)
            timesteps = timesteps.to(device)
            masks = masks.to(device)

            logits = model(states, actions, rtg, timesteps)

            # Masked loss
            B, T, A = logits.shape
            logits_flat = logits.reshape(B * T, A)
            actions_flat = actions.reshape(B * T)
            mask_flat = masks.reshape(B * T).bool()

            if mask_flat.sum() == 0:
                continue

            loss = loss_fn(logits_flat[mask_flat], actions_flat[mask_flat])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses) if batch_losses else float("nan")
        epoch_losses.append(avg_loss)

    return epoch_losses


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate_dt(
    model: DecisionTransformer,
    env: EVCorridorEnv,
    n_episodes: int = 20,
    target_return: float = 0.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate a trained DT model.

    Returns:
        Dict with mean_return, std_return, mean_length, mean_ev_travel_time, etc.
    """
    model.eval()
    model.to(device)

    returns = []
    lengths = []
    ev_travel_times = []

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep_idx)
        done = False
        t = 0
        ep_return = 0.0

        # Build running context tensors
        max_len = model.max_length
        state_dim = model.state_dim
        states = torch.zeros(1, max_len, state_dim, device=device)
        actions = torch.zeros(1, max_len, dtype=torch.long, device=device)
        rtg = torch.zeros(1, max_len, 1, device=device)
        timesteps = torch.zeros(1, max_len, dtype=torch.long, device=device)

        states[0, 0] = torch.tensor(obs, dtype=torch.float32, device=device)
        rtg[0, 0, 0] = target_return
        timesteps[0, 0] = 0

        while not done:
            ctx_len = min(t + 1, max_len)
            action = model.get_action(
                states[:, :ctx_len],
                actions[:, :ctx_len],
                rtg[:, :ctx_len],
                timesteps[:, :ctx_len],
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            t += 1

            if t < max_len:
                states[0, t] = torch.tensor(obs, dtype=torch.float32, device=device)
                actions[0, t - 1] = action
                rtg[0, t, 0] = rtg[0, t - 1, 0] - reward
                timesteps[0, t] = t

        returns.append(ep_return)
        lengths.append(t)

        ev_time = info.get("ev_travel_time", -1.0)
        ev_travel_times.append(float(ev_time))

    valid_ev_times = [t for t in ev_travel_times if t >= 0]

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_ev_travel_time": float(np.mean(valid_ev_times)) if valid_ev_times else -1.0,
        "std_ev_travel_time": float(np.std(valid_ev_times)) if valid_ev_times else -1.0,
        "num_episodes": n_episodes,
    }


# -----------------------------------------------------------------------
# Ablation definitions
# -----------------------------------------------------------------------

def run_ablation() -> Dict[str, Any]:
    """Run the full ablation study and return results dict."""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create environment
    env = EVCorridorEnv(rows=GRID_ROWS, cols=GRID_COLS, max_steps=MAX_STEPS, seed=SEED)

    # Determine dimensions
    obs, _ = env.reset(seed=SEED)
    state_dim = len(obs)
    # DT uses a flat discrete action: we pick from the first sub-action's range
    act_dim = 4  # 4 signal phases

    print("=" * 65)
    print("  EV-DT Quick Ablation Study")
    print("=" * 65)
    print(f"  Grid size     : {GRID_ROWS}x{GRID_COLS}")
    print(f"  State dim     : {state_dim}")
    print(f"  Action dim    : {act_dim}")
    print(f"  Train episodes: {NUM_TRAIN_EPISODES}")
    print(f"  Train epochs  : {NUM_TRAIN_EPOCHS}")
    print(f"  Eval episodes : {NUM_EVAL_EPISODES}")
    print("=" * 65)

    # ---- Collect data ----
    print("\n[1/5] Collecting training data (mixed: expert + random) ...")
    mixed_episodes = collect_episodes(
        env, NUM_TRAIN_EPISODES, policy="mixed", seed=SEED
    )
    print(f"  Collected {len(mixed_episodes)} mixed episodes")

    print("[2/5] Collecting expert-only data ...")
    expert_episodes = collect_episodes(
        env, NUM_TRAIN_EPISODES, policy="expert", seed=SEED + 1
    )
    print(f"  Collected {len(expert_episodes)} expert episodes")

    # ---- Define ablation configs ----
    ablations = {
        "full_dt": {
            "description": "Full DT (context=15, 2 layers)",
            "context_length": 15,
            "n_layers": 2,
            "episodes": mixed_episodes,
            "zero_rtg": False,
        },
        "no_rtg": {
            "description": "DT without return conditioning (R_t=0)",
            "context_length": 15,
            "n_layers": 2,
            "episodes": mixed_episodes,
            "zero_rtg": True,
        },
        "short_context": {
            "description": "DT with short context (context=3)",
            "context_length": 3,
            "n_layers": 2,
            "episodes": mixed_episodes,
            "zero_rtg": False,
        },
        "expert_only": {
            "description": "DT with expert-only data (no random/suboptimal)",
            "context_length": 15,
            "n_layers": 2,
            "episodes": expert_episodes,
            "zero_rtg": False,
        },
    }

    results: Dict[str, Any] = {
        "metadata": {
            "grid_size": f"{GRID_ROWS}x{GRID_COLS}",
            "state_dim": state_dim,
            "act_dim": act_dim,
            "num_train_episodes": NUM_TRAIN_EPISODES,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "num_eval_episodes": NUM_EVAL_EPISODES,
            "seed": SEED,
        },
        "ablations": {},
    }

    # ---- Run each ablation ----
    print(f"\n[3/5] Running {len(ablations)} ablation experiments ...\n")

    for name, cfg in ablations.items():
        print(f"  --- {name}: {cfg['description']} ---")
        t0 = time.time()

        # Build dataset
        dataset = episodes_to_dataset(
            cfg["episodes"],
            context_length=cfg["context_length"],
            zero_rtg=cfg["zero_rtg"],
        )
        print(f"    Dataset size: {len(dataset)} segments")

        # Build model
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=64,
            n_layers=cfg["n_layers"],
            n_heads=2,
            max_length=cfg["context_length"],
            max_ep_len=MAX_STEPS + 10,
            dropout=0.1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Model params: {n_params:,}")

        # Train
        epoch_losses = train_dt(
            model, dataset,
            num_epochs=NUM_TRAIN_EPOCHS,
            batch_size=32,
            lr=1e-3,
            device=DEVICE,
        )
        train_time = time.time() - t0
        print(f"    Final loss: {epoch_losses[-1]:.4f} (trained in {train_time:.1f}s)")

        # Evaluate
        eval_env = EVCorridorEnv(
            rows=GRID_ROWS, cols=GRID_COLS, max_steps=MAX_STEPS, seed=SEED + 100
        )
        eval_results = evaluate_dt(
            model, eval_env,
            n_episodes=NUM_EVAL_EPISODES,
            target_return=0.0,
            device=DEVICE,
        )
        print(f"    Eval: mean_return={eval_results['mean_return']:.2f}, "
              f"mean_length={eval_results['mean_length']:.1f}, "
              f"mean_ev_time={eval_results['mean_ev_travel_time']:.2f}")

        results["ablations"][name] = {
            "description": cfg["description"],
            "config": {
                "context_length": cfg["context_length"],
                "n_layers": cfg["n_layers"],
                "zero_rtg": cfg["zero_rtg"],
                "data_policy": "expert" if name == "expert_only" else "mixed",
            },
            "training": {
                "final_loss": float(epoch_losses[-1]),
                "epoch_losses": [float(l) for l in epoch_losses],
                "training_time_sec": round(train_time, 2),
                "num_params": n_params,
                "dataset_size": len(dataset),
            },
            "evaluation": eval_results,
        }
        print()

    # ---- Summary ----
    print("\n[4/5] Ablation Summary")
    print("-" * 75)
    print(f"  {'Variant':<20s} {'Loss':>8s} {'Return':>10s} {'Length':>8s} {'EV Time':>10s}")
    print("-" * 75)
    for name, res in results["ablations"].items():
        ev = res["evaluation"]
        tr = res["training"]
        print(f"  {name:<20s} {tr['final_loss']:>8.4f} "
              f"{ev['mean_return']:>10.2f} {ev['mean_length']:>8.1f} "
              f"{ev['mean_ev_travel_time']:>10.2f}")
    print("-" * 75)

    # ---- Save results ----
    print("\n[5/5] Saving results ...")
    output_path = _PROJECT_ROOT / "results" / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


if __name__ == "__main__":
    run_ablation()
