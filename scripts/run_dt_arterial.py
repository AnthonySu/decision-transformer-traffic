#!/usr/bin/env python3
"""Decision Transformer experiment on arterial network topology.

Demonstrates that DT works on non-grid topologies by training and evaluating
on a linear arterial corridor with 6 intersections and cross streets.

Steps:
  1. Generate 100 episodes (70 expert, 15 random, 15 suboptimal)
  2. Train DT with hidden_dim=64, 2 layers, 4 heads, context=10, 20 epochs
  3. Evaluate DT and FT-EVP for 30 episodes
  4. Save all results to results/dt_arterial.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.fixed_time_evp import FixedTimeEVP  # noqa: E402
from src.envs.ev_corridor_env import EVCorridorEnv  # noqa: E402
from src.envs.network_utils import build_arterial_network  # noqa: E402
from src.models.decision_transformer import DecisionTransformer  # noqa: E402
from src.models.trajectory_dataset import TrajectoryDataset  # noqa: E402
from src.utils.data_collector import DataCollector  # noqa: E402
from src.utils.metrics import aggregate_metrics  # noqa: E402

# ======================================================================
# Configuration
# ======================================================================

NUM_INTERSECTIONS = 6

DATA_PATH = str(_PROJECT_ROOT / "data" / "dt_arterial.h5")
MODEL_DIR = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"

# The arterial has 3 rows (north stubs, main arterial, south stubs) and
# num_intersections columns.  The EV corridor runs west-to-east along row 1.
ENV_KWARGS: dict = dict(
    rows=3,
    cols=NUM_INTERSECTIONS,
    max_steps=150,
    seed=42,
    origin="n1_0",
    destination=f"n1_{NUM_INTERSECTIONS - 1}",
)

# Dataset
NUM_EXPERT = 70
NUM_RANDOM = 15
NUM_SUBOPTIMAL = 15

# Model
HIDDEN_DIM = 64
N_LAYERS = 2
N_HEADS = 4
CONTEXT_LENGTH = 10
MAX_EP_LEN = 200
DROPOUT = 0.1

# Training
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 200

# Evaluation
EVAL_EPISODES = 30


# ======================================================================
# Helper: create an env backed by the arterial network
# ======================================================================

def make_arterial_env(**extra_kwargs) -> EVCorridorEnv:
    """Build an EVCorridorEnv with the arterial network injected."""
    kw = {**ENV_KWARGS, **extra_kwargs}
    env = EVCorridorEnv(**kw)
    # Replace the default grid network with the arterial topology
    env._network = build_arterial_network(num_intersections=NUM_INTERSECTIONS)
    return env


# ======================================================================
# Greedy policy adapted for the arterial env
# ======================================================================

class _ArterialGreedyPolicy:
    """Greedy preemption policy that reads env state directly.

    For each route intersection, set the phase to match the EV's
    approaching link.  For intersections the EV has already passed,
    use a max-pressure heuristic.
    """

    def __init__(self, env: EVCorridorEnv):
        self.env = env

    def select_action(self, obs: np.ndarray, ev_info: dict) -> np.ndarray:
        env = self.env
        network = env._network
        route = env._route
        route_intersections = env._route_intersections
        ev_link_idx = env._ev_link_idx

        actions = np.zeros(env._max_route_len, dtype=np.int64)

        for i, node_id in enumerate(route_intersections):
            if i >= env._max_route_len:
                break

            best_phase = 0
            for _ri, (_, link_id) in enumerate(route):
                if link_id is None:
                    continue
                link = network["links"][link_id]
                if link["target"] == node_id:
                    best_phase = link["phase_index"]
                    break

            dist_to_ev = i - ev_link_idx
            if dist_to_ev >= -1:
                actions[i] = best_phase
            else:
                node = network["nodes"][node_id]
                max_density = -1.0
                for p_idx in range(node["num_phases"]):
                    phase_density = 0.0
                    for lid in node["incoming_links"]:
                        lk = network["links"][lid]
                        if lk["phase_index"] == p_idx:
                            phase_density += lk["density"]
                    if phase_density > max_density:
                        max_density = phase_density
                        actions[i] = p_idx

        return actions

    def reset(self) -> None:
        pass


# ======================================================================
# Step 1: Generate dataset
# ======================================================================

def generate_dataset() -> str:
    """Generate 100 mixed-quality episodes on the arterial network."""
    print("\n" + "=" * 60)
    print("  STEP 1: Generate Arterial Dataset")
    print("=" * 60)

    env = make_arterial_env()
    expert = _ArterialGreedyPolicy(env)

    collector = DataCollector(env=env, save_path=DATA_PATH)
    t0 = time.time()

    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=NUM_EXPERT,
        num_random=NUM_RANDOM,
        num_suboptimal=NUM_SUBOPTIMAL,
    )

    elapsed = time.time() - t0
    n_episodes = len(collector._episodes)
    returns = [ep["episode_return"] for ep in collector._episodes]
    lengths = [ep["episode_length"] for ep in collector._episodes]

    print(f"  Collected {n_episodes} episodes in {elapsed:.1f}s")
    print(f"  Return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"  Length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")

    policy_names = {ep["policy_name"] for ep in collector._episodes}
    for pname in sorted(policy_names):
        eps = [ep for ep in collector._episodes if ep["policy_name"] == pname]
        rets = [ep["episode_return"] for ep in eps]
        print(f"    {pname:12s}: n={len(eps)}, return={np.mean(rets):.2f}")

    collector.save_dataset()
    return DATA_PATH


# ======================================================================
# Step 2: Train DT
# ======================================================================

def train_dt(data_path: str, device: str) -> tuple:
    """Train Decision Transformer on arterial data."""
    print("\n" + "=" * 60)
    print("  STEP 2: Train Decision Transformer")
    print("=" * 60)

    dataset = TrajectoryDataset(
        data_path=data_path,
        context_length=CONTEXT_LENGTH,
        normalize_states=True,
        normalize_returns=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    state_dim = dataset.state_dim
    act_dim = dataset.act_dim

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_length=CONTEXT_LENGTH,
        max_ep_len=MAX_EP_LEN,
        dropout=DROPOUT,
        activation="gelu",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    total_steps = NUM_EPOCHS * len(dataloader)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Dataset: {len(dataset)} segments")
    print(f"  state_dim={state_dim}, act_dim={act_dim}")
    print(f"  Model params: {model.get_num_params():,}")
    print(f"  Training for {NUM_EPOCHS} epochs ({total_steps} steps)")
    print("-" * 60)

    t0 = time.time()
    training_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            masks = batch["masks"].to(device)

            action_logits = model(states, actions, returns_to_go, timesteps)
            logits_flat = action_logits.reshape(-1, act_dim)
            targets_flat = actions.reshape(-1)
            mask_flat = masks.reshape(-1).bool()

            if mask_flat.any():
                loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])
            else:
                loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        training_losses.append(float(avg_loss))

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}"
            )

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Final loss: {training_losses[-1]:.4f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    save_path = MODEL_DIR / "dt_arterial.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "embed_dim": HIDDEN_DIM,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "context_length": CONTEXT_LENGTH,
                "max_ep_len": MAX_EP_LEN,
                "dropout": DROPOUT,
                "activation": "gelu",
            },
            "state_dim": state_dim,
            "act_dim": act_dim,
            "state_mean": dataset.state_mean.tolist(),
            "state_std": dataset.state_std.tolist(),
            "return_scale": dataset.return_scale,
            "training_losses": training_losses,
        },
        save_path,
    )
    print(f"  Model saved to {save_path}")

    return model, dataset


# ======================================================================
# Evaluation helpers
# ======================================================================

def _normalize_obs(
    obs: np.ndarray,
    state_mean: np.ndarray | None,
    state_std: np.ndarray | None,
) -> np.ndarray:
    if state_mean is not None and state_std is not None:
        return ((obs - state_mean) / state_std).astype(np.float32)
    return obs.astype(np.float32)


def evaluate_dt_episodes(
    model: DecisionTransformer,
    target_return: float,
    n_episodes: int,
    device: str,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    return_scale: float = 1.0,
) -> list[dict]:
    """Run DT for n_episodes with given target return."""
    model.eval()
    episodes_info = []

    for ep_idx in range(n_episodes):
        # Create a fresh env each episode so the arterial network is injected
        # before the first reset, and each episode gets a different seed.
        ep_env = make_arterial_env(seed=42 + ep_idx)
        obs, info = ep_env.reset()

        done = False
        episode_return = 0.0
        t = 0

        states_buf = torch.zeros(
            1, model.max_length, model.state_dim, device=device
        )
        actions_buf = torch.zeros(
            1, model.max_length, dtype=torch.long, device=device
        )
        rtg_buf = torch.zeros(1, model.max_length, 1, device=device)
        timesteps_buf = torch.zeros(
            1, model.max_length, dtype=torch.long, device=device
        )

        obs_norm = _normalize_obs(obs, state_mean, state_std)
        states_buf[0, 0] = torch.tensor(obs_norm, dtype=torch.float32, device=device)
        rtg_buf[0, 0, 0] = float(target_return) / float(return_scale)
        timesteps_buf[0, 0] = 0

        step_infos = []

        while not done:
            ctx_len = min(t + 1, model.max_length)
            action = model.get_action(
                states_buf[:, :ctx_len],
                actions_buf[:, :ctx_len],
                rtg_buf[:, :ctx_len],
                timesteps_buf[:, :ctx_len],
            )

            obs, reward, terminated, truncated, info = ep_env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

            if t < model.max_length:
                obs_norm = _normalize_obs(obs, state_mean, state_std)
                states_buf[0, t] = torch.tensor(
                    obs_norm, dtype=torch.float32, device=device
                )
                actions_buf[0, t - 1] = action
                rtg_buf[0, t, 0] = (
                    rtg_buf[0, t - 1, 0] - float(reward) / float(return_scale)
                )
                timesteps_buf[0, t] = min(t, MAX_EP_LEN - 1)

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })

    return episodes_info


def _run_ft_evp(n_episodes: int) -> list[dict]:
    """Run Fixed-Time EVP baseline on the arterial network."""
    ft_evp = FixedTimeEVP()
    episodes_info = []

    for _ in range(n_episodes):
        env = make_arterial_env()
        obs, info = env.reset()
        ft_evp.reset()
        done = False
        episode_return = 0.0
        t = 0
        step_infos = []

        while not done:
            action = ft_evp.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return, "length": t, "step_infos": step_infos,
        })
    return episodes_info


def _run_random(n_episodes: int) -> list[dict]:
    """Run uniform random baseline on the arterial network."""
    episodes_info = []

    for _ in range(n_episodes):
        env = make_arterial_env()
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        t = 0
        step_infos = []

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return, "length": t, "step_infos": step_infos,
        })
    return episodes_info


def _run_greedy(n_episodes: int) -> list[dict]:
    """Run greedy preemption baseline on the arterial network."""
    episodes_info = []

    for _ in range(n_episodes):
        env = make_arterial_env()
        greedy = _ArterialGreedyPolicy(env)
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        t = 0
        step_infos = []

        while not done:
            action = greedy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return, "length": t, "step_infos": step_infos,
        })
    return episodes_info


# ======================================================================
# Step 3: Evaluate
# ======================================================================

def evaluate(
    model: DecisionTransformer,
    dataset: TrajectoryDataset,
    device: str,
) -> dict:
    """Evaluate DT and FT-EVP on the arterial network."""
    print("\n" + "=" * 60)
    print("  STEP 3: Evaluate (30 episodes each)")
    print("=" * 60)

    state_mean, state_std = dataset.get_state_stats()
    return_scale = dataset.get_return_scale()

    results: dict = {}

    # --- DT (target=0, best return) ---
    print("  Evaluating DT (target=0)...")
    dt_infos = evaluate_dt_episodes(
        model, target_return=0.0,
        n_episodes=EVAL_EPISODES, device=device,
        state_mean=state_mean, state_std=state_std,
        return_scale=return_scale,
    )
    results["DT (target=0)"] = aggregate_metrics(dt_infos)

    # --- FT-EVP ---
    print("  Evaluating FT-EVP...")
    ft_infos = _run_ft_evp(EVAL_EPISODES)
    results["FT-EVP"] = aggregate_metrics(ft_infos)

    # --- Greedy ---
    print("  Evaluating Greedy Preempt...")
    greedy_infos = _run_greedy(EVAL_EPISODES)
    results["Greedy"] = aggregate_metrics(greedy_infos)

    # --- Random ---
    print("  Evaluating Random...")
    random_infos = _run_random(EVAL_EPISODES)
    results["Random"] = aggregate_metrics(random_infos)

    # Print summary
    print("\n  " + "-" * 56)
    print(
        f"  {'Method':<20s} {'Return':>10s} {'EV Time':>10s} "
        f"{'BG Delay':>10s} {'Length':>8s}"
    )
    print("  " + "-" * 56)
    for method, m in results.items():
        ret = m.get("mean_return", 0)
        evt = m.get("mean_ev_travel_time", -1)
        bgd = m.get("background_delay_mean", 0)
        length = m.get("mean_length", 0)
        print(
            f"  {method:<20s} {ret:>10.2f} {evt:>10.1f} "
            f"{bgd:>10.2f} {length:>8.1f}"
        )
    print("  " + "-" * 56)

    return results


# ======================================================================
# Step 4: Save results
# ======================================================================

def save_results(
    eval_results: dict,
    training_losses: list,
) -> None:
    """Save all results to results/dt_arterial.json."""
    print("\n" + "=" * 60)
    print("  STEP 4: Save Results")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "dt_arterial.json"

    def _serialize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    results = {
        "experiment": "dt_arterial",
        "description": (
            "Decision Transformer on arterial (non-grid) topology with "
            f"{NUM_INTERSECTIONS} intersections, demonstrating DT generalises "
            "beyond grid networks."
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "topology": "arterial",
            "num_intersections": NUM_INTERSECTIONS,
            "env": ENV_KWARGS,
            "dataset": {
                "num_expert": NUM_EXPERT,
                "num_random": NUM_RANDOM,
                "num_suboptimal": NUM_SUBOPTIMAL,
                "total": NUM_EXPERT + NUM_RANDOM + NUM_SUBOPTIMAL,
            },
            "model": {
                "hidden_dim": HIDDEN_DIM,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "context_length": CONTEXT_LENGTH,
                "max_ep_len": MAX_EP_LEN,
                "dropout": DROPOUT,
            },
            "training": {
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
            },
            "eval": {
                "eval_episodes": EVAL_EPISODES,
            },
        },
        "training_losses": training_losses,
        "evaluation": _serialize(eval_results),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"  Results saved to {output_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Topology: Arterial with {NUM_INTERSECTIONS} intersections")

    t_total = time.time()

    # Step 1: Generate dataset
    data_path = generate_dataset()

    # Step 2: Train DT
    model, dataset = train_dt(data_path, device)

    # Step 3: Evaluate
    eval_results = evaluate(model, dataset, device)

    # Step 4: Save results
    ckpt = torch.load(
        MODEL_DIR / "dt_arterial.pt", map_location=device, weights_only=False
    )
    training_losses = ckpt.get("training_losses", [])
    save_results(eval_results, training_losses)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Arterial experiment complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
