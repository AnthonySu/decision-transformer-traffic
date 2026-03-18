#!/usr/bin/env python3
"""3-seed Decision Transformer experiment on 4x4 grid for statistical significance.

For each seed in [42, 123, 456]:
  1. Generate 200 episodes (140 expert, 30 random, 30 suboptimal)
     with fixed OD: origin='n0_0', destination='n3_3'
  2. Train DT (hidden=128, layers=3, heads=4, context=20) for 30 epochs
  3. Evaluate DT, FT-EVP, Greedy for 50 episodes each

Aggregates across 3 seeds to get mean +/- std for each method.
Saves results to results/dt_4x4_3seed.json.
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
from src.models.decision_transformer import DecisionTransformer  # noqa: E402
from src.models.trajectory_dataset import TrajectoryDataset  # noqa: E402
from src.utils.data_collector import DataCollector  # noqa: E402
from src.utils.metrics import aggregate_metrics  # noqa: E402

# ======================================================================
# Configuration
# ======================================================================

SEEDS = [42, 123, 456]

DATA_DIR = _PROJECT_ROOT / "data"
MODEL_DIR = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"

ENV_KWARGS_BASE = dict(
    rows=4, cols=4, max_steps=150,
    origin="n0_0", destination="n3_3",
)

# Dataset
NUM_EXPERT = 140
NUM_RANDOM = 30
NUM_SUBOPTIMAL = 30

# Model
HIDDEN_DIM = 128
N_LAYERS = 3
N_HEADS = 4
CONTEXT_LENGTH = 20
MAX_EP_LEN = 200
DROPOUT = 0.1

# Training
NUM_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500

# Evaluation
EVAL_EPISODES = 50
DT_TARGET_RETURN = 50.0


# ======================================================================
# Greedy policy (env-aware, copied from run_dt_4x4_full.py)
# ======================================================================

class _EnvAwareGreedyPolicy:
    """Greedy preemption policy that reads the env state directly."""

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
            for ri, (_, link_id) in enumerate(route):
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
# Helpers
# ======================================================================

def _normalize_obs(
    obs: np.ndarray,
    state_mean: np.ndarray | None,
    state_std: np.ndarray | None,
) -> np.ndarray:
    if state_mean is not None and state_std is not None:
        return ((obs - state_mean) / state_std).astype(np.float32)
    return obs.astype(np.float32)


def generate_dataset(seed: int) -> str:
    """Generate 200 mixed-quality episodes for a given seed."""
    data_path = str(DATA_DIR / f"dt_4x4_3seed_s{seed}.h5")
    if Path(data_path).exists():
        print(f"  Dataset already exists: {data_path}")
        return data_path

    env_kwargs = {**ENV_KWARGS_BASE, "seed": seed}
    env = EVCorridorEnv(**env_kwargs)
    expert = _EnvAwareGreedyPolicy(env)
    collector = DataCollector(env=env, save_path=data_path)

    np.random.seed(seed)
    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=NUM_EXPERT,
        num_random=NUM_RANDOM,
        num_suboptimal=NUM_SUBOPTIMAL,
    )
    collector.save_dataset()
    return data_path


def train_dt(data_path: str, seed: int, device: str) -> tuple:
    """Train DT for 30 epochs. Returns (model, dataset)."""
    model_path = MODEL_DIR / f"dt_4x4_3seed_s{seed}.pt"
    dataset = TrajectoryDataset(
        data_path=data_path,
        context_length=CONTEXT_LENGTH,
        normalize_states=True,
        normalize_returns=True,
    )

    if model_path.exists():
        print(f"  Model already exists: {model_path}, loading...")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = DecisionTransformer(
            state_dim=ckpt["state_dim"],
            act_dim=ckpt["act_dim"],
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            max_length=CONTEXT_LENGTH,
            max_ep_len=MAX_EP_LEN,
            dropout=DROPOUT,
            activation="gelu",
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, dataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    print(f"  Dataset: {len(dataset)} segments, "
          f"state_dim={state_dim}, act_dim={act_dim}")
    print(f"  Model params: {model.get_num_params():,}")
    print(f"  Training for {NUM_EPOCHS} epochs ({total_steps} steps)")

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

            loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
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
        "seed": seed,
    }, model_path)
    print(f"  Model saved to {model_path}")

    return model, dataset


def evaluate_dt_episodes(
    model: DecisionTransformer,
    env: EVCorridorEnv,
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

    for _ in range(n_episodes):
        obs, info = env.reset()
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
        states_buf[0, 0] = torch.tensor(
            obs_norm, dtype=torch.float32, device=device
        )
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

            obs, reward, terminated, truncated, info = env.step(action)
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


def evaluate_ft_evp(seed: int, n_episodes: int) -> list[dict]:
    """Evaluate Fixed-Time EVP baseline."""
    ft_evp = FixedTimeEVP()
    episodes_info = []
    for _ in range(n_episodes):
        env = EVCorridorEnv(**{**ENV_KWARGS_BASE, "seed": seed})
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


def evaluate_greedy(seed: int, n_episodes: int) -> list[dict]:
    """Evaluate Greedy Preempt baseline."""
    episodes_info = []
    for _ in range(n_episodes):
        env = EVCorridorEnv(**{**ENV_KWARGS_BASE, "seed": seed})
        greedy = _EnvAwareGreedyPolicy(env)
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


def run_seed(seed: int, device: str) -> dict:
    """Run full pipeline for one seed. Returns per-method metrics."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # Step 1: Generate dataset
    print(f"\n  [1/3] Generating dataset (seed={seed})...")
    data_path = generate_dataset(seed)

    # Step 2: Train DT
    print(f"\n  [2/3] Training DT (seed={seed})...")
    model, dataset = train_dt(data_path, seed, device)

    # Step 3: Evaluate all methods
    print(f"\n  [3/3] Evaluating methods (seed={seed})...")
    state_mean, state_std = dataset.get_state_stats()
    return_scale = dataset.get_return_scale()

    # DT
    print(f"    DT (target={DT_TARGET_RETURN})...")
    env_dt = EVCorridorEnv(**{**ENV_KWARGS_BASE, "seed": seed})
    dt_infos = evaluate_dt_episodes(
        model, env_dt, target_return=DT_TARGET_RETURN,
        n_episodes=EVAL_EPISODES, device=device,
        state_mean=state_mean, state_std=state_std,
        return_scale=return_scale,
    )
    dt_metrics = aggregate_metrics(dt_infos)

    # FT-EVP
    print("    FT-EVP...")
    ft_infos = evaluate_ft_evp(seed, EVAL_EPISODES)
    ft_metrics = aggregate_metrics(ft_infos)

    # Greedy
    print("    Greedy...")
    greedy_infos = evaluate_greedy(seed, EVAL_EPISODES)
    greedy_metrics = aggregate_metrics(greedy_infos)

    # Print seed summary
    print(f"\n  Seed {seed} Summary:")
    print(f"  {'Method':<12s} {'Return':>10s} {'EV Time':>10s}")
    print(f"  {'-'*34}")
    for name, m in [("DT", dt_metrics), ("FT-EVP", ft_metrics),
                    ("Greedy", greedy_metrics)]:
        ret = m.get("mean_return", 0)
        evt = m.get("mean_ev_travel_time", -1)
        print(f"  {name:<12s} {ret:>10.2f} {evt:>10.1f}")

    return {
        "DT": dt_metrics,
        "FT-EVP": ft_metrics,
        "Greedy": greedy_metrics,
    }


def _serialize(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    t_total = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Run each seed
    per_seed_results = {}
    for seed in SEEDS:
        per_seed_results[seed] = run_seed(seed, device)

    # Build output structure
    methods_data: dict = {}
    for method in ["DT", "FT-EVP", "Greedy"]:
        ev_times = []
        returns = []
        for seed in SEEDS:
            m = per_seed_results[seed][method]
            ev_times.append(float(m["mean_ev_travel_time"]))
            returns.append(float(m["mean_return"]))
        methods_data[method] = {
            "ev_times": ev_times,
            "returns": returns,
        }

    # Aggregate across seeds
    aggregate: dict = {}
    for method in ["DT", "FT-EVP", "Greedy"]:
        ev_arr = np.array(methods_data[method]["ev_times"])
        ret_arr = np.array(methods_data[method]["returns"])
        aggregate[method] = {
            "ev_time_mean": float(np.mean(ev_arr)),
            "ev_time_std": float(np.std(ev_arr)),
            "return_mean": float(np.mean(ret_arr)),
            "return_std": float(np.std(ret_arr)),
        }

    results = {
        "seeds": SEEDS,
        "methods": _serialize(methods_data),
        "aggregate": _serialize(aggregate),
    }

    # Print final aggregate summary
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS (mean +/- std across 3 seeds)")
    print(f"{'='*60}")
    print(f"  {'Method':<12s} {'Return':>18s} {'EV Time':>18s}")
    print(f"  {'-'*50}")
    for method in ["DT", "FT-EVP", "Greedy"]:
        a = aggregate[method]
        ret_str = f"{a['return_mean']:.2f} +/- {a['return_std']:.2f}"
        evt_str = f"{a['ev_time_mean']:.1f} +/- {a['ev_time_std']:.1f}"
        print(f"  {method:<12s} {ret_str:>18s} {evt_str:>18s}")
    print(f"  {'-'*50}")

    # Save results
    output_path = RESULTS_DIR / "dt_4x4_3seed.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    elapsed = time.time() - t_total
    print(f"\n  Results saved to {output_path}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
