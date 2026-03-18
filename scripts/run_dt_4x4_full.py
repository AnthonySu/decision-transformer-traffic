#!/usr/bin/env python3
"""Full Decision Transformer experiment on 4x4 grid.

Steps:
  1. Generate 500 episodes (350 expert, 75 random, 75 suboptimal)
  2. Train DT with hidden_dim=128, 3 layers, 4 heads, context=20, 50 epochs
  3. Evaluate with 50 episodes against: FT-EVP, Greedy, Random
  4. Return conditioning sweep: targets [50,25,0,-25,-50,-100,-200] x 30 episodes
  5. Save all results to results/dt_4x4_full.json
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

DATA_PATH = str(_PROJECT_ROOT / "data" / "dt_4x4_full.h5")
MODEL_DIR = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"

ENV_KWARGS = dict(
    rows=4, cols=4, max_steps=150, seed=42,
    origin="n0_0", destination="n3_3",
)

# Dataset
NUM_EXPERT = 350
NUM_RANDOM = 75
NUM_SUBOPTIMAL = 75

# Model
HIDDEN_DIM = 128
N_LAYERS = 3
N_HEADS = 4
CONTEXT_LENGTH = 20
MAX_EP_LEN = 200
DROPOUT = 0.1

# Training
NUM_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500

# Evaluation
EVAL_EPISODES = 50
SWEEP_TARGETS = [50, 25, 0, -25, -50, -100, -200]
SWEEP_EPISODES = 30


# ======================================================================
# Step 1: Generate dataset
# ======================================================================

def generate_dataset() -> str:
    """Generate 500 mixed-quality episodes and save to HDF5."""
    print("\n" + "=" * 60)
    print("  STEP 1: Generate Dataset")
    print("=" * 60)

    env = EVCorridorEnv(**ENV_KWARGS)

    # The GreedyPreemptPolicy needs special handling for this env's API.
    # The env doesn't expose ev_info in a way the policy expects, so
    # we create a wrapper policy that works with the env's observation.
    expert = _EnvAwareGreedyPolicy(env)

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

    # Print per-policy stats
    policy_names = set(ep["policy_name"] for ep in collector._episodes)
    for pname in sorted(policy_names):
        eps = [ep for ep in collector._episodes if ep["policy_name"] == pname]
        rets = [ep["episode_return"] for ep in eps]
        print(f"    {pname:12s}: n={len(eps)}, return={np.mean(rets):.2f}")

    collector.save_dataset()
    return DATA_PATH


class _EnvAwareGreedyPolicy:
    """Greedy policy adapted for the EVCorridorEnv observation interface.

    The env uses a flat obs vector; the greedy heuristic just picks the phase
    that matches the EV's current link direction at each intersection.
    Since GreedyPreemptPolicy expects ev_info with 'phase' etc., and the
    data collector passes (obs, ev_info={}) -- we implement a simpler version
    that reads the phase from the env's network directly.
    """

    def __init__(self, env: EVCorridorEnv):
        self.env = env

    def select_action(self, obs: np.ndarray, ev_info: dict) -> int:
        """Pick the phase that gives green to the EV at each intersection."""
        env = self.env
        network = env._network
        route = env._route
        route_intersections = env._route_intersections
        ev_link_idx = env._ev_link_idx

        # For each route intersection, try to give green to EV's approach link
        actions = np.zeros(env._max_route_len, dtype=np.int64)

        for i, node_id in enumerate(route_intersections):
            if i >= env._max_route_len:
                break
            # Find the link from the route that enters this intersection
            # The EV approaches intersection i via route link i
            # Check if the EV is approaching this intersection
            if i < len(route) - 1:
                _, link_id = route[i]
                if link_id is not None:
                    link = network["links"][link_id]
                    actions[i] = link["phase_index"]
            else:
                # Default to phase 0 for the last intersection
                actions[i] = 0

        # Return scalar action (first intersection phase) for compatibility
        return int(actions[0])

    def reset(self) -> None:
        pass


# ======================================================================
# Step 2: Train DT
# ======================================================================

def train_dt(data_path: str, device: str) -> tuple:
    """Train Decision Transformer. Returns (model, dataset) tuple."""
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

            loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Final loss: {training_losses[-1]:.4f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    save_path = MODEL_DIR / "dt_4x4_full.pt"
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
    }, save_path)
    print(f"  Model saved to {save_path}")

    return model, dataset


# ======================================================================
# Step 3: Evaluate against baselines
# ======================================================================

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

        # Normalize initial obs
        obs_norm = _normalize_obs(obs, state_mean, state_std)
        states_buf[0, 0] = torch.tensor(obs_norm, dtype=torch.float32, device=device)
        # Normalize target return
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
                rtg_buf[0, t, 0] = rtg_buf[0, t - 1, 0] - reward / return_scale
                timesteps_buf[0, t] = min(t, MAX_EP_LEN - 1)

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })

    return episodes_info


def _normalize_obs(
    obs: np.ndarray,
    state_mean: np.ndarray | None,
    state_std: np.ndarray | None,
) -> np.ndarray:
    """Z-score normalize observation if stats are provided."""
    if state_mean is not None and state_std is not None:
        return ((obs - state_mean) / state_std).astype(np.float32)
    return obs.astype(np.float32)


def evaluate_baselines(
    model: DecisionTransformer,
    dataset: TrajectoryDataset,
    device: str,
) -> dict:
    """Evaluate DT + baselines, return results dict."""
    print("\n" + "=" * 60)
    print("  STEP 3: Evaluate Against Baselines")
    print("=" * 60)

    state_mean, state_std = dataset.get_state_stats()
    return_scale = dataset.get_return_scale()

    results = {}

    # --- DT (best target) ---
    print("  Evaluating Decision Transformer (target=50)...")
    env = EVCorridorEnv(**ENV_KWARGS)
    dt_infos = evaluate_dt_episodes(
        model, env, target_return=50.0,
        n_episodes=EVAL_EPISODES, device=device,
        state_mean=state_mean, state_std=state_std,
        return_scale=return_scale,
    )
    results["DT (target=50)"] = aggregate_metrics(dt_infos)

    # --- Fixed-Time EVP ---
    print("  Evaluating Fixed-Time EVP...")
    ft_evp = FixedTimeEVP()
    ft_infos = _run_baseline_policy(ft_evp, EVAL_EPISODES)
    results["FT-EVP"] = aggregate_metrics(ft_infos)

    # --- Greedy Preempt ---
    print("  Evaluating Greedy Preempt...")
    env2 = EVCorridorEnv(**ENV_KWARGS)
    greedy = _EnvAwareGreedyPolicy(env2)
    greedy_infos = _run_baseline_with_env(greedy, env2, EVAL_EPISODES)
    results["Greedy"] = aggregate_metrics(greedy_infos)

    # --- Random ---
    print("  Evaluating Random...")
    random_infos = _run_random_policy(EVAL_EPISODES)
    results["Random"] = aggregate_metrics(random_infos)

    # Print summary
    print("\n  " + "-" * 56)
    print(f"  {'Method':<20s} {'Return':>10s} {'EV Time':>10s} "
          f"{'BG Delay':>10s} {'Length':>8s}")
    print("  " + "-" * 56)
    for method, m in results.items():
        ret = m.get("mean_return", 0)
        evt = m.get("mean_ev_travel_time", -1)
        bgd = m.get("background_delay_mean", 0)
        length = m.get("mean_length", 0)
        print(f"  {method:<20s} {ret:>10.2f} {evt:>10.1f} "
              f"{bgd:>10.2f} {length:>8.1f}")
    print("  " + "-" * 56)

    return results


def _run_baseline_policy(policy, n_episodes: int) -> list[dict]:
    """Run a baseline policy that uses select_action(obs, ev_info)."""
    episodes_info = []
    for _ in range(n_episodes):
        env = EVCorridorEnv(**ENV_KWARGS)
        obs, info = env.reset()
        policy.reset()
        done = False
        episode_return = 0.0
        t = 0
        step_infos = []

        while not done:
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return, "length": t, "step_infos": step_infos,
        })
    return episodes_info


def _run_baseline_with_env(policy, env, n_episodes: int) -> list[dict]:
    """Run a baseline policy that needs access to the env instance."""
    episodes_info = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        episode_return = 0.0
        t = 0
        step_infos = []

        while not done:
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return, "length": t, "step_infos": step_infos,
        })
    return episodes_info


def _run_random_policy(n_episodes: int) -> list[dict]:
    """Run uniform random policy."""
    episodes_info = []
    for _ in range(n_episodes):
        env = EVCorridorEnv(**ENV_KWARGS)
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


# ======================================================================
# Step 4: Return conditioning sweep
# ======================================================================

def return_conditioning_sweep(
    model: DecisionTransformer,
    dataset: TrajectoryDataset,
    device: str,
) -> dict:
    """Sweep target returns to demonstrate controllability."""
    print("\n" + "=" * 60)
    print("  STEP 4: Return Conditioning Sweep")
    print("=" * 60)

    state_mean, state_std = dataset.get_state_stats()
    return_scale = dataset.get_return_scale()

    sweep_results = {}

    for target in SWEEP_TARGETS:
        print(f"  target_return = {target:>6.0f} ...", end="", flush=True)
        env = EVCorridorEnv(**ENV_KWARGS)
        infos = evaluate_dt_episodes(
            model, env, target_return=float(target),
            n_episodes=SWEEP_EPISODES, device=device,
            state_mean=state_mean, state_std=state_std,
            return_scale=return_scale,
        )
        metrics = aggregate_metrics(infos)
        per_ep_returns = [ep["return"] for ep in infos]
        metrics["target_return"] = target
        metrics["actual_return_mean"] = float(np.mean(per_ep_returns))
        metrics["actual_return_std"] = float(np.std(per_ep_returns))

        sweep_results[f"target_{target}"] = metrics
        print(f"  actual_return={metrics['actual_return_mean']:.1f} "
              f"+/- {metrics['actual_return_std']:.1f}, "
              f"ev_time={metrics['mean_ev_travel_time']:.1f}")

    return sweep_results


# ======================================================================
# Step 5: Save all results
# ======================================================================

def save_results(
    baseline_results: dict,
    sweep_results: dict,
    training_losses: list,
) -> None:
    """Save everything to results/dt_4x4_full.json."""
    print("\n" + "=" * 60)
    print("  STEP 5: Save Results")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "dt_4x4_full.json"

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

    results = {
        "experiment": "dt_4x4_full",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "env": ENV_KWARGS,
            "dataset": {
                "num_expert": NUM_EXPERT,
                "num_random": NUM_RANDOM,
                "num_suboptimal": NUM_SUBOPTIMAL,
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
                "sweep_targets": SWEEP_TARGETS,
                "sweep_episodes": SWEEP_EPISODES,
            },
        },
        "training_losses": training_losses,
        "baseline_comparison": _serialize(baseline_results),
        "return_conditioning_sweep": _serialize(sweep_results),
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

    t_total = time.time()

    model_path = MODEL_DIR / "dt_4x4_full.pt"
    data_path = DATA_PATH

    # Step 1: Generate dataset (skip if exists)
    if not Path(data_path).exists():
        data_path = generate_dataset()
    else:
        print(f"\n  Dataset already exists at {data_path}, skipping generation.")

    # Step 2: Train DT (skip if model exists)
    if not model_path.exists():
        model, dataset = train_dt(data_path, device)
    else:
        print(f"\n  Model already exists at {model_path}, loading...")
        dataset = TrajectoryDataset(
            data_path=data_path,
            context_length=CONTEXT_LENGTH,
            normalize_states=True,
            normalize_returns=True,
        )
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

    # Step 3: Evaluate against baselines
    baseline_results = evaluate_baselines(model, dataset, device)

    # Step 4: Return conditioning sweep
    sweep_results = return_conditioning_sweep(model, dataset, device)

    # Step 5: Save results
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    training_losses = ckpt.get("training_losses", [])
    save_results(baseline_results, sweep_results, training_losses)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Experiment complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
