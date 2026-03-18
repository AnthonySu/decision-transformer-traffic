#!/usr/bin/env python3
"""End-to-end smoke test: verify full pipeline works with tiny parameters.

Exercises the complete EV-DT pipeline in under 2 minutes on CPU:
  1. Create a small 3x3 grid environment
  2. Collect 20 episodes (14 expert + 3 random + 3 suboptimal)
  3. Save tiny dataset to data/smoke_test.h5
  4. Train DT for 3 epochs with tiny model (hidden=32, layers=1, heads=2, context=5)
  5. Train MADT for 3 epochs with tiny model
  6. Create and briefly train PPO for 500 steps
  7. Evaluate all methods for 5 episodes each
  8. Print comparison table
  9. Generate one demo figure
 10. Assert no errors occurred

Exit code 0 on success, 1 on failure.

Usage::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-baselines   # skip SB3 if not installed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIVIDER = "=" * 60
_errors: list[str] = []


def _banner(msg: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {msg}")
    print(_DIVIDER)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str, exc: Exception | None = None) -> None:
    detail = f"{msg}: {exc}" if exc else msg
    print(f"  [FAIL] {detail}")
    _errors.append(detail)
    if exc:
        traceback.print_exc()


def _load_config(path: str = "configs/smoke_test.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# Step 1 & 2 & 3: Environment + Data Collection
# ======================================================================

def step_collect_dataset(cfg: dict) -> str:
    """Create env, collect mixed dataset, save to HDF5. Returns save_path."""
    _banner("STEP 1-3: Environment + Dataset Collection")

    from src.baselines.greedy_preempt import GreedyPreemptPolicy
    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.utils.data_collector import DataCollector

    env_cfg = cfg["env"]
    ds_cfg = cfg["dataset"]

    rows = env_cfg.get("rows", 3)
    cols = env_cfg.get("cols", 3)
    max_steps = env_cfg.get("max_episode_steps", 50)

    # --- 1. Create environment ---
    env = EVCorridorEnv(rows=rows, cols=cols, max_steps=max_steps)
    obs, info = env.reset()
    print(f"  Environment: {rows}x{cols} grid, obs_dim={obs.shape[0]}, "
          f"action_space={env.action_space}")
    _ok("Environment created")

    # --- 2. Collect episodes ---
    save_path = ds_cfg.get("save_path", "data/smoke_test.h5")
    total_episodes = ds_cfg.get("num_episodes", 20)
    suboptimal_ratio = ds_cfg.get("suboptimal_ratio", 0.3)

    num_suboptimal = int(total_episodes * suboptimal_ratio * 0.5)
    num_random = int(total_episodes * suboptimal_ratio * 0.5)
    num_expert = total_episodes - num_suboptimal - num_random

    print(f"  Collecting: {num_expert} expert + {num_random} random + "
          f"{num_suboptimal} suboptimal = {total_episodes} episodes")

    # Build a greedy expert (uses network topology + route from the env)
    network = env._network
    route_intersections = env._route_intersections
    expert = GreedyPreemptPolicy(network=network, route=route_intersections)

    collector = DataCollector(env=env, save_path=save_path)
    t0 = time.time()
    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=num_expert,
        num_random=num_random,
        num_suboptimal=num_suboptimal,
    )
    elapsed = time.time() - t0
    print(f"  Collection took {elapsed:.1f}s")

    # --- 3. Save ---
    collector.save_dataset()
    assert Path(save_path).exists(), f"Dataset file missing: {save_path}"
    file_kb = Path(save_path).stat().st_size / 1024
    _ok(f"Dataset saved: {save_path} ({file_kb:.1f} KB, {total_episodes} episodes)")

    return save_path


# ======================================================================
# Step 4: Train Decision Transformer
# ======================================================================

def step_train_dt(cfg: dict, save_path: str) -> Path | None:
    """Train DT with tiny params. Returns path to saved model or None on failure."""
    _banner("STEP 4: Train Decision Transformer")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.models.decision_transformer import DecisionTransformer
    from src.models.trajectory_dataset import TrajectoryDataset

    dt_cfg = cfg["dt"]

    try:
        dataset = TrajectoryDataset(
            data_path=save_path,
            context_length=dt_cfg["context_length"],
        )
        print(f"  Dataset: {len(dataset)} segments, "
              f"state_dim={dataset.state_dim if hasattr(dataset, 'state_dim') else '?'}")

        # Infer dims from first sample
        sample = dataset[0]
        state_dim = sample["states"].shape[-1]
        # act_dim: infer from env or use a reasonable default
        env_cfg = cfg["env"]
        tmp_env = EVCorridorEnv(
            rows=env_cfg.get("rows", 3),
            cols=env_cfg.get("cols", 3),
            max_steps=env_cfg.get("max_episode_steps", 50),
        )
        tmp_env.reset()
        # action_space is MultiDiscrete; act_dim = number of per-intersection choices
        act_dim = int(tmp_env.action_space.nvec[0])

        dataloader = DataLoader(
            dataset,
            batch_size=dt_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
        )

        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=dt_cfg["embed_dim"],
            n_layers=dt_cfg["n_layers"],
            n_heads=dt_cfg["n_heads"],
            max_length=dt_cfg["context_length"],
            max_ep_len=dt_cfg.get("max_ep_len", 50),
            dropout=dt_cfg["dropout"],
            activation=dt_cfg["activation"],
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} parameters")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=dt_cfg["lr"], weight_decay=dt_cfg["weight_decay"]
        )
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(1, dt_cfg["num_epochs"] + 1):
            model.train()
            epoch_losses = []
            for batch in dataloader:
                states = batch["states"]
                actions = batch["actions"]
                rtg = batch["returns_to_go"]
                timesteps = batch["timesteps"]

                action_logits = model(states, actions, rtg, timesteps)

                # Flatten and compute loss
                logits_flat = action_logits.reshape(-1, act_dim)
                targets_flat = actions.reshape(-1)
                # Clamp action targets to valid range
                targets_flat = targets_flat.clamp(0, act_dim - 1)

                # Use attention_mask to mask padding
                mask = batch["attention_mask"].reshape(-1).bool()
                if mask.any():
                    loss = loss_fn(logits_flat[mask], targets_flat[mask])
                else:
                    loss = loss_fn(logits_flat, targets_flat)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            print(f"    Epoch {epoch}/{dt_cfg['num_epochs']} | Loss: {avg_loss:.4f}")

        # Save
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "dt_smoke.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": dt_cfg,
            "state_dim": state_dim,
            "act_dim": act_dim,
        }, model_path)

        _ok(f"DT trained and saved to {model_path}")
        return model_path

    except Exception as exc:
        _fail("DT training failed", exc)
        return None


# ======================================================================
# Step 5: Train MADT
# ======================================================================

def step_train_madt(cfg: dict, save_path: str) -> Path | None:
    """Train MADT with tiny params. Returns path to saved model or None."""
    _banner("STEP 5: Train Multi-Agent Decision Transformer")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
    from src.models.madt import MultiAgentDecisionTransformer

    madt_cfg = cfg["madt"]
    env_cfg = cfg["env"]

    try:
        rows = env_cfg.get("rows", 3)
        cols = env_cfg.get("cols", 3)
        max_steps = env_cfg.get("max_episode_steps", 50)

        # Create MA env to determine n_agents and obs/act dims
        ma_env = EVCorridorMAEnv(rows=rows, cols=cols, max_steps=max_steps)
        obs_dict, info_dict = ma_env.reset()
        n_agents = len(ma_env.agents)
        state_dim = next(iter(obs_dict.values())).shape[0]
        act_dim = 4  # Discrete(4) per agent

        print(f"  MA env: {n_agents} agents, state_dim={state_dim}, act_dim={act_dim}")

        # Build adjacency matrix from the network topology
        # Use route intersections as nodes, connect adjacent ones
        adj = torch.eye(n_agents, dtype=torch.float32)
        for i in range(n_agents - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0

        # Create a synthetic MA dataset from the single-agent dataset
        # (the real pipeline would use a separate MA dataset, but for the
        # smoke test we fabricate compatible tensors)
        print("  Building synthetic MA training batch...")

        # Run a few episodes in the MA env to get training data
        all_states = []
        all_actions = []
        all_rtg = []

        n_collect = 10
        for _ in range(n_collect):
            obs_dict, _ = ma_env.reset()
            ep_states = []
            ep_actions = []
            ep_rewards = []
            done = False
            agents_snapshot = list(ma_env.agents)

            for _step in range(max_steps):
                if not ma_env.agents:
                    break
                # Random actions
                action_dict = {a: ma_env.action_space(a).sample() for a in ma_env.agents}
                s_arr = np.stack([obs_dict[a] for a in agents_snapshot])
                a_arr = np.array([action_dict.get(a, 0) for a in agents_snapshot])
                ep_states.append(s_arr)
                ep_actions.append(a_arr)

                obs_dict, rew_dict, term_dict, trunc_dict, info_dict = ma_env.step(action_dict)
                r_arr = np.array([rew_dict.get(a, 0.0) for a in agents_snapshot])
                ep_rewards.append(r_arr)

                if not ma_env.agents:
                    break

            if len(ep_states) < 2:
                continue

            ep_states = np.stack(ep_states)    # [T, N, state_dim]
            ep_actions = np.stack(ep_actions)  # [T, N]
            ep_rewards = np.stack(ep_rewards)  # [T, N]

            # Compute per-agent returns-to-go
            T_ep = ep_rewards.shape[0]
            rtg = np.zeros_like(ep_rewards)
            for ai in range(n_agents):
                running = 0.0
                for t in reversed(range(T_ep)):
                    running += ep_rewards[t, ai]
                    rtg[t, ai] = running

            all_states.append(ep_states)
            all_actions.append(ep_actions)
            all_rtg.append(rtg)

        if not all_states:
            _fail("No MA episodes collected")
            return None

        # Build a simple training loop over these episodes
        K = madt_cfg["context_length"]

        model = MultiAgentDecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_agents=n_agents,
            adj_matrix=adj,
            hidden_dim=madt_cfg["embed_dim"],
            n_layers=madt_cfg["n_layers"],
            n_heads=madt_cfg["n_heads"],
            gat_heads=madt_cfg["gat_heads"],
            gat_layers=madt_cfg["gat_layers"],
            max_length=K,
            max_ep_len=madt_cfg.get("max_ep_len", 50),
            dropout=madt_cfg["dropout"],
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} parameters")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=madt_cfg["lr"], weight_decay=madt_cfg["weight_decay"]
        )
        loss_fn = nn.CrossEntropyLoss()

        # Assemble fixed-size training batch from collected episodes
        def _make_batch(episodes_states, episodes_actions, episodes_rtg):
            """Sample context windows and build a batch."""
            batch_s, batch_a, batch_r, batch_t, batch_m = [], [], [], [], []

            for s_ep, a_ep, r_ep in zip(episodes_states, episodes_actions, episodes_rtg):
                T_ep = s_ep.shape[0]
                # Pick a random end point
                end = min(T_ep, K)
                start = max(end - K, 0)
                actual = end - start
                pad = K - actual

                # s_ep: [T, N, D] -> extract window -> [N, K, D]
                sw = s_ep[start:end].transpose(1, 0, 2)  # [N, actual, D]
                aw = a_ep[start:end].T                     # [N, actual]
                rw = r_ep[start:end].T                     # [N, actual]
                tw = np.arange(start, end, dtype=np.int64)

                ps = np.zeros((n_agents, K, state_dim), dtype=np.float32)
                pa = np.zeros((n_agents, K), dtype=np.int64)
                pr = np.zeros((n_agents, K), dtype=np.float32)
                pt = np.zeros((n_agents, K), dtype=np.int64)
                pm = np.zeros(K, dtype=np.float32)

                ps[:, pad:, :] = sw
                pa[:, pad:] = aw
                pr[:, pad:] = rw
                pt[:, pad:] = np.broadcast_to(tw, (n_agents, actual))
                pm[pad:] = 1.0

                batch_s.append(ps)
                batch_a.append(pa)
                batch_r.append(pr)
                batch_t.append(pt)
                batch_m.append(pm)

            return (
                torch.tensor(np.stack(batch_s), dtype=torch.float32),
                torch.tensor(np.stack(batch_a), dtype=torch.long),
                torch.tensor(np.stack(batch_r), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.stack(batch_t), dtype=torch.long),
                torch.tensor(np.stack(batch_m), dtype=torch.float32),
            )

        states_b, actions_b, rtg_b, timesteps_b, masks_b = _make_batch(
            all_states, all_actions, all_rtg
        )

        for epoch in range(1, madt_cfg["num_epochs"] + 1):
            model.train()
            # Forward: model expects [B, N, T, ...]
            logits = model(states_b, actions_b, rtg_b, timesteps_b)
            # logits: [B, N, T, act_dim]
            logits_flat = logits.reshape(-1, act_dim)
            targets_flat = actions_b.reshape(-1).clamp(0, act_dim - 1)
            mask_flat = masks_b.unsqueeze(1).expand(-1, n_agents, -1).reshape(-1).bool()

            if mask_flat.any():
                loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])
            else:
                loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f"    Epoch {epoch}/{madt_cfg['num_epochs']} | Loss: {loss.item():.4f}")

        # Save
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "madt_smoke.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": madt_cfg,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "n_agents": n_agents,
            "adj_matrix": adj,
        }, model_path)

        _ok(f"MADT trained and saved to {model_path}")
        return model_path

    except Exception as exc:
        _fail("MADT training failed", exc)
        return None


# ======================================================================
# Step 6: Train PPO baseline
# ======================================================================

def step_train_ppo(cfg: dict) -> bool:
    """Create and briefly train PPO. Returns True on success."""
    _banner("STEP 6: Train PPO Baseline (500 steps)")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("  stable-baselines3 not installed -- skipping PPO")
        return False

    from src.envs.ev_corridor_env import EVCorridorEnv

    env_cfg = cfg["env"]
    ppo_cfg = cfg["baselines"]["ppo"]

    try:
        env = EVCorridorEnv(
            rows=env_cfg.get("rows", 3),
            cols=env_cfg.get("cols", 3),
            max_steps=env_cfg.get("max_episode_steps", 50),
        )
        wrapped = DummyVecEnv([lambda: Monitor(env)])

        ppo = PPO(
            policy="MlpPolicy",
            env=wrapped,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 64),
            batch_size=ppo_cfg.get("batch_size", 32),
            n_epochs=ppo_cfg.get("n_epochs", 2),
            gamma=ppo_cfg.get("gamma", 0.99),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            verbose=0,
            device="cpu",
        )

        ppo.learn(total_timesteps=ppo_cfg.get("total_timesteps", 500))

        model_dir = Path("models/ppo_smoke")
        model_dir.mkdir(parents=True, exist_ok=True)
        ppo.save(str(model_dir / "model"))

        _ok(f"PPO trained ({ppo_cfg.get('total_timesteps', 500)} steps)")
        return True

    except Exception as exc:
        _fail("PPO training failed", exc)
        return False


# ======================================================================
# Step 7 & 8: Evaluate all methods
# ======================================================================

def step_evaluate(cfg: dict, dt_path: Path | None, madt_path: Path | None,
                  ppo_trained: bool) -> dict | None:
    """Evaluate all available methods and print comparison table."""
    _banner("STEP 7-8: Evaluation")

    import torch

    from src.baselines.fixed_time_evp import FixedTimeEVP
    from src.baselines.greedy_preempt import GreedyPreemptPolicy
    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.utils.metrics import aggregate_metrics, compare_methods

    env_cfg = cfg["env"]
    n_episodes = cfg["eval"].get("num_eval_episodes", 5)
    rows = env_cfg.get("rows", 3)
    cols = env_cfg.get("cols", 3)
    max_steps = env_cfg.get("max_episode_steps", 50)

    try:
        env = EVCorridorEnv(rows=rows, cols=cols, max_steps=max_steps)

        results: dict[str, dict] = {}

        # --- Evaluate a generic policy ---
        def _eval_policy(env, policy_fn, n_ep):
            episodes_info = []
            for _ in range(n_ep):
                obs, info = env.reset()
                done = False
                ep_return = 0.0
                t = 0
                step_infos = []
                while not done:
                    action = policy_fn(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_return += reward
                    step_infos.append(info)
                    t += 1
                episodes_info.append({
                    "return": ep_return, "length": t, "step_infos": step_infos
                })
            return episodes_info

        # 1. Fixed-Time EVP
        print("  Evaluating Fixed-Time EVP...")
        fixed = FixedTimeEVP()
        fixed_info = _eval_policy(env, lambda o, i: fixed.select_action(o, i), n_episodes)
        results["Fixed-Time EVP"] = aggregate_metrics(fixed_info)

        # 2. Greedy Preemption
        print("  Evaluating Greedy Preemption...")
        greedy = GreedyPreemptPolicy(network=env._network, route=env._route_intersections)
        greedy_info = _eval_policy(env, lambda o, i: greedy.select_action(o, i), n_episodes)
        results["Greedy Preempt"] = aggregate_metrics(greedy_info)

        # 3. Random baseline
        print("  Evaluating Random Policy...")
        random_info = _eval_policy(
            env, lambda o, i: env.action_space.sample(), n_episodes
        )
        results["Random"] = aggregate_metrics(random_info)

        # 4. DT (if trained)
        if dt_path and dt_path.exists():
            print("  Evaluating Decision Transformer...")
            from src.models.decision_transformer import DecisionTransformer
            ckpt = torch.load(dt_path, map_location="cpu", weights_only=False)
            dt_model = DecisionTransformer(
                state_dim=ckpt["state_dim"],
                act_dim=ckpt["act_dim"],
                hidden_dim=ckpt["config"]["embed_dim"],
                n_layers=ckpt["config"]["n_layers"],
                n_heads=ckpt["config"]["n_heads"],
                max_length=ckpt["config"]["context_length"],
                max_ep_len=ckpt["config"].get("max_ep_len", 50),
            )
            dt_model.load_state_dict(ckpt["model_state_dict"])
            dt_model.eval()

            dt_episodes = []
            for _ in range(n_episodes):
                obs, info = env.reset()
                done = False
                ep_return = 0.0
                t = 0
                step_infos = []
                s_dim = ckpt["state_dim"]
                a_dim = ckpt["act_dim"]
                ctx = ckpt["config"]["context_length"]

                states_buf = torch.zeros(1, ctx, s_dim)
                actions_buf = torch.zeros(1, ctx, dtype=torch.long)
                rtg_buf = torch.zeros(1, ctx, 1)
                ts_buf = torch.zeros(1, ctx, dtype=torch.long)

                states_buf[0, 0] = torch.tensor(obs[:s_dim], dtype=torch.float32)
                rtg_buf[0, 0, 0] = 0.0  # target return
                ts_buf[0, 0] = 0

                while not done:
                    c = min(t + 1, ctx)
                    action = dt_model.get_action(
                        states_buf[:, :c], actions_buf[:, :c],
                        rtg_buf[:, :c], ts_buf[:, :c],
                    )
                    # Convert scalar action to MultiDiscrete array
                    act_array = np.full(env.action_space.shape, action, dtype=int)
                    obs, reward, terminated, truncated, info = env.step(act_array)
                    done = terminated or truncated
                    ep_return += reward
                    step_infos.append(info)
                    t += 1
                    if t < ctx:
                        states_buf[0, t] = torch.tensor(obs[:s_dim], dtype=torch.float32)
                        actions_buf[0, t - 1] = action
                        rtg_buf[0, t, 0] = rtg_buf[0, t - 1, 0] - reward
                        ts_buf[0, t] = min(t, 49)

                dt_episodes.append({
                    "return": ep_return, "length": t, "step_infos": step_infos
                })
            results["DT (target=0)"] = aggregate_metrics(dt_episodes)

        # 5. PPO (if trained)
        if ppo_trained:
            try:
                from stable_baselines3 import PPO
                ppo_path = Path("models/ppo_smoke/model.zip")
                if ppo_path.exists():
                    print("  Evaluating PPO...")
                    ppo_model = PPO.load(str(ppo_path.with_suffix("")))
                    ppo_info = _eval_policy(
                        env,
                        lambda o, i: ppo_model.predict(o, deterministic=True)[0],
                        n_episodes,
                    )
                    results["PPO"] = aggregate_metrics(ppo_info)
            except Exception as exc:
                _fail("PPO evaluation failed", exc)

        # --- Print comparison table ---
        print()
        try:
            df = compare_methods(results)
            # Select key columns if available
            display_cols = [c for c in [
                "mean_return", "mean_length", "mean_ev_travel_time",
                "background_delay_mean", "throughput_mean",
            ] if c in df.columns]
            if display_cols:
                print(df[display_cols].to_string())
            else:
                print(df.to_string())
        except Exception:
            # Fallback: manual table
            print(f"  {'Method':<20s} {'Return':>10s} {'Length':>8s} {'EV Time':>10s}")
            print("  " + "-" * 50)
            for name, m in results.items():
                print(f"  {name:<20s} {m.get('mean_return', 0):>10.1f} "
                      f"{m.get('mean_length', 0):>8.1f} "
                      f"{m.get('mean_ev_travel_time', -1):>10.1f}")

        _ok(f"Evaluated {len(results)} methods x {n_episodes} episodes")

        # Save results
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        serializable = {}
        for method, metrics in results.items():
            serializable[method] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in metrics.items()
                if not isinstance(v, (list, np.ndarray))
            }
        with open(logs_dir / "smoke_test_results.json", "w") as f:
            json.dump(serializable, f, indent=2)

        return results

    except Exception as exc:
        _fail("Evaluation failed", exc)
        return None


# ======================================================================
# Step 9: Generate demo figure
# ======================================================================

def step_generate_figure(cfg: dict) -> bool:
    """Generate a single demo figure showing EV progress over an episode."""
    _banner("STEP 9: Generate Demo Figure")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping figure generation")
        return False

    from src.envs.ev_corridor_env import EVCorridorEnv

    env_cfg = cfg["env"]

    try:
        env = EVCorridorEnv(
            rows=env_cfg.get("rows", 3),
            cols=env_cfg.get("cols", 3),
            max_steps=env_cfg.get("max_episode_steps", 50),
            seed=42,
        )
        obs, info = env.reset()

        steps = []
        ev_progress = []
        rewards_cum = []
        queues = []

        cum_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cum_reward += reward

            steps.append(info["step"])
            route_len = max(info["route_length"] - 1, 1)
            ev_progress.append((info["ev_link_idx"] + info["ev_progress"]) / route_len)
            rewards_cum.append(cum_reward)
            queues.append(info["total_queue"])

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), tight_layout=True)

        axes[0].plot(steps, ev_progress, "b-o", markersize=2)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("EV Route Progress")
        axes[0].set_title("EV Progress")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, rewards_cum, "r-")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Cumulative Reward")
        axes[1].set_title("Reward")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, queues, "g-")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Total Queue Length")
        axes[2].set_title("Network Congestion")
        axes[2].grid(True, alpha=0.3)

        fig_dir = Path("logs/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "smoke_test_demo.png"
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)

        _ok(f"Figure saved to {fig_path}")
        return True

    except Exception as exc:
        _fail("Figure generation failed", exc)
        return False


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="EV-DT end-to-end smoke test")
    parser.add_argument(
        "--config", default="configs/smoke_test.yaml",
        help="Path to smoke test config YAML",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip SB3 baseline training (if not installed)",
    )
    args = parser.parse_args()

    _banner("EV-DT END-TO-END SMOKE TEST")
    print(f"  Config: {args.config}")
    t_start = time.time()

    cfg = _load_config(args.config)

    # Step 1-3: Dataset
    try:
        save_path = step_collect_dataset(cfg)
    except Exception as exc:
        _fail("Dataset collection crashed", exc)
        save_path = cfg["dataset"]["save_path"]

    # Step 4: DT
    dt_path = step_train_dt(cfg, save_path)

    # Step 5: MADT
    madt_path = step_train_madt(cfg, save_path)

    # Step 6: PPO
    if args.skip_baselines:
        print("\n  [SKIP] PPO baseline (--skip-baselines)")
        ppo_ok = False
    else:
        ppo_ok = step_train_ppo(cfg)

    # Step 7-8: Evaluate
    results = step_evaluate(cfg, dt_path, madt_path, ppo_ok)

    # Step 9: Figure
    step_generate_figure(cfg)

    # Step 10: Summary
    elapsed = time.time() - t_start
    _banner("SMOKE TEST SUMMARY")
    print(f"  Total time: {elapsed:.1f}s")

    if _errors:
        print(f"\n  FAILURES ({len(_errors)}):")
        for i, err in enumerate(_errors, 1):
            print(f"    {i}. {err}")
        print(f"\n  RESULT: FAIL")
        return 1
    else:
        print(f"\n  All pipeline stages completed successfully.")
        print(f"  RESULT: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
