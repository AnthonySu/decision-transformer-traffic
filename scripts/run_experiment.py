#!/usr/bin/env python3
"""Full experiment pipeline for EV-DT project.

Runs dataset generation, DT training (10 epochs), MADT training (10 epochs),
return conditioning sweep, and full evaluation. All results saved to logs/.

Usage:
    python scripts/run_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src.baselines.fixed_time_evp import FixedTimeEVP
from src.baselines.greedy_preempt import GreedyPreemptPolicy
from src.envs.ev_corridor_env import EVCorridorEnv
from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.madt import MultiAgentDecisionTransformer
from src.models.trajectory_dataset import TrajectoryDataset
from src.utils.data_collector import DataCollector
from src.utils.metrics import aggregate_metrics, compare_methods

# ---------------------------------------------------------------------------
CONFIG_PATH = "configs/experiment.yaml"
DIVIDER = "=" * 70

def banner(msg: str):
    print(f"\n{DIVIDER}")
    print(f"  {msg}")
    print(DIVIDER)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# STEP 1: Generate Dataset (50 episodes)
# ======================================================================
def step_generate_dataset(cfg: dict) -> str:
    banner("STEP 1: Generate Dataset (50 episodes)")
    env_cfg = cfg["env"]
    ds_cfg = cfg["dataset"]

    rows = env_cfg.get("rows", 3)
    cols = env_cfg.get("cols", 3)
    max_steps = env_cfg.get("max_episode_steps", 50)

    env = EVCorridorEnv(rows=rows, cols=cols, max_steps=max_steps)
    obs, info = env.reset()
    print(f"  Environment: {rows}x{cols} grid, obs_dim={obs.shape[0]}")

    save_path = ds_cfg.get("save_path", "data/experiment.h5")
    total_episodes = ds_cfg.get("num_episodes", 50)
    suboptimal_ratio = ds_cfg.get("suboptimal_ratio", 0.3)

    num_suboptimal = int(total_episodes * suboptimal_ratio * 0.5)
    num_random = int(total_episodes * suboptimal_ratio * 0.5)
    num_expert = total_episodes - num_suboptimal - num_random

    print(f"  Collecting: {num_expert} expert + {num_random} random + "
          f"{num_suboptimal} suboptimal = {total_episodes} episodes")

    expert = GreedyPreemptPolicy(network=env.network, route=env.ev_route)
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

    collector.save_dataset()

    # Print dataset stats
    episodes = collector._episodes
    returns = np.array([ep["episode_return"] for ep in episodes])
    lengths = np.array([ep["episode_length"] for ep in episodes])
    print(f"  Episode return: {returns.mean():.2f} +/- {returns.std():.2f}")
    print(f"  Episode length: {lengths.mean():.1f} +/- {lengths.std():.1f}")

    file_kb = Path(save_path).stat().st_size / 1024
    print(f"  Saved: {save_path} ({file_kb:.1f} KB)")
    return save_path


# ======================================================================
# STEP 2: Train Decision Transformer (10 epochs)
# ======================================================================
def step_train_dt(cfg: dict, save_path: str) -> Path | None:
    banner("STEP 2: Train Decision Transformer (10 epochs)")
    dt_cfg = cfg["dt"]
    env_cfg = cfg["env"]

    dataset = TrajectoryDataset(
        data_path=save_path,
        context_length=dt_cfg["context_length"],
    )
    print(f"  Dataset: {len(dataset)} segments")

    sample = dataset[0]
    state_dim = sample["states"].shape[-1]

    tmp_env = EVCorridorEnv(
        rows=env_cfg.get("rows", 3),
        cols=env_cfg.get("cols", 3),
        max_steps=env_cfg.get("max_episode_steps", 50),
    )
    tmp_env.reset()
    act_dim = int(tmp_env.action_space.nvec[0])
    print(f"  state_dim={state_dim}, act_dim={act_dim}")

    dataloader = DataLoader(
        dataset, batch_size=dt_cfg["batch_size"], shuffle=True, num_workers=0,
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

    num_epochs = dt_cfg["num_epochs"]  # 10
    all_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in dataloader:
            states = batch["states"]
            actions = batch["actions"]
            rtg = batch["returns_to_go"]
            timesteps = batch["timesteps"]

            action_logits = model(states, actions, rtg, timesteps)
            logits_flat = action_logits.reshape(-1, act_dim)
            targets_flat = actions.reshape(-1).clamp(0, act_dim - 1)

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
        all_losses.append(avg_loss)
        print(f"    Epoch {epoch:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "dt_experiment.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dt_cfg,
        "state_dim": state_dim,
        "act_dim": act_dim,
        "training_losses": all_losses,
    }, model_path)

    print(f"  DT saved to {model_path}")
    print(f"  Final loss: {all_losses[-1]:.4f}")
    return model_path


# ======================================================================
# STEP 3: Train MADT (10 epochs)
# ======================================================================
def step_train_madt(cfg: dict) -> Path | None:
    banner("STEP 3: Train Multi-Agent Decision Transformer (10 epochs)")
    madt_cfg = cfg["madt"]
    env_cfg = cfg["env"]

    rows = env_cfg.get("rows", 3)
    cols = env_cfg.get("cols", 3)
    max_steps = env_cfg.get("max_episode_steps", 50)

    ma_env = EVCorridorMAEnv(
        rows=rows, cols=cols, max_steps=max_steps,
        origin="n0_0", destination=f"n{rows-1}_{cols-1}",
    )
    obs_dict, info_dict = ma_env.reset()
    n_agents = len(ma_env.agents)
    state_dim = next(iter(obs_dict.values())).shape[0]
    act_dim = 4
    print(f"  MA env: {n_agents} agents, state_dim={state_dim}, act_dim={act_dim}")

    adj = torch.eye(n_agents, dtype=torch.float32)
    for i in range(n_agents - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # Collect MA training episodes
    print("  Collecting MA training data (20 episodes)...")
    all_states, all_actions, all_rtg = [], [], []
    n_collect = 20

    for _ in range(n_collect):
        obs_dict, _ = ma_env.reset()
        current_n = len(ma_env.agents)
        if current_n != n_agents:
            continue
        ep_states, ep_actions, ep_rewards = [], [], []
        agents_snapshot = list(ma_env.agents)

        for _step in range(max_steps):
            if not ma_env.agents:
                break
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

        ep_states = np.stack(ep_states)
        ep_actions = np.stack(ep_actions)
        ep_rewards = np.stack(ep_rewards)

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

    print(f"  Collected {len(all_states)} usable MA episodes")

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

    # Build batch
    def _make_batch(episodes_states, episodes_actions, episodes_rtg):
        batch_s, batch_a, batch_r, batch_t, batch_m = [], [], [], [], []
        for s_ep, a_ep, r_ep in zip(episodes_states, episodes_actions, episodes_rtg):
            T_ep = s_ep.shape[0]
            end = min(T_ep, K)
            start = max(end - K, 0)
            actual = end - start
            pad = K - actual

            sw = s_ep[start:end].transpose(1, 0, 2)
            aw = a_ep[start:end].T
            rw = r_ep[start:end].T
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

    num_epochs = madt_cfg["num_epochs"]  # 10
    all_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        logits = model(states_b, actions_b, rtg_b, timesteps_b)
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
        all_losses.append(loss.item())
        print(f"    Epoch {epoch:2d}/{num_epochs} | Loss: {loss.item():.4f}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "madt_experiment.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": madt_cfg,
        "state_dim": state_dim,
        "act_dim": act_dim,
        "n_agents": n_agents,
        "adj_matrix": adj,
        "training_losses": all_losses,
    }, model_path)

    print(f"  MADT saved to {model_path}")
    print(f"  Final loss: {all_losses[-1]:.4f}")
    return model_path


# ======================================================================
# STEP 4: Return Conditioning Sweep
# ======================================================================
def step_return_sweep(cfg: dict, dt_path: Path) -> dict:
    banner("STEP 4: Return Conditioning Sweep")
    env_cfg = cfg["env"]
    rows = env_cfg.get("rows", 3)
    cols = env_cfg.get("cols", 3)
    max_steps = env_cfg.get("max_episode_steps", 50)

    target_returns = [0, -25, -50, -100, -200]
    n_episodes = 10

    print(f"  Target returns: {target_returns}")
    print(f"  Episodes per target: {n_episodes}")

    ckpt = torch.load(dt_path, map_location="cpu", weights_only=False)
    dt_cfg = ckpt["config"]
    state_dim = ckpt["state_dim"]
    act_dim = ckpt["act_dim"]

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=dt_cfg["embed_dim"],
        n_layers=dt_cfg["n_layers"],
        n_heads=dt_cfg["n_heads"],
        max_length=dt_cfg["context_length"],
        max_ep_len=dt_cfg.get("max_ep_len", 50),
        dropout=dt_cfg.get("dropout", 0.1),
        activation=dt_cfg.get("activation", "gelu"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = EVCorridorEnv(rows=rows, cols=cols, max_steps=max_steps)
    ctx = dt_cfg["context_length"]

    sweep_results = {}

    for target_return in target_returns:
        print(f"\n  Target RTG = {target_return}")
        episode_returns = []
        episode_lengths = []
        episode_ev_times = []
        all_step_infos = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_return = 0.0
            t = 0
            step_infos = []

            states_buf = torch.zeros(1, ctx, state_dim)
            actions_buf = torch.zeros(1, ctx, dtype=torch.long)
            rtg_buf = torch.zeros(1, ctx, 1)
            ts_buf = torch.zeros(1, ctx, dtype=torch.long)

            states_buf[0, 0] = torch.tensor(obs[:state_dim], dtype=torch.float32)
            rtg_buf[0, 0, 0] = target_return
            ts_buf[0, 0] = 0

            while not done:
                c = min(t + 1, ctx)
                action = model.get_action(
                    states_buf[:, :c], actions_buf[:, :c],
                    rtg_buf[:, :c], ts_buf[:, :c],
                )
                act_array = np.full(env.action_space.shape, action, dtype=int)
                obs, reward, terminated, truncated, info = env.step(act_array)
                done = terminated or truncated
                ep_return += reward
                step_infos.append(info)
                t += 1
                if t < ctx:
                    states_buf[0, t] = torch.tensor(obs[:state_dim], dtype=torch.float32)
                    actions_buf[0, t - 1] = action
                    rtg_buf[0, t, 0] = rtg_buf[0, t - 1, 0] - reward
                    ts_buf[0, t] = min(t, 49)

            episode_returns.append(ep_return)
            episode_lengths.append(t)
            ev_time = info.get("ev_travel_time", -1)
            episode_ev_times.append(ev_time)
            all_step_infos.append(step_infos)

        # Compute metrics
        returns_arr = np.array(episode_returns)
        lengths_arr = np.array(episode_lengths)
        ev_times_arr = np.array(episode_ev_times)
        valid_ev = ev_times_arr[ev_times_arr >= 0]

        result = {
            "target_return": target_return,
            "actual_return_mean": float(returns_arr.mean()),
            "actual_return_std": float(returns_arr.std()),
            "mean_length": float(lengths_arr.mean()),
            "mean_ev_travel_time": float(valid_ev.mean()) if len(valid_ev) > 0 else -1.0,
            "ev_arrival_rate": float(len(valid_ev) / n_episodes),
            "n_episodes": n_episodes,
        }
        sweep_results[f"target_{target_return}"] = result

        print(f"    Actual return: {returns_arr.mean():.1f} +/- {returns_arr.std():.1f}")
        print(f"    EV arrival rate: {result['ev_arrival_rate']:.0%}")
        if len(valid_ev) > 0:
            print(f"    EV travel time: {valid_ev.mean():.1f}")

    # Save sweep results
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "return_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    print("\n  Sweep results saved to logs/return_sweep_results.json")
    return sweep_results


# ======================================================================
# STEP 5: Full Evaluation (20 episodes each)
# ======================================================================
def step_evaluate(cfg: dict, dt_path: Path | None, madt_path: Path | None) -> dict:
    banner("STEP 5: Full Evaluation (20 episodes each)")
    env_cfg = cfg["env"]
    n_episodes = cfg["eval"].get("num_eval_episodes", 20)
    rows = env_cfg.get("rows", 3)
    cols = env_cfg.get("cols", 3)
    max_steps = env_cfg.get("max_episode_steps", 50)

    env = EVCorridorEnv(rows=rows, cols=cols, max_steps=max_steps)
    results: dict[str, dict] = {}

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

    # 1. Random baseline
    print("  Evaluating Random Policy...")
    random_info = _eval_policy(env, lambda o, i: env.action_space.sample(), n_episodes)
    results["Random"] = aggregate_metrics(random_info)

    # 2. Fixed-Time EVP
    print("  Evaluating Fixed-Time EVP...")
    fixed = FixedTimeEVP()
    fixed_info = _eval_policy(env, lambda o, i: fixed.select_action(o, i), n_episodes)
    results["Fixed-Time EVP"] = aggregate_metrics(fixed_info)

    # 3. Greedy Preemption
    print("  Evaluating Greedy Preemption...")
    # Need to create fresh expert after reset so route is populated
    greedy = GreedyPreemptPolicy(network=env._network, route=env._route_intersections)
    greedy_info = _eval_policy(env, lambda o, i: greedy.select_action(o, i), n_episodes)
    results["Greedy Preempt"] = aggregate_metrics(greedy_info)

    # 4. DT (target_return=0)
    if dt_path and dt_path.exists():
        print("  Evaluating Decision Transformer (target=0)...")
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

        s_dim = ckpt["state_dim"]
        a_dim = ckpt["act_dim"]
        ctx = ckpt["config"]["context_length"]

        dt_episodes = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_return = 0.0
            t = 0
            step_infos = []

            states_buf = torch.zeros(1, ctx, s_dim)
            actions_buf = torch.zeros(1, ctx, dtype=torch.long)
            rtg_buf = torch.zeros(1, ctx, 1)
            ts_buf = torch.zeros(1, ctx, dtype=torch.long)

            states_buf[0, 0] = torch.tensor(obs[:s_dim], dtype=torch.float32)
            rtg_buf[0, 0, 0] = 0.0  # target best return
            ts_buf[0, 0] = 0

            while not done:
                c = min(t + 1, ctx)
                action = dt_model.get_action(
                    states_buf[:, :c], actions_buf[:, :c],
                    rtg_buf[:, :c], ts_buf[:, :c],
                )
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

        # Also evaluate with target=-50
        print("  Evaluating Decision Transformer (target=-50)...")
        dt_episodes_50 = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_return = 0.0
            t = 0
            step_infos = []

            states_buf = torch.zeros(1, ctx, s_dim)
            actions_buf = torch.zeros(1, ctx, dtype=torch.long)
            rtg_buf = torch.zeros(1, ctx, 1)
            ts_buf = torch.zeros(1, ctx, dtype=torch.long)

            states_buf[0, 0] = torch.tensor(obs[:s_dim], dtype=torch.float32)
            rtg_buf[0, 0, 0] = -50.0
            ts_buf[0, 0] = 0

            while not done:
                c = min(t + 1, ctx)
                action = dt_model.get_action(
                    states_buf[:, :c], actions_buf[:, :c],
                    rtg_buf[:, :c], ts_buf[:, :c],
                )
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

            dt_episodes_50.append({
                "return": ep_return, "length": t, "step_infos": step_infos
            })
        results["DT (target=-50)"] = aggregate_metrics(dt_episodes_50)

    # 5. MADT evaluation
    if madt_path and madt_path.exists():
        print("  Evaluating MADT...")
        ckpt = torch.load(madt_path, map_location="cpu", weights_only=False)
        madt_model = MultiAgentDecisionTransformer(
            state_dim=ckpt["state_dim"],
            act_dim=ckpt["act_dim"],
            n_agents=ckpt["n_agents"],
            adj_matrix=ckpt["adj_matrix"],
            hidden_dim=ckpt["config"]["embed_dim"],
            n_layers=ckpt["config"]["n_layers"],
            n_heads=ckpt["config"]["n_heads"],
            gat_heads=ckpt["config"]["gat_heads"],
            gat_layers=ckpt["config"]["gat_layers"],
            max_length=ckpt["config"]["context_length"],
            max_ep_len=ckpt["config"].get("max_ep_len", 50),
            dropout=ckpt["config"].get("dropout", 0.1),
        )
        madt_model.load_state_dict(ckpt["model_state_dict"])
        madt_model.eval()

        n_agents = ckpt["n_agents"]
        ma_env = EVCorridorMAEnv(
            rows=rows, cols=cols, max_steps=max_steps,
            origin="n0_0", destination=f"n{rows-1}_{cols-1}",
        )

        madt_episodes = []
        for ep in range(n_episodes):
            obs_dict = ma_env.reset()
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]
            if not ma_env.agents or len(ma_env.agents) != n_agents:
                continue

            done = {agent: False for agent in ma_env.agents}
            episode_return = 0.0
            t = 0
            step_infos = []
            agents_snapshot = list(ma_env.agents)

            m_ctx = ckpt["config"]["context_length"]
            s_dim = ckpt["state_dim"]

            states_buf = torch.zeros(1, n_agents, m_ctx, s_dim)
            actions_buf = torch.zeros(1, n_agents, m_ctx, dtype=torch.long)
            rtg_buf = torch.zeros(1, n_agents, m_ctx, 1)
            ts_buf = torch.zeros(1, n_agents, m_ctx, dtype=torch.long)

            for i, agent_id in enumerate(agents_snapshot):
                if agent_id in obs_dict:
                    states_buf[0, i, 0] = torch.tensor(
                        obs_dict[agent_id], dtype=torch.float32
                    )
                rtg_buf[0, i, 0, 0] = 0.0 / n_agents  # target=0

            while not all(done.values()) and t < max_steps:
                ctx_len = min(t + 1, m_ctx)
                action_dict = {}

                for i, agent_id in enumerate(agents_snapshot):
                    if not done.get(agent_id, True):
                        action = madt_model.get_action(
                            states_buf[:, :, :ctx_len],
                            actions_buf[:, :, :ctx_len],
                            rtg_buf[:, :, :ctx_len],
                            ts_buf[:, :, :ctx_len],
                            agent_idx=i,
                        )
                        action_dict[agent_id] = action

                if not action_dict:
                    break

                result = ma_env.step(action_dict)
                obs_dict, reward_dict, term_dict, trunc_dict, info_dict = result

                step_reward = sum(reward_dict.values())
                episode_return += step_reward
                step_infos.append(info_dict)

                for agent_id, terminated in term_dict.items():
                    if terminated or trunc_dict.get(agent_id, False):
                        done[agent_id] = True

                t += 1
                if t < m_ctx:
                    for i, agent_id in enumerate(agents_snapshot):
                        if agent_id in obs_dict:
                            states_buf[0, i, t] = torch.tensor(
                                obs_dict[agent_id], dtype=torch.float32
                            )
                            if agent_id in action_dict:
                                actions_buf[0, i, t - 1] = action_dict[agent_id]
                            agent_reward = reward_dict.get(agent_id, 0.0)
                            rtg_buf[0, i, t, 0] = rtg_buf[0, i, t - 1, 0] - agent_reward
                            ts_buf[0, i, t] = t

            madt_episodes.append({
                "return": episode_return, "length": t, "step_infos": step_infos
            })

        if madt_episodes:
            results["MADT (target=0)"] = aggregate_metrics(madt_episodes)

    # Save and print results
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    serializable = {}
    for method, metrics in results.items():
        serializable[method] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics.items()
            if not isinstance(v, (list, np.ndarray))
        }
    with open(logs_dir / "evaluation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print("  METHOD COMPARISON TABLE")
    print(f"{'='*70}")

    try:
        df = compare_methods(results)
        display_cols = [c for c in [
            "mean_return", "mean_length", "mean_ev_travel_time",
            "background_delay_mean", "throughput_mean",
        ] if c in df.columns]
        if display_cols:
            print(df[display_cols].to_string())
        else:
            print(df.to_string())
    except Exception:
        print(f"  {'Method':<20s} {'Return':>10s} {'Length':>8s} {'EV Time':>10s}")
        print("  " + "-" * 50)
        for name, m in results.items():
            print(f"  {name:<20s} {m.get('mean_return', 0):>10.1f} "
                  f"{m.get('mean_length', 0):>8.1f} "
                  f"{m.get('mean_ev_travel_time', -1):>10.1f}")

    print("\n  Results saved to logs/evaluation_results.json")
    return results


# ======================================================================
# MAIN
# ======================================================================
def main():
    banner("EV-DT FULL EXPERIMENT PIPELINE")
    t_total = time.time()
    cfg = load_config()

    # Step 1: Dataset
    save_path = step_generate_dataset(cfg)

    # Step 2: Train DT
    dt_path = step_train_dt(cfg, save_path)

    # Step 3: Train MADT
    madt_path = step_train_madt(cfg)

    # Step 4: Return conditioning sweep (DT only)
    sweep_results = {}
    if dt_path:
        sweep_results = step_return_sweep(cfg, dt_path)

    # Step 5: Full evaluation
    eval_results = step_evaluate(cfg, dt_path, madt_path)

    # Final summary
    elapsed = time.time() - t_total
    banner("EXPERIMENT COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Dataset: {save_path}")
    print(f"  DT model: {dt_path}")
    print(f"  MADT model: {madt_path}")
    print("  Logs: logs/evaluation_results.json, logs/return_sweep_results.json")

    # Print final summary table
    if sweep_results:
        print("\n  RETURN CONDITIONING SWEEP:")
        print(f"  {'Target RTG':<12} {'Actual Return':<20} {'EV Arrival':>12} {'EV Time':>10}")
        print(f"  {'-'*56}")
        for key, m in sweep_results.items():
            tr = m["target_return"]
            ar = m["actual_return_mean"]
            astd = m["actual_return_std"]
            ea = m["ev_arrival_rate"]
            et = m["mean_ev_travel_time"]
            print(f"  {tr:<12.0f} {ar:<8.1f}+/-{astd:<9.1f} {ea:>10.0%}   {et:>8.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
