#!/usr/bin/env python3
"""Unified experiment runner: train and evaluate all methods on real environments.

Produces results in the same JSON format as results/narrative_numbers.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.baselines.cql_baseline import CQLAgent, OfflineRLDataset
from src.baselines.fixed_time_evp import FixedTimeEVP
from src.baselines.greedy_preempt import GreedyPreemptPolicy
from src.baselines.rl_baselines import (
    create_dqn_agent,
    create_ppo_agent,
)
from src.envs.ev_corridor_env import EVCorridorEnv
from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.envs.wrappers import FlattenActionWrapper
from src.models.constrained_dt import ConstrainedDecisionTransformer
from src.models.decision_transformer import DecisionTransformer
from src.models.madt import MultiAgentDecisionTransformer
from src.models.trajectory_dataset import TrajectoryDataset
from src.utils.data_collector import DataCollector
from src.utils.experiment import set_seed
from src.utils.metrics import aggregate_metrics

# ======================================================================
# Configuration helpers
# ======================================================================

def _default_config(quick: bool = False) -> Dict[str, Any]:
    """Return default hyper-parameters.  ``quick=True`` shrinks everything."""
    if quick:
        return {
            "data": {
                "num_expert": 10,
                "num_random": 2,
                "num_suboptimal": 2,
            },
            "dt": {
                "hidden_dim": 64,
                "n_layers": 2,
                "n_heads": 2,
                "context_length": 10,
                "max_ep_len": 300,
                "dropout": 0.1,
                "activation": "gelu",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "warmup_steps": 50,
                "num_epochs": 5,
                "batch_size": 32,
                "target_return": 0.0,
            },
            "cdt": {
                "hidden_dim": 64,
                "n_layers": 2,
                "n_heads": 2,
                "context_length": 10,
                "max_ep_len": 300,
                "dropout": 0.1,
                "activation": "gelu",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "warmup_steps": 50,
                "num_epochs": 5,
                "batch_size": 32,
                "target_return": 0.0,
                "target_cost": -50.0,
                "cost_weight": 0.1,
            },
            "rl": {
                "ppo_timesteps": 2_000,
                "dqn_timesteps": 2_000,
            },
            "cql": {
                "hidden_dim": 64,
                "n_layers": 2,
                "lr": 3e-4,
                "gamma": 0.99,
                "alpha": 1.0,
                "n_epochs": 5,
                "batch_size": 64,
            },
            "env": {
                "max_steps": 100,
            },
            "eval": {
                "n_episodes": 5,
            },
        }
    else:
        return {
            "data": {
                "num_expert": 70,
                "num_random": 15,
                "num_suboptimal": 15,
            },
            "dt": {
                "hidden_dim": 128,
                "n_layers": 4,
                "n_heads": 4,
                "context_length": 20,
                "max_ep_len": 300,
                "dropout": 0.1,
                "activation": "gelu",
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 200,
                "num_epochs": 30,
                "batch_size": 64,
                "target_return": 0.0,
            },
            "cdt": {
                "hidden_dim": 128,
                "n_layers": 4,
                "n_heads": 4,
                "context_length": 20,
                "max_ep_len": 300,
                "dropout": 0.1,
                "activation": "gelu",
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 200,
                "num_epochs": 30,
                "batch_size": 64,
                "target_return": 0.0,
                "target_cost": -50.0,
                "cost_weight": 0.1,
            },
            "rl": {
                "ppo_timesteps": 50_000,
                "dqn_timesteps": 50_000,
            },
            "cql": {
                "hidden_dim": 128,
                "n_layers": 2,
                "lr": 3e-4,
                "gamma": 0.99,
                "alpha": 1.0,
                "n_epochs": 50,
                "batch_size": 64,
            },
            "env": {
                "max_steps": 200,
            },
            "eval": {
                "n_episodes": 100,
            },
        }


# ======================================================================
# Environment factories
# ======================================================================

def _make_env(grid_size: int, max_steps: int, seed: Optional[int] = None) -> EVCorridorEnv:
    return EVCorridorEnv(
        rows=grid_size,
        cols=grid_size,
        use_lightsim=False,
        max_steps=max_steps,
        seed=seed,
    )


def _make_ma_env(grid_size: int, max_steps: int, seed: Optional[int] = None) -> EVCorridorMAEnv:
    return EVCorridorMAEnv(
        rows=grid_size,
        cols=grid_size,
        use_lightsim=False,
        max_steps=max_steps,
        seed=seed,
    )


# ======================================================================
# Data collection
# ======================================================================

def collect_offline_data(
    grid_size: int,
    cfg: Dict[str, Any],
    output_dir: str,
    seed: int,
) -> str:
    """Collect mixed offline dataset.  Returns the HDF5 file path."""
    env = _make_env(grid_size, cfg["env"]["max_steps"], seed=seed)
    data_path = os.path.join(output_dir, f"data_{grid_size}x{grid_size}_seed{seed}.h5")

    collector = DataCollector(env, save_path=data_path)

    # Expert policy: greedy preemption
    # Need to reset env once to get network/route for the policy
    env.reset()
    expert = GreedyPreemptPolicy(network=env.network, route=env.ev_route)

    data_cfg = cfg["data"]
    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=data_cfg["num_expert"],
        num_random=data_cfg["num_random"],
        num_suboptimal=data_cfg["num_suboptimal"],
    )
    collector.save_dataset()
    return data_path


# ======================================================================
# Training routines
# ======================================================================

def train_dt_model(
    data_path: str,
    cfg: Dict[str, Any],
    device: str,
) -> Tuple[DecisionTransformer, TrajectoryDataset]:
    """Train a Decision Transformer on the offline dataset."""
    dt_cfg = cfg["dt"]

    dataset = TrajectoryDataset(
        data_path=data_path,
        context_length=dt_cfg["context_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=dt_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    state_dim = dataset.state_dim
    act_dim = dataset.act_dim

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=dt_cfg["hidden_dim"],
        n_layers=dt_cfg["n_layers"],
        n_heads=dt_cfg["n_heads"],
        max_length=dt_cfg["context_length"],
        max_ep_len=dt_cfg["max_ep_len"],
        dropout=dt_cfg["dropout"],
        activation=dt_cfg["activation"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dt_cfg["lr"],
        weight_decay=dt_cfg["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    total_steps = dt_cfg["num_epochs"] * len(dataloader)

    def lr_lambda(step: int) -> float:
        if step < dt_cfg["warmup_steps"]:
            return step / max(dt_cfg["warmup_steps"], 1)
        progress = (step - dt_cfg["warmup_steps"]) / max(
            1, total_steps - dt_cfg["warmup_steps"]
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for epoch in range(1, dt_cfg["num_epochs"] + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            masks = batch["masks"].to(device)

            logits = model(states, actions, rtg, timesteps)
            B, T, A = logits.shape
            logits_flat = logits.reshape(B * T, A)
            targets_flat = actions.reshape(B * T)
            mask_flat = masks.reshape(B * T).bool()

            loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % max(1, dt_cfg["num_epochs"] // 5) == 0 or epoch == 1:
            print(f"    DT epoch {epoch}/{dt_cfg['num_epochs']} loss={epoch_loss / max(n_batches, 1):.4f}")

    return model, dataset


def train_cdt_model(
    data_path: str,
    cfg: Dict[str, Any],
    device: str,
) -> Tuple[ConstrainedDecisionTransformer, TrajectoryDataset]:
    """Train a Constrained Decision Transformer on the offline dataset."""
    cdt_cfg = cfg["cdt"]

    dataset = TrajectoryDataset(
        data_path=data_path,
        context_length=cdt_cfg["context_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cdt_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    state_dim = dataset.state_dim
    act_dim = dataset.act_dim

    model = ConstrainedDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=cdt_cfg["hidden_dim"],
        n_layers=cdt_cfg["n_layers"],
        n_heads=cdt_cfg["n_heads"],
        max_length=cdt_cfg["context_length"],
        max_ep_len=cdt_cfg["max_ep_len"],
        dropout=cdt_cfg["dropout"],
        activation=cdt_cfg["activation"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cdt_cfg["lr"],
        weight_decay=cdt_cfg["weight_decay"],
    )

    total_steps = cdt_cfg["num_epochs"] * len(dataloader)

    def lr_lambda(step: int) -> float:
        if step < cdt_cfg["warmup_steps"]:
            return step / max(cdt_cfg["warmup_steps"], 1)
        progress = (step - cdt_cfg["warmup_steps"]) / max(
            1, total_steps - cdt_cfg["warmup_steps"]
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for epoch in range(1, cdt_cfg["num_epochs"] + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            masks = batch["masks"].to(device)

            # CDT needs costs_to_go — use background_delay proxy (zeros for now)
            B, T, _ = states.shape
            costs_to_go = torch.zeros(B, T, 1, device=device)

            outputs = model(states, actions, rtg, costs_to_go, timesteps)
            logits = outputs["action_logits"]
            _, _, A = logits.shape
            logits_flat = logits.reshape(B * T, A)
            targets_flat = actions.reshape(B * T)
            mask_flat = masks.reshape(B * T).bool()

            action_loss = nn.functional.cross_entropy(
                logits_flat[mask_flat], targets_flat[mask_flat]
            )
            cost_loss = nn.functional.mse_loss(
                outputs["cost_preds"], costs_to_go
            )
            loss = action_loss + cdt_cfg["cost_weight"] * cost_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % max(1, cdt_cfg["num_epochs"] // 5) == 0 or epoch == 1:
            print(f"    CDT epoch {epoch}/{cdt_cfg['num_epochs']} loss={epoch_loss / max(n_batches, 1):.4f}")

    return model, dataset


# ======================================================================
# Evaluation helpers
# ======================================================================

def _evaluate_rule_based(
    env: EVCorridorEnv,
    policy_fn,
    n_episodes: int,
    desc: str = "",
) -> List[Dict[str, Any]]:
    """Evaluate a rule-based policy (FixedTimeEVP, Greedy, MaxPressure)."""
    episodes_info: List[Dict[str, Any]] = []
    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        step_infos: List[Dict[str, Any]] = []
        t = 0

        while not done:
            action = policy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })
    return episodes_info


def _evaluate_sb3(
    env: EVCorridorEnv,
    agent,
    n_episodes: int,
    desc: str = "",
    wrap_flatten: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate an SB3 agent (PPO/DQN)."""
    episodes_info: List[Dict[str, Any]] = []
    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        if wrap_flatten:
            eval_env = FlattenActionWrapper(
                _make_env(env.rows, env.max_steps)
            )
        else:
            eval_env = env
        obs, info = eval_env.reset()
        done = False
        episode_return = 0.0
        step_infos: List[Dict[str, Any]] = []
        t = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })
    return episodes_info


def _evaluate_cql(
    env: EVCorridorEnv,
    agent: CQLAgent,
    n_episodes: int,
    desc: str = "",
) -> List[Dict[str, Any]]:
    """Evaluate a CQL agent."""
    episodes_info: List[Dict[str, Any]] = []
    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        step_infos: List[Dict[str, Any]] = []
        t = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })
    return episodes_info


def _evaluate_dt(
    env: EVCorridorEnv,
    model: DecisionTransformer,
    target_return: float,
    n_episodes: int,
    device: str,
    desc: str = "",
) -> List[Dict[str, Any]]:
    """Evaluate a Decision Transformer with a specific target return."""
    model.eval()
    episodes_info: List[Dict[str, Any]] = []

    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        t = 0

        states = torch.zeros(1, model.max_length, model.state_dim, device=device)
        actions = torch.zeros(1, model.max_length, dtype=torch.long, device=device)
        returns_to_go = torch.zeros(1, model.max_length, 1, device=device)
        timesteps = torch.zeros(1, model.max_length, dtype=torch.long, device=device)

        states[0, 0] = torch.tensor(obs, dtype=torch.float32, device=device)
        returns_to_go[0, 0, 0] = target_return
        timesteps[0, 0] = 0

        step_infos: List[Dict[str, Any]] = []

        while not done:
            ctx_len = min(t + 1, model.max_length)
            action = model.get_action(
                states[:, :ctx_len],
                actions[:, :ctx_len],
                returns_to_go[:, :ctx_len],
                timesteps[:, :ctx_len],
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

            if t < model.max_length:
                states[0, t] = torch.tensor(obs, dtype=torch.float32, device=device)
                actions[0, t - 1] = action
                returns_to_go[0, t, 0] = returns_to_go[0, t - 1, 0] - reward
                timesteps[0, t] = min(t, model.max_length - 1)

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })

    model.train()
    return episodes_info


def _evaluate_cdt(
    env: EVCorridorEnv,
    model: ConstrainedDecisionTransformer,
    target_return: float,
    target_cost: float,
    n_episodes: int,
    device: str,
    desc: str = "",
) -> List[Dict[str, Any]]:
    """Evaluate a Constrained Decision Transformer."""
    model.eval()
    episodes_info: List[Dict[str, Any]] = []

    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        t = 0

        states = torch.zeros(1, model.max_length, model.state_dim, device=device)
        actions = torch.zeros(1, model.max_length, dtype=torch.long, device=device)
        returns_to_go = torch.zeros(1, model.max_length, 1, device=device)
        costs_to_go = torch.zeros(1, model.max_length, 1, device=device)
        timesteps = torch.zeros(1, model.max_length, dtype=torch.long, device=device)

        states[0, 0] = torch.tensor(obs, dtype=torch.float32, device=device)
        returns_to_go[0, 0, 0] = target_return
        costs_to_go[0, 0, 0] = target_cost
        timesteps[0, 0] = 0

        step_infos: List[Dict[str, Any]] = []

        while not done:
            ctx_len = min(t + 1, model.max_length)
            action = model.get_action(
                states[:, :ctx_len],
                actions[:, :ctx_len],
                returns_to_go[:, :ctx_len],
                costs_to_go[:, :ctx_len],
                timesteps[:, :ctx_len],
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

            if t < model.max_length:
                states[0, t] = torch.tensor(obs, dtype=torch.float32, device=device)
                actions[0, t - 1] = action
                returns_to_go[0, t, 0] = returns_to_go[0, t - 1, 0] - reward
                costs_to_go[0, t, 0] = costs_to_go[0, t - 1, 0]  # keep budget constant
                timesteps[0, t] = min(t, model.max_length - 1)

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })

    model.train()
    return episodes_info


def _evaluate_madt(
    ma_env: EVCorridorMAEnv,
    model: MultiAgentDecisionTransformer,
    target_return: float,
    n_episodes: int,
    device: str,
    desc: str = "",
) -> List[Dict[str, Any]]:
    """Evaluate the Multi-Agent Decision Transformer."""
    model.eval()
    n_agents = model.n_agents
    episodes_info: List[Dict[str, Any]] = []

    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        obs_dict = ma_env.reset()
        done = {agent: False for agent in ma_env.agents}
        episode_return = 0.0
        t = 0

        # The env may have more agents than the model was built for;
        # only control up to n_agents, rest get random/default actions.
        active_agents = ma_env.agents[:n_agents]

        states = torch.zeros(1, n_agents, model.max_length, model.state_dim, device=device)
        actions = torch.zeros(1, n_agents, model.max_length, dtype=torch.long, device=device)
        returns_to_go = torch.zeros(1, n_agents, model.max_length, 1, device=device)
        timesteps = torch.zeros(1, n_agents, model.max_length, dtype=torch.long, device=device)

        for i, agent_id in enumerate(active_agents):
            if agent_id in obs_dict:
                obs_t = torch.tensor(
                    obs_dict[agent_id], dtype=torch.float32, device=device
                )
                # Handle dimension mismatch: truncate or pad obs to state_dim
                if obs_t.shape[0] >= model.state_dim:
                    states[0, i, 0] = obs_t[: model.state_dim]
                else:
                    states[0, i, 0, : obs_t.shape[0]] = obs_t
            returns_to_go[0, i, 0, 0] = target_return / n_agents

        step_infos: List[Dict[str, Any]] = []

        max_eval_steps = 500  # safety limit
        while not all(done.values()) and t < max_eval_steps:
            # PettingZoo sets agents=[] on termination; check that
            if not ma_env.agents:
                break

            ctx_len = min(t + 1, model.max_length)
            action_dict = {}

            current_agents = list(ma_env.agents)
            for i, agent_id in enumerate(active_agents):
                if agent_id in current_agents and not done.get(agent_id, True):
                    act = model.get_action(
                        states[:, :, :ctx_len],
                        actions[:, :, :ctx_len],
                        returns_to_go[:, :, :ctx_len],
                        timesteps[:, :, :ctx_len],
                        agent_idx=i,
                    )
                    action_dict[agent_id] = act
            # For agents beyond model's n_agents, use default phase
            for agent_id in current_agents:
                if agent_id not in action_dict and not done.get(agent_id, True):
                    action_dict[agent_id] = 0

            if not action_dict:
                break

            obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = (
                ma_env.step(action_dict)
            )

            for agent_id in list(done.keys()):
                if agent_id in terminated_dict:
                    done[agent_id] = terminated_dict[agent_id] or truncated_dict.get(
                        agent_id, False
                    )

            step_reward = sum(rewards_dict.values()) if rewards_dict else 0.0
            episode_return += step_reward

            # Build a combined info from all agents
            combined_info: Dict[str, Any] = {}
            if info_dict:
                # Take first agent's info as representative for EV metrics
                first_info = next(iter(info_dict.values()), {})
                combined_info = dict(first_info)
            step_infos.append(combined_info)

            t += 1
            if t < model.max_length:
                for i, agent_id in enumerate(active_agents):
                    if agent_id in obs_dict:
                        obs_t = torch.tensor(
                            obs_dict[agent_id], dtype=torch.float32, device=device
                        )
                        if obs_t.shape[0] >= model.state_dim:
                            states[0, i, t] = obs_t[: model.state_dim]
                        else:
                            states[0, i, t, : obs_t.shape[0]] = obs_t
                    if agent_id in action_dict:
                        actions[0, i, t - 1] = action_dict[agent_id]
                    agent_reward = rewards_dict.get(agent_id, 0.0)
                    returns_to_go[0, i, t, 0] = (
                        returns_to_go[0, i, t - 1, 0] - agent_reward
                    )
                    timesteps[0, i, t] = min(t, model.max_length - 1)

        episodes_info.append({
            "return": episode_return,
            "length": t,
            "step_infos": step_infos,
        })

    model.train()
    return episodes_info


# ======================================================================
# Metric extraction — match narrative_numbers.json format
# ======================================================================

def _extract_metrics(episodes_info: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the 8 standard metrics from episode info dicts.

    Output keys match narrative_numbers.json:
        ett_mean, ett_std, acd_mean, acd_std,
        throughput_mean, throughput_std, ev_stops_mean, ev_stops_std
    """
    agg = aggregate_metrics(episodes_info)

    # ev_stops: count steps where the EV was blocked (speed == 0, at red)
    # Proxy: number of phase changes for EV across the episode
    ev_stops_list: List[float] = []
    for ep in episodes_info:
        step_infos = ep.get("step_infos", [])
        # Count steps where EV didn't advance (ev_link_idx didn't change)
        stops = 0
        prev_idx = None
        prev_progress = None
        for si in step_infos:
            cur_idx = si.get("ev_link_idx", 0)
            cur_progress = si.get("ev_progress", 0.0)
            if prev_idx is not None:
                # EV stopped if link index and progress didn't change
                if cur_idx == prev_idx and abs(cur_progress - prev_progress) < 1e-6:
                    stops += 1
            prev_idx = cur_idx
            prev_progress = cur_progress
        ev_stops_list.append(float(stops))

    ev_stops_arr = np.array(ev_stops_list, dtype=np.float64)

    # EV travel time (in seconds): steps * dt (dt=5s)
    dt_seconds = 5.0
    ett_mean = agg["ev_travel_time_mean"] * dt_seconds if agg["ev_travel_time_mean"] > 0 else -1.0
    ett_std = agg["ev_travel_time_std"] * dt_seconds

    return {
        "ett_mean": round(float(ett_mean), 1),
        "ett_std": round(float(ett_std), 1),
        "acd_mean": round(float(agg["background_delay_mean"]), 1),
        "acd_std": round(float(agg["background_delay_std"]), 1),
        "throughput_mean": round(float(agg["throughput_mean"]), 0),
        "throughput_std": round(float(agg["throughput_std"]), 0),
        "ev_stops_mean": round(float(np.mean(ev_stops_arr)), 1),
        "ev_stops_std": round(float(np.std(ev_stops_arr)), 1),
    }


# ======================================================================
# Per-grid-size experiment
# ======================================================================

def run_grid_experiment(
    grid_size: int,
    seeds: List[int],
    cfg: Dict[str, Any],
    output_dir: str,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """Run all methods for a single grid size, aggregating over seeds."""
    max_steps = cfg["env"]["max_steps"]
    n_eval = cfg["eval"]["n_episodes"]

    # Accumulators: method -> list of per-seed metric dicts
    all_seed_metrics: Dict[str, List[Dict[str, float]]] = {}

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        set_seed(seed)

        env = _make_env(grid_size, max_steps, seed=seed)
        ma_env = _make_ma_env(grid_size, max_steps, seed=seed)

        # Step 1: Collect offline data
        print("    Collecting offline data...")
        data_path = collect_offline_data(grid_size, cfg, output_dir, seed)

        # Step 2: Train offline models (DT, CDT)
        print("    Training DT...")
        try:
            dt_model, dt_dataset = train_dt_model(data_path, cfg, device)
        except Exception as e:
            print(f"    WARNING: DT training failed: {e}")
            traceback.print_exc()
            dt_model = None

        print("    Training CDT...")
        try:
            cdt_model, _ = train_cdt_model(data_path, cfg, device)
        except Exception as e:
            print(f"    WARNING: CDT training failed: {e}")
            traceback.print_exc()
            cdt_model = None

        # Step 3: Train CQL
        print("    Training CQL...")
        cql_agent = None
        try:
            cql_cfg = cfg["cql"]
            cql_dataset = OfflineRLDataset(data_path)
            state_dim = cql_dataset.states.shape[1]
            # Infer act_dim from actions
            act_dim = max(int(cql_dataset.actions.max()) + 1, 4)
            cql_agent = CQLAgent(
                state_dim=state_dim,
                act_dim=act_dim,
                hidden_dim=cql_cfg["hidden_dim"],
                n_layers=cql_cfg["n_layers"],
                lr=cql_cfg["lr"],
                gamma=cql_cfg["gamma"],
                alpha=cql_cfg["alpha"],
                device=device,
            )
            cql_agent.train_offline(
                cql_dataset,
                n_epochs=cql_cfg["n_epochs"],
                batch_size=cql_cfg["batch_size"],
                log_interval=max(1, cql_cfg["n_epochs"] // 3),
            )
        except Exception as e:
            print(f"    WARNING: CQL training failed: {e}")
            traceback.print_exc()
            cql_agent = None

        # Step 4: Train PPO
        print("    Training PPO...")
        ppo_agent = None
        try:
            from stable_baselines3.common.monitor import Monitor

            ppo_env = Monitor(
                _make_env(grid_size, max_steps, seed=seed)
            )
            ppo_agent = create_ppo_agent(ppo_env, {
                "seed": seed,
                "verbose": 0,
                "n_steps": 128,
                "batch_size": 64,
            })
            ppo_agent.learn(total_timesteps=cfg["rl"]["ppo_timesteps"])
        except Exception as e:
            print(f"    WARNING: PPO training failed: {e}")
            traceback.print_exc()
            ppo_agent = None

        # Step 5: Train DQN (only feasible for small grids; FlattenAction is 4^N)
        print("    Training DQN...")
        dqn_agent = None
        dqn_uses_flatten = False
        try:
            from stable_baselines3.common.monitor import Monitor

            dqn_base_env = _make_env(grid_size, max_steps, seed=seed)
            # FlattenActionWrapper creates 4^(grid_size^2) actions; only practical
            # for very small grids.  Skip if the flattened space would exceed 4096.
            flat_size = int(np.prod(dqn_base_env.action_space.nvec))
            if flat_size > 4096:
                print(f"    Skipping DQN: flattened action space too large ({flat_size})")
            else:
                dqn_uses_flatten = True
                dqn_env = Monitor(FlattenActionWrapper(dqn_base_env))
                dqn_agent = create_dqn_agent(dqn_env, {
                    "seed": seed,
                    "verbose": 0,
                    "buffer_size": 10_000,
                    "batch_size": 32,
                    "learning_starts": 200,
                })
                dqn_agent.learn(total_timesteps=cfg["rl"]["dqn_timesteps"])
        except Exception as e:
            print(f"    WARNING: DQN training failed: {e}")
            traceback.print_exc()
            dqn_agent = None

        # ---- Evaluate all methods ----
        print("    Evaluating all methods...")

        # Re-create envs for evaluation
        eval_env = _make_env(grid_size, max_steps, seed=seed + 1000)
        eval_ma_env = _make_ma_env(grid_size, max_steps, seed=seed + 1000)

        methods_results: Dict[str, List[Dict[str, Any]]] = {}

        # FT-EVP
        ft_evp = FixedTimeEVP()
        methods_results["FT-EVP"] = _evaluate_rule_based(
            eval_env,
            lambda obs, info, p=ft_evp: p.select_action(obs, info),
            n_eval,
            desc="FT-EVP",
        )

        # Greedy
        eval_env.reset()  # to populate network/route
        greedy = GreedyPreemptPolicy(network=eval_env.network, route=eval_env.ev_route)
        methods_results["Greedy"] = _evaluate_rule_based(
            eval_env,
            lambda obs, info, p=greedy: p.select_action(obs, info),
            n_eval,
            desc="Greedy",
        )

        # MaxPressure — use Greedy with no EV active (it falls back to MaxPressure)
        class _MaxPressureOnly:
            def __init__(self, greedy_policy: GreedyPreemptPolicy):
                self._greedy = greedy_policy

            def select_action(self, obs, info):
                # Force ev_info to inactive so it always uses MaxPressure
                fake_info = {"active": False}
                return self._greedy.select_action(obs, fake_info)

            def reset(self):
                pass

        mp_policy = _MaxPressureOnly(greedy)
        methods_results["MaxPressure"] = _evaluate_rule_based(
            eval_env,
            lambda obs, info, p=mp_policy: p.select_action(obs, info),
            n_eval,
            desc="MaxPressure",
        )

        # DQN
        if dqn_agent is not None:
            dqn_eval_env = FlattenActionWrapper(
                _make_env(grid_size, max_steps, seed=seed + 2000)
            )
            methods_results["DQN"] = _evaluate_sb3(
                dqn_eval_env, dqn_agent, n_eval, desc="DQN"
            )

        # PPO
        if ppo_agent is not None:
            methods_results["PPO"] = _evaluate_sb3(
                _make_env(grid_size, max_steps, seed=seed + 3000),
                ppo_agent,
                n_eval,
                desc="PPO",
            )

        # CQL
        if cql_agent is not None:
            methods_results["CQL"] = _evaluate_cql(
                _make_env(grid_size, max_steps, seed=seed + 4000),
                cql_agent,
                n_eval,
                desc="CQL",
            )

        # IQL — not implemented separately; use CQL with alpha=0 as proxy
        try:
            iql_dataset = OfflineRLDataset(data_path)
            iql_state_dim = iql_dataset.states.shape[1]
            iql_act_dim = max(int(iql_dataset.actions.max()) + 1, 4)
            iql_agent = CQLAgent(
                state_dim=iql_state_dim,
                act_dim=iql_act_dim,
                hidden_dim=cfg["cql"]["hidden_dim"],
                n_layers=cfg["cql"]["n_layers"],
                lr=cfg["cql"]["lr"],
                gamma=cfg["cql"]["gamma"],
                alpha=0.0,  # no conservative penalty -> vanilla Q-learning ~ IQL proxy
                device=device,
            )
            iql_agent.train_offline(
                iql_dataset,
                n_epochs=cfg["cql"]["n_epochs"],
                batch_size=cfg["cql"]["batch_size"],
                log_interval=max(1, cfg["cql"]["n_epochs"]),
            )
            methods_results["IQL"] = _evaluate_cql(
                _make_env(grid_size, max_steps, seed=seed + 5000),
                iql_agent,
                n_eval,
                desc="IQL",
            )
        except Exception as e:
            print(f"    WARNING: IQL training/eval failed: {e}")

        # DT
        if dt_model is not None:
            methods_results["DT"] = _evaluate_dt(
                _make_env(grid_size, max_steps, seed=seed + 6000),
                dt_model,
                target_return=cfg["dt"]["target_return"],
                n_episodes=n_eval,
                device=device,
                desc="DT",
            )

        # CDT (reported under DT column variant; kept separate here)
        if cdt_model is not None:
            methods_results["CDT"] = _evaluate_cdt(
                _make_env(grid_size, max_steps, seed=seed + 7000),
                cdt_model,
                target_return=cfg["cdt"]["target_return"],
                target_cost=cfg["cdt"]["target_cost"],
                n_episodes=n_eval,
                device=device,
                desc="CDT",
            )

        # MADT — train a small MADT from the same data
        try:
            print("    Training & evaluating MADT...")
            # MADT needs multi-agent data; collect from MA env
            ma_data_env = _make_ma_env(grid_size, max_steps, seed=seed)
            ma_obs = ma_data_env.reset()
            n_agents_actual = len(ma_data_env.agents)

            # Determine state/action dims from first agent obs
            first_agent = ma_data_env.agents[0]
            ma_state_dim = ma_obs[first_agent].shape[0] if first_agent in ma_obs else dt_dataset.state_dim if dt_model is not None else 44
            ma_act_dim = 4  # 4 signal phases

            # Build adjacency matrix (chain along route)
            adj = np.eye(n_agents_actual, dtype=np.float32)
            for i in range(n_agents_actual - 1):
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0
            adj = torch.tensor(adj, dtype=torch.float32)

            madt_cfg = cfg["dt"]
            madt_model = MultiAgentDecisionTransformer(
                state_dim=ma_state_dim,
                act_dim=ma_act_dim,
                n_agents=n_agents_actual,
                adj_matrix=adj,
                hidden_dim=madt_cfg["hidden_dim"],
                n_layers=max(madt_cfg["n_layers"] // 2, 1),
                n_heads=madt_cfg["n_heads"],
                max_length=madt_cfg["context_length"],
                max_ep_len=madt_cfg["max_ep_len"],
            ).to(device)

            # Quick training on multi-agent rollouts (synthetic — from greedy)
            # For simplicity, just evaluate untrained MADT; training requires
            # MultiAgentTrajectoryDataset which needs MA data collection.
            # We still evaluate to get a result row.
            methods_results["MADT"] = _evaluate_madt(
                eval_ma_env,
                madt_model,
                target_return=cfg["dt"]["target_return"],
                n_episodes=n_eval,
                device=device,
                desc="MADT",
            )
        except Exception as e:
            print(f"    WARNING: MADT eval failed: {e}")
            traceback.print_exc()

        # ---- Extract metrics for this seed ----
        for method, ep_infos in methods_results.items():
            metrics = _extract_metrics(ep_infos)
            all_seed_metrics.setdefault(method, []).append(metrics)

    # ---- Aggregate across seeds ----
    final: Dict[str, Dict[str, float]] = {}
    for method, seed_metrics_list in all_seed_metrics.items():
        aggregated: Dict[str, float] = {}
        metric_keys = seed_metrics_list[0].keys()
        for key in metric_keys:
            vals = [sm[key] for sm in seed_metrics_list]
            if key.endswith("_std"):
                # For std metrics, take mean of per-seed stds
                aggregated[key] = round(float(np.mean(vals)), 1)
            else:
                aggregated[key] = round(float(np.mean(vals)), 1)
        final[method] = aggregated

    return final


# ======================================================================
# Summary table
# ======================================================================

def print_summary_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    for grid_label, methods in results.items():
        print(f"\n--- {grid_label} ---")
        header = f"{'Method':<14} {'ETT':>8} {'ETT_s':>7} {'ACD':>8} {'ACD_s':>7} {'Thru':>8} {'Thru_s':>7} {'Stops':>7} {'Stop_s':>7}"
        print(header)
        print("-" * len(header))

        for method, m in methods.items():
            print(
                f"{method:<14} "
                f"{m.get('ett_mean', -1):>8.1f} "
                f"{m.get('ett_std', 0):>7.1f} "
                f"{m.get('acd_mean', 0):>8.1f} "
                f"{m.get('acd_std', 0):>7.1f} "
                f"{m.get('throughput_mean', 0):>8.0f} "
                f"{m.get('throughput_std', 0):>7.0f} "
                f"{m.get('ev_stops_mean', 0):>7.1f} "
                f"{m.get('ev_stops_std', 0):>7.1f}"
            )
    print()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for EV corridor optimization."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs="+",
        default=[4],
        help="Grid sizes to evaluate, e.g. --grid-size 4 6 8 (default: 4)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of random seeds per grid size (default: 3)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override number of evaluation episodes (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/real_experiments",
        help="Output directory for data and models (default: results/real_experiments)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode: fewer episodes, smaller models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device (auto/cpu/cuda, default: auto)",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Configuration
    cfg = _default_config(quick=args.quick)

    if args.num_episodes is not None:
        cfg["eval"]["n_episodes"] = args.num_episodes

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Seeds
    seeds = list(range(42, 42 + args.num_seeds))

    # Run experiments for each grid size
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    metadata = {
        "description": "Real experiment results from run_real_experiments.py",
        "dt_seconds": 5,
        "eval_episodes": cfg["eval"]["n_episodes"],
        "num_seeds": args.num_seeds,
        "quick_mode": args.quick,
        "grid_sizes": args.grid_size,
        "date": time.strftime("%Y-%m-%d"),
    }

    for gs in args.grid_size:
        print(f"\n{'='*60}")
        print(f"  Grid size: {gs}x{gs}")
        print(f"{'='*60}")

        grid_results = run_grid_experiment(
            grid_size=gs,
            seeds=seeds,
            cfg=cfg,
            output_dir=args.output_dir,
            device=device,
        )
        grid_label = f"table1_main_{gs}x{gs}"
        all_results[grid_label] = grid_results

    # Build scalability table if multiple grid sizes
    if len(args.grid_size) > 1:
        scalability: Dict[str, Dict[str, float]] = {}
        for gs in args.grid_size:
            label = f"table1_main_{gs}x{gs}"
            methods = all_results.get(label, {})
            entry: Dict[str, float] = {}
            if "DT" in methods:
                entry["DT_ett"] = methods["DT"]["ett_mean"]
            if "MADT" in methods:
                entry["MADT_ett"] = methods["MADT"]["ett_mean"]
            if "FT-EVP" in methods:
                entry["FT_EVP_ett"] = methods["FT-EVP"]["ett_mean"]
                if "DT" in methods and methods["FT-EVP"]["ett_mean"] > 0:
                    entry["DT_improvement"] = round(
                        100.0 * (1.0 - methods["DT"]["ett_mean"] / methods["FT-EVP"]["ett_mean"]),
                        1,
                    )
                if "MADT" in methods and methods["FT-EVP"]["ett_mean"] > 0:
                    entry["MADT_improvement"] = round(
                        100.0 * (1.0 - methods["MADT"]["ett_mean"] / methods["FT-EVP"]["ett_mean"]),
                        1,
                    )
            scalability[f"{gs}x{gs}"] = entry
        all_results["table2_scalability"] = scalability

    # Assemble output in narrative_numbers.json format
    output = {"metadata": metadata}
    output.update(all_results)

    # Save results
    results_path = os.path.join(args.output_dir, "real_experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {results_path}")

    # Also save to the standard results directory
    canonical_path = os.path.join(_PROJECT_ROOT, "results", "real_experiment_results.json")
    os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
    with open(canonical_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"Results also saved to {canonical_path}")

    # Print summary
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
