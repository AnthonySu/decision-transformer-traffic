"""End-to-end integration tests for the multi-agent training pipeline.

Tests the full flow: generate dataset -> train MADT -> evaluate,
using tiny models and minimal episodes to keep tests fast (<30s total).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.generate_ma_dataset import (
    MARandomPolicy,
    collect_ma_episode,
    generate_dataset,
)
from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.models.madt import MultiAgentDecisionTransformer
from src.models.trajectory_dataset import MultiAgentTrajectoryDataset

# ======================================================================
# Helpers
# ======================================================================

ACT_DIM = 4
HIDDEN_DIM = 32
N_LAYERS = 1
N_HEADS = 2
GAT_HEADS = 2
GAT_LAYERS = 1
CONTEXT_LENGTH = 5
MAX_STEPS = 30


def _make_env(rows: int = 3, cols: int = 3) -> EVCorridorMAEnv:
    origin = "n0_0"
    destination = f"n{rows - 1}_{cols - 1}"
    return EVCorridorMAEnv(
        rows=rows, cols=cols, max_steps=MAX_STEPS,
        origin=origin, destination=destination, seed=42,
    )


def _collect_episodes(env: EVCorridorMAEnv, policy, n_agents: int, n_episodes: int = 5):
    """Collect episodes, retrying on None results."""
    episodes = []
    attempts = 0
    while len(episodes) < n_episodes and attempts < n_episodes * 5:
        attempts += 1
        ep = collect_ma_episode(env, policy, n_agents)
        if ep is not None:
            episodes.append(ep)
    return episodes


def _build_model(n_agents: int, state_dim: int) -> MultiAgentDecisionTransformer:
    adj = torch.eye(n_agents, dtype=torch.float32)
    for i in range(n_agents - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    return MultiAgentDecisionTransformer(
        state_dim=state_dim,
        act_dim=ACT_DIM,
        n_agents=n_agents,
        adj_matrix=adj,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        gat_heads=GAT_HEADS,
        gat_layers=GAT_LAYERS,
        max_length=CONTEXT_LENGTH,
        max_ep_len=MAX_STEPS + 10,
        dropout=0.0,
    )


def _train_model(model, dataloader, n_agents, epochs=2):
    """Train MADT for a few epochs. Returns list of epoch losses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for _epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch in dataloader:
            states = batch["states"]
            actions = batch["actions"]
            rtg = batch["returns_to_go"]
            timesteps = batch["timesteps"]
            mask = batch["attention_mask"]

            logits = model(states, actions, rtg, timesteps)
            logits_flat = logits.reshape(-1, ACT_DIM)
            targets_flat = actions.reshape(-1).clamp(0, ACT_DIM - 1)
            mask_expanded = mask.unsqueeze(1).expand(-1, n_agents, -1).reshape(-1).bool()

            if mask_expanded.any():
                loss = loss_fn(logits_flat[mask_expanded], targets_flat[mask_expanded])
            else:
                loss = loss_fn(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        losses.append(float(np.mean(epoch_losses)))
    return losses


def _evaluate_model(model, env, n_agents, n_episodes=1, target_return=10.0):
    """Evaluate MADT for n_episodes. Returns list of episode dicts."""
    model.eval()
    results = []

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        if len(env.agents) != n_agents:
            continue

        agents_snapshot = list(env.agents)
        state_dim = model.state_dim
        K = model.max_length
        N = n_agents

        states_buf = torch.zeros(1, N, K, state_dim)
        actions_buf = torch.zeros(1, N, K, dtype=torch.long)
        rtg_buf = torch.zeros(1, N, K, 1)
        ts_buf = torch.zeros(1, N, K, dtype=torch.long)

        for i, a in enumerate(agents_snapshot):
            states_buf[0, i, 0] = torch.tensor(obs_dict[a], dtype=torch.float32)
            rtg_buf[0, i, 0, 0] = target_return / N

        ep_return = 0.0
        t = 0

        for _ in range(MAX_STEPS):
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
                            obs_dict[agent_name], dtype=torch.float32
                        )
                    actions_buf[0, i, t - 1] = action_dict.get(agent_name, 0)
                    agent_rew = rew_dict.get(agent_name, 0.0)
                    rtg_buf[0, i, t, 0] = rtg_buf[0, i, t - 1, 0] - agent_rew
                    ts_buf[0, i, t] = min(t, MAX_STEPS)

            if not env.agents:
                break

        results.append({"return": ep_return, "length": t})

    return results


# ======================================================================
# Tests
# ======================================================================


class TestMADataCollection:
    """Test multi-agent data collection from EVCorridorMAEnv."""

    def test_ma_data_collection(self):
        """Collect trajectories using a random policy and verify format."""
        env = _make_env(3, 3)
        obs_dict, _ = env.reset()
        n_agents = len(env.agents)
        state_dim = next(iter(obs_dict.values())).shape[0]

        policy = MARandomPolicy()
        episodes = _collect_episodes(env, policy, n_agents, n_episodes=3)

        assert len(episodes) >= 1, "Should collect at least 1 episode"

        for ep in episodes:
            # Check required keys
            assert "states" in ep
            assert "actions" in ep
            assert "rewards" in ep
            assert "dones" in ep

            T = ep["states"].shape[0]
            assert T >= 2, "Episode should have at least 2 timesteps"

            # Check shapes: [T, n_agents, state_dim], [T, n_agents], etc.
            assert ep["states"].shape == (T, n_agents, state_dim)
            assert ep["actions"].shape == (T, n_agents)
            assert ep["rewards"].shape == (T, n_agents)
            assert ep["dones"].shape == (T,)

            # Check dtypes
            assert ep["states"].dtype == np.float32
            assert ep["actions"].dtype == np.int64
            assert ep["rewards"].dtype == np.float32
            assert ep["dones"].dtype == bool

            # Actions should be valid (in [0, ACT_DIM))
            assert np.all(ep["actions"] >= 0)
            assert np.all(ep["actions"] < ACT_DIM)

            # Last done should be True
            assert ep["dones"][-1] is True or ep["dones"][-1] == True  # noqa: E712


class TestMADTTrainAndEval:
    """Test the full pipeline: collect data -> train MADT -> evaluate."""

    def test_madt_train_and_eval(self):
        """Collect data, train MADT for 2 epochs, evaluate for 1 episode."""
        env = _make_env(3, 3)
        obs_dict, _ = env.reset()
        n_agents = len(env.agents)
        state_dim = next(iter(obs_dict.values())).shape[0]

        # Step 1: Collect data and save to temp HDF5
        policy = MARandomPolicy()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test_ma_dataset.h5")
            generate_dataset(
                rows=3, cols=3, max_steps=MAX_STEPS,
                num_expert=3, num_random=2, num_noisy=2,
                save_path=data_path, seed=42,
            )

            # Step 2: Load dataset
            dataset = MultiAgentTrajectoryDataset(
                data_path=data_path,
                n_agents=n_agents,
                context_length=CONTEXT_LENGTH,
            )
            assert len(dataset) > 0, "Dataset should have samples"

            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=True, num_workers=0,
            )

            # Step 3: Build and train model
            model = _build_model(n_agents, state_dim)
            losses = _train_model(model, dataloader, n_agents, epochs=2)

            assert len(losses) == 2
            for loss_val in losses:
                assert np.isfinite(loss_val), "Loss should be finite"
                assert loss_val > 0, "Loss should be positive"

            # Step 4: Evaluate
            eval_env = _make_env(3, 3)
            results = _evaluate_model(model, eval_env, n_agents, n_episodes=1)

            assert len(results) >= 1, "Should have at least 1 evaluation episode"
            for r in results:
                assert "return" in r
                assert "length" in r
                assert isinstance(r["return"], float)
                assert isinstance(r["length"], int)
                assert r["length"] > 0


class TestMADTWithGreedyExpert:
    """Test training MADT on expert data from GreedyPreemptPolicy."""

    def test_madt_with_greedy_expert(self):
        """Collect expert data, train MADT, verify it produces reasonable actions."""
        env = _make_env(3, 3)
        obs_dict, _ = env.reset()
        n_agents = len(env.agents)
        state_dim = next(iter(obs_dict.values())).shape[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test_expert_dataset.h5")

            # Generate expert-heavy dataset
            generate_dataset(
                rows=3, cols=3, max_steps=MAX_STEPS,
                num_expert=5, num_random=1, num_noisy=1,
                save_path=data_path, seed=42,
            )

            dataset = MultiAgentTrajectoryDataset(
                data_path=data_path,
                n_agents=n_agents,
                context_length=CONTEXT_LENGTH,
            )
            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=True, num_workers=0,
            )

            model = _build_model(n_agents, state_dim)
            losses = _train_model(model, dataloader, n_agents, epochs=2)

            # Loss should decrease or at least remain finite
            assert all(np.isfinite(l) for l in losses)

            # Evaluate: model should produce valid actions for all agents
            model.eval()
            eval_env = _make_env(3, 3)
            obs_dict, _ = eval_env.reset()

            if len(eval_env.agents) == n_agents:
                agents_snapshot = list(eval_env.agents)
                K = model.max_length

                states_buf = torch.zeros(1, n_agents, K, state_dim)
                actions_buf = torch.zeros(1, n_agents, K, dtype=torch.long)
                rtg_buf = torch.zeros(1, n_agents, K, 1)
                ts_buf = torch.zeros(1, n_agents, K, dtype=torch.long)

                for i, a in enumerate(agents_snapshot):
                    states_buf[0, i, 0] = torch.tensor(
                        obs_dict[a], dtype=torch.float32
                    )
                    rtg_buf[0, i, 0, 0] = 10.0 / n_agents

                # Get actions for all agents
                for i in range(n_agents):
                    action = model.get_action(
                        states_buf[:, :, :1],
                        actions_buf[:, :, :1],
                        rtg_buf[:, :, :1],
                        ts_buf[:, :, :1],
                        agent_idx=i,
                    )
                    assert isinstance(action, int)
                    assert 0 <= action < ACT_DIM


class TestPipelineDifferentGridSizes:
    """Test the pipeline handles different grid sizes."""

    @pytest.mark.parametrize("rows,cols", [(3, 3), (4, 4)])
    def test_pipeline_different_grid_sizes(self, rows, cols):
        """Full pipeline on different grid sizes."""
        env = _make_env(rows, cols)
        obs_dict, _ = env.reset()
        n_agents = len(env.agents)
        state_dim = next(iter(obs_dict.values())).shape[0]

        assert n_agents > 0, f"Should have agents for {rows}x{cols} grid"

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, f"test_dataset_{rows}x{cols}.h5")

            generate_dataset(
                rows=rows, cols=cols, max_steps=MAX_STEPS,
                num_expert=3, num_random=1, num_noisy=1,
                save_path=data_path, seed=42,
            )

            dataset = MultiAgentTrajectoryDataset(
                data_path=data_path,
                n_agents=n_agents,
                context_length=CONTEXT_LENGTH,
            )
            assert len(dataset) > 0

            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=True, num_workers=0,
            )

            model = _build_model(n_agents, state_dim)
            losses = _train_model(model, dataloader, n_agents, epochs=2)

            assert len(losses) == 2
            assert all(np.isfinite(l) for l in losses)

            # Evaluate
            eval_env = _make_env(rows, cols)
            results = _evaluate_model(model, eval_env, n_agents, n_episodes=1)

            assert len(results) >= 1
            assert results[0]["length"] > 0
