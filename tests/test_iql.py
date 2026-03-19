"""Tests for src.baselines.iql_baseline — Implicit Q-Learning."""

import numpy as np
import torch

from src.baselines.iql_baseline import IQLPolicy


class TestIQLConstruction:
    """test_iql_construction — verify model builds with correct dimensions."""

    def test_init_defaults(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        assert agent.state_dim == 10
        assert agent.act_dim == 4
        assert agent.q_net is not None
        assert agent.q_target is not None
        assert agent.v_net is not None
        assert agent.policy_net is not None

    def test_q_output_shape(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        out = agent.q_net(torch.randn(2, 10))
        assert out.shape == (2, 4)

    def test_v_output_shape(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        out = agent.v_net(torch.randn(2, 10))
        assert out.shape == (2, 1)

    def test_policy_output_shape(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        out = agent.policy_net(torch.randn(2, 10))
        assert out.shape == (2, 4)

    def test_custom_hyperparams(self):
        agent = IQLPolicy(
            state_dim=8,
            act_dim=6,
            hidden_dim=64,
            n_layers=3,
            tau=0.9,
            beta=5.0,
            gamma=0.95,
        )
        assert agent.tau == 0.9
        assert agent.beta == 5.0
        assert agent.gamma == 0.95

    def test_target_separate_from_q(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.add_(1.0)
        # Target should not have changed
        assert not torch.equal(
            list(agent.q_net.parameters())[0],
            list(agent.q_target.parameters())[0],
        )


class _SyntheticDataset:
    """Minimal in-memory dataset mimicking OfflineRLDataset interface."""

    def __init__(self, n=200, state_dim=10, act_dim=4):
        self.states = np.random.randn(n, state_dim).astype(np.float32)
        self.actions = np.random.randint(0, act_dim, size=n).astype(np.int64)
        self.rewards = np.random.randn(n).astype(np.float32)
        self.next_states = np.random.randn(n, state_dim).astype(np.float32)
        self.dones = np.zeros(n, dtype=np.float32)
        self.dones[-1] = 1.0

    def as_tensors(self, device="cpu"):
        return (
            torch.tensor(self.states, dtype=torch.float32, device=device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.next_states, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.states)


class TestIQLTraining:
    """test_iql_training — train for a few steps and verify loss decreases."""

    def test_loss_decreases(self):
        agent = IQLPolicy(state_dim=10, act_dim=4, lr=1e-3)
        dataset = _SyntheticDataset(n=200, state_dim=10, act_dim=4)

        history = agent.train(dataset, n_epochs=20, batch_size=64, log_interval=100)

        # Total loss should decrease from start to end
        first_losses = history["total_loss"][:3]
        last_losses = history["total_loss"][-3:]
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first, (
            f"Loss did not decrease: first={avg_first:.4f}, last={avg_last:.4f}"
        )

    def test_history_keys(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        dataset = _SyntheticDataset(n=100, state_dim=10, act_dim=4)
        history = agent.train(dataset, n_epochs=3, batch_size=64, log_interval=100)

        assert "v_loss" in history
        assert "q_loss" in history
        assert "policy_loss" in history
        assert "total_loss" in history
        assert len(history["total_loss"]) == 3

    def test_update_count_advances(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        dataset = _SyntheticDataset(n=100, state_dim=10, act_dim=4)
        assert agent._update_count == 0
        agent.train(dataset, n_epochs=2, batch_size=64, log_interval=100)
        assert agent._update_count > 0


class TestIQLAct:
    """test_iql_act — verify act() returns valid actions."""

    def test_act_returns_valid_action(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        obs = np.random.randn(10).astype(np.float32)
        action = agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_act_deterministic_consistent(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        obs = np.random.randn(10).astype(np.float32)
        a1 = agent.act(obs, deterministic=True)
        a2 = agent.act(obs, deterministic=True)
        assert a1 == a2

    def test_select_action_alias(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        obs = np.random.randn(10).astype(np.float32)
        a1 = agent.act(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_act_stochastic(self):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        obs = np.random.randn(10).astype(np.float32)
        action = agent.act(obs, deterministic=False)
        assert isinstance(action, int)
        assert 0 <= action < 4


class TestIQLSaveLoad:
    """test_iql_save_load — verify save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        dataset = _SyntheticDataset(n=100, state_dim=10, act_dim=4)
        agent.train(dataset, n_epochs=2, batch_size=64, log_interval=100)

        obs = np.random.randn(10).astype(np.float32)
        action_before = agent.act(obs, deterministic=True)

        ckpt_path = tmp_path / "iql_test.pt"
        agent.save(ckpt_path)

        # Load into a fresh agent
        agent2 = IQLPolicy(state_dim=10, act_dim=4)
        agent2.load(ckpt_path)

        action_after = agent2.act(obs, deterministic=True)
        assert action_before == action_after

    def test_save_creates_file(self, tmp_path):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        ckpt_path = tmp_path / "subdir" / "iql.pt"
        agent.save(ckpt_path)
        assert ckpt_path.exists()

    def test_load_restores_update_count(self, tmp_path):
        agent = IQLPolicy(state_dim=10, act_dim=4)
        dataset = _SyntheticDataset(n=100, state_dim=10, act_dim=4)
        agent.train(dataset, n_epochs=2, batch_size=64, log_interval=100)
        count = agent._update_count

        ckpt_path = tmp_path / "iql.pt"
        agent.save(ckpt_path)

        agent2 = IQLPolicy(state_dim=10, act_dim=4)
        agent2.load(ckpt_path)
        assert agent2._update_count == count
