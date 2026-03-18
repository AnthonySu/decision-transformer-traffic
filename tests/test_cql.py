"""Tests for src.baselines.cql_baseline — Conservative Q-Learning."""

import torch

from src.baselines.cql_baseline import CQLAgent


class TestCQLAgent:
    def test_init(self):
        agent = CQLAgent(state_dim=10, act_dim=4)
        assert agent.q_net is not None
        assert agent.q_target is not None

    def test_select_action(self):
        agent = CQLAgent(state_dim=10, act_dim=4)
        obs = torch.randn(10).numpy()
        action = agent.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_select_action_deterministic(self):
        agent = CQLAgent(state_dim=10, act_dim=4)
        obs = torch.randn(10).numpy()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_q_values_shape(self):
        agent = CQLAgent(state_dim=10, act_dim=4)
        state = torch.randn(1, 10)
        q_vals = agent.q_net(state)
        assert q_vals.shape == (1, 4)

    def test_target_separate_from_q(self):
        agent = CQLAgent(state_dim=10, act_dim=4)
        # Modify Q-net weights
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.add_(1.0)
        # Target should not have changed
        q_out = agent.q_net(torch.randn(1, 10))
        t_out = agent.q_target(torch.randn(1, 10))
        # They should differ since we only modified q_net
        assert not torch.equal(
            list(agent.q_net.parameters())[0],
            list(agent.q_target.parameters())[0],
        )
