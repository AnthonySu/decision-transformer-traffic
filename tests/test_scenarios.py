"""Tests for src.envs.scenarios — scenario registry and factory."""

import pytest

from src.envs.scenarios import (
    create_env_from_scenario,
    get_scenario,
    list_scenarios,
)


class TestScenarioRegistry:
    def test_list_scenarios_non_empty(self):
        scenarios = list_scenarios()
        assert len(scenarios) >= 4

    def test_all_scenarios_have_required_keys(self):
        for name in list_scenarios():
            cfg = get_scenario(name)
            assert "description" in cfg

    def test_get_known_scenario(self):
        cfg = get_scenario("grid-4x4-v0")
        assert cfg["rows"] == 4
        assert cfg["cols"] == 4

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_scenario("nonexistent-v99")


class TestCreateEnvFromScenario:
    def test_grid_scenario(self):
        env = create_env_from_scenario("grid-3x3-v0")
        obs, info = env.reset()
        assert obs.shape[0] > 0

    def test_grid_4x4(self):
        env = create_env_from_scenario("grid-4x4-v0")
        obs, info = env.reset()
        action = env.action_space.sample()
        obs2, r, term, trunc, info2 = env.step(action)
        assert obs2.shape == obs.shape

    def test_multi_agent_flag(self):
        env = create_env_from_scenario("grid-3x3-v0", multi_agent=True)
        obs = env.reset()
        assert isinstance(obs, (dict, tuple))
