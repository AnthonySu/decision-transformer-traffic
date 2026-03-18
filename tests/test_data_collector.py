"""Tests for src.utils.data_collector — offline data collection utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.data_collector import DataCollector, _compute_returns_to_go


# ======================================================================
# Helpers
# ======================================================================

class _DummyEnv:
    """Minimal mock environment for testing DataCollector without real envs."""

    def __init__(self, episode_length: int = 10, state_dim: int = 5):
        self.episode_length = episode_length
        self.state_dim = state_dim
        self._step = 0
        self.action_space = MagicMock()
        self.action_space.sample.return_value = 0

    def reset(self, **kwargs):
        self._step = 0
        obs = np.zeros(self.state_dim, dtype=np.float32)
        info = {"ev_info": {"active": True, "distance": 5, "phase": 0}}
        return obs, info

    def step(self, action):
        self._step += 1
        obs = np.random.randn(self.state_dim).astype(np.float32)
        reward = -1.0 + np.random.randn() * 0.1
        terminated = self._step >= self.episode_length
        truncated = False
        info = {"ev_info": {"active": True, "distance": max(0, 5 - self._step), "phase": 0}}
        if terminated:
            info["ev_travel_time"] = float(self._step * 5.0)
        return obs, reward, terminated, truncated, info


class _DummyPolicy:
    """Minimal mock policy."""

    def select_action(self, obs, ev_info):
        return 0

    def reset(self):
        pass


# ======================================================================
# test_collect_single_episode
# ======================================================================

class TestCollectSingleEpisode:
    """Tests for collecting a single episode."""

    def test_episode_structure(self):
        """Collected episode should have all required keys."""
        env = _DummyEnv(episode_length=5, state_dim=4)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1, policy_name="test")
        assert len(episodes) == 1

        ep = episodes[0]
        required_keys = {
            "states", "actions", "rewards", "dones", "returns_to_go",
            "policy_name", "episode_return", "episode_length", "ev_travel_time",
        }
        assert required_keys.issubset(ep.keys())

    def test_episode_arrays_shape(self):
        """Arrays in the episode should have consistent shapes."""
        env = _DummyEnv(episode_length=8, state_dim=6)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1)
        ep = episodes[0]

        T = ep["episode_length"]
        assert ep["states"].shape == (T, 6)
        # Actions may be 1D (scalar) or 2D (MultiDiscrete)
        assert ep["actions"].shape[0] == T
        assert ep["rewards"].shape == (T,)
        assert ep["dones"].shape == (T,)
        assert ep["returns_to_go"].shape == (T,)

    def test_episode_length_matches(self):
        """episode_length should match the array lengths."""
        env = _DummyEnv(episode_length=7)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1)
        ep = episodes[0]
        assert ep["episode_length"] == len(ep["rewards"])

    def test_episode_return_is_sum_of_rewards(self):
        """episode_return should equal sum of rewards."""
        env = _DummyEnv(episode_length=5)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1)
        ep = episodes[0]
        assert ep["episode_return"] == pytest.approx(float(np.sum(ep["rewards"])))

    def test_dones_last_element_true(self):
        """The last dones entry should be True (episode terminated)."""
        env = _DummyEnv(episode_length=5)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1)
        ep = episodes[0]
        assert ep["dones"][-1] is True or ep["dones"][-1] == True

    def test_policy_name_stored(self):
        """policy_name should be stored in the episode dict."""
        env = _DummyEnv(episode_length=3)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1, policy_name="my_policy")
        assert episodes[0]["policy_name"] == "my_policy"

    def test_multiple_episodes(self):
        """collect_episodes with num_episodes > 1 should return that many episodes."""
        env = _DummyEnv(episode_length=3)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=5)
        assert len(episodes) == 5

    def test_episodes_accumulated(self):
        """Multiple calls to collect_episodes should accumulate episodes."""
        env = _DummyEnv(episode_length=3)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        collector.collect_episodes(policy, num_episodes=2)
        collector.collect_episodes(policy, num_episodes=3)
        assert len(collector._episodes) == 5


# ======================================================================
# test_returns_to_go_computation
# ======================================================================

class TestReturnsToGoComputation:
    """Tests for _compute_returns_to_go."""

    def test_basic_rtg(self):
        """RTG at time t should equal sum of rewards from t to T."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        rtg = _compute_returns_to_go(rewards)

        # rtg[0] = 1+2+3+4 = 10
        # rtg[1] = 2+3+4 = 9
        # rtg[2] = 3+4 = 7
        # rtg[3] = 4
        expected = np.array([10.0, 9.0, 7.0, 4.0])
        np.testing.assert_array_almost_equal(rtg, expected)

    def test_rtg_last_element(self):
        """RTG at the last timestep should equal the last reward."""
        rewards = np.array([5.0, -3.0, 2.0])
        rtg = _compute_returns_to_go(rewards)
        assert rtg[-1] == pytest.approx(2.0)

    def test_rtg_first_element_equals_episode_return(self):
        """RTG at time 0 should equal the total episode return."""
        rewards = np.array([1.0, -2.0, 3.0, -1.0, 5.0])
        rtg = _compute_returns_to_go(rewards)
        assert rtg[0] == pytest.approx(np.sum(rewards))

    def test_rtg_single_step(self):
        """RTG for a single-step episode should equal that single reward."""
        rewards = np.array([7.5])
        rtg = _compute_returns_to_go(rewards)
        assert rtg[0] == pytest.approx(7.5)

    def test_rtg_negative_rewards(self):
        """RTG should handle negative rewards correctly."""
        rewards = np.array([-1.0, -1.0, -1.0])
        rtg = _compute_returns_to_go(rewards)
        np.testing.assert_array_almost_equal(rtg, np.array([-3.0, -2.0, -1.0]))

    def test_rtg_monotonically_nonincreasing_for_positive_rewards(self):
        """With all positive rewards, RTG should be non-increasing."""
        rewards = np.array([2.0, 3.0, 1.0, 4.0])
        rtg = _compute_returns_to_go(rewards)
        for i in range(len(rtg) - 1):
            assert rtg[i] >= rtg[i + 1]

    def test_rtg_shape_matches_rewards(self):
        """RTG array should have the same shape as rewards."""
        rewards = np.random.randn(20).astype(np.float32)
        rtg = _compute_returns_to_go(rewards)
        assert rtg.shape == rewards.shape

    def test_rtg_with_collected_episode(self):
        """RTG computed inside DataCollector should match manual computation."""
        env = _DummyEnv(episode_length=5)
        collector = DataCollector(env, save_path="/tmp/test_dataset.h5")
        policy = _DummyPolicy()

        episodes = collector.collect_episodes(policy, num_episodes=1)
        ep = episodes[0]

        # Manually compute RTG
        expected_rtg = _compute_returns_to_go(ep["rewards"])
        np.testing.assert_array_almost_equal(ep["returns_to_go"], expected_rtg)
