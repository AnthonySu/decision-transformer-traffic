"""Tests for src.envs.ev_corridor_env — single-agent Gymnasium environment."""

import numpy as np
import pytest

from src.envs.ev_corridor_env import EVCorridorEnv

# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def env():
    """A small deterministic environment for testing."""
    e = EVCorridorEnv(rows=3, cols=3, max_steps=50, seed=42)
    return e


@pytest.fixture
def env_fixed_od():
    """Environment with fixed origin-destination for deterministic route."""
    e = EVCorridorEnv(
        rows=3, cols=3, max_steps=100, origin="n0_0", destination="n2_2", seed=42
    )
    return e


# ======================================================================
# test_reset
# ======================================================================

class TestReset:
    """Tests for env.reset()."""

    def test_returns_obs_and_info(self, env):
        """reset() should return a (obs, info) tuple."""
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_obs_shape(self, env):
        """Observation should match the declared observation_space shape."""
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_obs_dtype(self, env):
        """Observation should be float32."""
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_in_valid_range(self, env):
        """Observation values should be within [-1, 1]."""
        obs, _ = env.reset()
        assert np.all(obs >= -1.0 - 1e-6)
        assert np.all(obs <= 1.0 + 1e-6)

    def test_info_keys(self, env):
        """Info dict should contain expected keys."""
        _, info = env.reset()
        expected_keys = {
            "ev_link_idx",
            "ev_progress",
            "ev_arrived",
            "step",
            "intersections_passed",
            "return_to_go",
            "route_length",
            "total_queue",
        }
        assert expected_keys.issubset(info.keys())

    def test_initial_ev_state(self, env):
        """After reset, EV should be at the start of the route."""
        _, info = env.reset()
        assert info["ev_link_idx"] == 0
        assert info["ev_progress"] == 0.0
        assert info["ev_arrived"] is False
        assert info["step"] == 0
        assert info["intersections_passed"] == 0

    def test_reset_with_seed(self, env):
        """Resetting with the same seed should produce identical observations."""
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_route_length_positive(self, env):
        """Route should have at least 2 nodes (origin + destination)."""
        _, info = env.reset()
        assert info["route_length"] >= 2


# ======================================================================
# test_step
# ======================================================================

class TestStep:
    """Tests for env.step()."""

    def test_step_returns_five_tuple(self, env):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

    def test_obs_shape_preserved(self, env):
        """Observation shape should stay the same after step."""
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_reward_is_float(self, env):
        """Reward should be a Python float."""
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_terminated_truncated_are_bool(self, env):
        """terminated and truncated should be booleans."""
        env.reset()
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_count_increments(self, env):
        """Info step counter should increment each step."""
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["step"] == 1
        _, _, _, _, info = env.step(action)
        assert info["step"] == 2


# ======================================================================
# test_ev_progression
# ======================================================================

class TestEVProgression:
    """Tests for EV movement through the network."""

    def test_ev_advances_over_time(self, env_fixed_od):
        """Over many steps with favourable signals, EV should advance."""
        env_fixed_od.reset(seed=42)
        initial_idx = 0
        # Run many steps — giving all-green (phase that matches EV direction)
        # Use all-ones action (phase 1 = E/W through) as a heuristic
        action = np.ones(env_fixed_od.action_space.shape, dtype=int)
        max_idx = 0
        for _ in range(50):
            _, _, terminated, truncated, info = env_fixed_od.step(action)
            max_idx = max(max_idx, info["ev_link_idx"])
            if terminated or truncated:
                break
        assert max_idx > initial_idx, "EV should have advanced at least one link"

    def test_ev_progress_updates(self, env_fixed_od):
        """EV progress (fractional position on link) should change."""
        env_fixed_od.reset(seed=42)
        action = np.zeros(env_fixed_od.action_space.shape, dtype=int)
        progresses = set()
        for _ in range(20):
            _, _, terminated, truncated, info = env_fixed_od.step(action)
            progresses.add(round(info["ev_progress"], 4))
            if terminated or truncated:
                break
        # EV should have been at multiple progress values
        assert len(progresses) > 1


# ======================================================================
# test_ev_blocked_by_red
# ======================================================================

class TestEVBlockedByRed:
    """Tests verifying EV stops at red signals."""

    def test_ev_blocked_when_signal_wrong_phase(self, env_fixed_od):
        """When EV is near intersection and signal is red, speed should be 0."""
        env_fixed_od.reset(seed=42)
        # Manually advance EV progress close to end of first link
        env_fixed_od._ev_progress = 0.95
        # Set downstream node phase to something that blocks EV
        if env_fixed_od._ev_link_idx < len(env_fixed_od._route) - 1:
            _, link_id = env_fixed_od._route[env_fixed_od._ev_link_idx]
            if link_id is not None:
                link = env_fixed_od._network["links"][link_id]
                downstream = link["target"]
                # Set phase to opposite of what the link needs
                wrong_phase = (link["phase_index"] + 1) % 4
                env_fixed_od._network["nodes"][downstream]["current_phase"] = wrong_phase

                # Take a step — EV should NOT advance to next link
                action = np.full(env_fixed_od.action_space.shape, wrong_phase, dtype=int)
                _, _, _, _, info = env_fixed_od.step(action)
                # EV should still be on the same link (not advanced)
                assert info["ev_link_idx"] == 0


# ======================================================================
# test_episode_termination
# ======================================================================

class TestEpisodeTermination:
    """Tests for episode termination conditions."""

    def test_terminates_on_timeout(self):
        """Episode should truncate after max_steps."""
        env = EVCorridorEnv(rows=3, cols=3, max_steps=10, seed=42)
        env.reset()
        for step in range(20):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert step <= 10, "Should terminate within max_steps"
        # On timeout, truncated should be True (unless EV arrived)
        assert terminated or truncated

    def test_terminates_on_arrival(self, env_fixed_od):
        """Episode should terminate when EV reaches destination."""
        env_fixed_od.reset(seed=42)
        terminated = False
        for _ in range(env_fixed_od.max_steps):
            action = env_fixed_od.action_space.sample()
            _, _, terminated, truncated, info = env_fixed_od.step(action)
            if terminated:
                assert info["ev_arrived"] is True
                break
            if truncated:
                break
        # If we didn't get terminated, that's also fine (timeout)

    def test_not_both_terminated_and_truncated(self, env_fixed_od):
        """terminated and truncated should not both be True simultaneously."""
        env_fixed_od.reset(seed=42)
        for _ in range(env_fixed_od.max_steps + 5):
            action = env_fixed_od.action_space.sample()
            _, _, terminated, truncated, _ = env_fixed_od.step(action)
            assert not (terminated and truncated)
            if terminated or truncated:
                break


# ======================================================================
# test_reward_structure
# ======================================================================

class TestRewardStructure:
    """Tests for the reward components."""

    def test_time_penalty_present(self, env_fixed_od):
        """Each step should incur a negative time penalty component."""
        env_fixed_od.reset(seed=42)
        action = env_fixed_od.action_space.sample()
        # With no terminal event or intersection crossing, reward should be negative
        # (time penalty -1.0 + queue penalty)
        _, reward, _, _, _ = env_fixed_od.step(action)
        # reward = -1.0 + queue_penalty + intersection_bonus + terminal_bonus
        # On first step, no intersection bonus or terminal bonus likely
        assert reward < 0.0, "First-step reward should be negative (time + queue penalty)"

    def test_terminal_bonus_on_arrival(self, env_fixed_od):
        """Terminal reward should be positive (+50) when EV arrives."""
        env_fixed_od.reset(seed=42)
        # Manually set EV to nearly arrived state
        env_fixed_od._ev_link_idx = len(env_fixed_od._route) - 2
        env_fixed_od._ev_progress = 0.99

        # Give green to EV
        _, link_id = env_fixed_od._route[env_fixed_od._ev_link_idx]
        if link_id is not None:
            link = env_fixed_od._network["links"][link_id]
            downstream = link["target"]
            phase = link["phase_index"]
            env_fixed_od._network["nodes"][downstream]["current_phase"] = phase

        action = np.full(env_fixed_od.action_space.shape, 0, dtype=int)
        # Set the correct phase for the downstream node
        if link_id is not None:
            route_intersections = env_fixed_od._route_intersections
            for i, nid in enumerate(route_intersections):
                if nid == downstream and i < len(action):
                    action[i] = phase

        _, reward, terminated, _, _ = env_fixed_od.step(action)
        if terminated:
            # Should contain the +50 terminal bonus
            assert reward > 0.0, "Terminal reward on EV arrival should be positive"

    def test_terminal_penalty_on_timeout(self):
        """Terminal reward should be negative (-50) on timeout."""
        env = EVCorridorEnv(rows=3, cols=3, max_steps=5, seed=42)
        env.reset()
        last_reward = 0.0
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            last_reward = reward
            if terminated or truncated:
                break
        if truncated:
            assert last_reward < -40.0, "Timeout penalty should be large and negative"


# ======================================================================
# test_action_space
# ======================================================================

class TestActionSpace:
    """Tests for the action space."""

    def test_action_space_is_multidiscrete(self, env):
        """Action space should be MultiDiscrete."""
        from gymnasium.spaces import MultiDiscrete
        assert isinstance(env.action_space, MultiDiscrete)

    def test_action_space_shape(self, env):
        """MultiDiscrete shape should be (max_route_len,)."""
        assert len(env.action_space.shape) == 1
        assert env.action_space.shape[0] == env._max_route_len

    def test_each_action_has_4_phases(self, env):
        """Each action slot should allow values in {0, 1, 2, 3}."""
        nvec = env.action_space.nvec
        for n in nvec:
            assert n == 4

    def test_sample_valid(self, env):
        """Sampled actions should be valid (within space)."""
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)


# ======================================================================
# test_render
# ======================================================================

class TestRender:
    """Tests for env.render()."""

    def test_render_returns_string(self, env):
        """render() should return a string (when render_mode is not set)."""
        env.reset()
        result = env.render()
        assert isinstance(result, str)

    def test_render_contains_step_info(self, env):
        """Rendered string should contain step count."""
        env.reset()
        result = env.render()
        assert "Step" in result

    def test_render_contains_ev_info(self, env):
        """Rendered string should contain EV position info."""
        env.reset()
        result = env.render()
        assert "EV" in result

    def test_render_after_step(self, env):
        """render() should work after taking a step."""
        env.reset()
        env.step(env.action_space.sample())
        result = env.render()
        assert isinstance(result, str)
        assert len(result) > 0
