"""Tests for LightSim backend integration in EVCorridorEnv.

All tests mock the ``lightsim`` dependency so they run without it installed.
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.envs.ev_corridor_env import EVCorridorEnv

# ======================================================================
# Helpers
# ======================================================================

def _fake_topology() -> dict:
    """Return a minimal 2x2 lightsim-style topology dict."""
    nodes = [
        {"id": "n0", "x": 0, "y": 0, "row": 0, "col": 0, "num_phases": 4, "is_boundary": True},
        {"id": "n1", "x": 1, "y": 0, "row": 0, "col": 1, "num_phases": 4, "is_boundary": False},
        {"id": "n2", "x": 0, "y": 1, "row": 1, "col": 0, "num_phases": 4, "is_boundary": False},
        {"id": "n3", "x": 1, "y": 1, "row": 1, "col": 1, "num_phases": 4, "is_boundary": True},
    ]
    edges = [
        {"id": "e0", "source": "n0", "target": "n1", "length": 200, "lanes": 2, "speed_limit": 15.0, "direction": "E"},
        {"id": "e1", "source": "n1", "target": "n3", "length": 200, "lanes": 2, "speed_limit": 15.0, "direction": "S"},
        {"id": "e2", "source": "n0", "target": "n2", "length": 200, "lanes": 2, "speed_limit": 15.0, "direction": "S"},
        {"id": "e3", "source": "n2", "target": "n3", "length": 200, "lanes": 2, "speed_limit": 15.0, "direction": "E"},
    ]
    return {"nodes": nodes, "edges": edges}


def _make_fake_lightsim_module() -> ModuleType:
    """Create a fake ``lightsim`` module with a mock ``make()`` function."""
    mod = ModuleType("lightsim")

    topo = _fake_topology()

    fake_env = MagicMock()
    fake_env.topology = topo
    fake_env.action_space = None  # dict-style actions

    # reset returns (obs_dict, info_dict) with Layout-1 observations
    def _reset(seed=None):
        densities = {e["id"]: 0.02 for e in topo["edges"]}
        phases = {n["id"]: 0 for n in topo["nodes"]}
        obs = {"link_densities": densities, "signal_phases": phases}
        return obs, {}

    fake_env.reset.side_effect = _reset

    # step returns (obs, reward, terminated, truncated, info)
    def _step(actions):
        densities = {e["id"]: 0.03 for e in topo["edges"]}
        phases = {n["id"]: 0 for n in topo["nodes"]}
        obs = {"link_densities": densities, "signal_phases": phases}
        return obs, 0.0, False, False, {}

    fake_env.step.side_effect = _step

    mod.make = MagicMock(return_value=fake_env)  # type: ignore[attr-defined]
    mod._fake_env = fake_env  # stash for assertion access

    return mod


def _patch_lightsim(fake_mod: ModuleType):
    """Return a patcher that injects the fake lightsim module into sys.modules."""
    return patch.dict("sys.modules", {"lightsim": fake_mod})


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def fake_lightsim():
    """Provide a fake lightsim module and patch it into sys.modules."""
    mod = _make_fake_lightsim_module()
    with _patch_lightsim(mod):
        yield mod


# ======================================================================
# Tests
# ======================================================================

class TestLightSimAdapterConstruction:
    """Test that LightSimAdapter builds a valid Network from a lightsim topology."""

    def test_lightsim_adapter_construction(self, fake_lightsim):
        from src.envs.lightsim_adapter import LightSimAdapter

        adapter = LightSimAdapter(scenario="grid-4x4-v0")
        net = adapter.network

        # Network has the expected top-level keys
        assert "nodes" in net
        assert "links" in net
        assert "graph" in net

        # All four nodes present
        assert len(net["nodes"]) == 4
        for nid in ["n0", "n1", "n2", "n3"]:
            assert nid in net["nodes"]
            node = net["nodes"][nid]
            assert "current_phase" in node
            assert "num_phases" in node
            assert "incoming_links" in node
            assert "outgoing_links" in node

        # All four edges present with required link fields
        assert len(net["links"]) == 4
        for lid in ["e0", "e1", "e2", "e3"]:
            assert lid in net["links"]
            link = net["links"][lid]
            for key in ("source", "target", "length", "density", "flow",
                        "v_free", "k_jam", "num_lanes"):
                assert key in link, f"Missing key '{key}' in link {lid}"

        # Connectivity: n0 has outgoing links, n3 has incoming links
        assert len(net["nodes"]["n0"]["outgoing_links"]) == 2
        assert len(net["nodes"]["n3"]["incoming_links"]) == 2


class TestEnvWithLightSimBackend:
    """Test EVCorridorEnv with use_lightsim=True produces valid outputs."""

    def test_env_with_lightsim_backend(self, fake_lightsim):
        env = EVCorridorEnv(
            rows=2, cols=2, use_lightsim=True, max_steps=50, seed=42,
            lightsim_scenario="grid-4x4-v0",
        )

        obs, info = env.reset()

        # Observation shape and dtype
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

        # Info has expected keys
        assert "ev_link_idx" in info
        assert "ev_arrived" in info
        assert "step" in info
        assert info["step"] == 0

        # Take a step with a valid action
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info2["step"] == 1


class TestEnvDelegatesToAdapter:
    """Verify that use_lightsim=True delegates to adapter, not ctm_step."""

    def test_env_delegates_to_adapter(self, fake_lightsim):
        env = EVCorridorEnv(
            rows=2, cols=2, use_lightsim=True, max_steps=50, seed=42,
            lightsim_scenario="grid-4x4-v0",
        )

        # Access the underlying mock env to track calls
        ls_env_mock = fake_lightsim._fake_env

        # reset() should call adapter.reset -> lightsim_env.reset
        ls_env_mock.reset.reset_mock()
        env.reset()
        assert ls_env_mock.reset.call_count == 1

        # step() should call adapter.step -> lightsim_env.step
        ls_env_mock.step.reset_mock()
        action = env.action_space.sample()

        with patch("src.envs.ev_corridor_env.ctm_step") as mock_ctm:
            env.step(action)
            # lightsim env.step should be called
            assert ls_env_mock.step.call_count == 1
            # ctm_step should NOT be called
            mock_ctm.assert_not_called()


class TestLightSimUnavailableFallback:
    """When lightsim is not importable, use_lightsim=True should fail clearly."""

    def test_lightsim_unavailable_fallback(self):
        # Ensure lightsim is NOT in sys.modules (remove any fake)
        with patch.dict("sys.modules", {"lightsim": None}):
            with pytest.raises(ImportError, match="lightsim"):
                EVCorridorEnv(
                    rows=2, cols=2, use_lightsim=True, max_steps=50, seed=42,
                )


class TestEnvNetworkSync:
    """After adapter.step(), env's network densities should match the adapter's."""

    def test_env_network_sync(self, fake_lightsim):
        env = EVCorridorEnv(
            rows=2, cols=2, use_lightsim=True, max_steps=50, seed=42,
            lightsim_scenario="grid-4x4-v0",
        )
        env.reset()

        action = env.action_space.sample()
        env.step(action)

        # After step, the env's _network should be the same object as the
        # adapter's network (env does self._network = self._ls_adapter.network)
        adapter_net = env._ls_adapter.network
        assert env._network is adapter_net

        # Densities in the env's network should reflect the post-step values
        # Our fake step() sets all densities to 0.03
        for lid, lk in env._network["links"].items():
            assert lk["density"] == pytest.approx(0.03), (
                f"Link {lid} density not synced after step"
            )


class TestPhaseActionsTranslated:
    """Verify the action array is translated to {node_id: phase} for the adapter."""

    def test_phase_actions_translated(self, fake_lightsim):
        env = EVCorridorEnv(
            rows=2, cols=2, use_lightsim=True, max_steps=50, seed=42,
            lightsim_scenario="grid-4x4-v0",
        )
        env.reset()

        ls_env_mock = fake_lightsim._fake_env

        # Build a known action: set all route intersection phases to 2
        action = np.full(env._max_route_len, 2, dtype=np.int64)

        ls_env_mock.step.reset_mock()
        env.step(action)

        # The lightsim env's step should have been called with a dict
        # mapping node IDs to phase ints (since action_space is None -> dict mode)
        assert ls_env_mock.step.call_count == 1
        call_args = ls_env_mock.step.call_args
        ls_actions = call_args[0][0]

        # ls_actions should be a dict (adapter translates for dict-based action space)
        assert isinstance(ls_actions, dict), (
            f"Expected dict actions, got {type(ls_actions)}"
        )

        # Each node in the dict should have an integer phase value
        for node_id, phase_val in ls_actions.items():
            assert isinstance(phase_val, int)
            # Phase should be action[i] % num_phases = 2 % 4 = 2
            assert phase_val == 2, (
                f"Node {node_id}: expected phase 2, got {phase_val}"
            )
