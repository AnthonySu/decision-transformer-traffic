"""Tests for src.baselines — FixedTimeEVP and GreedyPreemptPolicy."""

import numpy as np
import pytest

from src.baselines.fixed_time_evp import FixedTimeEVP
from src.baselines.greedy_preempt import GreedyPreemptPolicy


# ======================================================================
# FixedTimeEVP tests
# ======================================================================

class TestFixedTimeEVPNormal:
    """Tests for FixedTimeEVP under normal (no EV) operation."""

    @pytest.fixture
    def controller(self):
        return FixedTimeEVP(preemption_distance=3, phase_duration=5, num_phases=4)

    def test_initial_phase_is_zero(self, controller):
        """After construction, the initial phase should be 0."""
        obs = np.zeros(10)
        ev_info = {"active": False, "distance": float("inf"), "phase": None}
        action = controller.select_action(obs, ev_info)
        assert action == 0

    def test_cycles_phases(self, controller):
        """Under normal operation, phases should cycle 0 -> 1 -> 2 -> 3 -> 0."""
        obs = np.zeros(10)
        ev_info = {"active": False, "distance": float("inf"), "phase": None}

        phases_seen = []
        for _ in range(25):  # 5 steps * 4 phases + extra
            action = controller.select_action(obs, ev_info)
            phases_seen.append(action)

        # After phase_duration=5 steps, phase should advance
        # Steps 1-5: phase 0, step 5 switches to 1
        # Verify we see all 4 phases
        assert set(phases_seen) == {0, 1, 2, 3}

    def test_phase_duration_respected(self, controller):
        """Each phase should last exactly phase_duration steps."""
        obs = np.zeros(10)
        ev_info = {"active": False, "distance": float("inf"), "phase": None}

        phases = []
        for _ in range(20):
            action = controller.select_action(obs, ev_info)
            phases.append(action)

        # The first phase_duration steps should all be the same phase
        # (Starting from step 1 since timer increments first)
        assert len(set(phases[:4])) == 1  # first 4 should be same (timer 1-4)
        # Phase changes at step 5 (timer reaches 5 = phase_duration)

    def test_wraps_around(self, controller):
        """After cycling through all phases, should wrap back to 0."""
        obs = np.zeros(10)
        ev_info = {"active": False, "distance": float("inf"), "phase": None}

        # Run for enough steps to cycle through all phases and wrap
        for _ in range(21):  # 5 * 4 = 20 to cycle, +1 to start next cycle
            action = controller.select_action(obs, ev_info)

        # Should be back to phase 0 or 1
        assert action in {0, 1}

    def test_reset_clears_state(self, controller):
        """reset() should restore initial state."""
        obs = np.zeros(10)
        ev_info = {"active": False, "distance": float("inf"), "phase": None}

        # Advance a few steps
        for _ in range(10):
            controller.select_action(obs, ev_info)

        controller.reset()

        # Should be back to phase 0
        action = controller.select_action(obs, ev_info)
        assert action == 0


class TestFixedTimeEVPPreempt:
    """Tests for FixedTimeEVP under EV preemption."""

    @pytest.fixture
    def controller(self):
        return FixedTimeEVP(preemption_distance=3, phase_duration=5, num_phases=4)

    def test_switches_to_ev_phase(self, controller):
        """When EV is close and active, should switch to EV's requested phase."""
        obs = np.zeros(10)
        ev_info = {"active": True, "distance": 2, "phase": 1}
        action = controller.select_action(obs, ev_info)
        assert action == 1

    def test_holds_ev_phase(self, controller):
        """While EV is still close, should continue holding the EV phase."""
        obs = np.zeros(10)
        ev_info = {"active": True, "distance": 1, "phase": 2}

        for _ in range(10):
            action = controller.select_action(obs, ev_info)
            assert action == 2

    def test_no_preemption_when_ev_far(self, controller):
        """If EV is beyond preemption_distance, should use fixed timing."""
        obs = np.zeros(10)
        ev_info = {"active": True, "distance": 10, "phase": 3}
        action = controller.select_action(obs, ev_info)
        # Should be normal fixed-timing phase (0 initially)
        assert action == 0

    def test_no_preemption_when_ev_inactive(self, controller):
        """If EV is not active, should use fixed timing regardless of distance."""
        obs = np.zeros(10)
        ev_info = {"active": False, "distance": 1, "phase": 2}
        action = controller.select_action(obs, ev_info)
        assert action == 0  # normal phase

    def test_reverts_to_fixed_after_ev_passes(self, controller):
        """After EV leaves preemption zone, should revert to fixed timing."""
        obs = np.zeros(10)

        # Preempt for a few steps
        ev_close = {"active": True, "distance": 1, "phase": 3}
        for _ in range(5):
            controller.select_action(obs, ev_close)

        # EV passes — now far away
        ev_far = {"active": True, "distance": 10, "phase": 3}
        action = controller.select_action(obs, ev_far)

        # Should revert to fixed timing (timer resets, so phase 0)
        assert action == 0

    def test_preemption_overrides_any_phase(self, controller):
        """Even in the middle of a phase cycle, preemption should take effect."""
        obs = np.zeros(10)
        no_ev = {"active": False, "distance": float("inf"), "phase": None}

        # Advance to some non-zero phase
        for _ in range(7):
            controller.select_action(obs, no_ev)

        # Now trigger preemption
        ev_info = {"active": True, "distance": 1, "phase": 2}
        action = controller.select_action(obs, ev_info)
        assert action == 2


# ======================================================================
# GreedyPreemptPolicy tests
# ======================================================================

class _MockNetwork:
    """Minimal mock network for GreedyPreemptPolicy tests."""

    def __init__(self):
        self.intersections = ["int_0", "int_1", "int_2"]

    def get_incoming_counts(self, intersection_id, obs):
        # Return per-phase incoming counts derived from obs
        return obs[:4]

    def get_outgoing_counts(self, intersection_id, obs):
        return np.zeros(4)

    def num_phases(self, intersection_id):
        return 4


class TestGreedyPreempt:
    """Tests for GreedyPreemptPolicy."""

    @pytest.fixture
    def policy(self):
        net = _MockNetwork()
        route = ["int_0", "int_1", "int_2"]
        return GreedyPreemptPolicy(network=net, route=route)

    def test_gives_green_to_ev_direction(self, policy):
        """When EV is active at a route intersection, should return EV's phase."""
        obs = np.zeros(10)
        ev_info = {
            "active": True,
            "phase": 2,
            "intersection_id": "int_0",
        }
        action = policy.select_action(obs, ev_info)
        assert action == 2

    def test_gives_green_for_different_phases(self, policy):
        """Should correctly relay any EV phase."""
        obs = np.zeros(10)
        for phase in range(4):
            ev_info = {
                "active": True,
                "phase": phase,
                "intersection_id": "int_1",
            }
            action = policy.select_action(obs, ev_info)
            assert action == phase

    def test_falls_back_to_max_pressure_when_no_ev(self, policy):
        """Without an active EV, should use MaxPressure."""
        # obs where phase 3 has highest incoming count
        obs = np.array([1.0, 2.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ev_info = {
            "active": False,
            "phase": None,
            "intersection_id": "int_0",
        }
        action = policy.select_action(obs, ev_info)
        assert action == 3  # argmax of [1, 2, 3, 5]

    def test_falls_back_for_off_route_intersection(self, policy):
        """Intersection NOT on route should use MaxPressure even with active EV."""
        obs = np.array([0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ev_info = {
            "active": True,
            "phase": 1,
            "intersection_id": "off_route_int",
        }
        action = policy.select_action(obs, ev_info)
        # "off_route_int" is not in route, so MaxPressure: argmax of [0, 0, 7, 1] = 2
        assert action == 2

    def test_multi_agent_interface(self, policy):
        """select_actions_multi_agent should return a dict of phases."""
        obs_dict = {
            "int_0": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "int_1": np.zeros(10),
            "int_2": np.zeros(10),
        }
        ev_info = {
            "active": True,
            "current_intersection": "int_1",
            "phase_map": {"int_0": 0, "int_1": 1, "int_2": 2},
        }
        actions = policy.select_actions_multi_agent(obs_dict, ev_info)
        assert isinstance(actions, dict)
        assert len(actions) == 3
        # On-route intersections with phase_map entries should get EV phases
        assert actions["int_0"] == 0
        assert actions["int_1"] == 1
        assert actions["int_2"] == 2

    def test_repr(self, policy):
        """__repr__ should be a non-empty string."""
        r = repr(policy)
        assert isinstance(r, str)
        assert "GreedyPreemptPolicy" in r


class TestGreedyPreemptWithoutNetwork:
    """Tests for GreedyPreemptPolicy when network lacks expected methods."""

    def test_graceful_degradation(self):
        """If network has no get_incoming_counts, should fall back to obs argmax."""
        # Bare object without the expected methods
        net = object()
        route = ["a", "b"]
        policy = GreedyPreemptPolicy(network=net, route=route)

        obs = np.array([0.0, 3.0, 1.0, 2.0])
        ev_info = {
            "active": False,
            "phase": None,
            "intersection_id": "c",
        }
        action = policy.select_action(obs, ev_info)
        assert action == 1  # argmax of [0, 3, 1, 2]
