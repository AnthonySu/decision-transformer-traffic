"""Tests for MaxPressurePolicy baseline."""

from src.baselines.max_pressure import MaxPressurePolicy
from src.envs.ev_corridor_env import EVCorridorEnv

# ======================================================================
# Unit tests
# ======================================================================


class TestMaxPressureSelectsHighDensityPhase:
    """MaxPressure should pick the phase with greatest upstream pressure."""

    def test_max_pressure_selects_high_density_phase(self):
        """Set up a network where one phase clearly has more upstream density
        and verify that MaxPressure picks that phase."""
        env = EVCorridorEnv(rows=4, cols=4, max_steps=50, seed=42)
        obs, _ = env.reset(seed=42)

        policy = MaxPressurePolicy(ev_priority=False)
        network = env.network

        # Pick the first route intersection and artificially inflate
        # density on incoming links that map to phase 1 (E/W).
        route_ints = [node for node, _ in env.ev_route]
        target_node_id = route_ints[1] if len(route_ints) > 1 else route_ints[0]
        node = network["nodes"][target_node_id]

        # Zero all incoming densities first, then spike phase-1 links
        for lid in node["incoming_links"]:
            network["links"][lid]["density"] = 0.0
        for lid in node["outgoing_links"]:
            network["links"][lid]["density"] = 0.0

        for lid in node["incoming_links"]:
            lk = network["links"][lid]
            if lk["phase_index"] == 1:
                lk["density"] = 0.12  # high density (near k_jam=0.15)

        chosen = policy._max_pressure_phase(network, target_node_id)
        assert chosen == 1, (
            f"Expected phase 1 (highest upstream density) but got {chosen}"
        )

    def test_picks_phase_0_when_ns_is_heavier(self):
        """When N/S incoming links are congested, phase 0 should win."""
        env = EVCorridorEnv(rows=4, cols=4, max_steps=50, seed=7)
        obs, _ = env.reset(seed=7)

        policy = MaxPressurePolicy(ev_priority=False)
        network = env.network

        route_ints = [node for node, _ in env.ev_route]
        # Find an interior intersection (has links from all directions)
        target_node_id = route_ints[min(2, len(route_ints) - 1)]
        node = network["nodes"][target_node_id]

        for lid in node["incoming_links"]:
            network["links"][lid]["density"] = 0.0
        for lid in node["outgoing_links"]:
            network["links"][lid]["density"] = 0.0

        for lid in node["incoming_links"]:
            lk = network["links"][lid]
            if lk["phase_index"] == 0:  # N/S
                lk["density"] = 0.10

        chosen = policy._max_pressure_phase(network, target_node_id)
        assert chosen == 0


class TestMaxPressureEVPriority:
    """When ev_priority=True, MaxPressure should force green for EV link."""

    def test_max_pressure_ev_priority(self):
        """The phase serving the EV link is forced when ev_priority=True."""
        env = EVCorridorEnv(rows=4, cols=4, max_steps=50, seed=42)
        obs, _ = env.reset(seed=42)

        policy = MaxPressurePolicy(ev_priority=True)
        network = env.network

        # Make sure the EV hasn't arrived and is on a valid link
        assert not env._ev_arrived
        assert env._ev_link_idx < len(env.ev_route) - 1

        _, ev_link_id = env.ev_route[env._ev_link_idx]
        assert ev_link_id is not None

        ev_link = network["links"][ev_link_id]
        ev_phase = ev_link["phase_index"]
        ev_target = ev_link["target"]

        # Zero out density on EV-phase incoming links and spike another
        # phase so that pure MaxPressure would NOT choose the EV phase.
        node = network["nodes"][ev_target]
        other_phase = 1 - ev_phase  # pick the opposite phase group
        for lid in node["incoming_links"]:
            lk = network["links"][lid]
            if lk["phase_index"] == other_phase:
                lk["density"] = 0.14  # very high
            else:
                lk["density"] = 0.0
        for lid in node["outgoing_links"]:
            network["links"][lid]["density"] = 0.0

        action = policy.act(obs, env)

        # Find the index of the EV's approaching intersection in the route
        route_ints = [n for n, _ in env.ev_route]
        ev_int_idx = route_ints.index(ev_target)

        assert action[ev_int_idx] == ev_phase, (
            f"EV priority should force phase {ev_phase} at intersection "
            f"{ev_target}, but got {action[ev_int_idx]}"
        )

    def test_no_ev_priority_uses_pressure(self):
        """With ev_priority=False, even the EV intersection uses pressure."""
        env = EVCorridorEnv(rows=4, cols=4, max_steps=50, seed=42)
        obs, _ = env.reset(seed=42)

        policy = MaxPressurePolicy(ev_priority=False)
        network = env.network

        _, ev_link_id = env.ev_route[env._ev_link_idx]
        ev_link = network["links"][ev_link_id]
        ev_phase = ev_link["phase_index"]
        ev_target = ev_link["target"]

        # Spike the OTHER phase so MaxPressure should NOT pick the EV phase
        node = network["nodes"][ev_target]
        other_phase = 1 - ev_phase
        for lid in node["incoming_links"]:
            lk = network["links"][lid]
            if lk["phase_index"] == other_phase:
                lk["density"] = 0.14
            else:
                lk["density"] = 0.0
        for lid in node["outgoing_links"]:
            network["links"][lid]["density"] = 0.0

        action = policy.act(obs, env)

        route_ints = [n for n, _ in env.ev_route]
        ev_int_idx = route_ints.index(ev_target)

        assert action[ev_int_idx] == other_phase, (
            f"Without EV priority, pressure should pick phase {other_phase} "
            f"but got {action[ev_int_idx]}"
        )


class TestMaxPressureEpisode:
    """Run a full episode with MaxPressure and verify it completes."""

    def test_max_pressure_episode(self):
        """A full episode should run without errors and return valid info."""
        env = EVCorridorEnv(rows=4, cols=4, max_steps=200, seed=123)
        obs, info = env.reset(seed=123)

        policy = MaxPressurePolicy(ev_priority=False)
        total_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = policy.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        assert steps > 0, "Episode had zero steps"
        assert steps <= 200, "Episode exceeded max_steps"
        assert "ev_arrived" in info
        assert "total_queue" in info

    def test_max_pressure_ev_priority_episode(self):
        """Full episode with ev_priority=True should also complete."""
        env = EVCorridorEnv(rows=4, cols=4, max_steps=200, seed=99)
        obs, info = env.reset(seed=99)

        policy = MaxPressurePolicy(ev_priority=True)
        done = False
        steps = 0

        while not done:
            action = policy.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        assert steps > 0

    def test_repr(self):
        """__repr__ should contain class name."""
        p = MaxPressurePolicy(ev_priority=True)
        assert "MaxPressurePolicy" in repr(p)
        assert "ev_priority=True" in repr(p)
