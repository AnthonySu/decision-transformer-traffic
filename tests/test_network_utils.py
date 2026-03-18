"""Tests for src.envs.network_utils — network construction and CTM simulation."""

import numpy as np
import pytest

from src.envs.network_utils import (
    build_grid_network,
    compute_shortest_path,
    ctm_step,
    get_total_queue_length,
    random_od_pair,
    reset_densities,
    signal_is_green_for_link,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def small_network():
    """A small 3x3 grid network for fast tests."""
    return build_grid_network(rows=3, cols=3)


@pytest.fixture
def default_network():
    """The default 4x4 grid network."""
    return build_grid_network()


# ======================================================================
# test_build_grid_network
# ======================================================================

class TestBuildGridNetwork:
    """Tests for build_grid_network."""

    def test_node_count(self, small_network):
        """A 3x3 grid should have exactly 9 nodes."""
        assert len(small_network["nodes"]) == 9

    def test_node_count_4x4(self, default_network):
        """A 4x4 grid should have exactly 16 nodes."""
        assert len(default_network["nodes"]) == 16

    def test_link_count_3x3(self, small_network):
        """A 3x3 grid should have 2 * (rows*(cols-1) + cols*(rows-1)) directed links."""
        # Horizontal edges: rows * (cols-1) = 3*2 = 6, bidirectional = 12
        # Vertical edges: cols * (rows-1) = 3*2 = 6, bidirectional = 12
        # Total: 24
        assert len(small_network["links"]) == 24

    def test_boundary_nodes_count(self, small_network):
        """In a 3x3 grid, all 8 perimeter nodes are boundary (centre is not)."""
        boundary = [
            nid for nid, n in small_network["nodes"].items() if n["is_boundary"]
        ]
        assert len(boundary) == 8

    def test_center_node_not_boundary(self, small_network):
        """The centre node of a 3x3 grid should NOT be a boundary node."""
        center = small_network["nodes"]["n1_1"]
        assert center["is_boundary"] is False

    def test_corner_is_boundary(self, small_network):
        """Corner nodes must be boundary nodes."""
        for nid in ["n0_0", "n0_2", "n2_0", "n2_2"]:
            assert small_network["nodes"][nid]["is_boundary"] is True

    def test_link_properties(self, small_network):
        """Every link should carry the expected default properties."""
        for lid, lk in small_network["links"].items():
            assert lk["length"] == 200.0
            assert lk["num_lanes"] == 2
            assert lk["v_free"] == 15.0
            assert lk["density"] == 0.0
            assert lk["flow"] == 0.0
            assert lk["direction"] in ("N", "S", "E", "W")
            assert lk["phase_index"] in (0, 1)

    def test_link_source_target_valid(self, small_network):
        """Each link's source and target must be valid node IDs."""
        nodes = small_network["nodes"]
        for lid, lk in small_network["links"].items():
            assert lk["source"] in nodes
            assert lk["target"] in nodes
            assert lk["source"] != lk["target"]

    def test_graph_edge_count_matches_links(self, small_network):
        """The networkx graph should have the same number of edges as links."""
        assert small_network["graph"].number_of_edges() == len(small_network["links"])

    def test_graph_node_count_matches_nodes(self, small_network):
        """The networkx graph should have the same number of nodes."""
        assert small_network["graph"].number_of_nodes() == len(small_network["nodes"])

    def test_custom_parameters(self):
        """build_grid_network should respect custom parameters."""
        net = build_grid_network(rows=2, cols=2, link_length=100.0, num_lanes=3)
        assert net["rows"] == 2
        assert net["cols"] == 2
        for lk in net["links"].values():
            assert lk["length"] == 100.0
            assert lk["num_lanes"] == 3

    def test_incoming_outgoing_links_populated(self, small_network):
        """Every node should have its incoming/outgoing link lists populated."""
        for nid, node in small_network["nodes"].items():
            # Interior nodes have 4 incoming and 4 outgoing; boundary have fewer
            assert isinstance(node["incoming_links"], list)
            assert isinstance(node["outgoing_links"], list)
            # At least 2 links for corner nodes
            if node["is_boundary"]:
                assert len(node["incoming_links"]) >= 2
                assert len(node["outgoing_links"]) >= 2


# ======================================================================
# test_compute_shortest_path
# ======================================================================

class TestComputeShortestPath:
    """Tests for compute_shortest_path."""

    def test_path_exists(self, small_network):
        """A valid path should be returned between connected nodes."""
        route = compute_shortest_path(small_network, "n0_0", "n2_2")
        assert len(route) > 0

    def test_correct_start_end(self, small_network):
        """The path should start at origin and end at destination."""
        route = compute_shortest_path(small_network, "n0_0", "n2_2")
        assert route[0][0] == "n0_0"
        assert route[-1][0] == "n2_2"

    def test_last_entry_has_none_link(self, small_network):
        """The final entry in the route should have outgoing_link_id = None."""
        route = compute_shortest_path(small_network, "n0_0", "n2_2")
        assert route[-1][1] is None

    def test_all_intermediate_edges_valid(self, small_network):
        """All non-terminal entries should reference valid link IDs."""
        route = compute_shortest_path(small_network, "n0_0", "n2_2")
        links = small_network["links"]
        for node_id, link_id in route[:-1]:
            assert link_id is not None
            assert link_id in links
            assert links[link_id]["source"] == node_id

    def test_consecutive_nodes_connected(self, small_network):
        """Consecutive nodes in the path should be graph neighbors."""
        route = compute_shortest_path(small_network, "n0_0", "n2_2")
        G = small_network["graph"]
        for i in range(len(route) - 1):
            src = route[i][0]
            dst = route[i + 1][0]
            assert G.has_edge(src, dst)

    def test_no_path_raises(self):
        """Requesting a path to a non-existent node should raise ValueError."""
        net = build_grid_network(rows=2, cols=2)
        with pytest.raises((ValueError, KeyError)):
            compute_shortest_path(net, "n0_0", "nonexistent")

    def test_adjacent_nodes(self, small_network):
        """Path between adjacent nodes should have exactly 2 entries."""
        route = compute_shortest_path(small_network, "n0_0", "n0_1")
        assert len(route) == 2


# ======================================================================
# test_random_od_pair
# ======================================================================

class TestRandomOdPair:
    """Tests for random_od_pair."""

    def test_both_are_boundary_nodes(self, small_network):
        """Both origin and destination should be boundary nodes."""
        rng = np.random.default_rng(42)
        origin, destination = random_od_pair(small_network, rng=rng)
        assert small_network["nodes"][origin]["is_boundary"]
        assert small_network["nodes"][destination]["is_boundary"]

    def test_distinct_origin_destination(self, small_network):
        """Origin and destination should not be the same node."""
        rng = np.random.default_rng(42)
        origin, destination = random_od_pair(small_network, rng=rng)
        assert origin != destination

    def test_path_exists_between_od(self, small_network):
        """There should be a valid path between the OD pair."""
        rng = np.random.default_rng(42)
        origin, destination = random_od_pair(small_network, rng=rng)
        route = compute_shortest_path(small_network, origin, destination)
        assert len(route) >= 2

    def test_returns_strings(self, small_network):
        """OD pair should be returned as strings."""
        rng = np.random.default_rng(42)
        origin, destination = random_od_pair(small_network, rng=rng)
        assert isinstance(origin, str)
        assert isinstance(destination, str)

    def test_reproducible_with_seed(self, small_network):
        """Same seed should produce the same OD pair."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        od1 = random_od_pair(small_network, rng=rng1)
        od2 = random_od_pair(small_network, rng=rng2)
        assert od1 == od2

    def test_default_rng(self, small_network):
        """Calling without rng should still work (uses default)."""
        origin, destination = random_od_pair(small_network)
        assert origin in small_network["nodes"]
        assert destination in small_network["nodes"]


# ======================================================================
# test_ctm_step
# ======================================================================

class TestCtmStep:
    """Tests for ctm_step (Cell Transmission Model)."""

    def test_densities_update(self, small_network):
        """After a CTM step, at least some densities should change."""
        reset_densities(small_network, base_density=0.05)
        old_densities = {
            lid: lk["density"] for lid, lk in small_network["links"].items()
        }
        ctm_step(small_network, dt=5.0)
        new_densities = {
            lid: lk["density"] for lid, lk in small_network["links"].items()
        }
        # At least one density should differ (due to CTM dynamics + boundary demand)
        changed = any(
            abs(old_densities[lid] - new_densities[lid]) > 1e-12
            for lid in old_densities
        )
        assert changed, "No densities changed after CTM step"

    def test_no_negative_densities(self, small_network):
        """After multiple CTM steps, no density should be negative."""
        reset_densities(small_network, base_density=0.03)
        for _ in range(50):
            ctm_step(small_network, dt=5.0)
        for lid, lk in small_network["links"].items():
            assert lk["density"] >= 0.0, f"Negative density on link {lid}"

    def test_densities_bounded_by_kjam(self, small_network):
        """Densities should never exceed k_jam."""
        reset_densities(small_network, base_density=0.1)
        for _ in range(100):
            ctm_step(small_network, dt=5.0)
        for lid, lk in small_network["links"].items():
            assert lk["density"] <= lk["k_jam"] + 1e-9, (
                f"Density {lk['density']} exceeds k_jam {lk['k_jam']} on link {lid}"
            )

    def test_returns_flow_dict(self, small_network):
        """ctm_step should return a dict mapping link IDs to inflow values."""
        reset_densities(small_network, base_density=0.02)
        flow_in = ctm_step(small_network, dt=5.0)
        assert isinstance(flow_in, dict)
        assert len(flow_in) == len(small_network["links"])
        for lid, flow_val in flow_in.items():
            assert lid in small_network["links"]
            assert isinstance(flow_val, float)
            assert flow_val >= 0.0

    def test_flow_conservation_approximate(self, small_network):
        """Total flow in the network should be non-negative (no flow destroyed)."""
        reset_densities(small_network, base_density=0.03)
        flow_in = ctm_step(small_network, dt=5.0)
        total_flow = sum(flow_in.values())
        assert total_flow >= 0.0


# ======================================================================
# test_reset_densities
# ======================================================================

class TestResetDensities:
    """Tests for reset_densities."""

    def test_all_links_reset_to_base(self, small_network):
        """After reset, all densities should equal the base value."""
        reset_densities(small_network, base_density=0.05)
        for lk in small_network["links"].values():
            assert lk["density"] == pytest.approx(0.05)

    def test_all_flows_reset_to_zero(self, small_network):
        """After reset, all flows should be zero."""
        # Run some steps first to accumulate flows
        reset_densities(small_network, base_density=0.03)
        ctm_step(small_network, dt=5.0)
        # Now reset
        reset_densities(small_network, base_density=0.02)
        for lk in small_network["links"].values():
            assert lk["flow"] == 0.0

    def test_default_base_density(self, small_network):
        """Default base_density should be 0.02."""
        reset_densities(small_network)
        for lk in small_network["links"].values():
            assert lk["density"] == pytest.approx(0.02)


# ======================================================================
# test_get_total_queue_length
# ======================================================================

class TestGetTotalQueueLength:
    """Tests for get_total_queue_length."""

    def test_zero_density_gives_zero(self, small_network):
        """When all densities are zero, total queue length should be zero."""
        reset_densities(small_network, base_density=0.0)
        total = get_total_queue_length(small_network)
        assert total == pytest.approx(0.0)

    def test_computation_correctness(self, small_network):
        """Manual computation should match the function result."""
        reset_densities(small_network, base_density=0.05)
        expected = sum(
            lk["density"] * lk["length"] * lk["num_lanes"]
            for lk in small_network["links"].values()
        )
        actual = get_total_queue_length(small_network)
        assert actual == pytest.approx(expected)

    def test_positive_for_nonzero_density(self, small_network):
        """Non-zero density should produce a positive total queue length."""
        reset_densities(small_network, base_density=0.03)
        assert get_total_queue_length(small_network) > 0.0

    def test_scales_with_density(self, small_network):
        """Doubling density should double the queue length."""
        reset_densities(small_network, base_density=0.02)
        q1 = get_total_queue_length(small_network)
        reset_densities(small_network, base_density=0.04)
        q2 = get_total_queue_length(small_network)
        assert q2 == pytest.approx(2.0 * q1)


# ======================================================================
# test_signal_is_green_for_link
# ======================================================================

class TestSignalIsGreenForLink:
    """Tests for signal_is_green_for_link."""

    def test_matching_phase_returns_true(self, small_network):
        """If the node's phase matches the link's phase_index, green is True."""
        # Find a link and set the node phase to match
        for lid, lk in small_network["links"].items():
            target_node = lk["target"]
            small_network["nodes"][target_node]["current_phase"] = lk["phase_index"]
            assert signal_is_green_for_link(small_network, target_node, lid) is True
            break  # one check is sufficient

    def test_non_matching_phase_returns_false(self, small_network):
        """If the node's phase does NOT match the link's phase_index, green is False."""
        for lid, lk in small_network["links"].items():
            target_node = lk["target"]
            # Set phase to something that doesn't match
            wrong_phase = (lk["phase_index"] + 1) % 4
            small_network["nodes"][target_node]["current_phase"] = wrong_phase
            assert signal_is_green_for_link(small_network, target_node, lid) is False
            break

    def test_ns_links_green_on_phase_0(self, small_network):
        """N/S direction links should be green when node phase is 0."""
        for lid, lk in small_network["links"].items():
            if lk["direction"] in ("N", "S"):
                target = lk["target"]
                small_network["nodes"][target]["current_phase"] = 0
                assert signal_is_green_for_link(small_network, target, lid) is True
                small_network["nodes"][target]["current_phase"] = 1
                assert signal_is_green_for_link(small_network, target, lid) is False
                break

    def test_ew_links_green_on_phase_1(self, small_network):
        """E/W direction links should be green when node phase is 1."""
        for lid, lk in small_network["links"].items():
            if lk["direction"] in ("E", "W"):
                target = lk["target"]
                small_network["nodes"][target]["current_phase"] = 1
                assert signal_is_green_for_link(small_network, target, lid) is True
                small_network["nodes"][target]["current_phase"] = 0
                assert signal_is_green_for_link(small_network, target, lid) is False
                break
