"""Tests for src.envs.ev_tracker — standalone EV tracking module."""


from src.envs.ev_tracker import EVTracker
from src.envs.network_utils import build_grid_network, compute_shortest_path


def _make_tracker(rows=3, cols=3):
    net = build_grid_network(rows=rows, cols=cols)
    route = compute_shortest_path(net, "n0_0", f"n{rows-1}_{cols-1}")
    return EVTracker(net, route, speed_factor=1.5), net, route


class TestEVTracker:
    def test_init(self):
        tracker, _, route = _make_tracker()
        assert tracker.link_idx == 0
        assert tracker.progress == 0.0
        assert not tracker.arrived
        assert tracker.intersections_passed == 0

    def test_step_advances(self):
        tracker, _, _ = _make_tracker()
        info = tracker.step(dt=5.0)
        assert isinstance(info, dict)
        assert "speed" in info
        assert "blocked" in info

    def test_multiple_steps_progress(self):
        tracker, _, _ = _make_tracker()
        initial_pos = tracker.position_fraction
        for _ in range(20):
            tracker.step(dt=5.0)
            if tracker.arrived:
                break
        assert tracker.position_fraction > initial_pos or tracker.arrived

    def test_arrival(self):
        tracker, _, _ = _make_tracker()
        for _ in range(200):
            tracker.step(dt=5.0)
            if tracker.arrived:
                break
        assert tracker.arrived
        assert tracker.intersections_passed > 0

    def test_reset(self):
        tracker, _, _ = _make_tracker()
        for _ in range(10):
            tracker.step(dt=5.0)
        tracker.reset()
        assert tracker.link_idx == 0
        assert tracker.progress == 0.0
        assert not tracker.arrived

    def test_distance_to_intersection(self):
        tracker, _, route = _make_tracker()
        d0 = tracker.distance_to_intersection(0)
        d_last = tracker.distance_to_intersection(len(route) - 1)
        assert d0 <= d_last  # first intersection is closer

    def test_position_fraction_bounded(self):
        tracker, _, _ = _make_tracker()
        for _ in range(50):
            tracker.step(dt=5.0)
            assert 0.0 <= tracker.position_fraction <= 1.0
            if tracker.arrived:
                break
