"""Tests for src.utils.metrics — evaluation metrics and aggregation."""

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    aggregate_metrics,
    compare_methods,
    compute_background_delay,
    compute_corridor_green_ratio,
    compute_ev_travel_time,
    compute_signal_disruptions,
    compute_throughput,
)

# ======================================================================
# Helpers
# ======================================================================

def _make_step_infos(
    n_steps: int = 10,
    ev_travel_time: float = 50.0,
    background_delay: float = 2.5,
    throughput: float = 1.0,
    phase_changed: bool = False,
    ev_active: bool = True,
    corridor_green: bool = True,
) -> list:
    """Create a list of per-step info dicts for testing."""
    infos = []
    for i in range(n_steps):
        info = {
            "ev_info": {"active": ev_active},
            "background_delay": background_delay,
            "throughput": throughput,
            "phase_changed_for_ev": phase_changed,
            "corridor_green": corridor_green,
        }
        infos.append(info)
    # Add ev_travel_time to the last info dict
    infos[-1]["ev_travel_time"] = ev_travel_time
    return infos


# ======================================================================
# test_aggregate_metrics
# ======================================================================

class TestAggregateMetrics:
    """Tests for aggregate_metrics."""

    def test_correct_stat_computation(self):
        """Mean and std should be computed correctly."""
        # 3 episodes with known travel times
        ep1 = _make_step_infos(n_steps=5, ev_travel_time=10.0)
        ep2 = _make_step_infos(n_steps=5, ev_travel_time=20.0)
        ep3 = _make_step_infos(n_steps=5, ev_travel_time=30.0)

        result = aggregate_metrics([ep1, ep2, ep3])

        assert result["mean_ev_travel_time"] == pytest.approx(20.0)
        expected_std = float(np.std([10.0, 20.0, 30.0]))
        assert result["ev_travel_time_std"] == pytest.approx(expected_std)

    def test_num_episodes_correct(self):
        """num_episodes should match the number of inputs."""
        episodes = [_make_step_infos() for _ in range(7)]
        result = aggregate_metrics(episodes)
        assert result["num_episodes"] == 7

    def test_single_episode(self):
        """Should work with a single episode."""
        ep = _make_step_infos(n_steps=5, ev_travel_time=42.0)
        result = aggregate_metrics([ep])
        assert result["mean_ev_travel_time"] == pytest.approx(42.0)
        assert result["ev_travel_time_std"] == pytest.approx(0.0)
        assert result["num_episodes"] == 1

    def test_background_delay_computed(self):
        """Background delay mean should be computed."""
        ep = _make_step_infos(n_steps=10, background_delay=3.0)
        result = aggregate_metrics([ep])
        assert result["background_delay_mean"] == pytest.approx(3.0)

    def test_throughput_computed(self):
        """Throughput mean should be computed."""
        ep = _make_step_infos(n_steps=10, throughput=2.0)
        result = aggregate_metrics([ep])
        # Total throughput = 10 * 2.0 = 20.0 per episode; mean across 1 ep = 20.0
        assert result["throughput_mean"] == pytest.approx(20.0)

    def test_signal_disruptions_computed(self):
        """Signal disruptions should be counted."""
        ep = _make_step_infos(n_steps=10, phase_changed=True)
        result = aggregate_metrics([ep])
        assert result["signal_disruptions_mean"] == pytest.approx(10.0)

    def test_corridor_green_ratio_computed(self):
        """Corridor green ratio should be in [0, 1]."""
        ep = _make_step_infos(n_steps=10, ev_active=True, corridor_green=True)
        result = aggregate_metrics([ep])
        assert result["corridor_green_ratio_mean"] == pytest.approx(1.0)

    def test_all_expected_keys_present(self):
        """aggregate_metrics should return all expected metric keys."""
        ep = _make_step_infos()
        result = aggregate_metrics([ep])
        expected_keys = {
            "mean_ev_travel_time",
            "ev_travel_time_mean",
            "ev_travel_time_std",
            "mean_return",
            "mean_length",
            "background_delay_mean",
            "background_delay_std",
            "throughput_mean",
            "throughput_std",
            "signal_disruptions_mean",
            "signal_disruptions_std",
            "corridor_green_ratio_mean",
            "corridor_green_ratio_std",
            "num_episodes",
        }
        assert expected_keys.issubset(result.keys())

    def test_dict_episode_format(self):
        """Should accept list of dicts with 'step_infos' key."""
        step_infos = _make_step_infos(n_steps=5, ev_travel_time=15.0)
        ep_dicts = [
            {"step_infos": step_infos, "return": 100.0, "length": 5},
            {"step_infos": step_infos, "return": 200.0, "length": 5},
        ]
        result = aggregate_metrics(ep_dicts)
        assert result["mean_return"] == pytest.approx(150.0)
        assert result["mean_length"] == pytest.approx(5.0)


# ======================================================================
# test_compare_methods
# ======================================================================

class TestCompareMethods:
    """Tests for compare_methods."""

    def test_dataframe_structure(self):
        """compare_methods should return a DataFrame indexed by method name."""
        ep1 = _make_step_infos(n_steps=5, ev_travel_time=10.0)
        ep2 = _make_step_infos(n_steps=5, ev_travel_time=20.0)

        results = {
            "DT": aggregate_metrics([ep1]),
            "MADT": aggregate_metrics([ep2]),
            "FixedTime": aggregate_metrics([ep1, ep2]),
        }
        df = compare_methods(results)

        assert isinstance(df, pd.DataFrame)
        assert "DT" in df.index
        assert "MADT" in df.index
        assert "FixedTime" in df.index

    def test_dataframe_columns(self):
        """DataFrame should contain metric columns."""
        ep = _make_step_infos()
        results = {"method_A": aggregate_metrics([ep])}
        df = compare_methods(results)

        assert "mean_ev_travel_time" in df.columns
        assert "background_delay_mean" in df.columns

    def test_dataframe_values(self):
        """Values in the DataFrame should match the input metrics."""
        ep = _make_step_infos(n_steps=5, ev_travel_time=42.0)
        results = {"test_method": aggregate_metrics([ep])}
        df = compare_methods(results)

        assert df.loc["test_method", "mean_ev_travel_time"] == pytest.approx(42.0)

    def test_single_method(self):
        """Should work with just one method."""
        ep = _make_step_infos()
        results = {"only_method": aggregate_metrics([ep])}
        df = compare_methods(results)
        assert len(df) == 1

    def test_empty_dict(self):
        """Empty input should produce an empty DataFrame."""
        df = compare_methods({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ======================================================================
# Individual metric function tests
# ======================================================================

class TestPerEpisodeMetrics:
    """Tests for individual metric computation functions."""

    def test_ev_travel_time_from_last_info(self):
        """Should extract ev_travel_time from the last info dict."""
        infos = [{"foo": "bar"}, {"ev_travel_time": 25.0}]
        assert compute_ev_travel_time(infos) == pytest.approx(25.0)

    def test_ev_travel_time_empty_list(self):
        """Empty list should return -1.0."""
        assert compute_ev_travel_time([]) == -1.0

    def test_ev_travel_time_fallback_to_active_count(self):
        """If no ev_travel_time key, should count active steps."""
        infos = [
            {"ev_info": {"active": True}},
            {"ev_info": {"active": True}},
            {"ev_info": {"active": False}},
        ]
        assert compute_ev_travel_time(infos) == pytest.approx(2.0)

    def test_background_delay_mean(self):
        """Should compute mean of background_delay values."""
        infos = [
            {"background_delay": 1.0},
            {"background_delay": 3.0},
            {"background_delay": 5.0},
        ]
        assert compute_background_delay(infos) == pytest.approx(3.0)

    def test_background_delay_empty(self):
        """No delay data should return 0.0."""
        assert compute_background_delay([{"foo": 1}]) == 0.0

    def test_throughput_sums(self):
        """Should sum throughput across all steps."""
        infos = [{"throughput": 3.0}, {"throughput": 2.0}, {"throughput": 5.0}]
        assert compute_throughput(infos) == pytest.approx(10.0)

    def test_signal_disruptions_count(self):
        """Should count steps where phase_changed_for_ev is True."""
        infos = [
            {"phase_changed_for_ev": True},
            {"phase_changed_for_ev": False},
            {"phase_changed_for_ev": True},
            {},
        ]
        assert compute_signal_disruptions(infos) == 2

    def test_corridor_green_ratio(self):
        """Should compute fraction of green steps while EV active."""
        infos = [
            {"ev_info": {"active": True}, "corridor_green": True},
            {"ev_info": {"active": True}, "corridor_green": False},
            {"ev_info": {"active": True}, "corridor_green": True},
            {"ev_info": {"active": False}, "corridor_green": True},  # inactive, ignored
        ]
        assert compute_corridor_green_ratio(infos) == pytest.approx(2.0 / 3.0)

    def test_corridor_green_ratio_no_active(self):
        """If EV never active, should return 0.0."""
        infos = [{"ev_info": {"active": False}, "corridor_green": True}]
        assert compute_corridor_green_ratio(infos) == 0.0
