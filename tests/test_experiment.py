"""Tests for src.utils.experiment — experiment management utilities."""

import numpy as np
import torch

from src.utils.experiment import ExperimentTracker, config_hash, set_seed


class TestSetSeed:
    def test_numpy_deterministic(self):
        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = np.random.randn(5)
        set_seed(123)
        b = np.random.randn(5)
        assert not np.array_equal(a, b)


class TestConfigHash:
    def test_deterministic(self):
        cfg = {"a": 1, "b": "hello"}
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2

    def test_different_configs_differ(self):
        h1 = config_hash({"a": 1})
        h2 = config_hash({"a": 2})
        assert h1 != h2

    def test_returns_string(self):
        h = config_hash({"x": [1, 2, 3]})
        assert isinstance(h, str)
        assert len(h) > 0


class TestExperimentTracker:
    def test_create(self, tmp_path):
        tracker = ExperimentTracker("test_exp", log_dir=str(tmp_path))
        assert tracker.name == "test_exp"

    def test_log_epoch(self, tmp_path):
        tracker = ExperimentTracker("test", log_dir=str(tmp_path))
        tracker.log_epoch(1, {"loss": 0.5, "reward": -100})
        tracker.log_epoch(2, {"loss": 0.3, "reward": -80})
        assert len(tracker.metrics_history) == 2

    def test_get_best(self, tmp_path):
        tracker = ExperimentTracker("test", log_dir=str(tmp_path))
        tracker.log_epoch(1, {"loss": 0.5})
        tracker.log_epoch(2, {"loss": 0.3})
        tracker.log_epoch(3, {"loss": 0.4})
        best = tracker.get_best("loss", mode="min")
        assert best["loss"] == 0.3

    def test_save_and_load(self, tmp_path):
        tracker = ExperimentTracker("test", log_dir=str(tmp_path))
        tracker.log_epoch(1, {"loss": 0.5})
        tracker.save()

        tracker2 = ExperimentTracker("test", log_dir=str(tmp_path))
        loaded = tracker2.load()
        assert loaded
        assert len(tracker2.metrics_history) == 1
