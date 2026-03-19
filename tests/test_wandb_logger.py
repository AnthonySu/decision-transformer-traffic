"""Tests for WandbLogger — verifies no-op behaviour when disabled or wandb missing."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from src.utils.wandb_logger import WandbLogger, create_logger

# ---------------------------------------------------------------------------
# Disabled logger (enabled=False) should be a complete no-op
# ---------------------------------------------------------------------------

class TestWandbLoggerDisabled:
    """All logging calls should silently do nothing when enabled=False."""

    def setup_method(self):
        self.logger = WandbLogger(enabled=False)

    def test_is_not_enabled(self):
        assert not self.logger.is_enabled

    def test_log_noop(self):
        self.logger.log({"loss": 0.5}, step=1)

    def test_log_table_noop(self):
        self.logger.log_table("t", ["a"], [[1]])

    def test_log_figure_noop(self):
        self.logger.log_figure("fig", MagicMock())

    def test_log_model_noop(self):
        self.logger.log_model("/tmp/model.pt", "model")

    def test_log_dataset_stats_noop(self):
        self.logger.log_dataset_stats({"size": 100})

    def test_log_episode_noop(self):
        self.logger.log_episode({"return": 5.0})

    def test_log_comparison_noop(self):
        self.logger.log_comparison({"DT": {"loss": 0.1}})

    def test_finish_noop(self):
        self.logger.finish()

    def test_context_manager(self):
        with WandbLogger(enabled=False) as lg:
            lg.log({"x": 1})
        assert not lg.is_enabled


# ---------------------------------------------------------------------------
# Warning when wandb requested but not installed
# ---------------------------------------------------------------------------

class TestWandbMissingWarning:
    """When enabled=True but wandb is absent, a warning should be emitted."""

    def test_warns_when_wandb_missing(self):
        with patch("src.utils.wandb_logger._WANDB_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                logger = WandbLogger(enabled=True)
                assert not logger.is_enabled
                assert len(w) == 1
                assert "wandb is not installed" in str(w[0].message)


# ---------------------------------------------------------------------------
# create_logger factory
# ---------------------------------------------------------------------------

class TestCreateLogger:
    """create_logger should respect the config dict."""

    def test_disabled_via_config(self):
        lg = create_logger({"logging": {"use_wandb": False}})
        assert not lg.is_enabled

    def test_default_disabled_without_wandb(self):
        with patch("src.utils.wandb_logger._WANDB_AVAILABLE", False):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                lg = create_logger({})
                assert not lg.is_enabled

    def test_empty_config(self):
        """Empty config should not crash."""
        with patch("src.utils.wandb_logger._WANDB_AVAILABLE", False):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                lg = create_logger({})
                assert not lg.is_enabled
                lg.finish()
