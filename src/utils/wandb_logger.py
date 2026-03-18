"""Weights & Biases logging for EV-DT experiments.

Provides a unified interface for logging training metrics, evaluation
results, model checkpoints, and figures across all methods.  Falls back
gracefully when ``wandb`` is not installed or logging is disabled via
config.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional wandb import
# ---------------------------------------------------------------------------
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandbLogger:
    """Unified W&B logger for the EV-DT project.

    Handles initialisation, metric logging, artifact saving, and graceful
    fallback when ``wandb`` is not installed or logging is explicitly
    disabled.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str | None
        Human-readable run name.  ``None`` lets W&B auto-generate one.
    config : dict | None
        Hyper-parameter / experiment configuration to record.
    tags : list[str] | None
        Tags attached to the run for filtering in the W&B UI.
    enabled : bool
        Master switch.  When ``False`` every logging call is a no-op.
    group : str | None
        Optional W&B run group (useful for sweep / multi-seed grouping).
    notes : str | None
        Free-form notes visible in the W&B dashboard.
    """

    def __init__(
        self,
        project: str = "ev-decision-transformer",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        enabled: bool = True,
        group: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._enabled = enabled and _WANDB_AVAILABLE
        self._run = None

        if enabled and not _WANDB_AVAILABLE:
            warnings.warn(
                "wandb is not installed; logging will be disabled.  "
                "Install with:  pip install wandb",
                stacklevel=2,
            )

        if self._enabled:
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                tags=tags,
                group=group,
                notes=notes,
                reinit=True,
            )
            logger.info("W&B run initialised: %s", self._run.url)

    # ------------------------------------------------------------------
    # Scalar / dict logging
    # ------------------------------------------------------------------
    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalar metrics.

        Parameters
        ----------
        metrics : dict[str, Any]
            Mapping of metric names to values (int, float, or str).
        step : int | None
            Global step.  When ``None`` W&B auto-increments.
        """
        if not self._enabled:
            return
        wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Table logging
    # ------------------------------------------------------------------
    def log_table(self, key: str, columns: list[str], data: list[list[Any]]) -> None:
        """Log a W&B table (e.g. evaluation results across targets).

        Parameters
        ----------
        key : str
            Name under which the table appears in the dashboard.
        columns : list[str]
            Column headers.
        data : list[list[Any]]
            Row data – each inner list corresponds to one row.
        """
        if not self._enabled:
            return
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})

    # ------------------------------------------------------------------
    # Figure logging
    # ------------------------------------------------------------------
    def log_figure(self, key: str, figure: Any) -> None:
        """Log a *matplotlib* figure.

        Parameters
        ----------
        key : str
            Name under which the figure appears in the dashboard.
        figure : matplotlib.figure.Figure
            The figure object to log.
        """
        if not self._enabled:
            return
        wandb.log({key: wandb.Image(figure)})

    # ------------------------------------------------------------------
    # Model / artifact logging
    # ------------------------------------------------------------------
    def log_model(
        self,
        path: str,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save a model checkpoint as a W&B artifact.

        Parameters
        ----------
        path : str
            Local path to the model file (e.g. ``models/dt_best.pt``).
        name : str
            Artifact name in W&B (e.g. ``dt-best``).
        metadata : dict | None
            Extra metadata attached to the artifact.
        """
        if not self._enabled:
            return
        artifact = wandb.Artifact(name, type="model", metadata=metadata or {})
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        logger.info("Logged model artifact '%s' from %s", name, path)

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------
    def log_dataset_stats(self, stats: dict[str, Any]) -> None:
        """Log dataset statistics as a summary entry.

        Parameters
        ----------
        stats : dict[str, Any]
            Arbitrary dataset statistics (sizes, feature dims, etc.).
        """
        if not self._enabled:
            return
        for key, value in stats.items():
            wandb.run.summary[f"dataset/{key}"] = value

    # ------------------------------------------------------------------
    # Per-episode evaluation logging
    # ------------------------------------------------------------------
    def log_episode(
        self,
        episode_info: dict[str, Any],
        prefix: str = "eval",
        step: Optional[int] = None,
    ) -> None:
        """Log a single evaluation episode's metrics.

        Parameters
        ----------
        episode_info : dict[str, Any]
            Keys such as ``return``, ``length``, ``ev_travel_time``, etc.
        prefix : str
            Namespace prefix for the logged metrics.
        step : int | None
            Global step.  ``None`` lets W&B auto-increment.
        """
        if not self._enabled:
            return
        prefixed = {f"{prefix}/{k}": v for k, v in episode_info.items()}
        wandb.log(prefixed, step=step)

    # ------------------------------------------------------------------
    # Cross-method comparison table
    # ------------------------------------------------------------------
    def log_comparison(self, results_dict: dict[str, dict[str, Any]]) -> None:
        """Log cross-method comparison as a W&B table.

        Parameters
        ----------
        results_dict : dict[str, dict[str, Any]]
            Outer key is the method name (e.g. ``"DT"``, ``"MADT"``).
            Inner dict holds metric name -> value pairs.
        """
        if not self._enabled:
            return

        # Collect all metric names across methods
        all_metrics: set[str] = set()
        for metrics in results_dict.values():
            all_metrics.update(metrics.keys())
        metric_names = sorted(all_metrics)

        columns = ["method"] + metric_names
        data = []
        for method, metrics in results_dict.items():
            row = [method] + [metrics.get(m, None) for m in metric_names]
            data.append(row)

        self.log_table("comparison", columns, data)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def finish(self) -> None:
        """Close the W&B run."""
        if self._enabled and self._run is not None:
            self._run.finish()
            self._run = None
            logger.info("W&B run finished.")

    @property
    def is_enabled(self) -> bool:
        """Whether W&B logging is active."""
        return self._enabled

    # Allow use as a context manager
    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_logger(config: dict) -> WandbLogger:
    """Create a :class:`WandbLogger` from an experiment config dict.

    Expects the config to optionally contain a ``logging`` section::

        logging:
          use_wandb: true
          project: ev-decision-transformer
          run_name: my-run
          tags: [dt, seed42]
          group: sweep-1
          notes: "first try"

    Parameters
    ----------
    config : dict
        Full experiment configuration (will be passed through to W&B as
        hyper-parameters).

    Returns
    -------
    WandbLogger
    """
    log_cfg = config.get("logging", {})
    return WandbLogger(
        project=log_cfg.get("project", "ev-decision-transformer"),
        run_name=log_cfg.get("run_name", None),
        config=config,
        tags=log_cfg.get("tags", None),
        enabled=log_cfg.get("use_wandb", True),
        group=log_cfg.get("group", None),
        notes=log_cfg.get("notes", None),
    )
