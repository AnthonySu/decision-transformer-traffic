"""Experiment management: reproducibility, checkpointing, config tracking.

Provides helpers for setting random seeds, saving / loading training
checkpoints, and tracking experiment metrics over time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        The seed value.  Applied to Python's ``random``, NumPy, and
        PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: dict[str, Any],
    path: str | Path,
    config: Optional[dict] = None,
) -> None:
    """Save a full training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose ``state_dict`` will be saved.
    optimizer : torch.optim.Optimizer
        Optimiser state to save.
    scheduler : lr_scheduler or None
        Learning-rate scheduler state (may be ``None``).
    epoch : int
        Current epoch number.
    metrics : dict
        Metric snapshot at the time of checkpointing.
    path : str | Path
        Destination file path.
    config : dict | None
        Optional experiment config dict to embed in the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore model / optimiser / scheduler state.

    Parameters
    ----------
    path : str | Path
        Path to the saved checkpoint file.
    model : torch.nn.Module
        Model instance whose weights will be loaded.
    optimizer : torch.optim.Optimizer | None
        If provided, optimiser state is restored.
    scheduler : lr_scheduler | None
        If provided, scheduler state is restored.
    device : str
        Device to map tensors onto (e.g. ``"cpu"`` or ``"cuda"``).

    Returns
    -------
    dict
        The full checkpoint dict (includes ``epoch``, ``metrics``, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Checkpoint loaded from %s (epoch %d)",
        path,
        checkpoint.get("epoch", -1),
    )
    return checkpoint


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def config_hash(config: dict) -> str:
    """Generate a short deterministic hash of *config*.

    Useful for tagging experiment runs so that identical configs map to
    the same identifier.

    Parameters
    ----------
    config : dict
        Experiment configuration (must be JSON-serialisable).

    Returns
    -------
    str
        An 8-character hexadecimal hash string.
    """
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Track experiment progress and results on disk.

    Saves periodic snapshots of metrics to a JSON file, enabling
    resume-from-checkpoint functionality and post-hoc analysis.

    Parameters
    ----------
    experiment_name : str
        Unique name for the experiment.
    log_dir : str
        Directory where the metrics JSON file will be written.
    """

    def __init__(self, experiment_name: str, log_dir: str = "logs") -> None:
        self.name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: list[dict[str, Any]] = []
        self.start_time: float = time.time()

    # ---- core API --------------------------------------------------------

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Record metrics for one epoch.

        Parameters
        ----------
        epoch : int
            The epoch number.
        metrics : dict
            Metric name -> value mapping for this epoch.
        """
        entry = {
            "epoch": epoch,
            "wall_time": time.time() - self.start_time,
            **metrics,
        }
        self.metrics_history.append(entry)

    def get_best(self, metric: str, mode: str = "min") -> dict[str, Any]:
        """Return the snapshot with the best value for *metric*.

        Parameters
        ----------
        metric : str
            Name of the metric to optimise.
        mode : str
            ``"min"`` or ``"max"``.

        Returns
        -------
        dict
            The full metrics snapshot from the best epoch, or an empty
            dict if no history exists.
        """
        if not self.metrics_history:
            return {}
        key_fn = min if mode == "min" else max
        valid = [m for m in self.metrics_history if metric in m]
        if not valid:
            return {}
        return key_fn(valid, key=lambda m: m[metric])

    # ---- persistence -----------------------------------------------------

    @property
    def _file_path(self) -> Path:
        return self.log_dir / f"{self.name}_metrics.json"

    def save(self) -> None:
        """Persist metrics history to disk as JSON."""
        with open(self._file_path, "w") as f:
            json.dump(
                {
                    "experiment_name": self.name,
                    "start_time": self.start_time,
                    "metrics_history": self.metrics_history,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info("Metrics saved to %s", self._file_path)

    def load(self) -> bool:
        """Load existing metrics history from disk.

        Returns
        -------
        bool
            ``True`` if a file was found and loaded, ``False`` otherwise.
        """
        if not self._file_path.exists():
            return False
        with open(self._file_path, "r") as f:
            data = json.load(f)
        self.metrics_history = data.get("metrics_history", [])
        self.start_time = data.get("start_time", self.start_time)
        logger.info(
            "Loaded %d metric snapshots from %s",
            len(self.metrics_history),
            self._file_path,
        )
        return True

    # ---- reporting -------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of the experiment.

        Returns
        -------
        str
            Multi-line string with experiment stats.
        """
        elapsed = time.time() - self.start_time
        n_epochs = len(self.metrics_history)
        lines = [
            f"Experiment : {self.name}",
            f"Epochs     : {n_epochs}",
            f"Wall time  : {elapsed:.1f}s",
        ]
        if n_epochs > 0:
            last = self.metrics_history[-1]
            lines.append(f"Last epoch : {last.get('epoch', '?')}")
            metric_keys = [k for k in last if k not in ("epoch", "wall_time")]
            for k in sorted(metric_keys):
                lines.append(f"  {k}: {last[k]}")
        return "\n".join(lines)
