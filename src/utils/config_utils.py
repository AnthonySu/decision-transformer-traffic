"""Configuration management for EV-DT experiments.

Provides helpers for loading YAML configs, merging overrides from
multiple sources (file + CLI), flattening nested dicts for W&B logging,
and saving configs back to disk.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    logger.info("Loaded config from %s", path)
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (override takes precedence).

    For nested dicts the merge is recursive.  For all other types the
    value in *override* simply replaces the one in *base*.

    Parameters
    ----------
    base : dict
        Base configuration.
    override : dict
        Override values.

    Returns
    -------
    dict
        A new dict with the merged result (neither input is mutated).
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_with_overrides(
    base_path: str | Path,
    override_path: Optional[str | Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Load a base config, optionally merging overrides from file and CLI.

    Merge order (later sources take precedence):

    1. Base config file
    2. Override config file (if provided)
    3. CLI overrides dict (if provided)

    CLI overrides support dotted keys for nested access.  For example
    ``{"dt.lr": 1e-4}`` will set ``config["dt"]["lr"]``.

    Parameters
    ----------
    base_path : str | Path
        Path to the base YAML config.
    override_path : str | Path | None
        Optional path to a second YAML file whose values override *base*.
    cli_overrides : dict | None
        Flat dict of dotted-key overrides from the command line.

    Returns
    -------
    dict
        The fully merged configuration.
    """
    config = load_config(base_path)

    if override_path is not None:
        override_cfg = load_config(override_path)
        config = merge_configs(config, override_cfg)

    if cli_overrides:
        # Expand dotted keys into nested dicts, then merge
        nested = _dotted_to_nested(cli_overrides)
        config = merge_configs(config, nested)

    return config


def _dotted_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dict with dotted keys into a nested dict.

    Example::

        {"dt.lr": 1e-4, "dt.n_layers": 6}
        -> {"dt": {"lr": 1e-4, "n_layers": 6}}
    """
    nested: dict[str, Any] = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        target = nested
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return nested


def config_to_flat(config: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict into dotted-key form.

    Useful for logging hyper-parameters to W&B where nested structures
    are not natively supported in some views.

    Example::

        {"dt": {"n_layers": 4, "embed_dim": 128}}
        -> {"dt.n_layers": 4, "dt.embed_dim": 128}

    Parameters
    ----------
    config : dict
        Possibly nested configuration dictionary.
    prefix : str
        Key prefix (used during recursion; callers can leave empty).

    Returns
    -------
    dict[str, Any]
        Flat dictionary with dotted keys.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(config_to_flat(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def save_config(config: dict, path: str | Path) -> None:
    """Save a configuration dict to a YAML file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    config : dict
        Configuration to persist.
    path : str | Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Config saved to %s", path)
