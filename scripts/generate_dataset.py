#!/usr/bin/env python
"""Generate an offline dataset for Decision Transformer training.

Reads configuration from a YAML file, creates the EV corridor environment,
runs the expert (GreedyPreempt) + random + suboptimal policies, and saves
the collected trajectories to HDF5.

Usage::

    python scripts/generate_dataset.py --config configs/default.yaml
    python scripts/generate_dataset.py --config configs/default.yaml --output data/my_dataset.h5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.greedy_preempt import GreedyPreemptPolicy  # noqa: E402
from src.envs import EVCorridorEnv, EVCorridorMAEnv  # noqa: E402
from src.utils.data_collector import DataCollector  # noqa: E402

# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML config.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Environment & policy factories
# ------------------------------------------------------------------

def create_environment(cfg: Dict[str, Any]) -> Any:
    """Instantiate the EV corridor environment from config.

    Parameters
    ----------
    cfg : dict
        The ``env`` section of the configuration.

    Returns
    -------
    EVCorridorEnv
        Configured environment instance.
    """
    return EVCorridorEnv(
        network=cfg.get("network", "grid-4x4-v0"),
        ev_speed_factor=cfg.get("ev_speed_factor", 1.5),
        ev_spawn_rate=cfg.get("ev_spawn_rate", 0.1),
        max_episode_steps=cfg.get("max_episode_steps", 300),
        decision_interval=cfg.get("decision_interval", 5),
        preemption_advance=cfg.get("preemption_advance", 3),
        background_demand_rate=cfg.get("background_demand_rate", 0.3),
    )


def create_ma_environment(cfg: Dict[str, Any]) -> Any:
    """Instantiate the multi-agent EV corridor environment from config.

    Parameters
    ----------
    cfg : dict
        The ``env`` section of the configuration.

    Returns
    -------
    EVCorridorMAEnv
        Configured multi-agent environment instance.
    """
    return EVCorridorMAEnv(
        network=cfg.get("network", "grid-4x4-v0"),
        ev_speed_factor=cfg.get("ev_speed_factor", 1.5),
        ev_spawn_rate=cfg.get("ev_spawn_rate", 0.1),
        max_episode_steps=cfg.get("max_episode_steps", 300),
        decision_interval=cfg.get("decision_interval", 5),
        preemption_advance=cfg.get("preemption_advance", 3),
        background_demand_rate=cfg.get("background_demand_rate", 0.3),
    )


def create_expert_policy(env: Any) -> GreedyPreemptPolicy:
    """Create the greedy-preempt expert policy.

    Parameters
    ----------
    env : EVCorridorEnv | EVCorridorMAEnv
        Environment (used to extract the network topology and default route).

    Returns
    -------
    GreedyPreemptPolicy
        Configured expert policy.
    """
    network = getattr(env, "network", env)
    route = getattr(env, "ev_route", getattr(env, "default_route", []))
    return GreedyPreemptPolicy(network=network, route=route)


# ------------------------------------------------------------------
# Episode count helpers
# ------------------------------------------------------------------

def compute_episode_splits(
    total_episodes: int,
    suboptimal_ratio: float,
    mixed_quality: bool,
) -> tuple[int, int, int]:
    """Determine how many expert / random / suboptimal episodes to collect.

    Parameters
    ----------
    total_episodes : int
        Total number of episodes desired.
    suboptimal_ratio : float
        Fraction of episodes that are non-expert (split evenly between
        random and noisy-expert).
    mixed_quality : bool
        If ``False``, all episodes come from the expert.

    Returns
    -------
    tuple[int, int, int]
        ``(num_expert, num_random, num_suboptimal)``
    """
    if not mixed_quality:
        return total_episodes, 0, 0

    num_suboptimal = int(total_episodes * suboptimal_ratio * 0.5)
    num_random = int(total_episodes * suboptimal_ratio * 0.5)
    num_expert = total_episodes - num_suboptimal - num_random
    return num_expert, num_random, num_suboptimal


# ------------------------------------------------------------------
# Dataset statistics
# ------------------------------------------------------------------

def print_statistics(collector: DataCollector, save_path: str) -> None:
    """Print a summary of the collected dataset.

    Parameters
    ----------
    collector : DataCollector
        Collector whose internal episodes will be summarised.
    save_path : str
        Path to the saved HDF5 file (for file-size reporting).
    """
    episodes = collector._episodes  # noqa: SLF001
    if not episodes:
        print("No episodes collected.")
        return

    returns = np.array([ep["episode_return"] for ep in episodes])
    lengths = np.array([ep["episode_length"] for ep in episodes])
    ev_times = np.array([
        ep["ev_travel_time"] for ep in episodes if ep["ev_travel_time"] >= 0
    ])

    policy_counts: Dict[str, int] = {}
    for ep in episodes:
        name = ep["policy_name"]
        policy_counts[name] = policy_counts.get(name, 0) + 1

    hr = "-" * 55
    print(f"\n{hr}")
    print("  Dataset Statistics")
    print(hr)
    print(f"  Total episodes       : {len(episodes)}")
    print(f"  Total transitions    : {int(np.sum(lengths))}")
    for policy_name, count in sorted(policy_counts.items()):
        print(f"    {policy_name:20s}: {count}")
    print(hr)
    print(f"  Episode return       : {returns.mean():.2f} +/- {returns.std():.2f}")
    print(f"  Episode length       : {lengths.mean():.1f} +/- {lengths.std():.1f}")
    if len(ev_times) > 0:
        print(f"  EV travel time       : {ev_times.mean():.2f} +/- {ev_times.std():.2f}")
    else:
        print("  EV travel time       : N/A")
    print(hr)

    save_file = Path(save_path)
    if save_file.exists():
        size_mb = save_file.stat().st_size / (1024 * 1024)
        print(f"  File size            : {size_mb:.2f} MB")
        print(f"  Saved to             : {save_file.resolve()}")
    print(hr + "\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline dataset for EV Decision Transformer."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path for the HDF5 dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-multi-agent",
        action="store_true",
        help="Skip multi-agent dataset generation.",
    )
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)

    # Load config
    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    env_cfg = cfg.get("env", {})

    save_path = args.output or ds_cfg.get("save_path", "data/offline_dataset.h5")

    # Determine episode counts
    total_episodes = ds_cfg.get("num_episodes", 5000)
    mixed_quality = ds_cfg.get("mixed_quality", True)
    suboptimal_ratio = ds_cfg.get("suboptimal_ratio", 0.3)

    num_expert, num_random, num_suboptimal = compute_episode_splits(
        total_episodes, suboptimal_ratio, mixed_quality
    )

    print("=" * 55)
    print("  Offline Dataset Generation for EV-DT")
    print("=" * 55)
    print(f"  Expert episodes      : {num_expert}")
    print(f"  Random episodes      : {num_random}")
    print(f"  Suboptimal episodes  : {num_suboptimal}")
    print("=" * 55)

    # ---- Single-agent dataset ----
    print("\n[1/2] Generating single-agent dataset ...")
    env = create_environment(env_cfg)
    expert = create_expert_policy(env)

    collector = DataCollector(env=env, save_path=save_path)
    t0 = time.time()

    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=num_expert,
        num_random=num_random,
        num_suboptimal=num_suboptimal,
    )

    elapsed = time.time() - t0
    print(f"Single-agent collection took {elapsed:.1f}s")
    collector.save_dataset()
    print_statistics(collector, save_path)

    # ---- Multi-agent dataset ----
    if not args.skip_multi_agent:
        print("[2/2] Generating multi-agent dataset ...")
        ma_save_path = save_path.replace(".h5", "_ma.h5")
        ma_env = create_ma_environment(env_cfg)
        ma_expert = create_expert_policy(ma_env)

        ma_collector = DataCollector(env=ma_env, save_path=ma_save_path)
        t0 = time.time()

        ma_collector.collect_mixed_dataset(
            expert_policy=ma_expert,
            num_expert=num_expert,
            num_random=num_random,
            num_suboptimal=num_suboptimal,
        )

        elapsed = time.time() - t0
        print(f"Multi-agent collection took {elapsed:.1f}s")
        ma_collector.save_dataset()
        print_statistics(ma_collector, ma_save_path)
    else:
        print("[2/2] Skipping multi-agent dataset (--skip-multi-agent).")

    print("=" * 55)
    print("  Dataset generation complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
