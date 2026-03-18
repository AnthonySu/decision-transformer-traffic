#!/usr/bin/env python3
"""Train RL baselines (PPO, DQN) for EV corridor optimization."""

import argparse
from pathlib import Path

import yaml

from src.baselines.rl_baselines import create_dqn_agent, create_ppo_agent, train_baseline
from src.envs.ev_corridor_env import EVCorridorEnv


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str, method: str = "ppo"):
    config = load_config(config_path)
    env_cfg = config["env"]
    baseline_cfg = config["baselines"][method]

    env = EVCorridorEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=env_cfg["max_episode_steps"],
    )

    eval_env = EVCorridorEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=env_cfg["max_episode_steps"],
    )

    log_dir = Path(f"logs/{method}")
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(f"models/{method}_evp")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {method.upper()} baseline")
    print(f"Config: {baseline_cfg}")

    if method == "ppo":
        agent = create_ppo_agent(env, baseline_cfg, log_dir=str(log_dir))
    elif method == "dqn":
        agent = create_dqn_agent(env, baseline_cfg, log_dir=str(log_dir))
    else:
        raise ValueError(f"Unknown method: {method}")

    train_baseline(
        agent,
        total_timesteps=baseline_cfg["total_timesteps"],
        log_dir=str(log_dir),
        eval_env=eval_env,
        save_path=str(save_dir / "best_model"),
    )

    print(f"\n{method.upper()} training complete. Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["ppo", "dqn"], default="ppo")
    args = parser.parse_args()
    train(args.config, args.method)
