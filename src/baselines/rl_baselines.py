"""Wrappers for Stable-Baselines3 PPO and DQN agents.

Provides convenience functions to create, train, and evaluate RL baselines
with the EV corridor environment.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ======================================================================
# Agent creation
# ======================================================================

def create_ppo_agent(env: Any, config: Dict[str, Any]) -> PPO:
    """Instantiate a PPO agent with the given config.

    Parameters
    ----------
    env : gym.Env
        A Gymnasium-compatible environment (single-agent).
    config : dict
        PPO hyper-parameters.  Recognised keys (all optional, sensible
        defaults are used when missing):

        * ``learning_rate``, ``n_steps``, ``batch_size``, ``n_epochs``,
          ``gamma``, ``clip_range``, ``policy`` (default ``"MlpPolicy"``),
          ``device``, ``seed``.

    Returns
    -------
    PPO
        Configured but untrained PPO agent.
    """
    wrapped = _wrap_env(env)
    return PPO(
        policy=config.get("policy", "MlpPolicy"),
        env=wrapped,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        clip_range=config.get("clip_range", 0.2),
        verbose=config.get("verbose", 0),
        device=config.get("device", "auto"),
        seed=config.get("seed", None),
    )


def create_dqn_agent(env: Any, config: Dict[str, Any]) -> DQN:
    """Instantiate a DQN agent with the given config.

    Parameters
    ----------
    env : gym.Env
        A Gymnasium-compatible environment (single-agent, discrete actions).
    config : dict
        DQN hyper-parameters.  Recognised keys (all optional):

        * ``learning_rate``, ``buffer_size``, ``batch_size``, ``gamma``,
          ``exploration_fraction``, ``target_update_interval``,
          ``policy`` (default ``"MlpPolicy"``), ``device``, ``seed``.

    Returns
    -------
    DQN
        Configured but untrained DQN agent.
    """
    wrapped = _wrap_env(env)
    return DQN(
        policy=config.get("policy", "MlpPolicy"),
        env=wrapped,
        learning_rate=config.get("learning_rate", 1e-4),
        buffer_size=config.get("buffer_size", 100_000),
        batch_size=config.get("batch_size", 32),
        gamma=config.get("gamma", 0.99),
        exploration_fraction=config.get("exploration_fraction", 0.3),
        target_update_interval=config.get("target_update_interval", 1000),
        verbose=config.get("verbose", 0),
        device=config.get("device", "auto"),
        seed=config.get("seed", None),
    )


# ======================================================================
# Training
# ======================================================================

def train_baseline(
    agent: PPO | DQN,
    total_timesteps: int,
    log_dir: str,
    eval_env: Optional[Any] = None,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    save_path: Optional[str] = None,
) -> PPO | DQN:
    """Train an SB3 agent and optionally run periodic evaluation.

    Parameters
    ----------
    agent : PPO | DQN
        An SB3 agent returned by :func:`create_ppo_agent` or
        :func:`create_dqn_agent`.
    total_timesteps : int
        Total environment steps to train for.
    log_dir : str
        Directory for TensorBoard logs and model checkpoints.
    eval_env : gym.Env, optional
        If provided, an ``EvalCallback`` is attached for periodic evaluation.
    eval_freq : int
        How often (in timesteps) to evaluate when *eval_env* is given.
    n_eval_episodes : int
        Episodes per evaluation round.

    Returns
    -------
    PPO | DQN
        The trained agent (same object, mutated in-place).
    """
    os.makedirs(log_dir, exist_ok=True)

    callbacks = []
    if eval_env is not None:
        eval_wrapped = _wrap_env(eval_env)
        eval_cb = EvalCallback(
            eval_wrapped,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_cb)

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=os.path.join(log_dir, "tb"),
    )
    final_path = save_path or os.path.join(log_dir, "final_model")
    agent.save(final_path)
    return agent


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_baseline(
    agent: PPO | DQN,
    env: Any,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """Evaluate a trained agent over *n_episodes* episodes.

    Parameters
    ----------
    agent : PPO | DQN
        Trained SB3 agent.
    env : gym.Env
        Environment to evaluate in (unwrapped is fine).
    n_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    dict
        Aggregated statistics:

        * ``mean_reward``  – mean cumulative reward across episodes.
        * ``std_reward``   – standard deviation of episode returns.
        * ``mean_length``  – mean episode length.
        * ``episode_rewards`` – list of per-episode returns.
        * ``episode_lengths`` – list of per-episode lengths.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

    rewards_arr = np.array(episode_rewards)
    return {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_length": float(np.mean(episode_lengths)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


# ======================================================================
# Internal helpers
# ======================================================================

def _wrap_env(env: Any) -> DummyVecEnv:
    """Wrap a plain Gymnasium env for SB3 if it is not already vectorised."""
    if isinstance(env, DummyVecEnv):
        return env
    monitored = Monitor(env)
    return DummyVecEnv([lambda: monitored])
