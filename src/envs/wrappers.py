"""Environment wrappers for observation/action/reward transformations.

Provides Gymnasium-compatible wrappers that can be composed around the
EV corridor environments to adapt their interfaces for different
algorithms (e.g., DQN requires ``Discrete`` actions) or to improve
training stability (observation normalisation, reward scaling).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ======================================================================
# Action wrappers
# ======================================================================

class FlattenActionWrapper(gym.ActionWrapper):
    """Convert a ``MultiDiscrete`` action space to a single ``Discrete`` space.

    This is useful for algorithms (e.g., DQN) that only support flat
    discrete action spaces.  The wrapper enumerates the Cartesian product
    of per-dimension choices and maps a single integer back to the
    multi-discrete tuple.

    .. warning::
       The resulting action space can be very large.  For an
       ``EVCorridorEnv`` with *N* route intersections and 4 phases each
       the Discrete space has ``4^N`` actions.  Only practical for small
       *N* (up to ~6).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        md = env.action_space
        if not isinstance(md, spaces.MultiDiscrete):
            raise TypeError(
                f"FlattenActionWrapper requires MultiDiscrete action space, "
                f"got {type(md).__name__}"
            )
        self._nvec = md.nvec.copy()
        self._total = int(np.prod(self._nvec))
        self.action_space = spaces.Discrete(self._total)

    def action(self, action: int) -> np.ndarray:
        """Decode a flat integer into a multi-discrete array."""
        md_action = np.zeros_like(self._nvec)
        remaining = int(action)
        for i in reversed(range(len(self._nvec))):
            md_action[i] = remaining % self._nvec[i]
            remaining //= self._nvec[i]
        return md_action


# ======================================================================
# Observation wrappers
# ======================================================================

class NormalizeObsWrapper(gym.ObservationWrapper):
    """Running mean/std normalisation of observations.

    Maintains exponential-moving-average statistics and normalises each
    observation dimension to approximately zero mean and unit variance.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    clip : float
        Observations are clipped to ``[-clip, clip]`` after normalisation.
    epsilon : float
        Small constant added to the standard deviation to avoid division
        by zero.
    decay : float
        Exponential decay factor for the running statistics.  Closer to 1
        means slower adaptation.
    """

    def __init__(
        self,
        env: gym.Env,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        decay: float = 0.999,
    ) -> None:
        super().__init__(env)
        obs_shape = env.observation_space.shape
        assert obs_shape is not None, "Observation space must have a shape"

        self.clip = clip
        self.epsilon = epsilon
        self.decay = decay

        self._running_mean = np.zeros(obs_shape, dtype=np.float64)
        self._running_var = np.ones(obs_shape, dtype=np.float64)
        self._count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalise a single observation using running statistics."""
        self._update_stats(obs)
        normalised = (obs - self._running_mean) / (
            np.sqrt(self._running_var) + self.epsilon
        )
        return np.clip(normalised, -self.clip, self.clip).astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        self._count += 1
        if self._count == 1:
            self._running_mean[:] = obs
            self._running_var[:] = 0.0
        else:
            delta = obs - self._running_mean
            self._running_mean += (1 - self.decay) * delta
            self._running_var = (
                self.decay * self._running_var
                + (1 - self.decay) * delta ** 2
            )


class FrameStackWrapper(gym.Wrapper):
    """Stack the most recent *k* observations for temporal context.

    The stacked observations are concatenated along the last axis, so
    the new observation shape is ``(*obs_shape[:-1], obs_shape[-1] * k)``.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    k : int
        Number of frames to stack.
    """

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = k
        obs_shape = env.observation_space.shape
        assert obs_shape is not None

        low = env.observation_space.low
        high = env.observation_space.high

        self.observation_space = spaces.Box(
            low=np.repeat(low, k, axis=-1),
            high=np.repeat(high, k, axis=-1),
            dtype=np.float32,
        )
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self._frames.append(obs)
        return self._get_stacked(), info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1).astype(np.float32)


# ======================================================================
# Reward wrappers
# ======================================================================

class RewardScaleWrapper(gym.RewardWrapper):
    """Scale rewards by a constant factor.

    Useful for keeping reward magnitudes in a range that is friendly
    for value-function approximation.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    scale : float
        Multiplicative factor applied to every reward.
    """

    def __init__(self, env: gym.Env, scale: float = 0.01) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


# ======================================================================
# Recording wrapper
# ======================================================================

class RecordEpisodeWrapper(gym.Wrapper):
    """Record full episode trajectories for offline dataset creation.

    Stores ``(obs, action, reward, next_obs, terminated, truncated, info)``
    tuples for every step.  After the episode ends the full trajectory is
    available in ``self.episode_data``.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    max_episodes : int
        Maximum number of episodes to buffer.  Oldest episodes are
        discarded when the buffer is full.
    """

    def __init__(self, env: gym.Env, max_episodes: int = 100) -> None:
        super().__init__(env)
        self.max_episodes = max_episodes

        # Current episode accumulator
        self._current_obs: np.ndarray | None = None
        self._current_trajectory: list[dict[str, Any]] = []

        # Completed episode buffer
        self.episode_data: deque[list[dict[str, Any]]] = deque(
            maxlen=max_episodes
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Flush any in-progress trajectory
        if self._current_trajectory:
            self.episode_data.append(list(self._current_trajectory))
        self._current_trajectory = []

        obs, info = self.env.reset(seed=seed, options=options)
        self._current_obs = obs.copy()
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._current_trajectory.append(
            {
                "obs": self._current_obs.copy() if self._current_obs is not None else None,
                "action": action if np.isscalar(action) else np.asarray(action).copy(),
                "reward": reward,
                "next_obs": obs.copy(),
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )
        self._current_obs = obs.copy()

        # Auto-flush on episode end
        if terminated or truncated:
            self.episode_data.append(list(self._current_trajectory))
            self._current_trajectory = []

        return obs, reward, terminated, truncated, info

    def get_dataset(self) -> dict[str, Any]:
        """Convert buffered episodes into flat numpy arrays.

        Returns a dict with keys ``observations``, ``actions``,
        ``rewards``, ``terminals``, ``timeouts`` — the standard format
        used by D4RL / minari offline RL datasets.
        """
        all_obs: list[np.ndarray] = []
        all_actions: list[Any] = []
        all_rewards: list[float] = []
        all_terminals: list[bool] = []
        all_timeouts: list[bool] = []

        for episode in self.episode_data:
            for transition in episode:
                all_obs.append(transition["obs"])
                all_actions.append(transition["action"])
                all_rewards.append(transition["reward"])
                all_terminals.append(transition["terminated"])
                all_timeouts.append(transition["truncated"])

        return {
            "observations": np.array(all_obs, dtype=np.float32) if all_obs else np.empty((0,)),
            "actions": np.array(all_actions) if all_actions else np.empty((0,)),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "terminals": np.array(all_terminals, dtype=bool),
            "timeouts": np.array(all_timeouts, dtype=bool),
        }
