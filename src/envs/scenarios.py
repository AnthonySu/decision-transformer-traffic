"""Pre-defined scenarios for EV corridor experiments.

Each scenario specifies a network topology, demand pattern,
and EV origin-destination configuration.
"""

from __future__ import annotations

from typing import Any

SCENARIOS: dict[str, dict[str, Any]] = {
    "grid-3x3-v0": {
        "type": "grid",
        "rows": 3,
        "cols": 3,
        "demand_rate": 0.03,
        "ev_speed_factor": 1.5,
        "max_steps": 150,
        "description": "Small 3x3 grid for fast testing",
    },
    "grid-4x4-v0": {
        "type": "grid",
        "rows": 4,
        "cols": 4,
        "demand_rate": 0.05,
        "ev_speed_factor": 1.5,
        "max_steps": 200,
        "description": "Standard 4x4 grid benchmark",
    },
    "grid-6x6-v0": {
        "type": "grid",
        "rows": 6,
        "cols": 6,
        "demand_rate": 0.04,
        "ev_speed_factor": 1.5,
        "max_steps": 300,
        "description": "Large 6x6 grid for scalability",
    },
    "grid-8x8-v0": {
        "type": "grid",
        "rows": 8,
        "cols": 8,
        "demand_rate": 0.03,
        "ev_speed_factor": 1.5,
        "max_steps": 400,
        "description": "Very large 8x8 grid stress test",
    },
    "arterial-v0": {
        "type": "arterial",
        "num_intersections": 8,
        "demand_rate": 0.06,
        "ev_speed_factor": 1.5,
        "max_steps": 200,
        "description": "Linear arterial corridor (8 intersections)",
    },
    "high-demand-v0": {
        "type": "grid",
        "rows": 4,
        "cols": 4,
        "demand_rate": 0.10,
        "ev_speed_factor": 1.3,
        "max_steps": 200,
        "description": "High congestion scenario",
    },
    "rush-hour-v0": {
        "type": "grid",
        "rows": 4,
        "cols": 4,
        "demand_rate": 0.08,
        "ev_speed_factor": 1.5,
        "directional_bias": 0.7,  # 70% of traffic flows E-W
        "max_steps": 200,
        "description": "Asymmetric rush-hour demand",
    },
}


def get_scenario(name: str) -> dict[str, Any]:
    """Get scenario config by name.

    Parameters
    ----------
    name : str
        Scenario identifier (e.g. ``"grid-4x4-v0"``).

    Returns
    -------
    dict
        A *copy* of the scenario configuration so callers can safely mutate it.

    Raises
    ------
    KeyError
        If the scenario name is not registered.
    """
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS.keys()))
        raise KeyError(
            f"Unknown scenario {name!r}. Available scenarios: {available}"
        )
    return dict(SCENARIOS[name])


def list_scenarios() -> list[str]:
    """List all available scenario names."""
    return sorted(SCENARIOS.keys())


def create_env_from_scenario(
    name: str,
    multi_agent: bool = False,
    **kwargs: Any,
):
    """Factory: create an environment from a scenario name.

    Parameters
    ----------
    name : str
        Registered scenario name.
    multi_agent : bool
        If ``True``, return an :class:`EVCorridorMAEnv`; otherwise
        return an :class:`EVCorridorEnv`.
    **kwargs
        Additional keyword arguments forwarded to the environment
        constructor (override scenario defaults).

    Returns
    -------
    EVCorridorEnv | EVCorridorMAEnv
        The instantiated environment.
    """
    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
    from src.envs.network_utils import build_arterial_network, build_grid_network

    scenario = get_scenario(name)
    scenario.update(kwargs)

    topo_type = scenario.pop("type", "grid")
    # Remove metadata keys the env constructors don't accept
    scenario.pop("description", None)
    demand_rate = scenario.pop("demand_rate", 0.05)
    directional_bias = scenario.pop("directional_bias", None)

    # Build the appropriate network
    if topo_type == "arterial":
        num_intersections = scenario.pop("num_intersections", 8)
        network = build_arterial_network(num_intersections=num_intersections)
        # Arterial envs need rows/cols for space sizing — derive from topology
        # The arterial has 3 rows, num_intersections cols
        scenario.setdefault("rows", 3)
        scenario.setdefault("cols", num_intersections)
    else:
        rows = scenario.get("rows", 4)
        cols = scenario.get("cols", 4)
        network = build_grid_network(rows=rows, cols=cols)

    env_cls = EVCorridorMAEnv if multi_agent else EVCorridorEnv
    env = env_cls(**scenario)

    # Inject the pre-built network so constructors don't rebuild
    env._network = network

    # Store demand metadata on the environment for boundary injection
    env._scenario_demand_rate = demand_rate
    env._scenario_directional_bias = directional_bias

    return env
