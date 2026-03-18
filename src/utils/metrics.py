"""Evaluation metrics for emergency vehicle corridor optimization.

All ``compute_*`` functions accept a list of per-step info dicts
(as returned by ``env.step()``) for a single episode.  The
``aggregate_metrics`` function summarises across multiple episodes,
and ``compare_methods`` produces a tidy pandas DataFrame for
cross-method comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ======================================================================
# Per-episode metrics
# ======================================================================

def compute_ev_travel_time(env_info_list: List[Dict[str, Any]]) -> float:
    """Total time (in decision steps) for the EV to traverse the network.

    Looks for the ``"ev_travel_time"`` key in the *last* info dict.  If it
    is absent, falls back to counting the steps where ``ev_info.active``
    is ``True``.

    Parameters
    ----------
    env_info_list : list[dict]
        Per-step info dicts from a single episode.

    Returns
    -------
    float
        EV travel time in decision steps.  Returns ``-1.0`` if the EV
        never appeared.
    """
    if not env_info_list:
        return -1.0

    # Prefer an explicit value from the environment
    last = env_info_list[-1]
    if "ev_travel_time" in last:
        return float(last["ev_travel_time"])

    # Fallback: count active steps
    active_steps = sum(
        1
        for info in env_info_list
        if info.get("ev_info", {}).get("active", False)
    )
    return float(active_steps) if active_steps > 0 else -1.0


def compute_background_delay(env_info_list: List[Dict[str, Any]]) -> float:
    """Mean delay experienced by non-EV (background) vehicles.

    Expects each info dict to contain ``"background_delay"`` (cumulative
    delay for that step) or ``"avg_delay"`` as a fallback.

    Returns
    -------
    float
        Average per-step background delay.  ``0.0`` if no data available.
    """
    delays: List[float] = []
    for info in env_info_list:
        if "background_delay" in info:
            delays.append(float(info["background_delay"]))
        elif "avg_delay" in info:
            delays.append(float(info["avg_delay"]))
    return float(np.mean(delays)) if delays else 0.0


def compute_throughput(env_info_list: List[Dict[str, Any]]) -> float:
    """Total number of vehicles that completed their trips during the episode.

    Looks for ``"throughput"`` or ``"vehicles_completed"`` in each step's
    info dict and sums them.

    Returns
    -------
    float
        Cumulative throughput (vehicle count).
    """
    total = 0.0
    for info in env_info_list:
        total += float(info.get("throughput", info.get("vehicles_completed", 0)))
    return total


def compute_signal_disruptions(env_info_list: List[Dict[str, Any]]) -> int:
    """Count the number of times a signal phase was changed due to EV preemption.

    A disruption is recorded whenever ``"phase_changed_for_ev"`` is ``True``
    in a step's info dict.

    Returns
    -------
    int
        Total preemption-induced phase changes.
    """
    return sum(
        1
        for info in env_info_list
        if info.get("phase_changed_for_ev", False)
    )


def compute_corridor_green_ratio(env_info_list: List[Dict[str, Any]]) -> float:
    """Fraction of time the EV corridor had a green signal while the EV was active.

    Expects each info dict to include:
    * ``ev_info.active`` – whether the EV is in the network.
    * ``corridor_green`` – whether the corridor phase is green this step.

    Returns
    -------
    float
        Ratio in [0, 1].  Returns ``0.0`` if the EV was never active.
    """
    active_steps = 0
    green_steps = 0
    for info in env_info_list:
        ev_active = info.get("ev_info", {}).get("active", False)
        if ev_active:
            active_steps += 1
            if info.get("corridor_green", False):
                green_steps += 1
    return float(green_steps / active_steps) if active_steps > 0 else 0.0


# ======================================================================
# Aggregation
# ======================================================================

def aggregate_metrics(
    episodes_info: List[Dict[str, Any]] | List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute summary statistics across multiple episodes.

    Parameters
    ----------
    episodes_info : list[dict] or list[list[dict]]
        Either a list of episode dicts (each with a ``"step_infos"`` key
        containing the per-step info list), OR a flat list of per-step
        info-dict lists (one inner list per episode).

    Returns
    -------
    dict
        Keys include ``mean_ev_travel_time``, ``ev_travel_time_std``,
        ``background_delay_mean``, ``throughput_mean``,
        ``signal_disruptions_mean``, ``corridor_green_ratio_mean``,
        and the corresponding ``_std`` variants.
    """
    ev_times: List[float] = []
    bg_delays: List[float] = []
    throughputs: List[float] = []
    disruptions: List[int] = []
    green_ratios: List[float] = []
    returns: List[float] = []
    lengths: List[int] = []

    for ep_info in episodes_info:
        # Normalise: extract the per-step info list
        if isinstance(ep_info, dict):
            step_infos = ep_info.get("step_infos", [])
            returns.append(float(ep_info.get("return", 0.0)))
            lengths.append(int(ep_info.get("length", len(step_infos))))
        else:
            step_infos = ep_info
            returns.append(0.0)
            lengths.append(len(step_infos))

        ev_t = compute_ev_travel_time(step_infos)
        if ev_t >= 0:
            ev_times.append(ev_t)
        bg_delays.append(compute_background_delay(step_infos))
        throughputs.append(compute_throughput(step_infos))
        disruptions.append(compute_signal_disruptions(step_infos))
        green_ratios.append(compute_corridor_green_ratio(step_infos))

    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    ev_stats = _stats(ev_times) if ev_times else {"mean": -1.0, "std": 0.0}

    return {
        "mean_ev_travel_time": ev_stats["mean"],
        "ev_travel_time_mean": ev_stats["mean"],
        "ev_travel_time_std": ev_stats["std"],
        "mean_return": _stats(returns)["mean"] if returns else 0.0,
        "mean_length": _stats([float(x) for x in lengths])["mean"] if lengths else 0.0,
        "background_delay_mean": _stats(bg_delays)["mean"],
        "background_delay_std": _stats(bg_delays)["std"],
        "throughput_mean": _stats(throughputs)["mean"],
        "throughput_std": _stats(throughputs)["std"],
        "signal_disruptions_mean": _stats([float(d) for d in disruptions])["mean"],
        "signal_disruptions_std": _stats([float(d) for d in disruptions])["std"],
        "corridor_green_ratio_mean": _stats(green_ratios)["mean"],
        "corridor_green_ratio_std": _stats(green_ratios)["std"],
        "num_episodes": len(episodes_info),
    }


# ======================================================================
# Cross-method comparison
# ======================================================================

def compare_methods(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison table from multiple methods' aggregated metrics.

    Parameters
    ----------
    results_dict : dict[str, dict]
        Mapping from method name to the dict returned by
        :func:`aggregate_metrics`.

    Returns
    -------
    pd.DataFrame
        Rows = methods, columns = metric names.  Mean +/- std are shown
        as separate columns for clarity.
    """
    rows: List[Dict[str, Any]] = []
    for method_name, metrics in results_dict.items():
        row: Dict[str, Any] = {"method": method_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "method" in df.columns:
        df = df.set_index("method")
    return df
