#!/usr/bin/env python3
"""Generate demo/placeholder figures using synthetic data.

Creates realistic-looking figures that illustrate the *expected* trends for
the EV-DT paper, allowing the paper layout to be previewed before actual
experiments are run.

Trends encoded in the synthetic data
-------------------------------------
- DT and MADT outperform RL baselines (PPO, DQN) and heuristics.
- MADT is slightly better than single-agent DT, especially on larger grids.
- Return conditioning shows a clear EV-time vs. background-delay tradeoff.
- Every ablated component degrades performance (each component matters).

Usage
-----
    python scripts/generate_demo_figures.py [--outdir paper/figures] [--format pdf]

All figures are saved to ``paper/figures/`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from envs.network_utils import build_grid_network, compute_shortest_path
from utils.visualize import (
    plot_ablation_results,
    plot_density_heatmap,
    plot_dt_architecture,
    plot_ev_trajectory,
    plot_grid_network,
    plot_learning_curves,
    plot_method_comparison_bar,
    plot_return_conditioning_sweep,
    plot_scalability,
    set_publication_style,
)


# ===================================================================
# Synthetic data generators
# ===================================================================

def _make_learning_curves(rng: np.random.Generator) -> dict:
    """Training curves: DT/MADT converge faster and lower than baselines."""
    epochs = np.arange(0, 101, 5)

    def _curve(final: float, start: float, noise: float, speed: float = 0.04):
        mean = final + (start - final) * np.exp(-speed * epochs)
        mean += rng.normal(0, noise, size=len(epochs))
        std = noise * (1 + 0.5 * np.exp(-0.03 * epochs))
        return {"x": epochs.tolist(), "mean": mean.tolist(), "std": std.tolist()}

    return {
        "DT (Ours)": _curve(final=82, start=160, noise=2.5, speed=0.06),
        "MADT (Ours)": _curve(final=78, start=155, noise=2.2, speed=0.065),
        "PPO": _curve(final=95, start=170, noise=4.0, speed=0.035),
        "DQN": _curve(final=101, start=175, noise=5.0, speed=0.03),
        "Greedy Preempt": _curve(final=88, start=88, noise=2.0, speed=0.0),
    }


def _make_comparison() -> dict[str, dict[str, tuple[float, float]]]:
    """Method comparison across key metrics (4x4 grid)."""
    return {
        "DT (Ours)": {
            "ev_travel_time": (82.3, 4.1),
            "background_delay": (12.8, 1.9),
            "throughput": (1820.0, 95.0),
            "signal_disruptions": (3.2, 0.8),
            "corridor_green_ratio": (0.72, 0.04),
        },
        "MADT (Ours)": {
            "ev_travel_time": (78.6, 3.7),
            "background_delay": (11.5, 1.6),
            "throughput": (1870.0, 88.0),
            "signal_disruptions": (2.9, 0.7),
            "corridor_green_ratio": (0.75, 0.03),
        },
        "PPO": {
            "ev_travel_time": (95.1, 6.3),
            "background_delay": (15.2, 2.8),
            "throughput": (1690.0, 110.0),
            "signal_disruptions": (5.1, 1.2),
            "corridor_green_ratio": (0.58, 0.06),
        },
        "DQN": {
            "ev_travel_time": (101.4, 7.8),
            "background_delay": (16.7, 3.1),
            "throughput": (1650.0, 125.0),
            "signal_disruptions": (5.8, 1.5),
            "corridor_green_ratio": (0.54, 0.07),
        },
        "Fixed-Time": {
            "ev_travel_time": (118.2, 5.5),
            "background_delay": (9.1, 1.2),
            "throughput": (1780.0, 70.0),
            "signal_disruptions": (0.0, 0.0),
            "corridor_green_ratio": (0.25, 0.0),
        },
        "Greedy Preempt": {
            "ev_travel_time": (88.5, 4.8),
            "background_delay": (19.3, 3.5),
            "throughput": (1580.0, 130.0),
            "signal_disruptions": (7.2, 1.8),
            "corridor_green_ratio": (0.82, 0.05),
        },
    }


def _make_conditioning() -> dict[str, dict[str, list[float]]]:
    """Return-conditioning sweep: tradeoff between EV time and background delay."""
    targets = [0.0, -50.0, -100.0, -150.0, -200.0, -250.0, -300.0]
    return {
        "DT (Ours)": {
            "target_returns": targets,
            "ev_travel_time": [125.1, 102.4, 82.3, 76.8, 73.1, 71.5, 70.8],
            "background_delay": [7.5, 9.6, 12.8, 16.3, 21.5, 28.7, 36.2],
        },
        "MADT (Ours)": {
            "target_returns": targets,
            "ev_travel_time": [119.3, 97.1, 78.6, 73.2, 69.8, 68.1, 67.5],
            "background_delay": [7.1, 8.9, 11.5, 14.7, 19.2, 25.3, 32.8],
        },
    }


def _make_scalability() -> dict[str, dict[str, tuple[float, float]]]:
    """Performance vs grid size.  MADT advantage grows with network size."""
    return {
        "DT (Ours)": {
            "4x4": (82.3, 4.1),
            "6x6": (125.7, 6.8),
            "8x8": (178.4, 9.2),
            "10x10": (241.5, 13.1),
        },
        "MADT (Ours)": {
            "4x4": (78.6, 3.7),
            "6x6": (115.2, 5.9),
            "8x8": (155.3, 7.8),
            "10x10": (198.7, 10.5),
        },
        "PPO": {
            "4x4": (95.1, 6.3),
            "6x6": (155.8, 11.2),
            "8x8": (238.6, 18.5),
            "10x10": (342.1, 28.3),
        },
        "DQN": {
            "4x4": (101.4, 7.8),
            "6x6": (168.3, 13.5),
            "8x8": (260.1, 22.0),
            "10x10": (378.9, 31.7),
        },
        "Greedy Preempt": {
            "4x4": (88.5, 4.8),
            "6x6": (138.9, 8.1),
            "8x8": (198.5, 12.4),
            "10x10": (268.3, 16.9),
        },
    }


def _make_ablation() -> dict[str, tuple[float, float]]:
    """Ablation study -- removing any component hurts performance."""
    return {
        "Full Model": (78.6, 3.7),
        "No Return Cond.": (96.4, 5.8),
        "No Preemption Emb.": (105.2, 7.1),
        "No GAT (single-agent)": (85.9, 4.5),
        "Context = 10": (88.3, 5.0),
        "Context = 5": (93.7, 5.6),
        "No Positional Enc.": (91.1, 5.3),
    }


def _make_trajectory(
    network: dict, route: list, rng: np.random.Generator,
) -> dict:
    """Synthetic episode info for trajectory visualisation."""
    link_times: dict[str, float] = {}
    for _, lid in route:
        if lid is not None:
            # Some links faster, some slower (simulate green wave / congestion)
            link_times[lid] = rng.uniform(5.0, 25.0)

    signal_phases: dict[str, list] = {}
    total_time = sum(link_times.values())
    for nid in [n for n, _ in route]:
        phases = []
        t = 0.0
        while t < total_time:
            duration = rng.uniform(15.0, 40.0)
            phase = rng.integers(0, 4)
            phases.append((t, t + duration, int(phase)))
            t += duration
        signal_phases[nid] = phases

    return {
        "link_times": link_times,
        "signal_phases": signal_phases,
        "total_time": total_time,
    }


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo figures for EV-DT paper.")
    parser.add_argument("--outdir", type=str, default="paper/figures",
                        help="Output directory for figures.")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output figure format.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    outdir = _PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ext = args.format
    rng = np.random.default_rng(args.seed)

    set_publication_style()

    # -----------------------------------------------------------------------
    # Figure 1: Network topology with EV route
    # -----------------------------------------------------------------------
    print("[1/9] Network topology with EV route ...")
    net = build_grid_network(rows=4, cols=4)
    for lk in net["links"].values():
        lk["density"] = rng.uniform(0.0, net["k_jam"] * 0.85)
    for nd in net["nodes"].values():
        nd["current_phase"] = int(rng.integers(0, 4))
    route = compute_shortest_path(net, "n0_0", "n3_3")
    plot_grid_network(
        net, route=route, ev_position="n1_1", densities=True,
        save_path=outdir / f"fig1_network.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 2: Architecture diagram
    # -----------------------------------------------------------------------
    print("[2/9] DT / MADT architecture diagram ...")
    plot_dt_architecture(save_path=outdir / f"fig2_architecture.{ext}")

    # -----------------------------------------------------------------------
    # Figure 3: Learning curves
    # -----------------------------------------------------------------------
    print("[3/9] Learning curves ...")
    curves = _make_learning_curves(rng)
    plot_learning_curves(
        curves,
        metric="ev_travel_time_mean",
        xlabel="Training Epoch",
        title="Training Convergence",
        save_path=outdir / f"fig3_learning_curves.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 4: Method comparison bar chart
    # -----------------------------------------------------------------------
    print("[4/9] Method comparison bar chart ...")
    comparison = _make_comparison()
    # Two-metric view: EV travel time and background delay
    bar_data = {
        m: {
            "EV Travel\nTime (s)": d["ev_travel_time"],
            "Background\nDelay (s)": d["background_delay"],
        }
        for m, d in comparison.items()
    }
    plot_method_comparison_bar(
        bar_data,
        ylabel="Time (s)",
        title="Method Comparison (4$\\times$4 Grid)",
        save_path=outdir / f"fig4_comparison.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 5: Return conditioning sweep
    # -----------------------------------------------------------------------
    print("[5/9] Return conditioning sweep ...")
    conditioning = _make_conditioning()
    plot_return_conditioning_sweep(
        conditioning,
        save_path=outdir / f"fig5_conditioning.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 6: Scalability analysis
    # -----------------------------------------------------------------------
    print("[6/9] Scalability analysis ...")
    scalability = _make_scalability()
    plot_scalability(
        scalability,
        metric="ev_travel_time",
        save_path=outdir / f"fig6_scalability.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 7: Ablation study
    # -----------------------------------------------------------------------
    print("[7/9] Ablation study ...")
    ablation = _make_ablation()
    plot_ablation_results(
        ablation,
        baseline_label="Full Model",
        save_path=outdir / f"fig7_ablation.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 8: Density heatmap
    # -----------------------------------------------------------------------
    print("[8/9] Density heatmap ...")
    # Create a congestion pattern: higher density toward the centre
    heatmap_net = build_grid_network(rows=6, cols=6)
    for lk in heatmap_net["links"].values():
        src_node = heatmap_net["nodes"][lk["source"]]
        r, c = src_node["row"], src_node["col"]
        # Distance from centre => lower distance = more congested
        centre_r, centre_c = 2.5, 2.5
        dist = np.sqrt((r - centre_r) ** 2 + (c - centre_c) ** 2) / 3.5
        base = heatmap_net["k_jam"] * (0.8 - 0.6 * dist)
        lk["density"] = float(np.clip(
            base + rng.normal(0, 0.01), 0, heatmap_net["k_jam"],
        ))
    plot_density_heatmap(
        heatmap_net, timestep=150,
        save_path=outdir / f"fig8_density_heatmap.{ext}",
    )

    # -----------------------------------------------------------------------
    # Figure 9: EV trajectory
    # -----------------------------------------------------------------------
    print("[9/9] EV trajectory visualisation ...")
    traj_net = build_grid_network(rows=4, cols=4)
    traj_route = compute_shortest_path(traj_net, "n0_0", "n3_3")
    episode_info = _make_trajectory(traj_net, traj_route, rng)
    plot_ev_trajectory(
        episode_info, traj_net, traj_route,
        save_path=outdir / f"fig9_ev_trajectory.{ext}",
    )

    # -----------------------------------------------------------------------
    print(f"\nAll {9} demo figures saved to {outdir}/")
    print("Formats: ." + ext)


if __name__ == "__main__":
    main()
