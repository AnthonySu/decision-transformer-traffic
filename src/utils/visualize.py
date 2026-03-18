"""Reusable plotting functions for the EV-DT research project.

Generates publication-quality figures for grid-network visualizations,
training curves, method comparisons, ablation studies, and more.
All functions accept an optional ``save_path`` to write the figure to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Custom colormaps
# ---------------------------------------------------------------------------

_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density",
    [(0.0, "#2ecc71"), (0.45, "#f1c40f"), (0.75, "#e67e22"), (1.0, "#e74c3c")],
)

# Method colours — consistent across all plots
METHOD_COLORS: dict[str, str] = {
    "DT (Ours)": "#2980b9",
    "MADT (Ours)": "#1a5276",
    "PPO": "#e74c3c",
    "DQN": "#e67e22",
    "Fixed-Time": "#95a5a6",
    "Greedy Preempt": "#27ae60",
    "No Preemption": "#bdc3c7",
}

METHOD_HATCHES: dict[str, str] = {
    "DT (Ours)": "",
    "MADT (Ours)": "//",
    "PPO": "\\\\",
    "DQN": "xx",
    "Fixed-Time": "..",
    "Greedy Preempt": "--",
    "No Preemption": "++",
}


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """Set matplotlib rcParams for publication-quality figures.

    Targets a two-column IEEE / Elsevier layout with serif fonts compatible
    with LaTeX rendering.
    """
    plt.rcParams.update({
        # --- Fonts ---
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        # --- Axes ---
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        # --- Ticks ---
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # --- Legend ---
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # --- Figure ---
        "figure.figsize": (3.5, 2.6),  # single-column width
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        # --- Grid ---
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        # --- Lines ---
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save figure to *save_path* (creating parent dirs) or show it."""
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p))
        plt.close(fig)
    else:
        plt.show()


def _get_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#7f8c8d")


def _get_hatch(method: str) -> str:
    return METHOD_HATCHES.get(method, "")


def _node_xy(node_data: dict[str, Any], spacing: float = 1.0) -> tuple[float, float]:
    """Return (x, y) for a node given its row/col."""
    return node_data["col"] * spacing, -node_data["row"] * spacing


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def plot_grid_network(
    network: dict[str, Any],
    route: list[tuple[str, str | None]] | None = None,
    ev_position: str | None = None,
    densities: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a grid network with optional EV route and link-density colouring.

    Parameters
    ----------
    network : Network dict from ``build_grid_network``.
    route : Optional EV route (list of (node_id, link_id) tuples).
    ev_position : Node ID where the EV currently is.
    densities : If True, colour links by their current density.
    save_path : If given, save figure to this path.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    nodes = network["nodes"]
    links = network["links"]
    rows, cols = network["rows"], network["cols"]
    spacing = 1.0
    k_jam = network["k_jam"]

    # Collect route link IDs for highlighting
    route_link_ids: set[str] = set()
    if route is not None:
        for _, lid in route:
            if lid is not None:
                route_link_ids.add(lid)

    # --- Draw links ---
    norm = Normalize(vmin=0, vmax=k_jam)
    for lid, lk in links.items():
        src = nodes[lk["source"]]
        dst = nodes[lk["target"]]
        x0, y0 = _node_xy(src, spacing)
        x1, y1 = _node_xy(dst, spacing)

        # Slight offset so bidirectional links don't overlap
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        perp_x, perp_y = -dy / length * 0.06, dx / length * 0.06

        sx, sy = x0 + perp_x, y0 + perp_y
        ex, ey = x1 + perp_x, y1 + perp_y

        # Shorten slightly so arrows don't overlap node circles
        shrink = 0.12
        sx += dx / length * shrink
        sy += dy / length * shrink
        ex -= dx / length * shrink
        ey -= dy / length * shrink

        if lid in route_link_ids:
            color = "#2980b9"
            linewidth = 2.8
            alpha = 1.0
            zorder = 3
        elif densities:
            color = _DENSITY_CMAP(norm(lk["density"]))
            linewidth = 1.6
            alpha = 0.85
            zorder = 2
        else:
            color = "#bdc3c7"
            linewidth = 1.2
            alpha = 0.7
            zorder = 2

        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=linewidth,
                mutation_scale=8,
            ),
            zorder=zorder,
        )

    # --- Draw nodes (intersections) ---
    for nid, nd in nodes.items():
        x, y = _node_xy(nd, spacing)
        phase = nd["current_phase"]
        # Colour rim by current phase
        phase_colors = ["#27ae60", "#3498db", "#f39c12", "#e74c3c"]
        rim_color = phase_colors[phase % len(phase_colors)]
        circle = plt.Circle(
            (x, y), 0.10, facecolor="white", edgecolor=rim_color,
            linewidth=1.5, zorder=5,
        )
        ax.add_patch(circle)

        if nd["is_boundary"]:
            ax.plot(x, y, "s", color="#ecf0f1", markersize=3, zorder=6)

    # --- EV position ---
    if ev_position is not None and ev_position in nodes:
        ex, ey = _node_xy(nodes[ev_position], spacing)
        ax.plot(
            ex, ey, "*", color="#e74c3c", markersize=16, markeredgecolor="white",
            markeredgewidth=0.8, zorder=10, label="EV",
        )

    # --- Density colour bar ---
    if densities:
        sm = plt.cm.ScalarMappable(cmap=_DENSITY_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.75)
        cbar.set_label("Link density (veh/m/lane)", fontsize=9)

    # --- Cosmetics ---
    ax.set_xlim(-0.3, (cols - 1) * spacing + 0.3)
    ax.set_ylim(-(rows - 1) * spacing - 0.3, 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"{rows}$\\times${cols} Grid Network", fontsize=11)

    # Legend
    legend_elements = []
    if route is not None:
        legend_elements.append(
            mpatches.Patch(color="#2980b9", label="EV route")
        )
    if ev_position is not None:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c",
                   markersize=12, label="EV position")
        )
    # Phase legend
    phase_labels = ["N/S through", "E/W through", "N/S left", "E/W left"]
    phase_colors_list = ["#27ae60", "#3498db", "#f39c12", "#e74c3c"]
    for pc, pl in zip(phase_colors_list, phase_labels):
        legend_elements.append(mpatches.Patch(facecolor="white", edgecolor=pc,
                                              linewidth=1.5, label=pl))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
                  framealpha=0.85)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_learning_curves(
    results_dict: dict[str, dict[str, Any]],
    metric: str = "ev_travel_time_mean",
    xlabel: str = "Training Epoch",
    ylabel: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training/evaluation curves for multiple methods on the same axes.

    Parameters
    ----------
    results_dict : ``{method_name: {"x": array, "mean": array, "std": array}}``.
    metric : Metric key (used as default y-label).
    save_path : If given, save figure to this path.
    """
    set_publication_style()
    fig, ax = plt.subplots()

    for method, data in results_dict.items():
        x = np.asarray(data["x"])
        mean = np.asarray(data["mean"])
        std = np.asarray(data.get("std", np.zeros_like(mean)))
        color = _get_color(method)

        ax.plot(x, mean, label=method, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or _metric_label(metric))
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_method_comparison_bar(
    results_dict: dict[str, dict[str, tuple[float, float]]],
    metrics: list[str] | None = None,
    ylabel: str = "Value",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing methods across metrics.

    Parameters
    ----------
    results_dict :
        ``{method_name: {metric_name: (mean, std), ...}, ...}``
    metrics : Subset of metric keys to plot.  Defaults to all.
    """
    set_publication_style()

    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(next(iter(results_dict.values())).keys())

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    total_width = 0.75
    bar_width = total_width / n_methods

    fig, ax = plt.subplots(figsize=(3.5 + 0.6 * n_metrics, 2.8))

    for i, method in enumerate(methods):
        means = [results_dict[method][m][0] for m in metrics]
        stds = [results_dict[method][m][1] for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        color = _get_color(method)
        hatch = _get_hatch(method)
        ax.bar(
            x + offset, means, bar_width, yerr=stds,
            label=method, color=color, hatch=hatch,
            edgecolor="white", linewidth=0.5,
            capsize=2, error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_metric_label(m) for m in metrics], fontsize=8)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=7, ncol=min(n_methods, 3))
    ax.axhline(0, color="black", linewidth=0.4)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_return_conditioning_sweep(
    sweep_results: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot how different target returns affect EV travel time and background delay.

    Parameters
    ----------
    sweep_results :
        ``{method_name: {"target_returns": [...], "ev_travel_time": [...],
        "background_delay": [...]}}``.
    """
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    ax2 = ax1.twinx()

    for method, data in sweep_results.items():
        targets = np.asarray(data["target_returns"])
        ev_time = np.asarray(data["ev_travel_time"])
        bg_delay = np.asarray(data["background_delay"])
        color = _get_color(method)

        ax1.plot(targets, ev_time, "o-", color=color, label=f"{method} (EV time)",
                 markersize=5, linewidth=1.4)
        ax2.plot(targets, bg_delay, "s--", color=color, alpha=0.6,
                 label=f"{method} (bg delay)", markersize=4, linewidth=1.2)

    ax1.set_xlabel("Target Return")
    ax1.set_ylabel("EV Travel Time (s)", color="#2c3e50")
    ax2.set_ylabel("Background Delay (s)", color="#7f8c8d")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    ax1.set_title("Return-Conditioning Tradeoff")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_ev_trajectory(
    episode_info: dict[str, Any],
    network: dict[str, Any],
    route: list[tuple[str, str | None]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualise a single episode's EV trajectory through the network.

    Parameters
    ----------
    episode_info :
        ``{"link_times": {link_id: time_spent_s, ...},
           "signal_phases": {node_id: [(t_start, t_end, phase), ...], ...},
           "total_time": float}``.
    network : Network dict.
    route : EV route.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})

    nodes = network["nodes"]
    links = network["links"]
    rows, cols = network["rows"], network["cols"]
    spacing = 1.0

    link_times = episode_info.get("link_times", {})
    max_time = max(link_times.values()) if link_times else 1.0

    # --- Left: Network with link-time heatmap ---
    ax = axes[0]
    route_link_ids = {lid for _, lid in route if lid is not None}

    norm = Normalize(vmin=0, vmax=max_time)
    heat_cmap = plt.cm.YlOrRd

    for lid, lk in links.items():
        src = nodes[lk["source"]]
        dst = nodes[lk["target"]]
        x0, y0 = _node_xy(src, spacing)
        x1, y1 = _node_xy(dst, spacing)
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        perp_x, perp_y = -dy / length * 0.06, dx / length * 0.06

        if lid in route_link_ids and lid in link_times:
            color = heat_cmap(norm(link_times[lid]))
            lw = 3.0
            alpha = 1.0
        elif lid in route_link_ids:
            color = "#3498db"
            lw = 2.0
            alpha = 0.8
        else:
            color = "#d5d8dc"
            lw = 0.8
            alpha = 0.5

        ax.plot(
            [x0 + perp_x, x1 + perp_x], [y0 + perp_y, y1 + perp_y],
            color=color, linewidth=lw, alpha=alpha, solid_capstyle="round",
        )

    for nid, nd in nodes.items():
        x, y = _node_xy(nd, spacing)
        ax.plot(x, y, "o", color="#2c3e50", markersize=4, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=heat_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label("Time on link (s)", fontsize=8)

    ax.set_xlim(-0.3, (cols - 1) * spacing + 0.3)
    ax.set_ylim(-(rows - 1) * spacing - 0.3, 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("EV Link Traversal Time", fontsize=10)

    # --- Right: Signal phase timeline ---
    ax2 = axes[1]
    route_nodes = [n for n, _ in route]
    signal_phases = episode_info.get("signal_phases", {})
    phase_colors = ["#27ae60", "#3498db", "#f39c12", "#e74c3c"]

    for i, nid in enumerate(route_nodes):
        if nid not in signal_phases:
            continue
        for t_start, t_end, phase in signal_phases[nid]:
            ax2.barh(
                i, t_end - t_start, left=t_start, height=0.6,
                color=phase_colors[phase % len(phase_colors)], edgecolor="none",
            )

    ax2.set_yticks(range(len(route_nodes)))
    ax2.set_yticklabels(route_nodes, fontsize=7)
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Signal Phases Along Route", fontsize=10)
    ax2.invert_yaxis()

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_scalability(
    results_by_size: dict[str, dict[str, tuple[float, float]]],
    metric: str = "ev_travel_time",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot performance vs network size for scalability analysis.

    Parameters
    ----------
    results_by_size :
        ``{method_name: {grid_label: (mean, std), ...}, ...}``
        where grid_label is e.g. "4x4", "6x6", "8x8".
    metric : Name of the metric being plotted.
    """
    set_publication_style()
    fig, ax = plt.subplots()

    methods = list(results_by_size.keys())
    # Gather all grid sizes (sorted by area)
    all_sizes: list[str] = sorted(
        {s for m_data in results_by_size.values() for s in m_data},
        key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]),
    )
    x = np.arange(len(all_sizes))

    for method in methods:
        data = results_by_size[method]
        means = [data[s][0] if s in data else np.nan for s in all_sizes]
        stds = [data[s][1] if s in data else 0.0 for s in all_sizes]
        color = _get_color(method)
        ax.errorbar(
            x, means, yerr=stds, label=method, color=color,
            marker="o", capsize=3, linewidth=1.4, markersize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title("Scalability Analysis")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_ablation_results(
    ablation_results: dict[str, tuple[float, float]],
    baseline_label: str = "Full Model",
    metric_label: str = "EV Travel Time (s)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart for ablation study results.

    Parameters
    ----------
    ablation_results :
        ``{variant_label: (mean, std), ...}``
        Should include the full model and each ablated variant.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    labels = list(ablation_results.keys())
    means = [ablation_results[l][0] for l in labels]
    stds = [ablation_results[l][1] for l in labels]

    colors = []
    for l in labels:
        if l == baseline_label:
            colors.append("#2980b9")
        else:
            colors.append("#95a5a6")

    bars = ax.bar(
        range(len(labels)), means, yerr=stds,
        color=colors, edgecolor="white", linewidth=0.6,
        capsize=3, error_kw={"linewidth": 0.8},
    )

    # Highlight the full model bar
    for i, l in enumerate(labels):
        if l == baseline_label:
            bars[i].set_edgecolor("#2c3e50")
            bars[i].set_linewidth(1.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title("Ablation Study")
    ax.axhline(means[0] if baseline_label in labels else 0,
               color="#2980b9", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_density_heatmap(
    network: dict[str, Any],
    timestep: int | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot link density as a heatmap over the grid.

    Each cell represents the average density of links entering that
    intersection.  Boundary nodes with fewer links are still shown.
    """
    set_publication_style()
    rows, cols = network["rows"], network["cols"]
    nodes = network["nodes"]
    links = network["links"]

    density_grid = np.zeros((rows, cols))

    for nid, nd in nodes.items():
        r, c = nd["row"], nd["col"]
        incoming = nd["incoming_links"]
        if incoming:
            avg_density = np.mean([links[lid]["density"] for lid in incoming])
        else:
            avg_density = 0.0
        density_grid[r, c] = avg_density

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(
        density_grid, cmap=_DENSITY_CMAP, interpolation="nearest",
        vmin=0, vmax=network["k_jam"], aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Avg. density (veh/m/lane)", fontsize=8)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([str(c) for c in range(cols)], fontsize=7)
    ax.set_yticklabels([str(r) for r in range(rows)], fontsize=7)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    title = "Link Density Heatmap"
    if timestep is not None:
        title += f" (t={timestep})"
    ax.set_title(title)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_dt_architecture(save_path: str | Path | None = None) -> plt.Figure:
    """Draw a simplified block diagram of the Decision Transformer architecture.

    Uses matplotlib patches and annotations -- no external image dependencies.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    box_kw = dict(
        boxstyle="round,pad=0.3", facecolor="#ebf5fb",
        edgecolor="#2980b9", linewidth=1.2,
    )
    embed_kw = dict(
        boxstyle="round,pad=0.25", facecolor="#fef9e7",
        edgecolor="#f39c12", linewidth=1.0,
    )
    out_kw = dict(
        boxstyle="round,pad=0.3", facecolor="#eafaf1",
        edgecolor="#27ae60", linewidth=1.2,
    )

    # Input tokens
    input_labels = [
        ("$\\hat{R}_t$", "Target\nReturn"),
        ("$s_t$", "State"),
        ("$a_{t-1}$", "Prev.\nAction"),
    ]
    token_x = [1.5, 4.0, 6.5]
    for i, (sym, desc) in enumerate(input_labels):
        x = token_x[i]
        ax.text(x, 0.5, f"{sym}\n{desc}", ha="center", va="center", fontsize=8,
                bbox=box_kw)

    # Embedding layer
    ax.text(4.0, 1.8, "Token Embedding + Positional Encoding", ha="center",
            va="center", fontsize=8, bbox=embed_kw)

    # Arrows: inputs -> embedding
    for x in token_x:
        ax.annotate("", xy=(4.0 if abs(x - 4.0) > 1 else x, 1.35),
                     xytext=(x, 0.95),
                     arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))

    # Transformer block
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.8, 2.5), 4.4, 1.2, boxstyle="round,pad=0.2",
        facecolor="#d6eaf8", edgecolor="#2980b9", linewidth=1.5,
    ))
    ax.text(4.0, 3.45, "Causal Self-Attention", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(4.0, 2.9, "$L$ Transformer Layers", ha="center", va="center",
            fontsize=7, color="#5d6d7e")

    ax.annotate("", xy=(4.0, 2.5), xytext=(4.0, 2.15),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))

    # Output head
    ax.text(4.0, 4.3, "Action Head $\\rightarrow a_t$", ha="center", va="center",
            fontsize=9, fontweight="bold", bbox=out_kw)
    ax.annotate("", xy=(4.0, 3.95), xytext=(4.0, 3.75),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))

    # --- MADT extension (right side) ---
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.5, 0.3), 4.5, 3.8, boxstyle="round,pad=0.2",
        facecolor="#fdf2e9", edgecolor="#e67e22", linewidth=1.2, linestyle="--",
    ))
    ax.text(10.75, 3.75, "MADT Extension", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#e67e22")

    # GAT block
    ax.add_patch(mpatches.FancyBboxPatch(
        (9.0, 2.2), 3.5, 1.0, boxstyle="round,pad=0.2",
        facecolor="#fdebd0", edgecolor="#e67e22", linewidth=1.0,
    ))
    ax.text(10.75, 2.7, "Graph Attention Network\n(Inter-Agent Comm.)",
            ha="center", va="center", fontsize=7.5)

    # Agent nodes
    for i, lbl in enumerate(["Agent 1\n(Intersection)", "Agent 2\n(Intersection)", "..."]):
        ax.text(9.5 + i * 1.3, 1.2, lbl, ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#e67e22", linewidth=0.8))
        ax.annotate("", xy=(10.75, 2.15), xytext=(9.5 + i * 1.3, 1.55),
                     arrowprops=dict(arrowstyle="->", color="#e67e22", lw=0.7))

    # Connection arrow from MADT to main transformer
    ax.annotate("", xy=(6.25, 3.1), xytext=(8.45, 2.7),
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2,
                                linestyle="--"))
    ax.text(7.3, 3.2, "fuse", fontsize=7, color="#e67e22", ha="center",
            fontstyle="italic")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Metric label helper
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "ev_travel_time": "EV Travel Time (s)",
    "ev_travel_time_mean": "EV Travel Time (s)",
    "background_delay": "Background Delay (s)",
    "throughput": "Throughput (veh/hr)",
    "signal_disruptions": "Signal Disruptions",
    "corridor_green_ratio": "Corridor Green Ratio",
}


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())
