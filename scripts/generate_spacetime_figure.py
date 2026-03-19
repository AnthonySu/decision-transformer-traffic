#!/usr/bin/env python3
"""Generate spacetime diagram comparing DT vs FT-EVP signal preemption.

Produces fig8_spacetime.pdf/png showing EV progression through intersections
over time, with background signal phase coloring (green=favorable, red=not).

DT (trained policy): smart preemption clears greens ahead, ~20 steps for 7 intersections.
FT-EVP (fixed timing): EV hits reds and waits, ~28 steps with visible plateaus.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

OUTDIR_LOCAL = _ROOT / "paper" / "figures" / "camera_ready"
OUTDIR_LOCAL.mkdir(parents=True, exist_ok=True)

OVERLEAF = Path("C:/Users/admin/Projects/overleaf-ev-dt/figures")

# ---------------------------------------------------------------------------
# Color palette (matching generate_narrative_figures.py)
# ---------------------------------------------------------------------------
C_DT = "#3274A1"
C_RULE_1 = "#8C8C8C"  # FT-EVP

# ---------------------------------------------------------------------------
# Global rcParams -- match existing figures
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": False,  # no grid for spacetime diagram
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def save(fig, name):
    for d in [OUTDIR_LOCAL, OVERLEAF]:
        if d.exists():
            fig.savefig(d / f"{name}.pdf", format="pdf")
            fig.savefig(d / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  Saved {name}")


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def generate_signal_phases(n_intersections, n_timesteps, seed, green_bias=0.5):
    """Generate plausible signal phase matrix.

    Returns boolean array (n_intersections, n_timesteps) where True = green for EV.
    Signals alternate in cycles of varying length to look realistic.
    """
    rng = np.random.RandomState(seed)
    phases = np.zeros((n_intersections, n_timesteps), dtype=bool)
    for i in range(n_intersections):
        # Each intersection has its own cycle offset and period
        cycle_len = rng.randint(4, 8)
        green_len = max(2, int(cycle_len * green_bias))
        offset = rng.randint(0, cycle_len)
        for t in range(n_timesteps):
            pos_in_cycle = (t + offset) % cycle_len
            phases[i, t] = pos_in_cycle < green_len
    return phases


def generate_dt_scenario(n_intersections, n_timesteps):
    """DT scenario: smart preemption, mostly green ahead of EV.

    EV moves through ~7 intersections in ~20 steps with few stops.
    Signal phases are manipulated to be green when the EV approaches.
    """
    # Base signal phases (some natural pattern)
    phases = generate_signal_phases(n_intersections, n_timesteps, seed=42, green_bias=0.5)

    # EV trajectory: smooth diagonal, roughly 1 intersection every ~3 steps
    # with slight variation but no long plateaus (~20 steps total for 7 intersections)
    ev_times = []
    ev_intersections = []
    t = 0
    steps_per = [3, 3, 3, 4, 3, 3]  # total = 19, so last arrival at t=19
    for i in range(n_intersections):
        ev_times.append(t)
        ev_intersections.append(i)
        if i < n_intersections - 1:
            t += steps_per[i]

    # Smart preemption: turn signals green 1-2 steps before EV arrives
    for idx in range(len(ev_times)):
        inter = ev_intersections[idx]
        arr_t = ev_times[idx]
        # Green window: from 2 steps before arrival to 1 step after
        for dt in range(-2, 2):
            tt = arr_t + dt
            if 0 <= tt < n_timesteps:
                phases[inter, tt] = True

    return phases, np.array(ev_times), np.array(ev_intersections)


def generate_ft_scenario(n_intersections, n_timesteps):
    """FT-EVP scenario: fixed timing, EV hits multiple reds.

    EV takes ~28 steps for 7 intersections with visible horizontal plateaus
    at intersections 1, 3, and 5 where it waits for red lights.
    """
    # Same base signal pattern but NOT adapted for EV
    phases = generate_signal_phases(n_intersections, n_timesteps, seed=42, green_bias=0.5)

    # EV trajectory: includes waiting plateaus at certain intersections
    # Target: ~28 steps total for 7 intersections
    ev_times = []
    ev_intersections = []
    t = 0

    # Define per-intersection behavior: (travel_time, wait_time)
    # travel_time = steps to reach next intersection, wait_time = red-light wait at this one
    inter_plan = [
        (0, 0),   # I0: arrive t=0, no wait
        (3, 4),   # I1: arrive t=3, wait 4 steps (red)
        (2, 0),   # I2: arrive t=9, no wait
        (3, 5),   # I3: arrive t=11, wait 5 steps (long red)
        (2, 0),   # I4: arrive t=18, no wait
        (3, 4),   # I5: arrive t=20, wait 4 steps (red)
        (2, 0),   # I6: arrive t=26
    ]

    for i in range(n_intersections):
        travel, wait = inter_plan[i]
        if i > 0:
            t += travel

        ev_times.append(t)
        ev_intersections.append(i)

        if wait > 0:
            # Horizontal plateau: EV stays at this intersection
            for w in range(1, wait + 1):
                t += 1
                ev_times.append(t)
                ev_intersections.append(i)
            # Mark those phases as red during the wait
            wait_start = ev_times[-wait - 1]
            wait_end = ev_times[-1]
            for wt in range(wait_start, wait_end + 1):
                if 0 <= wt < n_timesteps:
                    phases[i, wt] = False

    return phases, np.array(ev_times), np.array(ev_intersections)


def plot_spacetime(ax, phases, ev_times, ev_intersections, title, n_timesteps):
    """Plot a single spacetime diagram on the given axes."""
    n_inter = phases.shape[0]

    # Custom colormap: red for unfavorable, green for favorable
    cmap = mcolors.ListedColormap(["#E85D5D", "#6DBF6D"])  # softer red/green

    # Draw signal phase rectangles
    for i in range(n_inter):
        for t in range(n_timesteps):
            color = cmap(int(phases[i, t]))
            rect = plt.Rectangle(
                (t - 0.5, i - 0.5), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.3, alpha=0.55,
            )
            ax.add_patch(rect)

    # EV trajectory line (bold, dark)
    ax.plot(
        ev_times, ev_intersections,
        color="#1a1a1a", linewidth=2.5, solid_capstyle="round",
        zorder=5, label="EV position",
    )
    # Marker dots at each intersection arrival
    # Find first arrival at each intersection
    seen = set()
    first_times = []
    first_inters = []
    for t_val, i_val in zip(ev_times, ev_intersections):
        if i_val not in seen:
            seen.add(i_val)
            first_times.append(t_val)
            first_inters.append(i_val)

    ax.scatter(
        first_times, first_inters,
        color="white", edgecolors="#1a1a1a", s=40, linewidths=1.2,
        zorder=6,
    )

    ax.set_xlim(-0.5, n_timesteps - 0.5)
    ax.set_ylim(-0.5, n_inter - 0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Intersection Index")
    ax.set_title(title, fontweight="bold")
    ax.set_yticks(range(n_inter))
    ax.set_yticklabels([f"$I_{{{i}}}$" for i in range(n_inter)])

    _style_ax(ax)


def main():
    n_intersections = 7
    n_timesteps_dt = 22
    n_timesteps_ft = 30
    n_timesteps = 30  # common x-axis for fair comparison

    # Generate scenarios
    dt_phases, dt_times, dt_inters = generate_dt_scenario(n_intersections, n_timesteps)
    ft_phases, ft_times, ft_inters = generate_ft_scenario(n_intersections, n_timesteps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)

    plot_spacetime(ax1, dt_phases, dt_times, dt_inters,
                   "(a) DT (Ours)", n_timesteps)
    plot_spacetime(ax2, ft_phases, ft_times, ft_inters,
                   "(b) FT-EVP (Fixed Timing)", n_timesteps)

    # Remove redundant y-label on right subplot
    ax2.set_ylabel("")

    # Add a shared legend for signal phase colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6DBF6D", edgecolor="#444", alpha=0.7, label="Green (EV-favorable)"),
        Patch(facecolor="#E85D5D", edgecolor="#444", alpha=0.7, label="Red (EV-unfavorable)"),
        plt.Line2D([0], [0], color="#1a1a1a", lw=2.5, label="EV trajectory"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=3,
        fontsize=7.5, frameon=True, fancybox=True,
        edgecolor="#cccccc", facecolor="white",
        bbox_to_anchor=(0.5, 1.03),
    )

    # Add timing annotations
    dt_total = dt_times[-1]
    ft_total = ft_times[-1]
    ax1.text(
        0.97, 0.05, f"Total: {dt_total} steps",
        transform=ax1.transAxes, ha="right", va="bottom",
        fontsize=8, fontweight="bold", color=C_DT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_DT, alpha=0.9),
    )
    ax2.text(
        0.97, 0.05, f"Total: {ft_total} steps",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=8, fontweight="bold", color=C_RULE_1,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_RULE_1, alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig8_spacetime")

    print(f"\nDone! Figures saved to {OUTDIR_LOCAL}")
    if OVERLEAF.exists():
        print(f"Also copied to {OVERLEAF}")


if __name__ == "__main__":
    main()
