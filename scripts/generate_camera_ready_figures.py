#!/usr/bin/env python3
"""Generate publication-quality camera-ready figures for the EV-DT paper (AAJ 2026).

Produces 6 figures as both PDF (vector) and PNG (300 DPI) in paper/figures/camera_ready/.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Style configuration -- NeurIPS / camera-ready
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
    }
)

# Color palette
C_DT = "#2166AC"  # blue for DT
C_BASELINE = "#B2182B"  # red for baselines
C_MADT = "#1B7837"  # green for MADT
C_DT_LIGHT = "#6BAED6"
C_BASELINE_LIGHT = "#FC8D59"
C_GREEDY = "#D6604D"
C_RANDOM = "#878787"
C_FT_EVP = "#E08214"
C_DQN = "#B2ABD2"
C_PPO = "#5E4FA2"

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "results")
OUTDIR = os.path.join(PROJ, "paper", "figures", "camera_ready")
os.makedirs(OUTDIR, exist_ok=True)

SINGLE_COL = 3.25  # inches
DOUBLE_COL = 6.75


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  saved {path}")
    plt.close(fig)


# ===================================================================
# Figure 1: Architecture Diagram (double-column)
# ===================================================================
def fig1_architecture():
    """Draw DT + MADT architecture using matplotlib patches and arrows."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # --- helper functions ---
    def box(x, y, w, h, color, label, fs=7, ec="black", lw=0.8, alpha=0.85):
        r = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
        )
        return r

    def arrow(x1, y1, x2, y2, color="black"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    # ---- LEFT: Single-agent DT ----
    ax.text(2.35, 4.0, "Decision Transformer (Single-Agent)", ha="center",
            fontsize=9, fontweight="bold")

    # Input tokens: R, s, a
    token_colors = ["#DCEDC8", "#BBDEFB", "#FFE0B2"]  # green, blue, orange tint
    token_labels = [r"$\hat{R}_1$", r"$s_1$", r"$a_1$",
                    r"$\hat{R}_2$", r"$s_2$", r"$a_2$",
                    r"$\hat{R}_t$", r"$s_t$", r"$a_t$"]
    for i, (lbl, c) in enumerate(
        zip(token_labels, token_colors * 3)
    ):
        x0 = 0.15 + i * 0.5
        box(x0, 0.15, 0.45, 0.45, c, lbl, fs=6.5)

    # Ellipsis
    ax.text(3.65, 0.37, "...", fontsize=10, ha="center", va="center")

    # Token embeddings bar
    box(0.15, 0.85, 4.55, 0.35, "#E0E0E0", "Token + Positional Embeddings", fs=7)

    # Arrows up from tokens to embeddings
    for i in range(9):
        x0 = 0.15 + i * 0.5 + 0.225
        if i not in (6,):  # skip ellipsis area
            arrow(x0, 0.60, x0, 0.85)

    # Causal Transformer block
    box(0.15, 1.45, 4.55, 0.55, C_DT_LIGHT, "Causal Transformer (L layers)", fs=8)
    arrow(2.425, 1.20, 2.425, 1.45)

    # Causal mask illustration (small triangle)
    mask_x, mask_y = 3.8, 1.55
    triangle = plt.Polygon(
        [[mask_x, mask_y], [mask_x + 0.35, mask_y], [mask_x + 0.35, mask_y + 0.35]],
        closed=True, facecolor="#1565C0", alpha=0.4, edgecolor="black", linewidth=0.5,
    )
    ax.add_patch(triangle)
    ax.text(mask_x + 0.18, mask_y - 0.08, "mask", fontsize=5, ha="center")

    # Action prediction head
    box(0.15, 2.25, 4.55, 0.4, "#C5CAE9", "Linear Head → $\\hat{a}_t$", fs=8)
    arrow(2.425, 2.0, 2.425, 2.25)

    # Return-conditioned arrow
    ax.annotate(
        "Return\nConditioning\n$G^*$",
        xy=(0.15, 0.37),
        xytext=(-0.05, 1.5),
        fontsize=6,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_DT, lw=1.2, connectionstyle="arc3,rad=0.3"),
        color=C_DT,
        fontweight="bold",
    )

    # ---- RIGHT: MADT extension ----
    ax.text(7.75, 4.0, "MADT (Multi-Agent)", ha="center",
            fontsize=9, fontweight="bold")

    # Agent boxes
    agent_colors = [C_DT_LIGHT, "#A5D6A7", "#FFCC80"]
    agent_labels = ["Agent 1\n(DT)", "Agent 2\n(DT)", "Agent $N$\n(DT)"]
    agent_xs = [5.6, 7.1, 8.6]
    for i, (x0, c, lbl) in enumerate(zip(agent_xs, agent_colors, agent_labels)):
        box(x0, 0.15, 1.1, 0.85, c, lbl, fs=6.5)

    # Ellipsis
    ax.text(8.35, 0.57, "...", fontsize=10, ha="center", va="center")

    # Local embeddings
    for x0 in agent_xs:
        box(x0, 1.2, 1.1, 0.35, "#E0E0E0", "Embed", fs=6.5)
        arrow(x0 + 0.55, 1.0, x0 + 0.55, 1.2)

    # GAT layer
    box(5.6, 1.8, 4.1, 0.55, C_MADT, "", fs=8)
    ax.text(
        7.65, 2.075, "Graph Attention Network (GAT)", ha="center", va="center",
        fontsize=8, fontweight="bold", color="white",
    )
    for x0 in agent_xs:
        arrow(x0 + 0.55, 1.55, x0 + 0.55, 1.8)

    # Communication arrows between agents inside GAT
    arrow(6.3, 2.0, 7.5, 2.15, color=C_MADT)
    arrow(7.9, 2.15, 9.0, 2.0, color=C_MADT)
    arrow(9.0, 2.0, 7.5, 2.15, color=C_MADT)

    # Per-agent action heads
    for x0 in agent_xs:
        box(x0, 2.6, 1.1, 0.4, "#C5CAE9", "$\\hat{a}_t^i$", fs=7)
        arrow(x0 + 0.55, 2.35, x0 + 0.55, 2.6)

    # Divider line
    ax.axvline(x=5.1, ymin=0.02, ymax=0.95, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.5)

    _save(fig, "fig1_architecture")


# ===================================================================
# Figure 2: Main Results Bar Chart (single-column)
# ===================================================================
def fig2_main_results():
    dt_full = _load("dt_4x4_full.json")
    baselines_rl = _load("baseline_comparison.json")
    bc = dt_full["baseline_comparison"]

    # Methods: FT-EVP, Greedy, Random, DQN, PPO, DT
    methods = ["FT-EVP", "Greedy", "Random", "DQN", "PPO", "DT"]
    colors = [C_FT_EVP, C_GREEDY, C_RANDOM, C_DQN, C_PPO, C_DT]

    ev_times = [
        bc["FT-EVP"]["ev_travel_time_mean"],
        bc["Greedy"]["ev_travel_time_mean"],
        bc["Random"]["ev_travel_time_mean"],
        baselines_rl["DQN"]["avg_ev_travel_time"],
        baselines_rl["PPO"]["avg_ev_travel_time"],
        bc["DT (target=50)"]["ev_travel_time_mean"],
    ]
    ev_stds = [
        bc["FT-EVP"]["ev_travel_time_std"],
        bc["Greedy"]["ev_travel_time_std"],
        bc["Random"]["ev_travel_time_std"],
        0.0,
        0.0,
        bc["DT (target=50)"]["ev_travel_time_std"],
    ]
    returns = [
        bc["FT-EVP"]["mean_return"],
        bc["Greedy"]["mean_return"],
        bc["Random"]["mean_return"],
        baselines_rl["DQN"]["avg_return"],
        baselines_rl["PPO"]["avg_return"],
        bc["DT (target=50)"]["mean_return"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    x = np.arange(len(methods))
    w = 0.6

    # EV Travel Time
    bars1 = ax1.bar(x, ev_times, w, color=colors, edgecolor="black", linewidth=0.5,
                    yerr=ev_stds, capsize=2, error_kw={"linewidth": 0.8})
    ax1.set_ylabel("EV Travel Time (steps)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
    ax1.set_title("(a) EV Travel Time")

    # Return
    bars2 = ax2.bar(x, returns, w, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Return")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
    ax2.set_title("(b) Episode Return")

    # Highlight DT bar with a star
    for ax_i, vals in [(ax1, ev_times), (ax2, returns)]:
        dt_idx = 5
        val = vals[dt_idx]
        ax_i.annotate(
            "DT",
            xy=(dt_idx, val),
            xytext=(dt_idx, val + abs(val) * 0.08),
            fontsize=7,
            fontweight="bold",
            color=C_DT,
            ha="center",
        )

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig2_main_results")


# ===================================================================
# Figure 3: Return Conditioning Sweep (single-column)
# ===================================================================
def fig3_return_sweep():
    dt_full = _load("dt_4x4_full.json")
    sweep = dt_full["return_conditioning_sweep"]

    targets = [50, 25, 0, -25, -50, -100, -200]
    ev_times, bg_delays, ev_stds, bg_stds = [], [], [], []
    for t in targets:
        key = f"target_{t}"
        d = sweep[key]
        ev_times.append(d["ev_travel_time_mean"])
        ev_stds.append(d["ev_travel_time_std"])
        bg_delays.append(d["background_delay_mean"])
        bg_stds.append(d["background_delay_std"])

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.4))
    ax2 = ax1.twinx()

    x = np.arange(len(targets))
    labels = [str(t) for t in targets]

    l1 = ax1.errorbar(
        x, ev_times, yerr=ev_stds, fmt="o-", color=C_DT, markersize=4,
        capsize=2, linewidth=1.2, label="EV Travel Time",
    )
    l2 = ax2.errorbar(
        x, bg_delays, yerr=bg_stds, fmt="s--", color=C_BASELINE, markersize=4,
        capsize=2, linewidth=1.2, label="Background Delay",
    )

    ax1.set_xlabel("Target Return $G^*$")
    ax1.set_ylabel("EV Travel Time (steps)", color=C_DT)
    ax2.set_ylabel("Background Delay", color=C_BASELINE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor=C_DT)
    ax2.tick_params(axis="y", labelcolor=C_BASELINE)

    # Pareto-optimal region annotation
    ax1.axvspan(4.5, 6.5, alpha=0.08, color=C_MADT)
    ax1.annotate(
        "Pareto-optimal\nregion",
        xy=(5.5, min(ev_times) - 0.3),
        fontsize=7,
        ha="center",
        color=C_MADT,
        fontweight="bold",
    )

    # Combined legend
    lines = [l1, l2]
    labels_leg = [li.get_label() for li in lines]
    ax1.legend(lines, labels_leg, loc="upper left", framealpha=0.9, fontsize=7)

    ax1.set_title("Return Conditioning Sweep (4$\\times$4 Grid)")
    fig.tight_layout()
    _save(fig, "fig3_return_sweep")


# ===================================================================
# Figure 4: Scalability (single-column)
# ===================================================================
def fig4_scalability():
    data = _load("scalability.json")
    grids = ["3x3", "4x4", "6x6"]
    dt_times = [data[g]["methods"]["DT"]["ev_time"] for g in grids]
    ft_times = [data[g]["methods"]["FT-EVP"]["ev_time"] for g in grids]
    improvements = [
        (ft - dt) / ft * 100 for ft, dt in zip(ft_times, dt_times)
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    x = np.arange(len(grids))
    w = 0.32

    bars_ft = ax.bar(x - w / 2, ft_times, w, color=C_FT_EVP, edgecolor="black",
                     linewidth=0.5, label="FT-EVP")
    bars_dt = ax.bar(x + w / 2, dt_times, w, color=C_DT, edgecolor="black",
                     linewidth=0.5, label="DT")

    # Improvement labels
    for i, (ft_v, dt_v, imp) in enumerate(zip(ft_times, dt_times, improvements)):
        mid = max(ft_v, dt_v) + 1.0
        ax.text(i, mid, f"$\\downarrow${imp:.0f}%", ha="center", fontsize=7,
                fontweight="bold", color=C_DT)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("EV Travel Time (steps)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({data[g]['n_intersections']} int.)" for g in grids], fontsize=7)
    ax.legend(framealpha=0.9)
    ax.set_title("Scalability: DT vs FT-EVP")
    fig.tight_layout()
    _save(fig, "fig4_scalability")


# ===================================================================
# Figure 5: Ablation Study (single-column)
# ===================================================================
def fig5_ablation():
    data = _load("ablation_results.json")
    abl = data["ablations"]

    variants = ["full_dt", "no_rtg", "short_context", "expert_only"]
    labels = ["Full DT", "w/o RTG", "Short Context\n(K=3)", "Expert-Only"]
    returns = [abl[v]["evaluation"]["mean_return"] for v in variants]
    stds = [abl[v]["evaluation"]["std_return"] for v in variants]
    colors_abl = [C_DT, C_BASELINE_LIGHT, C_BASELINE_LIGHT, C_BASELINE_LIGHT]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))

    y = np.arange(len(variants))
    bars = ax.barh(y, returns, xerr=stds, height=0.55, color=colors_abl,
                   edgecolor="black", linewidth=0.5, capsize=2,
                   error_kw={"linewidth": 0.8})

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Return")
    ax.set_title("Ablation Study (3$\\times$3 Grid)")
    ax.invert_yaxis()

    # Degradation annotations
    full_ret = returns[0]
    for i in range(1, len(returns)):
        delta = returns[i] - full_ret
        pct = delta / abs(full_ret) * 100
        # Position label at the right end of the bar (inside, near tip)
        xpos = min(returns[i], -50)
        ax.text(
            xpos + 15, i, f"{pct:+.0f}%",
            va="center", ha="left", fontsize=7, color="white",
            fontweight="bold",
        )

    fig.tight_layout()
    _save(fig, "fig5_ablation")


# ===================================================================
# Figure 6: CDT 2-Knob Heatmap (single-column)
# ===================================================================
def fig6_cdt_heatmap():
    data = _load("cdt_2knob.json")

    g_vals = sorted(set(d["g_star"] for d in data))
    c_vals = sorted(set(d["c_star"] for d in data))

    queue_grid = np.full((len(c_vals), len(g_vals)), np.nan)
    for d in data:
        gi = g_vals.index(d["g_star"])
        ci = c_vals.index(d["c_star"])
        queue_grid[ci, gi] = d["queue"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    im = ax.imshow(queue_grid, cmap="YlOrRd", aspect="auto", origin="lower")

    ax.set_xticks(range(len(g_vals)))
    ax.set_xticklabels([f"{v:.0f}" for v in g_vals])
    ax.set_yticks(range(len(c_vals)))
    ax.set_yticklabels([f"{v:.0f}" for v in c_vals])
    ax.set_xlabel("Target Return $G^*$")
    ax.set_ylabel("Congestion Target $C^*$")
    ax.set_title("CDT 2-Knob Dispatch: Avg Queue")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label("Avg Queue Length", fontsize=8)

    # Annotate cells
    for ci in range(len(c_vals)):
        for gi in range(len(g_vals)):
            val = queue_grid[ci, gi]
            if not np.isnan(val):
                color = "white" if val > 100 else "black"
                ax.text(gi, ci, f"{val:.0f}", ha="center", va="center",
                        fontsize=6.5, color=color, fontweight="bold")

    # Region annotations
    ax.annotate(
        "Low\ndisruption",
        xy=(0, 0),
        xytext=(0.3, -0.1),
        fontsize=6,
        color=C_MADT,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_MADT, lw=0.8),
    )
    ax.annotate(
        "High\ndisruption",
        xy=(len(g_vals) - 1, len(c_vals) - 1),
        xytext=(len(g_vals) - 1 - 0.3, len(c_vals) - 1 + 0.15),
        fontsize=6,
        color=C_BASELINE,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_BASELINE, lw=0.8),
    )

    fig.tight_layout()
    _save(fig, "fig6_cdt_heatmap")


# ===================================================================
# Main
# ===================================================================
def main():
    print("Generating camera-ready figures...")
    fig1_architecture()
    print("[1/6] Architecture diagram done")
    fig2_main_results()
    print("[2/6] Main results done")
    fig3_return_sweep()
    print("[3/6] Return sweep done")
    fig4_scalability()
    print("[4/6] Scalability done")
    fig5_ablation()
    print("[5/6] Ablation done")
    fig6_cdt_heatmap()
    print("[6/6] CDT heatmap done")
    print(f"\nAll figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
