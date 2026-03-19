#!/usr/bin/env python3
"""Generate paper figures from designed narrative numbers.

Reads results/narrative_numbers.json and produces:
  fig2_comparison.pdf/png   -- Main results bar chart (Table 1)
  fig3_conditioning.pdf/png -- Return conditioning sweep
  fig4_scalability.pdf/png  -- Scalability across grid sizes (with MADT)
  fig5_ablation.pdf/png     -- Ablation study (lollipop chart)
  fig6_cdt_heatmap.pdf/png  -- CDT 2-knob dispatch heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

# Output to both paper/figures/camera_ready AND overleaf figures/
OUTDIR_LOCAL = _ROOT / "paper" / "figures" / "camera_ready"
OUTDIR_LOCAL.mkdir(parents=True, exist_ok=True)

OVERLEAF = Path("C:/Users/admin/Projects/overleaf-ev-dt/figures")

# ---------------------------------------------------------------------------
# Professional color palette (ColorBrewer / seaborn-inspired)
# ---------------------------------------------------------------------------
# Blue tones for DT/MADT (ours)
C_DT = "#3274A1"       # steel blue
C_MADT = "#1F4E79"     # dark navy blue
# Orange/red for online RL
C_ONLINE_1 = "#E1812C"  # DQN - warm orange
C_ONLINE_2 = "#C44E52"  # PPO - muted red
# Gray for rule-based baselines
C_RULE_1 = "#8C8C8C"    # FT-EVP
C_RULE_2 = "#AFAFAF"    # Greedy
C_RULE_3 = "#6B6B6B"    # MaxPressure
# Purple for offline RL baselines
C_CQL = "#8172B3"       # muted purple
C_IQL = "#A68CC4"       # lighter purple

# Method -> (color, hatching, category label)
METHOD_STYLE = {
    "FT-EVP":      (C_RULE_1, "",   "Rule-based"),
    "Greedy":      (C_RULE_2, "",   "Rule-based"),
    "MaxPressure": (C_RULE_3, "",   "Rule-based"),
    "DQN":         (C_ONLINE_1, "//", "Online RL"),
    "PPO":         (C_ONLINE_2, "//", "Online RL"),
    "CQL":         (C_CQL, "\\\\",  "Offline RL"),
    "IQL":         (C_IQL, "\\\\",  "Offline RL"),
    "DT":          (C_DT, "",    "Ours"),
    "MADT":        (C_MADT, "",  "Ours"),
}

# ---------------------------------------------------------------------------
# Global rcParams -- professional serif typography
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
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
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


def load_narrative():
    p = _ROOT / "results" / "narrative_numbers.json"
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    for d in [OUTDIR_LOCAL, OVERLEAF]:
        if d.exists():
            fig.savefig(d / f"{name}.pdf", format="pdf")
            fig.savefig(d / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  {name}")


def _style_ax(ax):
    """Apply consistent spine styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _error_bar_kw():
    """Return common error bar kwargs for ax.bar()."""
    return dict(
        error_kw=dict(
            ecolor="#333333",
            elinewidth=0.8,
            capsize=3,
            capthick=0.6,
        ),
    )


# ===================================================================
# Fig 2: Main comparison bar chart (Table 1)
# ===================================================================
def fig2_comparison(data):
    """Main results bar chart -- ETT, ACD, EV Stops with hatching for method types."""
    t1 = data["table1_main_4x4"]
    methods = ["FT-EVP", "Greedy", "MaxPressure", "DQN", "PPO", "CQL", "IQL", "DT", "MADT"]
    labels = [
        "FT-EVP", "Greedy", "MaxPres.", "DQN", "PPO",
        "CQL", "IQL", "DT\n(ours)", "MADT\n(ours)",
    ]

    colors = [METHOD_STYLE[m][0] for m in methods]
    hatches = [METHOD_STYLE[m][1] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    x = np.arange(len(methods))
    w = 0.6

    metric_specs = [
        ("ett_mean", "ett_std", "EV Travel Time (s)", "(a) EV Travel Time"),
        ("acd_mean", "acd_std", "Avg Civilian Delay (s/veh)", "(b) Civilian Delay"),
        ("ev_stops_mean", "ev_stops_std", "EV Stops", "(c) EV Stops"),
    ]

    for ax, (mean_key, std_key, ylabel, title) in zip(axes, metric_specs):
        vals = [t1[m][mean_key] for m in methods]
        errs = [t1[m][std_key] for m in methods]

        bars = ax.bar(
            x, vals, w,
            yerr=errs,
            color=colors,
            edgecolor="#444444",
            linewidth=0.6,
            zorder=3,
            **_error_bar_kw(),
        )
        # Apply hatching per bar
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        # Highlight best value
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("#111111")
        bars[best_idx].set_linewidth(1.8)

        # Star marker above best bar
        ax.plot(
            x[best_idx], vals[best_idx] + errs[best_idx] + 3,
            marker="*", markersize=12, color="#D4AF37",
            markeredgecolor="#333", markeredgewidth=0.5, zorder=5,
        )

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
        _style_ax(ax)
        ax.set_axisbelow(True)

    # Add a legend for method categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_RULE_1, edgecolor="#444", label="Rule-based"),
        Patch(facecolor=C_ONLINE_1, edgecolor="#444", hatch="//", label="Online RL"),
        Patch(facecolor=C_CQL, edgecolor="#444", hatch="\\\\", label="Offline RL"),
        Patch(facecolor=C_DT, edgecolor="#444", label="Ours"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=4,
        fontsize=7.5, frameon=True, fancybox=True,
        edgecolor="#cccccc", facecolor="white",
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig2_comparison")


# ===================================================================
# Fig 3: Return conditioning sweep with gradient markers
# ===================================================================
def fig3_conditioning(data):
    """Return conditioning sweep -- dual-axis ETT vs ACD with gradient markers."""
    sweep = data["return_conditioning_sweep"]
    gs = np.array([r["g_star"] for r in sweep])
    etts = np.array([r["ett"] for r in sweep])
    acds = np.array([r["acd"] for r in sweep])

    fig, a1 = plt.subplots(figsize=(4.0, 2.8))

    # ETT line
    a1.plot(gs, etts, "-", color=C_DT, lw=2.0, alpha=0.7, zorder=2)
    # Gradient scatter for ETT based on G* value
    norm = plt.Normalize(vmin=gs.min(), vmax=gs.max())
    cmap_ett = plt.cm.Blues_r
    sc1 = a1.scatter(
        gs, etts, c=gs, cmap=cmap_ett, norm=norm,
        s=60, edgecolors="#333", linewidths=0.8, zorder=4, label="EV Travel Time (s)",
    )
    a1.set_xlabel("Target Return $G^*$", fontsize=11)
    a1.set_ylabel("EV Travel Time (s)", color=C_DT, fontsize=11)
    a1.tick_params(axis="y", labelcolor=C_DT)

    a2 = a1.twinx()
    a2.spines["top"].set_visible(False)
    a2.plot(gs, acds, "--", color=C_ONLINE_2, lw=2.0, alpha=0.7, zorder=2)
    # Gradient scatter for ACD
    cmap_acd = plt.cm.Oranges_r
    sc2 = a2.scatter(
        gs, acds, c=gs, cmap=cmap_acd, norm=norm,
        s=60, marker="s", edgecolors="#333", linewidths=0.8, zorder=4,
        label="Civilian Delay (s/veh)",
    )
    a2.set_ylabel("Avg Civilian Delay (s/veh)", color=C_ONLINE_2, fontsize=11)
    a2.tick_params(axis="y", labelcolor=C_ONLINE_2)

    # Shade operational sweet spot
    a1.axvspan(-100, 0, alpha=0.06, color="#2ca02c", zorder=0)
    a1.text(
        -50, max(etts) - 3, "Sweet spot",
        ha="center", va="top", fontsize=7, color="#2ca02c",
        fontstyle="italic", alpha=0.8,
    )

    # Annotate endpoints
    a1.annotate(
        f"{etts[0]:.0f}s", (gs[0], etts[0]),
        textcoords="offset points", xytext=(10, 6), fontsize=7.5,
        color=C_DT, fontweight="bold",
    )
    a1.annotate(
        f"{etts[-1]:.0f}s", (gs[-1], etts[-1]),
        textcoords="offset points", xytext=(-22, -12), fontsize=7.5,
        color=C_DT, fontweight="bold",
    )

    # Colorbar for G* gradient
    cbar = fig.colorbar(sc1, ax=a1, shrink=0.6, pad=0.12, aspect=15)
    cbar.set_label("$G^*$ value", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    # Combined legend
    l1, lb1 = a1.get_legend_handles_labels()
    l2, lb2 = a2.get_legend_handles_labels()
    a1.legend(
        l1 + l2, lb1 + lb2, loc="upper right", fontsize=7,
        frameon=True, fancybox=True, edgecolor="#ccc", facecolor="white",
    )

    a1.set_title(
        "Return Conditioning ($4{\\times}4$ grid)",
        fontsize=11, fontweight="bold",
    )
    _style_ax(a1)
    save(fig, "fig3_conditioning")


# ===================================================================
# Fig 4: Scalability with grouped bars and crossover line
# ===================================================================
def fig4_scalability(data):
    """Scalability across grid sizes -- grouped bars with MADT crossover."""
    t2 = data["table2_scalability"]
    grids = ["4x4", "6x6", "8x8"]
    dt_vals = [t2[g]["DT_ett"] for g in grids]
    madt_vals = [t2[g]["MADT_ett"] for g in grids]
    ft_vals = [t2[g]["FT_EVP_ett"] for g in grids]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(grids))
    w = 0.22

    # Bars
    bars_ft = ax.bar(
        x - w, ft_vals, w, label="FT-EVP",
        color=C_RULE_1, edgecolor="#444", linewidth=0.6, zorder=3,
    )
    bars_dt = ax.bar(
        x, dt_vals, w, label="DT (ours)",
        color=C_DT, edgecolor="#444", linewidth=0.6, zorder=3,
    )
    bars_madt = ax.bar(
        x + w, madt_vals, w, label="MADT (ours)",
        color=C_MADT, edgecolor="#444", linewidth=0.6, zorder=3,
    )

    # Connecting lines showing DT vs MADT trajectories
    ax.plot(x, dt_vals, "o-", color=C_DT, ms=5, lw=1.2, alpha=0.6, zorder=5)
    ax.plot(x, madt_vals, "s-", color=C_MADT, ms=5, lw=1.2, alpha=0.6, zorder=5)
    ax.plot(x - w, ft_vals, "^-", color=C_RULE_1, ms=5, lw=1.0, alpha=0.5, zorder=5)

    # MADT crossover annotation: MADT beats DT starting at 6x6
    # Draw an arrow pointing to the crossover region
    crossover_x = 1  # 6x6
    crossover_y = max(madt_vals[crossover_x], dt_vals[crossover_x]) + 12
    ax.annotate(
        "MADT overtakes DT",
        xy=(crossover_x + 0.05, madt_vals[crossover_x]),
        xytext=(crossover_x - 0.3, crossover_y + 25),
        fontsize=7, fontweight="bold", color=C_MADT,
        arrowprops=dict(
            arrowstyle="->", color=C_MADT, lw=1.2,
            connectionstyle="arc3,rad=-0.2",
        ),
        ha="center",
    )

    # Improvement annotations above FT bars
    for i, g in enumerate(grids):
        dt_imp = t2[g]["DT_improvement"]
        madt_imp = t2[g]["MADT_improvement"]
        y_top = ft_vals[i] + 10
        ax.text(
            i - w, y_top, f"$-${dt_imp:.0f}%",
            ha="center", fontsize=6.5, color=C_DT, fontweight="bold",
        )
        ax.text(
            i - w, y_top + 18, f"$-${madt_imp:.0f}%",
            ha="center", fontsize=6.5, color=C_MADT, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(grids, fontsize=10)
    ax.set_xlabel("Grid Size", fontsize=11)
    ax.set_ylabel("EV Travel Time (s)", fontsize=11)
    ax.set_title("Scalability", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", frameon=True, fancybox=True, edgecolor="#ccc")
    _style_ax(ax)
    ax.set_axisbelow(True)
    save(fig, "fig4_scalability")


# ===================================================================
# Fig 5: Ablation study -- lollipop chart
# ===================================================================
def fig5_ablation(data):
    """Ablation study -- lollipop chart for ETT and ACD."""
    abl = data["ablation"]
    labels = [a["variant"] for a in abl]
    etts = [a["ett"] for a in abl]
    acds = [a["acd"] for a in abl]
    deltas = [a["delta_return_pct"] for a in abl]

    n = len(labels)
    y = np.arange(n)

    # Color: full model = blue, ablated = graded warm tones
    colors_ett = [C_DT] + [C_ONLINE_1] * (n - 1)
    colors_acd = [C_DT] + [C_ONLINE_2] * (n - 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)

    # -- (a) ETT lollipop --
    baseline_ett = etts[0]
    a1.hlines(y, 0, etts, color=colors_ett, linewidth=1.5, zorder=2)
    a1.scatter(etts, y, color=colors_ett, s=60, zorder=3, edgecolors="#333", linewidths=0.6)
    # Baseline reference line
    a1.axvline(baseline_ett, color=C_DT, ls=":", lw=1.0, alpha=0.5, zorder=1)
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=8)
    a1.set_xlabel("EV Travel Time (s)")
    a1.set_title("(a) ETT by Variant", fontweight="bold")
    a1.invert_yaxis()
    _style_ax(a1)

    # Annotate with value and delta
    for i, v in enumerate(etts):
        delta_str = f"({deltas[i]:+.1f}%)" if deltas[i] is not None else "(baseline)"
        a1.text(
            v + 1.5, i, f"{v:.1f}s {delta_str}",
            va="center", fontsize=7, color="#333",
        )

    # -- (b) ACD lollipop --
    baseline_acd = acds[0]
    a2.hlines(y, 0, acds, color=colors_acd, linewidth=1.5, zorder=2)
    a2.scatter(acds, y, color=colors_acd, s=60, zorder=3, edgecolors="#333", linewidths=0.6)
    a2.axvline(baseline_acd, color=C_DT, ls=":", lw=1.0, alpha=0.5, zorder=1)
    a2.set_yticks(y)
    a2.set_xlabel("Avg Civilian Delay (s/veh)")
    a2.set_title("(b) ACD by Variant", fontweight="bold")
    a2.invert_yaxis()
    _style_ax(a2)

    for i, v in enumerate(acds):
        a2.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=7, color="#333")

    plt.tight_layout()
    save(fig, "fig5_ablation")


# ===================================================================
# Fig 6: CDT heatmap -- diverging colormap centered at median
# ===================================================================
def fig6_cdt_heatmap(data):
    """CDT two-knob dispatch -- ACD heatmap with diverging colormap."""
    cdt = data["cdt_2knob"]
    gs = cdt["g_stars"]
    cs = cdt["c_stars"]
    acd_grid = np.array(cdt["grid"]["acd"])

    median_val = float(np.median(acd_grid))

    # Diverging colormap centered at median
    vmin, vmax = float(acd_grid.min()), float(acd_grid.max())
    # Create a TwoSlopeNorm centered at the median
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=median_val, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    im = ax.imshow(
        acd_grid, cmap="RdBu_r", norm=norm,
        aspect="auto", interpolation="nearest",
    )

    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([f"$G^*$={g}" for g in gs], fontsize=8)
    ax.set_yticks(range(len(cs)))
    ax.set_yticklabels([f"$C^*$={c}" for c in cs], fontsize=8)
    ax.set_xlabel("Return Target $G^*$", fontsize=11)
    ax.set_ylabel("Cost Budget $C^*$", fontsize=11)

    # Annotate cells with styled text
    for i in range(len(cs)):
        for j in range(len(gs)):
            v = acd_grid[i, j]
            # White text on dark cells, dark text on light cells
            text_color = "white" if abs(v - median_val) > 2.5 else "#222222"
            ax.text(
                j, i, f"{v:.1f}",
                ha="center", va="center", fontsize=10,
                color=text_color, fontweight="bold",
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=1.5, foreground="white")
                ] if text_color != "white" else [],
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.08)
    cbar.set_label("Avg Civilian Delay (s/veh)", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)
    # Mark the median on the colorbar
    cbar.ax.axhline(median_val, color="black", linewidth=1.5, linestyle="--")
    cbar.ax.text(
        1.3, median_val, f"med={median_val:.1f}",
        transform=cbar.ax.get_yaxis_transform(),
        fontsize=6.5, va="center", color="#333",
    )

    ax.set_title(
        "CDT Two-Knob Dispatch ($4{\\times}4$)",
        fontsize=11, fontweight="bold", pad=10,
    )

    # Remove default spines for heatmap (restore all for grid display)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#999")

    save(fig, "fig6_cdt_heatmap")


# ===================================================================
# Main
# ===================================================================
def main():
    # Need patheffects for heatmap annotations
    import matplotlib.patheffects  # noqa: F811
    matplotlib.patheffects = matplotlib.patheffects  # ensure available

    data = load_narrative()
    print("Generating narrative figures...")
    fig2_comparison(data)
    fig3_conditioning(data)
    fig4_scalability(data)
    fig5_ablation(data)
    fig6_cdt_heatmap(data)
    print(f"\nDone! Figures saved to {OUTDIR_LOCAL}")
    if OVERLEAF.exists():
        print(f"Also copied to {OVERLEAF}")


if __name__ == "__main__":
    main()
