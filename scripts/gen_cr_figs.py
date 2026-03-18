#!/usr/bin/env python3
"""Generate camera-ready figures from real experimental data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUTDIR = _ROOT / "paper" / "figures" / "camera_ready"
OUTDIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
})


def load(name):
    p = _ROOT / "results" / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save(fig, name):
    fig.savefig(OUTDIR / f"{name}.pdf", format="pdf")
    fig.savefig(OUTDIR / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  {name}")


def main():
    print("Generating camera-ready figures...")

    # Fig 1: Architecture
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    c = {"r": "#4ECDC4", "s": "#45B7D1", "a": "#F7DC6F",
         "t": "#BB8FCE", "o": "#E74C3C", "e": "#85C1E9"}
    toks = [(0.5, "R\u0302\u2081", c["r"]), (1.5, "s\u2081", c["s"]),
            (2.5, "a\u2081", c["a"]), (4.0, "R\u0302\u2082", c["r"]),
            (5.0, "s\u2082", c["s"]), (5.95, "a\u2082", c["a"]),
            (7.5, "R\u0302\u2095", c["r"]), (8.5, "s\u2095", c["s"]),
            (9.5, "a\u2095", c["a"])]
    for x, lb, cl in toks:
        ax.add_patch(FancyBboxPatch((x - 0.35, 0.1), 0.7, 0.45,
                     boxstyle="round,pad=0.05", fc=cl, ec="k", lw=0.5, alpha=0.85))
        ax.text(x, 0.32, lb, ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(6.7, 0.32, "...", ha="center", fontsize=12, fontweight="bold")
    for x, _, _ in toks:
        ax.annotate("", xy=(x, 1.05), xytext=(x, 0.55),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
    ax.add_patch(FancyBboxPatch((-0.1, 1.05), 10.2, 0.35,
                 boxstyle="round,pad=0.05", fc=c["e"], ec="k", lw=0.5, alpha=0.6))
    ax.text(5, 1.22, "Token Embeddings + Positional Encoding", ha="center", fontsize=7.5)
    ax.annotate("", xy=(5, 2.05), xytext=(5, 1.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    for i in range(3):
        y = 2.2 + i * 0.45
        lbl = f"Causal Transformer Layer {i + 1}"
        if i == 0:
            lbl += " (+ GAT for MADT)"
        ax.add_patch(FancyBboxPatch((0.5, y - 0.15), 9, 0.35,
                     boxstyle="round,pad=0.05", fc=c["t"], ec="k", lw=0.5, alpha=0.7 - i * 0.1))
        ax.text(5, y + 0.02, lbl, ha="center", fontsize=7)
    for i, x in enumerate([1.5, 5.0, 8.5]):
        ax.add_patch(FancyBboxPatch((x - 0.5, 3.55), 1.0, 0.35,
                     boxstyle="round,pad=0.05", fc=c["o"], ec="k", lw=0.5, alpha=0.8))
        ax.text(x, 3.72, ["a\u0302\u2081", "a\u0302\u2082", "a\u0302\u2095"][i],
                ha="center", fontsize=7.5, fontweight="bold", color="white")
        ax.annotate("", xy=(x, 3.55), xytext=(x, 2.2 + 2 * 0.45 + 0.2),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.text(5, 4.15, "Predicted Actions (from state positions)", ha="center", fontsize=8, style="italic")
    save(fig, "fig1_architecture")

    # Fig 2: Comparison
    full = load("dt_4x4_full.json")
    bl = load("baseline_comparison.json")
    if full and bl:
        bc = full["baseline_comparison"]
        labs = ["FT-EVP", "Greedy", "PPO", "DQN", "DT (ours)"]
        evt = [bc["FT-EVP"]["ev_travel_time_mean"], bc["Greedy"]["ev_travel_time_mean"],
               bl["PPO"]["avg_ev_travel_time"], bl["DQN"]["avg_ev_travel_time"],
               bc["DT (target=50)"]["ev_travel_time_mean"]]
        ret = [bc["FT-EVP"]["mean_return"], bc["Greedy"]["mean_return"],
               bl["PPO"]["avg_return"], bl["DQN"]["avg_return"],
               bc["DT (target=50)"]["mean_return"]]
        clrs = ["#95a5a6", "#95a5a6", "#e67e22", "#e67e22", "#2196F3"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.2))
        x = np.arange(len(labs))
        a1.bar(x, evt, color=clrs, ec="k", lw=0.5, width=0.6)
        a1.set_xticks(x); a1.set_xticklabels(labs, rotation=25, ha="right")
        a1.set_ylabel("EV Travel Time"); a1.set_title("(a) EV Travel Time")
        a2.bar(x, [-r for r in ret], color=clrs, ec="k", lw=0.5, width=0.6)
        a2.set_xticks(x); a2.set_xticklabels(labs, rotation=25, ha="right")
        a2.set_ylabel("Neg. Return"); a2.set_title("(b) Return (lower=better)")
        plt.tight_layout()
        save(fig, "fig2_comparison")

    # Fig 3: Conditioning sweep
    if full:
        sw = full["return_conditioning_sweep"]
        targs, evts, dels = [], [], []
        for k in sorted(sw.keys(), key=lambda k: float(k.split("_")[1])):
            d = sw[k]
            targs.append(d["target_return"]); evts.append(d["ev_travel_time_mean"])
            dels.append(d["background_delay_mean"])
        fig, a1 = plt.subplots(figsize=(3.25, 2.3))
        a1.plot(targs, evts, "o-", color="#2196F3", lw=1.5, ms=4, label="EV Time")
        a1.set_xlabel("Target Return G*"); a1.set_ylabel("EV Travel Time", color="#2196F3")
        a2 = a1.twinx()
        a2.plot(targs, dels, "s--", color="#FF5722", lw=1.5, ms=4, label="Bg. Delay")
        a2.set_ylabel("Background Delay", color="#FF5722")
        a1.axvspan(-150, -50, alpha=0.08, color="green")
        l1, lb1 = a1.get_legend_handles_labels()
        l2, lb2 = a2.get_legend_handles_labels()
        a1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=6.5)
        a1.set_title("Return Conditioning (4x4)", fontsize=9, fontweight="bold")
        save(fig, "fig3_conditioning")

    # Fig 4: Scalability
    scl = load("scalability.json")
    if scl:
        grids = sorted(scl.keys())
        dt_t = [scl[g]["methods"]["DT"]["ev_time"] for g in grids]
        ft_t = [scl[g]["methods"]["FT-EVP"]["ev_time"] for g in grids]
        fig, ax = plt.subplots(figsize=(3.25, 2.0))
        x = np.arange(len(grids)); w = 0.3
        ax.bar(x - w / 2, ft_t, w, label="FT-EVP", color="#95a5a6", ec="k", lw=0.5)
        ax.bar(x + w / 2, dt_t, w, label="DT (ours)", color="#2196F3", ec="k", lw=0.5)
        for i in range(len(grids)):
            imp = (ft_t[i] - dt_t[i]) / ft_t[i] * 100
            ax.text(i, max(ft_t[i], dt_t[i]) + 0.5, f"-{imp:.0f}%",
                    ha="center", fontsize=7, fontweight="bold", color="#2196F3")
        ax.set_xticks(x); ax.set_xticklabels(grids)
        ax.set_xlabel("Grid Size"); ax.set_ylabel("EV Travel Time")
        ax.set_title("Scalability", fontsize=9, fontweight="bold"); ax.legend(fontsize=7)
        save(fig, "fig4_scalability")

    # Fig 5: Ablation
    abl = load("ablation_results.json")
    if abl:
        ab = abl["ablations"]
        vs = ["full_dt", "no_rtg", "short_context", "expert_only"]
        lbs = ["Full DT", "w/o RTG", "Short Ctx", "Expert Only"]
        rs = [ab[v]["evaluation"]["mean_return"] for v in vs]
        cs = ["#2196F3", "#FF9800", "#FF9800", "#FF9800"]
        fig, ax = plt.subplots(figsize=(3.25, 1.8))
        y = np.arange(len(lbs))
        ax.barh(y, [-r for r in rs], color=cs, ec="k", lw=0.5, height=0.5)
        ax.set_yticks(y); ax.set_yticklabels(lbs)
        ax.set_xlabel("Neg. Return (lower=better)")
        ax.set_title("Ablation Study", fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        for b, v in zip(ax.patches, rs):
            ax.text(b.get_width() + 5, b.get_y() + b.get_height() / 2,
                    f"{v:.0f}", ha="left", va="center", fontsize=7)
        save(fig, "fig5_ablation")

    # Fig 6: CDT Heatmap
    cdt = load("cdt_2knob.json")
    if cdt:
        gs = sorted(set(r["g_star"] for r in cdt))
        cs_list = sorted(set(r["c_star"] for r in cdt))
        grid = np.zeros((len(cs_list), len(gs)))
        for r in cdt:
            grid[cs_list.index(r["c_star"]), gs.index(r["g_star"])] = r["queue"]
        fig, ax = plt.subplots(figsize=(3.25, 2.5))
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(gs))); ax.set_xticklabels([f"{int(g)}" for g in gs])
        ax.set_yticks(range(len(cs_list))); ax.set_yticklabels([f"{int(c)}" for c in cs_list])
        ax.set_xlabel("Return Target G*"); ax.set_ylabel("Cost Budget C*")
        for i in range(len(cs_list)):
            for j in range(len(gs)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 90 else "black", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.9).set_label("Avg Queue", fontsize=8)
        ax.set_title("CDT Two-Knob Dispatch", fontsize=9, fontweight="bold")
        save(fig, "fig6_cdt_heatmap")

    print(f"\nDone! {len(list(OUTDIR.iterdir()))} files in {OUTDIR}")


if __name__ == "__main__":
    main()
