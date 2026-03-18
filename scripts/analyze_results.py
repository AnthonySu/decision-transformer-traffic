#!/usr/bin/env python3
"""Master analysis script for the EV-DT paper.

Loads all experiment results (main evaluation, ablation studies, scalability
tests, and return-conditioning sweep), then generates:

    1. LaTeX tables for the paper (saved to paper/tables/)
    2. Publication-quality figures (saved to paper/figures/)
    3. A text summary report printed to stdout

Usage::

    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-dir logs --paper-dir paper
    python scripts/analyze_results.py --skip-figures   # tables + report only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# I/O helpers
# ===================================================================

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        print(f"  [SKIP] {path} not found.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===================================================================
# LaTeX table generators
# ===================================================================

def _fmt(val: Any, precision: int = 2) -> str:
    """Format a numeric value for LaTeX."""
    if val is None or val == "N/A" or (isinstance(val, float) and val < 0):
        return "--"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
    return str(val)


def _fmt_pm(mean: Any, std: Any, precision: int = 2) -> str:
    """Format mean +/- std for LaTeX."""
    if mean is None or (isinstance(mean, float) and mean < 0):
        return "--"
    m = f"{float(mean):.{precision}f}"
    s = f"{float(std):.{precision}f}" if std is not None else "0"
    return f"${m} \\pm {s}$"


def generate_main_results_table(
    eval_data: Dict[str, Any],
    scenario: str = "grid-4x4-v0",
) -> str:
    """Generate the main comparison table (Table 1 in the paper).

    Columns: Method | EV Travel Time | Background Delay | Throughput |
             Signal Disruptions | Corridor Green Ratio
    """
    methods_data = eval_data.get(scenario, {})
    if not methods_data:
        return "% No data for main results table.\n"

    rows: List[str] = []
    for method_name, metrics in methods_data.items():
        ev_time = _fmt_pm(
            metrics.get("ev_travel_time_mean", metrics.get("mean_ev_travel_time")),
            metrics.get("ev_travel_time_std"),
        )
        bg_delay = _fmt_pm(
            metrics.get("background_delay_mean"),
            metrics.get("background_delay_std"),
        )
        throughput = _fmt_pm(
            metrics.get("throughput_mean"),
            metrics.get("throughput_std"),
            precision=1,
        )
        disruptions = _fmt(metrics.get("signal_disruptions_mean"), precision=1)
        green_ratio = _fmt(metrics.get("corridor_green_ratio_mean"), precision=3)

        # Escape underscores in method name for LaTeX
        safe_name = method_name.replace("_", r"\_")
        rows.append(
            f"        {safe_name} & {ev_time} & {bg_delay} & "
            f"{throughput} & {disruptions} & {green_ratio} \\\\"
        )

    table = r"""\begin{table}[t]
    \centering
    \caption{Comparison of all methods on the %s network. Lower EV travel time is better; lower background delay and higher throughput indicate less disruption to normal traffic.}
    \label{tab:main_results}
    \begin{tabular}{lccccc}
        \toprule
        Method & EV Time & BG Delay & Throughput & Disruptions & Green Ratio \\
        \midrule
%s
        \bottomrule
    \end{tabular}
\end{table}
""" % (scenario.replace("_", r"\_"), "\n".join(rows))

    return table


def generate_ablation_table(ablation_data: Dict[str, Any]) -> str:
    """Generate the ablation study table (Table 2).

    Each row is an ablation; columns show impact relative to full model.
    """
    ablations = ablation_data.get("ablations", {})
    if not ablations:
        return "% No ablation data.\n"

    rows: List[str] = []
    for abl_name, abl_result in ablations.items():
        desc = abl_result.get("description", abl_name)
        # Try to extract the best EV time from evaluation
        eval_info = abl_result.get("evaluation", {})
        ev_time = "--"
        bg_delay = "--"
        for scenario, methods in eval_info.items():
            for method_name, metrics in methods.items():
                if "DT" in method_name or "MADT" in method_name:
                    ev_val = metrics.get("mean_ev_travel_time",
                                         metrics.get("ev_travel_time_mean"))
                    if ev_val is not None and (ev_time == "--" or float(ev_val) < float(ev_time)):
                        ev_time = _fmt(ev_val)
                    bg_val = metrics.get("background_delay_mean")
                    if bg_val is not None:
                        bg_delay = _fmt(bg_val)

        train_time = _fmt(abl_result.get("training_time_sec"), precision=0)
        safe_name = abl_name.replace("_", r"\_")
        rows.append(
            f"        {safe_name} & {desc} & {ev_time} & {bg_delay} & {train_time}s \\\\"
        )

    table = r"""\begin{table}[t]
    \centering
    \caption{Ablation study results. Each row removes one component from the full model and reports the impact on key metrics.}
    \label{tab:ablation}
    \begin{tabular}{llccc}
        \toprule
        Ablation & Description & EV Time & BG Delay & Train Time \\
        \midrule
%s
        \bottomrule
    \end{tabular}
\end{table}
""" % "\n".join(rows)

    return table


def generate_scalability_table(scale_data: Dict[str, Any]) -> str:
    """Generate the scalability table (Table 3).

    Rows: grid sizes. Columns: DT train time, MADT train time,
    DT inference, MADT inference, GPU memory.
    """
    experiments = scale_data.get("grid_experiments", {})
    if not experiments:
        return "% No scalability data.\n"

    rows: List[str] = []
    for label in ["3x3", "4x4", "6x6", "8x8"]:
        exp = experiments.get(label, {})
        if not exp or "error" in exp:
            rows.append(f"        {label} & -- & -- & -- & -- & -- \\\\")
            continue

        methods = exp.get("methods", {})
        n = exp.get("n_intersections", "?")

        dt_train = _fmt(methods.get("DT", {}).get("training_time_sec"), 0)
        madt_train = _fmt(methods.get("MADT", {}).get("training_time_sec"), 0)

        dt_inf = methods.get("DT", {}).get("inference_time_per_step_sec")
        dt_inf_str = f"{float(dt_inf)*1000:.1f}" if dt_inf else "--"

        madt_inf = methods.get("MADT", {}).get("inference_time_per_step_sec")
        madt_inf_str = f"{float(madt_inf)*1000:.1f}" if madt_inf else "--"

        gpu = _fmt(exp.get("peak_gpu_memory_mb"), 0)

        rows.append(
            f"        {label} ({n}) & {dt_train}s & {madt_train}s & "
            f"{dt_inf_str}ms & {madt_inf_str}ms & {gpu}MB \\\\"
        )

    table = r"""\begin{table}[t]
    \centering
    \caption{Scalability across network sizes. Training and inference times, plus peak GPU memory for MADT.}
    \label{tab:scalability}
    \begin{tabular}{lccccc}
        \toprule
        Grid (N) & DT Train & MADT Train & DT Inf/Step & MADT Inf/Step & GPU Mem \\
        \midrule
%s
        \bottomrule
    \end{tabular}
\end{table}
""" % "\n".join(rows)

    return table


def generate_return_sweep_table(sweep_data: Dict[str, Any]) -> str:
    """Generate the return-conditioning sweep table (Table 4).

    Shows how different target returns map to actual behavior.
    """
    methods = sweep_data.get("methods", {})
    if not methods:
        return "% No return sweep data.\n"

    # Build rows for DT (primary); add MADT columns if available
    dt_data = methods.get("DT", {})
    madt_data = methods.get("MADT", {})

    target_returns = sweep_data.get("target_returns", [])

    rows: List[str] = []
    for tr in target_returns:
        key = f"target_{tr}" if not isinstance(tr, str) else tr

        dt_m = dt_data.get(key, {})
        madt_m = madt_data.get(key, {})

        dt_ret = _fmt(dt_m.get("actual_return_mean", dt_m.get("mean_return")))
        dt_ev = _fmt(dt_m.get("mean_ev_travel_time"))
        dt_bg = _fmt(dt_m.get("background_delay_mean"))
        dt_tp = _fmt(dt_m.get("throughput_mean"), 1)

        madt_ret = _fmt(madt_m.get("actual_return_mean", madt_m.get("mean_return")))
        madt_ev = _fmt(madt_m.get("mean_ev_travel_time"))

        rows.append(
            f"        {_fmt(tr, 0)} & {dt_ret} & {dt_ev} & {dt_bg} & "
            f"{dt_tp} & {madt_ret} & {madt_ev} \\\\"
        )

    table = r"""\begin{table}[t]
    \centering
    \caption{Return-conditioning sweep (dispatch knob). Different target returns produce different EV urgency levels. Lower target returns request more aggressive corridor preemption.}
    \label{tab:return_sweep}
    \begin{tabular}{rcccccc}
        \toprule
        & \multicolumn{4}{c}{DT} & \multicolumn{2}{c}{MADT} \\
        \cmidrule(lr){2-5} \cmidrule(lr){6-7}
        Target RTG & Return & EV Time & BG Delay & Throughput & Return & EV Time \\
        \midrule
%s
        \bottomrule
    \end{tabular}
\end{table}
""" % "\n".join(rows)

    return table


# ===================================================================
# Figure generators
# ===================================================================

def _try_import_matplotlib():
    """Import matplotlib with Agg backend for headless environments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_return_sweep_figure(
    sweep_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure: target return vs actual metrics (dispatch knob visualization)."""
    plt = _try_import_matplotlib()

    methods = sweep_data.get("methods", {})
    target_returns = sweep_data.get("target_returns", [])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for method_name, color, marker in [("DT", "#2196F3", "o"), ("MADT", "#F44336", "s")]:
        method_data = methods.get(method_name, {})
        if not method_data:
            continue

        targets = []
        actual_returns = []
        ev_times = []
        bg_delays = []

        for tr in target_returns:
            key = f"target_{tr}"
            m = method_data.get(key, {})
            if not m:
                continue
            targets.append(float(tr))
            actual_returns.append(
                float(m.get("actual_return_mean", m.get("mean_return", 0)))
            )
            ev_times.append(float(m.get("mean_ev_travel_time", 0)))
            bg_delays.append(float(m.get("background_delay_mean", 0)))

        if not targets:
            continue

        axes[0].plot(targets, actual_returns, color=color, marker=marker,
                     label=method_name, linewidth=2, markersize=6)
        axes[1].plot(targets, ev_times, color=color, marker=marker,
                     label=method_name, linewidth=2, markersize=6)
        axes[2].plot(targets, bg_delays, color=color, marker=marker,
                     label=method_name, linewidth=2, markersize=6)

    # Reference line: target = actual
    if target_returns:
        tr_range = [min(target_returns), max(target_returns)]
        axes[0].plot(tr_range, tr_range, "k--", alpha=0.3, label="Target = Actual")

    axes[0].set_xlabel("Target Return")
    axes[0].set_ylabel("Actual Return")
    axes[0].set_title("(a) Return Conditioning")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Target Return")
    axes[1].set_ylabel("EV Travel Time (steps)")
    axes[1].set_title("(b) EV Travel Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Target Return")
    axes[2].set_ylabel("Background Delay")
    axes[2].set_title("(c) Background Delay")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "return_sweep.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "return_sweep.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved return_sweep.pdf / .png")


def generate_scalability_figure(
    scale_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure: network size vs training time and inference time."""
    plt = _try_import_matplotlib()

    experiments = scale_data.get("grid_experiments", {})
    grid_labels = ["3x3", "4x4", "6x6", "8x8"]
    n_intersections = [9, 16, 36, 64]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for method_name, color, marker in [
        ("DT", "#2196F3", "o"),
        ("MADT", "#F44336", "s"),
        ("PPO", "#4CAF50", "^"),
        ("DQN", "#FF9800", "D"),
    ]:
        ns = []
        train_times = []
        inf_times = []
        ev_times = []

        for label, n in zip(grid_labels, n_intersections):
            exp = experiments.get(label, {})
            methods = exp.get("methods", {})
            m = methods.get(method_name, {})
            if not m:
                continue

            tt = m.get("training_time_sec")
            if tt is not None:
                ns.append(n)
                train_times.append(float(tt))

                it = m.get("inference_time_per_step_sec")
                inf_times.append(float(it) * 1000 if it else None)

                # Extract EV time from evaluation
                eval_info = exp.get("evaluation", {})
                ev_t = None
                for scenario, scenario_methods in eval_info.items():
                    for sm_name, sm_metrics in scenario_methods.items():
                        if method_name in sm_name:
                            ev_t = sm_metrics.get("mean_ev_travel_time")
                            break
                ev_times.append(ev_t)

        if not ns:
            continue

        axes[0].plot(ns, train_times, color=color, marker=marker,
                     label=method_name, linewidth=2, markersize=6)

        valid_inf = [(n, t) for n, t in zip(ns, inf_times) if t is not None]
        if valid_inf:
            inf_ns, inf_ts = zip(*valid_inf)
            axes[1].plot(inf_ns, inf_ts, color=color, marker=marker,
                         label=method_name, linewidth=2, markersize=6)

        valid_ev = [(n, t) for n, t in zip(ns, ev_times) if t is not None]
        if valid_ev:
            ev_ns, ev_ts = zip(*valid_ev)
            axes[2].plot(ev_ns, ev_ts, color=color, marker=marker,
                         label=method_name, linewidth=2, markersize=6)

    for ax in axes:
        ax.set_xlabel("Number of Intersections")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Training Time (s)")
    axes[0].set_title("(a) Training Time")

    axes[1].set_ylabel("Inference Time (ms/step)")
    axes[1].set_title("(b) Inference Time")

    axes[2].set_ylabel("EV Travel Time (steps)")
    axes[2].set_title("(c) EV Travel Time")

    fig.tight_layout()
    fig.savefig(output_dir / "scalability.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "scalability.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved scalability.pdf / .png")


def generate_ablation_figure(
    ablation_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure: bar chart of ablation impacts on EV travel time."""
    plt = _try_import_matplotlib()

    ablations = ablation_data.get("ablations", {})
    if not ablations:
        return

    names: List[str] = []
    ev_times: List[float] = []

    for abl_name, abl_result in ablations.items():
        eval_info = abl_result.get("evaluation", {})
        best_ev = None
        for scenario, methods in eval_info.items():
            for method_name, metrics in methods.items():
                ev = metrics.get("mean_ev_travel_time",
                                 metrics.get("ev_travel_time_mean"))
                if ev is not None and ev >= 0:
                    if best_ev is None or float(ev) < best_ev:
                        best_ev = float(ev)
        if best_ev is not None:
            names.append(abl_name.replace("_", "\n"))
            ev_times.append(best_ev)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726", "#AB47BC"]
    bars = ax.bar(range(len(names)), ev_times,
                  color=colors[:len(names)], edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("EV Travel Time (steps)")
    ax.set_title("Ablation Study: Impact on EV Travel Time")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, ev_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "ablation_bar.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "ablation_bar.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved ablation_bar.pdf / .png")


def generate_method_comparison_figure(
    eval_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure: grouped bar chart comparing all methods on key metrics."""
    plt = _try_import_matplotlib()

    # Use the first scenario found
    if not eval_data:
        return
    scenario = list(eval_data.keys())[0]
    methods_data = eval_data[scenario]

    method_names = list(methods_data.keys())
    metrics_to_plot = [
        ("mean_ev_travel_time", "EV Travel Time"),
        ("background_delay_mean", "Background Delay"),
        ("throughput_mean", "Throughput"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    cmap = plt.cm.Set2
    colors = [cmap(i / max(1, len(method_names) - 1)) for i in range(len(method_names))]

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        vals = []
        valid_names = []
        for mn in method_names:
            v = methods_data[mn].get(metric_key)
            if v is not None and (not isinstance(v, float) or v >= 0):
                vals.append(float(v))
                valid_names.append(mn)

        if not vals:
            continue

        bars = ax.bar(range(len(valid_names)), vals,
                      color=colors[:len(valid_names)],
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(valid_names)))
        ax.set_xticklabels(valid_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Method Comparison ({scenario})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "method_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "method_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved method_comparison.pdf / .png")


# ===================================================================
# Summary report
# ===================================================================

def print_summary_report(
    eval_data: Optional[Dict],
    ablation_data: Optional[Dict],
    scale_data: Optional[Dict],
    sweep_data: Optional[Dict],
) -> None:
    """Print a text summary of all results."""
    print("\n" + "=" * 70)
    print("  EV-DT EXPERIMENT SUMMARY REPORT")
    print("=" * 70)

    # Main results
    if eval_data:
        print("\n--- Main Evaluation Results ---")
        for scenario, methods in eval_data.items():
            print(f"\n  Scenario: {scenario}")
            df = pd.DataFrame.from_dict(methods, orient="index")
            cols = [c for c in ["mean_ev_travel_time", "background_delay_mean",
                                "throughput_mean", "corridor_green_ratio_mean"]
                    if c in df.columns]
            if cols:
                print(df[cols].to_string())

    # Ablation results
    if ablation_data:
        print("\n--- Ablation Study ---")
        for name, result in ablation_data.get("ablations", {}).items():
            print(f"\n  {name}: {result.get('description', '')}")
            print(f"    Training time: {result.get('training_time_sec', 'N/A'):.0f}s")

    # Scalability results
    if scale_data:
        print("\n--- Scalability ---")
        for label, exp in scale_data.get("grid_experiments", {}).items():
            n = exp.get("n_intersections", "?")
            methods = exp.get("methods", {})
            dt_time = methods.get("DT", {}).get("training_time_sec", "N/A")
            madt_time = methods.get("MADT", {}).get("training_time_sec", "N/A")
            print(f"  {label} ({n} nodes): DT train={dt_time}s, MADT train={madt_time}s")

    # Return sweep results
    if sweep_data:
        print("\n--- Return Conditioning Sweep ---")
        for method_name, method_data in sweep_data.get("methods", {}).items():
            print(f"\n  {method_name}:")
            for key, m in sorted(method_data.items()):
                tr = m.get("target_return", key)
                ev = m.get("mean_ev_travel_time", "N/A")
                ret = m.get("actual_return_mean", m.get("mean_return", "N/A"))
                print(f"    target={tr}: actual_return={ret}, ev_time={ev}")

    print("\n" + "=" * 70)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze EV-DT experiment results and generate paper artifacts."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="logs",
        help="Directory containing result JSONs.",
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        default="paper",
        help="Directory for paper outputs (figures/, tables/).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (useful if matplotlib is unavailable).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip LaTeX table generation.",
    )
    args = parser.parse_args()

    results_dir = Path(_PROJECT_ROOT) / args.results_dir
    paper_dir = Path(_PROJECT_ROOT) / args.paper_dir
    figures_dir = _ensure_dir(paper_dir / "figures")
    tables_dir = _ensure_dir(paper_dir / "tables")

    print("=" * 70)
    print("  EV-DT Results Analysis")
    print("=" * 70)
    print(f"  Results dir : {results_dir}")
    print(f"  Paper dir   : {paper_dir}")

    # ---- Load all results ----
    print("\n--- Loading Results ---")
    eval_data = _load_json(results_dir / "evaluation_results.json")
    ablation_data = _load_json(results_dir / "ablation_results.json")
    scale_data = _load_json(results_dir / "scalability_results.json")
    sweep_data = _load_json(results_dir / "return_sweep_results.json")

    has_any = any(d is not None for d in [eval_data, ablation_data, scale_data, sweep_data])
    if not has_any:
        print("\n  No result files found. Run experiments first.")
        print("  Expected files in logs/:")
        print("    - evaluation_results.json")
        print("    - ablation_results.json")
        print("    - scalability_results.json")
        print("    - return_sweep_results.json")
        return

    # ---- Generate LaTeX tables ----
    if not args.skip_tables:
        print("\n--- Generating LaTeX Tables ---")

        if eval_data:
            table = generate_main_results_table(eval_data)
            (tables_dir / "main_results.tex").write_text(table)
            print("  Saved main_results.tex")

        if ablation_data:
            table = generate_ablation_table(ablation_data)
            (tables_dir / "ablation.tex").write_text(table)
            print("  Saved ablation.tex")

        if scale_data:
            table = generate_scalability_table(scale_data)
            (tables_dir / "scalability.tex").write_text(table)
            print("  Saved scalability.tex")

        if sweep_data:
            table = generate_return_sweep_table(sweep_data)
            (tables_dir / "return_sweep.tex").write_text(table)
            print("  Saved return_sweep.tex")

    # ---- Generate figures ----
    if not args.skip_figures:
        print("\n--- Generating Figures ---")
        try:
            if sweep_data:
                generate_return_sweep_figure(sweep_data, figures_dir)

            if scale_data:
                generate_scalability_figure(scale_data, figures_dir)

            if ablation_data:
                generate_ablation_figure(ablation_data, figures_dir)

            if eval_data:
                generate_method_comparison_figure(eval_data, figures_dir)

        except ImportError as exc:
            print(f"  WARNING: Could not generate figures: {exc}")
            print("  Install matplotlib to generate figures: pip install matplotlib")

    # ---- Summary report ----
    print_summary_report(eval_data, ablation_data, scale_data, sweep_data)

    print(f"\nAll artifacts saved to {paper_dir}/")
    print(f"  Tables : {tables_dir}/")
    print(f"  Figures: {figures_dir}/")


if __name__ == "__main__":
    main()
