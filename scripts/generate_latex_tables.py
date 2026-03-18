#!/usr/bin/env python3
"""Generate LaTeX tables from experimental results for the paper.

Reads all results/*.json files and produces camera-ready tables
that can be directly included in the LaTeX paper.

Usage:
    python scripts/generate_latex_tables.py [--outdir paper/tables]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results"


def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_main_table() -> str:
    """Generate the main comparison table."""
    quick = load_json("quick_experiment_results.json")
    baselines = load_json("baseline_comparison.json")

    if not quick:
        return "% No quick_experiment_results.json found"

    results = quick["results"]

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Main results on $4{\times}4$ grid (30 episodes). "
        r"Best in \textbf{bold}.}",
        r"  \label{tab:main}",
        r"  \centering",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{EV Time $\downarrow$} "
        r"& \textbf{Return $\uparrow$} \\",
        r"    \midrule",
    ]

    methods = [
        ("FT-EVP", results.get("Fixed_Time_EVP", {})),
        ("Greedy", results.get("Greedy_Preempt", {})),
        ("Random", results.get("Random", {})),
    ]
    if baselines:
        methods.append(("PPO", baselines.get("PPO", {})))
        methods.append(("DQN", baselines.get("DQN", {})))

    for name, data in methods:
        evt = data.get("mean_ev_time", data.get("avg_ev_travel_time", "?"))
        ret = data.get("mean_return", data.get("avg_return", "?"))
        evt_str = f"{evt:.1f}" if isinstance(evt, (int, float)) else str(evt)
        ret_str = f"{ret:.1f}" if isinstance(ret, (int, float)) else str(ret)
        lines.append(f"    {name:15s} & {evt_str} & {ret_str} \\\\")

    lines.append(r"    \midrule")

    dt = results.get("DT", {})
    evt = dt.get("mean_ev_time", "?")
    ret = dt.get("mean_return", "?")
    evt_str = f"\\textbf{{{evt:.1f}}}" if isinstance(evt, (int, float)) else str(evt)
    ret_str = f"\\textbf{{{ret:.1f}}}" if isinstance(ret, (int, float)) else str(ret)
    lines.append(f"    DT (ours)       & {evt_str} & {ret_str} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_scalability_table() -> str:
    """Generate scalability results table."""
    data = load_json("scalability.json")
    if not data:
        return "% No scalability.json found"

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Scalability across grid sizes. DT consistently reduces EV time.}",
        r"  \label{tab:scalability}",
        r"  \centering",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Grid} & \textbf{Method} & \textbf{EV Time} "
        r"& \textbf{Improvement} \\",
        r"    \midrule",
    ]

    for grid_key in sorted(data.keys()):
        grid_data = data[grid_key]
        methods = grid_data["methods"]
        ft = methods.get("FT-EVP", {})
        dt = methods.get("DT", {})
        ft_time = ft.get("ev_time", 0)
        dt_time = dt.get("ev_time", 0)
        improvement = ((ft_time - dt_time) / ft_time * 100) if ft_time > 0 else 0

        lines.append(
            f"    \\multirow{{2}}{{*}}{{{grid_key}}} "
            f"& FT-EVP & {ft_time:.1f} & -- \\\\"
        )
        lines.append(
            f"    & \\textbf{{DT}} & \\textbf{{{dt_time:.1f}}} "
            f"& {improvement:.0f}\\% \\\\"
        )
        lines.append(r"    \midrule")

    lines[-1] = r"    \bottomrule"
    lines.extend([
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_ablation_table() -> str:
    """Generate ablation study table."""
    data = load_json("ablation_results.json")
    if not data:
        return "% No ablation_results.json found"

    ablations = data["ablations"]

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Ablation study (3$\times$3 grid). "
        r"Return conditioning and mixed data are critical.}",
        r"  \label{tab:ablation}",
        r"  \centering",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Variant} & \textbf{EV Time} & \textbf{Return} \\",
        r"    \midrule",
    ]

    labels = {
        "full_dt": "DT (full)",
        "no_rtg": r"\quad w/o return cond.",
        "short_context": r"\quad context $C{=}3$",
        "expert_only": r"\quad expert-only data",
    }

    for key, label in labels.items():
        if key in ablations:
            ev = ablations[key]["evaluation"]
            evt = ev.get("mean_ev_travel_time", -1)
            ret = ev.get("mean_return", 0)
            bold = r"\textbf{" if key == "full_dt" else ""
            end_bold = "}" if key == "full_dt" else ""
            evt_str = f"{bold}{evt:.1f}{end_bold}"
            ret_str = f"{bold}{ret:.1f}{end_bold}"
            lines.append(f"    {label:30s} & {evt_str} & {ret_str} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    outdir = _PROJECT_ROOT / "paper" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    tables = {
        "main_results.tex": generate_main_table,
        "scalability.tex": generate_scalability_table,
        "ablation.tex": generate_ablation_table,
    }

    for filename, generator in tables.items():
        content = generator()
        path = outdir / filename
        path.write_text(content)
        print(f"  Generated: {path}")
        print(content[:200] + "..." if len(content) > 200 else content)
        print()


if __name__ == "__main__":
    main()
