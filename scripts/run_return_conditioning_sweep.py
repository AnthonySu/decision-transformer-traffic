#!/usr/bin/env python3
"""End-to-end return-conditioning sweep: train a DT, sweep G* targets, plot.

Demonstrates the key DT selling point: dispatchers can dial the return-to-go
target G* to smoothly trade off EV speed vs civilian disruption.

Pipeline:
  1. Collect mixed offline data on a 4x4 grid (expert + random + noisy).
  2. Train a Decision Transformer from scratch.
  3. Sweep return-to-go targets: [100, 50, 0, -50, -100, -200, -400].
  4. For each G*, evaluate for N episodes recording ETT, ACD, throughput.
  5. Save results to results/return_conditioning_sweep_real.json.
  6. Generate dual-axis figure (ETT left, ACD right) and save to
     paper/figures/camera_ready/fig3_conditioning_real.{pdf,png}.

Usage::

    python scripts/run_return_conditioning_sweep.py --quick
    python scripts/run_return_conditioning_sweep.py --num-episodes 50 --seeds 3
    python scripts/run_return_conditioning_sweep.py --device cuda --num-episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_TARGETS = [100, 50, 0, -50, -100, -200, -400]


# ---------------------------------------------------------------------------
# Step 1: Collect offline data
# ---------------------------------------------------------------------------

def collect_data(
    num_expert: int,
    num_random: int,
    num_noisy: int,
    max_episode_steps: int,
    save_path: str,
) -> str:
    """Collect a mixed-quality offline dataset and save to HDF5."""
    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.utils.data_collector import DataCollector

    env = EVCorridorEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=max_episode_steps,
    )

    # Need to reset env once so that network/route are populated
    from src.baselines.greedy_preempt import GreedyPreemptPolicy

    env.reset()
    network = env.network if hasattr(env, "network") else env
    route = env.ev_route if hasattr(env, "ev_route") else []
    expert = GreedyPreemptPolicy(network, route)

    collector = DataCollector(env, save_path=save_path)
    collector.collect_mixed_dataset(
        expert_policy=expert,
        num_expert=num_expert,
        num_random=num_random,
        num_suboptimal=num_noisy,
    )
    collector.save_dataset()
    return save_path


# ---------------------------------------------------------------------------
# Step 2: Train DT
# ---------------------------------------------------------------------------

def train_dt(
    data_path: str,
    device: str,
    context_length: int,
    n_layers: int,
    n_heads: int,
    embed_dim: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    max_ep_len: int,
    save_path: str,
) -> str:
    """Train a Decision Transformer and save the best checkpoint."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.models.decision_transformer import DecisionTransformer
    from src.models.trajectory_dataset import TrajectoryDataset

    dataset = TrajectoryDataset(
        data_path=data_path,
        context_length=context_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    state_dim = dataset.state_dim
    act_dim = dataset.act_dim

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_length=context_length,
        max_ep_len=max_ep_len,
        dropout=0.1,
        activation="gelu",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = num_epochs * len(dataloader)
    warmup = min(200, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Dataset: {len(dataset)} segments, "
          f"state_dim={state_dim}, act_dim={act_dim}")
    print(f"  Model params: {model.get_num_params():,}")
    print(f"  Training for {num_epochs} epochs ({total_steps} steps)")

    best_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            masks = batch["masks"].to(device)

            logits = model(states, actions, rtg, timesteps)
            logits_flat = logits.reshape(-1, act_dim)
            targets_flat = actions.reshape(-1)
            mask_flat = masks.reshape(-1).bool()

            loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
            print(f"    Epoch {epoch:3d}/{num_epochs} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "embed_dim": embed_dim,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "context_length": context_length,
                        "max_ep_len": max_ep_len,
                        "dropout": 0.1,
                        "activation": "gelu",
                    },
                    "state_dim": state_dim,
                    "act_dim": act_dim,
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                save_path,
            )

    print(f"  Best training loss: {best_loss:.4f}")
    return save_path


# ---------------------------------------------------------------------------
# Step 3: Sweep return-to-go targets
# ---------------------------------------------------------------------------

def sweep_targets(
    model_path: str,
    target_returns: List[float],
    n_episodes: int,
    max_episode_steps: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a trained DT across different return-to-go targets."""
    import torch

    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.models.decision_transformer import DecisionTransformer
    from src.utils.metrics import aggregate_metrics

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    dt_cfg = ckpt["config"]

    model = DecisionTransformer(
        state_dim=ckpt["state_dim"],
        act_dim=ckpt["act_dim"],
        hidden_dim=dt_cfg["embed_dim"],
        n_layers=dt_cfg["n_layers"],
        n_heads=dt_cfg["n_heads"],
        max_length=dt_cfg["context_length"],
        max_ep_len=dt_cfg.get("max_ep_len", max_episode_steps),
        dropout=dt_cfg.get("dropout", 0.1),
        activation=dt_cfg.get("activation", "gelu"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = EVCorridorEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=max_episode_steps,
    )

    np.random.seed(seed)

    results: Dict[str, Any] = {}

    for target_return in target_returns:
        print(f"    G*={target_return:>5.0f}", end="", flush=True)
        episodes_info: List[Dict[str, Any]] = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0.0
            t = 0

            states = torch.zeros(
                1, model.max_length, model.state_dim, device=device
            )
            actions = torch.zeros(
                1, model.max_length, dtype=torch.long, device=device
            )
            returns_to_go = torch.zeros(
                1, model.max_length, 1, device=device
            )
            timesteps = torch.zeros(
                1, model.max_length, dtype=torch.long, device=device
            )

            states[0, 0] = torch.tensor(obs, dtype=torch.float32, device=device)
            returns_to_go[0, 0, 0] = target_return
            timesteps[0, 0] = 0

            step_infos: List[Dict[str, Any]] = []

            while not done:
                ctx_len = min(t + 1, model.max_length)
                action = model.get_action(
                    states[:, :ctx_len],
                    actions[:, :ctx_len],
                    returns_to_go[:, :ctx_len],
                    timesteps[:, :ctx_len],
                )

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_return += reward
                step_infos.append(info)
                t += 1

                if t < model.max_length:
                    states[0, t] = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    )
                    actions[0, t - 1] = action
                    returns_to_go[0, t, 0] = returns_to_go[0, t - 1, 0] - reward
                    timesteps[0, t] = t

            episodes_info.append({
                "return": episode_return,
                "length": t,
                "step_infos": step_infos,
            })

        metrics = aggregate_metrics(episodes_info)
        metrics["target_return"] = target_return

        episode_returns = [ep["return"] for ep in episodes_info]
        metrics["actual_return_mean"] = float(np.mean(episode_returns))
        metrics["actual_return_std"] = float(np.std(episode_returns))

        ett = metrics.get("mean_ev_travel_time", -1)
        acd = metrics.get("background_delay_mean", 0)
        thr = metrics.get("throughput_mean", 0)
        print(f"  | ETT={ett:.1f}  ACD={acd:.2f}  thr={thr:.0f}")

        results[f"target_{target_return}"] = metrics

    return results


# ---------------------------------------------------------------------------
# Step 4: Plot dual-axis figure
# ---------------------------------------------------------------------------

def plot_conditioning_figure(
    sweep_results: Dict[str, Any],
    target_returns: List[float],
    output_dir: str,
) -> None:
    """Generate a dual-axis ETT vs ACD figure over G* targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Professional styling
    matplotlib.rcParams.update({
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
    })

    C_BLUE = "#3274A1"
    C_RED = "#C44E52"

    gs = np.array(target_returns, dtype=float)
    etts = []
    acds = []
    thrs = []
    for g in target_returns:
        key = f"target_{g}"
        m = sweep_results.get(key, {})
        etts.append(m.get("mean_ev_travel_time", m.get("ev_travel_time_mean", 0)))
        acds.append(m.get("background_delay_mean", 0))
        thrs.append(m.get("throughput_mean", 0))
    etts = np.array(etts)
    acds = np.array(acds)

    fig, a1 = plt.subplots(figsize=(4.5, 3.0))

    # ETT (left y-axis, blue)
    a1.plot(gs, etts, "o-", color=C_BLUE, lw=2.0, ms=7, markeredgecolor="#333",
            markeredgewidth=0.8, zorder=4, label="EV Travel Time (s)")
    a1.set_xlabel("Target Return $G^*$", fontsize=11)
    a1.set_ylabel("EV Travel Time (s)", color=C_BLUE, fontsize=11)
    a1.tick_params(axis="y", labelcolor=C_BLUE)

    # ACD (right y-axis, red)
    a2 = a1.twinx()
    a2.spines["top"].set_visible(False)
    a2.plot(gs, acds, "s--", color=C_RED, lw=2.0, ms=7, markeredgecolor="#333",
            markeredgewidth=0.8, zorder=4, label="Civilian Delay (s/veh)")
    a2.set_ylabel("Avg Civilian Delay (s/veh)", color=C_RED, fontsize=11)
    a2.tick_params(axis="y", labelcolor=C_RED)

    # Shade operational sweet spot (G* in [-100, 0])
    a1.axvspan(-100, 0, alpha=0.06, color="#2ca02c", zorder=0)
    y_text = etts.min() + 0.8 * (etts.max() - etts.min())
    a1.text(-50, y_text, "Sweet spot", ha="center", va="top",
            fontsize=7, color="#2ca02c", fontstyle="italic", alpha=0.8)

    # Annotate endpoints
    a1.annotate(f"{etts[0]:.0f}s", (gs[0], etts[0]),
                textcoords="offset points", xytext=(10, 6), fontsize=7.5,
                color=C_BLUE, fontweight="bold")
    a1.annotate(f"{etts[-1]:.0f}s", (gs[-1], etts[-1]),
                textcoords="offset points", xytext=(-22, -12), fontsize=7.5,
                color=C_BLUE, fontweight="bold")

    # Combined legend
    l1, lb1 = a1.get_legend_handles_labels()
    l2, lb2 = a2.get_legend_handles_labels()
    a1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7,
              frameon=True, fancybox=True, edgecolor="#ccc", facecolor="white")

    a1.set_title("Return Conditioning ($4{\\times}4$ grid)", fontsize=11,
                 fontweight="bold")
    a1.spines["top"].set_visible(False)
    a1.spines["right"].set_visible(False)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig3_conditioning_real.pdf", format="pdf")
    fig.savefig(out / "fig3_conditioning_real.png", format="png")
    plt.close(fig)
    print(f"  Figure saved to {out / 'fig3_conditioning_real.pdf'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end return-conditioning sweep: train DT, sweep G* targets, "
            "generate dual-axis figure."
        ),
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Evaluation episodes per target return per seed (default: 50).",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds to average over (default: 3).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto / cpu / cuda (default: auto).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke-test mode: tiny dataset, short training, few episodes.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_PROJECT_ROOT / "results"),
        help="Directory for results JSON.",
    )
    parser.add_argument(
        "--targets", nargs="+", type=float, default=None,
        help="Custom G* targets (default: [100, 50, 0, -50, -100, -200, -400]).",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    target_returns = args.targets if args.targets is not None else DEFAULT_TARGETS

    # Quick mode overrides
    if args.quick:
        num_expert, num_random, num_noisy = 10, 5, 5
        num_epochs = 3
        batch_size = 16
        n_episodes = 2
        n_seeds = 1
        max_episode_steps = 50
        context_length = 10
    else:
        num_expert, num_random, num_noisy = 200, 80, 80
        num_epochs = 30
        batch_size = 64
        n_episodes = args.num_episodes
        n_seeds = args.seeds
        max_episode_steps = 300
        context_length = 30

    print("=" * 70)
    print("  EV-DT Return Conditioning Sweep (end-to-end)")
    print("=" * 70)
    print(f"  Device          : {device}")
    print(f"  Quick mode      : {args.quick}")
    print(f"  Target returns  : {target_returns}")
    print(f"  Episodes/target : {n_episodes}")
    print(f"  Seeds           : {n_seeds}")
    print(f"  Training epochs : {num_epochs}")
    print(f"  Max ep steps    : {max_episode_steps}")
    print("=" * 70)

    t_start = time.time()

    # --- Step 1: Collect data ---
    data_path = str(_PROJECT_ROOT / "data" / "sweep_offline_dataset.h5")
    print("\n[1/4] Collecting offline data ...")
    collect_data(
        num_expert=num_expert,
        num_random=num_random,
        num_noisy=num_noisy,
        max_episode_steps=max_episode_steps,
        save_path=data_path,
    )

    # --- Step 2: Train DT ---
    model_path = str(_PROJECT_ROOT / "models" / "dt_sweep.pt")
    print("\n[2/4] Training Decision Transformer ...")
    train_dt(
        data_path=data_path,
        device=device,
        context_length=context_length,
        n_layers=3,
        n_heads=4,
        embed_dim=128,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=1e-4,
        max_ep_len=max_episode_steps,
        save_path=model_path,
    )

    # --- Step 3: Sweep targets across seeds ---
    print("\n[3/4] Sweeping return-to-go targets ...")
    all_seed_results: List[Dict[str, Any]] = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed + 1}/{n_seeds}:")
        seed_results = sweep_targets(
            model_path=model_path,
            target_returns=target_returns,
            n_episodes=n_episodes,
            max_episode_steps=max_episode_steps,
            device=device,
            seed=seed * 42,
        )
        all_seed_results.append(seed_results)

    # Aggregate across seeds
    aggregated: Dict[str, Any] = {}
    for g in target_returns:
        key = f"target_{g}"
        ett_vals = [
            sr[key].get("mean_ev_travel_time", sr[key].get("ev_travel_time_mean", 0))
            for sr in all_seed_results
            if key in sr
        ]
        acd_vals = [
            sr[key].get("background_delay_mean", 0)
            for sr in all_seed_results
            if key in sr
        ]
        thr_vals = [
            sr[key].get("throughput_mean", 0)
            for sr in all_seed_results
            if key in sr
        ]
        ret_vals = [
            sr[key].get("actual_return_mean", 0)
            for sr in all_seed_results
            if key in sr
        ]

        aggregated[key] = {
            "g_star": g,
            "ett_mean": float(np.mean(ett_vals)) if ett_vals else 0,
            "ett_std": float(np.std(ett_vals)) if ett_vals else 0,
            "acd_mean": float(np.mean(acd_vals)) if acd_vals else 0,
            "acd_std": float(np.std(acd_vals)) if acd_vals else 0,
            "throughput_mean": float(np.mean(thr_vals)) if thr_vals else 0,
            "throughput_std": float(np.std(thr_vals)) if thr_vals else 0,
            "actual_return_mean": float(np.mean(ret_vals)) if ret_vals else 0,
            "actual_return_std": float(np.std(ret_vals)) if ret_vals else 0,
            "mean_ev_travel_time": float(np.mean(ett_vals)) if ett_vals else 0,
            "background_delay_mean": float(np.mean(acd_vals)) if acd_vals else 0,
        }

    # Build flat list matching narrative_numbers.json format
    sweep_list = []
    for g in target_returns:
        key = f"target_{g}"
        a = aggregated[key]
        sweep_list.append({
            "g_star": g,
            "ett": a["ett_mean"],
            "acd": a["acd_mean"],
            "throughput": a["throughput_mean"],
        })

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "return_conditioning_sweep_real.json"

    save_payload = {
        "metadata": {
            "description": "Real return-conditioning sweep from trained DT",
            "target_returns": target_returns,
            "n_episodes_per_target": n_episodes,
            "n_seeds": n_seeds,
            "training_epochs": num_epochs,
            "quick_mode": args.quick,
            "device": device,
            "total_time_sec": time.time() - t_start,
        },
        "return_conditioning_sweep": sweep_list,
        "per_target_details": aggregated,
    }

    with open(results_path, "w") as f:
        json.dump(save_payload, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # --- Step 4: Plot figure ---
    print("\n[4/4] Generating figure ...")
    fig_dir = str(_PROJECT_ROOT / "paper" / "figures" / "camera_ready")
    plot_conditioning_figure(aggregated, target_returns, fig_dir)

    # --- Summary ---
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print("  RETURN CONDITIONING SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'G*':>6s}  {'ETT (s)':>10s}  {'ACD (s/veh)':>12s}  {'Throughput':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*10}")
    for entry in sweep_list:
        print(f"  {entry['g_star']:>6.0f}  {entry['ett']:>10.1f}  "
              f"{entry['acd']:>12.2f}  {entry['throughput']:>10.0f}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results: {results_path}")
    print(f"  Figure:  {fig_dir}/fig3_conditioning_real.pdf")


if __name__ == "__main__":
    main()
