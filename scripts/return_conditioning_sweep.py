#!/usr/bin/env python3
"""Sweep different target returns during inference to demonstrate the
controllability (dispatch knob) of the Decision Transformer.

For each target return level, we run evaluation episodes and record the
actual return, EV travel time, background delay, and throughput. This
shows that different urgency levels produce different corridor strategies.

Target returns swept: [0, -25, -50, -75, -100, -150, -200, -300]

Results are saved to logs/return_sweep_results.json.

Usage::

    python scripts/return_conditioning_sweep.py
    python scripts/return_conditioning_sweep.py --device cuda --n-episodes 100
    python scripts/return_conditioning_sweep.py --targets 0 -50 -100 -200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


DEFAULT_TARGET_RETURNS = [0, -25, -50, -75, -100, -150, -200, -300]


# ---------------------------------------------------------------------------
# DT return sweep
# ---------------------------------------------------------------------------

def sweep_dt(
    config: Dict[str, Any],
    target_returns: List[float],
    n_episodes: int,
    device: str,
) -> Dict[str, Any]:
    """Run return-conditioning sweep for single-agent Decision Transformer."""
    import torch

    from src.envs.ev_corridor_env import EVCorridorEnv
    from src.models.decision_transformer import DecisionTransformer
    from src.utils.metrics import aggregate_metrics

    dt_path = Path(_PROJECT_ROOT) / "models" / "dt_best.pt"
    if not dt_path.exists():
        print("  DT model not found, skipping DT sweep.")
        return {}

    ckpt = torch.load(dt_path, map_location=device, weights_only=False)
    dt_cfg = ckpt["config"]

    model = DecisionTransformer(
        state_dim=ckpt["state_dim"],
        act_dim=ckpt["act_dim"],
        hidden_dim=dt_cfg["embed_dim"],
        n_layers=dt_cfg["n_layers"],
        n_heads=dt_cfg["n_heads"],
        max_length=dt_cfg["context_length"],
        max_ep_len=dt_cfg.get("max_ep_len", config["env"]["max_episode_steps"]),
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
        max_episode_steps=config["env"]["max_episode_steps"],
    )

    results: Dict[str, Any] = {}

    for target_return in target_returns:
        print(f"\n  DT | target_return = {target_return}")
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

            if (ep + 1) % 10 == 0:
                print(f"    Episode {ep+1}/{n_episodes}, "
                      f"return={episode_return:.1f}")

        metrics = aggregate_metrics(episodes_info)
        metrics["target_return"] = target_return

        # Also record per-episode returns for distribution analysis
        episode_returns = [ep["return"] for ep in episodes_info]
        metrics["per_episode_returns"] = episode_returns
        metrics["actual_return_mean"] = float(np.mean(episode_returns))
        metrics["actual_return_std"] = float(np.std(episode_returns))

        results[f"target_{target_return}"] = metrics

    return results


# ---------------------------------------------------------------------------
# MADT return sweep
# ---------------------------------------------------------------------------

def sweep_madt(
    config: Dict[str, Any],
    target_returns: List[float],
    n_episodes: int,
    device: str,
) -> Dict[str, Any]:
    """Run return-conditioning sweep for multi-agent Decision Transformer."""
    import torch

    from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
    from src.models.madt import MultiAgentDecisionTransformer
    from src.utils.metrics import aggregate_metrics

    madt_path = Path(_PROJECT_ROOT) / "models" / "madt_best.pt"
    if not madt_path.exists():
        print("  MADT model not found, skipping MADT sweep.")
        return {}

    ckpt = torch.load(madt_path, map_location=device, weights_only=False)
    madt_cfg = ckpt["config"]

    model = MultiAgentDecisionTransformer(
        state_dim=ckpt["state_dim"],
        act_dim=ckpt["act_dim"],
        n_agents=ckpt["n_agents"],
        adj_matrix=ckpt["adj_matrix"],
        hidden_dim=madt_cfg["embed_dim"],
        n_layers=madt_cfg["n_layers"],
        n_heads=madt_cfg["n_heads"],
        gat_heads=madt_cfg["gat_heads"],
        gat_layers=madt_cfg["gat_layers"],
        max_length=madt_cfg["context_length"],
        max_ep_len=madt_cfg.get("max_ep_len", config["env"]["max_episode_steps"]),
        dropout=madt_cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_agents = ckpt["n_agents"]

    env = EVCorridorMAEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=config["env"]["max_episode_steps"],
    )

    results: Dict[str, Any] = {}

    for target_return in target_returns:
        print(f"\n  MADT | target_return = {target_return}")
        episodes_info: List[Dict[str, Any]] = []

        for ep in range(n_episodes):
            obs_dict = env.reset()
            done = {agent: False for agent in env.agents}
            episode_return = 0.0
            t = 0

            states = torch.zeros(
                1, n_agents, model.max_length, model.state_dim, device=device
            )
            actions = torch.zeros(
                1, n_agents, model.max_length, dtype=torch.long, device=device
            )
            returns_to_go = torch.zeros(
                1, n_agents, model.max_length, 1, device=device
            )
            timesteps = torch.zeros(
                1, n_agents, model.max_length, dtype=torch.long, device=device
            )

            for i, agent_id in enumerate(env.agents):
                if agent_id in obs_dict:
                    states[0, i, 0] = torch.tensor(
                        obs_dict[agent_id], dtype=torch.float32, device=device
                    )
                returns_to_go[0, i, 0, 0] = target_return / n_agents

            step_infos: List[Dict[str, Any]] = []

            while not all(done.values()):
                ctx_len = min(t + 1, model.max_length)
                action_dict = {}

                for i, agent_id in enumerate(env.agents):
                    if not done.get(agent_id, True):
                        action = model.get_action(
                            states[:, :, :ctx_len],
                            actions[:, :, :ctx_len],
                            returns_to_go[:, :, :ctx_len],
                            timesteps[:, :, :ctx_len],
                            agent_idx=i,
                        )
                        action_dict[agent_id] = action

                obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(
                    action_dict
                )

                step_reward = sum(reward_dict.values())
                episode_return += step_reward
                step_infos.append(info_dict)

                for agent_id, terminated in term_dict.items():
                    if terminated or trunc_dict.get(agent_id, False):
                        done[agent_id] = True

                t += 1
                if t < model.max_length:
                    for i, agent_id in enumerate(env.agents):
                        if agent_id in obs_dict:
                            states[0, i, t] = torch.tensor(
                                obs_dict[agent_id], dtype=torch.float32, device=device
                            )
                            if agent_id in action_dict:
                                actions[0, i, t - 1] = action_dict[agent_id]
                            agent_reward = reward_dict.get(agent_id, 0.0)
                            returns_to_go[0, i, t, 0] = (
                                returns_to_go[0, i, t - 1, 0] - agent_reward
                            )
                            timesteps[0, i, t] = t

            episodes_info.append({
                "return": episode_return,
                "length": t,
                "step_infos": step_infos,
            })

            if (ep + 1) % 10 == 0:
                print(f"    Episode {ep+1}/{n_episodes}, "
                      f"return={episode_return:.1f}")

        metrics = aggregate_metrics(episodes_info)
        metrics["target_return"] = target_return

        episode_returns = [ep["return"] for ep in episodes_info]
        metrics["per_episode_returns"] = episode_returns
        metrics["actual_return_mean"] = float(np.mean(episode_returns))
        metrics["actual_return_std"] = float(np.std(episode_returns))

        results[f"target_{target_return}"] = metrics

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Return-conditioning sweep for DT/MADT dispatch knob analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=None,
        help="Target returns to sweep. Default: [0, -25, -50, -75, -100, -150, -200, -300].",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per target return.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dt", "madt"],
        choices=["dt", "madt"],
        help="Which methods to sweep. Default: both.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/return_sweep_results.json",
        help="Output path for results JSON.",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    target_returns = args.targets if args.targets is not None else DEFAULT_TARGET_RETURNS
    config = load_config(args.config)

    print("=" * 70)
    print("  EV-DT Return Conditioning Sweep")
    print("=" * 70)
    print(f"  Config          : {args.config}")
    print(f"  Target returns  : {target_returns}")
    print(f"  Episodes/target : {args.n_episodes}")
    print(f"  Device          : {device}")
    print(f"  Methods         : {args.methods}")
    print("=" * 70)

    all_results: Dict[str, Any] = {
        "config": args.config,
        "target_returns": target_returns,
        "n_episodes": args.n_episodes,
        "device": device,
        "methods": {},
    }

    t_start = time.time()

    if "dt" in args.methods:
        print("\n--- Decision Transformer Sweep ---")
        dt_results = sweep_dt(config, target_returns, args.n_episodes, device)
        all_results["methods"]["DT"] = dt_results

    if "madt" in args.methods:
        print("\n--- Multi-Agent Decision Transformer Sweep ---")
        madt_results = sweep_madt(config, target_returns, args.n_episodes, device)
        all_results["methods"]["MADT"] = madt_results

    all_results["total_time_sec"] = time.time() - t_start

    # Save results
    output_path = Path(_PROJECT_ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip per_episode_returns for the saved JSON to keep file size reasonable;
    # store summary stats instead.
    save_results = json.loads(json.dumps(all_results, default=float))
    for method_name, method_data in save_results.get("methods", {}).items():
        for target_key, target_data in method_data.items():
            if isinstance(target_data, dict):
                target_data.pop("per_episode_returns", None)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("  RETURN SWEEP SUMMARY")
    print(f"{'='*70}")

    for method_name in ["DT", "MADT"]:
        method_data = all_results["methods"].get(method_name, {})
        if not method_data:
            continue

        print(f"\n  {method_name}:")
        print(f"  {'Target RTG':<12} {'Actual Return':<18} {'EV Time':<12} "
              f"{'BG Delay':<12} {'Throughput':<12}")
        print(f"  {'-'*12} {'-'*18} {'-'*12} {'-'*12} {'-'*12}")

        for target_return in target_returns:
            key = f"target_{target_return}"
            if key not in method_data:
                continue
            m = method_data[key]
            actual_ret = m.get("actual_return_mean", m.get("mean_return", 0))
            actual_std = m.get("actual_return_std", 0)
            ev_time = m.get("mean_ev_travel_time", -1)
            bg_delay = m.get("background_delay_mean", 0)
            throughput = m.get("throughput_mean", 0)

            print(f"  {target_return:<12.0f} "
                  f"{actual_ret:<8.1f}+/-{actual_std:<7.1f} "
                  f"{ev_time:<12.1f} {bg_delay:<12.2f} {throughput:<12.1f}")

    print(f"\nTotal time: {all_results['total_time_sec']:.1f}s")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
