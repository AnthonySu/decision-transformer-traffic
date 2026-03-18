#!/usr/bin/env python3
"""Evaluate all methods and generate comparison tables/plots."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.baselines.fixed_time_evp import FixedTimeEVP
from src.baselines.greedy_preempt import GreedyPreemptPolicy
from src.envs.ev_corridor_env import EVCorridorEnv
from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.madt import MultiAgentDecisionTransformer
from src.utils.metrics import aggregate_metrics, compare_methods


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_policy(env, policy_fn, n_episodes: int = 100) -> list[dict]:
    """Run policy for n_episodes, collect info dicts."""
    episodes_info = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        t = 0
        step_infos = []

        while not done:
            action = policy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step_infos.append(info)
            t += 1

        episodes_info.append(
            {"return": episode_return, "length": t, "step_infos": step_infos}
        )
    return episodes_info


def run_evaluation(config_path: str, scenarios: list[str] | None = None):
    config = load_config(config_path)
    eval_cfg = config["eval"]
    n_episodes = eval_cfg["num_eval_episodes"]

    if scenarios is None:
        scenarios = eval_cfg["scenarios"]

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Evaluating scenario: {scenario}")
        print(f"{'='*60}")

        # Create environments
        # Parse grid size from scenario name if applicable
        if "grid" in scenario:
            parts = scenario.replace("-v0", "").split("-")
            grid_spec = parts[1]  # e.g. "4x4"
            rows, cols = map(int, grid_spec.split("x"))
        else:
            rows, cols = 4, 4  # default

        env = EVCorridorEnv(
            network_type="grid",
            grid_rows=rows,
            grid_cols=cols,
            use_lightsim=False,
            max_episode_steps=config["env"]["max_episode_steps"],
        )

        ma_env = EVCorridorMAEnv(
            network_type="grid",
            grid_rows=rows,
            grid_cols=cols,
            use_lightsim=False,
            max_episode_steps=config["env"]["max_episode_steps"],
        )

        scenario_results = {}

        # 1. Fixed-Time EVP
        print("\n  Fixed-Time EVP...")
        fixed_evp = FixedTimeEVP()
        fixed_results = evaluate_policy(
            env, lambda obs, info: fixed_evp.select_action(obs, info), n_episodes
        )
        scenario_results["Fixed-Time EVP"] = aggregate_metrics(fixed_results)

        # 2. Greedy Preemption
        print("  Greedy Preemption...")
        greedy = GreedyPreemptPolicy(network=env.network, route=env.ev_route)
        greedy_results = evaluate_policy(
            env, lambda obs, info: greedy.select_action(obs, info), n_episodes
        )
        scenario_results["Greedy Preempt"] = aggregate_metrics(greedy_results)

        # 3. PPO (if model exists)
        ppo_path = Path("models/ppo_evp")
        if ppo_path.exists():
            print("  PPO...")
            from stable_baselines3 import PPO

            ppo_model = PPO.load(str(ppo_path / "best_model"))
            ppo_results = evaluate_policy(
                env,
                lambda obs, info: ppo_model.predict(obs, deterministic=True)[0],
                n_episodes,
            )
            scenario_results["PPO"] = aggregate_metrics(ppo_results)

        # 4. DQN (if model exists)
        dqn_path = Path("models/dqn_evp")
        if dqn_path.exists():
            print("  DQN...")
            from stable_baselines3 import DQN

            dqn_model = DQN.load(str(dqn_path / "best_model"))
            dqn_results = evaluate_policy(
                env,
                lambda obs, info: dqn_model.predict(obs, deterministic=True)[0],
                n_episodes,
            )
            scenario_results["DQN"] = aggregate_metrics(dqn_results)

        # 5. Decision Transformer (multiple target returns)
        dt_path = Path("models/dt_best.pt")
        if dt_path.exists():
            print("  Decision Transformer...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(dt_path, map_location=device, weights_only=False)
            dt_model = DecisionTransformer(
                state_dim=ckpt["state_dim"],
                act_dim=ckpt["act_dim"],
                hidden_dim=ckpt["config"]["embed_dim"],
                n_layers=ckpt["config"]["n_layers"],
                n_heads=ckpt["config"]["n_heads"],
                max_length=ckpt["config"]["context_length"],
            ).to(device)
            dt_model.load_state_dict(ckpt["model_state_dict"])
            dt_model.eval()

            from scripts.train_dt import evaluate_dt

            dt_results = evaluate_dt(
                dt_model,
                env,
                config["dt"]["target_returns"],
                n_episodes=n_episodes,
                device=device,
            )
            for key, metrics in dt_results.items():
                scenario_results[f"DT ({key})"] = metrics

        # 6. MADT (multiple target returns)
        madt_path = Path("models/madt_best.pt")
        if madt_path.exists():
            print("  Multi-Agent Decision Transformer...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(madt_path, map_location=device, weights_only=False)
            madt_model = MultiAgentDecisionTransformer(
                state_dim=ckpt["state_dim"],
                act_dim=ckpt["act_dim"],
                n_agents=ckpt["n_agents"],
                adj_matrix=ckpt["adj_matrix"],
                hidden_dim=ckpt["config"]["embed_dim"],
                n_layers=ckpt["config"]["n_layers"],
                n_heads=ckpt["config"]["n_heads"],
                gat_heads=ckpt["config"]["gat_heads"],
                gat_layers=ckpt["config"]["gat_layers"],
                max_length=ckpt["config"]["context_length"],
            ).to(device)
            madt_model.load_state_dict(ckpt["model_state_dict"])
            madt_model.eval()

            from scripts.train_madt import evaluate_madt

            madt_results = evaluate_madt(
                madt_model,
                ma_env,
                config["madt"]["target_returns"],
                n_episodes=n_episodes,
                device=device,
            )
            for key, metrics in madt_results.items():
                scenario_results[f"MADT ({key})"] = metrics

        all_results[scenario] = scenario_results

        # Print scenario summary
        df = compare_methods(scenario_results)
        print(f"\n  Results for {scenario}:")
        print(df.to_string(index=True))

    # Save all results
    results_dir = Path("logs")
    results_dir.mkdir(exist_ok=True)

    # Save raw results as JSON
    serializable = {}
    for scenario, methods in all_results.items():
        serializable[scenario] = {}
        for method, metrics in methods.items():
            serializable[scenario][method] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
                if k != "step_infos"
            }

    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {results_dir / 'evaluation_results.json'}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--scenarios", nargs="+", default=None)
    args = parser.parse_args()
    run_evaluation(args.config, args.scenarios)
