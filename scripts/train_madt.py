#!/usr/bin/env python3
"""Train Multi-Agent Decision Transformer for EV corridor optimization."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
from src.models.madt import MultiAgentDecisionTransformer
from src.models.trajectory_dataset import MultiAgentTrajectoryDataset
from src.utils.metrics import aggregate_metrics


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_madt(
    model: MultiAgentDecisionTransformer,
    env: EVCorridorMAEnv,
    target_returns: list[float],
    n_episodes: int = 20,
    device: str = "cpu",
) -> dict:
    """Evaluate MADT with different return-conditioning targets."""
    model.eval()
    results = {}
    n_agents = model.n_agents

    for target_return in target_returns:
        episodes_info = []

        for _ in range(n_episodes):
            obs_dict = env.reset()
            done = {agent: False for agent in env.agents}
            episode_return = 0.0
            t = 0

            # Initialize per-agent sequences
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

            # Set initial observations
            for i, agent_id in enumerate(env.agents):
                if agent_id in obs_dict:
                    states[0, i, 0] = torch.tensor(
                        obs_dict[agent_id], dtype=torch.float32, device=device
                    )
                returns_to_go[0, i, 0, 0] = target_return / n_agents

            step_infos = []

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

            episodes_info.append(
                {"return": episode_return, "length": t, "step_infos": step_infos}
            )

        metrics = aggregate_metrics(episodes_info)
        metrics["target_return"] = target_return
        results[f"target_{target_return}"] = metrics

    model.train()
    return results


def train(config_path: str, device: str = "auto"):
    config = load_config(config_path)
    madt_cfg = config["madt"]
    env_cfg = config["env"]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training Multi-Agent Decision Transformer on {device}")

    # Load dataset
    dataset = MultiAgentTrajectoryDataset(
        data_path=config["dataset"]["save_path"],
        context_length=madt_cfg["context_length"],
        max_ep_len=madt_cfg.get("max_ep_len", env_cfg["max_episode_steps"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=madt_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    state_dim = dataset.state_dim
    act_dim = dataset.act_dim
    n_agents = dataset.n_agents
    adj_matrix = dataset.adj_matrix  # loaded from dataset metadata

    model = MultiAgentDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_agents=n_agents,
        adj_matrix=adj_matrix,
        hidden_dim=madt_cfg["embed_dim"],
        n_layers=madt_cfg["n_layers"],
        n_heads=madt_cfg["n_heads"],
        gat_heads=madt_cfg["gat_heads"],
        gat_layers=madt_cfg["gat_layers"],
        max_length=madt_cfg["context_length"],
        max_ep_len=madt_cfg.get("max_ep_len", env_cfg["max_episode_steps"]),
        dropout=madt_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=madt_cfg["lr"],
        weight_decay=madt_cfg["weight_decay"],
    )

    total_steps = madt_cfg["num_epochs"] * len(dataloader)

    def lr_lambda(step):
        if step < madt_cfg["warmup_steps"]:
            return step / madt_cfg["warmup_steps"]
        progress = (step - madt_cfg["warmup_steps"]) / max(
            1, total_steps - madt_cfg["warmup_steps"]
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Eval environment
    eval_env = EVCorridorMAEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=env_cfg["max_episode_steps"],
    )

    loss_fn = nn.CrossEntropyLoss()
    best_ev_time = float("inf")
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    print(f"Dataset: {len(dataset)} segments")
    print(f"  n_agents={n_agents}, state_dim={state_dim}, act_dim={act_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    for epoch in range(1, madt_cfg["num_epochs"] + 1):
        model.train()
        epoch_losses = []

        for batch in dataloader:
            # shapes: [batch, n_agents, seq_len, ...]
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            masks = batch["masks"].to(device)

            action_logits = model(states, actions, returns_to_go, timesteps)
            # action_logits: [batch, n_agents, seq_len, act_dim]

            logits_flat = action_logits.reshape(-1, act_dim)
            targets_flat = actions.reshape(-1)
            mask_flat = masks.reshape(-1).bool()

            loss = loss_fn(logits_flat[mask_flat], targets_flat[mask_flat])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)

        if epoch % madt_cfg["eval_interval"] == 0:
            eval_results = evaluate_madt(
                model,
                eval_env,
                madt_cfg["target_returns"],
                n_episodes=20,
                device=device,
            )

            best_target = f"target_{madt_cfg['target_returns'][0]}"
            ev_time = eval_results[best_target].get("mean_ev_travel_time", float("inf"))

            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"EV Time: {ev_time:.1f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            if ev_time < best_ev_time:
                best_ev_time = ev_time
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": madt_cfg,
                        "state_dim": state_dim,
                        "act_dim": act_dim,
                        "n_agents": n_agents,
                        "adj_matrix": adj_matrix,
                        "epoch": epoch,
                        "best_ev_time": best_ev_time,
                    },
                    save_dir / "madt_best.pt",
                )
                print(f"  -> New best MADT model (EV time: {best_ev_time:.1f})")
        else:
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": madt_cfg,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "n_agents": n_agents,
            "adj_matrix": adj_matrix,
            "epoch": madt_cfg["num_epochs"],
        },
        save_dir / "madt_final.pt",
    )
    print(f"\nTraining complete. Best EV travel time: {best_ev_time:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MADT for EV corridor")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config path"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda)"
    )
    args = parser.parse_args()
    train(args.config, args.device)
