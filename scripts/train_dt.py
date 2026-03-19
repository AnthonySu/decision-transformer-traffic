#!/usr/bin/env python3
"""Train Decision Transformer for EV corridor optimization."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.envs.ev_corridor_env import EVCorridorEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import TrajectoryDataset
from src.utils.metrics import aggregate_metrics
from src.utils.wandb_logger import WandbLogger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_dt(
    model: DecisionTransformer,
    env: EVCorridorEnv,
    target_returns: list[float],
    n_episodes: int = 20,
    device: str = "cpu",
) -> dict:
    """Evaluate DT with different return-conditioning targets."""
    model.eval()
    results = {}

    for target_return in target_returns:
        episodes_info = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0.0
            t = 0

            # Initialize sequences
            states = torch.zeros(1, model.max_length, model.state_dim, device=device)
            actions = torch.zeros(1, model.max_length, dtype=torch.long, device=device)
            returns_to_go = torch.zeros(1, model.max_length, 1, device=device)
            timesteps = torch.zeros(1, model.max_length, dtype=torch.long, device=device)

            states[0, 0] = torch.tensor(obs, dtype=torch.float32, device=device)
            returns_to_go[0, 0, 0] = target_return
            timesteps[0, 0] = 0

            step_infos = []

            while not done:
                # Get context window
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
                    states[0, t] = torch.tensor(obs, dtype=torch.float32, device=device)
                    actions[0, t - 1] = action
                    returns_to_go[0, t, 0] = returns_to_go[0, t - 1, 0] - reward
                    timesteps[0, t] = t

            episodes_info.append(
                {"return": episode_return, "length": t, "step_infos": step_infos}
            )

        metrics = aggregate_metrics(episodes_info)
        metrics["target_return"] = target_return
        results[f"target_{target_return}"] = metrics

    model.train()
    return results


def train(config_path: str, device: str = "auto", use_wandb: bool = False):
    config = load_config(config_path)
    dt_cfg = config["dt"]
    env_cfg = config["env"]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb logger (no-op if use_wandb=False or wandb not installed)
    logger = WandbLogger(
        project="ev-decision-transformer",
        run_name="dt-train",
        config=config,
        tags=["dt"],
        enabled=use_wandb,
    )

    print(f"Training Decision Transformer on {device}")
    print(f"Config: {config_path}")

    # Load dataset
    dataset = TrajectoryDataset(
        data_path=config["dataset"]["save_path"],
        context_length=dt_cfg["context_length"],
        max_ep_len=dt_cfg["max_ep_len"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=dt_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Create model
    state_dim = dataset.state_dim
    act_dim = dataset.act_dim
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=dt_cfg["embed_dim"],
        n_layers=dt_cfg["n_layers"],
        n_heads=dt_cfg["n_heads"],
        max_length=dt_cfg["context_length"],
        max_ep_len=dt_cfg["max_ep_len"],
        dropout=dt_cfg["dropout"],
        activation=dt_cfg["activation"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dt_cfg["lr"],
        weight_decay=dt_cfg["weight_decay"],
    )

    # Warmup + cosine decay scheduler
    total_steps = dt_cfg["num_epochs"] * len(dataloader)

    def lr_lambda(step):
        if step < dt_cfg["warmup_steps"]:
            return step / dt_cfg["warmup_steps"]
        progress = (step - dt_cfg["warmup_steps"]) / max(
            1, total_steps - dt_cfg["warmup_steps"]
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create eval environment
    eval_env = EVCorridorEnv(
        network_type="grid",
        grid_rows=4,
        grid_cols=4,
        use_lightsim=False,
        max_episode_steps=env_cfg["max_episode_steps"],
    )

    # Training loop
    loss_fn = nn.CrossEntropyLoss()
    best_ev_time = float("inf")
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    print(f"Dataset: {len(dataset)} segments, state_dim={state_dim}, act_dim={act_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {dt_cfg['num_epochs']} epochs")
    print("-" * 60)

    for epoch in range(1, dt_cfg["num_epochs"] + 1):
        model.train()
        epoch_losses = []

        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            masks = batch["masks"].to(device)

            # Forward pass
            action_logits = model(states, actions, returns_to_go, timesteps)

            # Loss on valid (non-padded) positions
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

        # Log training metrics
        logger.log({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]}, step=epoch)

        # Evaluate periodically
        if epoch % dt_cfg["eval_interval"] == 0:
            eval_results = evaluate_dt(
                model, eval_env, dt_cfg["target_returns"], n_episodes=20, device=device
            )

            # Use target_return=0 (best case) for model selection
            best_target = f"target_{dt_cfg['target_returns'][0]}"
            ev_time = eval_results[best_target].get("mean_ev_travel_time", float("inf"))

            # Log eval metrics
            eval_metrics = {"eval/ev_travel_time": ev_time}
            for key, metrics in eval_results.items():
                for mk, mv in metrics.items():
                    if isinstance(mv, (int, float)):
                        eval_metrics[f"eval/{key}/{mk}"] = mv
            logger.log(eval_metrics, step=epoch)

            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"EV Time: {ev_time:.1f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            for key, metrics in eval_results.items():
                print(f"  {key}: {metrics}")

            if ev_time < best_ev_time:
                best_ev_time = ev_time
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": dt_cfg,
                        "state_dim": state_dim,
                        "act_dim": act_dim,
                        "epoch": epoch,
                        "best_ev_time": best_ev_time,
                    },
                    save_dir / "dt_best.pt",
                )
                print(f"  -> New best model saved (EV time: {best_ev_time:.1f})")
        else:
            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

    # Final save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dt_cfg,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "epoch": dt_cfg["num_epochs"],
        },
        save_dir / "dt_final.pt",
    )
    logger.finish()
    print(f"\nTraining complete. Best EV travel time: {best_ev_time:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Transformer for EV corridor")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    args = parser.parse_args()
    train(args.config, args.device, use_wandb=args.wandb)
