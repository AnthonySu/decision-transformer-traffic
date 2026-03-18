# EV-DT: Return-Conditioned Emergency Corridor Optimization via Decision Transformer

**Offline reinforcement learning for emergency vehicle signal preemption using Decision Transformers.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

Emergency vehicles (EVs) lose critical response time at signalized intersections. Traditional signal preemption strategies (fixed green waves, greedy heuristics) either ignore background traffic impacts or require hand-tuned parameters for every network. Online reinforcement learning methods can learn better policies but are unsafe to train in live traffic and suffer from sample inefficiency.

**EV-DT** reframes emergency corridor optimization as an offline, return-conditioned sequence modeling problem using the **Decision Transformer** (DT) architecture. Instead of learning a value function, the DT directly maps a desired cumulative return (e.g., "minimize EV travel time while keeping background delay below X") to a sequence of signal phase actions. This enables a single trained model to produce a *spectrum* of policies at inference time simply by varying the target return -- from aggressive corridor clearing to balanced traffic management -- without retraining.

The repository also implements a **Multi-Agent Decision Transformer (MADT)** that extends the single-agent DT with Graph Attention Network (GAT) layers. Each intersection acts as an agent with its own observation-action sequence, while GAT layers enable spatial communication over the traffic network topology. We compare both DT variants against standard online RL baselines (PPO, DQN) and classical heuristics (greedy preemption, fixed-time EVP) on grid networks of varying scale.

## Architecture

```
                          TRAINING (Offline)
  ========================================================================

  +------------------+     +---------------------+     +------------------+
  |  Offline Dataset |     |   Trajectory        |     |  Decision        |
  |  (H5)           | --> |   Sampling          | --> |  Transformer     |
  |                  |     |   (R, s, a) triples |     |  Training        |
  | - Expert demos   |     |   context_len = 30  |     |  (CrossEntropy)  |
  | - Mixed quality  |     +---------------------+     +------------------+
  | - 5000 episodes  |                                         |
  +------------------+                                         v
                                                  +------------------------+
                                                  |  Trained DT / MADT     |
                                                  |  Checkpoint (.pt)      |
                                                  +------------------------+

                         INFERENCE (Return-Conditioned)
  ========================================================================

  Target Return R*        Trained Model           Traffic Environment
  (dispatch knob)         (DT or MADT)            (CTM / LightSim)
       |                       |                        |
       v                       v                        v
  +----------+    +------------------------+    +----------------+
  | R* = 0   | -> | Autoregressive action  | -> | Step simulation|
  | R* = -50 |    | prediction given       |    | Advance EV     |
  | R* = -100|    | (R_t, s_t, a_{t-1})   |    | Update queues  |
  +----------+    +------------------------+    +----------------+
                           |                           |
                           +------ feedback loop ------+

  MADT Extension (Multi-Agent):
  ========================================================================

  Per-intersection        GAT Layers              Shared Causal
  Embeddings              (Spatial Fusion)        Transformer
  +--------+          +------------------+       +---------------+
  | Agent 1| --+      |  Graph Attention |       | Transformer   |
  | Agent 2| --+--->  |  over network    | --->  | Blocks        | ---> Actions
  | Agent N| --+      |  adjacency       |       | (shared wts)  |
  +--------+          +------------------+       +---------------+
```

## Key Features

- **Return conditioning as a dispatch knob.** A single model produces diverse policies by varying the target return at inference time -- no retraining needed. Dispatchers can dial between aggressive EV prioritization and balanced traffic flow.
- **Decision Transformer for offline RL.** Trains entirely on pre-collected demonstrations (expert + mixed-quality), avoiding unsafe online exploration in live traffic networks.
- **Multi-Agent Decision Transformer (MADT).** Extends DT with Graph Attention layers so intersection agents can communicate with topological neighbors, improving coordination on larger networks.
- **Built-in CTM traffic simulation.** Ships with a Cell Transmission Model (CTM) grid simulator, no external dependencies required. Also supports LightSim integration for higher-fidelity scenarios.
- **Comprehensive baselines.** Includes PPO, DQN (via Stable-Baselines3), greedy preemption, and fixed-time EVP for fair comparison.
- **Configurable experiments.** Single YAML config controls environment, model architecture, training, and evaluation. Swap scenarios, tune hyperparameters, or add new baselines without touching code.
- **Five evaluation metrics.** EV travel time, background vehicle delay, network throughput, signal disruption count, and corridor green ratio.
- **Scalability testing.** Evaluate on 4x4, 6x6, and Manhattan-style networks out of the box.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/decision-transformer-traffic.git
cd decision-transformer-traffic

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, PyTorch 2.1+. See `pyproject.toml` for the full dependency list. GPU is recommended for training but not required.

## Quick Start

### 1. Environment Demo

```python
from src.envs.ev_corridor_env import EVCorridorEnv

env = EVCorridorEnv(rows=4, cols=4, max_steps=200, render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.action_space.sample()  # random signal phases
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

print(f"EV arrived: {info['ev_arrived']}, Steps: {info['step']}")
```

### 2. Train a Decision Transformer

```bash
# Generate the offline dataset (expert + mixed-quality trajectories)
python scripts/generate_dataset.py --config configs/default.yaml

# Train the Decision Transformer
python scripts/train_dt.py --config configs/default.yaml --device auto
```

### 3. Evaluate with Different Target Returns

```python
import torch
from src.models.decision_transformer import DecisionTransformer
from src.envs.ev_corridor_env import EVCorridorEnv

# Load trained model
checkpoint = torch.load("models/dt_best.pt")
model = DecisionTransformer(
    state_dim=checkpoint["state_dim"],
    act_dim=checkpoint["act_dim"],
    **{k: v for k, v in checkpoint["config"].items()
       if k in ("embed_dim", "n_layers", "n_heads", "context_length",
                "max_ep_len", "dropout", "activation")},
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Evaluate with different target returns
# R* = 0   -> aggressive EV prioritization
# R* = -100 -> balanced traffic management
for target_return in [0.0, -50.0, -100.0, -200.0]:
    print(f"\n--- Target Return: {target_return} ---")
    # ... run evaluation loop (see scripts/evaluate.py for full example)
```

## Project Structure

```
decision-transformer-traffic/
|
|-- configs/
|   +-- default.yaml              # Main experiment configuration
|
|-- data/                          # Generated offline datasets (HDF5)
|
|-- logs/                          # Training logs and evaluation results
|
|-- models/                        # Saved model checkpoints
|
|-- paper/                         # Paper drafts and figures
|
|-- scripts/
|   |-- generate_dataset.py        # Collect offline trajectories
|   |-- train_dt.py                # Train single-agent Decision Transformer
|   |-- train_madt.py              # Train Multi-Agent Decision Transformer
|   |-- train_baselines.py         # Train PPO / DQN baselines
|   |-- evaluate.py                # Evaluate all methods, produce tables
|   +-- run_all.py                 # Full pipeline: data -> train -> eval
|
|-- src/
|   |-- envs/
|   |   |-- ev_corridor_env.py     # Single-agent Gymnasium environment
|   |   |-- ev_corridor_ma_env.py  # Multi-agent PettingZoo environment
|   |   |-- ev_tracker.py          # EV position and progress tracking
|   |   +-- network_utils.py       # CTM grid network builder and simulation
|   |
|   |-- models/
|   |   |-- decision_transformer.py  # DT: causal transformer + embeddings
|   |   |-- madt.py                  # MADT: DT + graph attention layers
|   |   +-- trajectory_dataset.py    # HDF5 dataset loader for (R, s, a) triples
|   |
|   |-- baselines/
|   |   |-- greedy_preempt.py      # Expert heuristic: green wave + MaxPressure
|   |   |-- fixed_time_evp.py      # Fixed-time signal plan with EV priority
|   |   +-- rl_baselines.py        # PPO / DQN wrappers (Stable-Baselines3)
|   |
|   +-- utils/
|       |-- data_collector.py      # Trajectory collection for offline dataset
|       +-- metrics.py             # EV travel time, delay, throughput, etc.
|
|-- tests/
|   +-- test_network_utils.py      # Unit tests for CTM simulation
|
+-- pyproject.toml                 # Package metadata and dependencies
```

## Experiments

### Full Pipeline

Run the complete experiment pipeline (data generation, training all methods, evaluation):

```bash
python scripts/run_all.py --config configs/default.yaml
```

This will sequentially:
1. Generate the offline dataset (5000 episodes, mixed quality)
2. Train the Decision Transformer (100 epochs)
3. Train the Multi-Agent Decision Transformer (100 epochs)
4. Train PPO and DQN baselines (500K timesteps each)
5. Evaluate all methods and save results to `logs/evaluation_results.json`

### Individual Steps

```bash
# Skip data generation if dataset already exists
python scripts/run_all.py --config configs/default.yaml --skip-data

# Train only specific methods
python scripts/train_dt.py --config configs/default.yaml --device cuda
python scripts/train_madt.py --config configs/default.yaml --device cuda
python scripts/train_baselines.py --config configs/default.yaml --method ppo
python scripts/train_baselines.py --config configs/default.yaml --method dqn

# Evaluate all trained models
python scripts/evaluate.py --config configs/default.yaml
```

### Custom Configuration

Edit `configs/default.yaml` to change:
- **Network size:** `env.network` (grid-4x4-v0, grid-6x6-v0, manhattan-v0)
- **Dataset size:** `dataset.num_episodes` and `dataset.suboptimal_ratio`
- **Model architecture:** `dt.n_layers`, `dt.embed_dim`, `madt.gat_heads`, etc.
- **Target returns:** `dt.target_returns` to control the conditioning spectrum

## Results

Results across methods on the **grid-4x4-v0** scenario (100 evaluation episodes):

| Method | EV Travel Time | Background Delay | Throughput | Signal Disruptions | Corridor Green Ratio |
|---|---|---|---|---|---|
| **DT (R\*=0)** | -- | -- | -- | -- | -- |
| **DT (R\*=-50)** | -- | -- | -- | -- | -- |
| **MADT (R\*=0)** | -- | -- | -- | -- | -- |
| PPO | -- | -- | -- | -- | -- |
| DQN | -- | -- | -- | -- | -- |
| Greedy Preempt | -- | -- | -- | -- | -- |
| Fixed-Time EVP | -- | -- | -- | -- | -- |

*Table will be populated after running the full experiment pipeline.*

## Configuration Reference

Key configuration parameters in `configs/default.yaml`:

| Section | Parameter | Default | Description |
|---|---|---|---|
| `env` | `network` | grid-4x4-v0 | LightSim scenario identifier |
| `env` | `ev_speed_factor` | 1.5 | EV speed multiplier over free-flow |
| `env` | `max_episode_steps` | 300 | Episode timeout (~25 min simulated) |
| `dataset` | `num_episodes` | 5000 | Episodes for offline data collection |
| `dataset` | `suboptimal_ratio` | 0.3 | Fraction of random/suboptimal demos |
| `dt` | `context_length` | 30 | Trajectory context window |
| `dt` | `embed_dim` | 128 | Transformer embedding dimension |
| `dt` | `n_layers` | 4 | Number of transformer blocks |
| `dt` | `target_returns` | [0, -50, -100, -200] | Return-conditioning targets |
| `madt` | `gat_heads` | 4 | Graph attention heads |
| `madt` | `gat_layers` | 2 | Stacked GAT layers |

## Citation

```bibtex
@article{su2026evdt,
  title     = {EV-DT: Return-Conditioned Emergency Corridor Optimization
               via Decision Transformer},
  author    = {Su, Haoran},
  year      = {2026},
  note      = {Preprint}
}
```

## License

This project is licensed under the MIT License. See [pyproject.toml](pyproject.toml) for details.
