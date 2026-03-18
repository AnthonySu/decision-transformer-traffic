"""Run PPO and DQN baselines on EVCorridorEnv and save results."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.envs.ev_corridor_env import EVCorridorEnv
from src.envs.wrappers import FlattenActionWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor


def main():
    # ------------------------------------------------------------------
    # Train PPO (works with MultiDiscrete natively)
    # ------------------------------------------------------------------
    print("Training PPO...")
    ppo_env = Monitor(EVCorridorEnv(rows=3, cols=3, max_steps=80, seed=42))
    ppo = PPO(
        "MlpPolicy", ppo_env, verbose=0,
        n_steps=128, batch_size=64, n_epochs=4,
    )
    ppo.learn(total_timesteps=5000)
    print("PPO training done.")

    # ------------------------------------------------------------------
    # Train DQN (needs Discrete action space via FlattenActionWrapper)
    # ------------------------------------------------------------------
    print("Training DQN...")
    dqn_base_env = EVCorridorEnv(rows=3, cols=3, max_steps=80, seed=42)
    dqn_env = Monitor(FlattenActionWrapper(dqn_base_env))
    dqn = DQN(
        "MlpPolicy", dqn_env, verbose=0,
        buffer_size=2000, batch_size=32, learning_starts=200,
    )
    dqn.learn(total_timesteps=5000)
    print("DQN training done.")

    # ------------------------------------------------------------------
    # Evaluate both
    # ------------------------------------------------------------------
    results = {}
    num_eval = 30

    for name, model in [("PPO", ppo), ("DQN", dqn)]:
        print(f"Evaluating {name}...")
        returns, ev_times = [], []
        bg_delays, throughputs = [], []
        phase_changes_count = []

        for ep in range(num_eval):
            eval_env = EVCorridorEnv(rows=3, cols=3, max_steps=80, seed=123 + ep)
            if name == "DQN":
                eval_env = FlattenActionWrapper(eval_env)

            obs, info = eval_env.reset()
            done, total_r = False, 0.0
            ep_throughput = 0.0
            ep_phase_changes = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                done = term or trunc
                total_r += r
                ep_throughput += info.get("throughput", 0.0)
                if info.get("phase_changed_for_ev", False):
                    ep_phase_changes += 1

            returns.append(total_r)
            ev_times.append(info.get("ev_travel_time", -1))
            bg_delays.append(info.get("background_delay", 0.0))
            throughputs.append(ep_throughput)
            phase_changes_count.append(ep_phase_changes)

        arrived = [t for t in ev_times if t > 0]
        avg_return = float(np.mean(returns))
        avg_ev_time = float(np.mean(arrived)) if arrived else -1.0
        arrival_rate = len(arrived) / num_eval
        avg_bg_delay = float(np.mean(bg_delays))
        avg_throughput = float(np.mean(throughputs))
        avg_phase_changes = float(np.mean(phase_changes_count))

        results[name] = {
            "avg_return": round(avg_return, 2),
            "avg_ev_travel_time": round(avg_ev_time, 2),
            "arrival_rate": round(arrival_rate, 3),
            "avg_background_delay": round(avg_bg_delay, 2),
            "avg_throughput": round(avg_throughput, 2),
            "avg_phase_changes_for_ev": round(avg_phase_changes, 2),
            "num_episodes": num_eval,
        }

        print(
            f"  {name}: return={avg_return:.1f}, "
            f"ev_time={avg_ev_time:.1f}, "
            f"arrival_rate={arrival_rate:.2f}, "
            f"bg_delay={avg_bg_delay:.1f}, "
            f"throughput={avg_throughput:.1f}"
        )

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_comparison.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
