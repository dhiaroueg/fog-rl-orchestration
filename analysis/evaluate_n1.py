"""Compare the N1 PPO model with simple policies on shared traffic seeds.

Run from the repository root, for example:
    python -m analysis.evaluate_n1 --model models/ppo_n1_smoke.zip

This evaluates a short smoke-trained model. Results are diagnostic, not a
scientific claim about the final policy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from sim.fog_env import FogEnv
from state_spec import DEFAULT_NORMALIZATION, FEATURES_PER_NODE


def greedy_action(obs: np.ndarray, env: FogEnv) -> int:
    """Choose the node with the smallest estimated response time."""

    n_nodes = len(env.profiles)
    request_start = n_nodes * FEATURES_PER_NODE
    work_ms = float(obs[request_start + 3]) * DEFAULT_NORMALIZATION.work_ms
    scores = []
    for i, profile in enumerate(env.profiles):
        queue_ms = float(obs[i * FEATURES_PER_NODE]) * DEFAULT_NORMALIZATION.queue_ms
        wait_ms = queue_ms / (profile.speed * profile.concurrency)
        service_ms = work_ms / profile.speed
        network_ms = float(obs[i * FEATURES_PER_NODE + 2]) * DEFAULT_NORMALIZATION.net_rtt_ms
        scores.append(wait_ms + service_ms + network_ms)
    return int(np.argmin(scores))


def evaluate_episode(policy_name: str, model: PPO | None, seed: int, n_requests: int) -> dict:
    env = FogEnv(n_requests=n_requests)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    next_round_robin = 0
    latencies: list[float] = []
    missed: list[bool] = []
    rewards: list[float] = []
    actions = np.zeros(len(env.profiles), dtype=np.int64)

    while True:
        if policy_name == "PPO":
            action = int(model.predict(obs, deterministic=True)[0])
        elif policy_name == "Random":
            action = int(rng.integers(len(env.profiles)))
        elif policy_name == "Round-robin":
            action = next_round_robin
            next_round_robin = (next_round_robin + 1) % len(env.profiles)
        elif policy_name == "Greedy":
            action = greedy_action(obs, env)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        actions[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        latencies.append(float(info["response_ms"]))
        missed.append(bool(info["deadline_missed"]))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    row = {
        "policy": policy_name,
        "seed": seed,
        "requests": n_requests,
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "deadline_miss_rate": float(np.mean(missed)),
        "mean_reward": float(np.mean(rewards)),
    }
    row.update({f"share_{p.node_id}": float(actions[i] / n_requests)
                for i, p in enumerate(env.profiles)})
    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/ppo_n1_smoke.zip")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--output", default="results/n1_smoke_eval.csv")
    args = parser.parse_args()
    if args.episodes < 1 or args.requests < 1:
        parser.error("--episodes et --requests doivent etre >= 1")

    model_path = Path(args.model)
    if not model_path.exists() and not Path(f"{model_path}.zip").exists():
        parser.error(f"Modèle introuvable : {model_path}")
    model = PPO.load(str(model_path))

    policies = ("PPO", "Random", "Round-robin", "Greedy")
    rows = []
    for policy_name in policies:
        policy_rows = [
            evaluate_episode(
                policy_name,
                model,
                seed,
                args.requests,
            )
            for seed in range(args.seed_start, args.seed_start + args.episodes)
        ]
        rows.extend(policy_rows)
        print(
            f"{policy_name:12s} "
            f"latence moyenne={np.mean([r['mean_latency_ms'] for r in policy_rows]):.2f} ms | "
            f"P95 moyen={np.mean([r['p95_latency_ms'] for r in policy_rows]):.2f} ms | "
            f"échéances manquées={100 * np.mean([r['deadline_miss_rate'] for r in policy_rows]):.2f}%"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRésultats détaillés : {output.resolve()}")


if __name__ == "__main__":
    main()
