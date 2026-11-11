"""Évalue des politiques sur un environnement Gymnasium et exporte un CSV.

Exemple (adapter le chemin de classe à votre dépôt) :
    python -m analysis.evaluate --env-factory package.module:ToyFogEnv \
        --episodes 30 --output results/baselines.csv

L'environnement doit suivre l'API Gymnasium :
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)

La latence est lue depuis info['latency_ms'], info['latency'] ou
info['response_time_ms']. Si aucune de ces clés n'existe, le script utilise
-reward UNIQUEMENT si --reward-is-negative-latency est fourni.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import csv
import math
import statistics
from typing import Callable

import numpy as np

from baselines.random_policy import RandomPolicy
from baselines.round_robin import RoundRobinPolicy
from baselines.greedy import GreedyPolicy


def load_factory(spec: str) -> Callable:
    """Charge une classe/fonction via 'module:attribute'."""
    if ":" not in spec:
        raise ValueError("--env-factory doit être au format module:NomClasse")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"{spec} n'est pas appelable.")
    return factory


def get_latency(info: dict, reward: float, reward_is_negative_latency: bool) -> float:
    for key in ("latency_ms", "latency", "response_time_ms", "response_time"):
        if key in info and info[key] is not None:
            value = float(info[key])
            if math.isfinite(value) and value >= 0:
                return value
    if reward_is_negative_latency:
        value = -float(reward)
        if math.isfinite(value) and value >= 0:
            return value
    raise KeyError(
        "Latence absente de info. Ajoutez-la à info['latency_ms'] dans ToyFogEnv, "
        "ou passez --reward-is-negative-latency si reward = -latence."
    )


def evaluate_episode(env, policy, seed: int, reward_is_negative_latency: bool) -> dict:
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _reset_info = reset_result
    else:  # compatibilité Gym ancien, à éviter si possible
        obs = reset_result

    policy.reset(seed=seed)
    latencies = []
    rewards = []
    steps = 0

    while True:
        action = policy.select(obs)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
            done = bool(done)
        else:
            raise RuntimeError("env.step() doit retourner 4 ou 5 éléments.")

        info = info or {}
        latencies.append(get_latency(info, float(reward), reward_is_negative_latency))
        rewards.append(float(reward))
        steps += 1

        if done:
            break

    return {
        "seed": seed,
        "steps": steps,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else float("nan"),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
    }


def make_policies():
    return [
        RandomPolicy(),
        RoundRobinPolicy(),
        GreedyPolicy(speeds=(2.0, 1.0)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-factory", required=True,
                        help="Import de l'environnement, format module:ToyFogEnv")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output", default="results/baselines.csv")
    parser.add_argument("--reward-is-negative-latency", action="store_true",
                        help="À utiliser seulement si reward = -latence.")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes doit être >= 1")

    env_factory = load_factory(args.env_factory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # Chaque politique reçoit les mêmes seeds. On crée un environnement neuf
    # par épisode pour éviter que l'état d'une politique contamine la suivante.
    for policy in make_policies():
        episode_rows = []
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            env = env_factory()
            try:
                result = evaluate_episode(
                    env, policy, seed, args.reward_is_negative_latency
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            result["policy"] = policy.name
            episode_rows.append(result)
            rows.append(result)

        values = [r["mean_latency_ms"] for r in episode_rows]
        print(
            f"{policy.name:18s} "
            f"latence={statistics.mean(values):.3f} ms "
            f"écart-type={statistics.stdev(values) if len(values) > 1 else 0.0:.3f} ms"
        )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["policy", "seed", "steps", "mean_latency_ms",
                           "p95_latency_ms", "mean_reward"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV écrit dans : {output.resolve()}")


if __name__ == "__main__":
    main()
