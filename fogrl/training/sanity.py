"""Test de coherence N0 (cahier, Phase 2 etape 9 ; jalon G2).

2 noeuds (un lent, un rapide), recompense = latence seule : un PPO doit
apprendre a privilegier le noeud rapide. Sinon, il y a un bug dans
l'environnement (ou dans la recompense, ou dans l'encodage de l'etat).

    python -m fogrl.training.sanity [--timesteps 60000]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from typing import Any, Optional, Sequence

import numpy as np

from ..config import EnvConfig, load_yaml
from ..evaluation.evaluate import run_episode
from ..policies.baselines import make_baseline
from ..policies.rl import RLPolicy
from ..sim.env import FogEnv
from .train import train


def run_n0_check(
    timesteps: int = 60000,
    seed: int = 0,
    config_path: str = "configs/train/n0.yaml",
    fast_node: str = "fogC",
    eval_seeds: Sequence[int] = (1000, 1001, 1002),
    min_fast_share: float = 0.8,
    out_root: Optional[str] = None,
) -> dict[str, Any]:
    config = {**load_yaml(config_path), "eval": {"freq": 0}}
    with tempfile.TemporaryDirectory() as tmp:
        root = out_root or tmp
        model_dir = train(config, "ppo", seed, out_root=root, runs_root=f"{root}/runs", total_timesteps=timesteps)
        env_cfg = EnvConfig.from_yaml(config["env"])
        env = FogEnv(env_cfg)
        profiles = {p.node_id: p for p in env_cfg.pool}
        agent = RLPolicy.load(model_dir, slot_ids=env_cfg.slot_ids, normalization=env_cfg.normalization)
        rl = [run_episode(env, agent, s) for s in eval_seeds]
        rnd = [run_episode(env, make_baseline("random", profiles), s) for s in eval_seeds]
    result = {
        "fast_node": fast_node,
        "fast_node_share": float(np.mean([m[f"share_{fast_node}"] for m in rl])),
        "latency_rl_ms": float(np.mean([m["latency_mean_ms"] for m in rl])),
        "latency_random_ms": float(np.mean([m["latency_mean_ms"] for m in rnd])),
        "timesteps": timesteps,
    }
    result["passed"] = bool(
        result["fast_node_share"] >= min_fast_share and result["latency_rl_ms"] < result["latency_random_ms"]
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_n0_check(args.timesteps, args.seed)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Test N0 :", "REUSSI" if result["passed"] else "ECHEC (verifier l'environnement)")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
