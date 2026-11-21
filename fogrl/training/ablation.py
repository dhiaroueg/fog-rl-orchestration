"""Etude d'ablation des poids de la recompense (cahier, Phase 4 ; front de Pareto energie / latence).

    python -m fogrl.training.ablation --config configs/experiments/ablation.yaml --out results/ablation

Pour chaque variation (un poids change a la fois), entraine un agent puis
l'evalue sur les graines d'evaluation. Produit ``ablation.csv`` (une ligne par
variation, scenario et graine, avec les poids w1..w5) et la figure ``pareto.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..config import EnvConfig, load_yaml
from ..evaluation.evaluate import evaluate_policy
from ..evaluation.figures import plot_pareto
from ..policies.rl import RLPolicy
from ..sim.env import FogEnv
from .train import train


def run_ablation(spec: Mapping[str, Any], out_dir, out_root: str = "models") -> Path:
    base = load_yaml(spec["train_config"])
    evaluation = spec["eval"]
    rows = []
    for variation in spec["variations"]:
        tag = "_".join(f"{k}{v}" for k, v in variation.items())
        config = {
            **base,
            "level": f"ablation_{tag}",
            "env_overrides": {**(base.get("env_overrides") or {}), "reward": variation},
            "eval": {"freq": 0},
        }
        model_dir = train(config, spec.get("algo", "ppo"), spec.get("train_seed", 0), out_root, f"{out_root}/runs", spec.get("timesteps"))
        eval_cfg = EnvConfig.from_yaml(
            evaluation["env"],
            {
                "layout": base_layout(base),
                "configuration": evaluation["configuration"],
                "episode_requests": evaluation["n_requests"],
                "reward": variation,
            },
        )
        env = FogEnv(eval_cfg)
        agent = RLPolicy.load(model_dir, slot_ids=eval_cfg.slot_ids, normalization=eval_cfg.normalization)
        for scenario in evaluation["scenarios"]:
            for row in evaluate_policy(
                agent, env, evaluation["seeds"], policy_name=spec.get("algo", "ppo"), policy_type="rl",
                scenario=scenario, configuration=evaluation["configuration"],
            ):
                rows.append({**{f"w{i}": getattr(eval_cfg.reward, f"w{i}") for i in range(1, 6)}, "variation": tag, **row})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "ablation.csv", index=False)
    varied_w2 = frame[frame["variation"].str.startswith("w2")]
    if varied_w2["w2"].nunique() > 1:
        plot_pareto(varied_w2, out / "pareto.png", "w2")
    return out / "ablation.csv"


def base_layout(base: Mapping[str, Any]) -> str:
    return EnvConfig.from_yaml(base["env"], base.get("env_overrides")).layout


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/ablation.yaml")
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--models", default="models")
    args = parser.parse_args(argv)
    print("Resultats :", run_ablation(load_yaml(args.config), args.out, args.models))


if __name__ == "__main__":
    main()
