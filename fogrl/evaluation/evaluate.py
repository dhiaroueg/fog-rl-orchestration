"""evaluate.py : execute une politique sur un scenario et des graines, exporte les metriques en CSV.

Exemples :
    python -m fogrl.evaluation.evaluate --policy greedy --policy round_robin \\
        --scenario medium --configuration n4 --seeds 1000-1004 --out results/baselines.csv
    python -m fogrl.evaluation.evaluate --policy rl:models/n3/ppo/seed0 --scenario high \\
        --configuration n6 --seeds 1000-1004 --out results/ppo.csv

Une ligne par (politique, scenario, taille, graine) avec toutes les metriques
(cahier des charges, section 6.2).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config import EnvConfig
from ..metrics import percentile
from ..policies.base import Policy
from ..policies.baselines import BASELINES
from ..policies.factory import is_rl_spec, make_policy
from ..sim.env import FogEnv
from ..utils import check_eval_seeds, git_commit, parse_seeds


def run_episode(
    env: FogEnv,
    policy: Policy,
    seed: int,
    options: Optional[dict] = None,
    latencies: Optional[list] = None,
) -> dict:
    """Joue un episode complet ; retourne les metriques (+ temps de decision)."""

    env.reset(seed=seed, options=options)
    policy.reset(seed)
    decision_ms = []
    while True:
        nodes, request, ctx = env.raw_state()
        start = time.perf_counter()
        action = policy.select(nodes, request, ctx)
        decision_ms.append((time.perf_counter() - start) * 1000.0)
        _, _, _, truncated, info = env.step(action)
        if latencies is not None and info["latency_ms"] is not None:
            latencies.append(info["latency_ms"])
        if truncated:
            break
    metrics = dict(info["episode_metrics"])
    metrics["size"] = metrics.pop("n_active")
    metrics["decision_ms_mean"] = float(np.mean(decision_ms))
    metrics["decision_ms_p99"] = percentile(np.asarray(decision_ms), 99)
    metrics["fallback_rate"] = 0.0  # pas de repli en simulation
    return metrics


def evaluate_policy(
    policy: Policy,
    env: FogEnv,
    seeds: Sequence[int],
    *,
    policy_name: str,
    policy_type: str,
    scenario: Optional[str] = None,
    configuration: Optional[str] = None,
    train_seed: Optional[int] = None,
    latencies: Optional[list] = None,
) -> list[dict]:
    check_eval_seeds(seeds)
    options = {k: v for k, v in (("scenario", scenario), ("configuration", configuration)) if v}
    commit = git_commit()
    rows = []
    for seed in seeds:
        metrics = run_episode(env, policy, seed, options, latencies)
        rows.append(
            {
                "policy": policy_name,
                "policy_type": policy_type,
                "train_seed": train_seed,
                "seed": seed,
                **metrics,
                "commit": commit,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="configs/env/eval.yaml")
    parser.add_argument("--policy", action="append", required=True, help=f"baseline {BASELINES} ou rl:<dossier>")
    parser.add_argument("--scenario", action="append", help="repetable ; defaut : celui du fichier env")
    parser.add_argument("--configuration", default=None, help="n2, n3, n4 ou n6")
    parser.add_argument("--seeds", default="1000-1004")
    parser.add_argument("--n-requests", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    overrides = {"episode_requests": args.n_requests} if args.n_requests else {}
    config = EnvConfig.from_yaml(args.env, overrides)
    env = FogEnv(config)
    profiles = {p.node_id: p for p in config.pool}
    scenarios = args.scenario or [config.scenarios[0].name]
    rows = []
    for spec in args.policy:
        policy = make_policy(spec, profiles, config.slot_ids, config.normalization)
        kind = "rl" if is_rl_spec(spec) else "baseline"
        name = Path(spec[3:]).parent.name if kind == "rl" else spec
        for scenario in scenarios:
            rows += evaluate_policy(
                policy, env, parse_seeds(args.seeds), policy_name=name, policy_type=kind,
                scenario=scenario, configuration=args.configuration,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    frame = pd.DataFrame(rows)
    print(
        frame.groupby(["policy", "scenario"])[["latency_mean_ms", "deadline_rate", "energy_fog_j_per_req", "cloud_cost_per_req"]]
        .mean()
        .round(3)
        .to_string()
    )
    print(f"\n{len(rows)} lignes -> {out}")


if __name__ == "__main__":
    main()
