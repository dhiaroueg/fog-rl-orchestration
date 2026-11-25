"""Campagne d'experiences sur le prototype reel (cahier, Phase 4) : politiques x scenarios x graines.

    python -m fogrl.loadgen.campaign --matrix configs/experiments/real_campaign.yaml --out results/real_campaign

Pour chaque politique, la campagne la charge a chaud dans l'orchestrateur (``POST /policy``), puis
rejoue chaque scenario pour chaque graine apres une remise a zero. Le CSV produit a les memes
colonnes que celui du simulateur (``results.csv`` de ``fogrl.evaluation.experiments``) : il se
compare directement avec ``python -m fogrl.evaluation.compare`` (jalon G4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import httpx
import pandas as pd

from ..config import EnvConfig, load_yaml
from ..utils import check_eval_seeds, git_commit, library_versions
from .generator import run_scenario


def policy_spec(policy: Mapping[str, Any]) -> str:
    return f"rl:{policy['model']}" if policy.get("type", "baseline") == "rl" else policy["name"]


async def run_campaign(
    matrix: Mapping[str, Any],
    out_dir,
    *,
    client: Optional[httpx.AsyncClient] = None,
    verbose: bool = True,
    save_requests: bool = False,
) -> Path:
    check_eval_seeds(matrix["seeds"])
    scenarios = list(matrix["scenarios"])
    configuration = matrix.get("configuration", "n4")
    config = EnvConfig.from_yaml(
        matrix.get("env", "configs/env/eval.yaml"), {"configuration": configuration, "scenarios": scenarios}
    )
    out = Path(out_dir)
    (out / "requests").mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=matrix.get("orchestrator_url", "http://localhost:9000"), timeout=60.0)

    rows: list[dict] = []
    started = time.time()
    total = len(matrix["policies"]) * len(scenarios) * len(matrix["seeds"])
    done = 0
    try:
        for policy in matrix["policies"]:
            reply = await client.post("/policy", json={"spec": policy_spec(policy), "name": policy["name"]})
            if reply.status_code != 200:
                raise RuntimeError(f"Chargement de {policy['name']} refuse par l'orchestrateur : {reply.text}")
            for scenario in scenarios:
                for seed in matrix["seeds"]:
                    await asyncio.sleep(matrix.get("cooldown_s", 0.0))  # laisse les files se vider
                    summary, request_rows = await run_scenario(
                        client,
                        config,
                        configuration,
                        scenario,
                        seed,
                        duration_s=matrix.get("duration_s", 60.0),
                        max_in_flight=matrix.get("max_in_flight", 1000),
                    )
                    rows.append(
                        {
                            "policy": policy["name"],
                            "policy_type": policy.get("type", "baseline"),
                            "train_seed": policy.get("train_seed"),
                            "seed": seed,
                            **summary,
                            "commit": commit,
                            "source": "real",
                        }
                    )
                    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)  # sauvegarde au fil de l'eau
                    if save_requests:
                        pd.DataFrame(request_rows).to_csv(out / "requests" / f"{policy['name']}_{scenario}_{seed}.csv", index=False)
                    done += 1
                    if verbose:
                        print(
                            f"[{done}/{total}] {policy['name']:<18} {scenario:<12} seed {seed} : "
                            f"{summary['latency_mean_ms']:.1f} ms, echeances {summary['deadline_rate']:.3f}, "
                            f"rejets {summary['rejection_rate']:.3f}, repli {summary['fallback_rate']:.3f}",
                            flush=True,
                        )
        final_state = (await client.get("/state")).json()
    finally:
        if owns_client:
            await client.aclose()

    info = {
        "name": matrix.get("name"),
        "commit": commit,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "n_rows": len(rows),
        "versions": library_versions(),
        "matrix": matrix,
        "env_digest": config.digest(),
        "orchestrator_final_state": final_state,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return out / "results.csv"


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - necessite une pile en marche
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default=None, help="URL de l'orchestrateur (defaut : celle de la matrice)")
    parser.add_argument("--save-requests", action="store_true", help="un CSV par run avec toutes les requetes")
    args = parser.parse_args(argv)
    matrix = dict(load_yaml(args.matrix))
    if args.url:
        matrix["orchestrator_url"] = args.url
    path = asyncio.run(run_campaign(matrix, args.out, save_requests=args.save_requests))
    print(f"Resultats : {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
