"""Matrice d'experiences pilotee par un fichier YAML (cahier, Phase 3 etape 6 et Phase 4).

    python -m fogrl.evaluation.experiments --matrix configs/experiments/sim_full.yaml --out results/sim_full

Produit ``results.csv`` (une ligne par politique x scenario x taille x graine
[x graine d'entrainement]) et ``run_info.json`` (commit git, configuration,
versions) : chaque CSV est ainsi associe au commit et a la config qui l'ont
produit (regle 6.3).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..config import EnvConfig, load_yaml, resolve_path
from ..sim.env import FogEnv
from ..utils import check_eval_seeds, git_commit, library_versions
from .evaluate import evaluate_policy, make_policy


def expand_tasks(matrix: Mapping[str, Any]) -> list[dict]:
    """Une tache = (politique [x modele], scenario, taille) ; chaque tache joue toutes les graines."""

    tasks = []
    check_eval_seeds(matrix["seeds"])
    for policy in matrix["policies"]:
        kind = policy.get("type", "baseline")
        sizes = policy.get("sizes", matrix.get("sizes", [4]))
        scenarios = policy.get("scenarios", matrix["scenarios"])
        if kind == "rl":
            model_dirs = sorted(
                p for p in glob.glob(str(resolve_path(policy["model_glob"]).as_posix())) if Path(p).is_dir()
            )
            if not model_dirs:
                warnings.warn(f"Aucun modele pour {policy['name']} ({policy['model_glob']}) : politique ignoree")
                continue
        else:
            model_dirs = [None]
        for model_dir in model_dirs:
            train_seed = None
            if model_dir:
                found = re.search(r"seed(\d+)", Path(model_dir).name)
                train_seed = int(found.group(1)) if found else None
            for scenario in scenarios:
                for size in sizes:
                    tasks.append(
                        {
                            "name": policy["name"],
                            "kind": kind,
                            "spec": f"rl:{model_dir}" if model_dir else policy["name"],
                            "train_seed": train_seed,
                            "scenario": scenario,
                            "size": size,
                            "env": matrix.get("env", "configs/env/eval.yaml"),
                            "env_overrides": policy.get("env_overrides", {}),
                            "seeds": list(matrix["seeds"]),
                            "n_requests": matrix.get("n_requests", 5000),
                        }
                    )
    return tasks


def run_task(task: Mapping[str, Any]) -> list[dict]:
    overrides = {
        "configuration": f"n{task['size']}",
        "scenarios": [task["scenario"]],
        "episode_requests": task["n_requests"],
        **task["env_overrides"],
    }
    config = EnvConfig.from_yaml(task["env"], overrides)
    env = FogEnv(config)
    profiles = {p.node_id: p for p in config.pool}
    policy = make_policy(task["spec"], profiles, config.slot_ids, config.normalization, name=task["name"])
    return evaluate_policy(
        policy,
        env,
        task["seeds"],
        policy_name=task["name"],
        policy_type=task["kind"],
        scenario=task["scenario"],
        configuration=f"n{task['size']}",
        train_seed=task["train_seed"],
    )


def run_matrix(
    matrix: Mapping[str, Any], out_dir, n_jobs: int = 1, verbose: bool = True
) -> Path:
    tasks = expand_tasks(matrix)
    if not tasks:
        raise ValueError("Matrice vide : aucune tache a executer")
    started = time.time()
    rows: list[dict] = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for done, result in enumerate(pool.map(run_task, tasks), 1):
                rows += result
                if verbose:
                    print(f"[{done}/{len(tasks)}] taches terminees", flush=True)
    else:
        for done, task in enumerate(tasks, 1):
            rows += run_task(task)
            if verbose:
                print(f"[{done}/{len(tasks)}] {task['name']} {task['scenario']} n{task['size']}", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    info = {
        "name": matrix.get("name"),
        "commit": git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "n_tasks": len(tasks),
        "n_rows": len(rows),
        "versions": library_versions(),
        "matrix": matrix,
        "env_digest": EnvConfig.from_yaml(matrix.get("env", "configs/env/eval.yaml")).digest(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return out / "results.csv"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    path = run_matrix(load_yaml(args.matrix), args.out, args.n_jobs)
    print(f"Resultats : {path}")


if __name__ == "__main__":
    main()
