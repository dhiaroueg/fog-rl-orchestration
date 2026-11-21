"""Fidelite simulation / reel (OS-7, ENF-08, jalon G4).

Critere : ecart relatif |reel - sim| / sim < 20 % sur la latence moyenne pour
chaque (politique, scenario) ET meme classement des politiques par scenario.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional, Sequence

import pandas as pd
from scipy import stats


def compare_sim_real(
    sim: pd.DataFrame,
    real: pd.DataFrame,
    metric: str = "latency_mean_ms",
    tolerance: float = 0.20,
) -> dict[str, Any]:
    keys = ["policy", "scenario"]
    sim_mean = sim.groupby(keys)[metric].mean().rename("sim")
    real_mean = real.groupby(keys)[metric].mean().rename("real")
    joined = pd.concat([sim_mean, real_mean], axis=1, join="inner").reset_index()
    if joined.empty:
        raise ValueError("Aucune paire (politique, scenario) commune entre simulation et reel")
    joined["relative_error"] = (joined["real"] - joined["sim"]).abs() / joined["sim"]

    rankings = {}
    for scenario, group in joined.groupby("scenario"):
        sim_order = list(group.sort_values("sim")["policy"])
        real_order = list(group.sort_values("real")["policy"])
        tau = stats.kendalltau(group["sim"], group["real"]).statistic if len(group) > 1 else float("nan")
        rankings[scenario] = {
            "sim_order": sim_order,
            "real_order": real_order,
            "same_ranking": sim_order == real_order,
            "kendall_tau": float(tau),
        }
    within = bool((joined["relative_error"] < tolerance).all())
    same = all(r["same_ranking"] for r in rankings.values())
    return {
        "metric": metric,
        "tolerance": tolerance,
        "mean_relative_error": float(joined["relative_error"].mean()),
        "max_relative_error": float(joined["relative_error"].max()),
        "all_within_tolerance": within,
        "same_ranking_everywhere": same,
        "passed": within and same,
        "pairs": joined.to_dict(orient="records"),
        "rankings": rankings,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="CSV de resultats de simulation")
    parser.add_argument("--real", required=True, help="CSV de resultats reels")
    parser.add_argument("--tolerance", type=float, default=0.20)
    args = parser.parse_args(argv)
    report = compare_sim_real(pd.read_csv(args.sim), pd.read_csv(args.real), tolerance=args.tolerance)
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
    print("G4 :", "ATTEINT" if report["passed"] else "NON ATTEINT")


if __name__ == "__main__":
    main()
