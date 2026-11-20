"""Point d'entrée local pour exécuter le générateur à travers les workers."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from generator.schema import TrafficConfig
from generator.traffic import generate_arrivals

from .loop import run_arrivals
from .nodes import load_workers
from .policy import build_policy
from .worker_client import WorkerClient


async def run(args: argparse.Namespace) -> None:
    values = {}
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        if not isinstance(values, dict):
            raise ValueError("Le fichier --config doit contenir un objet YAML")
    configured_policy = values.pop("policy", None)
    configured_fallback_policy = values.pop("fallback_policy", None)
    configured_decision_timeout_ms = values.pop("decision_timeout_ms", None)
    overrides = {
        "scenario": args.scenario,
        "rho": args.rho,
        "n_requests": args.requests,
        "duration_ms": args.duration_ms,
        "trace_path": args.trace,
        "trace_count_column": args.trace_count_column,
        "trace_timestamp_column": args.trace_timestamp_column,
        "trace_interval_ms": args.trace_interval_ms,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if args.requests is not None:
        values.pop("duration_ms", None)
    if args.duration_ms is not None:
        values.pop("n_requests", None)
    if args.trace_no_timestamp:
        values["trace_timestamp_column"] = None
    values.setdefault("scenario", "constant")
    values.setdefault("rho", 0.6)
    if "n_requests" not in values and "duration_ms" not in values:
        values["n_requests"] = 30
    config = TrafficConfig(**values)
    arrivals = generate_arrivals(config, seed=args.seed).to_dict(orient="records")
    workers = load_workers()
    policy = build_policy(args.policy or configured_policy or "greedy")
    fallback_policy = build_policy(
        args.fallback_policy or configured_fallback_policy or "least_connections"
    )
    decision_timeout_ms = (
        args.decision_timeout_ms
        if args.decision_timeout_ms is not None
        else float(configured_decision_timeout_ms or 20.0)
    )
    client = WorkerClient(timeout_s=args.timeout)
    try:
        results = await run_arrivals(
            arrivals,
            workers=workers,
            client=client,
            policy=policy,
            fallback_policy=fallback_policy,
            decision_timeout_ms=decision_timeout_ms,
        )
    finally:
        await client.aclose()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    completed = [result for result in results if result["status"] == "completed"]
    met = sum(bool(result["deadline_met"]) for result in completed)
    print(f"Requêtes: {len(results)} | terminées: {len(completed)} | échecs: {len(results) - len(completed)}")
    print(f"Deadlines respectées: {met}/{len(completed)} requêtes terminées")
    print(f"Résultats JSONL: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Configuration YAML de TrafficConfig")
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument("--requests", type=int)
    count_group.add_argument("--duration-ms", type=float)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--policy",
        choices=("greedy", "least_connections"),
        help="Politique principale de routage",
    )
    parser.add_argument(
        "--fallback-policy",
        choices=("greedy", "least_connections"),
        help="Politique de secours en cas d'échec de décision",
    )
    parser.add_argument(
        "--decision-timeout-ms",
        type=float,
        help="Seuil de basculement si la décision dépasse ce délai",
    )
    parser.add_argument("--rho", type=float)
    parser.add_argument(
        "--scenario",
        choices=("constant", "peaks", "diurnal", "flash_crowd", "trace"),
        help="Profil d'arrivée à rejouer",
    )
    parser.add_argument(
        "--trace",
        help="CSV de comptages, requis avec --scenario trace (colonne invocations)",
    )
    parser.add_argument("--trace-count-column")
    parser.add_argument("--trace-timestamp-column")
    parser.add_argument(
        "--trace-no-timestamp",
        action="store_true",
        help="Traiter chaque ligne comme un intervalle de durée --trace-interval-ms",
    )
    parser.add_argument("--trace-interval-ms", type=float)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", default="results/orchestration.jsonl")
    args = parser.parse_args()
    if args.requests is not None and args.requests < 1:
        parser.error("--requests doit être >= 1")
    if args.timeout <= 0:
        parser.error("--timeout doit être > 0")
    if args.decision_timeout_ms is not None and args.decision_timeout_ms <= 0:
        parser.error("--decision-timeout-ms doit être > 0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
