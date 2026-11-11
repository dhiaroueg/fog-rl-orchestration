"""Point d'entrée local pour exécuter le générateur à travers les workers."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from baselines.greedy import GreedyPolicy
from generator.schema import TrafficConfig
from generator.traffic import generate_arrivals

from .loop import run_arrivals
from .nodes import load_workers
from .worker_client import WorkerClient


async def run(args: argparse.Namespace) -> None:
    config = TrafficConfig(rho=args.rho, n_requests=args.requests)
    arrivals = generate_arrivals(config, seed=args.seed).to_dict(orient="records")
    workers = load_workers()
    # crée GreedyPolicy() et la transmet à la boucle.
    policy = GreedyPolicy()
    client = WorkerClient(timeout_s=args.timeout)
    try:
        results = await run_arrivals(arrivals, workers=workers, client=client, policy=policy)
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
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", default="results/orchestration.jsonl")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests doit être >= 1")
    if args.timeout <= 0:
        parser.error("--timeout doit être > 0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
