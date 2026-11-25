"""Generateur de charge asynchrone (EF-06) : rejoue un scenario de trafic contre l'orchestrateur.

Les instants d'arrivee et les types de requetes viennent de ``fogrl.traffic`` : meme graine = meme
sequence qu'en simulation, avec les memes parametres de scenario. Les metriques d'un run sont
calculees par ``fogrl.metrics.summarize_arrays`` (memes definitions qu'en simulation), ce qui rend
les CSV reels et simules directement comparables (``fogrl.evaluation.compare``).

Exemple (une politique deja chargee dans l'orchestrateur) :
    python -m fogrl.loadgen.generator --url http://localhost:9000 --scenario medium \\
        --duration 60 --seed 1000 --out results/run_medium.csv
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import httpx
import numpy as np
import pandas as pd

from ..config import EnvConfig, NodeProfile, RewardConfig
from ..metrics import jain_index, percentile, summarize_arrays
from ..reward import compute_reward
from ..traffic import RequestStream, base_rate_rps, generate_stream_until


async def run_load(
    client: httpx.AsyncClient,
    stream: RequestStream,
    deadlines: Mapping[str, float],
    *,
    max_in_flight: int = 1000,
    request_timeout_s: float = 30.0,
) -> list[dict]:
    """Envoie les requetes de ``stream`` a l'heure prevue et retourne une ligne de resultat par requete."""

    semaphore = asyncio.Semaphore(max_in_flight)
    rows: list[Optional[dict]] = [None] * len(stream)
    origin = time.perf_counter()

    async def send(i: int, target_s: float) -> None:
        req_type = stream.type_of(i)
        work_ms = float(stream.work_ms[i])
        row = {"index": i, "type": req_type, "work_ms": work_ms, "deadline_ms": deadlines[req_type], "target_s": target_s}
        async with semaphore:
            sent = time.perf_counter()
            row["lag_ms"] = (sent - origin - target_s) * 1000.0
            try:
                reply = await client.post(
                    "/route",
                    json={"type": req_type, "work_ms": work_ms, "deadline_ms": deadlines[req_type]},
                    timeout=request_timeout_s,
                )
                reply.raise_for_status()
                row.update(reply.json())
            except (httpx.HTTPError, ValueError) as exc:
                row.update(rejected=True, reason=f"loadgen_{type(exc).__name__}", node=None, latency_ms=None,
                           decision_ms=0.0, fallback=False, energy_j=0.0, cost=0.0)
            row["e2e_ms"] = (time.perf_counter() - sent) * 1000.0
        rows[i] = row

    tasks = []
    for i, target in enumerate(stream.times_s):
        delay = origin + float(target) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(send(i, float(target))))
    await asyncio.gather(*tasks)
    return [row for row in rows if row is not None]


def summarize_run(
    rows: Sequence[dict],
    *,
    slot_ids: Sequence[str],
    profiles: Mapping[str, NodeProfile],
    reward: RewardConfig,
    duration_s: float,
    nodes_before: Mapping[str, dict],
    nodes_after: Mapping[str, dict],
) -> dict:
    """Metriques d'un run reel (memes colonnes que ``FogEnv`` + temps de decision et retard du generateur)."""

    index = {slot: i for i, slot in enumerate(slot_ids)}
    rejected = np.asarray([bool(r["rejected"]) for r in rows])
    latency = np.asarray([np.nan if r["latency_ms"] is None else r["latency_ms"] for r in rows], dtype=float)
    deadline = np.asarray([r["deadline_ms"] for r in rows], dtype=float)
    energy = np.asarray([r["energy_j"] for r in rows], dtype=float)
    cost = np.asarray([r["cost"] for r in rows], dtype=float)
    node_index = np.asarray([index.get(r["node"], -1) for r in rows])
    rewards = [
        compute_reward(reward, 0.0 if rej else lat, dl, e, c, bool(rej))
        for rej, lat, dl, e, c in zip(rejected, latency, deadline, energy, cost)
    ]

    fog = [n for n in nodes_after if n in profiles and not profiles[n].is_cloud and n in nodes_before]
    idle = sum(nodes_after[n]["energy_idle_j"] - nodes_before[n]["energy_idle_j"] for n in fog)
    utilization = [
        min(1.0, (nodes_after[n]["busy_time_s"] - nodes_before[n]["busy_time_s"]) / (profiles[n].servers * duration_s))
        for n in fog
    ]
    summary = summarize_arrays(
        latency_ms=np.nan_to_num(latency),
        deadline_ms=deadline,
        rejected=rejected,
        node_index=node_index,
        energy_j=energy,
        cost=cost,
        slot_ids=slot_ids,
        duration_s=duration_s,
        idle_energy_j=float(idle),
        jain_cpu=jain_index(utilization) if utilization else float("nan"),
        reward_sum=float(sum(rewards)),
    )
    decision = np.asarray([r["decision_ms"] for r in rows], dtype=float)
    summary["decision_ms_mean"] = float(decision.mean()) if len(decision) else float("nan")
    summary["decision_ms_p99"] = percentile(decision, 99)
    summary["fallback_rate"] = float(np.mean([bool(r["fallback"]) for r in rows])) if rows else float("nan")
    summary["loadgen_lag_ms_p99"] = percentile(np.asarray([r["lag_ms"] for r in rows]), 99)
    return summary


async def fetch_nodes(client: httpx.AsyncClient) -> dict:
    reply = await client.get("/nodes", params={"fresh": True})
    reply.raise_for_status()
    return reply.json()


async def run_scenario(
    client: httpx.AsyncClient,
    config: EnvConfig,
    configuration: str,
    scenario: str,
    seed: int,
    *,
    duration_s: float,
    reset: bool = True,
    max_in_flight: int = 1000,
) -> tuple[dict, list[dict]]:
    """Un run complet : remise a zero, flux de ``duration_s`` secondes, metriques. Retourne (resume, lignes)."""

    specs = {s.name: s for s in config.scenarios}
    if scenario not in specs:
        raise ValueError(f"Scenario {scenario!r} absent de la configuration (disponibles : {sorted(specs)})")
    active = [config.profile(i) for i in config.active_ids(configuration)]
    rate = base_rate_rps(specs[scenario], active, config.requests)
    stream = generate_stream_until(specs[scenario], rate, duration_s, config.requests, seed)
    if len(stream) == 0:
        raise ValueError("Flux vide : allongez la duree ou augmentez la charge")

    if reset:
        reply = await client.post("/admin/reset", json={"seed": seed, "reset_workers": True})
        reply.raise_for_status()
    before = await fetch_nodes(client)
    started = time.perf_counter()
    rows = await run_load(
        client, stream, {r.name: r.deadline_ms for r in config.requests}, max_in_flight=max_in_flight
    )
    wall_s = time.perf_counter() - started
    after = await fetch_nodes(client)

    horizon = float(stream.times_s[-1])
    summary = summarize_run(
        rows,
        slot_ids=config.slot_ids,
        profiles={p.node_id: p for p in config.pool},
        reward=config.reward,
        duration_s=horizon,
        nodes_before=before,
        nodes_after=after,
    )
    summary.update(
        configuration=configuration,
        scenario=scenario,
        base_rate_rps=rate,
        size=len(active),
        duration_wall_s=wall_s,
    )
    return summary, rows


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - necessite un orchestrateur en marche
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9000", help="URL de l'orchestrateur")
    parser.add_argument("--env", default="configs/env/eval.yaml")
    parser.add_argument("--configuration", default="n4")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--duration", type=float, default=60.0, help="duree du flux (s)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", required=True, help="CSV des resultats par requete")
    args = parser.parse_args(argv)

    config = EnvConfig.from_yaml(args.env, {"configuration": args.configuration, "scenarios": [args.scenario]})

    async def main_async() -> None:
        async with httpx.AsyncClient(base_url=args.url, timeout=30.0) as client:
            summary, rows = await run_scenario(
                client, config, args.configuration, args.scenario, args.seed, duration_s=args.duration
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        for key in ("latency_mean_ms", "latency_p95_ms", "deadline_rate", "rejection_rate", "energy_fog_j_per_req",
                    "cloud_cost_per_req", "decision_ms_p99", "fallback_rate", "loadgen_lag_ms_p99"):
            print(f"{key:>24} : {summary[key]:.4g}")
        print(f"{len(rows)} requetes -> {out}")

    asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover
    main()
