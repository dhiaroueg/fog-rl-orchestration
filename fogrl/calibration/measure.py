"""Mesure des durees de service par noeud, type de requete et niveau de charge.

Pour chaque worker, chaque type de requete (travail nominal de ``configs/requests.yaml``) et chaque
niveau de concurrence c = 1 .. servers + 1, c clients envoient des requetes en boucle fermee. Le
worker rapporte la charge a l'arrivee (fraction de serveurs deja occupes : l'abscisse du modele de
service du simulateur) et la duree de service. La latence reseau est mesuree par des sondages /state.

    python -m fogrl.calibration.measure --plan configs/calibration/plan.yaml --out data/calibration/2026-11-20

Resultat : raw.csv (une ligne par requete), rtt.csv, meta.json. ``fogrl.calibration.fit`` les reduit
en ``calibration.json`` (lu par ``service_model: calibrated``) et propose des valeurs de profil.
Le modele d'energie n'est PAS mesurable sur des VM : il reste une hypothese (modele lineaire).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import httpx
import pandas as pd

from ..config import load_requests, load_yaml
from ..utils import git_commit


@dataclass(frozen=True)
class CalibrationPlan:
    workers: Mapping[str, str]  # node_id -> URL directe du worker
    requests_file: str = "configs/requests.yaml"
    req_types: Sequence[str] = ("light", "medium", "heavy")
    levels: Union[str, Sequence[int]] = "auto"  # "auto" : 1 .. servers + 1
    samples_per_level: int = 30
    warmup_requests: int = 5
    rtt_samples: int = 100
    timeout_s: float = 60.0
    extra: Mapping = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path) -> "CalibrationPlan":
        data = dict(load_yaml(path))
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Cles de plan inconnues : {sorted(unknown)}")
        return cls(**data)

    def levels_for(self, servers: int) -> list[int]:
        if self.levels == "auto":
            return list(range(1, servers + 2))
        return [int(c) for c in self.levels]


async def closed_loop(
    client: httpx.AsyncClient, req_type: str, work_ms: float, concurrency: int, n_requests: int, timeout_s: float
) -> list[dict]:
    """``concurrency`` clients envoient des requetes a la suite jusqu'a ``n_requests`` au total."""

    remaining = n_requests
    rows: list[dict] = []

    async def client_loop() -> None:
        nonlocal remaining
        while remaining > 0:
            remaining -= 1
            started = time.perf_counter()
            reply = await client.post("/process", json={"type": req_type, "work_ms": work_ms}, timeout=timeout_s)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if reply.status_code != 200:
                continue  # rejet (file pleine, noeud eteint) : pas de mesure de service
            data = reply.json()
            rows.append(
                {
                    "req_type": req_type,
                    "work_ms": work_ms,
                    "concurrency": concurrency,
                    "load": data["load"],
                    "service_ms": data["service_ms"],
                    "wait_ms": data["wait_ms"],
                    "latency_ms": latency_ms,
                    "energy_j": data["energy_j"],
                }
            )

    await asyncio.gather(*(client_loop() for _ in range(concurrency)))
    return rows


async def measure_rtt(client: httpx.AsyncClient, n: int) -> list[float]:
    samples = []
    for _ in range(n):
        started = time.perf_counter()
        (await client.get("/state")).raise_for_status()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


async def run_calibration(
    plan: CalibrationPlan,
    clients: Optional[Mapping[str, httpx.AsyncClient]] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nominal = {r.name: r.work_ms for r in load_requests(plan.requests_file)}
    unknown = [t for t in plan.req_types if t not in nominal]
    if unknown:
        raise ValueError(f"Types de requete inconnus : {unknown}")
    owned = clients is None
    if clients is None:
        clients = {node: httpx.AsyncClient(base_url=url, timeout=plan.timeout_s) for node, url in plan.workers.items()}

    raw_rows: list[dict] = []
    rtt_rows: list[dict] = []
    infos: dict[str, dict] = {}
    try:
        for node_id, client in clients.items():
            info = (await client.get("/info")).json()
            infos[node_id] = info
            servers = int(info["servers"])
            for req_type in plan.req_types:
                for _ in range(plan.warmup_requests):  # echauffe processus, caches et frequences CPU
                    await client.post("/process", json={"type": req_type, "work_ms": nominal[req_type]})
                for level in plan.levels_for(servers):
                    rows = await closed_loop(
                        client, req_type, nominal[req_type], level, plan.samples_per_level, plan.timeout_s
                    )
                    raw_rows += [{"node": node_id, **row} for row in rows]
                    if verbose and rows:
                        mean_service = sum(r["service_ms"] for r in rows) / len(rows)
                        print(f"{node_id:<6} {req_type:<7} c={level:<2} service moyen {mean_service:8.1f} ms ({len(rows)} mesures)", flush=True)
            rtt_rows += [{"node": node_id, "rtt_ms": v} for v in await measure_rtt(client, plan.rtt_samples)]
    finally:
        if owned:
            await asyncio.gather(*(c.aclose() for c in clients.values()))

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "plan": {k: v for k, v in plan.__dict__.items() if k != "extra"},
        "nominal_work_ms": nominal,
        "workers": infos,
    }
    return pd.DataFrame(raw_rows), pd.DataFrame(rtt_rows), meta


def save_measurements(raw: pd.DataFrame, rtt: pd.DataFrame, meta: dict, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw.csv", index=False)
    rtt.to_csv(out / "rtt.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - necessite des workers en marche
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="configs/calibration/plan.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    plan = CalibrationPlan.from_yaml(args.plan)
    raw, rtt, meta = asyncio.run(run_calibration(plan))
    out = save_measurements(raw, rtt, meta, args.out)
    print(f"{len(raw)} mesures -> {out}/raw.csv ; ensuite : python -m fogrl.calibration.fit --data {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
