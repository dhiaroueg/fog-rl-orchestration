"""Worker FastAPI : execute les requetes d'un noeud et expose son etat.

Endpoints :
    POST /process      execute une requete (charge ``burn`` ou reelle), renvoie les durees
    GET  /state        mesures utilisees par l'orchestrateur (file, CPU, batterie, RTT sonde)
    GET  /metrics      exposition Prometheus (file, CPU, batterie, energie, durees de service)
    GET  /health, /info
    POST /admin/fail | /admin/recover | /admin/reset      injection de panne (EF-09) et remise a zero

Configuration par variables d'environnement (``WorkerSettings.from_env``) :
    NODE_ID (obligatoire), NODES_FILE, WORKER_MODE=burn|real, WORKER_EXECUTOR=process|thread,
    EMULATE_SPEED=0|1, ONNX_MODEL, SIMULATED_RTT_MS (0 si tc netem injecte la latence), PORT.

L'energie et la batterie sont VIRTUELLES (modele lineaire, voir ``sim.energy``) :
les VM ne fournissent aucune mesure physique.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from ..config import NodeProfile, load_nodes
from ..sim.service import lognormal_factor
from .runtime import NodeRuntime
from .workloads import WorkloadExecutor

SERVICE_BUCKETS = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


@dataclass(frozen=True)
class WorkerSettings:
    node_id: str
    nodes_file: str = "configs/nodes/pool6.yaml"
    mode: str = "burn"
    executor: str = "process"
    emulate_speed: bool = False
    onnx_model: Optional[str] = None
    simulated_rtt_ms: float = 0.0
    host: str = "0.0.0.0"
    port: Optional[int] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "WorkerSettings":
        env = os.environ if env is None else env
        if not env.get("NODE_ID"):
            raise ValueError("Variable d'environnement NODE_ID obligatoire (ex. fogA)")
        return cls(
            node_id=env["NODE_ID"],
            nodes_file=env.get("NODES_FILE", cls.nodes_file),
            mode=env.get("WORKER_MODE", cls.mode),
            executor=env.get("WORKER_EXECUTOR", cls.executor),
            emulate_speed=env.get("EMULATE_SPEED", "0").lower() in ("1", "true", "yes"),
            onnx_model=env.get("ONNX_MODEL") or None,
            simulated_rtt_ms=float(env.get("SIMULATED_RTT_MS", "0")),
            host=env.get("HOST", cls.host),
            port=int(env["PORT"]) if env.get("PORT") else None,
        )

    def load_profile(self) -> NodeProfile:
        profiles, _ = load_nodes(self.nodes_file)
        by_id = {p.node_id: p for p in profiles}
        if self.node_id not in by_id:
            raise ValueError(f"Noeud {self.node_id!r} absent de {self.nodes_file} (disponibles : {sorted(by_id)})")
        return by_id[self.node_id]


class ProcessRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    type: Literal["light", "medium", "heavy"]
    work_ms: float = Field(gt=0, le=60_000)


class _NodeCollector:
    """Lit l'etat du noeud au moment du scrape : les jauges ne sont jamais perimees."""

    def __init__(self, runtime: NodeRuntime, clock: Callable[[], float]) -> None:
        self.runtime = runtime
        self.clock = clock

    def collect(self):
        snap = self.runtime.snapshot(self.clock())
        node = snap["node_id"]
        for name, doc, value in (
            ("fog_node_queue_ms", "Attente estimee d'une nouvelle requete (ms)", snap["queue_ms"]),
            ("fog_node_cpu_load", "Charge CPU lissee (fraction de serveurs occupes)", snap["cpu_load"]),
            ("fog_node_in_flight", "Requetes en cours (file + service)", snap["in_flight"]),
            ("fog_node_up", "1 si le noeud accepte des requetes", 1.0 if snap["available"] else 0.0),
        ):
            gauge = GaugeMetricFamily(name, doc, labels=["node"])
            gauge.add_metric([node], value)
            yield gauge
        if snap["battery_fraction"] is not None:
            gauge = GaugeMetricFamily("fog_node_battery_fraction", "Batterie virtuelle (0-1)", labels=["node"])
            gauge.add_metric([node], snap["battery_fraction"])
            yield gauge
        energy = CounterMetricFamily(
            "fog_node_energy_joules", "Energie virtuelle cumulee (modele lineaire)", labels=["node", "kind"]
        )
        energy.add_metric([node, "dynamic"], snap["energy_dynamic_j"])
        energy.add_metric([node, "idle"], snap["energy_idle_j"])
        yield energy


def create_app(
    settings: WorkerSettings,
    profile: Optional[NodeProfile] = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    profile = profile or settings.load_profile()
    runtime = NodeRuntime(
        profile, now=clock(), initial_speed_hint=profile.speed if settings.emulate_speed else 1.0
    )
    registry = CollectorRegistry()
    registry.register(_NodeCollector(runtime, clock))
    requests_total = Counter(
        "fog_worker_requests_total", "Requetes recues par le noeud", ["node", "type", "outcome"], registry=registry
    )
    service_seconds = Histogram(
        "fog_worker_service_seconds", "Duree de service", ["node", "type"], buckets=SERVICE_BUCKETS, registry=registry
    )
    wait_seconds = Histogram(
        "fog_worker_wait_seconds", "Attente avant service", ["node"], buckets=SERVICE_BUCKETS, registry=registry
    )
    rng = np.random.default_rng()
    holder: dict = {"executor": None, "semaphore": None}

    def get_executor() -> WorkloadExecutor:
        if holder["executor"] is None:
            holder["executor"] = WorkloadExecutor(
                profile.servers,
                mode=settings.mode,
                speed=profile.speed,
                emulate_speed=settings.emulate_speed,
                onnx_path=settings.onnx_model,
                kind=settings.executor,
            )
        return holder["executor"]

    def get_semaphore() -> asyncio.Semaphore:
        if holder["semaphore"] is None:
            holder["semaphore"] = asyncio.Semaphore(profile.servers)
        return holder["semaphore"]

    async def network_delay() -> None:
        """Demi-RTT injecte cote application (0 si ``tc netem`` s'en charge)."""

        if settings.simulated_rtt_ms > 0:
            rtt = settings.simulated_rtt_ms * lognormal_factor(rng, profile.rtt_sigma)
            await asyncio.sleep(rtt / 2000.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = get_executor()
        await asyncio.get_running_loop().run_in_executor(None, executor.warm_up)
        runtime.reset(clock())
        yield
        executor.close()

    app = FastAPI(title=f"fog worker {profile.node_id}", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.profile = profile
    app.state.registry = registry

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "node": profile.node_id}

    @app.get("/info")
    async def info() -> dict:
        return {
            "node_id": profile.node_id,
            "kind": profile.kind,
            "servers": profile.servers,
            "speed": profile.speed,
            "mode": settings.mode,
            "executor": settings.executor,
            "emulate_speed": settings.emulate_speed,
            "simulated_rtt_ms": settings.simulated_rtt_ms,
        }

    @app.get("/state")
    async def state() -> dict:
        await network_delay()
        snap = runtime.snapshot(clock())
        await network_delay()
        return snap

    @app.post("/process")
    async def process(req: ProcessRequest):
        await network_delay()
        arrived = clock()
        # Charge a l'arrivee (serveurs deja occupes) : meme grandeur que ``busy_fraction(arrive)``
        # du simulateur, abscisse de la table de calibration.
        load = runtime.busy_fraction()
        reason = runtime.admit(arrived, req.request_id, req.type, req.work_ms)
        if reason is not None:
            requests_total.labels(profile.node_id, req.type, reason).inc()
            await network_delay()
            return JSONResponse(
                status_code=503 if reason == "unavailable" else 429,
                content={"node": profile.node_id, "request_id": req.request_id, "reason": reason},
            )

        done = False
        try:
            async with get_semaphore():
                started = clock()
                runtime.start(started, req.request_id)
                begin = time.perf_counter()
                await asyncio.wrap_future(get_executor().submit(req.type, req.work_ms))
                service_ms = (time.perf_counter() - begin) * 1000.0
            completion = runtime.finish(clock(), req.request_id, req.work_ms, service_ms)
            done = True
        finally:
            if not done:
                runtime.abort(clock(), req.request_id)

        wait_ms = (started - arrived) * 1000.0
        requests_total.labels(profile.node_id, req.type, "ok").inc()
        service_seconds.labels(profile.node_id, req.type).observe(service_ms / 1000.0)
        wait_seconds.labels(profile.node_id).observe(wait_ms / 1000.0)
        await network_delay()
        return {
            "node": profile.node_id,
            "request_id": req.request_id,
            "type": req.type,
            "work_ms": req.work_ms,
            "wait_ms": wait_ms,
            "service_ms": service_ms,
            "load": load,
            "energy_j": completion.energy_j,
            "cost": completion.cost,
            "battery_fraction": completion.battery_fraction,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/fail")
    async def fail() -> dict:
        runtime.advance(clock())
        runtime.down = True
        return {"node": profile.node_id, "available": runtime.up}

    @app.post("/admin/recover")
    async def recover() -> dict:
        runtime.advance(clock())
        runtime.down = False
        return {"node": profile.node_id, "available": runtime.up}

    @app.post("/admin/reset")
    async def reset() -> dict:
        runtime.reset(clock())
        return runtime.snapshot(clock())

    return app


def main() -> None:  # pragma: no cover - lance un serveur
    import uvicorn

    settings = WorkerSettings.from_env()
    profile = settings.load_profile()
    uvicorn.run(
        create_app(settings, profile),
        host=settings.host,
        port=settings.port or profile.port,
        log_level=os.environ.get("LOG_LEVEL", "warning"),
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
