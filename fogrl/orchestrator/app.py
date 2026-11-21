"""API HTTP de l'orchestrateur.

    POST /route                    route une requete {type, work_ms[, deadline_ms, request_id]}
    GET  /state                    vue des noeuds, politique active, debit, repli
    GET  /nodes?fresh=true         etat brut des workers (apres sondage immediat)
    GET|POST /policy               politique active / remplacement a chaud ({"spec": "greedy" | "rl:<dossier>"})
    GET  /metrics, /health
    POST /admin/reset              remise a zero (politique, mesures, workers)
    POST /admin/nodes/{id}/fail|recover     injection de panne d'un noeud (EF-09)

Lancement : ``python -m fogrl.orchestrator.app`` (configuration : ORCH_CONFIG, ORCH_POLICY, ...).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Callable, Literal, Mapping, Optional

import httpx
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from .router import Router
from .settings import OrchestratorSettings


class RouteRequest(BaseModel):
    request_id: Optional[str] = None
    type: Literal["light", "medium", "heavy"]
    work_ms: float = Field(gt=0, le=60_000)
    deadline_ms: Optional[float] = Field(default=None, gt=0)


class PolicyRequest(BaseModel):
    spec: str
    name: Optional[str] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    reset_workers: bool = True


class PolicyFaultRequest(BaseModel):
    mode: Literal["none", "error", "slow"]
    delay_ms: float = Field(default=50.0, ge=0)


def create_app(
    settings: Optional[OrchestratorSettings] = None,
    transports: Optional[Mapping[str, httpx.AsyncBaseTransport]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    settings = settings or OrchestratorSettings.from_env()
    router = Router(settings.resolve(), transports=transports, clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await router.start()
        yield
        await router.stop()

    app = FastAPI(title="fog orchestrator", lifespan=lifespan)
    app.state.router = router

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "policy": router.engine.policy.name}

    @app.post("/route")
    async def route(req: RouteRequest) -> dict:
        return await router.route(req.type, req.work_ms, req.deadline_ms, req.request_id)

    @app.get("/state")
    async def state() -> dict:
        return router.view()

    @app.get("/nodes")
    async def nodes(fresh: bool = False) -> dict:
        if fresh:
            return await router.fresh_nodes()
        return {node_id: view.raw for node_id, view in router.monitor.views.items()}

    @app.get("/policy")
    async def get_policy() -> dict:
        return router.policy_info()

    @app.post("/policy")
    async def set_policy(req: PolicyRequest) -> dict:
        try:
            return await router.set_policy(req.spec, req.name)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(router.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reset")
    async def reset(req: ResetRequest = ResetRequest()) -> dict:
        await router.reset(req.seed, req.reset_workers)
        return router.view()

    @app.post("/admin/policy-fault")
    async def policy_fault(req: PolicyFaultRequest) -> dict:
        """Test EF-07 : simule la panne de l'agent (error), sa lenteur (slow) ou la leve (none)."""

        try:
            return router.set_policy_fault(req.mode, req.delay_ms)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/nodes/{node_id}/{action}")
    async def node_failure(node_id: str, action: Literal["fail", "recover"]) -> dict:
        try:
            return await router.set_node_failure(node_id, action == "fail")
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Noeud inconnu : {node_id}") from None

    return app


def main() -> None:  # pragma: no cover - lance un serveur
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "9000")),
        log_level=os.environ.get("LOG_LEVEL", "warning"),
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
