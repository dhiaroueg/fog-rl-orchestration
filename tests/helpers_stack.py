"""Aide aux tests : pile complete (workers + orchestrateur) en memoire, sans reseau."""

from contextlib import AsyncExitStack, asynccontextmanager

import httpx

from fogrl.orchestrator.app import create_app as create_orchestrator
from fogrl.orchestrator.settings import OrchestratorSettings
from fogrl.worker.app import WorkerSettings
from fogrl.worker.app import create_app as create_worker


class FlakyTransport(httpx.AsyncBaseTransport):
    """Transport qui peut simuler un noeud injoignable ou trop lent."""

    def __init__(self, inner):
        self.inner = inner
        self.mode = None  # None | "refuse" | "timeout"

    async def handle_async_request(self, request):
        if self.mode == "refuse":
            raise httpx.ConnectError("connexion refusee", request=request)
        if self.mode == "timeout":
            raise httpx.ReadTimeout("delai depasse", request=request)
        return await self.inner.handle_async_request(request)


@asynccontextmanager
async def stack(policy="greedy", configuration="n4", env_file="configs/env/eval.yaml", **overrides):
    settings = OrchestratorSettings(env_file=env_file, configuration=configuration, policy=policy, **overrides)
    resolved = settings.resolve()
    async with AsyncExitStack() as exit_stack:
        transports = {}
        for node_id in resolved.active_ids:
            worker = create_worker(
                WorkerSettings(node_id=node_id, executor="thread", emulate_speed=True), resolved.profiles[node_id]
            )
            await exit_stack.enter_async_context(worker.router.lifespan_context(worker))
            transports[node_id] = FlakyTransport(httpx.ASGITransport(app=worker))
        orchestrator = create_orchestrator(settings, transports=transports)
        await exit_stack.enter_async_context(orchestrator.router.lifespan_context(orchestrator))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=orchestrator), base_url="http://orchestrator"
        ) as client:
            client.router, client.transports, client.resolved = orchestrator.state.router, transports, resolved
            yield client


def route_body(req_type="light", work_ms=5.0, **extra):
    return {"type": req_type, "work_ms": work_ms, **extra}
