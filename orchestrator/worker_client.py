"""Client HTTP asynchrone pour les API exposées par nodes/worker.py."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .nodes import WorkerNode


class WorkerRequestError(RuntimeError):
    """Erreur de communication avec un worker, avec son nom et son endpoint."""

    def __init__(self, node: WorkerNode, endpoint: str, detail: str) -> None:
        self.node = node
        self.endpoint = endpoint
        super().__init__(f"{node.name} ({node.url}{endpoint}): {detail}")


class WorkerClient:
    def __init__(self, timeout_s: float = 5.0) -> None:
        self._http = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_state(self, node: WorkerNode) -> dict[str, Any]:
        """Appelle GET /state et retourne l'état JSON du worker."""
        started = time.perf_counter()
        state = await self._request_json(node, "GET", "/state")
        state["net_rtt_ms"] = (time.perf_counter() - started) * 1000.0
        return state

    async def process(
        self,
        node: WorkerNode,
        *,
        request_id: str,
        task_type: str,
        size: float,
    ) -> dict[str, Any]:
        """Soumet une tâche au format attendu par Req dans nodes/worker.py."""
        if not request_id:
            raise ValueError("request_id ne doit pas être vide")
        if task_type not in {"light", "medium", "heavy"}:
            raise ValueError(f"type de tâche inconnu: {task_type!r}")
        if size <= 0:
            raise ValueError("size doit être strictement positif")
        payload = {"request_id": request_id, "type": task_type, "size": size}
        return await self._request_json(node, "POST", "/process", json=payload)

    async def _request_json(
        self,
        node: WorkerNode,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = await self._http.request(method, f"{node.url}{endpoint}", **kwargs)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = f"HTTP {exc.response.status_code}: {exc.response.text}"
            raise WorkerRequestError(node, endpoint, detail) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise WorkerRequestError(node, endpoint, str(exc)) from exc
        if not isinstance(data, dict):
            raise WorkerRequestError(node, endpoint, "la réponse JSON doit être un objet")
        return data
