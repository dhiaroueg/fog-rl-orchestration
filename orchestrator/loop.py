"""Boucle d'orchestration indépendante de la politique de routage utilisée."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable, Mapping

from .nodes import WorkerNode
from .policy import RoutingPolicy
from .worker_client import WorkerClient, WorkerRequestError


async def _collect_states(
    client: WorkerClient,
    workers: Mapping[str, WorkerNode],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    names = list(workers)
    results = await asyncio.gather(
        *(client.get_state(workers[name]) for name in names), return_exceptions=True
    )
    states: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            errors[name] = str(result)
        else:
            states[name] = result
    return states, errors

# dispatch_request() : traite une requête
async def dispatch_request(
    request: Mapping[str, Any],
    *,
    workers: Mapping[str, WorkerNode],
    client: WorkerClient,
    policy: RoutingPolicy,
) -> dict[str, Any]:
    """Observe les nœuds, choisit le meilleur candidat, soumet puis rapporte."""
    request_id = str(request["request_id"])
    request_started = time.perf_counter()
    # Démarre le chronométrage et récupère l’état des workers avec GET /state.
    states, state_errors = await _collect_states(client, workers)
    # Demande à la politique de classer les workers. Aujourd’hui, GreedyPolicy.rank_workers() fait ce classement.
    ranked = policy.rank_workers(request, states, workers)
    # Si aucun worker n’est disponible, renvoie no_worker_available.
    if not ranked:
        return {
            "request_id": request_id,
            "status": "no_worker_available",
            "state_errors": state_errors,
            "observed_states": states,
        }
# Sinon, essaie les workers dans l’ordre fourni par la politique en envoyant type, size et request_id à POST /process.
# Si un worker refuse la requête ou est inaccessible, essaie le candidat suivant.
    failures: dict[str, str] = {}
    for name in ranked:
        node = workers[name]
        try:
            result = await client.process(
                node,
                request_id=request_id,
                task_type=str(request["type"]),
                size=float(request["size"]),
            )
        except WorkerRequestError as exc:
            failures[name] = str(exc)
            continue
        elapsed_ms = (time.perf_counter() - request_started) * 1000.0
        deadline_ms = float(request["deadline_ms"])
        return {
            "request_id": request_id,
            "status": "completed",
            "selected_node": name,
            "observed_states": states,
            "worker_result": result,
            "elapsed_ms": round(elapsed_ms, 3),
            "deadline_ms": deadline_ms,
            "deadline_met": elapsed_ms <= deadline_ms,
            "dispatch_failures": failures,
        }
    return {
        "request_id": request_id,
        "status": "dispatch_failed",
        "dispatch_failures": failures,
        "state_errors": state_errors,
        "observed_states": states,
    }


async def run_arrivals(
    arrivals: Iterable[Mapping[str, Any]],
    *,
    workers: Mapping[str, WorkerNode],
    client: WorkerClient,
    policy: RoutingPolicy,
) -> list[dict[str, Any]]:
    """Émet les requêtes aux instants t_arrival_ms et les traite concurremment."""
    rows = list(arrivals)
    if not rows:
        return []
    first_arrival_ms = float(rows[0]["t_arrival_ms"])
    start = time.perf_counter()
    tasks = []
    for request in rows:
        due_s = max(0.0, (float(request["t_arrival_ms"]) - first_arrival_ms) / 1000.0)
        delay_s = start + due_s - time.perf_counter()
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        tasks.append(asyncio.create_task(
            dispatch_request(request, workers=workers, client=client, policy=policy)
        ))
    return list(await asyncio.gather(*tasks))
