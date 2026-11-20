"""Boucle d'orchestration indépendante de la politique de routage utilisée."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from state_spec import NodeState, RequestState, encode_state

from .nodes import WorkerNode
from .policy import PolicyDecisionError, RoutingPolicy
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
    fallback_policy: RoutingPolicy,
    decision_timeout_ms: float = 20.0,
) -> dict[str, Any]:
    """Observe les nœuds, choisit le meilleur candidat, soumet puis rapporte."""
    request_id = str(request["request_id"])
    request_started = time.perf_counter()
    # Démarre le chronométrage et récupère l’état des workers avec GET /state.
    states, state_errors = await _collect_states(client, workers)
    observed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    ordered_states = []
    for name in workers:
        state = states.get(name, {})
        ordered_states.append(
            NodeState(
                node_id=name,
                queue_ms=float(state.get("queue_ms", 0.0)),
                cpu_load=float(state.get("cpu_load", 0.0)),
                net_rtt_ms=float(state.get("net_rtt_ms", 0.0)),
                battery_fraction=state.get("battery"),
                available=name in states and not state.get("depleted", False),
            )
        )
    try:
        observation, action_mask = encode_state(
            ordered_states,
            RequestState(
                request_type=str(request["type"]),
                work_ms=float(request["work_ms"]),
                deadline_ms=float(request["deadline_ms"]),
            ),
            arrival_rate_rps=float(request.get("_arrival_rate_rps", 0.0)),
        )
    except ValueError as exc:
        return {
            "request_id": request_id,
            "status": "no_worker_available",
            "error": str(exc),
            "state_errors": state_errors,
            "observed_states": states,
            "observed_at": observed_at,
        }

    decision_started = time.perf_counter()
    fallback = False
    fallback_reason = None
    try:
        ranked = policy.rank_workers(
            request,
            states,
            workers,
            observation=observation,
            action_mask=action_mask,
        )
    except PolicyDecisionError as exc:
        ranked = []
        fallback = True
        fallback_reason = f"policy_error: {exc}"
    decision_ms = (time.perf_counter() - decision_started) * 1000.0
    if decision_ms > decision_timeout_ms:
        ranked = []
        fallback = True
        fallback_reason = f"decision_timeout_ms>{decision_timeout_ms:g}"
    if not ranked:
        ranked = fallback_policy.rank_workers(
            request,
            states,
            workers,
            observation=observation,
            action_mask=action_mask,
        )
        fallback = True
        fallback_reason = fallback_reason or "no_candidate"
    # Si aucun worker n’est disponible, renvoie no_worker_available.
    if not ranked:
        return {
            "request_id": request_id,
            "status": "no_worker_available",
            "state_errors": state_errors,
            "observed_states": states,
            "observed_at": observed_at,
            "observation": observation.tolist(),
            "action_mask": action_mask.tolist(),
            "decision_ms": round(decision_ms, 3),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
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
            "observed_at": observed_at,
            "worker_result": result,
            "elapsed_ms": round(elapsed_ms, 3),
            "decision_ms": round(decision_ms, 3),
            "deadline_ms": deadline_ms,
            "deadline_met": elapsed_ms <= deadline_ms,
            "dispatch_failures": failures,
            "observation": observation.tolist(),
            "action_mask": action_mask.tolist(),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
        }
    return {
        "request_id": request_id,
        "status": "dispatch_failed",
        "dispatch_failures": failures,
        "state_errors": state_errors,
        "observed_states": states,
        "observed_at": observed_at,
        "observation": observation.tolist(),
        "action_mask": action_mask.tolist(),
        "decision_ms": round(decision_ms, 3),
        "fallback": fallback,
        "fallback_reason": fallback_reason,
    }


async def run_arrivals(
    arrivals: Iterable[Mapping[str, Any]],
    *,
    workers: Mapping[str, WorkerNode],
    client: WorkerClient,
    policy: RoutingPolicy,
    fallback_policy: RoutingPolicy,
    decision_timeout_ms: float = 20.0,
) -> list[dict[str, Any]]:
    """Émet les requêtes aux instants t_arrival_ms et les traite concurremment."""
    rows = list(arrivals)
    if not rows:
        return []
    first_arrival_ms = float(rows[0]["t_arrival_ms"])
    start = time.perf_counter()
    tasks = []
    for index, request in enumerate(rows):
        due_s = max(0.0, (float(request["t_arrival_ms"]) - first_arrival_ms) / 1000.0)
        delay_s = start + due_s - time.perf_counter()
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        recent_start = float(request["t_arrival_ms"]) - 1000.0
        recent_count = sum(
            recent_start <= float(previous["t_arrival_ms"]) <= float(request["t_arrival_ms"])
            for previous in rows[: index + 1]
        )
        request_with_rate = dict(request)
        request_with_rate["_arrival_rate_rps"] = recent_count
        tasks.append(asyncio.create_task(
            dispatch_request(
                request_with_rate,
                workers=workers,
                client=client,
                policy=policy,
                fallback_policy=fallback_policy,
                decision_timeout_ms=decision_timeout_ms,
            )
        ))
    return list(await asyncio.gather(*tasks))
