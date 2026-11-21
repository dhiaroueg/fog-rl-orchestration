"""Coeur de l'orchestrateur : mesure -> decision -> envoi -> journal.

Une requete suit toujours le meme chemin, que la politique soit une baseline ou un agent RL
(EF-08) : ``NodeMonitor.states`` -> ``DecisionEngine.decide`` -> POST /process du noeud choisi.
La latence rapportee est la duree entre l'envoi au noeud et sa reponse (reseau + attente +
service), la meme grandeur que ``latency_ms`` du simulateur ; le temps de decision est
rapporte a part.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from typing import Callable, Mapping, Optional

import httpx

from ..policies.base import Policy
from ..policies.factory import is_rl_spec, make_policy
from ..state_spec import DecisionContext, RequestState, check_contract
from .decision import DecisionEngine, NoNodeAvailable, warm_up_policy
from .decision_log import DecisionLog
from .metrics import RouterMetrics
from .monitor import NodeMonitor
from .settings import ResolvedSettings


class RateWindow:
    """Debit recent des requetes sur ``window_s`` (meme formule que ``FogEnv._recent_rate``)."""

    def __init__(self, window_s: float, t0: float) -> None:
        self.window_s = window_s
        self.t0 = t0
        self._times: deque[float] = deque()

    def reset(self, t0: float) -> None:
        self.t0 = t0
        self._times.clear()

    def _trim(self, now: float) -> None:
        while self._times and self._times[0] <= now - self.window_s:
            self._times.popleft()

    def record(self, now: float) -> float:
        self._times.append(now)
        return self.current(now)

    def current(self, now: float) -> float:
        self._trim(now)
        return len(self._times) / max(min(self.window_s, now - self.t0), 0.1)


class Router:
    def __init__(
        self,
        resolved: ResolvedSettings,
        transports: Optional[Mapping[str, httpx.AsyncBaseTransport]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.resolved = resolved
        self.settings = resolved.settings
        self.clock = clock
        timeout = httpx.Timeout(self.settings.request_timeout_s, connect=2.0)
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        self.clients: dict[str, httpx.AsyncClient] = {}
        for node_id in resolved.active_ids:
            kwargs = {"transport": transports[node_id]} if transports and node_id in transports else {}
            self.clients[node_id] = httpx.AsyncClient(
                base_url=resolved.urls[node_id], timeout=timeout, limits=limits, **kwargs
            )
        self.monitor = NodeMonitor(
            [resolved.profiles[i] for i in resolved.active_ids],
            self.clients,
            poll_interval_s=resolved.poll_interval_s,
            timeout_s=self.settings.poll_timeout_ms / 1000.0,
            failure_threshold=self.settings.failure_threshold,
            stale_after_s=resolved.stale_after_s,
            clock=clock,
        )
        self.in_flight: Counter = Counter()
        self.rate = RateWindow(resolved.rate_window_s, clock())
        self.t0 = clock()
        self.log = DecisionLog(self.settings.log_file)
        self.engine: DecisionEngine  # cree dans start()
        self.metrics = RouterMetrics(self)
        self._started = False

    # ------------------------------------------------------------- cycle de vie

    async def start(self) -> None:
        await self.monitor.poll_once()
        policy, slot_ids = await self._build_policy(self.settings.policy, None)
        self.engine = DecisionEngine(
            policy,
            slot_ids,
            self.resolved.normalization,
            budget_ms=self.settings.decision_budget_ms,
            failure_threshold=self.settings.policy_failure_threshold,
            cooldown_s=self.settings.policy_cooldown_s,
            clock=self.clock,
        )
        self.engine.warm_up(self.monitor.states(self.in_flight), self.settings.warmup_decisions)
        self.monitor.start()
        self._started = True

    async def stop(self) -> None:
        await self.monitor.stop()
        if self._started:
            self.engine.close()
        self.log.close()
        await asyncio.gather(*(client.aclose() for client in self.clients.values()))

    # ----------------------------------------------------------------- politique

    async def _build_policy(self, spec: str, name: Optional[str]) -> tuple[Policy, tuple[str, ...]]:
        """Charge la politique (hors de la boucle d'evenements : torch met ~1 s) et verifie son contrat."""

        profiles, resolved = self.resolved.profiles, self.resolved

        def build() -> tuple[Policy, tuple[str, ...]]:
            policy = make_policy(spec, profiles, None, None, name)
            if is_rl_spec(spec):
                contract = policy.metadata["contract"]
                check_contract(contract, policy.slot_ids, resolved.normalization)
                missing = [i for i in resolved.active_ids if i not in policy.slot_ids]
                if missing:
                    raise ValueError(
                        f"Le modele couvre {list(policy.slot_ids)} mais les noeuds actifs incluent {missing}"
                    )
                return policy, tuple(policy.slot_ids)
            return policy, tuple(resolved.active_ids)

        return await asyncio.get_running_loop().run_in_executor(None, build)

    async def set_policy(self, spec: str, name: Optional[str] = None) -> dict:
        """Remplace la politique a chaud (EF-08). Leve ValueError/FileNotFoundError si le chargement echoue."""

        policy, slot_ids = await self._build_policy(spec, name)
        states = self.monitor.states(self.in_flight)
        await asyncio.get_running_loop().run_in_executor(
            None,
            warm_up_policy,
            policy,
            slot_ids,
            self.resolved.normalization,
            states,
            self.settings.warmup_decisions,
        )
        self.engine.swap(policy, slot_ids)
        return self.policy_info()

    def policy_info(self) -> dict:
        policy = self.engine.policy
        return {
            "name": policy.name,
            "kind": "rl" if policy.needs_budget else "baseline",
            "slot_ids": list(self.engine.slot_ids),
            "decision_budget_ms": self.settings.decision_budget_ms,
            "fault": self._fault_info(),
        }

    def _fault_info(self) -> Optional[dict]:
        fault = self.engine.fault
        if fault is None:
            return None
        return {"mode": fault[0], "delay_ms": fault[1] * 1000.0}

    def set_policy_fault(self, mode: str, delay_ms: float = 0.0) -> dict:
        """Injecte ou leve une panne de l'agent (EF-07). Leve ValueError si le mode est inconnu."""

        self.engine.set_fault(mode, delay_ms)
        return self.policy_info()

    # ------------------------------------------------------------------- routage

    async def route(
        self,
        req_type: str,
        work_ms: float,
        deadline_ms: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> dict:
        arrived = self.clock()
        deadline_ms = deadline_ms if deadline_ms is not None else self.resolved.deadlines[req_type]
        request_id = request_id or f"{int(arrived * 1e6):x}-{id(object()) & 0xFFFF:04x}"
        rate = self.rate.record(arrived)
        states = self.monitor.states(self.in_flight)
        request = RequestState(req_type, work_ms, deadline_ms)
        ctx = DecisionContext(now_s=arrived - self.t0, arrival_rate_rps=rate)
        policy_name = self.engine.policy.name

        result = {
            "request_id": request_id,
            "policy": policy_name,
            "node": None,
            "rejected": True,
            "reason": "",
            "latency_ms": None,
            "deadline_ms": deadline_ms,
            "deadline_met": False,
            "decision_ms": 0.0,
            "fallback": False,
            "fallback_reason": "",
            "wait_ms": None,
            "service_ms": None,
            "energy_j": 0.0,
            "cost": 0.0,
        }
        try:
            decision = await self.engine.decide(states, request, ctx)
        except NoNodeAvailable:
            result["reason"] = "no_node_available"
            self._record(result, req_type, work_ms, rate, states)
            return result
        result.update(
            node=decision.node_id,
            decision_ms=decision.decision_ms,
            fallback=decision.fallback,
            fallback_reason=decision.reason,
        )

        client = self.clients[decision.node_id]
        self.in_flight[decision.node_id] += 1
        sent = time.perf_counter()
        try:
            response = await client.post(
                "/process", json={"request_id": request_id, "type": req_type, "work_ms": work_ms}
            )
            latency_ms = (time.perf_counter() - sent) * 1000.0
            if response.status_code == 200:
                data = response.json()
                result.update(
                    rejected=False,
                    latency_ms=latency_ms,
                    deadline_met=latency_ms <= deadline_ms,
                    wait_ms=data["wait_ms"],
                    service_ms=data["service_ms"],
                    energy_j=data["energy_j"],
                    cost=data["cost"],
                )
            else:
                result["reason"] = _reason(response)
        except httpx.TimeoutException:
            result["reason"] = "timeout"
            self.monitor.mark_failure(decision.node_id)
        except httpx.TransportError:
            result["reason"] = "unreachable"
            self.monitor.mark_failure(decision.node_id)
        finally:
            self.in_flight[decision.node_id] -= 1

        self._record(result, req_type, work_ms, rate, states)
        return result

    def _record(self, result: dict, req_type: str, work_ms: float, rate: float, states) -> None:
        self.metrics.observe(result["policy"], result["node"] or "none", req_type, result)
        if self.log.enabled:
            self.log.write(
                {
                    "ts": time.time(),
                    "type": req_type,
                    "work_ms": work_ms,
                    "rate_rps": rate,
                    "state": {
                        s.node_id: [s.queue_ms, s.cpu_load, s.net_rtt_ms, s.battery_fraction, s.available, s.in_flight]
                        for s in states
                    },
                    **result,
                }
            )

    # ---------------------------------------------------------------------- admin

    def view(self) -> dict:
        now = self.clock()
        return {
            "policy": self.policy_info(),
            "arrival_rate_rps": self.rate.current(now),
            "nodes": [
                {
                    "node_id": node_id,
                    "available": self.monitor.is_available(node_id),
                    "queue_ms": view.queue_ms,
                    "cpu_load": view.cpu_load,
                    "rtt_ms": view.rtt_ms,
                    "battery_fraction": view.battery_fraction,
                    "in_flight": self.in_flight.get(node_id, 0),
                    "age_ms": self.monitor.age_ms(node_id),
                    "failures": view.failures,
                }
                for node_id, view in self.monitor.views.items()
            ],
            "decisions": self.engine.decisions,
            "fallbacks": dict(self.engine.fallbacks),
        }

    async def fresh_nodes(self) -> dict:
        """Etat brut des workers apres un sondage immediat (compteurs d'energie des campagnes)."""

        await self.monitor.poll_once()
        return {node_id: view.raw for node_id, view in self.monitor.views.items()}

    async def reset(self, seed: Optional[int] = None, reset_workers: bool = True) -> None:
        """Remet a zero la politique, les mesures, le debit et (optionnellement) les workers."""

        self.engine.reset(seed)
        self.monitor.reset()
        self.rate.reset(self.clock())
        self.t0 = self.clock()
        if reset_workers:
            await asyncio.gather(*(c.post("/admin/reset") for c in self.clients.values()))
        await self.monitor.poll_once()

    async def set_node_failure(self, node_id: str, failed: bool) -> dict:
        if node_id not in self.clients:
            raise KeyError(node_id)
        reply = await self.clients[node_id].post("/admin/fail" if failed else "/admin/recover")
        reply.raise_for_status()
        await self.monitor.poll_node(node_id)
        return reply.json()


def _reason(response: httpx.Response) -> str:
    try:
        return str(response.json().get("reason") or f"http_{response.status_code}")
    except ValueError:
        return f"http_{response.status_code}"
