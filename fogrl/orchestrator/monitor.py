"""Sondage periodique des noeuds (toutes les 200 ms) et disjoncteur par noeud.

L'orchestrateur decide sur ces mesures, donc potentiellement perimees de
``poll_interval_ms`` : c'est la decision D5 du simulateur (``FogEnv``), reproduite ici.
La latence reseau est la duree mesuree du sondage, lissee comme dans le simulateur
(EMA, alpha = 0,3).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import httpx

from ..config import NodeProfile
from ..state_spec import NodeState

RTT_ALPHA = 0.3  # identique a ``NodeSim.RTT_ALPHA``


@dataclass
class NodeView:
    profile: NodeProfile
    queue_ms: float = 0.0
    cpu_load: float = 0.0
    rtt_ms: float = 0.0
    battery_fraction: Optional[float] = None
    reported_available: bool = False
    failures: int = 0
    last_ok: Optional[float] = None
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rtt_ms = self.profile.rtt_ms  # valeur nominale tant qu'aucun sondage n'a abouti


class NodeMonitor:
    def __init__(
        self,
        profiles: Sequence[NodeProfile],
        clients: Mapping[str, httpx.AsyncClient],
        *,
        poll_interval_s: float = 0.2,
        timeout_s: float = 0.4,
        failure_threshold: int = 2,
        stale_after_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.views = {p.node_id: NodeView(p) for p in profiles}
        self.clients = dict(clients)
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.failure_threshold = failure_threshold
        self.stale_after_s = stale_after_s
        self.clock = clock
        self._task: Optional[asyncio.Task] = None

    # ----------------------------------------------------------------- sondage

    async def poll_node(self, node_id: str) -> None:
        view = self.views[node_id]
        start = time.perf_counter()
        try:
            response = await self.clients[node_id].get("/state", timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            view.failures += 1
            return
        rtt_ms = (time.perf_counter() - start) * 1000.0
        view.rtt_ms += RTT_ALPHA * (rtt_ms - view.rtt_ms)
        view.queue_ms = float(data["queue_ms"])
        view.cpu_load = float(data["cpu_load"])
        view.battery_fraction = data.get("battery_fraction")
        view.reported_available = bool(data["available"])
        view.raw = data
        view.failures = 0
        view.last_ok = self.clock()

    async def poll_once(self) -> None:
        await asyncio.gather(*(self.poll_node(node_id) for node_id in self.views))

    async def run(self) -> None:
        while True:
            started = time.perf_counter()
            await self.poll_once()
            await asyncio.sleep(max(0.0, self.poll_interval_s - (time.perf_counter() - started)))

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="node-monitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------- etat

    def is_available(self, node_id: str) -> bool:
        view = self.views[node_id]
        if view.last_ok is None or not view.reported_available:
            return False
        if view.failures >= self.failure_threshold:
            return False
        return self.clock() - view.last_ok <= self.stale_after_s

    def mark_failure(self, node_id: str) -> None:
        """Un envoi a echoue (connexion refusee, delai depasse) : le disjoncteur se rapproche."""

        self.views[node_id].failures += 1

    def age_ms(self, node_id: str) -> Optional[float]:
        last = self.views[node_id].last_ok
        return None if last is None else (self.clock() - last) * 1000.0

    def states(self, in_flight: Mapping[str, int]) -> list[NodeState]:
        """Mesures (ordre des noeuds actifs) telles que la politique les recoit.

        Si plus aucun noeud ne parait disponible, les mesures sont tenues pour perimees et tous
        les noeuds sont proposes (comme ``FogEnv._observe``) : l'envoi dira s'ils repondent.
        """

        states = [
            NodeState(
                node_id=node_id,
                queue_ms=view.queue_ms,
                cpu_load=view.cpu_load,
                net_rtt_ms=view.rtt_ms,
                battery_fraction=view.battery_fraction,
                available=self.is_available(node_id),
                in_flight=in_flight.get(node_id, 0),
            )
            for node_id, view in self.views.items()
        ]
        if not any(s.available for s in states):
            states = [NodeState(**{**s.__dict__, "available": True}) for s in states]
        return states

    def reset(self) -> None:
        for view in self.views.values():
            view.rtt_ms = view.profile.rtt_ms
            view.failures = 0
