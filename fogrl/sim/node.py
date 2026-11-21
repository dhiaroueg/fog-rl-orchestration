"""Noeud simule : file FIFO a ``servers`` serveurs, energie et batterie virtuelle.

Les temps de service sont tires au moment ou la requete est affectee : la fin
de chaque travail est alors connue exactement, ce qui permet une simulation
sans file d'evenements (la file FIFO se resume aux dates de liberation des
serveurs).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import NodeProfile
from .energy import BatteryModel, dynamic_energy_j
from .service import ServiceModel, lognormal_factor


@dataclass
class SubmitResult:
    accepted: bool
    reason: str = ""  # "", "unavailable", "queue_full"
    latency_ms: float = 0.0  # temps de reponse bout en bout (reseau + attente + service)
    wait_ms: float = 0.0
    service_ms: float = 0.0
    rtt_ms: float = 0.0
    energy_j: float = 0.0  # energie dynamique de la requete
    cost: float = 0.0
    finish_s: float = 0.0


@dataclass(frozen=True)
class NodeSnapshot:
    """Mesures telles que l'orchestrateur les obtient en interrogeant /state."""

    queue_ms: float
    cpu_load: float
    net_rtt_ms: float
    battery_fraction: Optional[float]
    in_flight: int
    available: bool


class NodeSim:
    TAU_CPU_S = 1.0  # constante de temps du lissage de la charge CPU
    RTT_ALPHA = 0.3  # lissage de la latence mesuree

    def __init__(
        self,
        profile: NodeProfile,
        service_model: ServiceModel,
        rng: np.random.Generator,
        service_scale: float = 1.0,
        rtt_scale: float = 1.0,
        battery_fraction: Optional[float] = None,
    ) -> None:
        self.profile = profile
        self.model = service_model
        self.rng = rng
        self.service_scale = service_scale
        self.rtt_base_ms = profile.rtt_ms * rtt_scale
        self.free_at = [0.0] * profile.servers
        self._finish_heap: list[float] = []
        self.cpu_ema = 0.0
        self.rtt_ema = self.rtt_base_ms
        self.battery = (
            BatteryModel(profile.battery, battery_fraction) if profile.battery is not None else None
        )
        self.t = 0.0
        self.down = False  # panne simulee
        # Compteurs de l'episode
        self.dyn_energy_j = 0.0
        self.on_time_s = 0.0
        self.busy_time_s = 0.0
        self.accepted = 0
        self.rejected = 0

    # -- etat -----------------------------------------------------------------

    @property
    def up(self) -> bool:
        return not self.down and (self.battery is None or self.battery.on)

    def busy_fraction(self, t: float) -> float:
        return sum(1 for f in self.free_at if f > t) / self.profile.servers

    def queue_wait_s(self, t: float) -> float:
        """Attente d'une requete qui arriverait maintenant (temps de travail en attente)."""

        return max(0.0, min(self.free_at) - t)

    def in_flight(self, t: float) -> int:
        heap = self._finish_heap
        while heap and heap[0] <= t:
            heapq.heappop(heap)
        return len(heap)

    def advance(self, t: float) -> None:
        """Fait avancer l'horloge : lissage de la charge, batterie, temps allume."""

        dt = t - self.t
        if dt <= 0:
            return
        alpha = 1.0 - math.exp(-dt / self.TAU_CPU_S)
        self.cpu_ema += alpha * (self.busy_fraction(t) - self.cpu_ema)
        if self.battery is not None:
            if self.battery.on:
                self.on_time_s += dt
            self.battery.step(dt, self.profile.p_idle_w if self.battery.on else 0.0)
        else:
            self.on_time_s += dt
        self.t = t

    def sample_rtt_ms(self) -> float:
        return self.rtt_base_ms * lognormal_factor(self.rng, self.profile.rtt_sigma)

    def probe_rtt(self) -> None:
        """Mesure de latence faite lors de l'interrogation periodique du noeud."""

        self.rtt_ema += self.RTT_ALPHA * (self.sample_rtt_ms() - self.rtt_ema)

    def snapshot(self, t: float) -> NodeSnapshot:
        return NodeSnapshot(
            queue_ms=self.queue_wait_s(t) * 1000.0,
            cpu_load=self.cpu_ema,
            net_rtt_ms=self.rtt_ema,
            battery_fraction=None if self.battery is None else self.battery.fraction,
            in_flight=self.in_flight(t),
            available=self.up,
        )

    # -- traitement -------------------------------------------------------------

    def submit(self, t: float, req_type: str, work_ms: float) -> SubmitResult:
        self.advance(t)
        if not self.up:
            self.rejected += 1
            return SubmitResult(False, "unavailable")
        if self.in_flight(t) >= self.profile.max_queue:
            self.rejected += 1
            return SubmitResult(False, "queue_full")

        rtt_ms = self.sample_rtt_ms()
        arrive = t + rtt_ms / 2000.0
        server = min(range(len(self.free_at)), key=self.free_at.__getitem__)
        start = max(arrive, self.free_at[server])
        wait_s = start - arrive
        load = self.busy_fraction(arrive)
        service_ms = (
            self.model.sample_ms(self.rng, self.profile, req_type, work_ms, load) * self.service_scale
        )
        service_s = service_ms / 1000.0
        finish = start + service_s
        self.free_at[server] = finish
        heapq.heappush(self._finish_heap, finish)

        energy = dynamic_energy_j(self.profile, service_s)
        self.dyn_energy_j += energy
        self.busy_time_s += service_s
        if self.battery is not None:
            self.battery.drain(energy)
        self.accepted += 1
        return SubmitResult(
            accepted=True,
            latency_ms=rtt_ms + wait_s * 1000.0 + service_ms,
            wait_ms=wait_s * 1000.0,
            service_ms=service_ms,
            rtt_ms=rtt_ms,
            energy_j=energy,
            cost=self.profile.cost_per_request,
            finish_s=finish,
        )

    # -- bilan ----------------------------------------------------------------

    def idle_energy_j(self) -> float:
        return self.profile.p_idle_w * self.on_time_s

    def utilization(self, duration_s: float) -> float:
        if duration_s <= 0:
            return 0.0
        return min(1.0, self.busy_time_s / (self.profile.servers * duration_s))
