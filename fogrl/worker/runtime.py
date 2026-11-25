"""Etat d'un noeud reel : machine a etats pure (sans asyncio, sans horloge).

Le worker FastAPI (``app.py``) appelle ces methodes avec ``time.monotonic()`` ;
les tests les appellent avec des dates choisies, ce qui rend le comptage de la
file, de la charge CPU, de l'energie et de la batterie deterministe.

Les definitions sont celles du simulateur (``fogrl.sim.node``) pour que les
mesures reelles et simulees soient comparables :
    - ``queue_ms`` : attente estimee d'une requete qui arriverait maintenant ;
    - ``cpu_load`` : fraction de serveurs occupes, lissee (EMA, tau = 1 s) ;
    - energie dynamique d'une requete = (Pmax - Pidle) / servers x duree de service ;
    - energie a vide = Pidle x temps allume ; batterie virtuelle (voir ``sim.energy``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import NodeProfile
from ..sim.energy import BatteryModel, dynamic_energy_j


@dataclass
class _Job:
    req_type: str
    est_ms: float  # duree de service estimee (sert au calcul de queue_ms)
    start_s: Optional[float] = None  # None tant que la requete attend un serveur


@dataclass(frozen=True)
class Completion:
    energy_j: float
    cost: float
    battery_fraction: Optional[float]


class NodeRuntime:
    TAU_CPU_S = 1.0  # meme constante que le simulateur
    RATIO_ALPHA = 0.2  # lissage du rapport duree mesuree / travail demande

    def __init__(self, profile: NodeProfile, now: float = 0.0, initial_speed_hint: Optional[float] = None) -> None:
        self.profile = profile
        # Rapport service/travail attendu, appris en ligne : l'estimation de la file
        # suit ainsi le debit reel du conteneur (limites CPU Docker comprises).
        self._ratio_init = 1.0 / (initial_speed_hint or 1.0)
        self.reset(now)

    # ------------------------------------------------------------------ etat

    def reset(self, now: float = 0.0) -> None:
        """Remet les compteurs, la batterie et la charge a zero (entre deux runs)."""

        p = self.profile
        self.t = now
        self.t_start = now
        self.cpu_ema = 0.0
        self.battery = BatteryModel(p.battery) if p.battery is not None else None
        self.down = False
        self._jobs: dict[str, _Job] = {}
        self._ratio: dict[str, float] = {}
        # Compteurs (depuis le dernier reset)
        self.on_time_s = 0.0
        self.busy_time_s = 0.0
        self.dyn_energy_j = 0.0
        self.served = 0
        self.rejected_unavailable = 0
        self.rejected_queue_full = 0

    @property
    def up(self) -> bool:
        return not self.down and (self.battery is None or self.battery.on)

    @property
    def in_flight(self) -> int:
        return len(self._jobs)

    @property
    def busy(self) -> int:
        return sum(1 for job in self._jobs.values() if job.start_s is not None)

    def busy_fraction(self) -> float:
        return self.busy / self.profile.servers

    def advance(self, now: float) -> None:
        """Avance l'horloge : lissage de la charge, batterie, temps allume."""

        dt = now - self.t
        if dt <= 0:
            return
        alpha = 1.0 - math.exp(-dt / self.TAU_CPU_S)
        self.cpu_ema += alpha * (self.busy_fraction() - self.cpu_ema)
        if self.battery is not None:
            if self.battery.on:
                self.on_time_s += dt
            self.battery.step(dt, self.profile.p_idle_w if self.battery.on else 0.0)
        else:
            self.on_time_s += dt
        self.t = now

    # ------------------------------------------------------------- requetes

    def estimate_ms(self, req_type: str, work_ms: float) -> float:
        return work_ms * self._ratio.get(req_type, self._ratio_init)

    def admit(self, now: float, request_id: str, req_type: str, work_ms: float) -> Optional[str]:
        """Admet une requete. Retourne None, ou le motif du rejet (unavailable | queue_full)."""

        self.advance(now)
        if not self.up:
            self.rejected_unavailable += 1
            return "unavailable"
        if self.in_flight >= self.profile.max_queue:
            self.rejected_queue_full += 1
            return "queue_full"
        if request_id in self._jobs:
            raise ValueError(f"request_id deja en cours : {request_id}")
        self._jobs[request_id] = _Job(req_type, self.estimate_ms(req_type, work_ms))
        return None

    def start(self, now: float, request_id: str) -> float:
        """Un serveur commence la requete. Retourne la charge (fraction de serveurs occupes) a cet instant."""

        self.advance(now)
        load = self.busy_fraction()  # sans la requete elle-meme, comme ``busy_fraction(arrive)`` du simulateur
        self._jobs[request_id].start_s = now
        return load

    def finish(self, now: float, request_id: str, work_ms: float, service_ms: float) -> Completion:
        self.advance(now)
        job = self._jobs.pop(request_id)
        service_s = max(service_ms, 0.0) / 1000.0
        energy = dynamic_energy_j(self.profile, service_s)
        self.dyn_energy_j += energy
        self.busy_time_s += service_s
        if self.battery is not None:
            self.battery.drain(energy)
        self.served += 1
        if work_ms > 0:
            old = self._ratio.get(job.req_type, self._ratio_init)
            self._ratio[job.req_type] = old + self.RATIO_ALPHA * (service_ms / work_ms - old)
        return Completion(
            energy_j=energy,
            cost=self.profile.cost_per_request,
            battery_fraction=None if self.battery is None else self.battery.fraction,
        )

    def abort(self, now: float, request_id: str) -> None:
        """Requete annulee (client deconnecte) : elle libere sa place sans compter comme servie."""

        self.advance(now)
        self._jobs.pop(request_id, None)

    # ---------------------------------------------------------------- mesures

    def queue_wait_ms(self, now: float) -> float:
        """Attente d'une requete qui arriverait maintenant (0 tant qu'un serveur est libre)."""

        if len(self._jobs) < self.profile.servers:
            return 0.0
        remaining_ms = 0.0
        for job in self._jobs.values():
            if job.start_s is None:
                remaining_ms += job.est_ms
            else:
                remaining_ms += max(job.est_ms - (now - job.start_s) * 1000.0, 0.0)
        return remaining_ms / self.profile.servers

    def snapshot(self, now: float) -> dict:
        """Reponse de ``GET /state`` : mesures de decision + compteurs d'energie."""

        self.advance(now)
        return {
            "node_id": self.profile.node_id,
            "kind": self.profile.kind,
            "queue_ms": self.queue_wait_ms(now),
            "cpu_load": self.cpu_ema,
            "battery_fraction": None if self.battery is None else self.battery.fraction,
            "in_flight": self.in_flight,
            "available": self.up,
            "servers": self.profile.servers,
            "uptime_s": now - self.t_start,
            "on_time_s": self.on_time_s,
            "busy_time_s": self.busy_time_s,
            "energy_dynamic_j": self.dyn_energy_j,
            "energy_idle_j": self.profile.p_idle_w * self.on_time_s,
            "served": self.served,
            "rejected_unavailable": self.rejected_unavailable,
            "rejected_queue_full": self.rejected_queue_full,
        }
