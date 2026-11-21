"""Les 5 baselines du cahier des charges (OS-4) : Random, Round-robin,
Least-connections, Seuils statiques et Greedy.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from ..config import NodeProfile
from ..state_spec import DecisionContext, NodeState, RequestState
from .base import Policy


class RandomPolicy(Policy):
    """Borne basse : noeud disponible tire au hasard."""

    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select(self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext) -> int:
        candidates = self.available_indices(nodes)
        return int(candidates[self._rng.integers(len(candidates))])


class RoundRobinPolicy(Policy):
    """Chaque noeud disponible a tour de role. Ignore puissance et charge."""

    name = "round_robin"

    def __init__(self) -> None:
        self._last = -1

    def reset(self, seed: Optional[int] = None) -> None:
        self._last = -1

    def select(self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext) -> int:
        self.available_indices(nodes)  # leve s'il n'y en a aucun
        n = len(nodes)
        for step in range(1, n + 1):
            candidate = (self._last + step) % n
            if nodes[candidate].available:
                self._last = candidate
                return candidate
        raise ValueError("Aucun noeud disponible")


class LeastConnectionsPolicy(Policy):
    """Noeud avec le moins de requetes en cours (egalite : file la plus courte).

    Sert aussi de regle de repli de l'orchestrateur (EF-07).
    """

    name = "least_connections"

    def select(self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext) -> int:
        candidates = self.available_indices(nodes)
        return min(candidates, key=lambda i: (nodes[i].in_flight, nodes[i].queue_ms, i))


class StaticThresholdPolicy(Policy):
    """Fog le plus proche tant que sa charge est sous des seuils fixes, sinon cloud.

    Les seuils sont fixes a la main (c'est la limite de cette methode) :
    ``cpu_load < cpu_threshold``, ``queue_ms < queue_threshold_ms`` et batterie
    au-dessus de ``battery_min``.
    """

    name = "static_threshold"

    def __init__(
        self,
        profiles: Mapping[str, NodeProfile],
        cpu_threshold: float = 0.7,
        queue_threshold_ms: float = 200.0,
        battery_min: float = 0.15,
    ) -> None:
        self.profiles = profiles
        self.cpu_threshold = cpu_threshold
        self.queue_threshold_ms = queue_threshold_ms
        self.battery_min = battery_min

    def _is_cloud(self, node: NodeState) -> bool:
        profile = self.profiles.get(node.node_id)
        return profile is not None and profile.is_cloud

    def select(self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext) -> int:
        candidates = self.available_indices(nodes)
        fogs = sorted(
            (i for i in candidates if not self._is_cloud(nodes[i])),
            key=lambda i: (nodes[i].net_rtt_ms, i),
        )
        for i in fogs:
            node = nodes[i]
            battery_ok = node.battery_fraction is None or node.battery_fraction >= self.battery_min
            if (
                node.cpu_load < self.cpu_threshold
                and node.queue_ms < self.queue_threshold_ms
                and battery_ok
            ):
                return i
        clouds = [i for i in candidates if self._is_cloud(nodes[i])]
        if clouds:
            return clouds[0]
        return min(candidates, key=lambda i: (nodes[i].queue_ms, i))  # pas de cloud : le moins charge


class GreedyPolicy(Policy):
    """Choisit le noeud qui finira le plus tot : RTT + file + temps de service estime.

    Meilleure heuristique et cible reelle a battre ; myope (ignore energie,
    cout et effet sur les requetes suivantes).
    """

    name = "greedy"

    def __init__(self, profiles: Mapping[str, NodeProfile]) -> None:
        self.profiles = profiles

    def estimated_completion_ms(self, node: NodeState, request: RequestState) -> float:
        profile = self.profiles[node.node_id]
        service = request.work_ms / profile.speed * (1.0 + profile.contention * node.cpu_load)
        return node.net_rtt_ms + node.queue_ms + service

    def select(self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext) -> int:
        candidates = self.available_indices(nodes)
        return min(candidates, key=lambda i: (self.estimated_completion_ms(nodes[i], request), i))


BASELINES = ("random", "round_robin", "least_connections", "static_threshold", "greedy")


def make_baseline(name: str, profiles: Mapping[str, NodeProfile]) -> Policy:
    if name == "random":
        return RandomPolicy()
    if name == "round_robin":
        return RoundRobinPolicy()
    if name == "least_connections":
        return LeastConnectionsPolicy()
    if name == "static_threshold":
        return StaticThresholdPolicy(profiles)
    if name == "greedy":
        return GreedyPolicy(profiles)
    raise ValueError(f"Baseline inconnue : {name!r} (disponibles : {BASELINES})")
