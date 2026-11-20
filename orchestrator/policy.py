"""Contrats et politiques de routage utilisées par l'orchestrateur."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .nodes import WorkerNode


class PolicyDecisionError(RuntimeError):
    """Erreur explicite produite lorsqu'une politique ne peut pas décider."""


class RoutingPolicy(Protocol):
    """Classe les nœuds par préférence; le premier est le choix principal.

    Retourner les autres candidats après le premier permet à l'orchestrateur
    d'essayer un nœud de secours si la soumission échoue.
    """

    def rank_workers(
        self,
        request: Mapping[str, Any],
        states: Mapping[str, Mapping[str, Any]],
        workers: Mapping[str, WorkerNode],
        *,
        observation: Any = None,
        action_mask: Any = None,
    ) -> list[str]: ...


class LeastConnectionsPolicy:
    """Politique de secours qui choisit la file la moins chargée."""

    name = "least_connections"

    def rank_workers(
        self,
        request: Mapping[str, Any],
        states: Mapping[str, Mapping[str, Any]],
        workers: Mapping[str, WorkerNode],
        *,
        observation: Any = None,
        action_mask: Any = None,
    ) -> list[str]:
        del request, observation, action_mask
        candidates: list[tuple[int, str]] = []
        for name in workers:
            state = states.get(name)
            if state is None or state.get("depleted", False):
                continue
            waiting = int(state.get("waiting", 0))
            max_queue = int(state.get("max_queue", 0))
            if waiting < max_queue:
                candidates.append((waiting, name))
        candidates.sort()
        return [name for _, name in candidates]


def build_policy(name: str) -> RoutingPolicy:
    """Construit une politique réelle à partir de sa valeur de configuration."""
    if name == "least_connections":
        return LeastConnectionsPolicy()
    if name == "greedy":
        from baselines.greedy import GreedyPolicy

        return GreedyPolicy()
    raise ValueError(
        f"Politique inconnue : {name!r}. "
        "Valeurs disponibles : greedy, least_connections"
    )
