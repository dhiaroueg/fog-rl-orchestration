"""Contrat minimal que toute politique de routage doit respecter."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .nodes import WorkerNode

# décrit la méthode attendue : rank_workers(request, states, workers).
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
    ) -> list[str]: ...
