"""Politique Greedy : choisit le temps de fin estimé le plus faible.

Hypothèse d'observation :
    obs[0] = travail en attente sur le nœud rapide
    obs[1] = travail en attente sur le nœud lent
    obs[2] = travail de la nouvelle requête

Vitesses du jouet décrites dans le projet :
    nœud 0 : vitesse 2
    nœud 1 : vitesse 1

Important :
    Si ToyFogEnv normalise différemment les composantes de l'observation,
    il faut adapter cette politique à son state_spec exact.
"""

from __future__ import annotations
from typing import Any, Mapping

import numpy as np
from .base import Policy

# fournit cette méthode dans la classe GreedyPolicy : c’est là que se trouve le calcul greedy.
class GreedyPolicy(Policy):
    name = "Greedy"

    def __init__(self, speeds=(2.0, 1.0)) -> None:
        super().__init__()
        if len(speeds) != 2 or any(float(s) <= 0 for s in speeds):
            raise ValueError("speeds doit contenir deux valeurs strictement positives.")
        self.speeds = tuple(float(s) for s in speeds)

    def select(self, obs) -> int:
        values = np.asarray(obs, dtype=float).reshape(-1)
        if values.size < 3:
            raise ValueError(
                "Greedy attend obs=[queue_fast, queue_slow, request_work]. "
                f"Observation reçue avec {values.size} valeur(s)."
            )

        queue_fast, queue_slow, work = map(float, values[:3])
        if min(queue_fast, queue_slow, work) < 0:
            raise ValueError("Les files et le travail ne peuvent pas être négatifs.")

        finish_fast = queue_fast + work / self.speeds[0]
        finish_slow = queue_slow + work / self.speeds[1]
        return 0 if finish_fast <= finish_slow else 1

    def rank_workers(
        self,
        request: Mapping[str, Any],
        states: Mapping[str, Mapping[str, Any]],
        workers: Mapping[str, Any],
    ) -> list[str]:
        """Classe les workers réels selon leur temps de fin estimé.

        Cette méthode partage le contrat de l'orchestrateur, tout en gardant
        ``select`` compatible avec l'environnement jouet à deux nœuds.
        """
        work_ms = float(request["work_ms"])
        if work_ms < 0:
            raise ValueError("work_ms ne peut pas être négatif")

        candidates: list[tuple[float, str]] = []
        for name, state in states.items():
            node = workers[name]
            if state.get("depleted", False):
                continue
            if int(state.get("waiting", 0)) >= int(state.get("max_queue", 0)):
                continue
            queue_ms = float(state.get("queue_ms", 0.0))
            estimated_service_ms = work_ms / float(node.speed_factor)
            candidates.append((queue_ms + estimated_service_ms, name))

        candidates.sort()
        return [name for _, name in candidates]
