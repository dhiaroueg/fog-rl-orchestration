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
import numpy as np
from .base import Policy


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
