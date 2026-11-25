"""Interface commune des politiques de routage (EF-08).

Les 8 politiques (Random, Round-robin, Least-connections, Seuils statiques,
Greedy, DQN, PPO, A2C) implementent ``select(etat) -> action`` et tournent a
l'identique dans le simulateur et dans l'orchestrateur reel.

``select`` recoit les mesures BRUTES (non normalisees) des noeuds, dans l'ordre
des emplacements de l'agent, et retourne l'index du noeud choisi. Les noeuds
indisponibles (``available=False``) ne doivent jamais etre choisis tant qu'un
autre noeud l'est.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..state_spec import DecisionContext, NodeState, RequestState


class Policy(ABC):
    name: str = "policy"
    # True pour les agents dont le temps de decision doit etre borne (EF-07).
    needs_budget: bool = False

    def reset(self, seed: Optional[int] = None) -> None:
        """Reinitialise l'etat interne avant un episode ou un run."""

    @abstractmethod
    def select(
        self,
        nodes: Sequence[NodeState],
        request: RequestState,
        ctx: DecisionContext,
    ) -> int:
        """Retourne l'index du noeud (dans ``nodes``) qui traitera la requete."""

    @staticmethod
    def available_indices(nodes: Sequence[NodeState]) -> list[int]:
        indices = [i for i, node in enumerate(nodes) if node.available]
        if not indices:
            raise ValueError("Aucun noeud disponible")
        return indices
