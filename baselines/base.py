"""Interface commune des politiques de routage.

Contrat :
    policy.reset(seed): réinitialise l'état interne de la politique.
    policy.select(obs): retourne une action entière (0 ou 1).

Hypothèse du jouet P1 :
    obs = [queue_fast, queue_slow, request_work]
Les valeurs peuvent être normalisées dans [0, 1]. Si elles le sont,
Greedy reste une estimation cohérente uniquement si les trois composantes
utilisent une échelle compatible. Vérifier state_spec.py / l'environnement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Policy(ABC):
    """Classe de base commune aux politiques."""

    name = "policy"
    n_actions = 2

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """Réinitialise l'état interne et la source aléatoire."""
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select(self, obs: Any) -> int:
        """Choisit une action valide : 0 (nœud rapide) ou 1 (nœud lent)."""
        raise NotImplementedError

    @staticmethod
    def validate_action(action: int, n_actions: int = 2) -> int:
        action = int(action)
        if not 0 <= action < n_actions:
            raise ValueError(f"Action invalide {action}; attendu entre 0 et {n_actions - 1}.")
        return action
