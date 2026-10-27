"""Structures de données du générateur de trafic."""
from dataclasses import dataclass, field
from typing import Optional

ZONES = ("A", "B", "C")


@dataclass(frozen=True)
class TypeSpec:
    """Un type de requête (légère, moyenne, lourde)."""
    mean_work_ms: float
    deadline_ms: float
    share: float


DEFAULT_TYPES = {
    # share part dans la requete
    "light":  TypeSpec(mean_work_ms=5,   deadline_ms=100,  share=0.6),
    "medium": TypeSpec(mean_work_ms=40,  deadline_ms=300,  share=0.3),
    "heavy":  TypeSpec(mean_work_ms=150, deadline_ms=1000, share=0.1),
}


@dataclass(frozen=True)
class TrafficConfig:
    """Configuration complète d'un scénario de trafic."""
    # Avec "poisson", les requêtes arrivent parfois rapprochées, parfois espacées.
    scenario: str = "poisson"
    rho: float = 0.6 
    # la requete demande 60 unités de travail par seconde
    capacity: float = 3.0
    n_requests: Optional[int] = 2000
    duration_ms: Optional[float] = None
    types: dict = field(default_factory=lambda: dict(DEFAULT_TYPES))
    work_cv: float = 0.3
    zone_weights: tuple = (0.34, 0.33, 0.33)

    def __post_init__(self):
        if self.rho <= 0:
            raise ValueError("rho doit être > 0")
        if self.capacity <= 0:
            raise ValueError("capacity doit être > 0")
        if (self.n_requests is None) == (self.duration_ms is None):
            raise ValueError("Définir exactement un des deux : n_requests ou duration_ms")
        if self.n_requests is not None and self.n_requests <= 0:
            raise ValueError("n_requests doit être > 0")
        if self.duration_ms is not None and self.duration_ms <= 0:
            raise ValueError("duration_ms doit être > 0")
        if self.work_cv < 0:
            raise ValueError("work_cv doit être >= 0")
        if len(self.zone_weights) != len(ZONES) or abs(sum(self.zone_weights) - 1) > 1e-6:
            raise ValueError("zone_weights : une valeur par zone, somme égale à 1")
        if abs(sum(t.share for t in self.types.values()) - 1) > 1e-6:
            raise ValueError("La somme des parts des types doit valoir 1")
        for name, t in self.types.items():
            if t.mean_work_ms <= 0 or t.deadline_ms <= 0:
                raise ValueError(f"Type {name} : travail et échéance doivent être > 0")