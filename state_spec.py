"""Contrat partage des observations et actions du routeur fog.

Pour chaque noeud, l'observation contient dans cet ordre :
    queue_ms, cpu_load, net_rtt_ms, battery_fraction
Puis viennent les caracteristiques de la requete :
    type_light, type_medium, type_heavy, work_ms, deadline_ms
Enfin, le debit recent global : arrival_rate_rps.

Avec quatre noeuds (fogA, fogB, fogC et cloud), cela fait 22 valeurs,
comme dans le cahier des charges. La taille est fixe pour une configuration
donnee : un modele SB3 entraine avec N noeuds attend 4*N + 6 valeurs.

Les constantes de normalisation sont des valeurs de depart. Elles devront
etre revisees avec les mesures de calibration des VM.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


REQUEST_TYPES = ("light", "medium", "heavy")
NODE_FEATURES = ("queue_ms", "cpu_load", "net_rtt_ms", "battery_fraction")
REQUEST_FEATURES = (
    "type_light",
    "type_medium",
    "type_heavy",
    "work_ms",
    "deadline_ms",
)
GLOBAL_FEATURES = ("arrival_rate_rps",)
FEATURES_PER_NODE = len(NODE_FEATURES)
FEATURES_AFTER_NODES = len(REQUEST_FEATURES) + len(GLOBAL_FEATURES)


@dataclass(frozen=True)
class Normalization:
    """Valeurs de reference servant a ramener les mesures vers [0, 1]."""

    queue_ms: float = 1000.0
    net_rtt_ms: float = 200.0
    work_ms: float = 300.0
    deadline_ms: float = 2000.0
    arrival_rate_rps: float = 100.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"La normalisation {name} doit etre > 0")


DEFAULT_NORMALIZATION = Normalization()


@dataclass(frozen=True)
class NodeState:
    """Mesures courantes d'un noeud dans l'ordre de la configuration."""

    node_id: str
    queue_ms: float
    cpu_load: float
    net_rtt_ms: float
    battery_fraction: Optional[float] = None
    available: bool = True


@dataclass(frozen=True)
class RequestState:
    """Caracteristiques de la requete a router."""

    request_type: str
    work_ms: float
    deadline_ms: float


def observation_dim(n_nodes: int) -> int:
    """Taille du vecteur d'observation pour une configuration de N noeuds."""

    if n_nodes <= 0:
        raise ValueError("Il faut au moins un noeud")
    return FEATURES_PER_NODE * n_nodes + FEATURES_AFTER_NODES


def observation_feature_names(node_ids: Sequence[str]) -> tuple[str, ...]:
    """Retourne les noms des valeurs dans l'ordre exact de l'observation."""

    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids doit contenir des identifiants uniques")

    names = tuple(
        f"{node_id}.{feature}"
        for node_id in node_ids
        for feature in NODE_FEATURES
    )
    return names + REQUEST_FEATURES + GLOBAL_FEATURES


def encode_state(
    nodes: Sequence[NodeState],
    request: RequestState,
    arrival_rate_rps: float,
    normalization: Normalization = DEFAULT_NORMALIZATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode l'etat pour l'agent et renvoie le masque des actions disponibles.

    L'index d'action correspond exactement a la position du noeud dans
    ``nodes``. Le masque vaut True pour un noeud selectionnable. Un noeud sans
    batterie est represente par battery_fraction=1.0 (aucune contrainte de
    batterie); un noeud indisponible garde ses mesures mais son action est
    masquee.

    Les valeurs continues sont bornees dans [0, 1]. L'ordre des noeuds doit
    rester identique entre entrainement, evaluation et orchestration.
    """

    if not nodes:
        raise ValueError("nodes ne peut pas etre vide")
    node_ids = [node.node_id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Chaque node_id doit etre unique")
    if request.request_type not in REQUEST_TYPES:
        raise ValueError(
            f"Type de requete inconnu : {request.request_type!r}; "
            f"attendu parmi {REQUEST_TYPES}"
        )

    values: list[float] = []
    for node in nodes:
        battery = 1.0 if node.battery_fraction is None else node.battery_fraction
        values.extend(
            (
                _normalized(node.queue_ms, normalization.queue_ms),
                _unit_interval(node.cpu_load),
                _normalized(node.net_rtt_ms, normalization.net_rtt_ms),
                _unit_interval(battery),
            )
        )

    values.extend(float(request.request_type == kind) for kind in REQUEST_TYPES)
    values.extend(
        (
            _normalized(request.work_ms, normalization.work_ms),
            _normalized(request.deadline_ms, normalization.deadline_ms),
            _normalized(arrival_rate_rps, normalization.arrival_rate_rps),
        )
    )

    observation = np.asarray(values, dtype=np.float32)
    action_mask = np.asarray([node.available for node in nodes], dtype=np.bool_)
    if not action_mask.any():
        raise ValueError("Aucun noeud disponible pour traiter la requete")
    return observation, action_mask


def _unit_interval(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalized(value: float, reference: float) -> float:
    return float(np.clip(value / reference, 0.0, 1.0))
