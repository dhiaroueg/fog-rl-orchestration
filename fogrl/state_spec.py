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

Robustesse (version 2 du contrat) :
    - une mesure non finie (NaN, inf) ou negative est refusee avec une
      ValueError au lieu d'etre transformee en silence ;
    - ``contract_metadata`` produit un dictionnaire (a sauvegarder en JSON a
      cote de chaque modele) et ``check_contract`` le compare a la
      configuration courante, pour detecter un ordre de noeuds ou une
      normalisation differents de ceux de l'entrainement.

Dispositions des noeuds (decision D1, voir docs/decisions.md) :
    - disposition compacte : les N noeuds actifs, dans l'ordre de la
      configuration (4 noeuds -> 22 valeurs) ;
    - disposition remplie : un nombre fixe d'emplacements (par exemple 6),
      les noeuds absents etant representes par ``absent_node`` (satures,
      batterie vide, action masquee). Un meme modele peut alors servir pour
      3, 4 ou 6 noeuds. ``pad_nodes`` construit cette liste.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SPEC_VERSION = 2

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
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"La normalisation {name} doit etre > 0 et finie")


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
    # Requetes en cours (non encode dans l'observation de l'agent ; utilise
    # par la baseline least-connections).
    in_flight: int = 0


@dataclass(frozen=True)
class RequestState:
    """Caracteristiques de la requete a router."""

    request_type: str
    work_ms: float
    deadline_ms: float


@dataclass(frozen=True)
class DecisionContext:
    """Contexte d'une decision de routage, en plus des noeuds et de la requete."""

    now_s: float = 0.0
    arrival_rate_rps: float = 0.0


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

    Une mesure non finie ou negative leve une ValueError : l'appelant
    (routeur) doit alors utiliser sa strategie de repli plutot que de
    laisser l'agent decider sur une donnee corrompue. Les valeurs cpu_load et
    battery_fraction legerement hors de [0, 1] restent simplement bornees.
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
        where = f"noeud {node.node_id}"
        battery = 1.0 if node.battery_fraction is None else node.battery_fraction
        values.extend(
            (
                _normalized(node.queue_ms, normalization.queue_ms, f"{where}.queue_ms"),
                _unit_interval(node.cpu_load, f"{where}.cpu_load"),
                _normalized(
                    node.net_rtt_ms, normalization.net_rtt_ms, f"{where}.net_rtt_ms"
                ),
                _unit_interval(battery, f"{where}.battery_fraction"),
            )
        )

    values.extend(float(request.request_type == kind) for kind in REQUEST_TYPES)
    values.extend(
        (
            _normalized(request.work_ms, normalization.work_ms, "request.work_ms"),
            _normalized(
                request.deadline_ms, normalization.deadline_ms, "request.deadline_ms"
            ),
            _normalized(
                arrival_rate_rps, normalization.arrival_rate_rps, "arrival_rate_rps"
            ),
        )
    )

    observation = np.asarray(values, dtype=np.float32)
    action_mask = np.asarray([node.available for node in nodes], dtype=np.bool_)
    if not action_mask.any():
        raise ValueError("Aucun noeud disponible pour traiter la requete")
    return observation, action_mask


def absent_node(
    node_id: str, normalization: Normalization = DEFAULT_NORMALIZATION
) -> NodeState:
    """Emplacement vide d'une disposition remplie : sature, batterie vide, masque."""

    return NodeState(
        node_id=node_id,
        queue_ms=normalization.queue_ms,
        cpu_load=1.0,
        net_rtt_ms=normalization.net_rtt_ms,
        battery_fraction=0.0,
        available=False,
    )


def pad_nodes(
    nodes: Sequence[NodeState],
    slot_ids: Sequence[str],
    normalization: Normalization = DEFAULT_NORMALIZATION,
) -> list[NodeState]:
    """Place les noeuds mesures dans l'ordre des emplacements ``slot_ids``.

    Les emplacements sans noeud recoivent ``absent_node``. Un noeud dont
    l'identifiant n'est pas dans ``slot_ids`` leve une ValueError.
    """

    by_id = {node.node_id: node for node in nodes}
    unknown = set(by_id) - set(slot_ids)
    if unknown:
        raise ValueError(f"Noeuds hors des emplacements : {sorted(unknown)}")
    return [
        by_id[slot] if slot in by_id else absent_node(slot, normalization)
        for slot in slot_ids
    ]


def contract_metadata(
    node_ids: Sequence[str],
    normalization: Normalization = DEFAULT_NORMALIZATION,
) -> dict[str, Any]:
    """Decrit le contrat d'observation, a sauvegarder en JSON avec le modele."""

    return {
        "spec_version": SPEC_VERSION,
        "node_ids": list(node_ids),
        "observation_dim": observation_dim(len(node_ids)),
        "feature_names": list(observation_feature_names(node_ids)),
        "normalization": asdict(normalization),
    }


def check_contract(
    metadata: Mapping[str, Any],
    node_ids: Sequence[str],
    normalization: Normalization = DEFAULT_NORMALIZATION,
) -> None:
    """Verifie qu'un modele entraine est compatible avec la configuration.

    Leve une ValueError si la version du contrat, l'ordre des noeuds, la
    taille de l'observation ou la normalisation different.
    """

    expected = contract_metadata(node_ids, normalization)
    for key in ("spec_version", "node_ids", "observation_dim", "normalization"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"Contrat incompatible sur {key!r} : le modele a "
                f"{metadata.get(key)!r}, la configuration courante a "
                f"{expected[key]!r}"
            )


def _checked(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Mesure non finie pour {label} : {value}")
    if value < 0:
        raise ValueError(f"Mesure negative pour {label} : {value}")
    return value


def _unit_interval(value: float, label: str) -> float:
    return float(np.clip(_checked(value, label), 0.0, 1.0))


def _normalized(value: float, reference: float, label: str) -> float:
    return float(np.clip(_checked(value, label) / reference, 0.0, 1.0))
