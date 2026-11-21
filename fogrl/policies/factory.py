"""Construction d'une politique depuis une specification texte (simulation et orchestrateur reel).

``spec`` : nom de baseline (random, round_robin, least_connections, static_threshold,
greedy) ou ``rl:<repertoire du modele>``. Ce module n'importe ni gymnasium ni torch
tant qu'une politique RL n'est pas demandee : l'image Docker de l'orchestrateur
peut donc se passer de torch quand elle ne sert que des baselines.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..config import NodeProfile
from ..state_spec import Normalization
from .base import Policy
from .baselines import make_baseline


def is_rl_spec(spec: str) -> bool:
    return spec.startswith("rl:")


def make_policy(
    spec: str,
    profiles: Mapping[str, NodeProfile],
    slot_ids: Optional[Sequence[str]] = None,
    normalization: Optional[Normalization] = None,
    name: Optional[str] = None,
) -> Policy:
    """``slot_ids`` et ``normalization`` servent a verifier le contrat d'un modele RL."""

    if is_rl_spec(spec):
        from .rl import RLPolicy

        return RLPolicy.load(spec[3:], name=name, slot_ids=slot_ids, normalization=normalization)
    return make_baseline(spec, profiles)
