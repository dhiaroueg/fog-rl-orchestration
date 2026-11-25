"""Politique RL : charge un modele SB3 entraine et decide a partir de l'etat brut.

Chemin d'inference identique en simulation et dans l'orchestrateur reel :
etat brut -> ``encode_state`` (contrat partage) -> reseau -> argmax sur les
actions VALIDES. DQN, PPO et A2C sont entraines sans masque (comparaison
equitable : SB3 ne masque nativement aucun des trois) ; le masquage est
applique ici a l'inference, ce qui couvre aussi la panne d'un noeud (EF-09).

Un repertoire de modele contient :
    model.zip, contract.json (+ vecnormalize.pkl si l'observation est normalisee)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..config import resolve_path
from ..state_spec import (
    DEFAULT_NORMALIZATION,
    DecisionContext,
    NodeState,
    Normalization,
    RequestState,
    check_contract,
    encode_state,
    pad_nodes,
)
from .base import Policy

ALGOS = ("ppo", "dqn", "a2c")


def algo_class(algo: str):
    """Classe SB3 de l'algorithme (import paresseux : torch n'est charge qu'ici)."""

    from stable_baselines3 import A2C, DQN, PPO

    try:
        return {"ppo": PPO, "dqn": DQN, "a2c": A2C}[algo]
    except KeyError:
        raise ValueError(f"Algorithme inconnu : {algo!r} (disponibles : {ALGOS})") from None


class RLPolicy(Policy):
    needs_budget = True

    def __init__(
        self,
        model: Any,
        algo: str,
        slot_ids: Sequence[str],
        normalization: Normalization = DEFAULT_NORMALIZATION,
        obs_stats: Optional[dict] = None,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.algo = algo
        self.slot_ids = tuple(slot_ids)
        self.normalization = normalization
        self.obs_stats = obs_stats
        self.name = name or algo
        self.metadata = metadata or {}

    # ------------------------------------------------------------- chargement

    @classmethod
    def load(
        cls,
        model_dir,
        name: Optional[str] = None,
        slot_ids: Optional[Sequence[str]] = None,
        normalization: Optional[Normalization] = None,
    ) -> "RLPolicy":
        """Charge un modele et verifie son contrat d'observation.

        ``slot_ids`` et ``normalization`` (ceux de l'environnement ou de
        l'orchestrateur) sont compares a ceux de l'entrainement : une
        incompatibilite leve une ValueError au chargement.
        """

        import torch

        directory = resolve_path(model_dir)
        with open(directory / "contract.json", encoding="utf-8") as handle:
            meta = json.load(handle)
        contract = meta["contract"]
        norm = Normalization(**contract["normalization"])
        if slot_ids is not None:
            check_contract(contract, slot_ids, normalization or norm)

        torch.set_num_threads(1)  # latence de decision stable (ENF-01)
        model = algo_class(meta["algo"]).load(directory / "model.zip", device="cpu")

        stats = None
        stats_path = directory / "vecnormalize.pkl"
        if meta.get("obs_normalized") and stats_path.exists():
            stats = _load_obs_stats(stats_path)
        return cls(
            model,
            meta["algo"],
            contract["node_ids"],
            norm,
            stats,
            name=name or meta["algo"],
            metadata=meta,
        )

    # ------------------------------------------------------------- inference

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_stats is None:
            return obs
        s = self.obs_stats
        return np.clip((obs - s["mean"]) / np.sqrt(s["var"] + s["epsilon"]), -s["clip"], s["clip"]).astype(
            np.float32
        )

    def action_values(self, obs: np.ndarray) -> np.ndarray:
        """Score de chaque action : Q-values (DQN) ou logits (PPO, A2C)."""

        import torch

        tensor = torch.as_tensor(self._normalize(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.algo == "dqn":
                values = self.model.q_net(tensor)[0]
            else:
                values = self.model.policy.get_distribution(tensor).distribution.logits[0]
        return values.cpu().numpy()

    def select(self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext) -> int:
        if tuple(n.node_id for n in nodes) != self.slot_ids:
            nodes = pad_nodes(nodes, self.slot_ids, self.normalization)
        obs, mask = encode_state(nodes, request, ctx.arrival_rate_rps, self.normalization)
        values = np.where(mask, self.action_values(obs), -np.inf)
        return int(np.argmax(values))


def _load_obs_stats(path: Path) -> dict:
    # pickle : ne charger que des fichiers produits par ce depot (confiance requise).
    with open(path, "rb") as handle:
        vec_normalize = pickle.load(handle)
    rms = vec_normalize.obs_rms
    return {
        "mean": np.asarray(rms.mean, dtype=np.float64),
        "var": np.asarray(rms.var, dtype=np.float64),
        "epsilon": float(vec_normalize.epsilon),
        "clip": float(vec_normalize.clip_obs),
    }
