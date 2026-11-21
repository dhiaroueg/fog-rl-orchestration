"""Modeles de temps de service du simulateur.

Deux variantes :
    - ``AnalyticServiceModel`` : valeurs theoriques (avant calibration) ;
    - ``CalibratedServiceModel`` : distributions mesurees sur les VM (CSV de
      calibration reduit par ``fogrl.calibration.fit``), cahier des charges
      Phase 2, etape 4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ..config import NodeProfile, RequestProfile, resolve_path


def lognormal_factor(rng: np.random.Generator, cv: float) -> float:
    """Facteur multiplicatif log-normal de moyenne 1 et de coefficient de variation ``cv``."""

    if cv <= 0:
        return 1.0
    sigma2 = math.log(1.0 + cv * cv)
    return math.exp(rng.normal(-sigma2 / 2.0, math.sqrt(sigma2)))


class AnalyticServiceModel:
    """temps = travail / vitesse * (1 + contention * charge) * bruit log-normal."""

    def mean_ms(self, profile: NodeProfile, req_type: str, work_ms: float, load: float) -> float:
        return work_ms / profile.speed * (1.0 + profile.contention * min(max(load, 0.0), 1.0))

    def sample_ms(
        self,
        rng: np.random.Generator,
        profile: NodeProfile,
        req_type: str,
        work_ms: float,
        load: float,
    ) -> float:
        return self.mean_ms(profile, req_type, work_ms, load) * lognormal_factor(rng, profile.service_cv)


class CalibratedServiceModel:
    """Moyenne et dispersion interpolees selon la charge, par (noeud, type de requete).

    La table vient des mesures ; la moyenne est ramenee a la taille de la
    requete (``work_ms`` / travail nominal du type) et ``cv`` est le bruit
    residuel. Les couples absents de la table retombent sur le modele analytique.
    """

    def __init__(
        self,
        table: Mapping[tuple[str, str], Mapping[str, Sequence[float]]],
        nominal_work_ms: Mapping[str, float],
        fallback: Optional[AnalyticServiceModel] = None,
    ) -> None:
        self.table = {
            key: {
                "loads": np.asarray(entry["loads"], dtype=float),
                "mean_ms": np.asarray(entry["mean_ms"], dtype=float),
                "cv": np.asarray(entry["cv"], dtype=float),
            }
            for key, entry in table.items()
        }
        self.nominal_work_ms = dict(nominal_work_ms)
        self.fallback = fallback or AnalyticServiceModel()

    @classmethod
    def from_json(
        cls, path: Union[str, Path], requests: Sequence[RequestProfile]
    ) -> "CalibratedServiceModel":
        with open(resolve_path(path), encoding="utf-8") as handle:
            data = json.load(handle)
        table = {(e["node"], e["req_type"]): e for e in data["entries"]}
        # Les durees de la table sont celles du travail nominal UTILISE PAR LA CALIBRATION : on s'y
        # refere (et non a requests.yaml, qui a pu changer depuis) pour remettre chaque requete a l'echelle.
        nominal = data.get("nominal_work_ms") or {r.name: r.work_ms for r in requests}
        return cls(table, nominal)

    def covers(self, node_id: str, req_type: str) -> bool:
        return (node_id, req_type) in self.table

    def mean_ms(self, profile: NodeProfile, req_type: str, work_ms: float, load: float) -> float:
        entry = self.table.get((profile.node_id, req_type))
        if entry is None:
            return self.fallback.mean_ms(profile, req_type, work_ms, load)
        mean = float(np.interp(load, entry["loads"], entry["mean_ms"]))
        return mean * work_ms / self.nominal_work_ms[req_type]

    def sample_ms(
        self,
        rng: np.random.Generator,
        profile: NodeProfile,
        req_type: str,
        work_ms: float,
        load: float,
    ) -> float:
        entry = self.table.get((profile.node_id, req_type))
        if entry is None:
            return self.fallback.sample_ms(rng, profile, req_type, work_ms, load)
        cv = float(np.interp(load, entry["loads"], entry["cv"]))
        return self.mean_ms(profile, req_type, work_ms, load) * lognormal_factor(rng, cv)


ServiceModel = Union[AnalyticServiceModel, CalibratedServiceModel]


def build_service_model(
    kind: str, calibration_file: Optional[str], requests: Sequence[RequestProfile]
) -> ServiceModel:
    if kind == "analytic":
        return AnalyticServiceModel()
    if not calibration_file:
        raise ValueError("service_model: calibrated exige calibration_file")
    return CalibratedServiceModel.from_json(calibration_file, requests)
