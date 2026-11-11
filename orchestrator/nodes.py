"""Configuration des workers; leurs URL peuvent être remplacées par variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WorkerNode:
    name: str
    url: str
    speed_factor: float


def load_workers() -> dict[str, WorkerNode]:
    """Retourne les workers configurés, dans l'ordre stable du profil.

    Les URL par défaut ciblent les ports publiés localement pour le développement.
    Au déploiement, définir FOGA_URL, FOGB_URL, FOGC_URL et CLOUD_URL avec
    les adresses réellement joignables depuis la machine de l'orchestrateur.
    """
    defaults = {
        "fogA": "http://127.0.0.1:8001",
        "fogB": "http://127.0.0.1:8002",
        "fogC": "http://127.0.0.1:8003",
        "cloud": "http://127.0.0.1:8004",
    }
    env_names = {"fogA": "FOGA_URL", "fogB": "FOGB_URL", "fogC": "FOGC_URL", "cloud": "CLOUD_URL"}
    profile_path = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "profiles.yaml"
    with profile_path.open(encoding="utf-8") as profile_file:
        profiles = yaml.safe_load(profile_file)["nodes"]
    return {
        name: WorkerNode(
            name=name,
            url=os.getenv(env_names[name], default).rstrip("/"),
            speed_factor=float(profiles[name]["speed"]),
        )
        for name, default in defaults.items()
    }
