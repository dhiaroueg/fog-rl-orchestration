"""Petits utilitaires : version git, graines, versions des bibliotheques."""

from __future__ import annotations

import subprocess
from importlib import metadata
from typing import Sequence

from .config import PROJECT_ROOT

TRAIN_SEED_MAX = 99  # graines d'entrainement : 0-99 (cahier 6.3)
EVAL_SEED_MIN = 1000  # graines d'evaluation : 1000+


def git_commit() -> str:
    """Hash court du commit courant (+ '-dirty' si des modifications ne sont pas commitees)."""

    try:
        run = lambda *args: subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        commit = run("rev-parse", "--short", "HEAD")
        return commit + ("-dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def parse_seeds(text: str) -> list[int]:
    """'1000-1004' -> [1000..1004] ; '1,5,9' -> [1, 5, 9]."""

    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"Graines invalides : {text!r}")
    return seeds


def check_eval_seeds(seeds: Sequence[int]) -> None:
    """Les graines d'evaluation doivent etre separees de celles d'entrainement (>= 1000)."""

    bad = [s for s in seeds if s < EVAL_SEED_MIN]
    if bad:
        raise ValueError(
            f"Graines d'evaluation {bad} < {EVAL_SEED_MIN} : reservees a l'entrainement (cahier 6.3)"
        )


def check_train_seed(seed: int) -> None:
    if not 0 <= seed <= TRAIN_SEED_MAX:
        raise ValueError(f"Graine d'entrainement {seed} hors de 0-{TRAIN_SEED_MAX} (cahier 6.3)")


def library_versions(names: Sequence[str] = ("torch", "stable_baselines3", "gymnasium", "numpy")) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
