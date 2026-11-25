"""Configuration de l'orchestrateur (YAML + variables d'environnement).

Le fichier ``env_file`` (par defaut ``configs/env/eval.yaml``) reste la SOURCE UNIQUE des
profils de noeuds, des requetes (echeances), de la normalisation et de la recompense :
simulation et prototype lisent les memes valeurs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Optional

from ..config import EnvConfig, NodeProfile, load_yaml, resolve_path

DEFAULT_CONFIG = "configs/deploy/orchestrator.yaml"


@dataclass(frozen=True)
class OrchestratorSettings:
    env_file: str = "configs/env/eval.yaml"
    configuration: str = "n4"  # noeuds actifs (voir `configurations` dans configs/nodes/*.yaml)
    urls: Mapping[str, str] = field(default_factory=dict)  # node_id -> URL ; defaut http://<node_id en minuscules>:<port>
    policy: str = "least_connections"  # baseline ou rl:<repertoire du modele>
    decision_budget_ms: float = 20.0  # EF-07 : au-dela, repli sur least-connections
    poll_interval_ms: Optional[float] = None  # defaut : celui de env_file (200 ms)
    poll_timeout_ms: float = 400.0
    rate_window_s: Optional[float] = None
    request_timeout_s: float = 10.0
    failure_threshold: int = 2  # echecs consecutifs avant de declarer un noeud indisponible
    stale_after_s: Optional[float] = None  # mesure plus vieille que ca => noeud indisponible
    policy_failure_threshold: int = 20  # echecs consecutifs de la politique avant de s'en passer
    policy_cooldown_s: float = 5.0
    warmup_decisions: int = 5
    log_file: Optional[str] = None  # journal JSONL des decisions

    def __post_init__(self) -> None:
        if self.decision_budget_ms <= 0 or self.request_timeout_s <= 0 or self.poll_timeout_ms <= 0:
            raise ValueError("decision_budget_ms, request_timeout_s et poll_timeout_ms doivent etre > 0")
        if self.failure_threshold < 1 or self.policy_failure_threshold < 1:
            raise ValueError("Les seuils d'echecs doivent etre >= 1")

    # ------------------------------------------------------------ construction

    @classmethod
    def from_dict(cls, data: Mapping) -> "OrchestratorSettings":
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Cles de configuration inconnues : {sorted(unknown)} (attendu : {sorted(allowed)})")
        return cls(**data)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "OrchestratorSettings":
        """YAML (``ORCH_CONFIG``) puis surcharges ORCH_ENV_FILE, ORCH_POLICY, ORCH_CONFIGURATION, ORCH_NODE_URLS, DECISION_LOG."""

        env = os.environ if env is None else env
        path = env.get("ORCH_CONFIG", DEFAULT_CONFIG)
        data = load_yaml(path) if resolve_path(path).exists() else {}
        if env.get("ORCH_ENV_FILE"):
            data["env_file"] = env["ORCH_ENV_FILE"]
        if env.get("ORCH_POLICY"):
            data["policy"] = env["ORCH_POLICY"]
        if env.get("ORCH_CONFIGURATION"):
            data["configuration"] = env["ORCH_CONFIGURATION"]
        if env.get("DECISION_LOG"):
            data["log_file"] = env["DECISION_LOG"]
        if env.get("ORCH_NODE_URLS"):
            data["urls"] = parse_urls(env["ORCH_NODE_URLS"])
        return cls.from_dict(data)

    # ------------------------------------------------------------------ vues

    def env_config(self) -> EnvConfig:
        return EnvConfig.from_yaml(self.env_file)

    def resolve(self) -> "ResolvedSettings":
        config = self.env_config()
        if self.configuration not in config.configurations:
            raise ValueError(
                f"Configuration inconnue : {self.configuration!r} (disponibles : {sorted(config.configurations)})"
            )
        active = tuple(i for i in (p.node_id for p in config.pool) if i in config.configurations[self.configuration])
        profiles = {p.node_id: p for p in config.pool}
        urls = {}
        for node_id in active:
            urls[node_id] = self.urls.get(node_id) or f"http://{node_id.lower()}:{profiles[node_id].port}"
        poll_ms = self.poll_interval_ms if self.poll_interval_ms is not None else config.poll_interval_ms
        return ResolvedSettings(
            settings=self,
            config=config,
            profiles=profiles,
            active_ids=active,
            urls=urls,
            deadlines={r.name: r.deadline_ms for r in config.requests},
            poll_interval_s=poll_ms / 1000.0,
            rate_window_s=self.rate_window_s if self.rate_window_s is not None else config.rate_window_s,
            stale_after_s=self.stale_after_s if self.stale_after_s is not None else max(1.0, 5.0 * poll_ms / 1000.0),
        )


@dataclass(frozen=True)
class ResolvedSettings:
    """Reglages + valeurs deduites de ``env_file`` (noeuds actifs, URLs, echeances)."""

    settings: OrchestratorSettings
    config: EnvConfig
    profiles: Mapping[str, NodeProfile]
    active_ids: tuple[str, ...]
    urls: Mapping[str, str]
    deadlines: Mapping[str, float]
    poll_interval_s: float
    rate_window_s: float
    stale_after_s: float

    @property
    def normalization(self):
        return self.config.normalization


def parse_urls(text: str) -> dict[str, str]:
    """'fogA=http://h1:8001,fogB=http://h2:8002' -> dict."""

    urls = {}
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        node_id, sep, url = part.partition("=")
        if not sep or not url:
            raise ValueError(f"URL de noeud invalide : {part!r} (attendu : id=http://hote:port)")
        urls[node_id.strip()] = url.strip()
    return urls
