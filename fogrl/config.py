"""Chargement et validation des configurations YAML (noeuds, requetes, scenarios, env).

Une seule source de verite decrit les noeuds : ``configs/nodes/*.yaml``. Elle
sert au simulateur (vitesse, energie, batterie), aux baselines (estimation de
la latence de Greedy), au deploiement Docker (cpus, memoire, netem) et au
worker reel (energie et batterie virtuelles).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

from .state_spec import REQUEST_TYPES, Normalization

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PathLike = Union[str, Path]


def resolve_path(path: PathLike) -> Path:
    """Resout un chemin relatif : d'abord le dossier courant, puis la racine du depot."""

    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(path: PathLike) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} doit contenir un dictionnaire YAML")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Fusion recursive : ``override`` l'emporte sur ``base``."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build(cls, data: Mapping[str, Any], where: str):
    """Construit une dataclass depuis un dictionnaire en refusant les cles inconnues."""

    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{where} : cles inconnues {sorted(unknown)} (attendu : {sorted(allowed)})")
    return cls(**data)


# --------------------------------------------------------------------------- noeuds


@dataclass(frozen=True)
class BatteryConfig:
    """Batterie virtuelle a recharge lente (fogA et fogB dans le cahier)."""

    capacity_j: float
    recharge_w: float
    restart_fraction: float = 0.25
    initial_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_j <= 0 or self.recharge_w < 0:
            raise ValueError("Batterie : capacity_j > 0 et recharge_w >= 0 requis")
        if not 0.0 <= self.restart_fraction <= 1.0 or not 0.0 <= self.initial_fraction <= 1.0:
            raise ValueError("Batterie : les fractions doivent etre dans [0, 1]")


@dataclass(frozen=True)
class NodeProfile:
    """Caracteristiques d'un noeud (hypotheses de modele, a reviser apres calibration)."""

    node_id: str
    kind: str = "fog"  # "fog" ou "cloud"
    speed: float = 1.0  # vitesse relative par serveur (1.0 = duree nominale des requetes)
    servers: int = 1  # requetes traitees en parallele
    rtt_ms: float = 20.0  # latence reseau aller-retour moyenne
    rtt_sigma: float = 0.1  # gigue relative (ecart-type log-normal)
    service_cv: float = 0.1  # variabilite residuelle du temps de service
    contention: float = 0.3  # inflation du temps de service a pleine charge
    p_idle_w: float = 0.0
    p_max_w: float = 0.0
    cost_per_request: float = 0.0
    max_queue: int = 100  # au-dela, la requete est rejetee
    battery: Optional[BatteryConfig] = None
    # Deploiement (scripts/gen_compose.py)
    cpus: float = 1.0
    mem_mb: int = 512
    vm: str = "vm1"
    port: int = 8000

    def __post_init__(self) -> None:
        if self.kind not in ("fog", "cloud"):
            raise ValueError(f"{self.node_id} : kind doit etre 'fog' ou 'cloud'")
        if self.speed <= 0 or self.servers < 1 or self.rtt_ms < 0 or self.max_queue < 1:
            raise ValueError(f"{self.node_id} : speed > 0, servers >= 1, rtt_ms >= 0, max_queue >= 1 requis")
        if self.p_max_w < self.p_idle_w or self.p_idle_w < 0:
            raise ValueError(f"{self.node_id} : 0 <= p_idle_w <= p_max_w requis")

    @property
    def is_cloud(self) -> bool:
        return self.kind == "cloud"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeProfile":
        data = dict(data)
        if "id" in data:
            data["node_id"] = data.pop("id")
        battery = data.pop("battery", None)
        profile = _build(cls, {**data, "battery": None}, f"noeud {data.get('node_id')}")
        if battery:
            profile = _replace(profile, battery=_build(BatteryConfig, battery, f"batterie {profile.node_id}"))
        return profile


def _replace(obj, **changes):
    values = {f.name: getattr(obj, f.name) for f in fields(obj)}
    values.update(changes)
    return type(obj)(**values)


@dataclass(frozen=True)
class RequestProfile:
    name: str
    work_ms: float  # travail nominal a la vitesse 1.0
    deadline_ms: float
    weight: float = 1.0
    work_cv: float = 0.2  # variabilite de la taille de la requete

    def __post_init__(self) -> None:
        if self.name not in REQUEST_TYPES:
            raise ValueError(f"Type de requete {self.name!r} inconnu ; attendu {REQUEST_TYPES}")
        if self.work_ms <= 0 or self.deadline_ms <= 0 or self.weight < 0:
            raise ValueError(f"{self.name} : work_ms, deadline_ms > 0 et weight >= 0 requis")


def load_nodes(path: PathLike) -> tuple[list[NodeProfile], dict[str, tuple[str, ...]]]:
    """Retourne (profils dans l'ordre du fichier, configurations nommees)."""

    data = load_yaml(path)
    profiles = [NodeProfile.from_dict(item) for item in data.get("nodes", [])]
    ids = [p.node_id for p in profiles]
    if not profiles or len(set(ids)) != len(ids):
        raise ValueError(f"{path} : nodes doit contenir des identifiants uniques")
    configurations = {
        name: tuple(members) for name, members in (data.get("configurations") or {}).items()
    }
    for name, members in configurations.items():
        missing = set(members) - set(ids)
        if missing:
            raise ValueError(f"Configuration {name} : noeuds inconnus {sorted(missing)}")
    return profiles, configurations


def load_requests(path: PathLike) -> tuple[RequestProfile, ...]:
    data = load_yaml(path).get("requests", {})
    profiles = tuple(
        RequestProfile(name=name, **params) for name, params in data.items()
    )
    if not profiles:
        raise ValueError(f"{path} : aucune requete definie")
    return profiles


def mean_work_ms(requests: Sequence[RequestProfile]) -> float:
    total = sum(r.weight for r in requests)
    return sum(r.weight * r.work_ms for r in requests) / total


def capacity_rps(profiles: Sequence[NodeProfile], requests: Sequence[RequestProfile]) -> float:
    """Debit maximal theorique (req/s) d'un groupe de noeuds, sans contention."""

    work = mean_work_ms(requests)
    return sum(p.servers * p.speed / work * 1000.0 for p in profiles)


# ----------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class ScenarioSpec:
    """Scenario de trafic. ``load`` = fraction de la capacite des noeuds fog actifs."""

    name: str
    type: str  # poisson | mmpp | diurnal | flash_crowd | trace
    load: float
    params: Mapping[str, Any] = field(default_factory=dict)

    KNOWN_TYPES = ("poisson", "mmpp", "diurnal", "flash_crowd", "trace")

    def __post_init__(self) -> None:
        if self.type not in self.KNOWN_TYPES:
            raise ValueError(f"Scenario {self.name} : type {self.type!r} inconnu ({self.KNOWN_TYPES})")
        if self.load <= 0:
            raise ValueError(f"Scenario {self.name} : load doit etre > 0")


def load_scenarios(path: PathLike) -> dict[str, ScenarioSpec]:
    data = load_yaml(path).get("scenarios", {})
    scenarios = {}
    for name, params in data.items():
        params = dict(params)
        scenarios[name] = ScenarioSpec(
            name=name, type=params.pop("type"), load=float(params.pop("load")), params=params
        )
    return scenarios


# --------------------------------------------------------------------- recompense


@dataclass(frozen=True)
class RewardConfig:
    """r = -(w1 T/Tref + w2 E/Eref + w3 C/Cref + w4 [echeance] + w5 [rejet])."""

    w1: float = 1.0
    w2: float = 0.0
    w3: float = 0.0
    w4: float = 2.0
    w5: float = 5.0
    t_ref_ms: float = 200.0
    e_ref_j: float = 1.0
    c_ref: float = 1.0
    # La latence est plafonnee a latency_cap_factor x echeance : un rejet ne
    # peut jamais couter moins qu'une reponse tres tardive (sinon l'agent
    # apprendrait a se faire rejeter volontairement).
    latency_cap_factor: float = 3.0

    def __post_init__(self) -> None:
        if min(self.w1, self.w2, self.w3, self.w4, self.w5) < 0:
            raise ValueError("Les poids de recompense doivent etre >= 0")
        if min(self.t_ref_ms, self.e_ref_j, self.c_ref, self.latency_cap_factor) <= 0:
            raise ValueError("Les references de recompense doivent etre > 0")


@dataclass(frozen=True)
class RandomizationConfig:
    """Randomisation de domaine (niveau N3) : variations par episode."""

    enabled: bool = False
    service_scale: tuple[float, float] = (0.8, 1.2)  # temps de service +-20 %
    rtt_scale: tuple[float, float] = (0.7, 1.3)  # latence +-30 %
    rate_scale: tuple[float, float] = (0.8, 1.2)  # debit
    battery_initial: tuple[float, float] = (0.6, 1.0)

    def __post_init__(self) -> None:
        for name in ("service_scale", "rtt_scale", "rate_scale", "battery_initial"):
            low, high = getattr(self, name)
            if not 0 < low <= high:
                raise ValueError(f"randomization.{name} : 0 < min <= max requis")


@dataclass(frozen=True)
class Outage:
    """Panne simulee d'un noeud entre deux instants (secondes d'episode)."""

    node: str
    start_s: float
    end_s: float


# --------------------------------------------------------------------------- env


@dataclass(frozen=True)
class EnvConfig:
    pool: tuple[NodeProfile, ...]
    configurations: Mapping[str, tuple[str, ...]]
    configuration_choices: tuple[str, ...]
    layout: str
    requests: tuple[RequestProfile, ...]
    scenarios: tuple[ScenarioSpec, ...]
    episode_requests: int
    poll_interval_ms: float
    rate_window_s: float
    service_model: str
    calibration_file: Optional[str]
    normalization: Normalization
    reward: RewardConfig
    randomization: RandomizationConfig
    outages: tuple[Outage, ...]

    # -- construction -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: PathLike, overrides: Optional[Mapping[str, Any]] = None) -> "EnvConfig":
        data = load_yaml(path)
        if overrides:
            data = deep_merge(data, overrides)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvConfig":
        data = dict(data)
        pool, configurations = load_nodes(data.pop("nodes_file", "configs/nodes/pool6.yaml"))
        overrides = data.pop("node_overrides", None) or {}
        unknown_nodes = set(overrides) - {p.node_id for p in pool}
        if unknown_nodes:
            raise ValueError(f"node_overrides : noeuds inconnus {sorted(unknown_nodes)}")
        pool = [
            NodeProfile.from_dict(deep_merge(_node_as_dict(p), overrides.get(p.node_id, {})))
            for p in pool
        ]
        requests = load_requests(data.pop("requests_file", "configs/requests.yaml"))
        known_scenarios = load_scenarios(data.pop("scenarios_file", "configs/scenarios.yaml"))
        names = data.pop("scenarios", ["medium"])
        missing = [n for n in names if n not in known_scenarios]
        if missing:
            raise ValueError(f"Scenarios inconnus : {missing} (disponibles : {sorted(known_scenarios)})")

        choice = data.pop("configuration", "n4")
        choices = tuple([choice] if isinstance(choice, str) else choice)
        unknown_conf = [c for c in choices if c not in configurations]
        if unknown_conf:
            raise ValueError(f"Configurations inconnues : {unknown_conf} (disponibles : {sorted(configurations)})")

        layout = data.pop("layout", "compact")
        if layout not in ("compact", "padded"):
            raise ValueError("layout doit etre 'compact' ou 'padded'")
        if layout == "compact" and len(choices) != 1:
            raise ValueError("La disposition compacte exige une seule configuration (voir layout: padded)")

        randomization = dict(data.pop("randomization", {}) or {})
        for key in ("service_scale", "rtt_scale", "rate_scale", "battery_initial"):
            if key in randomization:
                randomization[key] = tuple(randomization[key])
        outages = tuple(_build(Outage, o, "outage") for o in data.pop("outages", []) or [])

        config = cls(
            pool=tuple(pool),
            configurations=configurations,
            configuration_choices=choices,
            layout=layout,
            requests=requests,
            scenarios=tuple(known_scenarios[n] for n in names),
            episode_requests=int(data.pop("episode_requests", 2000)),
            poll_interval_ms=float(data.pop("poll_interval_ms", 200.0)),
            rate_window_s=float(data.pop("rate_window_s", 1.0)),
            service_model=data.pop("service_model", "analytic"),
            calibration_file=data.pop("calibration_file", None),
            normalization=Normalization(**(data.pop("normalization", {}) or {})),
            reward=_build(RewardConfig, data.pop("reward", {}) or {}, "reward"),
            randomization=_build(RandomizationConfig, randomization, "randomization"),
            outages=outages,
        )
        if data:
            raise ValueError(f"Cles de configuration inconnues : {sorted(data)}")
        if config.service_model not in ("analytic", "calibrated"):
            raise ValueError("service_model doit etre 'analytic' ou 'calibrated'")
        if config.episode_requests < 1 or config.poll_interval_ms < 0:
            raise ValueError("episode_requests >= 1 et poll_interval_ms >= 0 requis")
        return config

    # -- noeuds -------------------------------------------------------------

    def profile(self, node_id: str) -> NodeProfile:
        for profile in self.pool:
            if profile.node_id == node_id:
                return profile
        raise KeyError(node_id)

    @property
    def slot_ids(self) -> tuple[str, ...]:
        """Identifiants des emplacements de l'observation, dans l'ordre des actions."""

        if self.layout == "padded":
            return tuple(p.node_id for p in self.pool)
        return self.configurations[self.configuration_choices[0]]

    def active_ids(self, configuration: str) -> tuple[str, ...]:
        members = set(self.configurations[configuration])
        return tuple(i for i in self.slot_ids if i in members)

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable (sauvegardee dans contract.json)."""

        return json.loads(json.dumps(self.summary(), default=_json_default))

    def summary(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "slot_ids": list(self.slot_ids),
            "configurations": list(self.configuration_choices),
            "scenarios": [s.name for s in self.scenarios],
            "episode_requests": self.episode_requests,
            "poll_interval_ms": self.poll_interval_ms,
            "service_model": self.service_model,
            "normalization": self.normalization.__dict__,
            "reward": self.reward.__dict__,
            "randomization": self.randomization.__dict__,
            "nodes": {p.node_id: _node_as_dict(p) for p in self.pool},
        }

    def digest(self) -> str:
        payload = json.dumps(self.summary(), sort_keys=True, default=_json_default)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _node_as_dict(profile: NodeProfile) -> dict[str, Any]:
    data = {f.name: getattr(profile, f.name) for f in fields(profile)}
    data["id"] = data.pop("node_id")
    if profile.battery is not None:
        data["battery"] = {f.name: getattr(profile.battery, f.name) for f in fields(profile.battery)}
    return data


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Non serialisable : {type(value)}")
