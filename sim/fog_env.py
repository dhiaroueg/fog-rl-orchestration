"""Environnement Gymnasium fog multi-noeuds pour le niveau N1.

Ce premier niveau simule les files, le temps de service, le reseau et les
echeances. L'energie, le cout, les batteries dynamiques et les rejets seront
ajoutes dans les niveaux suivants apres validation des mesures.

Les profils sont lus depuis experiments/configs/profiles.yaml. Le champ
netem_delay_ms est un delai configure dans un sens ; en attendant une mesure
RTT, on utilise 2 * netem_delay_ms comme approximation provisoire.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from generator.schema import TrafficConfig
from generator.traffic import generate_arrivals
from state_spec import NodeState, RequestState, encode_state, observation_dim


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_PATH = ROOT / "experiments" / "configs" / "profiles.yaml"


@dataclass(frozen=True)
class NodeProfile:
    """Parametres fixes d'un noeud utilises par le simulateur."""

    node_id: str
    speed: float
    concurrency: int
    net_rtt_ms: float
    battery: bool


def load_node_profiles(path: Path = DEFAULT_PROFILES_PATH) -> tuple[NodeProfile, ...]:
    """Charge les noeuds du YAML, dans l'ordre declare dans le fichier."""

    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    profiles = []
    for node_id, raw in config["nodes"].items():
        delay_ms = float(raw["netem_delay_ms"])
        profiles.append(
            NodeProfile(
                node_id=node_id,
                speed=float(raw["speed"]),
                concurrency=int(raw["concurrency"]),
                net_rtt_ms=float(raw.get("net_rtt_ms", 2 * delay_ms)),
                battery=bool(raw.get("battery", False)),
            )
        )

    if not profiles:
        raise ValueError("Aucun profil de noeud dans le fichier YAML")
    if any(p.speed <= 0 or p.concurrency <= 0 or p.net_rtt_ms < 0 for p in profiles):
        raise ValueError("Vitesse/concurrence doivent etre positives et RTT >= 0")
    return tuple(profiles)


class FogEnv(gym.Env):
    """Route des requetes vers les noeuds declares dans les profils.

    Action 0 correspond au premier noeud de profiles.yaml, action 1 au
    deuxieme, etc. L'observation suit exactement state_spec.encode_state().
    Un modele entraine avec un nombre de noeuds donne conserve cette taille.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_requests: int = 2000,
        traffic_config: Optional[TrafficConfig] = None,
        profiles_path: Path = DEFAULT_PROFILES_PATH,
        profiles: Optional[Sequence[NodeProfile]] = None,
        deadline_penalty: float = 2.0,
        latency_reference_ms: float = 1000.0,
    ) -> None:
        super().__init__()
        if n_requests <= 0:
            raise ValueError("n_requests doit etre > 0")
        if deadline_penalty < 0 or latency_reference_ms <= 0:
            raise ValueError("Penalite >= 0 et reference de latence > 0 requises")

        self.profiles = (
            tuple(profiles) if profiles is not None else load_node_profiles(profiles_path)
        )
        if not self.profiles:
            raise ValueError("Il faut au moins un profil de noeud")
        if len({p.node_id for p in self.profiles}) != len(self.profiles):
            raise ValueError("Les identifiants de noeuds doivent etre uniques")

        self.n_requests = n_requests
        self.traffic_config = traffic_config or TrafficConfig(
            rho=0.6,
            n_requests=n_requests,
        )
        if self.traffic_config.n_requests != n_requests:
            raise ValueError("traffic_config.n_requests doit egaler n_requests")
        self.deadline_penalty = float(deadline_penalty)
        self.latency_reference_ms = float(latency_reference_ms)
        self.node_ids = tuple(profile.node_id for profile in self.profiles)

        self.action_space = spaces.Discrete(len(self.profiles))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_dim(len(self.profiles)),),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        self.arrivals = generate_arrivals(self.traffic_config, seed=seed)
        self.index = 0
        self.backlog_work_ms = np.zeros(len(self.profiles), dtype=np.float64)
        return self._observation(), {"node_order": self.node_ids}

    def _recent_arrival_rate_rps(self, current_ms: float) -> float:
        start = max(0.0, current_ms - 1000.0)
        times = self.arrivals["t_arrival_ms"].to_numpy()
        left = int(np.searchsorted(times, start, side="left"))
        right = int(np.searchsorted(times, current_ms, side="right"))
        return float(right - left)

    def _observation(self) -> np.ndarray:
        row = self.arrivals.iloc[self.index]
        nodes = []
        for i, profile in enumerate(self.profiles):
            capacity = profile.speed * profile.concurrency
            # Proxy de pression calculee depuis le travail restant, pas une
            # mesure de CPU de VM. La calibration pourra remplacer ce proxy.
            cpu_pressure = min(1.0, self.backlog_work_ms[i] / (100.0 * capacity))
            nodes.append(
                NodeState(
                    node_id=profile.node_id,
                    queue_ms=float(self.backlog_work_ms[i]),
                    cpu_load=cpu_pressure,
                    net_rtt_ms=profile.net_rtt_ms,
                    battery_fraction=1.0,
                    available=True,
                )
            )

        request = RequestState(
            request_type=str(row["type"]),
            work_ms=float(row["work_ms"]),
            deadline_ms=float(row["deadline_ms"]),
        )
        return encode_state(
            nodes=nodes,
            request=request,
            arrival_rate_rps=self._recent_arrival_rate_rps(float(row["t_arrival_ms"])),
        )[0]

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")
        if self.index >= self.n_requests:
            raise RuntimeError("Episode termine : appelez reset() avant step()")

        row = self.arrivals.iloc[self.index]
        profile = self.profiles[int(action)]
        i = int(action)
        work_ms = float(row["work_ms"])
        arrival_ms = float(row["t_arrival_ms"])

        wait_ms = self.backlog_work_ms[i] / (profile.speed * profile.concurrency)
        service_ms = work_ms / profile.speed
        network_ms = profile.net_rtt_ms
        response_ms = wait_ms + service_ms + network_ms
        missed_deadline = response_ms > float(row["deadline_ms"])

        self.backlog_work_ms[i] += work_ms
        next_index = self.index + 1
        terminated = next_index >= self.n_requests
        if not terminated:
            next_arrival_ms = float(self.arrivals.iloc[next_index]["t_arrival_ms"])
            elapsed_ms = max(0.0, next_arrival_ms - arrival_ms)
            for j, other in enumerate(self.profiles):
                capacity = other.speed * other.concurrency
                self.backlog_work_ms[j] = max(
                    0.0,
                    self.backlog_work_ms[j] - elapsed_ms * capacity,
                )

        reward = -response_ms / self.latency_reference_ms
        if missed_deadline:
            reward -= self.deadline_penalty

        self.index = next_index
        info = {
            "node": profile.node_id,
            "request_type": str(row["type"]),
            "work_ms": work_ms,
            "wait_ms": wait_ms,
            "service_ms": service_ms,
            "network_ms": network_ms,
            "response_ms": response_ms,
            "deadline_ms": float(row["deadline_ms"]),
            "deadline_missed": missed_deadline,
            "node_order": self.node_ids,
        }
        observation = self._observation() if not terminated else np.zeros(
            observation_dim(len(self.profiles)), dtype=np.float32
        )
        return observation, float(reward), terminated, False, info
