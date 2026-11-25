"""FogEnv : environnement Gymnasium du routage de requetes en fog computing.

Un pas de decision = l'arrivee d'une requete. L'action choisit le noeud
(index = emplacement dans ``config.slot_ids``). L'agent observe l'etat tel que
l'orchestrateur le voit : mesures des noeuds rafraichies toutes les
``poll_interval_ms`` (200 ms dans le cahier des charges), donc potentiellement
perimees, alors que les consequences de la decision sont calculees sur l'etat reel.

Un episode compte ``episode_requests`` requetes (2000 en entrainement, 5000 en
evaluation) et se termine par une troncature (limite de temps, pas un etat
terminal). Les metriques de l'episode sont dans ``info["episode_metrics"]``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import EnvConfig
from ..metrics import EpisodeMetrics, jain_index
from ..reward import compute_reward
from ..state_spec import (
    DecisionContext,
    NodeState,
    RequestState,
    absent_node,
    encode_state,
    observation_dim,
)
from ..traffic import RequestStream, base_rate_rps, generate_stream
from .node import NodeSim, NodeSnapshot, SubmitResult
from .service import build_service_model


class FogEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Union[EnvConfig, Mapping[str, Any], str, Path],
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if isinstance(config, (str, Path)):
            config = EnvConfig.from_yaml(config, overrides)
        elif isinstance(config, Mapping):
            config = EnvConfig.from_dict(config)
        self.config: EnvConfig = config
        self.slot_ids = config.slot_ids
        self.n_slots = len(self.slot_ids)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(observation_dim(self.n_slots),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_slots)
        self._service_model = build_service_model(
            config.service_model, config.calibration_file, config.requests
        )
        self._deadline = {r.name: r.deadline_ms for r in config.requests}
        self._poll_s = config.poll_interval_ms / 1000.0

        self.nodes: list[Optional[NodeSim]] = []
        self.stream: Optional[RequestStream] = None
        self.episode_info: dict[str, Any] = {}
        self._i = 0
        self._t = 0.0
        self._last_poll = -math.inf
        self._cache: list[Optional[NodeSnapshot]] = []
        self._mask = np.ones(self.n_slots, dtype=bool)
        self._raw: Optional[tuple[list[NodeState], RequestState, float]] = None
        self._metrics = EpisodeMetrics(self.slot_ids)
        self._last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._needs_reset = True

    # ------------------------------------------------------------------ reset

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        cfg, rng = self.config, self.np_random

        choices = cfg.configuration_choices
        configuration = options.get("configuration") or (
            choices[0] if len(choices) == 1 else choices[int(rng.integers(len(choices)))]
        )
        if configuration not in cfg.configurations:
            raise ValueError(f"Configuration inconnue : {configuration}")
        active_ids = cfg.active_ids(configuration)

        scenarios = {s.name: s for s in cfg.scenarios}
        if options.get("scenario"):
            if options["scenario"] not in scenarios:
                raise ValueError(f"Scenario inconnu pour cette config : {options['scenario']}")
            scenario = scenarios[options["scenario"]]
        else:
            scenario = (
                cfg.scenarios[0]
                if len(cfg.scenarios) == 1
                else cfg.scenarios[int(rng.integers(len(cfg.scenarios)))]
            )

        dr = cfg.randomization

        def draw(bounds: tuple[float, float]) -> float:
            return float(rng.uniform(*bounds)) if dr.enabled else 1.0

        self.nodes = []
        for slot in self.slot_ids:
            if slot not in active_ids:
                self.nodes.append(None)
                continue
            profile = cfg.profile(slot)
            node_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            battery = (
                float(rng.uniform(*dr.battery_initial))
                if dr.enabled and profile.battery is not None
                else None
            )
            self.nodes.append(
                NodeSim(
                    profile,
                    self._service_model,
                    node_rng,
                    service_scale=draw(dr.service_scale),
                    rtt_scale=draw(dr.rtt_scale),
                    battery_fraction=battery,
                )
            )

        active_profiles = [n.profile for n in self.nodes if n is not None]
        rate = base_rate_rps(scenario, active_profiles, cfg.requests) * draw(dr.rate_scale)
        self.stream = generate_stream(
            scenario, rate, cfg.episode_requests, cfg.requests, int(rng.integers(0, 2**31 - 1))
        )
        self.episode_info = {
            "configuration": configuration,
            "scenario": scenario.name,
            "base_rate_rps": rate,
            "n_active": len(active_profiles),
        }

        self._i = 0
        self._t = float(self.stream.times_s[0])
        self._last_poll = -math.inf
        self._metrics = EpisodeMetrics(self.slot_ids)
        self._needs_reset = False
        self._advance_world()
        obs = self._observe()
        self._last_obs = obs
        return obs, {"action_mask": self._mask.copy(), **self.episode_info}

    # ------------------------------------------------------------------- step

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._needs_reset or self.stream is None:
            raise RuntimeError("Appelez reset() avant step() (episode termine ou non demarre)")
        action = int(action)
        i, t, cfg = self._i, self._t, self.config
        req_type = self.stream.type_of(i)
        work_ms = float(self.stream.work_ms[i])
        deadline = self._deadline[req_type]

        node = self.nodes[action] if 0 <= action < self.n_slots else None
        result = node.submit(t, req_type, work_ms) if node is not None else SubmitResult(False, "absent")
        rejected = not result.accepted
        reward = compute_reward(
            cfg.reward, result.latency_ms, deadline, result.energy_j, result.cost, rejected
        )
        self._metrics.record(
            action, result.latency_ms, deadline, rejected, result.energy_j, result.cost, reward
        )
        info: dict[str, Any] = {
            "node": self.slot_ids[action] if 0 <= action < self.n_slots else None,
            "request_type": req_type,
            "rejected": rejected,
            "reason": result.reason,
            "latency_ms": None if rejected else result.latency_ms,
            "deadline_met": (not rejected) and result.latency_ms <= deadline,
            "energy_j": result.energy_j,
            "cost": result.cost,
        }

        self._i += 1
        truncated = self._i >= len(self.stream)
        if truncated:
            self._needs_reset = True
            info["episode_metrics"] = self._episode_summary()
            info["action_mask"] = self._mask.copy()
            return self._last_obs.copy(), float(reward), False, True, info

        self._t = float(self.stream.times_s[self._i])
        self._advance_world()
        obs = self._observe()
        self._last_obs = obs
        info["action_mask"] = self._mask.copy()
        return obs, float(reward), False, False, info

    # ---------------------------------------------------------------- interne

    def _advance_world(self) -> None:
        t = self._t
        for slot, node in zip(self.slot_ids, self.nodes):
            if node is None:
                continue
            for outage in self.config.outages:
                if outage.node == slot:
                    node.down = outage.start_s <= t < outage.end_s
            node.advance(t)
        if t - self._last_poll >= self._poll_s:
            self._cache = [None if n is None else self._poll(n, t) for n in self.nodes]
            self._last_poll = t

    @staticmethod
    def _poll(node: NodeSim, t: float) -> NodeSnapshot:
        node.probe_rtt()
        return node.snapshot(t)

    def _recent_rate(self) -> float:
        window = self.config.rate_window_s
        times = self.stream.times_s
        first = int(np.searchsorted(times, self._t - window, side="right"))
        return (self._i + 1 - first) / max(min(window, self._t), 0.1)

    def _observe(self) -> np.ndarray:
        norm = self.config.normalization
        states: list[NodeState] = []
        for slot, node, snap in zip(self.slot_ids, self.nodes, self._cache):
            if node is None or snap is None:
                states.append(absent_node(slot, norm))
            else:
                # Les mesures viennent du sondage (perimees de <= poll_interval_ms), mais
                # ``in_flight`` est compte par l'orchestrateur lui-meme (requetes envoyees
                # et non terminees) : il est donc a jour, comme dans le prototype reel.
                states.append(
                    NodeState(
                        slot,
                        snap.queue_ms,
                        snap.cpu_load,
                        snap.net_rtt_ms,
                        snap.battery_fraction,
                        snap.available,
                        node.in_flight(self._t),
                    )
                )
        if not any(s.available for s in states):
            # Mesures perimees : tous les noeuds presents paraissent indisponibles.
            states = [
                s if node is None else replace(s, available=True)
                for s, node in zip(states, self.nodes)
            ]
        i = self._i
        request_type = self.stream.type_of(i)
        request = RequestState(request_type, float(self.stream.work_ms[i]), self._deadline[request_type])
        rate = self._recent_rate()
        obs, mask = encode_state(states, request, rate, norm)
        self._mask = mask
        self._raw = (states, request, rate)
        return obs

    def _episode_summary(self) -> dict:
        duration = float(self.stream.times_s[-1])
        fog = [n for n in self.nodes if n is not None and not n.profile.is_cloud]
        idle = sum(n.idle_energy_j() for n in fog)
        jain = jain_index([n.utilization(duration) for n in fog]) if fog else None
        summary = self._metrics.summary(duration, idle, jain)
        summary.update(self.episode_info)
        return summary

    # ------------------------------------------------------------- public API

    def action_masks(self) -> np.ndarray:
        """Masque des actions valides pour la requete courante (compatible sb3_contrib)."""

        return self._mask.copy()

    def raw_state(self) -> tuple[list[NodeState], RequestState, DecisionContext]:
        """Etat brut (non normalise) de la decision courante, pour les baselines."""

        if self._raw is None:
            raise RuntimeError("Appelez reset() d'abord")
        nodes, request, rate = self._raw
        return list(nodes), request, DecisionContext(now_s=self._t, arrival_rate_rps=rate)
