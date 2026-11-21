"""Metriques systeme (cahier des charges, section 8.1), communes a la simulation et au reel."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def jain_index(values: Sequence[float]) -> float:
    """Indice de Jain (somme x)^2 / (n * somme x^2) : 1 = parfaitement equilibre."""

    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.any(x):
        return 1.0
    return float(x.sum() ** 2 / (x.size * np.sum(x**2)))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def summarize_arrays(
    *,
    latency_ms: np.ndarray,
    deadline_ms: np.ndarray,
    rejected: np.ndarray,
    node_index: np.ndarray,
    energy_j: np.ndarray,
    cost: np.ndarray,
    slot_ids: Sequence[str],
    duration_s: float,
    idle_energy_j: float = 0.0,
    jain_cpu: float = float("nan"),
    reward_sum: float = float("nan"),
) -> dict:
    """Calcule toutes les metriques d'un run a partir des resultats par requete.

    ``latency_ms`` n'est pris en compte que pour les requetes acceptees. Un
    rejet compte comme une echeance manquee.
    """

    n = len(latency_ms)
    accepted = ~rejected.astype(bool)
    latency = latency_ms[accepted]
    met = accepted & (latency_ms <= deadline_ms)
    dyn = float(energy_j[accepted].sum())
    total_cost = float(cost[accepted].sum())
    summary = {
        "n_requests": int(n),
        "n_rejected": int(n - accepted.sum()),
        "rejection_rate": float((n - accepted.sum()) / n) if n else float("nan"),
        "latency_mean_ms": float(latency.mean()) if len(latency) else float("nan"),
        "latency_p95_ms": percentile(latency, 95),
        "latency_p99_ms": percentile(latency, 99),
        "deadline_rate": float(met.sum() / n) if n else float("nan"),
        "throughput_rps": float(accepted.sum() / duration_s) if duration_s > 0 else float("nan"),
        "energy_fog_dyn_j_per_req": dyn / n if n else float("nan"),
        "energy_fog_total_j": dyn + idle_energy_j,
        "energy_fog_j_per_req": (dyn + idle_energy_j) / n if n else float("nan"),
        "cloud_cost": total_cost,
        "cloud_cost_per_req": total_cost / n if n else float("nan"),
        "jain_cpu": jain_cpu,
        "reward_sum": reward_sum,
    }
    for i, slot in enumerate(slot_ids):
        summary[f"share_{slot}"] = float(np.mean(node_index == i)) if n else float("nan")
    return summary


class EpisodeMetrics:
    """Accumule les resultats par requete d'un episode simule."""

    def __init__(self, slot_ids: Sequence[str]) -> None:
        self.slot_ids = tuple(slot_ids)
        self.latency_ms: list[float] = []
        self.deadline_ms: list[float] = []
        self.rejected: list[bool] = []
        self.node_index: list[int] = []
        self.energy_j: list[float] = []
        self.cost: list[float] = []
        self.reward_sum = 0.0

    def record(
        self,
        node_index: int,
        latency_ms: float,
        deadline_ms: float,
        rejected: bool,
        energy_j: float,
        cost: float,
        reward: float,
    ) -> None:
        self.node_index.append(node_index)
        self.latency_ms.append(latency_ms)
        self.deadline_ms.append(deadline_ms)
        self.rejected.append(rejected)
        self.energy_j.append(energy_j)
        self.cost.append(cost)
        self.reward_sum += reward

    def summary(
        self, duration_s: float, idle_energy_j: float, jain_cpu: Optional[float]
    ) -> dict:
        return summarize_arrays(
            latency_ms=np.asarray(self.latency_ms),
            deadline_ms=np.asarray(self.deadline_ms),
            rejected=np.asarray(self.rejected, dtype=bool),
            node_index=np.asarray(self.node_index),
            energy_j=np.asarray(self.energy_j),
            cost=np.asarray(self.cost),
            slot_ids=self.slot_ids,
            duration_s=duration_s,
            idle_energy_j=idle_energy_j,
            jain_cpu=float("nan") if jain_cpu is None else jain_cpu,
            reward_sum=self.reward_sum,
        )
