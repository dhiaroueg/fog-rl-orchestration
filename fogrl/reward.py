"""Recompense multi-objectif (cahier des charges, section 5.2).

    r = -( w1 T/Tref + w2 E/Eref + w3 C/Cref + w4 [T > echeance] + w5 [rejet] )

La latence est plafonnee a ``latency_cap_factor`` x echeance. Un rejet recoit
la latence plafonnee, la penalite d'echeance et la penalite de rejet : il ne
peut donc jamais etre plus avantageux qu'une reponse tres tardive.
"""

from __future__ import annotations

from .config import RewardConfig


def compute_reward(
    cfg: RewardConfig,
    latency_ms: float,
    deadline_ms: float,
    energy_j: float,
    cost: float,
    rejected: bool,
) -> float:
    cap = cfg.latency_cap_factor * deadline_ms
    effective_latency = cap if rejected else min(latency_ms, cap)
    missed = rejected or latency_ms > deadline_ms
    return -(
        cfg.w1 * effective_latency / cfg.t_ref_ms
        + cfg.w2 * energy_j / cfg.e_ref_j
        + cfg.w3 * cost / cfg.c_ref
        + cfg.w4 * float(missed)
        + cfg.w5 * float(rejected)
    )
