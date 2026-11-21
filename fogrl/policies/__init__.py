"""Politiques de routage. ``RLPolicy`` (torch) est importee a la demande depuis ``fogrl.policies.rl``."""

from .base import Policy
from .baselines import (
    BASELINES,
    GreedyPolicy,
    LeastConnectionsPolicy,
    RandomPolicy,
    RoundRobinPolicy,
    StaticThresholdPolicy,
    make_baseline,
)

__all__ = [
    "BASELINES",
    "GreedyPolicy",
    "LeastConnectionsPolicy",
    "Policy",
    "RandomPolicy",
    "RoundRobinPolicy",
    "StaticThresholdPolicy",
    "make_baseline",
]
