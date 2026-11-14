"""Générateur de trafic partagé par le simulateur, l'évaluation et le cluster réel."""
from .schema import TrafficConfig, TypeSpec, ZONES
from .traffic import generate_arrivals, arrival_rate, mean_work_ms

__all__ = ["TrafficConfig", "TypeSpec", "ZONES",
           "generate_arrivals", "arrival_rate", "mean_work_ms"]