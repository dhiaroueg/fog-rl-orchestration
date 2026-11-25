"""Modele d'energie lineaire et batterie virtuelle.

Partage par le simulateur (``NodeSim``) et par le worker reel : les deux
appliquent exactement les memes equations (cahier des charges, section 5.3) :

    P(t) = P_idle + (P_max - P_idle) * U_cpu(t),    E = integrale de P(t) dt

Les VM ne fournissant aucune mesure physique, ce modele est une hypothese a
presenter comme telle dans le memoire.
"""

from __future__ import annotations

from typing import Optional

from ..config import BatteryConfig, NodeProfile


def power_w(profile: NodeProfile, utilization: float) -> float:
    """Puissance instantanee (W) pour une utilisation CPU dans [0, 1]."""

    u = min(max(utilization, 0.0), 1.0)
    return profile.p_idle_w + (profile.p_max_w - profile.p_idle_w) * u


def dynamic_energy_j(profile: NodeProfile, service_s: float) -> float:
    """Energie dynamique (J) d'un serveur occupe ``service_s`` secondes.

    Un serveur occupe represente 1/servers de l'utilisation CPU du noeud.
    """

    return (profile.p_max_w - profile.p_idle_w) / profile.servers * max(service_s, 0.0)


class BatteryModel:
    """Batterie virtuelle : decharge selon la puissance, recharge lente constante.

    Quand l'energie tombe a zero, le noeud s'eteint (il refuse les requetes et
    ne consomme plus) et ne redemarre qu'une fois la fraction
    ``restart_fraction`` atteinte par la seule recharge.
    """

    def __init__(self, config: BatteryConfig, initial_fraction: Optional[float] = None) -> None:
        fraction = config.initial_fraction if initial_fraction is None else initial_fraction
        self.config = config
        self.energy_j = config.capacity_j * min(max(fraction, 0.0), 1.0)
        self.on = self.energy_j > 0.0

    @property
    def fraction(self) -> float:
        return self.energy_j / self.config.capacity_j

    def step(self, dt_s: float, power_w_now: float) -> None:
        """Avance de ``dt_s`` secondes avec une puissance constante (0 si eteint)."""

        if dt_s <= 0:
            return
        cfg = self.config
        if self.on:
            self.energy_j += (cfg.recharge_w - power_w_now) * dt_s
            if self.energy_j <= 0.0:
                self.energy_j = 0.0
                self.on = False
        else:
            self.energy_j += cfg.recharge_w * dt_s
            if self.energy_j >= max(cfg.restart_fraction * cfg.capacity_j, 1e-9):
                self.on = True
        self.energy_j = min(self.energy_j, cfg.capacity_j)

    def drain(self, joules: float) -> None:
        """Retire une quantite d'energie ponctuelle (energie dynamique d'une requete)."""

        if not self.on or joules <= 0:
            return
        self.energy_j -= joules
        if self.energy_j <= 0.0:
            self.energy_j = 0.0
            self.on = False
