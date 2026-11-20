"""Structures de données du générateur de trafic."""
from dataclasses import dataclass, field
import math
from typing import Optional

ZONES = ("A", "B", "C")


@dataclass(frozen=True)
class TypeSpec:
    """Un type de requête (légère, moyenne, lourde)."""
    mean_work_ms: float
    deadline_ms: float
    share: float


DEFAULT_TYPES = {
    # share part dans la requete
    "light":  TypeSpec(mean_work_ms=5,   deadline_ms=100,  share=0.6),
    "medium": TypeSpec(mean_work_ms=40,  deadline_ms=300,  share=0.3),
    "heavy":  TypeSpec(mean_work_ms=150, deadline_ms=1000, share=0.1),
}


@dataclass(frozen=True)
class TrafficConfig:
    """Configuration complète d'un scénario de trafic."""
    # "poisson" remains an alias for backward compatibility with earlier configs.
    scenario: str = "constant"
    rho: float = 0.6
    capacity: float = 3.0
    n_requests: Optional[int] = 2000
    duration_ms: Optional[float] = None
    types: dict = field(default_factory=lambda: dict(DEFAULT_TYPES))
    work_cv: float = 0.3
    zone_weights: tuple = (0.34, 0.33, 0.33)

    # Two-state MMPP. Residence times are means in milliseconds.
    peak_low_mean_ms: float = 30_000.0
    peak_high_mean_ms: float = 10_000.0
    peak_low_multiplier: float = 0.5
    peak_high_multiplier: float = 2.5

    # A compressed day can be selected for short simulations by overriding period.
    diurnal_period_ms: float = 86_400_000.0
    diurnal_amplitude: float = 0.8
    diurnal_peak_phase: float = 0.5
    profile_resolution_ms: float = 1_000.0

    # Flash profile: baseline -> ramp -> peak plateau -> recovery -> baseline.
    flash_peak_multiplier: float = 5.0
    flash_start_fraction: float = 0.2
    flash_ramp_up_fraction: float = 0.15
    flash_plateau_fraction: float = 0.2
    flash_ramp_down_fraction: float = 0.15

    # Trace input: one count per row, optionally with one timestamp per bin.
    trace_path: Optional[str] = None
    trace_count_column: str = "invocations"
    trace_timestamp_column: Optional[str] = "timestamp"
    trace_interval_ms: float = 60_000.0

    def __post_init__(self):
        if self.scenario not in {
            "constant", "poisson", "peaks", "diurnal", "flash_crowd", "trace"
        }:
            raise ValueError(
                "scenario doit être parmi : constant, peaks, diurnal, flash_crowd, trace"
            )
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho doit être > 0")
        if not math.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("capacity doit être > 0")
        if (self.n_requests is None) == (self.duration_ms is None):
            raise ValueError("Définir exactement un des deux : n_requests ou duration_ms")
        if self.n_requests is not None and (
            isinstance(self.n_requests, bool) or not isinstance(self.n_requests, int)
            or self.n_requests <= 0
        ):
            raise ValueError("n_requests doit être > 0")
        if self.duration_ms is not None and (
            not math.isfinite(self.duration_ms) or self.duration_ms <= 0
        ):
            raise ValueError("duration_ms doit être > 0")
        if not math.isfinite(self.work_cv) or self.work_cv < 0:
            raise ValueError("work_cv doit être >= 0")
        if (
            len(self.zone_weights) != len(ZONES)
            or not all(math.isfinite(w) and w >= 0 for w in self.zone_weights)
            or abs(sum(self.zone_weights) - 1) > 1e-6
        ):
            raise ValueError("zone_weights : une valeur par zone, somme égale à 1")
        if not self.types or any(not isinstance(t, TypeSpec) for t in self.types.values()):
            raise ValueError("types doit contenir au moins un TypeSpec")
        if (
            any(not math.isfinite(t.share) or t.share < 0 for t in self.types.values())
            or abs(sum(t.share for t in self.types.values()) - 1) > 1e-6
        ):
            raise ValueError("La somme des parts des types doit valoir 1")
        for name, t in self.types.items():
            if (
                not math.isfinite(t.mean_work_ms) or t.mean_work_ms <= 0
                or not math.isfinite(t.deadline_ms) or t.deadline_ms <= 0
            ):
                raise ValueError(f"Type {name} : travail et échéance doivent être > 0")

        positive = {
            "peak_low_mean_ms": self.peak_low_mean_ms,
            "peak_high_mean_ms": self.peak_high_mean_ms,
            "diurnal_period_ms": self.diurnal_period_ms,
            "profile_resolution_ms": self.profile_resolution_ms,
            "trace_interval_ms": self.trace_interval_ms,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("Les durées de scénario doivent être finies et > 0")
        if (
            not math.isfinite(self.peak_low_multiplier) or self.peak_low_multiplier <= 0
            or not math.isfinite(self.peak_high_multiplier) or self.peak_high_multiplier <= 0
            or self.peak_high_multiplier <= self.peak_low_multiplier
        ):
            raise ValueError("Le multiplicateur haut MMPP doit dépasser le multiplicateur bas")
        if (
            not math.isfinite(self.diurnal_amplitude)
            or not 0 <= self.diurnal_amplitude < 1
            or not math.isfinite(self.diurnal_peak_phase)
            or not 0 <= self.diurnal_peak_phase <= 1
        ):
            raise ValueError("Amplitude diurne dans [0, 1) et phase dans [0, 1] requises")
        fractions = (
            self.flash_start_fraction, self.flash_ramp_up_fraction,
            self.flash_plateau_fraction, self.flash_ramp_down_fraction,
        )
        if (
            not all(math.isfinite(value) and value >= 0 for value in fractions)
            or self.flash_ramp_up_fraction <= 0
            or self.flash_ramp_down_fraction <= 0
            or sum(fractions) > 1
        ):
            raise ValueError("Les phases flash crowd doivent être positives et tenir dans l'horizon")
        if not math.isfinite(self.flash_peak_multiplier) or self.flash_peak_multiplier < 1:
            raise ValueError("flash_peak_multiplier doit être fini et >= 1")
        if not self.trace_count_column:
            raise ValueError("Le nom de la colonne de comptage de trace ne peut pas être vide")
        if self.trace_timestamp_column is not None and not self.trace_timestamp_column:
            raise ValueError("Utiliser None pour une trace sans horodatages")
        if self.scenario == "trace" and not self.trace_path:
            raise ValueError("Le scénario 'trace' exige trace_path vers un CSV de comptages")
