"""Génération d'une séquence de requêtes : le cœur du module."""
import numpy as np
import pandas as pd

from .schema import TrafficConfig, ZONES
from .scenarios import ARRIVAL_PROCESSES


def mean_work_ms(cfg: TrafficConfig) -> float:
    """Travail moyen d'une requête, pondéré par la part de chaque type."""
    return sum(t.share * t.mean_work_ms for t in cfg.types.values())


def arrival_rate(cfg: TrafficConfig) -> float:
    """Débit (requêtes par ms) qui produit la charge rho."""
    return cfg.rho * cfg.capacity / mean_work_ms(cfg)


def generate_arrivals(cfg: TrafficConfig, seed: int) -> pd.DataFrame:
    try:
        process = ARRIVAL_PROCESSES[cfg.scenario]
    except KeyError:
        raise ValueError(
            f"Scénario inconnu : {cfg.scenario!r}. Disponibles : {sorted(ARRIVAL_PROCESSES)}"
        )

    # Un flux aléatoire indépendant par aspect, tous dérivés de la même seed.
    ss = np.random.SeedSequence(seed)
    rng_t, rng_type, rng_work, rng_zone = (
        np.random.default_rng(s) for s in ss.spawn(4)
    )

    # 1. QUAND : instants d'arrivée
    lam = arrival_rate(cfg)
    process_options = {
        "peaks": {
            "low_mean_ms": cfg.peak_low_mean_ms,
            "high_mean_ms": cfg.peak_high_mean_ms,
            "low_multiplier": cfg.peak_low_multiplier,
            "high_multiplier": cfg.peak_high_multiplier,
        },
        "diurnal": {
            "period_ms": cfg.diurnal_period_ms,
            "amplitude": cfg.diurnal_amplitude,
            "peak_phase": cfg.diurnal_peak_phase,
            "resolution_ms": cfg.profile_resolution_ms,
        },
        "flash_crowd": {
            "peak_multiplier": cfg.flash_peak_multiplier,
            "start_fraction": cfg.flash_start_fraction,
            "ramp_up_fraction": cfg.flash_ramp_up_fraction,
            "plateau_fraction": cfg.flash_plateau_fraction,
            "ramp_down_fraction": cfg.flash_ramp_down_fraction,
            "resolution_ms": cfg.profile_resolution_ms,
        },
        "trace": {
            "trace_path": cfg.trace_path,
            "trace_count_column": cfg.trace_count_column,
            "trace_timestamp_column": cfg.trace_timestamp_column,
            "trace_interval_ms": cfg.trace_interval_ms,
        },
    }
    t = process(
        rng_t,
        lam,
        cfg.n_requests,
        cfg.duration_ms,
        **process_options.get(cfg.scenario, {}),
    )
    n = len(t)

    # 2. QUOI : type, travail, échéance
    names = sorted(cfg.types)
    shares = np.array([cfg.types[k].share for k in names])
    type_idx = rng_type.choice(len(names), size=n, p=shares)

    means = np.array([cfg.types[k].mean_work_ms for k in names])[type_idx]
    sigma = np.sqrt(np.log(1.0 + cfg.work_cv ** 2))
    mu = np.log(means) - sigma ** 2 / 2.0
    work = rng_work.lognormal(mu, sigma)

    # Le worker reçoit type + size et calcule son travail comme
    # WORK_MS[type] * size / SPEED_FACTOR. Utiliser la moyenne de référence
    # du type permet de représenter le work_ms tiré par le générateur.
    size = work / means

    deadlines = np.array([cfg.types[k].deadline_ms for k in names])[type_idx]

    # 3. D'OÙ : zone d'origine
    zone_idx = rng_zone.choice(len(ZONES), size=n, p=np.array(cfg.zone_weights))

    return pd.DataFrame({
        "request_id": [f"r-{i:06d}" for i in range(n)],
        "t_arrival_ms": t,
        "type": np.array(names)[type_idx],
        "work_ms": work,
        "size": size,
        "deadline_ms": deadlines,
        "zone": np.array(ZONES)[zone_idx],
    })


if __name__ == "__main__":
    cfg = TrafficConfig(rho=0.6, n_requests=5000, scenario="constant")
    df = generate_arrivals(cfg, seed=1000)

    print("Travail moyen :", mean_work_ms(cfg), "ms")
    print("Débit moyen visé :", arrival_rate(cfg) * 1000, "req/s")
    print(df.head(10))
    # print("Nombre total de requêtes :", len(df))
    # print(df)

    horizon = df["t_arrival_ms"].iloc[-1]
    print("rho observé   :", round(df["work_ms"].sum() / (horizon * cfg.capacity), 3))
    print(df["type"].value_counts(normalize=True).round(3))
