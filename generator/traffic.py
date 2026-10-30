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
    t = process(rng_t, arrival_rate(cfg), cfg.n_requests, cfg.duration_ms)
    n = len(t)

    # 2. QUOI : type, travail, échéance
    names = sorted(cfg.types)
    shares = np.array([cfg.types[k].share for k in names])
    type_idx = rng_type.choice(len(names), size=n, p=shares)

    means = np.array([cfg.types[k].mean_work_ms for k in names])[type_idx]
    sigma = np.sqrt(np.log(1.0 + cfg.work_cv ** 2))
    mu = np.log(means) - sigma ** 2 / 2.0
    work = rng_work.lognormal(mu, sigma)

    deadlines = np.array([cfg.types[k].deadline_ms for k in names])[type_idx]

    # 3. D'OÙ : zone d'origine
    zone_idx = rng_zone.choice(len(ZONES), size=n, p=np.array(cfg.zone_weights))

    return pd.DataFrame({
        "request_id": [f"r-{i:06d}" for i in range(n)],
        "t_arrival_ms": t,
        "type": np.array(names)[type_idx],
        "work_ms": work,
        "deadline_ms": deadlines,
        "zone": np.array(ZONES)[zone_idx],
    })


if __name__ == "__main__":
    cfg = TrafficConfig(rho=0.6, n_requests=5000)
    df = generate_arrivals(cfg, seed=1000)

    print("Travail moyen :", mean_work_ms(cfg), "ms")
    print("Débit visé    :", arrival_rate(cfg) * 1000, "req/s")
    print(df.head(10))
    # print("Nombre total de requêtes :", len(df))
    # print(df)


    horizon = df["t_arrival_ms"].iloc[-1]
    print("rho observé   :", round(df["work_ms"].sum() / (horizon * cfg.capacity), 3))
    print(df["type"].value_counts(normalize=True).round(3))