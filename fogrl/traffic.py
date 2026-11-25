"""Generateurs de trafic reproductibles (EF-06 : meme graine = meme sequence).

Scenarios : Poisson (constant), MMPP a 2 etats (pics), sinusoide (diurne),
rampe x5 (flash crowd) et rejeu d'une trace (Azure Functions 2019, remise a
l'echelle). Les processus non homogenes sont generes exactement : des arrivees
de taux 1 sont deformees par l'inverse du taux cumule (constant par morceaux
sur un pas de 50 ms).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .config import NodeProfile, RequestProfile, ScenarioSpec, capacity_rps, resolve_path

RateFn = Callable[[np.ndarray], np.ndarray]
_DT = 0.05  # pas de discretisation du taux (s)


@dataclass(frozen=True)
class RequestStream:
    """Sequence de requetes d'un episode : instants, types et tailles."""

    times_s: np.ndarray
    type_names: tuple[str, ...]
    type_index: np.ndarray
    work_ms: np.ndarray

    def __len__(self) -> int:
        return len(self.times_s)

    def type_of(self, i: int) -> str:
        return self.type_names[int(self.type_index[i])]


def base_rate_rps(
    spec: ScenarioSpec, active: Sequence[NodeProfile], requests: Sequence[RequestProfile]
) -> float:
    """Debit de base = load x capacite des noeuds fog actifs (tous les noeuds s'il n'y a pas de fog)."""

    fog = [p for p in active if not p.is_cloud] or list(active)
    return spec.load * capacity_rps(fog, requests)


# ------------------------------------------------------------------ taux variables


def _mmpp_multiplier(params, seed: int) -> RateFn:
    low = float(params.get("low_factor", 0.5))
    high = float(params.get("high_factor", 2.0))
    mean_low = float(params.get("mean_low_s", 4.0))
    mean_high = float(params.get("mean_high_s", 2.0))

    def fn(t: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed)  # meme prefixe quel que soit l'horizon
        horizon = float(t[-1]) + 1.0 if len(t) else 1.0
        state = 0 if rng.random() < mean_low / (mean_low + mean_high) else 1
        edges, states, now = [0.0], [state], 0.0
        while now <= horizon:
            now += rng.exponential(mean_low if state == 0 else mean_high)
            state = 1 - state
            edges.append(now)
            states.append(state)
        index = np.searchsorted(np.asarray(edges), t, side="right") - 1
        return np.where(np.asarray(states)[index] == 0, low, high)

    return fn


def _diurnal_multiplier(params) -> RateFn:
    amplitude = float(params.get("amplitude", 0.5))
    period = float(params.get("period_s", 30.0))
    if not 0 <= amplitude < 1:
        raise ValueError("diurnal : amplitude doit etre dans [0, 1[")
    return lambda t: 1.0 + amplitude * np.sin(2.0 * np.pi * t / period)


def _flash_multiplier(params) -> RateFn:
    factor = float(params.get("factor", 5.0))
    start = float(params.get("start_s", 5.0))
    ramp = float(params.get("ramp_s", 3.0))
    hold = float(params.get("hold_s", 6.0))
    decay = float(params.get("decay_s", 3.0))
    xs = [0.0, start, start + ramp, start + ramp + hold, start + ramp + hold + decay]
    ys = [1.0, 1.0, factor, factor, 1.0]
    return lambda t: np.interp(t, xs, ys)


def load_trace_counts(path) -> np.ndarray:
    """Charge une trace au format Azure Functions 2019 et somme les fonctions par minute.

    Colonnes attendues : HashOwner, HashApp, HashFunction, Trigger, 1 ... 1440
    (nombre d'invocations par minute).
    """

    return _load_trace_counts(str(resolve_path(path)))


@lru_cache(maxsize=8)
def _load_trace_counts(path: str) -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        minute_columns = [i for i, name in enumerate(header) if name.strip().isdigit()]
        if not minute_columns:
            raise ValueError(f"{path} : aucune colonne de minute (1..1440) trouvee")
        totals = np.zeros(len(minute_columns))
        for row in reader:
            totals += np.asarray([float(row[i] or 0) for i in minute_columns])
    if totals.sum() <= 0:
        raise ValueError(f"{path} : trace vide")
    return totals


def _trace_multiplier(params, rng: np.random.Generator) -> RateFn:
    counts = load_trace_counts(params["trace_file"])
    bin_s = float(params.get("bin_s", 1.0))
    offset = int(rng.integers(0, len(counts)))
    # Plancher a 5 % du debit moyen pour eviter les trous de plusieurs minutes.
    multiplier = np.maximum(counts / counts.mean(), 0.05)
    return lambda t: multiplier[(offset + np.floor(t / bin_s).astype(int)) % len(counts)]


def _multiplier_fn(spec: ScenarioSpec, rng: np.random.Generator) -> RateFn:
    if spec.type == "mmpp":
        return _mmpp_multiplier(spec.params, int(rng.integers(0, 2**31 - 1)))
    if spec.type == "diurnal":
        return _diurnal_multiplier(spec.params)
    if spec.type == "flash_crowd":
        return _flash_multiplier(spec.params)
    if spec.type == "trace":
        return _trace_multiplier(spec.params, rng)
    raise ValueError(f"Type de scenario non homogene inconnu : {spec.type}")


def _nhpp_times(rate_fn: RateFn, unit_arrivals: np.ndarray, mean_rate: float) -> np.ndarray:
    """Deforme des arrivees de taux 1 en arrivees de taux ``rate_fn`` (inverse du taux cumule)."""

    horizon = max(5.0, 2.0 * len(unit_arrivals) / mean_rate)
    while True:
        edges = np.arange(0.0, horizon + _DT, _DT)
        mids = edges[:-1] + _DT / 2.0
        rates = np.maximum(rate_fn(mids), 1e-6)
        cumulative = np.concatenate(([0.0], np.cumsum(rates * _DT)))
        if cumulative[-1] >= unit_arrivals[-1]:
            return np.interp(unit_arrivals, cumulative, edges)
        horizon *= 2.0


# ---------------------------------------------------------------------- flux


def generate_stream(
    spec: ScenarioSpec,
    base_rate: float,
    n: int,
    requests: Sequence[RequestProfile],
    seed: int,
) -> RequestStream:
    """Genere ``n`` requetes. Meme graine et memes parametres : meme sequence."""

    if base_rate <= 0 or n < 1:
        raise ValueError("base_rate > 0 et n >= 1 requis")
    streams = np.random.SeedSequence(seed).spawn(4)
    rng_arrivals, rng_types, rng_work, rng_path = (np.random.default_rng(s) for s in streams)

    if spec.type == "poisson":
        times = np.cumsum(rng_arrivals.exponential(1.0 / base_rate, n))
    else:
        multiplier = _multiplier_fn(spec, rng_path)
        unit = np.cumsum(rng_arrivals.exponential(1.0, n))
        times = _nhpp_times(lambda t: base_rate * multiplier(t), unit, base_rate)

    weights = np.asarray([r.weight for r in requests], dtype=float)
    type_index = rng_types.choice(len(requests), size=n, p=weights / weights.sum())
    nominal = np.asarray([r.work_ms for r in requests])[type_index]
    cv = np.asarray([r.work_cv for r in requests])[type_index]
    sigma2 = np.log1p(cv**2)
    work_ms = nominal * np.exp(rng_work.normal(-sigma2 / 2.0, np.sqrt(sigma2)))
    return RequestStream(times, tuple(r.name for r in requests), type_index, work_ms)


def generate_stream_until(
    spec: ScenarioSpec,
    base_rate: float,
    duration_s: float,
    requests: Sequence[RequestProfile],
    seed: int,
) -> RequestStream:
    """Requetes arrivant avant ``duration_s`` (campagnes reelles, EF-06)."""

    n = int(base_rate * duration_s * 1.3) + 50
    while True:
        stream = generate_stream(spec, base_rate, n, requests, seed)
        if stream.times_s[-1] >= duration_s:
            keep = int(np.searchsorted(stream.times_s, duration_s, side="right"))
            return RequestStream(
                stream.times_s[:keep],
                stream.type_names,
                stream.type_index[:keep],
                stream.work_ms[:keep],
            )
        n *= 2


# ------------------------------------------------------------- trace synthetique


def make_synthetic_trace(
    path, n_functions: int = 12, minutes: int = 1440, seed: int = 0
) -> Path:
    """Ecrit une trace SYNTHETIQUE au format Azure Functions 2019.

    Sert aux tests et a la CI. Pour les resultats du memoire, utilisez la
    vraie trace (scripts/download_azure_trace.sh).
    """

    rng = np.random.default_rng(seed)
    t = np.arange(minutes)
    rows = []
    for k in range(n_functions):
        base = rng.lognormal(mean=2.0, sigma=0.8)
        phase = rng.uniform(0, 2 * np.pi)
        daily = 1.0 + 0.6 * np.sin(2 * np.pi * t / minutes + phase)
        bursts = np.where(rng.random(minutes) < 0.01, rng.uniform(3, 8, minutes), 1.0)
        counts = rng.poisson(np.maximum(base * daily * bursts, 0.0))
        rows.append(
            [f"owner{k % 3:02d}", f"app{k:03d}", f"func{k:04d}", "http", *map(int, counts)]
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "Trigger", *range(1, minutes + 1)])
        writer.writerows(rows)
    return path
