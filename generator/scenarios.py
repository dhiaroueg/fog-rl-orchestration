"""Arrival processes for the shared traffic generator.

All times and rates are expressed in milliseconds and arrivals per millisecond.
For a fixed request count, non-stationary processes are conditioned on that
count over a reference horizon of ``n / mean_rate``. For a fixed duration they
are sampled as non-homogeneous Poisson processes, so the count remains random.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def _sort_and_clip(times: list[np.ndarray], horizon_ms: float) -> np.ndarray:
    if not times:
        return np.empty(0, dtype=np.float64)
    result = np.sort(np.concatenate(times).astype(np.float64, copy=False))
    return result[(result >= 0.0) & (result < horizon_ms)]


def _sample_segments(
    rng: np.random.Generator,
    lam: float,
    segments: list[tuple[float, float, float]],
    n: int | None,
) -> np.ndarray:
    """Sample a piecewise-constant rate profile, optionally conditioning on N."""
    if not segments:
        return np.empty(0, dtype=np.float64)
    starts = np.fromiter((s[0] for s in segments), dtype=np.float64)
    ends = np.fromiter((s[1] for s in segments), dtype=np.float64)
    multipliers = np.fromiter((s[2] for s in segments), dtype=np.float64)
    widths = ends - starts
    if np.any(widths <= 0) or np.any(multipliers < 0):
        raise ValueError("Profil de trafic invalide : durées positives et taux >= 0 requis")

    if n is not None:
        mass = widths * multipliers
        total_mass = float(mass.sum())
        if total_mass <= 0:
            raise ValueError("Le profil choisi ne produit aucune arrivée sur l'horizon")
        selected = rng.choice(len(segments), size=n, p=mass / total_mass)
        times = starts[selected] + rng.random(n) * widths[selected]
        return np.sort(times.astype(np.float64, copy=False))

    chunks = []
    for start, end, multiplier in segments:
        count = int(rng.poisson(lam * multiplier * (end - start)))
        if count:
            chunks.append(start + rng.random(count) * (end - start))
    return _sort_and_clip(chunks, float(ends[-1]))


def poisson_times(
    rng: np.random.Generator,
    lam: float,
    n: int | None = None,
    duration_ms: float | None = None,
) -> np.ndarray:
    """Homogeneous Poisson process (constant mean arrival rate)."""
    if lam <= 0:
        raise ValueError("Le débit de base doit être strictement positif")
    if n is not None:
        return np.cumsum(rng.exponential(1.0 / lam, size=n), dtype=np.float64)
    count = int(rng.poisson(lam * duration_ms))
    return np.sort(rng.random(count) * duration_ms).astype(np.float64, copy=False)


def _fixed_horizon(n: int | None, duration_ms: float | None, lam: float) -> float:
    return float(duration_ms if duration_ms is not None else n / lam)


def _regular_segments(
    horizon_ms: float,
    step_ms: float,
    multiplier_at,
) -> list[tuple[float, float, float]]:
    count = max(1, int(np.ceil(horizon_ms / step_ms)))
    edges = np.linspace(0.0, horizon_ms, count + 1, dtype=np.float64)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    return [
        (float(start), float(end), float(multiplier_at(float(mid))))
        for start, end, mid in zip(edges[:-1], edges[1:], midpoints)
    ]


def _segments_with_boundaries(horizon_ms, step_ms, boundaries, multiplier_at):
    grid = np.arange(0.0, horizon_ms, step_ms, dtype=np.float64)
    edges = np.unique(np.concatenate((grid, np.asarray(boundaries), [horizon_ms])))
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    return [
        (float(start), float(end), float(multiplier_at(float(mid))))
        for start, end, mid in zip(edges[:-1], edges[1:], midpoints)
    ]


def mmpp_times(rng, lam, n=None, duration_ms=None, *, low_mean_ms, high_mean_ms,
               low_multiplier, high_multiplier):
    """Two-state MMPP with exponential low/high state residence times."""
    horizon = _fixed_horizon(n, duration_ms, lam)
    high_probability = high_mean_ms / (low_mean_ms + high_mean_ms)
    segments = []
    t = 0.0
    high = bool(rng.random() < high_probability)
    while t < horizon:
        mean = high_mean_ms if high else low_mean_ms
        end = min(horizon, t + float(rng.exponential(mean)))
        if end <= t:  # Protect against an underflow-sized exponential draw.
            end = min(horizon, np.nextafter(t, np.inf))
        segments.append((t, end, high_multiplier if high else low_multiplier))
        t = end
        high = not high

    # Normalize from the CTMC stationary probabilities, preserving MMPP volume
    # variation from one run to another instead of forcing each path to its mean.
    stationary_mean = (
        (1.0 - high_probability) * low_multiplier
        + high_probability * high_multiplier
    )
    normalized = [(a, b, m / stationary_mean) for a, b, m in segments]
    return _sample_segments(rng, lam, normalized, n)


def diurnal_times(rng, lam, n=None, duration_ms=None, *, period_ms, amplitude,
                  peak_phase, resolution_ms):
    """Sinusoidal day/night intensity with mean rate normalized over full cycles."""
    horizon = _fixed_horizon(n, duration_ms, lam)

    def multiplier(t):
        phase = (t / period_ms - peak_phase) * 2.0 * np.pi
        # Formule de la fonction sinusoïde 
        return 1.0 + amplitude * np.cos(phase)

    segments = _regular_segments(horizon, min(period_ms / 1440.0, resolution_ms), multiplier)
    if n is None:
        return _sample_segments(rng, lam, segments, None)
    return _sample_segments(rng, lam, segments, n)

# scénario simule une hausse brutale et temporaire du trafic, suivie d’une récupération progressive.
def flash_crowd_times(rng, lam, n=None, duration_ms=None, *, peak_multiplier,
                      start_fraction, ramp_up_fraction, plateau_fraction,
                      ramp_down_fraction, resolution_ms):
    """Baseline traffic with a bounded linear flash-crowd pulse and recovery."""
    horizon = _fixed_horizon(n, duration_ms, lam)
    start = start_fraction * horizon
    ramp_end = start + ramp_up_fraction * horizon
    plateau_end = ramp_end + plateau_fraction * horizon
    recovery_end = plateau_end + ramp_down_fraction * horizon

    def multiplier(t):
        if t < start or t >= recovery_end:
            return 1.0
        if t < ramp_end:
            fraction = (t - start) / (ramp_end - start)
            return 1.0 + fraction * (peak_multiplier - 1.0)
        if t < plateau_end:
            return peak_multiplier
        fraction = (t - plateau_end) / (recovery_end - plateau_end)
        return peak_multiplier - fraction * (peak_multiplier - 1.0)

    segments = _segments_with_boundaries(
        horizon,
        resolution_ms,
        (start, ramp_end, plateau_end, recovery_end),
        multiplier,
    )
    return _sample_segments(rng, lam, segments, n)


@lru_cache(maxsize=16)
def _read_trace(path_text: str, modified_ns: int, count_column: str,
                timestamp_column: str, interval_ms: float) -> tuple[np.ndarray, np.ndarray]:
    del modified_ns  # Included in the cache key so changed trace files are reloaded.
    path = Path(path_text)
    frame = pd.read_csv(path)
    if count_column not in frame.columns:
        raise ValueError(
            f"Colonne de comptage {count_column!r} absente du fichier de trace {path}"
        )
    counts = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(dtype=np.float64)
    if len(counts) == 0 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Les comptages de la trace doivent être finis, positifs ou nuls")

    if timestamp_column in frame.columns:
        stamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
        if stamps.isna().any():
            raise ValueError(f"Horodatages invalides dans la colonne {timestamp_column!r}")
        elapsed = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy() * 1000.0
        if len(elapsed) > 1 and np.any(np.diff(elapsed) <= 0):
            raise ValueError("Les horodatages de la trace doivent être strictement croissants")
        widths = np.diff(elapsed, append=elapsed[-1] + float(np.median(np.diff(elapsed)))
                         if len(elapsed) > 1 else interval_ms)
    else:
        widths = np.full(len(counts), interval_ms, dtype=np.float64)
    if not np.isfinite(widths).all() or np.any(widths <= 0) or counts.sum() <= 0:
        raise ValueError("La trace doit contenir une durée et un total d'invocations positifs")
    return counts, widths


def trace_times(rng, lam, n=None, duration_ms=None, *, trace_path,
                trace_count_column, trace_timestamp_column, trace_interval_ms):
    """Replay binned trace counts, rescaled to the requested mean arrival rate."""
    if not trace_path:
        raise ValueError("Le scénario 'trace' exige trace_path vers un CSV de comptages")
    path = Path(trace_path).expanduser().resolve()
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise ValueError(f"Impossible de lire le fichier de trace {path}: {exc}") from exc
    counts, widths = _read_trace(
        str(path), modified_ns, trace_count_column, trace_timestamp_column, trace_interval_ms
    )
    horizon = _fixed_horizon(n, duration_ms, lam)
    source_edges = np.concatenate(([0.0], np.cumsum(widths)))
    source_horizon = float(source_edges[-1])
    # Preserve the source profile while mapping its complete time axis to this run.
    starts = source_edges[:-1] / source_horizon * horizon
    ends = source_edges[1:] / source_horizon * horizon
    weights = counts / widths
    mean_weight = float(np.average(weights, weights=ends - starts))
    segments = [
        (float(a), float(b), float(weight / mean_weight))
        for a, b, weight in zip(starts, ends, weights)
    ]
    return _sample_segments(rng, lam, segments, n)


ARRIVAL_PROCESSES = {
    "constant": poisson_times,
    "poisson": poisson_times,
    "peaks": mmpp_times,
    "diurnal": diurnal_times,
    "flash_crowd": flash_crowd_times,
    "trace": trace_times,
}
