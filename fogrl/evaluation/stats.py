"""Statistiques de comparaison (cahier, section 8.4) et verification des criteres de recette.

- moyenne +- intervalle de confiance a 95 % (loi de Student) ;
- test de Mann-Whitney entre la meilleure politique RL et la meilleure baseline (p < 0,05) ;
- criteres ENF-02, ENF-03, H1, H2 et OS-4 calcules a partir du CSV de resultats.

Pour toutes les metriques d'objectif (latence, energie, cout), plus petit = mieux.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

LATENCY = "latency_mean_ms"
OBJECTIVES = {
    "latence": LATENCY,
    "energie": "energy_fog_j_per_req",
    "cout": "cloud_cost_per_req",
}


def mean_ci(values: Iterable[float], level: float = 0.95) -> tuple[float, float]:
    """Moyenne et demi-largeur de l'intervalle de confiance."""

    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.sem(x) * stats.t.ppf((1 + level) / 2, len(x) - 1)
    return float(x.mean()), float(half)


def aggregate(
    df: pd.DataFrame,
    metrics: Sequence[str],
    by: Sequence[str] = ("policy", "scenario", "size"),
) -> pd.DataFrame:
    """Une ligne par groupe avec ``<metrique>_mean``, ``<metrique>_ci95`` et ``n``."""

    rows = []
    for key, group in df.groupby(list(by), sort=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def relative_reduction(reference: float, candidate: float) -> float:
    """Gain relatif de ``candidate`` sur ``reference`` (0,2 = 20 % de moins)."""

    if reference == 0 or math.isnan(reference) or math.isnan(candidate):
        return float("nan")
    return (reference - candidate) / reference


def mann_whitney_less(candidate: np.ndarray, reference: np.ndarray) -> float:
    """p-value du test de Mann-Whitney : ``candidate`` est-il plus petit que ``reference`` ?"""

    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    if len(candidate) == 0 or len(reference) == 0:
        return float("nan")
    if np.ptp(np.concatenate([candidate, reference])) == 0:
        return 1.0
    return float(stats.mannwhitneyu(candidate, reference, alternative="less").pvalue)


def _values(df: pd.DataFrame, policy: str, scenario: str, size: int, metric: str) -> np.ndarray:
    mask = (df["policy"] == policy) & (df["scenario"] == scenario) & (df["size"] == size)
    return df.loc[mask, metric].to_numpy(dtype=float)


def best_rl(df: pd.DataFrame, size: int, scenarios: Optional[Sequence[str]] = None) -> Optional[str]:
    """Politique RL de plus faible latence moyenne (sur ``scenarios``)."""

    rl = df[(df["policy_type"] == "rl") & (df["size"] == size)]
    if scenarios is not None:
        rl = rl[rl["scenario"].isin(scenarios)]
    if rl.empty:
        return None
    return str(rl.groupby("policy")[LATENCY].mean().idxmin())


def best_baseline(df: pd.DataFrame, scenario: str, size: int, metric: str) -> Optional[str]:
    base = df[(df["policy_type"] == "baseline") & (df["scenario"] == scenario) & (df["size"] == size)]
    if base.empty:
        return None
    return str(base.groupby("policy")[metric].mean().idxmin())


def acceptance_report(
    df: pd.DataFrame,
    size: int = 4,
    load_scenarios: Sequence[str] = ("medium", "high"),
    latency_tolerance: float = 0.05,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Verifie les criteres chiffres du cahier des charges sur un CSV de resultats.

    ``latency_tolerance`` definit la « latence comparable » de H2 (RL <= Greedy x 1,05).
    Un critere dont les politiques necessaires sont absentes du CSV vaut ``passed=None``.
    """

    available = set(df.loc[df["size"] == size, "scenario"])
    loads = [s for s in load_scenarios if s in available]
    rl = best_rl(df, size, loads or None)
    report: dict[str, Any] = {"size": size, "best_rl": rl, "checks": []}
    if rl is None:
        report["error"] = "aucune politique RL dans les resultats"
        return report

    def mean_of(policy: str, scenario: str, metric: str) -> float:
        values = _values(df, policy, scenario, size, metric)
        return float(values.mean()) if len(values) else float("nan")

    def add(check_id: str, description: str, passed: Optional[bool], details: dict) -> None:
        report["checks"].append(
            {"id": check_id, "description": description, "passed": passed, "details": details}
        )

    # ENF-02 / H1 : latence moyenne >= 20 % sous round-robin
    gains = {s: relative_reduction(mean_of("round_robin", s, LATENCY), mean_of(rl, s, LATENCY)) for s in loads}
    valid = {s: g for s, g in gains.items() if not math.isnan(g)}
    add(
        "ENF-02 / H1",
        "Latence moyenne reduite d'au moins 20 % face au round-robin (charge moyenne et elevee)",
        (all(g >= 0.20 for g in valid.values()) if valid and len(valid) == len(loads) else None),
        {"reduction": valid},
    )

    # ENF-03 : energie fog >= 15 % sous seuils statiques, a latence egale ou meilleure
    details, ok = {}, []
    for s in loads:
        energy = relative_reduction(
            mean_of("static_threshold", s, "energy_fog_j_per_req"), mean_of(rl, s, "energy_fog_j_per_req")
        )
        lat_ok = mean_of(rl, s, LATENCY) <= mean_of("static_threshold", s, LATENCY)
        details[s] = {"energy_reduction": energy, "latency_equal_or_better": bool(lat_ok)}
        ok.append(None if math.isnan(energy) else (energy >= 0.15 and lat_ok))
    add(
        "ENF-03",
        "Energie fog reduite d'au moins 15 % face aux seuils statiques, a latence egale ou meilleure",
        None if not ok or any(o is None for o in ok) else all(ok),
        details,
    )

    # H2 : face a Greedy, latence comparable et energie OU cout >= 15 % plus bas
    details, ok = {}, []
    for s in loads:
        greedy_lat = mean_of("greedy", s, LATENCY)
        energy = relative_reduction(mean_of("greedy", s, "energy_fog_j_per_req"), mean_of(rl, s, "energy_fog_j_per_req"))
        cost = relative_reduction(mean_of("greedy", s, "cloud_cost_per_req"), mean_of(rl, s, "cloud_cost_per_req"))
        comparable = mean_of(rl, s, LATENCY) <= greedy_lat * (1 + latency_tolerance)
        details[s] = {"latency_comparable": bool(comparable), "energy_reduction": energy, "cost_reduction": cost}
        if math.isnan(greedy_lat):
            ok.append(None)
        else:
            ok.append(bool(comparable and ((energy >= 0.15) or (cost >= 0.15))))
    add(
        "H2",
        "Face a Greedy : latence comparable et energie ou cout reduits d'au moins 15 %",
        None if not ok or any(o is None for o in ok) else all(ok),
        details,
    )

    # OS-4 : gain significatif (Mann-Whitney) sur au moins 2 objectifs
    scenarios = sorted(available)
    significant: dict[str, list[str]] = {}
    details = {}
    for label, metric in OBJECTIVES.items():
        wins = []
        for s in scenarios:
            reference = best_baseline(df, s, size, metric)
            if reference is None:
                continue
            candidate_values = _values(df, rl, s, size, metric)
            reference_values = _values(df, reference, s, size, metric)
            p = mann_whitney_less(candidate_values, reference_values)
            details.setdefault(label, {})[s] = {"vs": reference, "p_value": p}
            if not math.isnan(p) and p < alpha:
                wins.append(s)
        significant[label] = wins
    threshold = math.ceil(len(scenarios) / 2) if scenarios else 0
    winning_objectives = [k for k, wins in significant.items() if scenarios and len(wins) >= threshold]
    add(
        "OS-4",
        f"Gain significatif (p < {alpha}) sur au moins 2 objectifs face a la meilleure baseline "
        "(dans au moins la moitie des scenarios)",
        len(winning_objectives) >= 2 if scenarios else None,
        {"objectives_won": winning_objectives, "scenarios_won": significant, "tests": details},
    )
    return report


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
    if isinstance(value, float):
        return "n/d" if math.isnan(value) else f"{value:.3g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Criteres de recette (taille {report['size']})", ""]
    if "error" in report:
        return "\n".join(lines + [f"**{report['error']}**"])
    lines += [f"Meilleure politique RL : **{report['best_rl']}**", "", "| Critere | Resultat | Details |", "|---|---|---|"]
    symbol = {True: "OK", False: "NON ATTEINT", None: "non evaluable"}
    for check in report["checks"]:
        lines.append(f"| {check['id']} : {check['description']} | {symbol[check['passed']]} | {_fmt(check['details'])} |")
    return "\n".join(lines) + "\n"
