"""Reduit les mesures brutes en table de calibration et propose des valeurs de profil.

    python -m fogrl.calibration.fit --data data/calibration/2026-11-20 --out configs/calibration

Sorties :
    calibration.json       table (noeud, type) -> charge, duree moyenne, cv ; lue par
                           ``service_model: calibrated`` + ``calibration_file`` (``sim.service``)
    suggested_nodes.yaml   speed, contention, service_cv, rtt_ms, rtt_sigma mesures, a reporter
                           dans configs/nodes/pool6.yaml (puis re-entrainer, voir docs/runbook.md)
    report.md              hypotheses du cahier face aux mesures, avec les ecarts a verifier

La duree moyenne est ramenee au travail nominal du type (``mean_ms`` pour ``work_ms`` nominal) : le
simulateur la remet a l'echelle de chaque requete.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from ..config import NodeProfile, load_nodes, load_requests

MIN_SAMPLES = 5  # mesures minimales par point de charge pour qu'il entre dans la table


def fit_service_table(
    raw: pd.DataFrame, nominal_work_ms: Mapping[str, float], min_samples: int = MIN_SAMPLES
) -> tuple[list[dict], list[str]]:
    """Retourne (entrees, avertissements)."""

    entries, warnings = [], []
    for (node, req_type), group in raw.groupby(["node", "req_type"]):
        nominal = nominal_work_ms[req_type]
        service = group["service_ms"] * nominal / group["work_ms"]  # ramene au travail nominal
        frame = pd.DataFrame({"load": group["load"].round(3), "service": service})
        points = []
        for load, part in frame.groupby("load"):
            if len(part) < min_samples:
                continue
            mean = float(part["service"].mean())
            points.append((float(load), mean, float(part["service"].std(ddof=1) / mean) if mean > 0 else 0.0, len(part)))
        if not points:
            warnings.append(f"{node}/{req_type} : aucun niveau de charge avec >= {min_samples} mesures, couple ignore")
            continue
        points.sort()
        if len(points) == 1:
            warnings.append(f"{node}/{req_type} : un seul niveau de charge, contention non estimable")
        elif points[-1][0] < 0.5:
            warnings.append(f"{node}/{req_type} : charge maximale mesuree {points[-1][0]:.2f} < 0.5, extrapolation au-dela")
        entries.append(
            {
                "node": node,
                "req_type": req_type,
                "loads": [p[0] for p in points],
                "mean_ms": [p[1] for p in points],
                "cv": [p[2] for p in points],
                "n": [p[3] for p in points],
            }
        )
    return entries, warnings


def suggest_profiles(
    entries: Sequence[dict], rtt: Optional[pd.DataFrame], nominal_work_ms: Mapping[str, float]
) -> dict[str, dict]:
    """speed, contention et service_cv deduits des entrees ; rtt_ms et rtt_sigma des sondages."""

    suggestions: dict[str, dict] = {}
    for node in sorted({e["node"] for e in entries}):
        speeds, weights, ratios, cvs = [], [], [], []
        for e in (e for e in entries if e["node"] == node):
            base = e["mean_ms"][0]
            speeds.append(nominal_work_ms[e["req_type"]] / base)
            weights.append(e["n"][0])
            cvs.append(e["cv"][0])
            for load, mean in zip(e["loads"][1:], e["mean_ms"][1:]):
                ratios.append((load - e["loads"][0], mean / base - 1.0))
        suggestion = {
            "speed": round(float(np.average(speeds, weights=weights)), 3),
            "service_cv": round(float(np.average(cvs, weights=weights)), 3),
        }
        if ratios:  # regression a travers l'origine : mean(load) = base x (1 + contention x load)
            dl = np.asarray([r[0] for r in ratios])
            dr = np.asarray([r[1] for r in ratios])
            suggestion["contention"] = round(float(max(np.sum(dl * dr) / np.sum(dl * dl), 0.0)), 3)
        if rtt is not None and len(rtt) and node in set(rtt["node"]):
            samples = rtt.loc[rtt["node"] == node, "rtt_ms"]
            suggestion["rtt_ms"] = round(float(samples.mean()), 2)
            suggestion["rtt_sigma"] = round(float(samples.std(ddof=1) / samples.mean()), 3) if len(samples) > 1 else 0.0
        suggestions[node] = suggestion
    return suggestions


def build_calibration(
    raw: pd.DataFrame,
    rtt: Optional[pd.DataFrame],
    profiles: Mapping[str, NodeProfile],
    nominal_work_ms: Mapping[str, float],
    source: str = "",
    min_samples: int = MIN_SAMPLES,
) -> tuple[dict, dict, list[str]]:
    entries, warnings = fit_service_table(raw, nominal_work_ms, min_samples)
    suggestions = suggest_profiles(entries, rtt, nominal_work_ms)
    unknown = sorted(set(suggestions) - set(profiles))
    if unknown:
        warnings.append(f"Noeuds mesures absents de la configuration : {unknown}")
    calibration = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "nominal_work_ms": dict(nominal_work_ms),
        "entries": entries,
    }
    return calibration, suggestions, warnings


def render_report(
    suggestions: Mapping[str, dict], profiles: Mapping[str, NodeProfile], warnings: Sequence[str], threshold: float = 0.3
) -> str:
    lines = [
        "# Rapport de calibration",
        "",
        "Hypotheses du modele (`configs/nodes/pool6.yaml`) face aux mesures. Un ecart relatif superieur a "
        f"{threshold:.0%} est marque **A REVISER** : reportez la valeur mesuree dans le profil puis re-entrainez.",
        "",
        "| noeud | parametre | hypothese | mesure | ecart | decision |",
        "|---|---|---|---|---|---|",
    ]
    for node, suggestion in suggestions.items():
        profile = profiles.get(node)
        for key, measured in suggestion.items():
            assumed = getattr(profile, key, None) if profile else None
            if assumed in (None, 0):
                lines.append(f"| {node} | {key} | {assumed} | {measured} | - | a renseigner |")
                continue
            gap = (measured - assumed) / assumed
            lines.append(
                f"| {node} | {key} | {assumed} | {measured} | {gap:+.0%} | {'**A REVISER**' if abs(gap) > threshold else 'ok'} |"
            )
    lines += [
        "",
        "Energie : les VM ne fournissent aucune mesure de puissance ; `p_idle_w` / `p_max_w` restent des "
        "hypotheses (modele lineaire, valeurs de reference Raspberry Pi pour fogA). A presenter comme telles.",
    ]
    if warnings:
        lines += ["", "## Avertissements", *[f"- {w}" for w in warnings]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="dossier produit par fogrl.calibration.measure")
    parser.add_argument("--out", default="configs/calibration")
    parser.add_argument("--nodes-file", default="configs/nodes/pool6.yaml")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = parser.parse_args(argv)

    data = Path(args.data)
    raw = pd.read_csv(data / "raw.csv")
    rtt = pd.read_csv(data / "rtt.csv") if (data / "rtt.csv").exists() else None
    meta = json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else {}
    nominal = meta.get("nominal_work_ms") or {r.name: r.work_ms for r in load_requests("configs/requests.yaml")}
    profiles = {p.node_id: p for p in load_nodes(args.nodes_file)[0]}
    calibration, suggestions, warnings = build_calibration(
        raw, rtt, profiles, nominal, source=str(data), min_samples=args.min_samples
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (out / "suggested_nodes.yaml").write_text(
        "# Valeurs mesurees a reporter dans configs/nodes/pool6.yaml (voir report.md)\n"
        + yaml.safe_dump({"nodes": suggestions}, sort_keys=True),
        encoding="utf-8",
    )
    (out / "report.md").write_text(render_report(suggestions, profiles, warnings), encoding="utf-8")
    print(f"calibration.json, suggested_nodes.yaml et report.md ecrits dans {out}")
    for warning in warnings:
        print("ATTENTION :", warning)


if __name__ == "__main__":
    main()
