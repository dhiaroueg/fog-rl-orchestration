"""Figures du memoire (cahier, section 8.5), generees UNIQUEMENT par script depuis les CSV.

    python -m fogrl.evaluation.figures --results results/sim_full/results.csv \\
        --convergence models/n3 --out figures/
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import aggregate


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_convergence(npz_by_algo: Mapping[str, Sequence[str]], out: Path) -> Path:
    """Courbes de convergence : recompense moyenne par episode (moyenne +- ecart-type sur les graines)."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, paths in sorted(npz_by_algo.items()):
        curves = []
        for path in paths:
            data = np.load(path)
            curves.append((data["timesteps"], data["results"].mean(axis=1)))
        length = min(len(c[1]) for c in curves)
        steps = curves[0][0][:length]
        values = np.stack([c[1][:length] for c in curves])
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(steps, mean, label=f"{algo.upper()} ({len(curves)} graines)")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Pas d'entrainement")
    ax.set_ylabel("Recompense moyenne par episode")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_latency_by_policy(df: pd.DataFrame, out: Path, size: Optional[int] = None) -> Path:
    """Latence moyenne et P95 par politique et par scenario (barres, IC 95 %)."""

    plt = _plt()
    frame = df if size is None else df[df["size"] == size]
    agg = aggregate(frame, ["latency_mean_ms", "latency_p95_ms"], by=("policy", "scenario"))
    scenarios = list(dict.fromkeys(agg["scenario"]))
    policies = list(dict.fromkeys(agg["policy"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    width = 0.8 / len(policies)
    for ax, metric, title in zip(axes, ("latency_mean_ms", "latency_p95_ms"), ("Latence moyenne", "Latence P95")):
        for k, policy in enumerate(policies):
            rows = agg[agg["policy"] == policy].set_index("scenario").reindex(scenarios)
            ax.bar(
                np.arange(len(scenarios)) + k * width,
                rows[f"{metric}_mean"],
                width,
                yerr=rows[f"{metric}_ci95"].fillna(0),
                label=policy,
                capsize=2,
            )
        ax.set_xticks(np.arange(len(scenarios)) + 0.4 - width / 2)
        ax.set_xticklabels(scenarios, rotation=20)
        ax.set_ylabel("ms")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7, ncol=2)
    return _save(fig, out)


def plot_latency_cdf(latencies: Mapping[str, np.ndarray], out: Path) -> Path:
    """Fonction de repartition de la latence : meilleur RL vs meilleure baseline."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in latencies.items():
        x = np.sort(np.asarray(values))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Latence (ms)")
    ax.set_ylabel("Fraction des requetes")
    ax.set_title("CDF de la latence")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_pareto(df: pd.DataFrame, out: Path, label_column: str = "w2") -> Path:
    """Compromis energie / latence selon les poids de la recompense (etude d'ablation)."""

    plt = _plt()
    agg = df.groupby(label_column)[["latency_mean_ms", "energy_fog_j_per_req"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(agg["energy_fog_j_per_req"], agg["latency_mean_ms"], "o-")
    for weight, row in agg.iterrows():
        ax.annotate(f"{label_column}={weight}", (row["energy_fog_j_per_req"], row["latency_mean_ms"]), fontsize=8)
    ax.set_xlabel("Energie fog (J/requete)")
    ax.set_ylabel("Latence moyenne (ms)")
    ax.set_title("Compromis energie / latence")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_sim_vs_real(sim: pd.DataFrame, real: pd.DataFrame, out: Path, metric: str = "latency_mean_ms") -> Path:
    """Comparaison simulation / reel par politique (moyenne sur les scenarios communs)."""

    plt = _plt()
    sim_mean = sim.groupby("policy")[metric].mean().rename("simulation")
    real_mean = real.groupby("policy")[metric].mean().rename("reel")
    joined = pd.concat([sim_mean, real_mean], axis=1, join="inner")
    fig, ax = plt.subplots(figsize=(7, 4))
    joined.plot.bar(ax=ax)
    ax.set_ylabel("Latence moyenne (ms)")
    ax.set_title("Simulation vs reel")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", help="results.csv de la matrice d'experiences")
    parser.add_argument("--convergence", help="dossier de niveau, ex. models/n3 (cherche */seed*/evaluations.npz)")
    parser.add_argument("--ablation", help="CSV d'ablation (colonne w2)")
    parser.add_argument("--real", help="CSV de resultats reels (avec --results : figure sim vs reel)")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)
    out = Path(args.out)
    made = []
    if args.convergence:
        by_algo = {}
        for path in glob.glob(f"{args.convergence}/*/seed*/evaluations.npz"):
            by_algo.setdefault(Path(path).parts[-3], []).append(path)
        if by_algo:
            made.append(plot_convergence(by_algo, out / "convergence.png"))
    if args.results:
        df = pd.read_csv(args.results)
        made.append(plot_latency_by_policy(df, out / "latency_by_policy.png"))
        if args.real:
            made.append(plot_sim_vs_real(df, pd.read_csv(args.real), out / "sim_vs_real.png"))
    if args.ablation:
        made.append(plot_pareto(pd.read_csv(args.ablation), out / "pareto.png"))
    print("Figures :", *map(str, made), sep="\n  ")


if __name__ == "__main__":
    main()
