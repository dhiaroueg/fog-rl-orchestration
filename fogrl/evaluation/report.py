"""Rapport de recette : criteres ENF-02, ENF-03, H1, H2 et OS-4 calcules depuis un CSV de resultats.

    python -m fogrl.evaluation.report --results results/sim_full/results.csv --size 4 --out results/sim_full/recette_n4.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .stats import acceptance_report, render_markdown


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--load-scenarios", default="medium,high", help="scenarios de charge pour H2")
    parser.add_argument("--out", default=None, help="fichier Markdown (defaut : affichage)")
    args = parser.parse_args(argv)
    report = acceptance_report(
        pd.read_csv(args.results), args.size, tuple(s for s in args.load_scenarios.split(",") if s)
    )
    text = render_markdown(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Rapport -> {args.out}")
    print(text)


if __name__ == "__main__":
    main()
