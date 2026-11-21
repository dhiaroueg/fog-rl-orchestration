#!/usr/bin/env python3
"""Genere deploy/ (Compose, Prometheus, Grafana) : voir fogrl/deploy/generate.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fogrl.deploy.generate import main  # noqa: E402

if __name__ == "__main__":
    main()
