#!/usr/bin/env python3
"""Lance la pile complete sans Docker (workers + orchestrateur) : voir fogrl/deploy/local.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fogrl.deploy.local import main  # noqa: E402

if __name__ == "__main__":
    main()
