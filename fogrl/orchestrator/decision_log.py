"""Journal JSONL des decisions (une ligne par requete) : analyse hors ligne et rejeu."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class DecisionLog:
    def __init__(self, path: Optional[str], flush_every: int = 50) -> None:
        self.path = path
        self.flush_every = flush_every
        self._pending = 0
        self._handle = None
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(target, "a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._handle is not None

    def write(self, record: dict) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._pending = 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
