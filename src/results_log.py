"""Logging degli esperimenti su CSV con supporto multi-autore su Drive."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import get_author, get_paths


def log_path(author: str | None = None) -> Path:
    author = author or get_author()
    return get_paths()["results"] / f"results_log_{author}.csv"


def _serialize(v):
    return json.dumps(v) if isinstance(v, (list, dict, tuple)) else v


def append_result(record: dict, path: str | Path | None = None) -> Path:
    """Salva una riga di metriche sul CSV dell'autore, serializzando strutture dati e allineando le colonne."""
    path = Path(path) if path else log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "author": get_author()}
    row.update({k: _serialize(v) for k, v in record.items()})
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)
        if set(new.columns) <= set(old.columns):
            new.reindex(columns=old.columns).to_csv(path, mode="a", header=False, index=False)
            return path
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)
    return path


def load_results(results_dir: str | Path | None = None) -> pd.DataFrame:
    """Unisce tutti i log generati dagli autori in un unico DataFrame complessivo."""
    results_dir = Path(results_dir) if results_dir else get_paths()["results"]
    files = sorted(results_dir.glob("results_log_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
