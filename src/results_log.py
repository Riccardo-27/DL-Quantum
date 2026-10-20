"""Logging degli esperimenti su CSV con supporto multi-autore per Google Drive.

Gestisce la persistenza strutturata dei risultati sperimentali evitando conflitti
di scrittura concorrente su filesystem condivisi mediante file di log separati
per ciascun autore (es. 'results_log_ernesto.csv' e 'results_log_riccardo.csv'),
consentendo l'aggregazione trasparente in lettura in un unico DataFrame globale.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import get_author, get_paths


def log_path(author: str | None = None) -> Path:
    """Restituisce il percorso del file CSV di log associato all'autore specificato.

    Parametri:
        author: Nome dell'autore (es. 'ernesto' o 'riccardo'). Se None, viene
            rilevato dinamicamente dalla variabile d'ambiente DLQ_AUTHOR.
    """
    author = author or get_author()
    return get_paths()["results"] / f"results_log_{author}.csv"


def _serialize(v):
    """Serializza liste, tuple o dizionari in stringhe JSON per l'archiviazione tabellare."""
    return json.dumps(v) if isinstance(v, (list, dict, tuple)) else v


def append_result(record: dict, path: str | Path | None = None) -> Path:
    """Appende in modo atomico una riga di metriche e iperparametri al file di log.

    Gestisce l'evoluzione dinamica dello schema del log: se un nuovo modello
    produce metriche aggiuntive, le colonne esistenti vengono preservate e il
    file viene aggiornato con la nuova intestazione allineata.

    Parametri:
        record: Dizionario contenente iperparametri e metriche calcolate.
        path: Percorso opzionale del file di destinazione (default: log dell'autore corrente).

    Restituisce:
        Path del file CSV aggiornato.
    """
    path = Path(path) if path else log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Metadati standard di esecuzione
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "author": get_author(),
    }
    row.update({k: _serialize(v) for k, v in record.items()})
    new = pd.DataFrame([row])

    if path.exists():
        old = pd.read_csv(path)
        # Se le colonne coincidono, append veloce senza riscrittura completa
        if set(new.columns) <= set(old.columns):
            new.reindex(columns=old.columns).to_csv(path, mode="a", header=False, index=False)
            return path
        # Se sono presenti nuove colonne, unione con reindicizzazione trasparente
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)
    return path


def load_results(results_dir: str | Path | None = None) -> pd.DataFrame:
    """Carica e concatena tutti i log sperimentali generati dai diversi autori.

    Parametri:
        results_dir: Cartella contenente i file results_log_*.csv (default: results/).

    Restituisce:
        pd.DataFrame globale contenente lo storico unificato di tutti i run eseguiti.
    """
    results_dir = Path(results_dir) if results_dir else get_paths()["results"]
    files = sorted(results_dir.glob("results_log_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
