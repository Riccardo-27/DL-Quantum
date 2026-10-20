"""Statistiche descrittive e intervalli di confidenza al 95% con distribuzione t di Student.

Fornisce stime statistiche rigorose per esperimenti replicati su seed multipli (tipicamente n=3).
Dato il numero ridotto di gradi di libertà (df = n - 1 = 2), l'approssimazione normale asintotica
(z = 1.96) sottostima gravemente l'incertezza campionaria: questo modulo adotta la corretta
distribuzione t di Student bicaudale (quantile t_{0.975, 2} ≈ 4.303 per intervalli al 95%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st


def t_critical(n: int, confidence: float = 0.95) -> float:
    """Restituisce il valore critico t per un intervallo di confidenza bicaudale con df = n - 1.

    Parametri:
        n: Dimensione del campione (numero di repliche/seed).
        confidence: Livello di confidenza desiderato (default: 0.95 per il 95%).

    Restituisce:
        Valore critico t (float), oppure NaN se n < 2.
    """
    if n < 2:
        return float("nan")
    # Per confidence=0.95 e n=3: df=2, ppf(0.975) ≈ 4.30265
    return float(_st.t.ppf(0.5 + confidence / 2, df=n - 1))


def summarize(values, confidence: float = 0.95) -> dict[str, float]:
    """Calcola media, deviazione standard campionaria (ddof=1) e semi-ampiezza dell'intervallo di confidenza.

    La formula dell'intervallo è:
        IC = media ± t_{crit} * (std / sqrt(n))

    Parametri:
        values: Array o lista di valori scalari (ignora eventuali valori NaN).
        confidence: Livello di confidenza (default: 0.95).

    Restituisce:
        Dizionario con:
        - 'n': numero di campioni validi
        - 'mean': media campionaria
        - 'std': deviazione standard con correzione di Bessel (ddof=1)
        - 'ci_half': semi-ampiezza dell'intervallo di confidenza
        - 'ci_low': estremo inferiore dell'intervallo
        - 'ci_high': estremo superiore dell'intervallo
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    n = v.size
    mean = float(v.mean()) if n else float("nan")
    std = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = t_critical(n, confidence) * std / np.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_half": float(half),
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def format_mean_ci(values, digits: int = 4, confidence: float = 0.95) -> str:
    """Formatta la stima campionaria e la sua incertezza per tabelle e report scientifici.

    Esempio di output:
        "0.000161 ± 0.000001" (con digits=6)

    Parametri:
        values: Sequenza di metriche su cui calcolare la statistica.
        digits: Numero di cifre decimali dopo la virgola.
        confidence: Livello di confidenza per l'intervallo t di Student.
    """
    s = summarize(values, confidence)
    return f"{s['mean']:.{digits}f} ± {s['ci_half']:.{digits}f}"


def summarize_df(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggrega un DataFrame di esperimenti calcolando media e semi-intervallo di confidenza per ogni gruppo.

    Parametri:
        df: Tabella dei risultati sperimentali.
        group_cols: Colonne di raggruppamento (es. ['model_name', 'window']).
        metric_cols: Metriche da aggregare (es. ['val_mse', 'val_mae']).
        confidence: Livello di confidenza per il quantile di Student.

    Restituisce:
        pd.DataFrame con le colonne aggregate: <metrica>_mean, <metrica>_ci, <metrica>_n.
    """
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for m in metric_cols:
            s = summarize(g[m].to_numpy(), confidence)
            row[f"{m}_mean"] = s["mean"]
            row[f"{m}_ci"] = s["ci_half"]
            row[f"{m}_n"] = s["n"]
        rows.append(row)
    return pd.DataFrame(rows)
