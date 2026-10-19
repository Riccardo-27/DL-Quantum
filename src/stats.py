"""Statistiche descrittive e intervalli di confidenza al 95% con distribuzione t di Student."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st


def t_critical(n: int, confidence: float = 0.95) -> float:
    """Quantile critico t con n-1 gradi di libertà per la stima dell'incertezza con pochi campioni."""
    if n < 2:
        return float("nan")
    return float(_st.t.ppf(0.5 + confidence / 2, df=n - 1))


def summarize(values, confidence: float = 0.95) -> dict[str, float]:
    """Calcola media, deviazione standard campionaria (ddof=1) e semi-ampiezza dell'IC al 95%."""
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
    """Formatta il risultato per le tabelle di riepilogo nella forma: media ± incertezza."""
    s = summarize(values, confidence)
    return f"{s['mean']:.{digits}f} ± {s['ci_half']:.{digits}f}"


def summarize_df(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggrega i risultati degli esperimenti per gruppo (modello, iperparametri) calcolando media e IC."""
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
