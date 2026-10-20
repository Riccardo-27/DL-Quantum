"""Calcolo metriche globali ed errori disaggregati (per variabile, qubit e orizzonte).

Fornisce funzioni vettorializzate NumPy per valutare l'accuratezza predittiva dei modelli:
- Metriche aggregate: MSE, MAE, RMSE.
- Errori disaggregati fisicamente: magnetizzazioni a singolo sito (10 variabili, indici 0..9)
  vs correlazioni a due corpi (45 variabili, indici 10..54).
- Errori disaggregati spazialmente e temporalmente: profilo per qubit lungo la catena,
  matrice simmetrica 10x10 delle coppie (i, j) e accumulo dell'errore lungo i 40 passi futuri.
"""

from __future__ import annotations

import numpy as np

from .config import CORR_SLICE, MAG_SLICE, N_QUBITS
from .data import corr_pairs


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valida shape e dimensioni dei tensori di confronto prima del calcolo delle metriche.

    Richiede tensori tridimensionali a float64 con shape (n_finestre, t_out, 55).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape diverse: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"Attesi array (n_win, t_out, n_feat), ricevuto ndim={y_true.ndim}")
    return y_true, y_pred


def mse(y_true, y_pred) -> float:
    """Mean Squared Error (MSE) medio complessivo su tutte le finestre, timestep e feature."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error (MAE) medio complessivo su tutte le finestre, timestep e feature."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def per_variable_mse(y_true, y_pred) -> np.ndarray:
    """Calcola l'MSE singolarmente per ciascuna delle 55 variabili fisiche.

    Restituisce:
        Array 1D di 55 elementi, dove [0..9] corrispondono a m_1..m_10 e [10..54] a c_ij.
    """
    y_true, y_pred = _check(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2, axis=(0, 1))


def per_horizon_mse(y_true, y_pred) -> np.ndarray:
    """Calcola l'MSE mediato sulle feature per ciascuno dei t_out passi temporali futuri.

    Utile per quantificare l'accumulo dell'errore (drift) e la divergenza in free-running.

    Restituisce:
        Array 1D di t_out elementi (es. 40 passi temporali).
    """
    y_true, y_pred = _check(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2, axis=(0, 2))


def per_horizon_group_mse(y_true, y_pred) -> dict[str, np.ndarray]:
    """Separa la traiettoria temporale dell'errore tra magnetizzazioni (10) e correlazioni (45).

    Restituisce:
        Dizionario con array 1D di t_out punti:
        - 'mag': errore quadratico medio delle sole 10 magnetizzazioni a ogni passo futuro.
        - 'corr': errore quadratico medio delle sole 45 correlazioni a ogni passo futuro.
    """
    y_true, y_pred = _check(y_true, y_pred)
    se = (y_true - y_pred) ** 2
    return {
        "mag": se[..., MAG_SLICE].mean(axis=(0, 2)),
        "corr": se[..., CORR_SLICE].mean(axis=(0, 2)),
    }


def per_qubit_mse(y_true, y_pred) -> np.ndarray:
    """Calcola l'MSE delle sole magnetizzazioni locali per ciascuno dei 10 atomi di Rydberg.

    Restituisce:
        Array 1D di 10 elementi: MSE per il qubit 1, qubit 2, ..., qubit 10.
    """
    return per_variable_mse(y_true, y_pred)[MAG_SLICE]


def corr_pair_mse_matrix(y_true, y_pred, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Costruisce la matrice simmetrica 10x10 degli errori sulle correlazioni c_ij.

    La diagonale principale è impostata a NaN poiché le auto-correlazioni <sigma_z^i sigma_z^i> = 1
    sono costanti banali non incluse nel dataset delle 45 coppie (i < j).

    Restituisce:
        Array 2D di forma (10, 10).
    """
    errs = per_variable_mse(y_true, y_pred)[CORR_SLICE]
    mat = np.full((n_qubits, n_qubits), np.nan)
    for e, (i, j) in zip(errs, corr_pairs(n_qubits)):
        mat[i - 1, j - 1] = mat[j - 1, i - 1] = e
    return mat


def evaluate(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """Valutazione esaustiva che restituisce un dizionario completo di metriche scalari.

    Include metriche complessive, disaggregate per gruppo fisico e agli estremi dell'orizzonte:
    - 'mse': errore quadratico medio globale
    - 'mae': errore assoluto medio globale
    - 'rmse': radice dell'errore quadratico medio
    - 'mse_mag' / 'mae_mag': errore medio sulle 10 magnetizzazioni locali
    - 'mse_corr' / 'mae_corr': errore medio sulle 45 correlazioni
    - 'mse_first_step': errore al passo futuro t=1 (orizzonte immediato)
    - 'mse_last_step': errore al passo futuro t=40 (orizzonte lungo)

    Parametri:
        y_true: Target reali (n_finestre, t_out, 55).
        y_pred: Predizioni del modello (n_finestre, t_out, 55).
        prefix: Prefisso opzionale per le chiavi del dizionario (es. 'val_', 'test_').
    """
    y_true, y_pred = _check(y_true, y_pred)
    err = y_true - y_pred
    se, ae = err**2, np.abs(err)
    out = {
        "mse": se.mean(),
        "mae": ae.mean(),
        "rmse": np.sqrt(se.mean()),
        "mse_mag": se[..., MAG_SLICE].mean(),
        "mse_corr": se[..., CORR_SLICE].mean(),
        "mae_mag": ae[..., MAG_SLICE].mean(),
        "mae_corr": ae[..., CORR_SLICE].mean(),
        "mse_first_step": se[:, 0].mean(),
        "mse_last_step": se[:, -1].mean(),
    }
    return {f"{prefix}{k}": float(v) for k, v in out.items()}
