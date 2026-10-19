"""Calcolo metriche globali ed errori disaggregati (per variabile, qubit e orizzonte)."""

from __future__ import annotations

import numpy as np

from .config import CORR_SLICE, MAG_SLICE, N_QUBITS
from .data import corr_pairs


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape diverse: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"Attesi array (n_win, t_out, n_feat), ricevuto ndim={y_true.ndim}")
    return y_true, y_pred


def mse(y_true, y_pred) -> float:
    """Mean Squared Error medio complessivo."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error medio complessivo."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def per_variable_mse(y_true, y_pred) -> np.ndarray:
    """Calcola l'MSE singolarmente per ciascuna delle 55 variabili."""
    y_true, y_pred = _check(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2, axis=(0, 1))


def per_horizon_mse(y_true, y_pred) -> np.ndarray:
    """Calcola l'MSE lungo i 40 istanti temporali futuri."""
    y_true, y_pred = _check(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2, axis=(0, 2))


def per_horizon_group_mse(y_true, y_pred) -> dict[str, np.ndarray]:
    """Separa la curva temporale dell'MSE tra magnetizzazioni (10) e correlazioni (45)."""
    y_true, y_pred = _check(y_true, y_pred)
    se = (y_true - y_pred) ** 2
    return {
        "mag": se[..., MAG_SLICE].mean(axis=(0, 2)),
        "corr": se[..., CORR_SLICE].mean(axis=(0, 2)),
    }


def per_qubit_mse(y_true, y_pred) -> np.ndarray:
    """Calcola l'MSE delle sole magnetizzazioni per i 10 singoli qubit."""
    return per_variable_mse(y_true, y_pred)[MAG_SLICE]


def corr_pair_mse_matrix(y_true, y_pred, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Costruisce la matrice simmetrica 10x10 degli errori sulle correlazioni a due punti."""
    errs = per_variable_mse(y_true, y_pred)[CORR_SLICE]
    mat = np.full((n_qubits, n_qubits), np.nan)
    for e, (i, j) in zip(errs, corr_pairs(n_qubits)):
        mat[i - 1, j - 1] = mat[j - 1, i - 1] = e
    return mat


def evaluate(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """Calcola un dizionario completo di metriche scalari (MSE, MAE, RMSE, errori disaggregati)."""
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
