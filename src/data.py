"""Lettura, verifica e split del dataset ``trajectories.csv``.

Formato (data/README): 400 traiettorie x 1001 righe, 56 colonne
``[t, m_1..m_10, c_12..c_1N, c_23..c_2N, ..., c_(N-1)N]``.
Una nuova traiettoria inizia a ogni ``t = 0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DT, N_FEATURES, N_POINTS, N_QUBITS, N_TRAJ, get_paths

# ---------------------------------------------------------------------------
# Nomi delle feature
# ---------------------------------------------------------------------------


def corr_pairs(n_qubits: int = N_QUBITS) -> list[tuple[int, int]]:
    """Coppie (i, j) 1-based nell'ordine delle colonne: c12..c1N, c23..c2N, ..."""
    return [(i, j) for i in range(1, n_qubits + 1) for j in range(i + 1, n_qubits + 1)]


def feature_names(n_qubits: int = N_QUBITS) -> list[str]:
    """``['m1', ..., 'm10', 'c1_2', 'c1_3', ..., 'c9_10']`` (55 nomi)."""
    mags = [f"m{i}" for i in range(1, n_qubits + 1)]
    corrs = [f"c{i}_{j}" for i, j in corr_pairs(n_qubits)]
    return mags + corrs


# ---------------------------------------------------------------------------
# Lettura e verifica
# ---------------------------------------------------------------------------


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Legge il CSV come il notebook del professore (nessun header)."""
    return pd.read_csv(path, header=None, index_col=False)


def validate_raw(
    df: pd.DataFrame,
    n_traj: int = N_TRAJ,
    n_points: int = N_POINTS,
    dt: float = DT,
    atol: float = 1e-6,
) -> dict:
    """Esegue le verifiche dello step S2 e restituisce un report.

    Non solleva eccezioni: ogni check produce ``True/False`` cosi' le anomalie
    possono essere documentate nel ``decision_log`` invece di bloccare tutto.
    """
    report: dict = {}
    n_cols_expected = 1 + N_FEATURES
    report["shape"] = df.shape
    report["shape_ok"] = df.shape == (n_traj * n_points, n_cols_expected)
    report["n_nan"] = int(df.isna().sum().sum())
    report["no_nan"] = report["n_nan"] == 0

    t = df.iloc[:, 0].to_numpy(dtype=np.float64)
    report["n_traj_detected"] = int(np.sum(np.isclose(t, 0.0, atol=atol)))
    report["n_traj_ok"] = report["n_traj_detected"] == n_traj

    if t.size == n_traj * n_points:
        t_grid = t.reshape(n_traj, n_points)
        expected = np.arange(n_points) * dt
        max_dev = float(np.max(np.abs(t_grid - expected)))
        report["time_max_abs_dev"] = max_dev
        report["time_grid_ok"] = max_dev <= atol
    else:
        report["time_max_abs_dev"] = None
        report["time_grid_ok"] = False

    values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    report["min"] = float(np.nanmin(values))
    report["max"] = float(np.nanmax(values))
    report["in_unit_range"] = bool(report["min"] >= -1.0 and report["max"] <= 1.0)

    report["all_ok"] = all(
        report[k] for k in ("shape_ok", "no_nan", "n_traj_ok", "time_grid_ok")
    )
    return report


def to_array(df: pd.DataFrame, n_points: int = N_POINTS) -> np.ndarray:
    """Scarta la colonna tempo e restituisce ``(n_traj, n_points, 55)`` float32."""
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return values.reshape(-1, n_points, values.shape[1])


def feature_stats(data: np.ndarray) -> pd.DataFrame:
    """Min, max, media e std per variabile, calcolate su tutte le traiettorie."""
    flat = data.reshape(-1, data.shape[-1])
    return pd.DataFrame(
        {
            "min": flat.min(axis=0),
            "max": flat.max(axis=0),
            "mean": flat.mean(axis=0),
            "std": flat.std(axis=0),
        },
        index=feature_names(),
    )


def load_trajectories(
    csv_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Restituisce ``(400, 1001, 55)``. Usa la cache ``.npy`` se presente.

    La prima volta legge il CSV (~464 MB, lento), lo verifica e salva la cache.
    """
    paths = get_paths()
    csv_path = Path(csv_path) if csv_path else paths["data_csv"]
    cache_path = Path(cache_path) if cache_path else paths["cache"] / "trajectories.npy"

    if cache_path.exists():
        return np.load(cache_path)

    df = load_raw_csv(csv_path)
    if validate:
        report = validate_raw(df)
        if not report["all_ok"]:
            raise ValueError(f"Verifica del CSV fallita: {report}")
    data = to_array(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data)
    return data


# ---------------------------------------------------------------------------
# Split per traiettoria
# ---------------------------------------------------------------------------


def make_split(
    n_traj: int = N_TRAJ,
    n_train: int = 280,
    n_val: int = 60,
    n_test: int = 60,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Permuta gli indici delle traiettorie e li divide in train/val/test."""
    if n_train + n_val + n_test != n_traj:
        raise ValueError("n_train + n_val + n_test deve essere uguale a n_traj")
    perm = np.random.default_rng(seed).permutation(n_traj)
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }


def save_split(split: dict[str, np.ndarray], path: str | Path | None = None) -> Path:
    path = Path(path) if path else get_paths()["results"] / "split_indices.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **split)
    return path


def load_split(path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = Path(path) if path else get_paths()["results"] / "split_indices.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def kfold_splits(
    indices: np.ndarray, k: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold *per traiettoria*: restituisce ``[(train_idx, val_idx), ...]``.

    Da usare su ``train ∪ val``; il test set resta sempre fuori.
    """
    perm = np.random.default_rng(seed).permutation(np.asarray(indices))
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        val_idx = np.sort(folds[i])
        train_idx = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((train_idx, val_idx))
    return out
