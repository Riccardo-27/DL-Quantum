"""Lettura, validazione e partizionamento del dataset di traiettorie quantistiche."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DT, N_FEATURES, N_POINTS, N_QUBITS, N_TRAJ, get_paths

# ---------------------------------------------------------------------------
# Nomi e mappatura delle 55 feature osservabili
# ---------------------------------------------------------------------------


def corr_pairs(n_qubits: int = N_QUBITS) -> list[tuple[int, int]]:
    """Genera le 45 coppie (i, j) con i < j ordinate secondo le colonne del dataset."""
    return [(i, j) for i in range(1, n_qubits + 1) for j in range(i + 1, n_qubits + 1)]


def feature_names(n_qubits: int = N_QUBITS) -> list[str]:
    """Restituisce i 55 nomi delle variabili (10 magnetizzazioni e 45 correlazioni)."""
    mags = [f"m{i}" for i in range(1, n_qubits + 1)]
    corrs = [f"c{i}_{j}" for i, j in corr_pairs(n_qubits)]
    return mags + corrs


# ---------------------------------------------------------------------------
# Caricamento e verifiche di integrità
# ---------------------------------------------------------------------------


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Carica il file CSV grezzo senza header (400400 righe x 56 colonne)."""
    return pd.read_csv(path, header=None, index_col=False)


def validate_raw(
    df: pd.DataFrame,
    n_traj: int = N_TRAJ,
    n_points: int = N_POINTS,
    dt: float = DT,
    atol: float = 1e-6,
) -> dict:
    """Verifica l'integrità del dataset: dimensioni, assenza di NaN, griglia temporale e range fisici."""
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


def check_quantum_probabilities(
    data: np.ndarray, n_qubits: int = N_QUBITS, atol: float = 1e-5
) -> dict[str, float | int | bool]:
    """Verifica che le probabilità congiunte a due corpi P(s_i, s_j) siano non-negative (>= 0).

    Applicabile sia ai dati reali che alle predizioni dei modelli (array con shape (..., 55)).
    Per ogni coppia i < j e per ciascuna delle 4 configurazioni di spin:
        P(s_i, s_j) = (1 + s_i * m_i + s_j * m_j + s_i * s_j * c_ij) / 4 >= 0
    """
    flat = np.asarray(data, dtype=np.float64).reshape(-1, N_FEATURES)
    mags = flat[:, :n_qubits]
    corrs = flat[:, n_qubits:]

    min_prob = 1.0
    total_violations = 0
    total_checks = 0

    pairs = corr_pairs(n_qubits)
    for col_idx, (i, j) in enumerate(pairs):
        mi = mags[:, i - 1]
        mj = mags[:, j - 1]
        cij = corrs[:, col_idx]

        p_up_up = (1.0 + mi + mj + cij) / 4.0
        p_up_down = (1.0 + mi - mj - cij) / 4.0
        p_down_up = (1.0 - mi + mj - cij) / 4.0
        p_down_down = (1.0 - mi - mj + cij) / 4.0

        for p in (p_up_up, p_up_down, p_down_up, p_down_down):
            min_prob = min(min_prob, float(np.min(p)))
            total_violations += int(np.sum(p < -atol))
            total_checks += p.size

    return {
        "min_probability": float(min_prob),
        "violations_count": total_violations,
        "total_checks": total_checks,
        "violation_rate": float(total_violations / total_checks) if total_checks else 0.0,
        "is_physical": total_violations == 0,
    }


def to_array(df: pd.DataFrame, n_points: int = N_POINTS) -> np.ndarray:
    """Rimuove la colonna tempo e restituisce il tensore 3D (400, 1001, 55) in float32."""
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return values.reshape(-1, n_points, values.shape[1])


def feature_stats(data: np.ndarray) -> pd.DataFrame:
    """Calcola statistiche descrittive (min, max, media, deviazione standard) per ogni variabile."""
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
    """Carica i dati (400, 1001, 55), salvando o riutilizzando la cache binaria .npy per velocizzare l'I/O."""
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
# Partizionamento delle traiettorie (Train / Val / Test)
# ---------------------------------------------------------------------------


def make_split(
    n_traj: int = N_TRAJ,
    n_train: int = 280,
    n_val: int = 60,
    n_test: int = 60,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Partiziona le 400 traiettorie in 280 Train, 60 Val e 60 Test con seed fisso (zero data leakage)."""
    if n_train + n_val + n_test != n_traj:
        raise ValueError("n_train + n_val + n_test deve essere uguale a n_traj")
    perm = np.random.default_rng(seed).permutation(n_traj)
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }


def save_split(split: dict[str, np.ndarray], path: str | Path | None = None) -> Path:
    """Salva gli indici delle partizioni su file compresso .npz per congelare lo split."""
    path = Path(path) if path else get_paths()["results"] / "split_indices.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **split)
    return path


def load_split(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Carica gli indici congelati di Train, Val e Test dal file .npz, restituendo le chiavi canoniche ('train', 'val', 'test')."""
    path = Path(path) if path else get_paths()["results"] / "split_indices.npz"
    with np.load(path) as f:
        res = {k: f[k] for k in f.files}
        clean_split = {}
        for key in ["train", "val", "test"]:
            if key in res:
                clean_split[key] = res[key]
            elif f"{key}_idx" in res:
                clean_split[key] = res[f"{key}_idx"]
        return clean_split



def kfold_splits(
    indices: np.ndarray, k: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera K fold per la validazione incrociata operando strettamente a livello di traiettoria."""
    perm = np.random.default_rng(seed).permutation(np.asarray(indices))
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        val_idx = np.sort(folds[i])
        train_idx = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((train_idx, val_idx))
    return out
