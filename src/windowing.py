"""Generazione finestre temporali (input -> target) e controlli anti-leakage.

Tutte le finestre sono confinate rigorosamente all'interno di una singola traiettoria:
nessuna finestra può attraversare il confine tra due traiettorie indipendenti.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_starts(n_points: int, span: int, stride: int) -> np.ndarray:
    """Calcola gli indici temporali di partenza delle finestre in base allo stride."""
    if span > n_points:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_points - span + 1, stride)


def make_windows(
    data: np.ndarray,
    idx: np.ndarray,
    t_in: int,
    t_out: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estrae finestre temporali a frequenza nativa: X (input osservato) e Y (target futuro).

    Restituisce:
    - X: tensore (n_finestre, t_in, 55)
    - Y: tensore (n_finestre, t_out, 55)
    - meta: array (n_finestre, 2) con [traj_id, start_idx] per tracciabilità
    """
    span = t_in + t_out
    n_points, n_feat = data.shape[1], data.shape[2]
    starts = window_starts(n_points, span, stride)

    xs, ys, metas = [], [], []
    for tid in np.asarray(idx):
        # (n_points - span + 1, n_feat, span) -> (n_win, span, n_feat)
        w = sliding_window_view(data[tid], span, axis=0)[starts].transpose(0, 2, 1)
        xs.append(w[:, :t_in])
        ys.append(w[:, t_in:])
        metas.append(np.stack([np.full_like(starts, tid), starts], axis=1))

    if not xs:
        empty = np.empty((0, 0, n_feat), dtype=data.dtype)
        return empty, empty, np.empty((0, 2), dtype=np.int64)
    X = np.ascontiguousarray(np.concatenate(xs), dtype=np.float32)
    Y = np.ascontiguousarray(np.concatenate(ys), dtype=np.float32)
    return X, Y, np.concatenate(metas)


def make_sr_windows(
    data: np.ndarray,
    idx: np.ndarray,
    t_in: int,
    t_out: int,
    stride: int,
    factor: int = 2,
    return_dense_input: bool = False,
):
    """Genera finestre per super-risoluzione: input sottocampionato (x2) e target a frequenza nativa."""
    obs_span = (t_in - 1) * factor + 1          # intervallo continuo coperto dall'input
    span = obs_span + t_out
    n_points, n_feat = data.shape[1], data.shape[2]
    starts = window_starts(n_points, span, stride)

    xs, ys, xds, metas = [], [], [], []
    for tid in np.asarray(idx):
        w = sliding_window_view(data[tid], span, axis=0)[starts].transpose(0, 2, 1)
        dense_in = w[:, :obs_span]
        xs.append(dense_in[:, ::factor])
        ys.append(w[:, obs_span:])
        if return_dense_input:
            xds.append(dense_in)
        metas.append(np.stack([np.full_like(starts, tid), starts], axis=1))

    X = np.ascontiguousarray(np.concatenate(xs), dtype=np.float32)
    Y = np.ascontiguousarray(np.concatenate(ys), dtype=np.float32)
    meta = np.concatenate(metas)
    if return_dense_input:
        return X, Y, meta, np.ascontiguousarray(np.concatenate(xds), dtype=np.float32)
    return X, Y, meta


def decoder_inputs(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Prepara gli input del decoder (teacher forcing): target shiftato di un passo temporale."""
    return np.concatenate([X[:, -1:], Y[:, :-1]], axis=1)


def check_no_leakage(metas: dict[str, np.ndarray]) -> None:
    """Verifica formalmente che non ci sia sovrapposizione di traiettorie tra Train, Val e Test."""
    sets = {k: set(np.unique(m[:, 0]).tolist()) for k, m in metas.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sets[a] & sets[b]
            assert not common, f"Leakage tra {a} e {b}: traiettorie {sorted(common)[:10]}"
