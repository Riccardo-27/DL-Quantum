"""Costruzione delle finestre (input -> target) per traiettoria.

Ogni finestra e' contenuta in una sola traiettoria: si lavora sempre
traiettoria per traiettoria, quindi non c'e' modo di "attraversarne" due.
Insieme a X e Y viene restituito ``meta`` con ``(traj_id, start)`` per ogni
finestra, utile per i controlli di leakage e per ritrovare le finestre nei
grafici.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_starts(n_points: int, span: int, stride: int) -> np.ndarray:
    """Indici di partenza validi per una finestra lunga ``span``."""
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
    """Finestre a risoluzione nativa.

    Parameters
    ----------
    data : ``(n_traj, n_points, n_feat)``
    idx : indici delle traiettorie da usare (es. ``split['train']``)

    Returns
    -------
    X : ``(n_win, t_in, n_feat)``
    Y : ``(n_win, t_out, n_feat)``
    meta : ``(n_win, 2)`` int con ``(traj_id, start)``
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
    """Finestre per la super-resolution (sfida 2).

    L'input contiene ``t_in`` punti osservati ogni ``factor`` step
    (dt_in = factor * dt). Il target sono i ``t_out`` punti a risoluzione
    nativa che seguono immediatamente l'ultimo punto osservato.

    Con ``return_dense_input=True`` restituisce anche la finestra di input a
    risoluzione nativa (``(n_win, (t_in-1)*factor+1, n_feat)``), utile per i
    grafici "punti osservati vs ricostruiti" o per un target di ricostruzione.

    Returns
    -------
    X, Y, meta  (+ X_dense se richiesto)
    """
    obs_span = (t_in - 1) * factor + 1          # dal primo all'ultimo osservato
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
    """Input del decoder per il teacher forcing: target spostato di un passo.

    ``dec_in[:, 0] = X[:, -1]`` (ultimo punto osservato),
    ``dec_in[:, k] = Y[:, k-1]`` per ``k >= 1``.
    """
    return np.concatenate([X[:, -1:], Y[:, :-1]], axis=1)


def check_no_leakage(metas: dict[str, np.ndarray]) -> None:
    """Verifica che le partizioni non condividano traiettorie.

    ``metas`` e' ad esempio ``{'train': meta_tr, 'val': meta_va, 'test': meta_te}``.
    Solleva ``AssertionError`` se due partizioni hanno una traiettoria in comune.
    """
    sets = {k: set(np.unique(m[:, 0]).tolist()) for k, m in metas.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sets[a] & sets[b]
            assert not common, f"Leakage tra {a} e {b}: traiettorie {sorted(common)[:10]}"
