"""Generazione finestre temporali (input -> target) e controlli anti-leakage.

Tutte le finestre sono confinate rigorosamente all'interno di ciascuna singola traiettoria:
nessuna finestra può attraversare il confine temporale tra due traiettorie indipendenti.
Include il supporto all'allineamento target per il confronto equo multi-contesto (Sfida 1)
e al subsampling continuo per la super-risoluzione temporale (Sfida 2).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_starts(n_points: int, span: int, stride: int) -> np.ndarray:
    """Calcola gli indici temporali di inizio delle finestre all'interno di una traiettoria.

    Garantisce che nessuna finestra superi l'ultimo punto temporale della traiettoria (span <= n_points).

    Parametri:
        n_points: Numero totale di punti temporali per traiettoria (1001).
        span: Lunghezza totale della finestra (t_in + t_out).
        stride: Passo di scorrimento temporale tra finestre consecutive (default: 10).

    Restituisce:
        Array 1D di indici interi [0, stride, 2*stride, ...].
    """
    if span > n_points:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_points - span + 1, stride)


def make_windows(
    data: np.ndarray,
    idx: np.ndarray,
    t_in: int,
    t_out: int,
    stride: int,
    align_t_in: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estrae finestre temporali in-memory a frequenza nativa: X (input) e Y (target futuro).

    Sfrutta la vista strided vettorializzata (sliding_window_view) per estrarre decine di migliaia
    di finestre in RAM in frazioni di secondo senza duplicazioni di memoria.

    Allineamento Target (Sfida 1):
        Se ``align_t_in`` è specificato (es. 40), le finestre con t_in più breve (es. 5, 10, 20)
        vengono traslate in avanti di un offset = (align_t_in - t_in). In questo modo, per qualsiasi
        t_in, il target futuro Y inizia sempre allo stesso istante fisico (align_t_in * dt),
        garantendo un benchmark rigoroso e non affetto da bias di transiente.

    Parametri:
        data: Tensore 3D del dataset con shape (400, 1001, 55).
        idx: Indici delle traiettorie da finestrare (es. partizione train, val o test).
        t_in: Numero di punti passati osservati come input.
        t_out: Numero di punti futuri da prevedere (default: 40).
        stride: Passo di avanzamento della finestra temporale.
        align_t_in: Lunghezza massima di contesto per allineare temporalmente i target futuri.

    Restituisce:
        - X: Tensore float32 (n_finestre, t_in, 55)
        - Y: Tensore float32 (n_finestre, t_out, 55)
        - meta: Array int64 (n_finestre, 2) con [id_traiettoria, indice_inizio] per tracciabilità
    """
    span = t_in + t_out
    n_points, n_feat = data.shape[1], data.shape[2]
    if align_t_in is None:
        starts = window_starts(n_points, span, stride)
    else:
        if align_t_in < t_in:
            raise ValueError(f"align_t_in ({align_t_in}) deve essere >= t_in ({t_in})")
        offset = align_t_in - t_in
        starts = offset + window_starts(n_points - offset, span, stride)

    xs, ys, metas = [], [], []
    for tid in np.asarray(idx):
        # Vista vettorizzata: (n_points - span + 1, n_feat, span) -> trasposta a (n_win, span, n_feat)
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
    """Genera finestre temporali per la Super-Risoluzione Temporale (Sfida 2).

    Simula una condizione sperimentale in cui la dinamica passata è campionata a frequenza dimezzata
    (passo 2 * dt = 0.04s) estraendo t_in punti radi su uno span continuo di (t_in - 1) * factor + 1
    punti densi, mentre il target futuro Y è richiesto alla piena frequenza nativa (dt = 0.02s).

    Parametri:
        data: Tensore 3D del dataset (400, 1001, 55).
        idx: Indici delle traiettorie da campionare.
        t_in: Numero di punti radi osservati in input (es. 20 punti).
        t_out: Orizzonte futuro denso (40 punti).
        stride: Passo di avanzamento della finestra.
        factor: Fattore di subsampling temporale (default: 2).
        return_dense_input: Se True, restituisce anche la dinamica continua passata (ground truth denso).

    Restituisce:
        X (input rado), Y (target denso), meta, e opzionalmente X_dense.
    """
    obs_span = (t_in - 1) * factor + 1  # Intervallo temporale continuo coperto dall'input
    span = obs_span + t_out
    n_points, n_feat = data.shape[1], data.shape[2]
    starts = window_starts(n_points, span, stride)

    xs, ys, xds, metas = [], [], [], []
    for tid in np.asarray(idx):
        w = sliding_window_view(data[tid], span, axis=0)[starts].transpose(0, 2, 1)
        dense_in = w[:, :obs_span]
        xs.append(dense_in[:, ::factor])  # Sottocampionamento periodico x2
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


def target_time_index(meta: np.ndarray, t_in: int, t_out: int) -> np.ndarray:
    """Restituisce la matrice degli indici temporali assoluti (n_finestre, t_out) associati ai target."""
    return meta[:, 1:2] + t_in + np.arange(t_out)[None, :]


def decoder_inputs(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Costruisce gli input del decoder autoregressivo in regime di Teacher Forcing.

    Shifta il target Y a destra di un passo temporale, inserendo come primo token (t=0)
    l'ultimo istante osservato della sequenza di input (X[:, -1:]):
        dec_in = [X[:, -1:], Y[:, :-1]] con shape identica a Y: (n_finestre, t_out, 55).
    """
    return np.concatenate([X[:, -1:], Y[:, :-1]], axis=1)


def check_no_leakage(metas: dict[str, np.ndarray]) -> None:
    """Verifica formalmente l'assenza totale di data-leakage tra partizioni Train, Val e Test.

    Controlla che l'intersezione tra gli insiemi degli ID di traiettoria delle diverse partizioni sia vuota.
    """
    sets = {k: set(np.unique(m[:, 0]).tolist()) for k, m in metas.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sets[a] & sets[b]
            assert not common, f"Data leakage rilevato tra {a} e {b}: traiettorie {sorted(common)[:10]}"
