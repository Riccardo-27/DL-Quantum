"""Esportazione grafici standardizzati ad alta risoluzione (JPEG >= 300 dpi).

Gestisce il salvataggio uniforme e la visualizzazione delle traiettorie simulate
e delle finestre temporali, garantendo gli standard qualitativi per la relazione
tecnica e le slide dell'esposizione orale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DT, get_paths
from .data import feature_names

# Standard di risoluzione minima per pubblicazioni scientifiche
DPI = 300


def save_figure(fig, name: str, subdir: str | None = None, dpi: int = DPI) -> Path:
    """Esporta la figura in formato JPEG ad alta risoluzione con margini ottimizzati.

    Parametri:
        fig: Istanza matplotlib.figure.Figure da esportare.
        name: Nome del file (l'estensione viene forzata a .jpeg).
        subdir: Sottocartella opzionale all'interno di figures/ (es. '02_windowing', '03_rnn').
        dpi: Risoluzione in punti per pollice (default: 300 dpi).

    Restituisce:
        Path del file JPEG salvato.
    """
    out_dir = get_paths()["figures"]
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{Path(name).stem}.jpeg"
    fig.savefig(
        path,
        dpi=max(dpi, DPI),
        format="jpeg",
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    return path


def plot_trajectory(
    traj: np.ndarray,
    features: list[int],
    dt: float = DT,
    ax=None,
    title: str | None = None,
):
    """Visualizza l'evoluzione temporale continua di un sottoinsieme di variabili su una traiettoria.

    Parametri:
        traj: Array con shape (n_points, 55) contenente l'intera traiettoria (es. 1001 punti).
        features: Lista di indici di feature da tracciare (0..9 per magnetizzazioni, 10..54 per correlazioni).
        dt: Passo temporale di campionamento fisico (default: 0.02).
        ax: Axes matplotlib opzionale su cui disegnare.
        title: Titolo del subplot.
    """
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    names = feature_names()
    t = np.arange(traj.shape[0]) * dt
    for f in features:
        ax.plot(t, traj[:, f], label=names[f], lw=1)
    ax.set_xlabel("Tempo (t)")
    ax.legend(ncol=min(len(features), 5), fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_window(
    x: np.ndarray,
    y: np.ndarray,
    feature: int,
    y_pred: np.ndarray | None = None,
    dt: float = DT,
    ax=None,
    title: str | None = None,
):
    """Visualizza una finestra di previsione: contesto osservato (input), target futuro e predizione.

    Traccia una linea verticale tratteggiata per marcare il confine temporale esatto
    tra il passato noto al modello e l'orizzonte di predizione futura.

    Parametri:
        x: Finestra di input osservata con shape (t_in, 55).
        y: Finestra target con shape (t_out, 55).
        feature: Indice della feature da rappresentare (0..54).
        y_pred: Predizione del modello opzionale con shape (t_out, 55).
        dt: Passo di campionamento temporale (default: 0.02).
        ax: Axes matplotlib su cui disegnare.
        title: Titolo del grafico.
    """
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    t_in, t_out = x.shape[0], y.shape[0]
    t_x = np.arange(t_in) * dt
    t_y = (t_in + np.arange(t_out)) * dt

    # Contesto passato (input noto)
    ax.plot(t_x, x[:, feature], "o-", ms=3, label="Input (passato)")

    # Valore reale futuro (ground truth)
    ax.plot(t_y, y[:, feature], "-", label="Target reale")

    # Predizione opzionale della rete neurale
    if y_pred is not None:
        ax.plot(t_y, y_pred[:, feature], "--", label="Predizione")

    # Separatore visivo passato/futuro
    ax.axvline(t_in * dt - dt / 2, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Tempo relativo (t)")
    ax.set_ylabel(feature_names()[feature])
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax
