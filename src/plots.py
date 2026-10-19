"""Grafici: salvataggio sempre in ``.jpeg`` a 300 dpi (regola del professore)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DT, get_paths
from .data import feature_names

DPI = 300


def save_figure(fig, name: str, subdir: str | None = None, dpi: int = DPI) -> Path:
    """Salva ``fig`` come ``<figures>/<subdir>/<name>.jpeg`` a ``dpi`` dpi."""
    out_dir = get_paths()["figures"]
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(name).stem}.jpeg"
    fig.savefig(path, dpi=max(dpi, DPI), format="jpeg", bbox_inches="tight",
                pil_kwargs={"quality": 95})
    return path


def plot_trajectory(traj: np.ndarray, features: list[int], dt: float = DT, ax=None,
                    title: str | None = None):
    """Andamento nel tempo di alcune variabili di una traiettoria ``(T, 55)``."""
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    names = feature_names()
    t = np.arange(traj.shape[0]) * dt
    for f in features:
        ax.plot(t, traj[:, f], label=names[f], lw=1)
    ax.set_xlabel("t")
    ax.legend(ncol=min(len(features), 5), fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_window(x: np.ndarray, y: np.ndarray, feature: int, y_pred: np.ndarray | None = None,
                dt: float = DT, ax=None, title: str | None = None):
    """Una finestra: input osservato, target e (opzionale) predizione."""
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    t_in, t_out = x.shape[0], y.shape[0]
    t_x = np.arange(t_in) * dt
    t_y = (t_in + np.arange(t_out)) * dt
    ax.plot(t_x, x[:, feature], "o-", ms=3, label="input")
    ax.plot(t_y, y[:, feature], "-", label="target")
    if y_pred is not None:
        ax.plot(t_y, y_pred[:, feature], "--", label="predizione")
    ax.axvline(t_in * dt - dt / 2, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("t (relativo alla finestra)")
    ax.set_ylabel(feature_names()[feature])
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax
