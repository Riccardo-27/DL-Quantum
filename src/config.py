"""Configurazione centrale del progetto DL-Quantum.

Tutti gli iperparametri vivono in un unico dizionario piatto (facile da loggare
riga per riga in ``results_log``). Ogni esperimento parte da ``make_config`` e
sovrascrive solo le chiavi che cambia.

Percorsi
--------
Il layout delle cartelle di lavoro e' identico in locale e su Colab::

    <ROOT>/data/trajectories.csv
    <ROOT>/data/cache/            # .npy derivati dal CSV
    <ROOT>/results/               # results_log_<autore>.csv, split_indices.npz
    <ROOT>/figures/               # .jpeg a 300 dpi
    <ROOT>/checkpoints/           # modelli .keras

``ROOT`` viene risolta in quest'ordine:
1. variabile d'ambiente ``DLQ_ROOT``;
2. ``/content/drive/MyDrive/DL-Quantum`` se esiste (Colab con Drive montato);
3. la root del repository (uso locale).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Costanti del dataset (da data/README del professore)
# ---------------------------------------------------------------------------
N_QUBITS = 10
N_MAG = N_QUBITS                              # magnetizzazioni m_1..m_N
N_CORR = N_QUBITS * (N_QUBITS - 1) // 2       # correlazioni c_ij, i<j  -> 45
N_FEATURES = N_MAG + N_CORR                   # 55 (colonna tempo esclusa)
N_TRAJ = 400
N_POINTS = 1001
DT = 0.02
T_FIN = 20.0

MAG_SLICE = slice(0, N_MAG)
CORR_SLICE = slice(N_MAG, N_FEATURES)

# ---------------------------------------------------------------------------
# Iperparametri di default (GUIDA_UNICA §3.6 e §4.0)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    # --- riproducibilita' ---
    "seed": 42,                       # seed del singolo run
    "split_seed": 42,                 # seed dello split (NON cambiarlo mai)
    "replica_seeds": [42, 43, 44],

    # --- split per traiettoria ---
    "n_train": 280,
    "n_val": 60,
    "n_test": 60,
    "cv_folds": 5,

    # --- finestre ---
    "t_in": 10,
    "t_out": 40,
    "stride": 10,
    "windows": {"short": 10, "long": 20},
    "t_in_grid": [5, 10, 20, 40],     # sfida 1
    "sr_factor": 2,                   # sfida 2: input sottocampionato x2
    "sr_t_in": 20,                    # punti osservati in input (super-res)

    # --- preprocessing ---
    "normalization": "none",          # "none" | "zscore"  (da decidere in S2)
    "output_activation": "tanh",

    # --- modello ---
    "model": "rnn",                   # "rnn_direct" | "rnn" | "transformer"
    "rnn_units": 64,
    "rnn_layers": 1,
    "transformer_d_model": 64,
    "transformer_heads": 4,
    "transformer_d_ff": 128,
    "transformer_layers": 2,
    "dropout": 0.0,

    # --- train loop ibrido (TF -> MM -> SS) ---
    "epochs_tf": 10,
    "epochs_mm": 10,
    "epochs_ss": 5,
    "mask_ratio": 0.15,
    "ss_p_start": 0.0,
    "ss_p_end": 0.5,
    "batch_size": 64,
    "lr": 1e-3,
    "optimizer": "adam",

    # --- bookkeeping ---
    "run_name": None,
    "hp_name": None,                  # "HP1" | "HP2" | "HP3"
    "window_name": None,              # "short" | "long" | ...
}


def make_config(**overrides) -> dict:
    """Restituisce una copia della config di default con ``overrides`` applicati.

    Solleva ``KeyError`` per chiavi sconosciute, cosi' un refuso
    (es. ``learning_rate`` invece di ``lr``) non passa inosservato.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"Chiavi di config sconosciute: {sorted(unknown)}")
    cfg.update(copy.deepcopy(overrides))
    return cfg


# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/DL-Quantum")


def get_root() -> Path:
    env = os.environ.get("DLQ_ROOT")
    if env:
        return Path(env)
    if (COLAB_DRIVE_ROOT / "src" / "seeding.py").exists():
        return COLAB_DRIVE_ROOT
    colab_local = Path("/content/DL-Quantum")
    if (colab_local / "src" / "seeding.py").exists():
        return colab_local
    if COLAB_DRIVE_ROOT.exists():
        return COLAB_DRIVE_ROOT
    return REPO_ROOT


def find_data_csv(root: Path) -> Path:
    """Individua trajectories.csv in modo robusto (utile con cartelle condivise su Drive)."""
    env = os.environ.get("DLQ_DATA_CSV")
    if env:
        p = Path(env)
        if p.exists():
            return p

    candidates = [
        root / "data" / "trajectories.csv",
        root / "trajectories.csv",
        Path("/content/drive/MyDrive/DL-Quantum/data/trajectories.csv"),
        Path("/content/drive/MyDrive/DL-Quantum/trajectories.csv"),
        Path("/content/drive/MyDrive/Quantum Dynamics/data/trajectories.csv"),
        Path("/content/drive/MyDrive/Quantum Dynamics/trajectories.csv"),
        Path("/content/drive/MyDrive/trajectories.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Se siamo su Colab, ricerca rapida in MyDrive o cartelle condivise
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        found = list(drive_root.glob("**/trajectories.csv"))
        if found:
            return found[0]

    return root / "data" / "trajectories.csv"


def get_paths(create: bool = True) -> dict[str, Path]:
    """Dizionario dei percorsi di lavoro. Con ``create=True`` crea le cartelle."""
    root = get_root()
    paths = {
        "root": root,
        "data_csv": find_data_csv(root),
        "cache": root / "data" / "cache",
        "results": root / "results",
        "figures": root / "figures",
        "checkpoints": root / "checkpoints",
    }
    if create:
        for key in ("cache", "results", "figures", "checkpoints"):
            paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def get_author() -> str:
    """Nome breve di chi lancia il run (usato per separare i log su Drive)."""
    return os.environ.get("DLQ_AUTHOR", "unknown")
