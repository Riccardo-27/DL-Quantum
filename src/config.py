"""Configurazione centralizzata del progetto DL-Quantum.

Gestisce in un unico dizionario piatto (DEFAULT_CONFIG) tutti i parametri fisici,
le impostazioni di riproducibilità numerica, le dimensioni delle finestre temporali,
le architetture dei modelli e gli iperparametri di addestramento.
Risolve inoltre in modo dinamico i percorsi tra ambiente locale e Google Drive.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Parametri fisici del sistema PXP (catena unidimensionale a 10 qubit)
# ---------------------------------------------------------------------------
N_QUBITS = 10                                 # Numero di atomi di Rydberg neutri
N_MAG = N_QUBITS                              # 10 magnetizzazioni locali: <sigma_z^i> (i=1..10)
N_CORR = N_QUBITS * (N_QUBITS - 1) // 2       # 45 correlazioni a due corpi: <sigma_z^i sigma_z^j> (i < j)
N_FEATURES = N_MAG + N_CORR                   # 55 variabili fisiche osservabili complessive
N_TRAJ = 400                                  # Traiettorie quantistiche simulate indipendenti
N_POINTS = 1001                               # Istanti temporali campionati per ciascuna traiettoria
DT = 0.02                                     # Passo temporale di campionamento (delta_t = 0.02)
T_FIN = 20.0                                  # Orizzonte temporale finale della simulazione (t in [0, 20])

# Slice NumPy standard per estrarre le sotto-matrici di osservabili
MAG_SLICE = slice(0, N_MAG)                   # Indici delle magnetizzazioni: 0..9
CORR_SLICE = slice(N_MAG, N_FEATURES)         # Indici delle correlazioni connesse: 10..54

# ---------------------------------------------------------------------------
# Iperparametri di default per esperimenti, architetture e addestramento
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    # --- Riproducibilità e Repliche Statistiche ---
    "seed": 42,                               # Seed del singolo run
    "split_seed": 42,                         # Seed per il partizionamento delle traiettorie (congelato)
    "replica_seeds": [42, 43, 44],            # Terzetto di seed per la validazione t di Student

    # --- Partizionamento su Base Traiettoria ---
    "n_train": 280,                           # 70% traiettorie di training (280)
    "n_val": 60,                              # 15% traiettorie di validazione (60)
    "n_test": 60,                             # 15% traiettorie di test finale (60)
    "cv_folds": 5,                            # Numero di fold per eventuale cross-validation

    # --- Finestrature Temporali ---
    "t_in": 10,                               # Contesto passato osservato standard (0.20 s)
    "t_out": 40,                              # Orizzonte futuro previsto (0.80 s)
    "stride": 10,                             # Passo di avanzamento della finestra scorrevole
    "windows": {"short": 10, "long": 20},     # Finestre standard di benchmark
    "t_in_grid": [5, 10, 20, 40],             # Griglia della memoria passata per la Sfida 1
    "align_t_in": 40,                         # Allineamento dei target per confronto equo nella Sfida 1
    "sr_factor": 2,                           # Fattore di subsampling temporale (x2) per la Sfida 2
    "sr_t_in": 20,                            # Punti radi osservati in input per la super-risoluzione

    # --- Preprocessing e Vincoli Fisici ---
    "normalization": "none",                  # Dati grezzi non riscalati per preservare le gerarchie fisiche
    "output_activation": "tanh",              # Attivazione tanh per vincolare l'output al range [-1, 1]

    # --- Architetture dei Modelli ---
    "model": "rnn",                           # Famiglia: 'rnn_direct' (M1), 'rnn' (M2), 'transformer'
    "rnn_units": 64,                          # Neuroni nello stato nascosto delle celle ricorrenti
    "rnn_layers": 1,                          # Numero di layer ricorrenti stacked
    "rnn_cell": "lstm",                       # Cella ricorrente: 'lstm' o 'gru'
    "transformer_d_model": 64,                # Dimensione dello spazio di embedding dell'attenzione
    "transformer_heads": 4,                   # Numero di teste di Multi-Head Attention
    "transformer_d_ff": 128,                  # Dimensione dello strato Feed-Forward intermedio
    "transformer_layers": 2,                  # Numero di blocchi Transformer impilati
    "dropout": 0.0,                           # Tasso di regolarizzazione dropout

    # --- Train Loop Ibrido (TF -> MM -> SS) ---
    "epochs_tf": 10,                          # Epoche in Teacher Forcing (fase 1)
    "epochs_mm": 10,                          # Epoche in Masked Modeling (fase 2)
    "epochs_ss": 5,                           # Epoche in Scheduled Sampling (fase 3)
    "mask_ratio": 0.15,                       # Frazione di timestep azzerati in fase MM (15%)
    "ss_p_start": 0.0,                        # Probabilità iniziale di campionamento predizioni in SS
    "ss_p_end": 0.5,                          # Probabilità finale di campionamento predizioni in SS
    "batch_size": 64,                         # Dimensione del mini-batch per gradiente stocastico
    "lr": 1e-3,                               # Learning rate iniziale per l'ottimizzatore
    "optimizer": "adam",                      # Algoritmo di ottimizzazione (Adam)

    # --- Bookkeeping e Metadati ---
    "run_name": None,                         # Nome identificativo univoco del run
    "hp_name": None,                          # Configurazione iperparametri: 'HP1', 'HP2', 'HP3'
    "window_name": None,                      # Nome della finestra: 'short', 'long', ecc.
}


def make_config(**overrides) -> dict:
    """Restituisce una copia deep della configurazione predefinita con i parametri specificati sovrascritti.

    Solleva un KeyError esplicito se viene passata una chiave non prevista in DEFAULT_CONFIG,
    prevenendo refusi silenziosi (ad es. 'learning_rate' invece di 'lr').
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"Chiavi di config sconosciute: {sorted(unknown)}")
    cfg.update(copy.deepcopy(overrides))
    return cfg


# ---------------------------------------------------------------------------
# Risoluzione dinamica dei percorsi (Locale vs Google Drive Colab)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/DL-Quantum")


def get_root() -> Path:
    """Individua in modo robusto la radice del repository sia su Google Drive che in locale."""
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
    """Localizza il file pesante trajectories.csv testando i pattern di percorso standard su Drive e locale."""
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

    # Ricerca euristica su Google Drive nel caso la cartella condivisa abbia un nome personalizzato
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        found = list(drive_root.glob("**/trajectories.csv"))
        if found:
            return found[0]

    return root / "data" / "trajectories.csv"


def get_paths(create: bool = True) -> dict[str, Path]:
    """Costruisce il dizionario dei percorsi di lavoro (root, csv, cache, results, figures, checkpoints).

    Parametri:
        create: Se True, crea automaticamente le directory di output su disco se non ancora presenti.
    """
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
    """Restituisce il nome dell'autore attivo (es. 'ernesto' o 'riccardo') per il logging concorrente."""
    return os.environ.get("DLQ_AUTHOR", "unknown")
