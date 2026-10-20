"""Gestione centralizzata della riproducibilità numerica (seed e determinismo).

Fissa in modo congiunto i generatori pseudo-casuali dell'interprete Python,
di NumPy e di TensorFlow, garantendo la replicabilità numerica bit-a-bit
degli split, dell'estrazione delle finestre e dell'inizializzazione dei pesi.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Fissa i seed di sistema e attiva il determinismo opzionale su CPU/GPU.

    Parametri:
        seed: Valore intero del seed da applicare a tutti i generatori.
        deterministic: Se True, forza TensorFlow e cuDNN a utilizzare algoritmi
            deterministici su GPU (eliminando la variabilità introdotta dalle
            operazioni atomiche concorrenti di riduzione in virgola mobile).
    """
    # 1. Determinismo hashing interno delle strutture dati Python (dizionari/set)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Generatori standard di Python e NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 3. Backend di Deep Learning (TensorFlow/Keras) se disponibile nell'ambiente
    try:
        import tensorflow as tf
    except ImportError:
        # In ambienti di solo pre-processing, TF può essere assente
        return

    # 4. Configurazione flag deterministici per kernel cuDNN e operazioni TF
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Fissa contemporaneamente random, numpy e tf.random
    tf.keras.utils.set_random_seed(seed)

    # 5. Attiva il determinismo rigoroso a livello di grafo computazionale TF
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
