"""Riproducibilita': un solo punto in cui si fissano tutti i seed.

Nota: ``PYTHONHASHSEED`` ha effetto solo se impostato *prima* dell'avvio
dell'interprete. Lo impostiamo comunque (vale per eventuali sottoprocessi) e
in Colab l'hashing di stringhe non influenza i nostri risultati numerici.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Fissa i seed di ``random``, ``numpy`` e (se installato) TensorFlow.

    Con ``deterministic=True`` abilita anche il determinismo delle op di TF,
    necessario per avere risultati identici su GPU (a costo di un po' di
    velocita').
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:  # in locale TF puo' mancare: va bene per i test numpy
        return

    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # set_random_seed fissa insieme random, numpy e tf.random
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:  # TF < 2.9
            pass
