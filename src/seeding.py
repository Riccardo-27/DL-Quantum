"""Gestione centralizzata della riproducibilità numerica (seed e determinismo)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Fissa i seed di random, numpy e TensorFlow per garantire risultati identici.

    Con deterministic=True attiva il determinismo delle operazioni GPU di TensorFlow.
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
