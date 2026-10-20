"""Train loop ibrido e parametrico: teacher forcing -> masked modeling -> scheduled sampling.

Un unico ciclo di addestramento attraversa tre fasi consecutive, ciascuna con un numero
di epoche configurabile (``epochs_tf``, ``epochs_mm``, ``epochs_ss`` in ``src.config``):

1. **Teacher forcing (TF)**: il decoder riceve i target veri shiftati di un passo.
2. **Masked modeling (MM)**: una frazione ``mask_ratio`` di istanti temporali dell'input
   (encoder e decoder) viene azzerata; il modello deve ricostruire comunque l'intero
   target. Rende la rete robusta a osservazioni mancanti o rumorose.
3. **Scheduled sampling (SS)**: con probabilità ``p`` (crescente linearmente da
   ``ss_p_start`` a ``ss_p_end`` lungo la fase) l'input del decoder al passo t è la
   predizione del modello al passo t-1 invece del valore vero. Riduce il divario
   train/test dovuto all'accumulo dell'errore in free-running (exposure bias).
   Implementazione parallela (Duckworth et al., 2019): un primo passaggio senza
   gradiente produce le predizioni, un secondo passaggio con input misti calcola la loss.
   È valida identicamente per RNN seq2seq e Transformer.

Interfaccia richiesta al modello:
- seq2seq (RNN o Transformer): ``model((x, dec_in), training)`` -> ``y`` e metodo
  ``predict_free_running(x, batch_size)`` per la validazione autoregressiva;
- diretto (RNN one-shot): ``model(x, training)`` -> ``y``. Per questi modelli TF e SS
  coincidono con l'addestramento supervisionato standard; MM maschera solo ``x``.
"""

from __future__ import annotations

import time

import keras
import numpy as np
import tensorflow as tf

from .windowing import decoder_inputs

PHASES = ("tf", "mm", "ss")


def is_seq2seq(model) -> bool:
    """True se il modello ha un decoder autoregressivo (input ``(x, dec_in)``)."""
    return hasattr(model, "predict_free_running")


def predict(model, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Predizione realistica: free-running per i seq2seq, one-shot per i modelli diretti."""
    if is_seq2seq(model):
        return model.predict_free_running(x, batch_size=batch_size)
    return model.predict(x, batch_size=batch_size, verbose=0)


def ss_probability(k: int, n_epochs: int, p_start: float, p_end: float) -> float:
    """Probabilità di scheduled sampling all'epoca ``k`` della fase SS (rampa lineare)."""
    if n_epochs <= 1:
        return float(p_end)
    return float(p_start + (p_end - p_start) * k / (n_epochs - 1))


def mask_timesteps(arr: np.ndarray, ratio: float, rng: np.random.Generator,
                   protect: int | None = None) -> np.ndarray:
    """Azzera una frazione ``ratio`` di istanti temporali (tutte le 55 feature insieme).

    ``protect`` indica un indice temporale da non mascherare mai (es. -1 per l'ultimo
    istante osservato dell'encoder, 0 per il token iniziale del decoder).
    """
    if ratio <= 0:
        return arr
    mask = rng.random(arr.shape[:2]) < ratio
    if protect is not None:
        mask[:, protect] = False
    out = arr.copy()
    out[mask] = 0.0
    return out


def mix_decoder_inputs(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, p: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Input del decoder per scheduled sampling: ogni posizione t >= 1 usa ``y_hat[t-1]`` con prob. ``p``."""
    dec_true = decoder_inputs(x, y)
    dec_pred = decoder_inputs(x, y_hat)
    use_pred = rng.random(y.shape[:2]) < p
    use_pred[:, 0] = False          # il primo token è sempre l'ultimo istante osservato
    return np.where(use_pred[..., None], dec_pred, dec_true).astype(np.float32)


def train_hybrid(
    model: keras.Model,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    cfg: dict,
    seed: int | None = None,
    restore_best: bool = True,
    verbose: int = 1,
) -> list[dict]:
    """Addestra ``model`` attraverso le fasi TF -> MM -> SS e restituisce la history per epoca.

    La metrica di validazione è sempre quella realistica (free-running per i seq2seq);
    con ``restore_best=True`` a fine training vengono ripristinati i pesi migliori.
    """
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    s2s = is_seq2seq(model)
    bs = int(cfg["batch_size"])
    optimizer = keras.optimizers.get({"class_name": cfg["optimizer"],
                                      "config": {"learning_rate": cfg["lr"]}})
    loss_fn = keras.losses.MeanSquaredError()

    @tf.function(reduce_retracing=True)
    def train_step(inputs, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(inputs, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def forward(inputs):
        return model(inputs, training=False)

    schedule = [("tf", cfg["epochs_tf"]), ("mm", cfg["epochs_mm"]), ("ss", cfg["epochs_ss"])]
    history: list[dict] = []
    best_val, best_weights = np.inf, None
    n, epoch = len(X_tr), 0

    for phase, n_epochs in schedule:
        for k in range(int(n_epochs)):
            epoch += 1
            p = ss_probability(k, n_epochs, cfg["ss_p_start"], cfg["ss_p_end"]) if phase == "ss" else 0.0
            t0 = time.time()
            perm = rng.permutation(n)
            loss_sum = 0.0

            for i in range(0, n, bs):
                b = perm[i:i + bs]
                xb, yb = X_tr[b], Y_tr[b]
                if phase == "mm":
                    xb = mask_timesteps(xb, cfg["mask_ratio"], rng, protect=-1)
                if s2s:
                    dec = decoder_inputs(xb, yb)
                    if phase == "mm":
                        dec = mask_timesteps(dec, cfg["mask_ratio"], rng, protect=0)
                    elif phase == "ss" and p > 0:
                        y_hat = forward((tf.convert_to_tensor(xb), tf.convert_to_tensor(dec))).numpy()
                        dec = mix_decoder_inputs(xb, yb, y_hat, p, rng)
                    inputs = (tf.convert_to_tensor(xb), tf.convert_to_tensor(dec))
                else:
                    inputs = tf.convert_to_tensor(xb)
                loss = train_step(inputs, tf.convert_to_tensor(yb))
                loss_sum += float(loss) * len(b)

            val_mse = float(np.mean((predict(model, X_va, batch_size=max(bs, 512)) - Y_va) ** 2))
            rec = {"epoch": epoch, "phase": phase, "ss_p": p, "loss": loss_sum / n,
                   "val_mse": val_mse, "time_s": time.time() - t0}
            history.append(rec)
            if val_mse < best_val:
                best_val, best_weights = val_mse, model.get_weights()
            if verbose:
                print(f"[{phase.upper()}] epoca {epoch:3d} | loss={rec['loss']:.6f} | "
                      f"val_mse={val_mse:.6f} | p_ss={p:.2f} | {rec['time_s']:.1f}s")

    if restore_best and best_weights is not None:
        model.set_weights(best_weights)
    return history
