"""Train loop ibrido e parametrico: Teacher Forcing -> Masked Modeling -> Scheduled Sampling.

Implementa un unico ciclo di addestramento strutturato su tre fasi consecutive, ciascuna
con numero di epoche configurabile dinamicamente (epochs_tf, epochs_mm, epochs_ss):

1. **Teacher Forcing (TF)**:
   Il decoder riceve i target reali shiftati di un passo. Garantisce convergenza stabile
   e rapida discesa del gradiente nelle prime epoche.

2. **Masked Modeling (MM)**:
   Una frazione casuale (mask_ratio = 15%) di istanti temporali dell'input (sia encoder che decoder)
   viene azzerata con maschera uniforme. La rete impara a ricostruire la dinamica quantistica
   anche a fronte di osservazioni parzialmente mancanti o rumorose.

3. **Scheduled Sampling (SS)**:
   Con probabilità p (che cresce linearmente da ss_p_start=0.0 a ss_p_end=0.5 durante la fase),
   l'input del decoder al passo t riceve la predizione generata dalla rete al passo t-1 anziché il valore vero.
   Colma progressivamente la discrepanza tra addestramento e inferenza autoregressiva (exposure bias).
   Adotta l'algoritmo parallelo a due passaggi (Duckworth et al., 2019):
     - Passo 1: Forward senza gradiente per generare le predizioni y_hat.
     - Passo 2: Selezione probabilistica dei token e forward-backward con gradiente per calcolare la loss.
"""

from __future__ import annotations

import time

import keras
import numpy as np
import tensorflow as tf

from .windowing import decoder_inputs

# Nomi canonici delle tre fasi sequenziali
PHASES = ("tf", "mm", "ss")


def is_seq2seq(model) -> bool:
    """Verifica se il modello possiede un decoder autoregressivo (interfaccia seq2seq con predict_free_running)."""
    return hasattr(model, "predict_free_running")


def predict(model, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Esegue la predizione realistica: free-running a circuito chiuso per Seq2Seq, one-shot per modelli diretti."""
    if is_seq2seq(model):
        return model.predict_free_running(x, batch_size=batch_size)
    return model.predict(x, batch_size=batch_size, verbose=0)


def ss_probability(k: int, n_epochs: int, p_start: float, p_end: float) -> float:
    """Calcola la probabilità di scheduled sampling p all'epoca k mediante rampa lineare da p_start a p_end."""
    if n_epochs <= 1:
        return float(p_end)
    return float(p_start + (p_end - p_start) * k / (n_epochs - 1))


def mask_timesteps(
    arr: np.ndarray, ratio: float, rng: np.random.Generator, protect: int | None = None
) -> np.ndarray:
    """Azzera una frazione casuale (ratio) di istanti temporali lungo la dimensione temporale dell'array.

    Parametri:
        arr: Array con shape (batch_size, timesteps, 55).
        ratio: Percentuale di timestep da azzerare (es. 0.15).
        rng: Generatore di numeri casuali NumPy.
        protect: Indice temporale opzionale escluso dal mascheramento (es. -1 per l'ultimo istante noto).
    """
    if ratio <= 0:
        return arr
    mask = rng.random(arr.shape[:2]) < ratio
    if protect is not None:
        mask[:, protect] = False
    out = arr.copy()
    out[mask] = 0.0
    return out


def mix_decoder_inputs(
    x: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Costruisce gli input del decoder per lo Scheduled Sampling miscelando target veri e predizioni.

    Per ogni passo temporale t >= 1, seleziona con probabilità p la predizione y_hat[t-1]
    e con probabilità 1-p il valore vero y[t-1]. Il token iniziale t=0 è sempre rigidamente
    preservato come l'ultimo istante osservato dell'encoder (x[:, -1:]).
    """
    dec_true = decoder_inputs(x, y)
    dec_pred = decoder_inputs(x, y_hat)
    use_pred = rng.random(y.shape[:2]) < p
    use_pred[:, 0] = False  # Il primo token è sempre vincolato al passato osservato
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
    """Esegue l'addestramento ibrido completo (TF -> MM -> SS) con monitoraggio dell'errore di validazione.

    Caratteristiche:
    - Funzioni di training e forward compilate come grafi statici TensorFlow (@tf.function).
    - Metrica di validazione realistica calcolata all'inizio di ogni epoca (in free-running per i Seq2Seq).
    - Salvataggio e ripristino opzionale (restore_best=True) dello stato dei pesi che ha minimizzato il val_mse.

    Parametri:
        model: Modello Keras (RNN diretta, Seq2Seq o Transformer).
        X_tr, Y_tr: Array di training per input e target.
        X_va, Y_va: Array di validazione per input e target.
        cfg: Dizionario centralizzato degli iperparametri di configurazione.
        seed: Seed casuale locale per il generatore del train loop (default: cfg['seed']).
        restore_best: Se True, ripristina a fine training i pesi con il minor val_mse registrato.
        verbose: Se > 0, stampa a terminale l'andamento epoca per epoca.

    Restituisce:
        Lista di dizionari contenente la history per epoca: loss, val_mse, fase, probabilità ss_p e tempo.
    """
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    s2s = is_seq2seq(model)
    bs = int(cfg["batch_size"])
    optimizer = keras.optimizers.get({
        "class_name": cfg["optimizer"],
        "config": {"learning_rate": cfg["lr"]},
    })
    loss_fn = keras.losses.MeanSquaredError()

    # Compilazione del passo di ottimizzazione a grafo statico
    @tf.function(reduce_retracing=True)
    def train_step(inputs, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(inputs, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Compilazione del forward pass non-differenziabile per lo Scheduled Sampling
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

            # Iterazione sui mini-batch
            for i in range(0, n, bs):
                b = perm[i : i + bs]
                xb, yb = X_tr[b], Y_tr[b]

                # Fase 2 (Masked Modeling): azzeramento casuale di timesteps
                if phase == "mm":
                    xb = mask_timesteps(xb, cfg["mask_ratio"], rng, protect=-1)

                if s2s:
                    dec = decoder_inputs(xb, yb)
                    if phase == "mm":
                        dec = mask_timesteps(dec, cfg["mask_ratio"], rng, protect=0)
                    elif phase == "ss" and p > 0:
                        # Fase 3 (Scheduled Sampling): estrazione predizioni e miscelazione probabilistica
                        y_hat = forward((tf.convert_to_tensor(xb), tf.convert_to_tensor(dec))).numpy()
                        dec = mix_decoder_inputs(xb, yb, y_hat, p, rng)
                    inputs = (tf.convert_to_tensor(xb), tf.convert_to_tensor(dec))
                else:
                    inputs = tf.convert_to_tensor(xb)

                loss = train_step(inputs, tf.convert_to_tensor(yb))
                loss_sum += float(loss) * len(b)

            # Valutazione della metrica di validazione reale (Free-Running per i seq2seq)
            val_mse = float(np.mean((predict(model, X_va, batch_size=max(bs, 512)) - Y_va) ** 2))
            rec = {
                "epoch": epoch,
                "phase": phase,
                "ss_p": p,
                "loss": loss_sum / n,
                "val_mse": val_mse,
                "time_s": time.time() - t0,
            }
            history.append(rec)

            if val_mse < best_val:
                best_val, best_weights = val_mse, model.get_weights()

            if verbose:
                print(
                    f"[{phase.upper()}] epoca {epoch:3d} | loss={rec['loss']:.6f} | "
                    f"val_mse={val_mse:.6f} | p_ss={p:.2f} | {rec['time_s']:.1f}s"
                )

    # Ripristino dei pesi associati al miglior checkpoint di validazione
    if restore_best and best_weights is not None:
        model.set_weights(best_weights)

    return history
