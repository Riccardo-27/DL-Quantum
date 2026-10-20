"""Architetture ricorrenti (LSTM / GRU) per la previsione multi-orizzonte della dinamica quantistica.

Fornisce due famiglie architetturali, entrambe vincolate all'intervallo fisico [-1, 1]
mediante funzione di attivazione finale tanh:

1. **M1 – RNN Diretta (One-Shot)** (build_rnn_direct):
   L'encoder ricorrente comprime la sequenza di input (t_in, 55) in uno stato latente finale;
   una testa densa proietta in un singolo passaggio feed-forward l'intero tensore futuro
   di forma (t_out, 55). Non prevede retroazione: è esente per costruzione da accumulo di errore
   ricorsivo (exposure bias) e offre massima velocità di inferenza.

2. **M2 – RNN Seq2Seq Autoregressiva** (RNNSeq2Seq):
   Architettura encoder-decoder con trasferimento degli stati nascosti (h, c).
   - In fase di training (call): opera in Teacher Forcing parallelo ricevendo il target
     reale shiftato di un passo (vedi src.windowing.decoder_inputs).
   - In fase di test/validazione (free_running / predict_free_running): opera in circuito chiuso
     autoregressivo, reimmettendo ogni vettore predetto y_hat[t] come input al passo successivo t+1.
   Compatibile al 100% con la serializzazione nativa Keras 3 (@register_keras_serializable).
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from .config import N_FEATURES

# Dizionario dei costruttori di layer ricorrenti compatibili con Keras 3
_CELLS = {"lstm": keras.layers.LSTM, "gru": keras.layers.GRU}


def _rnn_layer(
    cell: str,
    units: int,
    return_sequences: bool,
    return_state: bool = False,
    name: str | None = None,
) -> keras.layers.Layer:
    """Istanzia un layer ricorrente (LSTM o GRU) configurato per accelerazione hardware cuDNN su GPU.

    Non impiega recurrent_dropout, permettendo a TensorFlow di utilizzare i kernel
    ottimizzati NVIDIA cuDNN per la massima velocità di addestramento su GPU (T4).
    """
    key = cell.lower()
    if key not in _CELLS:
        raise ValueError(f"Cella ricorrente non supportata: {cell!r} (attese: {sorted(_CELLS)})")
    return _CELLS[key](units, return_sequences=return_sequences, return_state=return_state, name=name)


# ---------------------------------------------------------------------------
# M1 - RNN diretta (one-shot)
# ---------------------------------------------------------------------------


def build_rnn_direct(
    t_out: int,
    t_in: int | None = None,
    n_features: int = N_FEATURES,
    units: int = 64,
    n_layers: int = 1,
    cell: str = "lstm",
    dropout: float = 0.0,
    output_activation: str = "tanh",
    name: str = "rnn_direct",
) -> keras.Model:
    """Costruisce un modello ricorrente diretto (M1) con testa densa one-shot.

    Architettura:
        Input(shape=(t_in, n_features))
          -> [Cella ricorrente(units)] * n_layers
          -> Dense(t_out * n_features, activation='tanh')
          -> Reshape((t_out, n_features))

    Parametri:
        t_out: Numero di passi temporali futuri da prevedere contemporaneamente (es. 40).
        t_in: Numero di passi temporali passati in input (None per supportare lunghezze variabili).
        n_features: Numero di osservabili quantistiche (default: 55).
        units: Dimensione dello stato nascosto della cella ricorrente (default: 64).
        n_layers: Numero di layer ricorrenti impilati (default: 1).
        cell: Tipo di cella: 'lstm' oppure 'gru'.
        dropout: Tasso di dropout inter-layer (applicato se n_layers > 1).
        output_activation: Funzione di attivazione finale ('tanh' per vincolare le predizioni a [-1, 1]).
        name: Nome identificativo del modello Keras.
    """
    inp = keras.Input(shape=(t_in, n_features), name="x")
    h = inp
    for i in range(n_layers):
        last = i == n_layers - 1
        h = _rnn_layer(cell, units, return_sequences=not last, name=f"{cell}_{i + 1}")(h)
        if dropout > 0 and not last:
            h = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(h)
    out = keras.layers.Dense(t_out * n_features, activation=output_activation, name="head")(h)
    out = keras.layers.Reshape((t_out, n_features), name="y")(out)
    return keras.Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# M2 - RNN seq2seq (encoder-decoder autoregressivo)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dlq")
class RNNSeq2Seq(keras.Model):
    """Architettura Seq2Seq Encoder-Decoder autoregressiva con supporto sia a LSTM che GRU.

    Gestisce:
    - Passaggio completo dello stato interno:
      * LSTM: coppia di tensori (h, c) per ciascun layer.
      * GRU: singolo tensore di stato h per ciascun layer.
    - Addestramento supervisionato in Teacher Forcing parallelo tramite il metodo call((x, dec_in)).
    - Inferenza realistica ad anello chiuso tramite il metodo free_running(x).
    - Serializzazione completa in formato standard .keras per persistenza su disco e riuso.
    """

    def __init__(
        self,
        t_out: int = 40,
        n_features: int = N_FEATURES,
        units: int = 64,
        n_layers: int = 1,
        cell: str = "lstm",
        dropout: float = 0.0,
        output_activation: str = "tanh",
        **kwargs,
    ):
        kwargs.setdefault("name", "rnn_seq2seq")
        super().__init__(**kwargs)
        self.t_out = int(t_out)
        self.n_features = int(n_features)
        self.units = int(units)
        self.n_layers = int(n_layers)
        self.cell = cell.lower()
        self.dropout_rate = float(dropout)
        self.output_activation = output_activation

        # Stack di layer ricorrenti per l'Encoder
        self.encoder_layers = [
            _rnn_layer(self.cell, units, return_sequences=True, return_state=True, name=f"enc_{i + 1}")
            for i in range(n_layers)
        ]
        # Stack speculare di layer ricorrenti per il Decoder
        self.decoder_layers = [
            _rnn_layer(self.cell, units, return_sequences=True, return_state=True, name=f"dec_{i + 1}")
            for i in range(n_layers)
        ]
        self.drop = keras.layers.Dropout(self.dropout_rate, name="dropout")
        self.head = keras.layers.Dense(n_features, activation=output_activation, name="head")

    # --- blocchi computazionali elementari ----------------------------------

    @staticmethod
    def _run(layer, x, initial_state=None, training=None):
        """Esegue un layer ricorrente estraendo separatamente la sequenza di output e gli stati interni."""
        out = layer(x, initial_state=initial_state, training=training)
        return out[0], list(out[1:])

    def encode(self, x, training=None) -> list[list[tf.Tensor]]:
        """Elabora la sequenza di input e restituisce gli stati finali di ciascun layer dell'encoder."""
        states, h = [], x
        for i, layer in enumerate(self.encoder_layers):
            h, st = self._run(layer, h, training=training)
            states.append(st)
            if i < self.n_layers - 1:
                h = self.drop(h, training=training)
        return states

    def decode(self, dec_in, states, training=None):
        """Avanza il decoder lungo dec_in inizializzando gli stati nascosti da states."""
        new_states, h = [], dec_in
        for layer, st in zip(self.decoder_layers, states):
            h, ns = self._run(layer, h, initial_state=st, training=training)
            new_states.append(ns)
            h = self.drop(h, training=training)
        return self.head(h), new_states

    # --- modalità operative di esecuzione -----------------------------------

    def call(self, inputs, training=None):
        """Modalità Teacher Forcing (training supervisionato).

        Parametri:
            inputs: Tupla (x, dec_in) dove:
                - x: Finestra passata osservata con shape (batch_size, t_in, 55).
                - dec_in: Target shiftato con shape (batch_size, t_out, 55).
        """
        x, dec_in = inputs
        y, _ = self.decode(dec_in, self.encode(x, training=training), training=training)
        return y

    def free_running(self, x, t_out: int | None = None, training=None):
        """Modalità Autoregressiva / Free-Running (valutazione realistica a circuito chiuso).

        Genera la traiettoria futura un passo alla volta: ciascun vettore predetto y_hat[t]
        viene reimmesso come input del decoder al passo successivo t+1.
        """
        t_out = int(t_out or self.t_out)
        states = self.encode(x, training=training)
        token = x[:, -1:, :]  # Il primo token è l'ultimo punto temporale osservato in input
        outs = []
        for _ in range(t_out):
            token, states = self.decode(token, states, training=training)
            outs.append(token)
        return tf.concat(outs, axis=1)

    def predict_free_running(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Inferenza Free-Running ad alte prestazioni compilata con tf.function e processata a batch.

        Parametri:
            x: Array NumPy con shape (n_finestre, t_in, 55).
            batch_size: Dimensione del sotto-batch di inferenza su GPU (default: 512).

        Restituisce:
            Array NumPy con shape (n_finestre, t_out, 55).
        """
        if not hasattr(self, "_fr_fn"):
            self._fr_fn = tf.function(lambda inp: self.free_running(inp, training=False))
        outs = [
            self._fr_fn(tf.convert_to_tensor(x[i : i + batch_size], dtype=tf.float32)).numpy()
            for i in range(0, len(x), batch_size)
        ]
        return np.concatenate(outs, axis=0)

    def get_config(self) -> dict:
        """Serializza la configurazione degli iperparametri per il salvataggio in formato Keras 3."""
        cfg = super().get_config()
        cfg.update({
            "t_out": self.t_out,
            "n_features": self.n_features,
            "units": self.units,
            "n_layers": self.n_layers,
            "cell": self.cell,
            "dropout": self.dropout_rate,
            "output_activation": self.output_activation,
        })
        return cfg


# ---------------------------------------------------------------------------
# Factory guidata dalla configurazione centralizzata
# ---------------------------------------------------------------------------


def build_rnn_from_config(cfg: dict, t_in: int | None = None) -> keras.Model:
    """Costruisce e istanzia il modello ricorrente specifico definito nel dizionario cfg.

    Supporta:
    - cfg['model'] == 'rnn_direct': Costruisce M1 (RNN diretta one-shot).
    - cfg['model'] == 'rnn': Costruisce M2 (Seq2Seq autoregressivo con build esplicito).
    """
    common = dict(
        t_out=cfg["t_out"],
        n_features=N_FEATURES,
        units=cfg["rnn_units"],
        n_layers=cfg["rnn_layers"],
        cell=cfg.get("rnn_cell", "lstm"),
        dropout=cfg["dropout"],
        output_activation=cfg["output_activation"],
    )
    if cfg["model"] == "rnn_direct":
        return build_rnn_direct(t_in=t_in, **common)
    if cfg["model"] == "rnn":
        model = RNNSeq2Seq(**common)
        # Esecuzione dummy di build per allocare i pesi prima di model.summary() o model.save()
        t = t_in or cfg["t_in"]
        model((tf.zeros((1, t, N_FEATURES)), tf.zeros((1, cfg["t_out"], N_FEATURES))))
        return model
    raise ValueError(f"Modello ricorrente sconosciuto: {cfg['model']!r} (attesi 'rnn_direct' | 'rnn')")
