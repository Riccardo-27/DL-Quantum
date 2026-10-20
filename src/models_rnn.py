"""Architetture ricorrenti (LSTM / GRU) per la previsione multi-orizzonte.

Due famiglie di modelli, entrambe con uscita ``tanh`` (dominio fisico [-1, 1]):

- **RNN diretta** (``build_rnn_direct``): l'encoder ricorrente comprime la finestra
  di input in un vettore, una testa densa emette in un colpo solo tutti i
  ``t_out`` passi futuri. Nessuna retroazione: l'errore non si accumula nel tempo.
- **RNN seq2seq** (``RNNSeq2Seq``): encoder e decoder ricorrenti. Il decoder genera
  un passo alla volta partendo dallo stato finale dell'encoder.
  * ``call((x, dec_in))``: teacher forcing, il decoder riceve i target veri
    shiftati (vedi ``src.windowing.decoder_inputs``); calcolo parallelo.
  * ``free_running(x)``: inferenza autoregressiva, ogni predizione viene
    reimmessa come input del passo successivo (scenario realistico di test).
  Le due modalità usano esattamente gli stessi pesi e la stessa logica di stato.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from .config import N_FEATURES

_CELLS = {"lstm": keras.layers.LSTM, "gru": keras.layers.GRU}


def _rnn_layer(cell: str, units: int, return_sequences: bool, return_state: bool = False,
               name: str | None = None) -> keras.layers.Layer:
    """Crea un layer LSTM o GRU compatibile con cuDNN (nessun recurrent_dropout)."""
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
    """RNN impilata + testa densa che predice tutti i ``t_out`` passi futuri in un colpo solo.

    Con ``t_in=None`` il modello accetta finestre di lunghezza arbitraria (utile per la
    griglia T_in della sfida 1, pur addestrando un modello separato per ogni T_in).
    """
    inp = keras.Input(shape=(t_in, n_features), name="x")
    h = inp
    for i in range(n_layers):
        last = i == n_layers - 1
        h = _rnn_layer(cell, units, return_sequences=not last, name=f"{cell}_{i + 1}")(h)
        if dropout > 0:
            h = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(h)
    out = keras.layers.Dense(t_out * n_features, activation=output_activation, name="head")(h)
    out = keras.layers.Reshape((t_out, n_features), name="y")(out)
    return keras.Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# M2 - RNN seq2seq (encoder-decoder autoregressivo)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dlq")
class RNNSeq2Seq(keras.Model):
    """Encoder-decoder ricorrente con teacher forcing (``call``) e free-running (``free_running``)."""

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

        self.encoder_layers = [
            _rnn_layer(self.cell, units, return_sequences=True, return_state=True, name=f"enc_{i + 1}")
            for i in range(n_layers)
        ]
        self.decoder_layers = [
            _rnn_layer(self.cell, units, return_sequences=True, return_state=True, name=f"dec_{i + 1}")
            for i in range(n_layers)
        ]
        self.drop = keras.layers.Dropout(self.dropout_rate, name="dropout")
        self.head = keras.layers.Dense(n_features, activation=output_activation, name="head")

    # --- blocchi elementari -------------------------------------------------

    @staticmethod
    def _run(layer, x, initial_state=None, training=None):
        """Esegue un layer ricorrente restituendo (sequenza, lista_stati) per LSTM e GRU."""
        out = layer(x, initial_state=initial_state, training=training)
        return out[0], list(out[1:])

    def encode(self, x, training=None) -> list[list[tf.Tensor]]:
        """Stati finali di ciascun layer dell'encoder (usati per inizializzare il decoder)."""
        states, h = [], x
        for i, layer in enumerate(self.encoder_layers):
            h, st = self._run(layer, h, training=training)
            states.append(st)
            if i < self.n_layers - 1:
                h = self.drop(h, training=training)
        return states

    def decode(self, dec_in, states, training=None):
        """Applica il decoder a una sequenza (anche di lunghezza 1) a partire da ``states``."""
        new_states, h = [], dec_in
        for layer, st in zip(self.decoder_layers, states):
            h, ns = self._run(layer, h, initial_state=st, training=training)
            new_states.append(ns)
            h = self.drop(h, training=training)
        return self.head(h), new_states

    # --- modalità di utilizzo -----------------------------------------------

    def call(self, inputs, training=None):
        """Teacher forcing: ``inputs = (x, dec_in)`` con ``dec_in = [x[:, -1], y[:, :-1]]``."""
        x, dec_in = inputs
        y, _ = self.decode(dec_in, self.encode(x, training=training), training=training)
        return y

    def free_running(self, x, t_out: int | None = None, training=None):
        """Generazione autoregressiva: ogni predizione diventa l'input del passo successivo."""
        t_out = int(t_out or self.t_out)
        states = self.encode(x, training=training)
        token = x[:, -1:, :]
        outs = []
        for _ in range(t_out):
            token, states = self.decode(token, states, training=training)
            outs.append(token)
        return tf.concat(outs, axis=1)

    def predict_free_running(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Free-running a batch su array NumPy (compilato con ``tf.function`` per velocità)."""
        if not hasattr(self, "_fr_fn"):
            self._fr_fn = tf.function(lambda inp: self.free_running(inp, training=False))
        outs = [self._fr_fn(tf.convert_to_tensor(x[i:i + batch_size], dtype=tf.float32)).numpy()
                for i in range(0, len(x), batch_size)]
        return np.concatenate(outs, axis=0)

    def get_config(self) -> dict:
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
    """Costruisce M1 (``cfg['model'] == 'rnn_direct'``) o M2 (``'rnn'``) dai parametri di ``cfg``."""
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
        # build esplicito: crea i pesi così che summary() e save() funzionino subito
        t = t_in or cfg["t_in"]
        model((tf.zeros((1, t, N_FEATURES)), tf.zeros((1, cfg["t_out"], N_FEATURES))))
        return model
    raise ValueError(f"Modello ricorrente sconosciuto: {cfg['model']!r} (attesi 'rnn_direct' | 'rnn')")
