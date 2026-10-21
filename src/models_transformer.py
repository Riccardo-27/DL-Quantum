"""Architetture Transformer (Lab 7) per la previsione multi-orizzonte della dinamica quantistica.

Implementa fedelmente i blocchi modulari introdotti nel **Lab 7 – Attention and Transformers**
(`Transformer.ipynb` e `Attention.ipynb`), adattandoli al dominio delle serie temporali continue
a 55 osservabili fisiche ($10$ magnetizzazioni locali + $45$ correlazioni a due corpi) vincolate
all'intervallo $[-1, 1]$ tramite attivazione finale $\\tanh$:

1. **Codifica Posizionale Sinusoidale e Proiezione Lineare Continua** (`positional_encoding`, `PositionalEmbedding`):
   Sostituisce il layer discreto `Embedding` (usato in NLP nel Lab 7) con una proiezione densa
   continua `Dense(d_model)`, riscalata per $\\sqrt{d_{\\text{model}}}$ e sommata alla codifica
   posizionale sinusoidale deterministica su frequenze geometriche $1 / 10000^{2i / d_{\\text{model}}}$.

2. **Gerarchia di Attenzione del Lab 7** (`BaseAttention`, `GlobalSelfAttention`, `CausalSelfAttention`, `CrossAttention`):
   - `BaseAttention`: incapsula `MultiHeadAttention`, `LayerNormalization` e connessione residuale `Add` (schema Post-LN del Lab 7).
   - `GlobalSelfAttention`: self-attention bidirezionale sull'intero contesto passato $T_{\\text{in}}$ nell'Encoder.
   - `CausalSelfAttention`: self-attention causale (`use_causal_mask=True`) nel Decoder per impedire qualsiasi
     data leakage dai timestep futuri $t' > t$ durante il training parallelo.
   - `CrossAttention`: attenzione incrociata in cui il Decoder interroga (`query`) la memoria codificata dall'Encoder (`key`, `value`),
     salvando in `last_attn_scores` i pesi di attenzione di shape `(batch, num_heads, t_out, t_in)` per la Sfida 3.

3. **Due Famiglie Architetturali complete**:
   - **M3 – Transformer Diretto (Encoder-Only One-Shot)** (`TransformerDirect`, `build_transformer_direct`):
     L'Encoder contestualizza la finestra passata $(T_{\\text{in}}, 55)$ attraverso $L$ blocchi di `GlobalSelfAttention`
     e `FeedForward`; la rappresentazione contestualizzata dell'ultimo istante osservato viene proiettata
     in un unico passaggio (`Dense(t_out * 55, activation='tanh')` + `Reshape((t_out, 55))`) sull'intero orizzonte futuro.
   - **M4 – Transformer Seq2Seq Causale Autoregressivo** (`TransformerSeq2Seq`):
     Architettura Encoder-Decoder completa come nel Lab 7 (`Transformer`).
     * In training (`call`): opera in parallelo su `(x, dec_in)` grazie alla maschera triangolare inferiore di `CausalSelfAttention`,
       supportando nativamente il loop ibrido a 3 fasi (Teacher Forcing $\\to$ Masked Modeling $\\to$ Scheduled Sampling).
     * In inferenza (`free_running` / `predict_free_running`): genera la traiettoria futura un passo alla volta in circuito chiuso,
       partendo dall'ultimo istante osservato $x[:, -1:, :]$ e concatenando iterativamente ogni vettore predetto $\\hat{y}_t$.
     * Per l'interpretabilità fisica (Sfida 3, `get_attention_scores`): ricalcola in modalità eager fuori da `@tf.function`
       (esattamente come nella classe `Translator` del Lab 7) le matrici di `CrossAttention` e `CausalSelfAttention`.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from .config import N_FEATURES
from .windowing import decoder_inputs


# ---------------------------------------------------------------------------
# 1. Codifica Posizionale Sinusoidale e Proiezione Continua (Lab 7)
# ---------------------------------------------------------------------------


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Calcola la matrice di codifica posizionale sinusoidale come nel Lab 7 (`Transformer.ipynb`).

    Parametri:
        length: Lunghezza massima della sequenza temporale supportata.
        depth: Dimensione dello spazio latente d_model (deve essere pari).

    Restituisce:
        Tensore float32 di shape (length, depth) con colonne sin e cos alternate/concatenate.
    """
    depth_half = depth / 2
    positions = np.arange(length)[:, np.newaxis]              # (seq, 1)
    depths = np.arange(depth_half)[np.newaxis, :] / depth_half  # (1, depth/2)
    angle_rates = 1 / (10000**depths)                         # (1, depth/2)
    angle_rads = positions * angle_rates                      # (pos, depth/2)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


@keras.saving.register_keras_serializable(package="dlq")
class PositionalEmbedding(keras.layers.Layer):
    """Proietta i vettori continui a 55 osservabili in d_model e somma il Positional Encoding (Lab 7).

    Nel Lab 7 (traduzione automatica discreta) si utilizza `tf.keras.layers.Embedding`;
    per serie temporali quantistiche continue a valori reali in [-1, 1], la proiezione
    in ingresso è realizzata tramite uno strato lineare `Dense(d_model)` seguito dal
    fattore di scala $\\sqrt{d_{\\text{model}}}$ e dalla somma con `positional_encoding`.
    """

    def __init__(self, d_model: int, max_length: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.max_length = int(max_length)
        self.embedding = keras.layers.Dense(self.d_model, name="input_projection")
        self.pos_encoding = positional_encoding(length=self.max_length, depth=self.d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "max_length": self.max_length})
        return cfg


# ---------------------------------------------------------------------------
# 2. Blocchi di Attenzione Multi-Testa (Lab 7: Base, Cross, Global, Causal)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dlq")
class BaseAttention(keras.layers.Layer):
    """Classe base per i blocchi di attenzione con connessione residuale Add & LayerNorm (Lab 7)."""

    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.dropout_rate = float(dropout)
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
        )
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()
        self.last_attn_scores = None

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate,
        })
        return cfg


@keras.saving.register_keras_serializable(package="dlq")
class CrossAttention(BaseAttention):
    """Attenzione incrociata Decoder -> Encoder con memorizzazione degli score per la Sfida 3 (Lab 7)."""

    def call(self, x, context, training=None):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True,
            training=training,
        )
        if tf.executing_eagerly():
            self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@keras.saving.register_keras_serializable(package="dlq")
class GlobalSelfAttention(BaseAttention):
    """Self-Attention globale bidirezionale per i blocchi dell'Encoder (Lab 7)."""

    def call(self, x, training=None):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True,
            training=training,
        )
        if tf.executing_eagerly():
            self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@keras.saving.register_keras_serializable(package="dlq")
class CausalSelfAttention(BaseAttention):
    """Self-Attention causale con maschera triangolare inferiore (`use_causal_mask=True`) per il Decoder (Lab 7)."""

    def call(self, x, training=None):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
            return_attention_scores=True,
            training=training,
        )
        if tf.executing_eagerly():
            self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# ---------------------------------------------------------------------------
# 3. Rete Feed-Forward Point-Wise e Layer di Encoder / Decoder (Lab 7)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dlq")
class FeedForward(keras.layers.Layer):
    """Blocco Feed-Forward point-wise a due layer densi (ReLU + lineare) con Add & LayerNorm (Lab 7)."""

    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.dff = int(dff)
        self.dropout_rate = float(dropout_rate)
        self.seq = keras.Sequential([
            keras.layers.Dense(self.dff, activation="relu"),
            keras.layers.Dense(self.d_model),
            keras.layers.Dropout(self.dropout_rate),
        ])
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x, training=None):
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x)
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.saving.register_keras_serializable(package="dlq")
class EncoderLayer(keras.layers.Layer):
    """Singolo blocco dell'Encoder composto da GlobalSelfAttention + FeedForward (Lab 7)."""

    def __init__(self, *, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dff = int(dff)
        self.dropout_rate = float(dropout_rate)
        self.self_attention = GlobalSelfAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dropout_rate,
        )
        self.ffn = FeedForward(self.d_model, self.dff, self.dropout_rate)
        self.last_attn_scores = None

    def call(self, x, training=None):
        x = self.self_attention(x, training=training)
        if tf.executing_eagerly():
            self.last_attn_scores = self.self_attention.last_attn_scores
        x = self.ffn(x, training=training)
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.saving.register_keras_serializable(package="dlq")
class Encoder(keras.layers.Layer):
    """Stack completo dell'Encoder: PositionalEmbedding -> Dropout -> N x EncoderLayer (Lab 7)."""

    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.0,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dff = int(dff)
        self.dropout_rate = float(dropout_rate)
        self.max_length = int(max_length)

        self.pos_embedding = PositionalEmbedding(d_model=self.d_model, max_length=self.max_length)
        self.enc_layers = [
            EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.last_attn_scores = None

    def call(self, x, training=None):
        x = self.pos_embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        if tf.executing_eagerly():
            self.last_attn_scores = self.enc_layers[-1].last_attn_scores
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "max_length": self.max_length,
        })
        return cfg


@keras.saving.register_keras_serializable(package="dlq")
class DecoderLayer(keras.layers.Layer):
    """Singolo blocco del Decoder: CausalSelfAttention -> CrossAttention -> FeedForward (Lab 7)."""

    def __init__(self, *, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dff = int(dff)
        self.dropout_rate = float(dropout_rate)

        self.causal_self_attention = CausalSelfAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dropout_rate,
        )
        self.cross_attention = CrossAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dropout_rate,
        )
        self.ffn = FeedForward(self.d_model, self.dff, self.dropout_rate)
        self.last_attn_scores = None
        self.last_causal_attn_scores = None

    def call(self, x, context, training=None):
        x = self.causal_self_attention(x=x, training=training)
        x = self.cross_attention(x=x, context=context, training=training)
        if tf.executing_eagerly():
            self.last_causal_attn_scores = self.causal_self_attention.last_attn_scores
            self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x, training=training)
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


@keras.saving.register_keras_serializable(package="dlq")
class Decoder(keras.layers.Layer):
    """Stack completo del Decoder: PositionalEmbedding -> Dropout -> N x DecoderLayer (Lab 7)."""

    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.0,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dff = int(dff)
        self.dropout_rate = float(dropout_rate)
        self.max_length = int(max_length)

        self.pos_embedding = PositionalEmbedding(d_model=self.d_model, max_length=self.max_length)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        self.last_attn_scores = None
        self.last_causal_attn_scores = None

    def call(self, x, context, training=None):
        x = self.pos_embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context, training=training)
        if tf.executing_eagerly():
            self.last_attn_scores = self.dec_layers[-1].last_attn_scores
            self.last_causal_attn_scores = self.dec_layers[-1].last_causal_attn_scores
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "max_length": self.max_length,
        })
        return cfg


# ---------------------------------------------------------------------------
# 4. M3 - Transformer Diretto (Encoder-Only One-Shot)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dlq")
class TransformerDirect(keras.Model):
    """M3 – Architettura Transformer Encoder-Only con proiezione diretta one-shot (t_out, 55).

    L'Encoder elabora la sequenza passata (batch, t_in, 55) tramite GlobalSelfAttention;
    il vettore contestualizzato dell'ultimo istante osservato `context[:, -1, :]` (che ha
    atteso sull'intera finestra passata) viene proiettato da uno strato `Dense(t_out * 55, activation='tanh')`
    e rimodellato in `(batch, t_out, 55)`. Non impiega retroazione autoregressiva, risultando
    immune per costruzione all'exposure bias e compatibile con finestre di lunghezza arbitraria.
    """

    def __init__(
        self,
        t_out: int = 40,
        n_features: int = N_FEATURES,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        output_activation: str = "tanh",
        **kwargs,
    ):
        kwargs.setdefault("name", "transformer_direct")
        super().__init__(**kwargs)
        self.t_out = int(t_out)
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.n_layers = int(n_layers)
        self.dropout_rate = float(dropout)
        self.output_activation = output_activation

        self.encoder = Encoder(
            num_layers=self.n_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.d_ff,
            dropout_rate=self.dropout_rate,
            name="encoder",
        )
        self.head = keras.layers.Dense(
            self.t_out * self.n_features,
            activation=self.output_activation,
            name="head",
        )
        self.reshape = keras.layers.Reshape((self.t_out, self.n_features), name="y")

    def call(self, x, training=None):
        context = self.encoder(x, training=training)  # (batch, t_in, d_model)
        last_step = context[:, -1, :]                 # (batch, d_model)
        out = self.head(last_step)                    # (batch, t_out * n_features)
        return self.reshape(out)                      # (batch, t_out, n_features)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "t_out": self.t_out,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "dropout": self.dropout_rate,
            "output_activation": self.output_activation,
        })
        return cfg


def build_transformer_direct(
    t_out: int = 40,
    t_in: int | None = 10,
    n_features: int = N_FEATURES,
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 128,
    n_layers: int = 2,
    dropout: float = 0.0,
    output_activation: str = "tanh",
    name: str = "transformer_direct",
) -> TransformerDirect:
    """Istanzia e inizializza i pesi del modello M3 (`TransformerDirect`)."""
    model = TransformerDirect(
        t_out=t_out,
        n_features=n_features,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        output_activation=output_activation,
        name=name,
    )
    t = int(t_in or 10)
    model(tf.zeros((1, t, n_features)))
    return model


# ---------------------------------------------------------------------------
# 5. M4 - Transformer Seq2Seq Causale Autoregressivo (Lab 7)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dlq")
class TransformerSeq2Seq(keras.Model):
    """M4 – Architettura Transformer Encoder-Decoder Causale completa (Lab 7 `Transformer`).

    Riproduce esattamente la classe `Transformer` del Lab 7 (`Encoder` + `Decoder` + `final_layer`),
    aggiungendo i metodi per il ciclo ibrido e per la previsione multi-step:
    - `call((x, dec_in))`: Teacher Forcing parallelo con maschera causale triangolare inferiore.
    - `free_running(x, t_out)`: Generazione autoregressiva passo-passo ad anello chiuso.
    - `predict_free_running(x, batch_size)`: Inferenza compilata con `@tf.function` per massima velocità su GPU.
    - `get_attention_scores(x, dec_in)`: Estrazione eager delle mappe di Cross-Attention e Causal Self-Attention (Sfida 3).
    """

    def __init__(
        self,
        t_out: int = 40,
        n_features: int = N_FEATURES,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        output_activation: str = "tanh",
        **kwargs,
    ):
        kwargs.setdefault("name", "transformer_seq2seq")
        super().__init__(**kwargs)
        self.t_out = int(t_out)
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.n_layers = int(n_layers)
        self.dropout_rate = float(dropout)
        self.output_activation = output_activation

        self.encoder = Encoder(
            num_layers=self.n_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.d_ff,
            dropout_rate=self.dropout_rate,
            name="encoder",
        )
        self.decoder = Decoder(
            num_layers=self.n_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.d_ff,
            dropout_rate=self.dropout_rate,
            name="decoder",
        )
        self.final_layer = keras.layers.Dense(
            self.n_features,
            activation=self.output_activation,
            name="head",
        )

    def call(self, inputs, training=None):
        """Modalità Teacher Forcing parallela con maschera causale triangolare nel Decoder."""
        x, dec_in = inputs
        context = self.encoder(x, training=training)               # (batch, t_in, d_model)
        dec_out = self.decoder(dec_in, context, training=training)  # (batch, t_out, d_model)
        return self.final_layer(dec_out)                           # (batch, t_out, n_features)

    def free_running(self, x, t_out: int | None = None, training=None):
        """Modalità Autoregressiva / Free-Running (valutazione realistica a circuito chiuso).

        Come nella classe `Translator` del Lab 7 (`Transformer.ipynb`), codifica una sola volta
        il contesto passato `x` nell'Encoder e costruisce iterativamente la sequenza del Decoder
        partendo dall'ultimo istante osservato `x[:, -1:, :]` e concatenando a ogni passo $t$
        l'ultimo token predetto `next_token = final_layer(dec_out[:, -1:, :])`.
        """
        t_out = int(t_out or self.t_out)
        context = self.encoder(x, training=training)
        dec_seq = x[:, -1:, :]  # Token iniziale t=0: ultimo passo osservato (batch, 1, 55)
        outs = []
        for step in range(t_out):
            dec_out = self.decoder(dec_seq, context, training=training)
            next_token = self.final_layer(dec_out[:, -1:, :])
            outs.append(next_token)
            if step < t_out - 1:
                dec_seq = tf.concat([dec_seq, next_token], axis=1)
        return tf.concat(outs, axis=1)

    def predict_free_running(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Inferenza Free-Running compilata con `@tf.function` e suddivisa in mini-batch."""
        if not hasattr(self, "_fr_fn"):
            self._fr_fn = tf.function(
                lambda inp: self.free_running(inp, training=False),
                reduce_retracing=True,
            )
        outs = [
            self._fr_fn(tf.convert_to_tensor(x[i : i + batch_size], dtype=tf.float32)).numpy()
            for i in range(0, len(x), batch_size)
        ]
        return np.concatenate(outs, axis=0)

    def get_attention_scores(
        self,
        x: np.ndarray,
        dec_in: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Estrae in modalità eager le mappe di attenzione dell'ultimo blocco per la Sfida 3.

        Seguendo la prescrizione del Lab 7 (`Translator.__call__`), poiché `@tf.function`
        impedisce di esportare direttamente gli attributi Python `last_attn_scores` calcolati
        dentro il grafo compilato, esegue un singolo passaggio forward eager fuori dal loop
        alimentando il decoder con la sequenza generata in free-running (o con `dec_in` se fornito).

        Restituisce un dizionario con:
        - `'cross_attention'`: array `(batch, num_heads, t_out, t_in)` (Decoder -> Encoder)
        - `'causal_self_attention'`: array `(batch, num_heads, t_out, t_out)` (Decoder causale triangolare)
        - `'encoder_self_attention'`: array `(batch, num_heads, t_in, t_in)` (Encoder self-attention)
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if dec_in is None:
            y_fr = self.predict_free_running(x_arr, batch_size=len(x_arr))
            dec_arr = decoder_inputs(x_arr, y_fr)
        else:
            dec_arr = np.asarray(dec_in, dtype=np.float32)

        _ = self(
            (tf.convert_to_tensor(x_arr, dtype=tf.float32), tf.convert_to_tensor(dec_arr, dtype=tf.float32)),
            training=False,
        )
        return {
            "cross_attention": self.decoder.last_attn_scores.numpy(),
            "causal_self_attention": self.decoder.last_causal_attn_scores.numpy(),
            "encoder_self_attention": self.encoder.last_attn_scores.numpy(),
        }

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "t_out": self.t_out,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "dropout": self.dropout_rate,
            "output_activation": self.output_activation,
        })
        return cfg


# ---------------------------------------------------------------------------
# 6. Factory guidata dalla configurazione centralizzata e Custom Objects
# ---------------------------------------------------------------------------

CUSTOM_OBJECTS: dict[str, type] = {
    "PositionalEmbedding": PositionalEmbedding,
    "BaseAttention": BaseAttention,
    "CrossAttention": CrossAttention,
    "GlobalSelfAttention": GlobalSelfAttention,
    "CausalSelfAttention": CausalSelfAttention,
    "FeedForward": FeedForward,
    "EncoderLayer": EncoderLayer,
    "Encoder": Encoder,
    "DecoderLayer": DecoderLayer,
    "Decoder": Decoder,
    "TransformerDirect": TransformerDirect,
    "TransformerSeq2Seq": TransformerSeq2Seq,
}


def build_transformer_from_config(cfg: dict, t_in: int | None = None) -> keras.Model:
    """Costruisce e istanzia il modello Transformer definito nel dizionario `cfg`.

    Supporta:
    - `cfg['model'] == 'transformer_direct'`: Costruisce M3 (`TransformerDirect` Encoder-Only one-shot).
    - `cfg['model'] == 'transformer'`: Costruisce M4 (`TransformerSeq2Seq` Encoder-Decoder causale).
    """
    common = dict(
        t_out=cfg["t_out"],
        n_features=N_FEATURES,
        d_model=cfg["transformer_d_model"],
        num_heads=cfg["transformer_heads"],
        d_ff=cfg["transformer_d_ff"],
        n_layers=cfg["transformer_layers"],
        dropout=cfg["dropout"],
        output_activation=cfg["output_activation"],
    )
    t = int(t_in or cfg["t_in"])
    if cfg["model"] == "transformer_direct":
        return build_transformer_direct(t_in=t, **common)
    if cfg["model"] == "transformer":
        model = TransformerSeq2Seq(**common)
        model((tf.zeros((1, t, N_FEATURES)), tf.zeros((1, cfg["t_out"], N_FEATURES))))
        return model
    raise ValueError(
        f"Modello Transformer sconosciuto: {cfg['model']!r} (attesi 'transformer_direct' | 'transformer')"
    )

