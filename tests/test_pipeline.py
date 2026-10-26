"""Suite di test unitari e di integrazione per la pipeline DL-Quantum.

Copertura delle verifiche:
1. Riproducibilità e Seeding: determinismo numerico dei generatori pseudo-casuali (CPU/GPU).
2. Costruzione Feature: coerenza del mapping tra 10 qubit e 55 osservabili quantistiche.
3. Finestrature Temporali: sliding window vincolata per traiettoria, target alignment e super-risoluzione.
4. Partizionamento e Anti-Leakage: split disgiunti per traiettoria e verifica metadati.
5. Metriche e Statistica: MSE/MAE complessivo e disaggregato, CI al 95% con t di Student (n-1 g.d.l.).
6. Fisica Quantistica: positività semidefinita delle matrici di densità a 2 corpi (P(s_i, s_j) >= 0).
7. Modelli Ricorrenti (RNN): architetture M1 e M2 (LSTM/GRU), range fisico tanh [-1, 1],
   free-running vs teacher forcing autoreferenziale.
8. Training Loop Ibrido: determinismo e progressione fasi TF -> MM -> SS (Scheduled Sampling).
9. Visualizzazione ed Export: conformità JPEG ad alta risoluzione (>= 300 DPI).
10. Checkpointing: persistenza e ripristino modelli nativi Keras 3 (.keras).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Inserimento della radice di progetto nel sys.path per importare src
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import N_FEATURES, N_POINTS, N_QUBITS, make_config
from src.data import (
    check_quantum_probabilities,
    corr_pairs,
    feature_names,
    kfold_splits,
    make_split,
)
from src.metrics import (
    corr_pair_mse_matrix,
    evaluate,
    per_horizon_group_mse,
    per_horizon_mse,
    per_qubit_mse,
    per_variable_mse,
)
from src.plots import plot_trajectory, plot_window, save_figure
from src.results_log import append_result, load_results
from src.seeding import set_seeds
from src.stats import format_mean_ci, summarize
from src.windowing import (
    check_no_leakage,
    decoder_inputs,
    make_sr_windows,
    make_windows,
)


def test_seeding_determinism():
    """Verifica che il seeding centralizzato garantisca la perfetta riproducibilità numerica."""
    # Primo campionamento con seed 42
    set_seeds(42)
    a = np.random.randn(10, 10)

    # Secondo campionamento indipendente reimpostando il seed 42
    set_seeds(42)
    b = np.random.randn(10, 10)

    # I due array devono coincidere esattamente elemento per elemento
    np.testing.assert_array_equal(
        a, b, err_msg="I tensori generati con lo stesso seed devono essere identici."
    )


def test_feature_names_count():
    """Verifica che il numero di osservabili quantistiche corrisponda a 10 magnetizzazioni e 45 correlazioni."""
    # Nomi di tutte le 55 osservabili: <sigma_z^i> (10) e <sigma_z^i sigma_z^j> (45)
    names = feature_names(N_QUBITS)
    assert len(names) == N_FEATURES, f"Attesi {N_FEATURES} nomi, trovati {len(names)}"

    # Verifica conteggio delle coppie non ordinate (i < j) per la catena di 10 atomi
    pairs = corr_pairs(N_QUBITS)
    assert len(pairs) == 45, f"Attese 45 correlazioni a due corpi, trovate {len(pairs)}"


def test_windows_shapes():
    """Verifica la generazione a finestra scorrevole per traiettoria e la costruzione degli input del decoder."""
    # Dati sintetici: 5 traiettorie da 100 istanti temporali e 55 feature
    n_traj, n_points, n_feat = 5, 100, 55
    dummy_data = np.random.randn(n_traj, n_points, n_feat).astype(np.float32)
    idx = np.array([0, 1, 2])

    t_in, t_out, stride = 10, 40, 10
    X, Y, meta = make_windows(dummy_data, idx, t_in=t_in, t_out=t_out, stride=stride)

    # Con span = t_in + t_out = 50 e stride = 10 su 100 punti:
    # partenze valide = {0, 10, 20, 30, 40, 50} -> 6 finestre per traiettoria
    expected_win_per_traj = (n_points - (t_in + t_out)) // stride + 1
    expected_total_win = len(idx) * expected_win_per_traj

    # Verifica dimensioni dei tensori estratti
    assert X.shape == (expected_total_win, t_in, n_feat), f"Shape X inattesa: {X.shape}"
    assert Y.shape == (expected_total_win, t_out, n_feat), f"Shape Y inattesa: {Y.shape}"
    assert meta.shape == (expected_total_win, 2)  # [traj_id, start_time_step]

    # Verifica allineamento per Teacher Forcing: dec_in[:, 0] = x[:, -1], dec_in[:, 1:] = y[:, :-1]
    dec_in = decoder_inputs(X, Y)
    assert dec_in.shape == Y.shape
    np.testing.assert_array_equal(dec_in[:, 0], X[:, -1])
    np.testing.assert_array_equal(dec_in[:, 1:], Y[:, :-1])


def test_sr_windows():
    """Verifica l'estrazione delle finestre di Super-Risoluzione con sottocampionamento temporale x2."""
    dummy_data = np.random.randn(2, 100, 55).astype(np.float32)
    idx = np.array([0])
    t_in, t_out, stride, factor = 20, 40, 10, 2

    # X_sr viene campionata a passo rado (dt = 0.04), mentre Y rimane a risoluzione piena (dt = 0.02)
    X_sr, Y, meta = make_sr_windows(
        dummy_data, idx, t_in=t_in, t_out=t_out, stride=stride, factor=factor
    )
    assert X_sr.shape[1] == t_in
    assert Y.shape[1] == t_out


def test_aligned_windows_share_targets():
    """Verifica che con align_t_in=40 finestre con T_in diversi condividano esattamente gli stessi target Y."""
    dummy_data = np.random.randn(3, 1001, 55).astype(np.float32)
    idx = np.array([0, 2])
    ys = {}

    # Genera finestre per la griglia di contesti passati T_in in {5, 10, 20, 40}
    for t_in in (5, 10, 20, 40):
        X, Y, meta = make_windows(
            dummy_data, idx, t_in=t_in, t_out=40, stride=10, align_t_in=40
        )
        assert X.shape[1] == t_in
        # Verifica che il confine temporale non superi la durata della simulazione (1001 step)
        assert np.all(meta[:, 1] + t_in + 40 <= 1001)
        # Verifica che l'istante di inizio della predizione futura sia un multiplo esatto dello stride
        assert np.all((meta[:, 1] + t_in - 40) % 10 == 0)
        ys[t_in] = Y

    # Controllo di identità: i target Y per T_in in {5, 10, 20} devono essere indistinguibili da T_in=40
    for t_in in (5, 10, 20):
        np.testing.assert_array_equal(ys[t_in], ys[40])


def test_split_and_no_leakage():
    """Verifica la correttezza del partizionamento per traiettoria e l'assenza di sovrapposizioni tra split."""
    split = make_split(n_traj=400, n_train=280, n_val=60, n_test=60, seed=42)
    assert len(split["train"]) == 280
    assert len(split["val"]) == 60
    assert len(split["test"]) == 60

    # Verifica rigorosa di disgiunzione insiemistica: zero data leakage tra train, val e test
    assert len(set(split["train"]) & set(split["val"])) == 0
    assert len(set(split["train"]) & set(split["test"])) == 0
    assert len(set(split["val"]) & set(split["test"])) == 0

    # Verifica della funzione sentinella check_no_leakage su metadati sintetici
    meta_tr = np.array([[tid, 0] for tid in split["train"]])
    meta_va = np.array([[tid, 0] for tid in split["val"]])
    meta_te = np.array([[tid, 0] for tid in split["test"]])
    check_no_leakage({"train": meta_tr, "val": meta_va, "test": meta_te})


def test_metrics_and_stats():
    """Verifica il calcolo delle metriche di regressione e degli intervalli di confidenza con t di Student."""
    # Dati fittizi con errore costante unitario: |1.0 - 0.0| = 1.0 -> MSE = 1.0, MAE = 1.0
    y_true = np.ones((10, 40, 55), dtype=np.float32)
    y_pred = np.zeros((10, 40, 55), dtype=np.float32)

    res = evaluate(y_true, y_pred)
    assert np.isclose(res["mse"], 1.0)
    assert np.isclose(res["mae"], 1.0)
    assert np.isclose(res["mse_mag"], 1.0)
    assert np.isclose(res["mse_corr"], 1.0)

    # Profilo per-variabile lungo le 55 osservabili
    pvar = per_variable_mse(y_true, y_pred)
    assert pvar.shape == (55,)
    assert np.allclose(pvar, 1.0)

    # Profilo per-orizzonte lungo i 40 step futuri
    phor = per_horizon_mse(y_true, y_pred)
    assert phor.shape == (40,)
    assert np.allclose(phor, 1.0)

    # Test statistico con t di Student su n=3 ripetizioni indipendenti
    stats = summarize([1.0, 1.1, 0.9])
    assert stats["n"] == 3
    assert np.isclose(stats["mean"], 1.0)
    assert stats["ci_half"] > 0
    # Con n=3 (dof=2) il quantile critico al 95% e' t_{0.975, 2} approx 4.30265
    assert np.isclose(stats["ci_half"], 4.30265 * stats["std"] / np.sqrt(3), atol=1e-3)


def test_kfold_splits():
    """Verifica che la suddivisione K-fold generi fold disgiunti ed esaustivi senza duplicati."""
    indices = np.arange(20)
    folds = kfold_splits(indices, k=5, seed=42)
    assert len(folds) == 5

    seen_val = []
    for train_idx, val_idx in folds:
        assert len(val_idx) == 4
        assert len(train_idx) == 16
        # Nessuna sovrapposizione tra training set e validation set del singolo fold
        assert len(set(train_idx) & set(val_idx)) == 0
        seen_val.extend(val_idx.tolist())

    # La validazione copre tutti i 20 indici esattamente una sola volta
    assert sorted(seen_val) == list(range(20))


def test_results_logging(tmp_path):
    """Verifica la persistenza incrementale e il caricamento dei risultati sperimentali in CSV."""
    log_file = tmp_path / "results_log_test.csv"

    # Scrittura del primo record sperimentale
    rec1 = {"model": "rnn", "val_mse": 0.001, "seeds": [42, 43]}
    append_result(rec1, path=log_file)
    assert log_file.exists()

    # Scrittura del secondo record contenente colonne eterogenee (schema evolutivo)
    rec2 = {"model": "rnn", "val_mse": 0.0009, "new_metric": 0.99}
    append_result(rec2, path=log_file)

    # Rilettura aggregata del file di log
    df = load_results(tmp_path)
    assert len(df) == 2
    assert "val_mse" in df.columns
    assert "new_metric" in df.columns
    assert df.iloc[0]["model"] == "rnn"


def test_quantum_probabilities():
    """Verifica il rispetto della positivita' semidefinita delle matrici di densita' a 2 corpi."""
    # 1. Stato fisico con magnetizzazioni e correlazioni nulle:
    # P(s_i, s_j) = 1/4 = 0.25 >= 0 per tutte le 45 coppie quantistiche
    zero_state = np.zeros((1, N_FEATURES), dtype=np.float32)
    res_zero = check_quantum_probabilities(zero_state)
    assert res_zero["is_physical"] is True
    assert res_zero["violations_count"] == 0
    assert np.isclose(res_zero["min_probability"], 0.25)

    # 2. Configurazione non fisica sintetica: m1 = 1, m2 = 1, c1_2 = -1
    # P(-1, -1) = (1 - 1 - 1 - (-1)) / 4 = 0, ma P(+1, -1) = (1 + 1 - 1 - (-1)) / 4 = 0.5
    # Per s1=-1, s2=-1: P(-1, -1) = (1 - 1*(1) - 1*(1) + (-1)) / 4 = -2/4 = -0.5 < 0 (violazione quantistica)
    unphysical_state = np.zeros((1, N_FEATURES), dtype=np.float32)
    unphysical_state[0, 0] = 1.0   # <sigma_z^1>
    unphysical_state[0, 1] = 1.0   # <sigma_z^2>
    unphysical_state[0, 10] = -1.0  # <sigma_z^1 sigma_z^2>
    res_unphys = check_quantum_probabilities(unphysical_state)
    assert res_unphys["is_physical"] is False
    assert res_unphys["violations_count"] > 0
    assert res_unphys["min_probability"] < 0


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_rnn_models(cell):
    """Verifica la costruzione, output shapes e confini fisici tanh delle architetture ricorrenti (LSTM/GRU)."""
    pytest.importorskip("tensorflow")
    from src.models_rnn import RNNSeq2Seq, build_rnn_direct, build_rnn_from_config

    set_seeds(42)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (4, 10, N_FEATURES)).astype(np.float32)
    y = rng.uniform(-1, 1, (4, 40, N_FEATURES)).astype(np.float32)

    # M1 diretta: shape corretta e output confinato nell'intervallo fisico [-1, 1] dalla tanh
    m1 = build_rnn_direct(t_out=40, t_in=10, units=8, n_layers=2, cell=cell)
    y1 = np.asarray(m1(x))
    assert y1.shape == (4, 40, N_FEATURES)
    assert np.all(np.abs(y1) <= 1.0)

    # M2 seq2seq: verifica inferenza in teacher forcing e free-running
    m2 = RNNSeq2Seq(t_out=40, units=8, n_layers=2, cell=cell)
    assert np.asarray(m2((x, decoder_inputs(x, y)))).shape == (4, 40, N_FEATURES)
    y_fr = np.asarray(m2.free_running(x))
    assert y_fr.shape == (4, 40, N_FEATURES)

    # Proprieta' fondamentale di coerenza: se il decoder riceve in input le sue stesse predizioni,
    # il teacher forcing deve replicare esattamente l'output del free-running
    y_tf_self = np.asarray(m2((x, decoder_inputs(x, y_fr))))
    np.testing.assert_allclose(y_tf_self, y_fr, atol=1e-5)
    np.testing.assert_allclose(m2.predict_free_running(x, batch_size=3), y_fr, atol=1e-5)

    # Verifica corretta istanziazione tramite factory da configurazione
    for model_name in ("rnn_direct", "rnn"):
        cfg = make_config(model=model_name, rnn_units=8, rnn_cell=cell)
        assert build_rnn_from_config(cfg, t_in=10).count_params() > 0


def test_training_helpers():
    """Verifica le utilita' ausiliarie del loop di addestramento: schedule SS, mascheramento e mix input."""
    pytest.importorskip("tensorflow")
    from src.training import mask_timesteps, mix_decoder_inputs, ss_probability

    # Verifica linear scheduling della probabilita' di Scheduled Sampling p_ss in [p_start, p_end]
    assert ss_probability(0, 5, 0.0, 0.5) == 0.0
    assert ss_probability(4, 5, 0.0, 0.5) == 0.5
    assert ss_probability(0, 1, 0.0, 0.5) == 0.5

    # Verifica mascheramento temporale del passato (Masked Modeling): azzera istanti interi preservando l'ultimo
    rng = np.random.default_rng(0)
    x = np.ones((50, 10, N_FEATURES), dtype=np.float32)
    xm = mask_timesteps(x, 0.5, rng, protect=-1)
    assert np.all(xm[:, -1] == 1.0)                                # Ultimo istante sempre protetto
    assert 0.3 < (xm[:, :-1, 0] == 0).mean() < 0.7                  # Circa 50% degli altri istanti mascherati
    assert np.all((xm == 0).all(axis=2) | (xm == 1).all(axis=2))    # Mascheramento sull'intero vettore spaziale

    # Verifica miscelazione dei target del decoder tra ground-truth e predizioni correnti
    y = np.zeros((50, 40, N_FEATURES), dtype=np.float32)
    y_hat = np.full_like(y, 0.5)
    np.testing.assert_array_equal(mix_decoder_inputs(x, y, y_hat, 0.0, rng), decoder_inputs(x, y))
    np.testing.assert_array_equal(mix_decoder_inputs(x, y, y_hat, 1.0, rng), decoder_inputs(x, y_hat))


@pytest.mark.parametrize("model_name", ["rnn_direct", "rnn"])
def test_train_hybrid_runs_and_is_deterministic(model_name):
    """Verifica che il training loop ibrido (TF -> MM -> SS) sia eseguito senza errori e sia deterministico."""
    pytest.importorskip("tensorflow")
    from src.models_rnn import build_rnn_from_config
    from src.training import train_hybrid

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (32, 10, N_FEATURES)).astype(np.float32)
    Y = rng.uniform(-1, 1, (32, 40, N_FEATURES)).astype(np.float32)
    cfg = make_config(
        model=model_name,
        rnn_units=8,
        epochs_tf=1,
        epochs_mm=1,
        epochs_ss=2,
        batch_size=16,
    )

    def run():
        set_seeds(42)
        model = build_rnn_from_config(cfg, t_in=10)
        return train_hybrid(model, X, Y, X[:8], Y[:8], cfg, verbose=0)

    # Due esecuzioni indipendenti con lo stesso seed devono produrre traiettorie di loss identiche
    h1, h2 = run(), run()
    assert [r["phase"] for r in h1] == ["tf", "mm", "ss", "ss"]
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["val_mse"]) for r in h1)
    assert [r["ss_p"] for r in h1][-2:] == [0.0, 0.5]
    assert [r["loss"] for r in h1] == [r["loss"] for r in h2]
    assert [r["val_mse"] for r in h1] == [r["val_mse"] for r in h2]


def test_disaggregated_metrics():
    """Verifica le metriche disaggregate: per gruppo temporale, per singolo qubit e matrice di correlazione 10x10."""
    # Tensori fittizi con errore costante pari a 1.0: (1 - 0)^2 = 1.0
    y_true = np.ones((5, 40, N_FEATURES), dtype=np.float32)
    y_pred = np.zeros((5, 40, N_FEATURES), dtype=np.float32)

    # 1. Separazione dell'orizzonte futuro tra magnetizzazioni e correlazioni a due corpi
    grp = per_horizon_group_mse(y_true, y_pred)
    assert grp["mag"].shape == (40,)
    assert grp["corr"].shape == (40,)
    assert np.allclose(grp["mag"], 1.0)
    assert np.allclose(grp["corr"], 1.0)

    # 2. Errore disaggregato per singolo qubit spaziale (10 atomi di Rydberg)
    q_mse = per_qubit_mse(y_true, y_pred)
    assert q_mse.shape == (N_QUBITS,)
    assert np.allclose(q_mse, 1.0)

    # 3. Matrice simmetrica 10x10 per le correlazioni quantistiche con diagonale NaN (auto-correlazioni banali omesse)
    mat = corr_pair_mse_matrix(y_true, y_pred, n_qubits=N_QUBITS)
    assert mat.shape == (N_QUBITS, N_QUBITS)
    assert np.all(np.isnan(np.diag(mat)))  # Diagonale assente per definizione
    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            assert np.isclose(mat[i, j], 1.0)
            assert np.isclose(mat[j, i], 1.0)


def test_plots_export(tmp_path, monkeypatch):
    """Verifica la generazione e l'esportazione di figure ad alta risoluzione conforme a standard JPEG (>= 300 DPI)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # Reindirizza temporaneamente la cartella figure verso la cartella isolata tmp_path
    monkeypatch.setenv("DLQ_ROOT", str(tmp_path))

    fig, ax = plt.subplots(figsize=(6, 3))
    x_dummy = np.random.randn(10, N_FEATURES).astype(np.float32)
    y_dummy = np.random.randn(40, N_FEATURES).astype(np.float32)
    plot_window(x_dummy, y_dummy, feature=0, ax=ax, title="Test Window")

    out_path = save_figure(fig, "test_plot.jpeg", subdir="test_sub")
    plt.close(fig)

    # Controllo presenza, estensione e densita' di pixel DPI >= 300
    assert out_path.exists()
    assert out_path.suffix == ".jpeg"
    with Image.open(out_path) as img:
        assert img.format == "JPEG"
        dpi = img.info.get("dpi", (300, 300))
        assert dpi[0] >= 300


def test_rnn_model_save_and_reload(tmp_path):
    """Verifica che il salvataggio e il ricaricamento del checkpoint Keras 3 preservino l'inferenza in free-running."""
    pytest.importorskip("tensorflow")
    import keras
    from src.models_rnn import RNNSeq2Seq, build_rnn_from_config

    set_seeds(42)
    cfg = make_config(model="rnn", rnn_units=8, rnn_cell="lstm")
    # Modello istanziato e costruito esplicitamente
    model = build_rnn_from_config(cfg, t_in=10)
    x = np.random.uniform(-1, 1, (2, 10, N_FEATURES)).astype(np.float32)
    pred_orig = model.predict_free_running(x, batch_size=2)

    # Salvataggio nel formato nativo .keras
    ckpt_file = tmp_path / "model_test.keras"
    model.save(ckpt_file)
    assert ckpt_file.exists()

    # Ricaricamento con registrazione del custom object RNNSeq2Seq
    loaded = keras.models.load_model(ckpt_file, custom_objects={"RNNSeq2Seq": RNNSeq2Seq})
    pred_loaded = loaded.predict_free_running(x, batch_size=2)

    # Le traiettorie predette devono coincidere numericamente elemento per elemento
    np.testing.assert_allclose(pred_orig, pred_loaded, atol=1e-5)


def test_transformer_models_and_attention():
    """Verifica M3 (TransformerDirect) e M4 (TransformerSeq2Seq): shapes, tanh, causal mask e attention scores."""
    pytest.importorskip("tensorflow")
    from src.models_transformer import (
        TransformerDirect,
        TransformerSeq2Seq,
        build_transformer_direct,
        build_transformer_from_config,
        positional_encoding,
    )

    set_seeds(42)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (4, 10, N_FEATURES)).astype(np.float32)
    y = rng.uniform(-1, 1, (4, 40, N_FEATURES)).astype(np.float32)

    # 1. Codifica posizionale sinusoidale (Lab 7)
    pe = np.asarray(positional_encoding(length=50, depth=16))
    assert pe.shape == (50, 16)
    assert np.all(np.abs(pe) <= 1.0)

    # 2. M3 (TransformerDirect): shape (batch, t_out, 55) e vincolo fisico tanh in [-1, 1]
    m3 = build_transformer_direct(t_out=40, t_in=10, d_model=16, num_heads=2, d_ff=32, n_layers=1)
    y3 = np.asarray(m3(x))
    assert isinstance(m3, TransformerDirect)
    assert y3.shape == (4, 40, N_FEATURES)
    assert np.all(np.abs(y3) <= 1.0)

    # 3. M4 (TransformerSeq2Seq): coerenza tra Teacher Forcing auto-alimentato e Free-Running
    m4 = TransformerSeq2Seq(t_out=40, d_model=16, num_heads=2, d_ff=32, n_layers=1)
    y4_tf = np.asarray(m4((x, decoder_inputs(x, y)), training=False))
    assert y4_tf.shape == (4, 40, N_FEATURES)
    assert np.all(np.abs(y4_tf) <= 1.0)

    y4_fr = np.asarray(m4.free_running(x, training=False))
    assert y4_fr.shape == (4, 40, N_FEATURES)
    y4_tf_self = np.asarray(m4((x, decoder_inputs(x, y4_fr)), training=False))
    np.testing.assert_allclose(y4_tf_self, y4_fr, atol=1e-5)
    np.testing.assert_allclose(m4.predict_free_running(x, batch_size=2), y4_fr, atol=1e-5)

    # 4. Sfida 3: Estrazione attention scores e verifica della maschera causale triangolare inferiore
    attn = m4.get_attention_scores(x[:2])
    cross = attn["cross_attention"]
    causal = attn["causal_self_attention"]
    enc_self = attn["encoder_self_attention"]
    assert cross.shape == (2, 2, 40, 10)
    assert causal.shape == (2, 2, 40, 40)
    assert enc_self.shape == (2, 2, 10, 10)
    # Le righe delle matrici di attenzione normalizzate con softmax sommano a 1
    np.testing.assert_allclose(cross.sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(causal.sum(axis=-1), 1.0, atol=1e-5)
    # Verifica causalita' temporale anti-leakage: la parte strettamente triangolare superiore (j > i) deve essere zero
    upper_tri = np.triu(causal, k=1)
    np.testing.assert_allclose(upper_tri, 0.0, atol=1e-6)

    # 5. Istanziazione tramite factory da DEFAULT_CONFIG
    for model_name in ("transformer_direct", "transformer"):
        cfg = make_config(
            model=model_name,
            transformer_d_model=16,
            transformer_heads=2,
            transformer_d_ff=32,
            transformer_layers=1,
        )
        assert build_transformer_from_config(cfg, t_in=10).count_params() > 0


@pytest.mark.parametrize("model_name", ["transformer_direct", "transformer"])
def test_transformer_train_hybrid_and_save_reload(tmp_path, model_name):
    """Verifica il ciclo ibrido (TF -> MM -> SS) e il salvataggio/ricaricamento .keras per M3 e M4."""
    pytest.importorskip("tensorflow")
    import keras
    from src.models_transformer import CUSTOM_OBJECTS, build_transformer_from_config
    from src.training import predict, train_hybrid

    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (16, 10, N_FEATURES)).astype(np.float32)
    Y = rng.uniform(-1, 1, (16, 8, N_FEATURES)).astype(np.float32)
    cfg = make_config(
        model=model_name,
        t_in=10,
        t_out=8,
        transformer_d_model=16,
        transformer_heads=2,
        transformer_d_ff=32,
        transformer_layers=1,
        epochs_tf=1,
        epochs_mm=1,
        epochs_ss=1,
        batch_size=8,
    )

    set_seeds(42)
    model = build_transformer_from_config(cfg, t_in=10)
    hist = train_hybrid(model, X, Y, X[:4], Y[:4], cfg, verbose=0)
    assert [r["phase"] for r in hist] == ["tf", "mm", "ss"]
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["val_mse"]) for r in hist)

    pred_orig = predict(model, X[:3], batch_size=2)
    ckpt_file = tmp_path / f"{model_name}_test.keras"
    model.save(ckpt_file)
    assert ckpt_file.exists()

    loaded = keras.models.load_model(ckpt_file, custom_objects=CUSTOM_OBJECTS)
    pred_loaded = predict(loaded, X[:3], batch_size=2)
    np.testing.assert_allclose(pred_orig, pred_loaded, atol=1e-5)
