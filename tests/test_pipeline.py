"""Test di validazione della pipeline e dei moduli di base:

1. Verifica determinismo: due generazioni con lo stesso seed producono tensori identici.
2. Verifica windowing: le shape di output X e Y su dati sintetici corrispondono a quanto atteso.
3. Verifica leakage: nessun overlap tra split train/val/test.
4. Verifica metriche e statistiche: calcolo MSE, MAE, disaggregazione e CI 95% con t di Student.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Aggiunge src al path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import N_FEATURES, N_POINTS, N_QUBITS, make_config
from src.data import check_quantum_probabilities, corr_pairs, feature_names, kfold_splits, make_split
from src.metrics import evaluate, per_horizon_mse, per_variable_mse
from src.results_log import append_result, load_results
from src.seeding import set_seeds
from src.stats import format_mean_ci, summarize
from src.windowing import check_no_leakage, decoder_inputs, make_sr_windows, make_windows


def test_seeding_determinism():
    set_seeds(42)
    a = np.random.randn(10, 10)
    set_seeds(42)
    b = np.random.randn(10, 10)
    np.testing.assert_array_equal(a, b, err_msg="I tensori generati con lo stesso seed devono essere identici")


def test_feature_names_count():
    names = feature_names(N_QUBITS)
    assert len(names) == N_FEATURES, f"Attesi {N_FEATURES} nomi, trovati {len(names)}"
    pairs = corr_pairs(N_QUBITS)
    assert len(pairs) == 45, f"Attese 45 correlazioni, trovate {len(pairs)}"


def test_windows_shapes():
    # Dati sintetici: 5 traiettorie da 100 punti con 55 feature
    n_traj, n_points, n_feat = 5, 100, 55
    dummy_data = np.random.randn(n_traj, n_points, n_feat).astype(np.float32)
    idx = np.array([0, 1, 2])

    t_in, t_out, stride = 10, 40, 10
    X, Y, meta = make_windows(dummy_data, idx, t_in=t_in, t_out=t_out, stride=stride)

    # span = 50. In 100 punti con stride 10: partenze 0, 10, 20, 30, 40, 50 -> 6 finestre per traiettoria
    expected_win_per_traj = (100 - (t_in + t_out)) // stride + 1
    expected_total_win = len(idx) * expected_win_per_traj

    assert X.shape == (expected_total_win, t_in, n_feat), f"Shape X inattesa: {X.shape}"
    assert Y.shape == (expected_total_win, t_out, n_feat), f"Shape Y inattesa: {Y.shape}"
    assert meta.shape == (expected_total_win, 2)

    # Verifica teacher forcing dec_in
    dec_in = decoder_inputs(X, Y)
    assert dec_in.shape == Y.shape
    np.testing.assert_array_equal(dec_in[:, 0], X[:, -1])
    np.testing.assert_array_equal(dec_in[:, 1:], Y[:, :-1])


def test_sr_windows():
    dummy_data = np.random.randn(2, 100, 55).astype(np.float32)
    idx = np.array([0])
    t_in, t_out, stride, factor = 20, 40, 10, 2

    X_sr, Y, meta = make_sr_windows(dummy_data, idx, t_in=t_in, t_out=t_out, stride=stride, factor=factor)
    assert X_sr.shape[1] == t_in
    assert Y.shape[1] == t_out


def test_split_and_no_leakage():
    split = make_split(n_traj=400, n_train=280, n_val=60, n_test=60, seed=42)
    assert len(split["train"]) == 280
    assert len(split["val"]) == 60
    assert len(split["test"]) == 60

    # Verifica che non ci siano intersezioni tra indici
    assert len(set(split["train"]) & set(split["val"])) == 0
    assert len(set(split["train"]) & set(split["test"])) == 0
    assert len(set(split["val"]) & set(split["test"])) == 0

    # Verifica check_no_leakage su metadati finti
    meta_tr = np.array([[tid, 0] for tid in split["train"]])
    meta_va = np.array([[tid, 0] for tid in split["val"]])
    meta_te = np.array([[tid, 0] for tid in split["test"]])
    check_no_leakage({"train": meta_tr, "val": meta_va, "test": meta_te})


def test_metrics_and_stats():
    y_true = np.ones((10, 40, 55), dtype=np.float32)
    y_pred = np.zeros((10, 40, 55), dtype=np.float32)

    res = evaluate(y_true, y_pred)
    assert np.isclose(res["mse"], 1.0)
    assert np.isclose(res["mae"], 1.0)
    assert np.isclose(res["mse_mag"], 1.0)
    assert np.isclose(res["mse_corr"], 1.0)

    pvar = per_variable_mse(y_true, y_pred)
    assert pvar.shape == (55,)
    assert np.allclose(pvar, 1.0)

    phor = per_horizon_mse(y_true, y_pred)
    assert phor.shape == (40,)
    assert np.allclose(phor, 1.0)

    # Test CI con t di Student su n=3
    stats = summarize([1.0, 1.1, 0.9])
    assert stats["n"] == 3
    assert np.isclose(stats["mean"], 1.0)
    assert stats["ci_half"] > 0
    # Con n=3 t critico è ~4.303
    assert np.isclose(stats["ci_half"], 4.30265 * stats["std"] / np.sqrt(3), atol=1e-3)


def test_kfold_splits():
    indices = np.arange(20)
    folds = kfold_splits(indices, k=5, seed=42)
    assert len(folds) == 5

    seen_val = []
    for train_idx, val_idx in folds:
        assert len(val_idx) == 4
        assert len(train_idx) == 16
        # Nessuna sovrapposizione tra train e val del singolo fold
        assert len(set(train_idx) & set(val_idx)) == 0
        seen_val.extend(val_idx.tolist())

    # La validazione copre tutti i 20 elementi esattamente una volta
    assert sorted(seen_val) == list(range(20))


def test_results_logging(tmp_path):
    log_file = tmp_path / "results_log_test.csv"

    # Primo record
    rec1 = {"model": "rnn", "val_mse": 0.001, "seeds": [42, 43]}
    append_result(rec1, path=log_file)
    assert log_file.exists()

    # Secondo record con colonna aggiuntiva
    rec2 = {"model": "rnn", "val_mse": 0.0009, "new_metric": 0.99}
    append_result(rec2, path=log_file)

    # Rilettura
    df = load_results(tmp_path)
    assert len(df) == 2
    assert "val_mse" in df.columns
    assert "new_metric" in df.columns
    assert df.iloc[0]["model"] == "rnn"


def test_quantum_probabilities():
    # 1. Stato fisico con magnetizzazioni e correlazioni nulle:
    # P(s_i, s_j) = 1/4 = 0.25 >= 0 per tutte le 45 coppie
    zero_state = np.zeros((1, N_FEATURES), dtype=np.float32)
    res_zero = check_quantum_probabilities(zero_state)
    assert res_zero["is_physical"] is True
    assert res_zero["violations_count"] == 0
    assert np.isclose(res_zero["min_probability"], 0.25)

    # 2. Configurazione non fisica: m1 = 1, m2 = 1, c12 = -1
    # P(-1, -1) = (1 - 1 - 1 - 1) / 4 = -0.5 < 0 (violazione quantistica)
    unphysical_state = np.zeros((1, N_FEATURES), dtype=np.float32)
    unphysical_state[0, 0] = 1.0   # m1
    unphysical_state[0, 1] = 1.0   # m2
    unphysical_state[0, 10] = -1.0  # c1_2
    res_unphys = check_quantum_probabilities(unphysical_state)
    assert res_unphys["is_physical"] is False
    assert res_unphys["violations_count"] > 0
    assert res_unphys["min_probability"] < 0
