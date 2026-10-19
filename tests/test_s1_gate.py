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
from src.data import corr_pairs, feature_names, make_split
from src.metrics import evaluate, per_horizon_mse, per_variable_mse
from src.seeding import set_seeds
from src.stats import summarize
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
