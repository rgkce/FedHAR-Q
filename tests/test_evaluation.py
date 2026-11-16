"""
Adım 4 Doğrulama Testi: Sınıflandırma Metrikleri ve Sistem/Kaynak Ölçümü Testleri.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import UCIHARDataset
from models.cnn1d import FedHAR1DCNN
from evaluation.metrics import evaluate_model
from evaluation.benchmark import (
    measure_model_size_kb,
    measure_inference_latency_ms,
    measure_peak_memory_kb,
    compute_fl_communication_cost_mb,
    run_full_benchmark,
)


def test_evaluate_model_metrics():
    """Accuracy, Macro-F1, sınıf bazlı F1 ve 6x6 Confusion Matrix hesabını doğrular."""
    X = np.random.randn(60, 9, 128).astype(np.float32)
    y = np.repeat(np.arange(6), 10).astype(np.int64)
    loader = DataLoader(UCIHARDataset(X, y), batch_size=20, shuffle=False)

    model = FedHAR1DCNN()
    res = evaluate_model(model, loader)

    assert 0.0 <= res["accuracy"] <= 100.0
    assert 0.0 <= res["macro_f1"] <= 100.0
    assert len(res["per_class_f1"]) == 6
    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (6, 6)
    assert cm.sum() == 60


def test_system_benchmarks_and_fl_comm():
    """Model boyutu (KB), gecikme (ms), tepe bellek (KB) ve FL iletişim yükü (MB) hesabını doğrular."""
    model = FedHAR1DCNN()

    size_kb = measure_model_size_kb(model)
    assert 140.0 <= size_kb <= 200.0, f"FP32 model boyutu ~165 KB bekleniyordu, gelen: {size_kb}"

    lat = measure_inference_latency_ms(model, warmup_runs=10, timed_runs=50)
    assert lat["latency_mean_ms"] > 0.0
    assert lat["latency_p95_ms"] >= lat["latency_median_ms"]

    mem = measure_peak_memory_kb(model)
    assert mem["peak_inference_footprint_kb"] > size_kb

    # 30 tur, 17 istemci için FL iletişim yükü testi
    comm = compute_fl_communication_cost_mb(num_rounds=30, num_clients_per_round=17, model_or_size_kb=model)
    # 30 * 17 * 2 * 161.27 KB = 164,495.4 KB ≈ 160.64 MB
    assert 150.0 <= comm["total_comm_mb"] <= 170.0


def test_run_full_benchmark():
    """Tek fonksiyonla tüm sınıflandırma ve sistem metriklerinin birleştiğini doğrular."""
    X = np.random.randn(30, 9, 128).astype(np.float32)
    y = np.tile(np.arange(6), 5).astype(np.int64)
    loader = DataLoader(UCIHARDataset(X, y), batch_size=15)

    model = FedHAR1DCNN()
    full = run_full_benchmark(
        model, loader, warmup_runs=5, timed_runs=20, fl_rounds=30, fl_clients_per_round=17
    )
    for key in ("accuracy", "macro_f1", "model_size_kb", "latency_mean_ms", "total_comm_mb"):
        assert key in full
