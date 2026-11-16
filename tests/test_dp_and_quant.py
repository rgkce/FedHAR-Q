"""
Adım 6 Doğrulama Testi: Opacus DP-SGD Gizlilik Muhasebesi ve INT8 Statik Kuantizasyon Testleri.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import UCIHARDataset
from models.cnn1d import FedHAR1DCNN
from privacy.dp_engine import prepare_federated_dp_clients, train_client_with_dp
from quantization.quantize import quantize_fp32_to_int8
from evaluation.benchmark import measure_model_size_kb


def test_dp_client_training_and_rdp_accountant():
    """
    Opacus DP-SGD yerel eğitiminin çalıştığını ve RDPAccountant'ın hedef
    epsilon bütçesini doğru takip ettiğini doğrular.
    """
    X = np.random.randn(64, 9, 128).astype(np.float32)
    y = np.random.randint(0, 6, size=(64,), dtype=np.int64)
    loader = DataLoader(UCIHARDataset(X, y), batch_size=32, shuffle=True)

    client_dp_configs, dp_summary = prepare_federated_dp_clients(
        client_loaders={0: loader},
        num_rounds=2,
        local_epochs=1,
        target_epsilon=5.0,
        delta=1e-4,
        max_grad_norm=1.0,
    )

    assert dp_summary["accountant"] == "RDPAccountant"
    assert dp_summary["noise_multiplier"] > 0.0

    model = FedHAR1DCNN()
    # 2 tur simüle et
    for _ in range(2):
        params, n_samples, metrics = train_client_with_dp(
            model=model,
            train_loader=loader,
            local_epochs=1,
            lr=0.002,
            weight_decay=1e-4,
            dp_config=client_dp_configs[0],
        )

    assert n_samples == 64
    assert 4.0 <= metrics["epsilon_spent"] <= 6.0, (
        f"Hedef ε=5.0 için harcanan ε ~5.0 olmalı, gelen: {metrics['epsilon_spent']}"
    )


def test_ptq_int8_compression_and_inference():
    """
    FP32 modelin INT8'e kuantize edildiğinde boyutunun >2.5x küçüldüğünü
    ve çıkarım yapabildiğini doğrular.
    """
    X = np.random.randn(64, 9, 128).astype(np.float32)
    y = np.random.randint(0, 6, size=(64,), dtype=np.int64)
    loader = DataLoader(UCIHARDataset(X, y), batch_size=32)

    fp32_model = FedHAR1DCNN().eval()
    int8_model = quantize_fp32_to_int8(fp32_model, loader, num_calibration_batches=2)

    fp32_kb = measure_model_size_kb(fp32_model)
    int8_kb = measure_model_size_kb(int8_model)

    assert int8_kb < fp32_kb / 2.5, f"INT8 model ({int8_kb} KB) FP32'den ({fp32_kb} KB) >2.5x küçük olmalı!"

    with torch.no_grad():
        out = int8_model(torch.randn(8, 9, 128))
    assert out.shape == (8, 6)
