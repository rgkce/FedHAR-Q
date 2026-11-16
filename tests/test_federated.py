"""
Adım 5 Doğrulama Testi: Merkezi (E1) ve Federatif Flower FedAvg (E2) Baseline Testleri.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import UCIHARDataset
from federated.client import FedHARFlowerClient, get_model_parameters, set_model_parameters
from models.cnn1d import FedHAR1DCNN


def test_parameter_roundtrip():
    """Model ağırlıklarının NumPy dizilerine dönüştürülüp kayıpsız geri yüklendiğini doğrular."""
    m1 = FedHAR1DCNN()
    m2 = FedHAR1DCNN()

    params = get_model_parameters(m1)
    set_model_parameters(m2, params)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_flower_client_fit_and_eval():
    """Tek bir Flower istemcisinin yerel eğitim (fit) ve değerlendirme (evaluate) adımlarını doğrular."""
    X = np.random.randn(32, 9, 128).astype(np.float32)
    y = np.random.randint(0, 6, size=(32,), dtype=np.int64)
    loader = DataLoader(UCIHARDataset(X, y), batch_size=16, shuffle=True)

    model = FedHAR1DCNN()
    client = FedHARFlowerClient(
        client_id=0,
        subject_id=5,
        model=model,
        train_loader=loader,
        lr=0.002,
        local_epochs=1,
    )

    init_params = client.get_parameters({})
    updated_params, num_examples, metrics = client.fit(init_params, {"server_round": 1})

    assert num_examples == 32
    assert "train_loss" in metrics
    assert len(updated_params) == len(init_params)

    loss, eval_examples, eval_metrics = client.evaluate(updated_params, {})
    assert eval_examples == 32
    assert "accuracy" in eval_metrics
    assert "macro_f1" in eval_metrics
