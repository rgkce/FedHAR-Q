"""Adım 1 Doğrulama Testi: Konfigürasyon ve Tekrarlanabilirlik (Seed) Testleri."""

import torch
from configs.config_loader import load_config, set_seed, get_path


def test_load_default_config():
    cfg = load_config()
    assert cfg["project"]["name"] == "FedHAR-Q"
    assert len(cfg["project"]["seeds"]) >= 3
    assert cfg["data"]["in_channels"] == 9
    assert cfg["data"]["seq_len"] == 128
    assert cfg["data"]["num_classes"] == 6
    assert cfg["privacy"]["epsilon_1"] < cfg["privacy"]["epsilon_2"]


def test_seed_reproducibility():
    set_seed(42)
    t1 = torch.randn(5, 5)
    set_seed(42)
    t2 = torch.randn(5, 5)
    assert torch.allclose(t1, t2), "Aynı tohum (seed) aynı tensörü üretmeli!"


def test_project_paths_exist():
    assert get_path("dataset/raw").exists()
    assert get_path("result/models").exists()
    assert get_path("result/tables").exists()
