"""
Adım 2 Doğrulama Testi: UCI HAR Veri Hattı ve Sıfır Veri Sızıntısı (Zero Data Leakage) Testleri.
"""

import numpy as np
import pytest
from data.preprocessing import prepare_subject_wise_data
from data.dataset import get_centralized_dataloaders
from data.partition import create_subject_client_dataloaders, get_client_summary


@pytest.fixture(scope="module")
def split_data():
    """UCI HAR veri kümesini indirir ve kişi bazlı (subject-wise) olarak hazırlar."""
    return prepare_subject_wise_data(val_subjects_count=4, split_seed=42, normalize=True)


def test_zero_subject_leakage(split_data):
    """
    Eğitim (17 kişi), Doğrulama (4 kişi) ve Test (9 kişi) kümelerindeki
    katılımcıların kesişiminin boş küme (sıfır veri sızıntısı) olduğunu doğrular.
    """
    train_s = set(split_data.train_subject_ids)
    val_s = set(split_data.val_subject_ids)
    test_s = set(split_data.test_subject_ids)

    assert len(train_s) == 17
    assert len(val_s) == 4
    assert len(test_s) == 9
    assert len(train_s | val_s | test_s) == 30, "Toplam 30 benzersiz katılımcı olmalı!"

    # Kesişim kontrolleri
    assert train_s.isdisjoint(val_s), "Train ve Validation arasında kişi sızıntısı var!"
    assert train_s.isdisjoint(test_s), "Train ve Test arasında kişi sızıntısı var!"
    assert val_s.isdisjoint(test_s), "Validation ve Test arasında kişi sızıntısı var!"


def test_shapes_and_labels(split_data):
    """Sinyal boyutlarının (N, 9, 128) ve etiketlerin 0..5 aralığında olduğunu doğrular."""
    # UCI HAR resmî train=7352, test=2947 örnektir
    assert split_data.X_train.shape[0] + split_data.X_val.shape[0] == 7352
    assert split_data.X_test.shape == (2947, 9, 128)
    assert split_data.X_train.shape[1:] == (9, 128)
    assert split_data.X_val.shape[1:] == (9, 128)

    for y in (split_data.y_train, split_data.y_val, split_data.y_test):
        assert set(np.unique(y).tolist()) == {0, 1, 2, 3, 4, 5}


def test_train_only_normalization(split_data):
    """
    Normalizasyonun yalnızca eğitim verisinden hesaplandığını doğrular:
    X_train ortalaması ~0.0 ve standart sapması ~1.0 olmalıdır.
    """
    train_mean = split_data.X_train.mean(axis=(0, 2))
    train_std = split_data.X_train.std(axis=(0, 2))
    assert np.allclose(train_mean, 0.0, atol=1e-5)
    assert np.allclose(train_std, 1.0, atol=1e-4)


def test_dataloaders_and_federated_partitions(split_data):
    """Merkezi ve Federatif (17 istemcili) DataLoader yapılarını doğrular."""
    train_loader, val_loader, test_loader, _ = get_centralized_dataloaders(
        split_data=split_data, batch_size=64
    )
    xb, yb = next(iter(train_loader))
    assert xb.shape == (64, 9, 128)
    assert yb.shape == (64,)

    client_loaders = create_subject_client_dataloaders(split_data, batch_size=32)
    summary = get_client_summary(split_data)
    assert len(client_loaders) == 17
    assert len(summary) == 17

    total_client_samples = sum(len(loader.dataset) for loader in client_loaders.values())
    assert total_client_samples == split_data.X_train.shape[0]
