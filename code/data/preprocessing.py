r"""
UCI HAR Ön İşleme ve Kişi Bazlı (Subject-Wise) Veri Ayrımı Modülü.

Bu modül görev dokümanındaki en kritik bilimsel kuralı uygular:
1. 9 kanallı ham eylemsizlik sinyallerini `(N, 9, 128)` boyutunda yükler.
2. Resmî 21 eğitim kişisini (subject) kişi bazında `17 Eğitim` ve `4 Doğrulama (Validation)`
   olarak ayırır; resmî `9 Test` kişisine hiçbir aşamada dokunulmaz.
3. Eğitim, doğrulama ve test kümelerindeki kişilerin kesişiminin boş küme (sıfır veri sızıntısı)
   olduğunu garanti altına alır (`assert`).
4. Kanal bazlı z-score normalizasyon istatistiklerini ($\mu, \sigma$) YALNIZCA
   17 eğitim kişisinin verisinden hesaplar ve tüm kümelere uygular.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from configs.config_loader import get_path
from data.download_uci_har import SIGNAL_PREFIXES, download_and_extract_uci_har

ACTIVITY_NAMES: Dict[int, str] = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


@dataclass
class SplitData:
    """Kişi bazlı ayrılmış ve normalize edilmiş UCI HAR veri yapısı."""

    X_train: np.ndarray        # (N_train, 9, 128) float32
    y_train: np.ndarray        # (N_train,) int64 [0..5]
    subjects_train: np.ndarray # (N_train,) int64

    X_val: np.ndarray          # (N_val, 9, 128) float32
    y_val: np.ndarray          # (N_val,) int64 [0..5]
    subjects_val: np.ndarray   # (N_val,) int64

    X_test: np.ndarray         # (N_test, 9, 128) float32
    y_test: np.ndarray         # (N_test,) int64 [0..5]
    subjects_test: np.ndarray  # (N_test,) int64

    channel_mean: np.ndarray   # (1, 9, 1) float32 - Sadece train'den hesaplanan ortalama
    channel_std: np.ndarray    # (1, 9, 1) float32 - Sadece train'den hesaplanan std

    train_subject_ids: List[int]
    val_subject_ids: List[int]
    test_subject_ids: List[int]


def _load_raw_split(dataset_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Belirtilen alt küme ('train' veya 'test') için:
    - X: (N, 9, 128) float32
    - y: (N,) int64 (0..5 aralığında)
    - subjects: (N,) int64 (1..30 aralığında katılımcı ID'si)
    döndürür.
    """
    signals = []
    sig_dir = dataset_dir / split / "Inertial Signals"
    for prefix in SIGNAL_PREFIXES:
        fpath = sig_dir / f"{prefix}_{split}.txt"
        arr = np.loadtxt(fpath, dtype=np.float32)  # (N, 128)
        signals.append(arr)

    # (9, N, 128) -> (N, 9, 128)
    X = np.stack(signals, axis=1).astype(np.float32)

    # Etiketler 1..6 -> 0..5
    y_path = dataset_dir / split / f"y_{split}.txt"
    y = np.loadtxt(y_path, dtype=np.int64).reshape(-1) - 1

    # Katılımcı (Subject) kimlikleri 1..30
    subj_path = dataset_dir / split / f"subject_{split}.txt"
    subjects = np.loadtxt(subj_path, dtype=np.int64).reshape(-1)

    assert X.shape[0] == y.shape[0] == subjects.shape[0], "Örnek sayıları uyuşmuyor!"
    assert X.shape[1:] == (9, 128), f"Beklenen boyut (N, 9, 128), gelen: {X.shape}"
    return X, y, subjects


def prepare_subject_wise_data(
    raw_dir: str | Path = "dataset/raw",
    val_subjects_count: int = 4,
    split_seed: int = 42,
    normalize: bool = True,
) -> SplitData:
    """
    UCI HAR veri kümesini kişi bazlı (subject-wise) olarak Train, Validation ve Test
    kümelerine ayırır ve veri sızıntısı olmadan kanal bazlı normalizasyon uygular.
    """
    dataset_dir = download_and_extract_uci_har(raw_dir)

    X_train_full, y_train_full, subj_train_full = _load_raw_split(dataset_dir, "train")
    X_test_raw, y_test, subj_test = _load_raw_split(dataset_dir, "test")

    # Resmî eğitim katılımcıları (21 kişi) ve test katılımcıları (9 kişi)
    unique_train_subjects = sorted(np.unique(subj_train_full).tolist())
    test_subject_ids = sorted(np.unique(subj_test).tolist())

    # Kişi bazlı doğrulama (validation) ayrımı:
    # Tekrarlanabilirlik için sabit tohumla 21 eğitim kişisinden 4 kişi seçilir.
    rng = np.random.RandomState(split_seed)
    shuffled_subjects = unique_train_subjects.copy()
    rng.shuffle(shuffled_subjects)

    val_subject_ids = sorted(shuffled_subjects[:val_subjects_count])
    train_subject_ids = sorted(shuffled_subjects[val_subjects_count:])

    # --- KRİTİK VERİ SIZINTISI KONTROLLERİ ---
    assert set(train_subject_ids).isdisjoint(set(val_subject_ids)), (
        "HATA: Eğitim ve Doğrulama kümelerinde ortak kişi (data leakage) var!"
    )
    assert set(train_subject_ids).isdisjoint(set(test_subject_ids)), (
        "HATA: Eğitim ve Test kümelerinde ortak kişi (data leakage) var!"
    )
    assert set(val_subject_ids).isdisjoint(set(test_subject_ids)), (
        "HATA: Doğrulama ve Test kümelerinde ortak kişi (data leakage) var!"
    )

    train_mask = np.isin(subj_train_full, train_subject_ids)
    val_mask = np.isin(subj_train_full, val_subject_ids)

    X_train_raw = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    subjects_train = subj_train_full[train_mask]

    X_val_raw = X_train_full[val_mask]
    y_val = y_train_full[val_mask]
    subjects_val = subj_train_full[val_mask]

    # --- KANAL BAZLI NORMALİZASYON (SADECE EĞİTİM VERİSİNDEN) ---
    # Her kanal (9 kanal) için tüm eğitim pencereleri ve 128 zaman adımı üzerinden mean ve std
    channel_mean = X_train_raw.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    channel_std = (X_train_raw.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)

    if normalize:
        X_train = ((X_train_raw - channel_mean) / channel_std).astype(np.float32)
        X_val = ((X_val_raw - channel_mean) / channel_std).astype(np.float32)
        X_test = ((X_test_raw - channel_mean) / channel_std).astype(np.float32)
    else:
        X_train = X_train_raw
        X_val = X_val_raw
        X_test = X_test_raw

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        subjects_train=subjects_train,
        X_val=X_val,
        y_val=y_val,
        subjects_val=subjects_val,
        X_test=X_test,
        y_test=y_test,
        subjects_test=subj_test,
        channel_mean=channel_mean,
        channel_std=channel_std,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=test_subject_ids,
    )
