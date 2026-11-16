"""
PyTorch Dataset Sınıfı ve Merkezi DataLoader Üreticisi.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import SplitData, prepare_subject_wise_data


class UCIHARDataset(Dataset):
    """
    (N, 9, 128) boyutlu eylemsizlik sinyallerini ve (N,) boyutlu sınıf
    etiketlerini PyTorch tensörleri olarak sunan Dataset sınıfı.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray | None = None) -> None:
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.subjects = torch.from_numpy(subjects).long() if subjects is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_centralized_dataloaders(
    split_data: SplitData | None = None,
    batch_size: int = 64,
    eval_batch_size: int = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SplitData]:
    """
    Merkezi eğitim (E1) ve küresel doğrulama/test için `train_loader`, `val_loader`
    ve `test_loader` nesnelerini döndürür.
    """
    if split_data is None:
        split_data = prepare_subject_wise_data(split_seed=42)

    train_ds = UCIHARDataset(split_data.X_train, split_data.y_train, split_data.subjects_train)
    val_ds = UCIHARDataset(split_data.X_val, split_data.y_val, split_data.subjects_val)
    test_ds = UCIHARDataset(split_data.X_test, split_data.y_test, split_data.subjects_test)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, split_data
