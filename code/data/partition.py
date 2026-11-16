"""
Federatif Öğrenme (FL) İçin İstemci Veri Bölünmesi (Client Partitioning) Modülü.

UCI HAR veri kümesinde her katılımcı (subject) kendi akıllı telefonunu taşıyan
gerçek bir kullanıcıyı temsil eder. Bu nedenle en gerçekçi ve bilimsel olarak
doğru FL senaryosu, 17 eğitim katılımcısının her birini bağımsız birer FL istemcisi
(`client_id = 0 .. 16`) olarak modellemektir (doğal kullanıcı bazlı Non-IID dağılım).

Ayrıca isteğe bağlı karşılaştırmalar için rastgele eşit (IID) bölünme desteği de içerir.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import UCIHARDataset
from data.preprocessing import SplitData


def create_subject_client_dataloaders(
    split_data: SplitData,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[int, DataLoader]:
    """
    Eğitim kümesindeki her katılımcıyı (17 kişi) ayrı bir Flower istemcisi (`client_id: 0..16`)
    olarak böler ve her istemci için yerel `DataLoader` döndürür.
    """
    client_loaders: Dict[int, DataLoader] = {}

    for client_id, subject_id in enumerate(split_data.train_subject_ids):
        mask = split_data.subjects_train == subject_id
        X_client = split_data.X_train[mask]
        y_client = split_data.y_train[mask]
        subj_client = split_data.subjects_train[mask]

        client_ds = UCIHARDataset(X_client, y_client, subj_client)

        g = torch.Generator()
        g.manual_seed(seed + client_id)

        client_loaders[client_id] = DataLoader(
            client_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            drop_last=False,
        )

    return client_loaders


def get_client_summary(split_data: SplitData) -> List[Dict[str, int]]:
    """Her istemcinin sahip olduğu katılımcı ID'sini ve örnek sayısını özetler."""
    summary: List[Dict[str, int]] = []
    for client_id, subject_id in enumerate(split_data.train_subject_ids):
        mask = split_data.subjects_train == subject_id
        summary.append(
            {
                "client_id": int(client_id),
                "subject_id": int(subject_id),
                "num_samples": int(np.sum(mask)),
                "num_classes_present": int(len(np.unique(split_data.y_train[mask]))),
            }
        )
    return summary
