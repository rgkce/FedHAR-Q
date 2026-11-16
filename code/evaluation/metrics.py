"""
FedHAR-Q Sınıflandırma Değerlendirme Metrikleri Modülü (`code/evaluation/metrics.py`).

Tüm deney koşullarında (E1–E7, hem FP32 hem INT8 modellerde) aynı test protokolüyle:
- Accuracy (%)
- Macro-F1 (%)
- Sınıf bazlı F1 skorları (6 aktivite sınıfı için ayrı ayrı)
- Confusion Matrix (6x6 karmaşıklık matrisi)
- Ortalama Cross-Entropy kaybı (loss)
hesaplar.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data.preprocessing import ACTIVITY_NAMES


def evaluate_model(
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Verilen modeli (FP32 veya INT8 kuantize PyTorch modeli) `dataloader` üzerinde
    değerlendirir ve tüm sınıflandırma metriklerini sözlük olarak döndürür.
    """
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            batch_size = yb.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    acc_pct = float(accuracy_score(y_true, y_pred) * 100.0)
    macro_f1_pct = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    per_class_f1_arr = f1_score(y_true, y_pred, average=None, labels=list(range(6)), zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))

    per_class_f1: Dict[str, float] = {
        ACTIVITY_NAMES[i]: round(float(per_class_f1_arr[i]), 2) for i in range(6)
    }

    return {
        "loss": round(total_loss / max(total_samples, 1), 4),
        "accuracy": round(acc_pct, 2),
        "macro_f1": round(macro_f1_pct, 2),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(total_samples),
    }
