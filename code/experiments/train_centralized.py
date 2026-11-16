r"""
Merkezi (Centralized) FP32 Baseline Eğitim Modülü — Deney Koşulu E1.

Bilimsel Protokol:
1. 17 eğitim katılımcısının verisi (`train_loader`) üzerinde `FedHAR1DCNN` modeli eğitilir.
2. Her epoch sonunda model **yalnızca 4 doğrulama (validation) katılımcısı** üzerinde değerlendirilir.
3. En yüksek Doğrulama Macro-F1 skorunu veren model ağırlıkları `result/models/E1_seed{seed}.pt`
   olarak kaydedilir (Test kümesi model seçiminde kesinlikle kullanılmaz!).
4. Eğitim tamamlandığında en iyi doğrulama modeli yüklenir ve **9 test katılımcısı** (`test_loader`)
   üzerinde Accuracy, Macro-F1, Confusion Matrix, Model Boyutu (KB) ve Gecikme (ms) ölçülür.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from configs.config_loader import get_path, load_config, set_seed
from data.dataset import get_centralized_dataloaders
from data.preprocessing import SplitData, prepare_subject_wise_data
from evaluation.benchmark import run_full_benchmark
from evaluation.metrics import evaluate_model
from models.cnn1d import build_model_from_config


def train_centralized_model(
    seed: int = 42,
    epochs: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    batch_size: int | None = None,
    split_data: SplitData | None = None,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    E1 (Merkezi FP32, DP Yok) deney koşulunu verilen `seed` ile çalıştırır ve
    tüm doğrulama/test/sistem metriklerini döndürür.
    """
    cfg = load_config()
    set_seed(seed)

    device = torch.device(cfg["project"].get("device", "cpu"))
    epochs = epochs if epochs is not None else int(cfg["centralized"]["epochs"])
    lr = lr if lr is not None else float(cfg["centralized"]["lr"])
    weight_decay = (
        weight_decay if weight_decay is not None else float(cfg["centralized"]["weight_decay"])
    )
    batch_size = batch_size if batch_size is not None else int(cfg["data"]["batch_size"])
    eval_batch_size = int(cfg["data"]["eval_batch_size"])

    if split_data is None:
        split_data = prepare_subject_wise_data(
            raw_dir=cfg["data"]["raw_dir"],
            val_subjects_count=int(cfg["data"]["val_subjects_count"]),
            split_seed=42,
            normalize=True,
        )

    train_loader, val_loader, test_loader, _ = get_centralized_dataloaders(
        split_data=split_data,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )

    model = build_model_from_config(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * yb.size(0)
            total_samples += yb.size(0)

        train_loss = running_loss / max(total_samples, 1)
        val_metrics = evaluate_model(model, val_loader, device=device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(
                f"[E1 | Seed {seed}] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.2f}%"
            )

    # En iyi doğrulama (validation) modelini geri yükle
    model.load_state_dict(best_state_dict)

    model_save_path: Path | None = None
    if save_artifacts:
        model_save_path = get_path(f"result/models/E1_seed{seed}.pt")

    # Dokunulmamış 9 kişilik Test kümesi üzerinde nihai değerlendirme ve sistem ölçümü
    test_benchmark = run_full_benchmark(
        model=model,
        test_loader=test_loader,
        save_path=model_save_path,
        warmup_runs=int(cfg["benchmark"]["warmup_runs"]),
        timed_runs=int(cfg["benchmark"]["timed_runs"]),
        fl_rounds=0,
        fl_clients_per_round=0,
        device=device,
    )

    result_payload: Dict[str, Any] = {
        "experiment_id": "E1",
        "training_mode": "Centralized",
        "privacy": "None",
        "epsilon_target": None,
        "epsilon_spent": None,
        "delta": None,
        "representation": "FP32",
        "seed": seed,
        "best_val_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        **test_benchmark,
        "history": history,
    }

    if save_artifacts:
        log_path = get_path(f"result/logs/E1_seed{seed}.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

    if verbose:
        print(
            f"--> [E1 TAMAMLANDI | Seed {seed}] En İyi Val Epoch: {best_epoch} | "
            f"TEST Accuracy: {test_benchmark['accuracy']:.2f}% | "
            f"TEST Macro-F1: {test_benchmark['macro_f1']:.2f}% | "
            f"Boyut: {test_benchmark['model_size_kb']:.2f} KB | "
            f"Gecikme: {test_benchmark['latency_mean_ms']:.4f} ms"
        )

    return result_payload


if __name__ == "__main__":
    train_centralized_model(seed=42)
