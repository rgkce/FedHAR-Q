r"""
Flower (`flwr`) Tabanlı Federatif Öğrenme İstemcisi (`code/federated/client.py`).

Bu modül:
1. PyTorch model ağırlıklarını (`state_dict`) Flower'ın beklediği NumPy dizisi listesine
   (`NDArrays`) dönüştüren ve geri yükleyen yardımcı fonksiyonları içerir.
2. Her katılımcıyı (subject) temsil eden `FedHARFlowerClient(fl.client.NumPyClient)`
   sınıfını tanımlar.
3. Hem gizliliksiz (E2) hem de Opacus Diferansiyel Gizlilik (DP-SGD) etkinleştirilmiş
   (E4, E6) yerel eğitimi destekler.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_model


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """PyTorch modelinin ağırlıklarını NumPy dizisi listesi (`NDArrays`) olarak döndürür."""
    return [val.detach().cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Flower sunucusundan gelen NumPy ağırlık listesini PyTorch modeline yükler."""
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in zip(keys, parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


class FedHARFlowerClient(fl.client.NumPyClient):
    """
    UCI HAR üzerindeki tek bir katılımcıyı (istemciyi) temsil eden Flower NumPyClient sınıfı.
    """

    def __init__(
        self,
        client_id: int,
        subject_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        local_epochs: int = 2,
        device: str | torch.device = "cpu",
        dp_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.subject_id = subject_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        self.dp_config = dp_config

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Küresel model ağırlıklarını alır, istemcinin yerel verisi üzerinde
        `local_epochs` boyunca eğitir ve güncellenmiş ağırlıkları sunucuya döndürür.
        """
        set_model_parameters(self.model, parameters)
        self.model.train()

        # Eğer DP-SGD aktifse (Adım 6 - E4/E6), ilgili DP eğitim motoruna delege et
        if self.dp_config is not None and self.dp_config.get("enabled", False):
            from privacy.dp_engine import train_client_with_dp

            updated_params, num_examples, dp_metrics = train_client_with_dp(
                model=self.model,
                train_loader=self.train_loader,
                local_epochs=self.local_epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                dp_config=self.dp_config,
                device=self.device,
            )
            return updated_params, num_examples, dp_metrics

        # Standart (DP'siz - E2) yerel eğitim döngüsü
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        num_samples = len(self.train_loader.dataset)

        return (
            get_model_parameters(self.model),
            num_samples,
            {
                "client_id": self.client_id,
                "subject_id": self.subject_id,
                "train_loss": round(avg_loss, 4),
            },
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """İstemci verisi üzerinde yerel değerlendirme yapar."""
        set_model_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.train_loader, device=self.device)
        return (
            float(metrics["loss"]),
            len(self.train_loader.dataset),
            {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
        )
