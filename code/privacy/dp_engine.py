r"""
Opacus Tabanlı Diferansiyel Gizlilik (DP-SGD) ve Rényi DP Muhasebesi Modülü (`code/privacy/dp_engine.py`).

Teorik Kapsam ve Belgeleme (Görev Dokümanı Zorunlu İsteri):
1. **Gizlilik Birimi (Privacy Unit):** Örnek düzeyi diferansiyel gizlilik (*Sample-level DP-SGD*,
   Abadi et al., 2016 [9]). Her istemcinin yerel eğitimindeki bireysel sensör pencerelerinin
   ($x_i$) gradyan katkısı $L_2$ norm eşiği $C$ (`max_grad_norm`) ile kırpılır ve üzerine
   $\mathcal{N}(0, \sigma^2 C^2 I)$ Gauss gürültüsü eklenir. (İstemci düzeyi / user-level DP
   değildir; bu ayrım bildiride McMahan et al., 2018 [11] referansıyla açıkça tartışılır.)
2. **Gizlilik Muhasebesi (Privacy Accountant):** Rényi Diferansiyel Gizlilik (*RDPAccountant*,
   Mironov, 2017 [10]; Yousefpour et al., 2021 [12]). Her istemci için 30 federatif tur boyunca
   kalıcı (persistent) bir `RDPAccountant` nesnesi tutulur ve her yerel adım (`step`) sonunda
   kümülatif $(\varepsilon, \delta)$ bütçesi hesaplanır.
3. **Kesin Adım Bazlı Gürültü Kalibrasyonu:** Hedef $\varepsilon \in \{\varepsilon_1=3.0, \varepsilon_2=8.0\}$
   ve $\delta = 10^{-4} < 1/N_{\text{train}}$ için Opacus `get_noise_multiplier` fonksiyonu ile
   istemcinin 30 tur boyunca atacağı tam adım sayısı ($T_{\text{steps}} = R \times E_{\text{local}} \times |\mathcal{B}_k|$)
   üzerinden $\sigma_k$ (`noise_multiplier`) kalibre edilir; böylece 30. tur sonunda harcanan bütçe
   tam olarak $\varepsilon_{\text{hedef}}$ değerine eşitlenir.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader

from federated.client import get_model_parameters


def prepare_federated_dp_clients(
    client_loaders: Dict[int, DataLoader],
    num_rounds: int,
    local_epochs: int,
    target_epsilon: float,
    delta: float = 1e-4,
    max_grad_norm: float = 2.0,
    dp_lr: float = 0.005,
    seed: int = 42,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Her FL istemcisi için toplam eğitim adımı (`num_rounds * local_epochs * steps_per_epoch`)
    sonunda hedef `target_epsilon` bütçesini tam olarak sağlayacak gürültü çarpanını
    (`noise_multiplier`) kalibre eder ve istemci başına kalıcı bir `RDPAccountant` oluşturur.
    """
    client_dp_configs: Dict[int, Dict[str, Any]] = {}
    sigmas: List[float] = []

    for cid, loader in client_loaders.items():
        steps_per_epoch = max(len(loader), 1)
        total_steps = num_rounds * local_epochs * steps_per_epoch
        sample_rate = 1.0 / float(steps_per_epoch)

        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant="rdp",
        )
        sigmas.append(float(sigma))

        accountant = RDPAccountant()
        client_dp_configs[cid] = {
            "enabled": True,
            "client_id": cid,
            "target_epsilon": float(target_epsilon),
            "delta": float(delta),
            "max_grad_norm": float(max_grad_norm),
            "dp_lr": float(dp_lr),
            "noise_multiplier": float(sigma),
            "sample_rate": float(sample_rate),
            "accountant": accountant,
            "seed": int(seed + cid * 100),
        }

    avg_sigma = float(np.mean(sigmas))
    dp_summary = {
        "privacy": f"DP-SGD (eps={target_epsilon})",
        "privacy_unit": "Sample-level (per-window within client)",
        "accountant": "RDPAccountant",
        "epsilon_target": float(target_epsilon),
        "epsilon_spent": None,
        "delta": float(delta),
        "noise_multiplier": round(avg_sigma, 4),
        "max_grad_norm": float(max_grad_norm),
    }

    return client_dp_configs, dp_summary


def train_client_with_dp(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    dp_config: Dict[str, Any],
    device: str | torch.device = "cpu",
) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
    """
    Opacus `GradSampleModule` ve `DPOptimizer` kullanarak tek bir istemcinin
    yerel eğitimini örnek-bazlı gradyan kırpma (`max_grad_norm`) ve Gauss gürültüsü
    (`noise_multiplier`) ile gerçekleştirir; kalıcı `RDPAccountant` üzerinde
    harcanan $\varepsilon$ bütçesini günceller.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

    model.to(device)
    model.train()

    max_grad_norm = float(dp_config["max_grad_norm"])
    noise_multiplier = float(dp_config["noise_multiplier"])
    sample_rate = float(dp_config["sample_rate"])
    delta = float(dp_config["delta"])
    effective_lr = float(dp_config.get("dp_lr", lr))
    accountant: RDPAccountant = dp_config["accountant"]

    dp_model = GradSampleModule(model)
    criterion = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.Adam(
        dp_model.parameters(), lr=effective_lr, weight_decay=weight_decay
    )

    total_loss = 0.0
    total_batches = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(local_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                actual_bs = max(int(yb.size(0)), 1)

                dp_optimizer = DPOptimizer(
                    optimizer=base_optimizer,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                    expected_batch_size=actual_bs,
                )

                dp_optimizer.zero_grad()
                logits = dp_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                dp_optimizer.step()

                accountant.step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                )

                total_loss += float(loss.item())
                total_batches += 1

    dp_model.remove_hooks()

    epsilon_spent = float(accountant.get_epsilon(delta=delta))
    avg_loss = total_loss / max(total_batches, 1)
    num_samples = len(train_loader.dataset)

    return (
        get_model_parameters(model),
        num_samples,
        {
            "client_id": dp_config.get("client_id", 0),
            "train_loss": round(avg_loss, 4),
            "epsilon_spent": round(epsilon_spent, 4),
            "noise_multiplier": round(noise_multiplier, 4),
        },
    )
