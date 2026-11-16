r"""
Flower (`flwr`) Tabanlı Federatif Sunucu ve FedAvg Simülasyon Motoru (`code/federated/server.py`).

Bu modül:
1. Flower'ın `FedAvg` ağırlıklı ortalama algoritmasını (`aggregate`) kullanarak
   17 eğitim katılımcısının yerel modellerini her turda (round) birleştirir:
   $$w^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{N} w_k^{(t+1)}$$
2. Her tur sonunda küresel modeli 4 doğrulama (validation) katılımcısı üzerinde değerlendirir
   ve en yüksek Doğrulama Macro-F1 skoruna sahip küresel modeli kaydeder.
3. Eğitim sonunda en iyi küresel modeli 9 test katılımcısı üzerinde değerlendirir ve
   toplam FL iletişim yükünü (MB) raporlar.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from flwr.server.strategy.aggregate import aggregate

import numpy as np
import torch

from configs.config_loader import get_path, load_config, set_seed
from data.dataset import get_centralized_dataloaders
from data.partition import create_subject_client_dataloaders
from data.preprocessing import SplitData, prepare_subject_wise_data
from evaluation.benchmark import run_full_benchmark
from evaluation.metrics import evaluate_model
from federated.client import FedHARFlowerClient, get_model_parameters, set_model_parameters
from models.cnn1d import build_model_from_config


def run_federated_training(
    experiment_id: str = "E2",
    seed: int = 42,
    num_rounds: int | None = None,
    local_epochs: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    batch_size: int | None = None,
    dp_config: Dict[str, Any] | None = None,
    split_data: SplitData | None = None,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Flower `FedHARFlowerClient` istemcileri ve `FedAvg` ağırlıklı agregasyonu ile
    federatif öğrenme simülasyonunu (E2, E4, E6) çalıştırır.
    """
    cfg = load_config()
    set_seed(seed)

    device = torch.device(cfg["project"].get("device", "cpu"))
    num_rounds = num_rounds if num_rounds is not None else int(cfg["federated"]["num_rounds"])
    local_epochs = (
        local_epochs if local_epochs is not None else int(cfg["federated"]["local_epochs"])
    )
    lr = lr if lr is not None else float(cfg["federated"]["lr"])
    weight_decay = (
        weight_decay if weight_decay is not None else float(cfg["federated"]["weight_decay"])
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

    _, val_loader, test_loader, _ = get_centralized_dataloaders(
        split_data=split_data,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )

    client_loaders = create_subject_client_dataloaders(
        split_data=split_data,
        batch_size=batch_size,
        seed=seed,
    )
    num_clients = len(client_loaders)

    # Küresel modelin başlatılması
    global_model = build_model_from_config(cfg).to(device)
    global_parameters = get_model_parameters(global_model)

    # Eğer DP aktifse (Adım 6 - E4/E6), her istemci için gürültü çarpanını (noise_multiplier)
    # toplam tur ve yerel epoch sayısına göre kalibre et ve istemci başına RDPAccountant oluştur
    client_dp_configs: Dict[int, Dict[str, Any] | None] = {cid: None for cid in range(num_clients)}
    dp_summary_info: Dict[str, Any] = {
        "privacy": "None",
        "epsilon_target": None,
        "epsilon_spent": None,
        "delta": None,
        "noise_multiplier": None,
        "max_grad_norm": None,
        "accountant": None,
    }

    if dp_config is not None and dp_config.get("enabled", False):
        from privacy.dp_engine import prepare_federated_dp_clients

        client_dp_configs, dp_summary_info = prepare_federated_dp_clients(
            client_loaders=client_loaders,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            target_epsilon=float(dp_config["target_epsilon"]),
            delta=float(dp_config.get("delta", cfg["privacy"]["delta"])),
            max_grad_norm=float(dp_config.get("max_grad_norm", cfg["privacy"]["max_grad_norm"])),
            dp_lr=float(dp_config.get("dp_lr", cfg["privacy"].get("dp_lr", 0.005))),
            seed=seed,
        )

    # 17 Flower İstemcisinin Oluşturulması
    clients: List[FedHARFlowerClient] = []
    for cid, subject_id in enumerate(split_data.train_subject_ids):
        client_model = build_model_from_config(cfg).to(device)
        clients.append(
            FedHARFlowerClient(
                client_id=cid,
                subject_id=subject_id,
                model=client_model,
                train_loader=client_loaders[cid],
                lr=lr,
                weight_decay=weight_decay,
                local_epochs=local_epochs,
                device=device,
                dp_config=client_dp_configs[cid],
            )
        )

    best_val_f1 = -1.0
    best_round = 0
    best_global_state = copy.deepcopy(global_model.state_dict())
    history: List[Dict[str, Any]] = []

    for rnd in range(1, num_rounds + 1):
        round_results: List[Tuple[List[np.ndarray], int]] = []
        client_losses: List[float] = []
        client_epsilons: List[float] = []

        for client in clients:
            updated_params, num_examples, client_metrics = client.fit(
                global_parameters, config={"server_round": rnd}
            )
            round_results.append((updated_params, num_examples))
            client_losses.append(float(client_metrics["train_loss"]) * num_examples)
            if "epsilon_spent" in client_metrics and client_metrics["epsilon_spent"] is not None:
                client_epsilons.append(float(client_metrics["epsilon_spent"]))

        # Flower FedAvg Ağırlıklı Agregasyon: sum(n_k * w_k) / sum(n_k)
        global_parameters = aggregate(round_results)
        set_model_parameters(global_model, global_parameters)

        total_train_samples = sum(n for _, n in round_results)
        avg_train_loss = sum(client_losses) / max(total_train_samples, 1)

        # Küresel modelin 4 doğrulama (validation) katılımcısı üzerinde değerlendirilmesi
        val_metrics = evaluate_model(global_model, val_loader, device=device)

        max_eps_so_far = round(max(client_epsilons), 4) if client_epsilons else None

        history.append(
            {
                "round": rnd,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "epsilon_spent": max_eps_so_far,
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_round = rnd
            best_global_state = copy.deepcopy(global_model.state_dict())

        if verbose and (rnd == 1 or rnd % 5 == 0 or rnd == num_rounds):
            eps_str = f" | ε: {max_eps_so_far:.2f}" if max_eps_so_far is not None else ""
            print(
                f"[{experiment_id} | Seed {seed}] Round {rnd:02d}/{num_rounds:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.2f}%{eps_str}"
            )

    # En iyi doğrulama (validation) turundaki küresel modeli yükle
    global_model.load_state_dict(best_global_state)

    if history and history[-1].get("epsilon_spent") is not None:
        dp_summary_info["epsilon_spent"] = history[-1]["epsilon_spent"]

    model_save_path: Path | None = None
    if save_artifacts:
        model_save_path = get_path(f"result/models/{experiment_id}_seed{seed}.pt")

    # Dokunulmamış 9 test katılımcısı üzerinde nihai değerlendirme ve sistem/iletişim ölçümü
    test_benchmark = run_full_benchmark(
        model=global_model,
        test_loader=test_loader,
        save_path=model_save_path,
        warmup_runs=int(cfg["benchmark"]["warmup_runs"]),
        timed_runs=int(cfg["benchmark"]["timed_runs"]),
        fl_rounds=num_rounds,
        fl_clients_per_round=num_clients,
        device=device,
    )

    result_payload: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "training_mode": "Federated (FedAvg)",
        **dp_summary_info,
        "representation": "FP32",
        "seed": seed,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "best_val_round": best_round,
        "best_val_macro_f1": best_val_f1,
        **test_benchmark,
        "history": history,
    }

    if save_artifacts:
        log_path = get_path(f"result/logs/{experiment_id}_seed{seed}.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

    if verbose:
        print(
            f"--> [{experiment_id} TAMAMLANDI | Seed {seed}] En İyi Val Round: {best_round} | "
            f"TEST Accuracy: {test_benchmark['accuracy']:.2f}% | "
            f"TEST Macro-F1: {test_benchmark['macro_f1']:.2f}% | "
            f"Boyut: {test_benchmark['model_size_kb']:.2f} KB | "
            f"İletişim: {test_benchmark['total_comm_mb']:.2f} MB"
        )

    return result_payload


if __name__ == "__main__":
    run_federated_training(experiment_id="E2", seed=42)
