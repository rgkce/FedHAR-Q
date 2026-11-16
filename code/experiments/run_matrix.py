r"""
FedHAR-Q Tam Deney Matrisi Koşucusu (`code/experiments/run_matrix.py`).

Çalışma planındaki (Bölüm 3.4) çekirdek deney matrisini (E1–E7) en az 3 farklı
rastgele tohumla (`seeds: [42, 123, 999]`, toplam 21 deney koşusu) çalıştırır
ve tüm ham sonuçları `result/tables/raw_all_21_runs.csv` dosyasına kaydeder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from configs.config_loader import get_path, load_config
from data.preprocessing import prepare_subject_wise_data
from experiments.train_centralized import train_centralized_model
from federated.server import run_federated_training
from quantization.quantize import run_quantization_experiment


def _load_cached_if_exists(exp_id: str, seed: int) -> Dict[str, Any] | None:
    log_path = get_path(f"result/logs/{exp_id}_seed{seed}.json")
    model_path = get_path(f"result/models/{exp_id}_seed{seed}.pt")
    if log_path.exists() and model_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_full_experiment_matrix(
    seeds: List[int] | None = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Verilen tüm tohumlar (`seeds`) için E1, E2, E3, E4, E5, E6 ve E7 koşullarını
    sırasıyla çalıştırır (veya `skip_existing=True` ise daha önce tamamlanmış
    koşuları önbellekten yükler) ve tüm sonuçların listesini döndürür.
    """
    cfg = load_config()
    if seeds is None:
        seeds = [int(s) for s in cfg["project"]["seeds"]]

    split_data = prepare_subject_wise_data(
        raw_dir=cfg["data"]["raw_dir"],
        val_subjects_count=int(cfg["data"]["val_subjects_count"]),
        split_seed=42,
        normalize=True,
    )

    eps_1 = float(cfg["privacy"]["epsilon_1"])
    eps_2 = float(cfg["privacy"]["epsilon_2"])

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n======================================================================")
        print(f"   DENEY MATRİSİ (E1 - E7) | TOHUM (SEED) = {seed}")
        print(f"======================================================================")

        # --- E1: Merkezi Eğitim | DP Yok | FP32 ---
        res_e1 = _load_cached_if_exists("E1", seed) if skip_existing else None
        if res_e1 is None:
            res_e1 = train_centralized_model(seed=seed, split_data=split_data, verbose=verbose)
        else:
            print(f"[ÖNBELLEK] E1 | Seed {seed} -> Acc: {res_e1['accuracy']:.2f}%, F1: {res_e1['macro_f1']:.2f}%")
        all_results.append(res_e1)

        # --- E2: Federatif Eğitim | DP Yok | FP32 ---
        res_e2 = _load_cached_if_exists("E2", seed) if skip_existing else None
        if res_e2 is None:
            res_e2 = run_federated_training(
                experiment_id="E2", seed=seed, split_data=split_data, verbose=verbose
            )
        else:
            print(f"[ÖNBELLEK] E2 | Seed {seed} -> Acc: {res_e2['accuracy']:.2f}%, F1: {res_e2['macro_f1']:.2f}%")
        all_results.append(res_e2)

        # --- E3: Federatif Eğitim | DP Yok | INT8 ---
        res_e3 = _load_cached_if_exists("E3", seed) if skip_existing else None
        if res_e3 is None:
            res_e3 = run_quantization_experiment(
                "E2", "E3", seed=seed, source_result=res_e2, split_data=split_data, verbose=verbose
            )
        else:
            print(f"[ÖNBELLEK] E3 | Seed {seed} -> Acc: {res_e3['accuracy']:.2f}%, F1: {res_e3['macro_f1']:.2f}%")
        all_results.append(res_e3)

        # --- E4: Federatif Eğitim | ε₁ = 3.0 Hedefi | FP32 ---
        res_e4 = _load_cached_if_exists("E4", seed) if skip_existing else None
        if res_e4 is None:
            res_e4 = run_federated_training(
                experiment_id="E4",
                seed=seed,
                dp_config={"enabled": True, "target_epsilon": eps_1, "dp_lr": 0.008},
                split_data=split_data,
                verbose=verbose,
            )
        else:
            print(f"[ÖNBELLEK] E4 | Seed {seed} -> Acc: {res_e4['accuracy']:.2f}%, F1: {res_e4['macro_f1']:.2f}%")
        all_results.append(res_e4)

        # --- E5: Federatif Eğitim | ε₁ = 3.0 Hedefi | INT8 ---
        res_e5 = _load_cached_if_exists("E5", seed) if skip_existing else None
        if res_e5 is None:
            res_e5 = run_quantization_experiment(
                "E4", "E5", seed=seed, source_result=res_e4, split_data=split_data, verbose=verbose
            )
        else:
            print(f"[ÖNBELLEK] E5 | Seed {seed} -> Acc: {res_e5['accuracy']:.2f}%, F1: {res_e5['macro_f1']:.2f}%")
        all_results.append(res_e5)

        # --- E6: Federatif Eğitim | ε₂ = 8.0 Hedefi | FP32 ---
        res_e6 = _load_cached_if_exists("E6", seed) if skip_existing else None
        if res_e6 is None:
            res_e6 = run_federated_training(
                experiment_id="E6",
                seed=seed,
                dp_config={"enabled": True, "target_epsilon": eps_2, "dp_lr": 0.005},
                split_data=split_data,
                verbose=verbose,
            )
        else:
            print(f"[ÖNBELLEK] E6 | Seed {seed} -> Acc: {res_e6['accuracy']:.2f}%, F1: {res_e6['macro_f1']:.2f}%")
        all_results.append(res_e6)

        # --- E7: Federatif Eğitim | ε₂ = 8.0 Hedefi | INT8 ---
        res_e7 = _load_cached_if_exists("E7", seed) if skip_existing else None
        if res_e7 is None:
            res_e7 = run_quantization_experiment(
                "E6", "E7", seed=seed, source_result=res_e6, split_data=split_data, verbose=verbose
            )
        else:
            print(f"[ÖNBELLEK] E7 | Seed {seed} -> Acc: {res_e7['accuracy']:.2f}%, F1: {res_e7['macro_f1']:.2f}%")
        all_results.append(res_e7)

    # 21 koşunun ham tablosunu kaydet
    rows = []
    for r in all_results:
        row = {
            "experiment_id": r["experiment_id"],
            "seed": r["seed"],
            "training_mode": r["training_mode"],
            "privacy": r["privacy"],
            "epsilon_target": r.get("epsilon_target"),
            "epsilon_spent": r.get("epsilon_spent"),
            "delta": r.get("delta"),
            "noise_multiplier": r.get("noise_multiplier"),
            "representation": r["representation"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "loss": r["loss"],
            "model_size_kb": r["model_size_kb"],
            "latency_mean_ms": r["latency_mean_ms"],
            "latency_median_ms": r["latency_median_ms"],
            "peak_inference_footprint_kb": r["peak_inference_footprint_kb"],
            "total_comm_mb": r["total_comm_mb"],
        }
        for act_name, f1_val in r.get("per_class_f1", {}).items():
            row[f"f1_{act_name}"] = f1_val
        rows.append(row)

    df_raw = pd.DataFrame(rows)
    raw_csv_path: Path = get_path("result/tables/raw_all_21_runs.csv")
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\n[MATRİS TAMAMLANDI] 21 koşunun ham tablosu kaydedildi: {raw_csv_path}")

    return all_results


if __name__ == "__main__":
    run_full_experiment_matrix()
