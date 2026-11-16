r"""
FedHAR-Q Raspberry Pi (ARM Uç Cihaz) Dağıtım ve Çıkarım Ölçüm Betiği
(`code/experiments/run_rpi_benchmark.py`).

Bu betik Raspberry Pi üzerinde çalıştırılmak üzere tasarlanmıştır:
- Ağır federatif eğitimleri (E1, E2, E4, E6) tekrar çalıştırmaz! Bilgisayarda eğitilip
  `result/models/` altına kaydedilmiş hazır FP32 model ağırlıklarını (`E1`, `E2`, `E4`, `E6`)
  doğrudan yükler.
- Raspberry Pi'nin ARM işlemcisi üzerinde `qnnpack` (veya mevcut kuantize motor) ile
  FP32 modelleri INT8'e (`E3`, `E5`, `E7`) dönüştürür.
- Her koşul (E1–E7) ve her tohum (`42, 123, 999`) için Raspberry Pi üzerinde:
  1. Test Accuracy (%) ve Macro-F1 (%)
  2. Model Boyutu (KB)
  3. Gerçek Cihaz Çıkarım Gecikmesi (Latency - ms/pencere, 50 ısınma + 500 ölçüm)
  4. Tepe Bellek Kullanımı (Peak Memory - KB)
  ölçümlerini yapar ve `result/tables/rpi_edge_benchmark_summary.csv` ile
  `result/tables/rpi_edge_benchmark_markdown.md` dosyalarına kaydeder.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

# Proje code/ dizinini Python yoluna ekle
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd
import torch

from configs.config_loader import get_path, load_config, set_seed
from data.dataset import get_centralized_dataloaders
from data.preprocessing import prepare_subject_wise_data
from evaluation.benchmark import run_full_benchmark
from models.cnn1d import build_model_from_config
from quantization.quantize import quantize_fp32_to_int8


def run_raspberry_pi_benchmarks() -> pd.DataFrame:
    cfg = load_config()
    seeds: List[int] = [int(s) for s in cfg["project"]["seeds"]]

    hw_info = f"{platform.system()} {platform.release()} | {platform.machine()} | Python {platform.python_version()} | PyTorch {torch.__version__}"
    supported_engines = torch.backends.quantized.supported_engines
    print("======================================================================")
    print("   FedHAR-Q UÇ CİHAZ (RASPBERRY PI / EDGE) ÇIKARIM ÖLÇÜMÜ")
    print(f"   Donanım: {hw_info}")
    print(f"   Desteklenen Kuantizasyon Motorları: {supported_engines}")
    print("======================================================================")

    split_data = prepare_subject_wise_data(
        raw_dir=cfg["data"]["raw_dir"],
        val_subjects_count=int(cfg["data"]["val_subjects_count"]),
        split_seed=42,
        normalize=True,
    )

    warmup_runs = int(cfg["benchmark"]["warmup_runs"])
    timed_runs = int(cfg["benchmark"]["timed_runs"])

    # (ExpID, Kaynak FP32 Modeli, Temsil, Gizlilik Etiketi)
    matrix_plan = [
        ("E1", "E1", "FP32", "None"),
        ("E2", "E2", "FP32", "None"),
        ("E3", "E2", "INT8", "None"),
        ("E4", "E4", "FP32", "DP-SGD (eps=3.0)"),
        ("E5", "E4", "INT8", "DP-SGD (eps=3.0)"),
        ("E6", "E6", "FP32", "DP-SGD (eps=8.0)"),
        ("E7", "E6", "INT8", "DP-SGD (eps=8.0)"),
    ]

    all_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        train_loader, _, test_loader, _ = get_centralized_dataloaders(
            split_data=split_data,
            batch_size=int(cfg["data"]["batch_size"]),
            eval_batch_size=int(cfg["data"]["eval_batch_size"]),
            seed=seed,
        )

        print(f"\n--- Tohum (Seed): {seed} Ölçülüyor ---")
        for exp_id, src_id, rep, priv in matrix_plan:
            ckpt_path = get_path(f"result/models/{src_id}_seed{seed}.pt")
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Eğitilmiş model bulunamadı: {ckpt_path}. "
                    f"Lütfen bilgisayardaki 'result/models/' klasörünü Raspberry Pi'ye kopyalayın."
                )

            fp32_model = build_model_from_config(cfg).cpu().eval()
            fp32_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

            if rep == "FP32":
                eval_model = fp32_model
            else:
                eval_model = quantize_fp32_to_int8(
                    fp32_model=fp32_model,
                    calibration_loader=train_loader,
                    num_calibration_batches=int(cfg["quantization"]["calibration_batches"]),
                )

            bench = run_full_benchmark(
                model=eval_model,
                test_loader=test_loader,
                save_path=None,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
                device="cpu",
            )

            print(
                f"[{exp_id} | {rep:4s} | Seed {seed}] "
                f"Acc: {bench['accuracy']:.2f}% | F1: {bench['macro_f1']:.2f}% | "
                f"Boyut: {bench['model_size_kb']:.2f} KB | "
                f"Gecikme: {bench['latency_mean_ms']:.4f} ms (Medyan: {bench['latency_median_ms']:.4f} ms)"
            )

            all_rows.append(
                {
                    "experiment_id": exp_id,
                    "seed": seed,
                    "privacy": priv,
                    "representation": rep,
                    "accuracy": bench["accuracy"],
                    "macro_f1": bench["macro_f1"],
                    "model_size_kb": bench["model_size_kb"],
                    "latency_mean_ms": bench["latency_mean_ms"],
                    "latency_median_ms": bench["latency_median_ms"],
                    "latency_p95_ms": bench["latency_p95_ms"],
                    "peak_memory_kb": bench["peak_inference_footprint_kb"],
                    "hardware": platform.machine(),
                }
            )

    df_raw = pd.DataFrame(all_rows)
    tables_dir = get_path("result/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(tables_dir / "rpi_raw_21_runs.csv", index=False)

    # Ortalama ± Std özet tablosu
    summary_rows = []
    for exp_id, _, rep, priv in matrix_plan:
        sub = df_raw[df_raw["experiment_id"] == exp_id]
        summary_rows.append(
            {
                "experiment_id": exp_id,
                "privacy": priv,
                "representation": rep,
                "accuracy_formatted": f"{sub['accuracy'].mean():.2f} ± {sub['accuracy'].std(ddof=1):.2f}",
                "macro_f1_formatted": f"{sub['macro_f1'].mean():.2f} ± {sub['macro_f1'].std(ddof=1):.2f}",
                "model_size_kb": round(float(sub["model_size_kb"].mean()), 2),
                "latency_mean_ms": round(float(sub["latency_mean_ms"].mean()), 4),
                "latency_std_ms": round(float(sub["latency_mean_ms"].std(ddof=1)), 4),
                "latency_formatted": f"{sub['latency_mean_ms'].mean():.3f} ± {sub['latency_mean_ms'].std(ddof=1):.3f}",
                "peak_memory_kb": round(float(sub["peak_memory_kb"].mean()), 2),
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = tables_dir / "rpi_edge_benchmark_summary.csv"
    df_sum.to_csv(sum_csv, index=False)

    md_lines = [
        f"# FedHAR-Q — Uç Cihaz (Raspberry Pi / Edge) Çıkarım Sonuçları",
        f"**Donanım Ortamı:** `{hw_info}`",
        "",
        "| Kod | Gizlilik (DP) | Temsil | Test Accuracy (%) | Test Macro-F1 (%) | Model Boyutu (KB) | Cihaz Gecikmesi (ms) | Tepe Bellek (KB) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_sum.iterrows():
        md_lines.append(
            f"| **{r['experiment_id']}** | {r['privacy']} | **{r['representation']}** | "
            f"**{r['accuracy_formatted']}** | **{r['macro_f1_formatted']}** | "
            f"{r['model_size_kb']:.2f} | **{r['latency_formatted']}** | {r['peak_memory_kb']:.2f} |"
        )

    md_path = tables_dir / "rpi_edge_benchmark_markdown.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[TAMAMLANDI] Raspberry Pi sonuçları kaydedildi:\n  - {sum_csv}\n  - {md_path}")
    return df_sum


if __name__ == "__main__":
    run_raspberry_pi_benchmarks()
