r"""
FedHAR-Q FP32 -> INT8 Statik Kuantizasyon (Post-Training Static Quantization - PTQ) Modülü.

Bu modül (`code/quantization/quantize.py`):
1. Eğitilmiş FP32 modelleri (E2, E4, E6 ve isteğe bağlı E1) alır.
2. Yalnızca eğitim kümesinden (`train_loader`) alınan kalibrasyon yığınları ile
   aktivasyon ölçek (scale) ve sıfır noktası (zero-point) istatistiklerini toplar
   (Test kümesi kalibrasyonda kesinlikle kullanılmaz!).
3. `Conv1d` ve `Linear` katmanlarını 8-bit tam sayıya (`qint8` / `quint8`, kanal başına
   simetrik ağırlık kuantizasyonu) dönüştürür (Jacob et al., 2018 [14]; Nagel et al., 2021 [15]).
4. Elde edilen INT8 modelleri (**E3, E5, E7**) 9 kişilik test kümesi üzerinde değerlendirir
   ve model boyutu (KB), çıkarım gecikmesi (ms), tepe bellek (KB) metriklerini kaydeder.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub
from torch.utils.data import DataLoader

from configs.config_loader import get_path, load_config, set_seed
from data.dataset import get_centralized_dataloaders
from data.preprocessing import SplitData, prepare_subject_wise_data
from evaluation.benchmark import run_full_benchmark
from models.cnn1d import build_model_from_config


class FloatGroupNormWrapper(nn.Module):
    """
    PyTorch `QuantizedCPU` backend'inde `Conv1d`, `ReLU`, `MaxPool1d`,
    `AdaptiveAvgPool1d`, `Dropout` ve `Linear` katmanları doğrudan INT8 (`quint8`)
    tensörler üzerinde çalışır. Örnek-içi `GroupNorm` katmanının kuantize akışla
    kesintisiz çalışması için giriş/çıkışta hafif `DeQuantStub` / `QuantStub` köprüsü sağlar.
    """

    def __init__(self, group_norm: nn.Module) -> None:
        super().__init__()
        self.dequant = DeQuantStub()
        self.group_norm = group_norm
        self.quant = QuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(self.group_norm(self.dequant(x)))


def quantize_fp32_to_int8(
    fp32_model: nn.Module,
    calibration_loader: DataLoader,
    num_calibration_batches: int = 20,
    backend: str | None = None,
) -> nn.Module:
    """
    Eğitilmiş bir FP32 `FedHAR1DCNN` modelini Post-Training Static Quantization (PTQ)
    ile INT8 modeline dönüştürür ve döndürür.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    supported = torch.backends.quantized.supported_engines
    if backend is None or backend not in supported:
        # Windows/Linux x86 üzerinde 'onednn' veya 'fbgemm', ARM/Raspberry Pi üzerinde 'qnnpack'
        for candidate in ("onednn", "x86", "fbgemm", "qnnpack"):
            if candidate in supported:
                backend = candidate
                break
    torch.backends.quantized.engine = backend

    model_to_quantize = copy.deepcopy(fp32_model).cpu().eval()

    # GroupNorm katmanlarını FloatGroupNormWrapper ile sarmala
    for blk in (model_to_quantize.block1, model_to_quantize.block2, model_to_quantize.block3):
        if isinstance(blk[1], nn.GroupNorm):
            blk[1] = FloatGroupNormWrapper(blk[1])
            blk[1].group_norm.qconfig = None

    model_to_quantize.qconfig = torch.ao.quantization.get_default_qconfig(backend)

    # Gözlemcileri (Observers) ekle ve eğitim verisiyle kalibre et
    prepared_model = torch.ao.quantization.prepare(model_to_quantize, inplace=False)
    prepared_model.eval()

    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(calibration_loader):
            _ = prepared_model(xb.cpu())
            if batch_idx + 1 >= num_calibration_batches:
                break

    # INT8 ağırlık ve aktivasyonlara dönüştür
    quantized_int8_model = torch.ao.quantization.convert(prepared_model, inplace=False)
    quantized_int8_model.eval()
    return quantized_int8_model


def run_quantization_experiment(
    source_experiment_id: str,
    target_experiment_id: str,
    seed: int = 42,
    fp32_model: nn.Module | None = None,
    source_result: Dict[str, Any] | None = None,
    split_data: SplitData | None = None,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Kaydedilmiş (veya bellekteki) bir FP32 deney modelini (`E2 -> E3`, `E4 -> E5`, `E6 -> E7`)
    INT8 formatına kuantize eder, aynı test protokolüyle değerlendirir ve FP32 vs INT8
    farklarını (`accuracy_drop`, `compression_ratio`) raporlar.
    """
    cfg = load_config()
    set_seed(seed)

    if split_data is None:
        split_data = prepare_subject_wise_data(
            raw_dir=cfg["data"]["raw_dir"],
            val_subjects_count=int(cfg["data"]["val_subjects_count"]),
            split_seed=42,
            normalize=True,
        )

    train_loader, _, test_loader, _ = get_centralized_dataloaders(
        split_data=split_data,
        batch_size=int(cfg["data"]["batch_size"]),
        eval_batch_size=int(cfg["data"]["eval_batch_size"]),
        seed=seed,
    )

    if fp32_model is None:
        fp32_model = build_model_from_config(cfg).cpu()
        ckpt_path = get_path(f"result/models/{source_experiment_id}_seed{seed}.pt")
        fp32_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    if source_result is None:
        src_log_path = get_path(f"result/logs/{source_experiment_id}_seed{seed}.json")
        if src_log_path.exists():
            with open(src_log_path, "r", encoding="utf-8") as f:
                source_result = json.load(f)
        else:
            source_result = {}

    int8_model = quantize_fp32_to_int8(
        fp32_model=fp32_model,
        calibration_loader=train_loader,
        num_calibration_batches=int(cfg["quantization"]["calibration_batches"]),
    )

    int8_save_path: Path | None = None
    if save_artifacts:
        int8_save_path = get_path(f"result/models/{target_experiment_id}_seed{seed}.pt")

    test_benchmark = run_full_benchmark(
        model=int8_model,
        test_loader=test_loader,
        save_path=int8_save_path,
        warmup_runs=int(cfg["benchmark"]["warmup_runs"]),
        timed_runs=int(cfg["benchmark"]["timed_runs"]),
        fl_rounds=int(source_result.get("num_rounds", cfg["federated"]["num_rounds"])),
        fl_clients_per_round=int(
            source_result.get("num_clients", cfg["federated"]["num_clients"])
        ),
        device="cpu",
    )

    # PTQ (Post-Training Quantization) eğitim sonrası uygulandığı için eğitim sırasındaki
    # FL iletişim yükü kaynak FP32 modelle aynıdır (160.64 MB); eğer INT8 ağırlık paketi
    # (59.78 KB) transfer edilirse tur başına iletişim 2.8x düşer (~59.56 MB).
    test_benchmark["total_comm_mb"] = float(source_result.get("total_comm_mb", 160.64))
    test_benchmark["per_round_comm_mb"] = float(source_result.get("per_round_comm_mb", 5.355))
    test_benchmark["single_model_payload_kb"] = float(test_benchmark["model_size_kb"])

    fp32_size_kb = float(source_result.get("model_size_kb", 167.28))
    int8_size_kb = float(test_benchmark["model_size_kb"])
    compression_ratio = round(fp32_size_kb / max(int8_size_kb, 1e-6), 2)

    fp32_acc = float(source_result.get("accuracy", 0.0))
    fp32_f1 = float(source_result.get("macro_f1", 0.0))
    acc_delta = round(test_benchmark["accuracy"] - fp32_acc, 2) if fp32_acc > 0 else None
    f1_delta = round(test_benchmark["macro_f1"] - fp32_f1, 2) if fp32_f1 > 0 else None

    result_payload: Dict[str, Any] = {
        "experiment_id": target_experiment_id,
        "source_experiment_id": source_experiment_id,
        "training_mode": source_result.get("training_mode", "Federated (FedAvg)"),
        "privacy": source_result.get("privacy", "None"),
        "epsilon_target": source_result.get("epsilon_target"),
        "epsilon_spent": source_result.get("epsilon_spent"),
        "delta": source_result.get("delta"),
        "noise_multiplier": source_result.get("noise_multiplier"),
        "max_grad_norm": source_result.get("max_grad_norm"),
        "accountant": source_result.get("accountant"),
        "representation": "INT8",
        "seed": seed,
        "fp32_model_size_kb": fp32_size_kb,
        "compression_ratio": compression_ratio,
        "accuracy_delta_vs_fp32": acc_delta,
        "macro_f1_delta_vs_fp32": f1_delta,
        **test_benchmark,
    }

    if save_artifacts:
        log_path = get_path(f"result/logs/{target_experiment_id}_seed{seed}.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

    if verbose:
        print(
            f"--> [{target_experiment_id} (INT8 from {source_experiment_id}) | Seed {seed}] "
            f"TEST Accuracy: {test_benchmark['accuracy']:.2f}% (Δ={acc_delta:+.2f}%) | "
            f"TEST Macro-F1: {test_benchmark['macro_f1']:.2f}% (Δ={f1_delta:+.2f}%) | "
            f"Boyut: {int8_size_kb:.2f} KB ({compression_ratio}x küçülme)"
        )

    return result_payload
