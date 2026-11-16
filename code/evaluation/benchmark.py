r"""
FedHAR-Q Sistem ve Kaynak Kullanımı Ölçüm Modülü (`code/evaluation/benchmark.py`).

Görev dokümanında zorunlu tutulan mühendislik ve kaynak metriklerini ölçer:
1. **Model Boyutu (KB):** Serileştirilmiş (`state_dict` / TorchScript) modelin disk üzerindeki
   gerçek boyutu (KB).
2. **Çıkarım Gecikmesi (Inference Latency - ms):** Isınma (warm-up) turları sonrasında
   tekil örnek (`batch_size=1`) için ortalama, standart sapma, medyan ve P95 gecikme süresi (ms).
3. **Tepe Bellek Kullanımı (Peak Memory - KB):** Model ağırlıkları, tamponlar (buffers) ve
   ileri yayılım (forward pass) sırasındaki aktivasyon tensörlerinin toplam tepe bellek izi.
4. **Federatif İletişim Yükü (Communication Cost - MB):** FL eğitimi boyunca istemciler ve
   sunucu arasında transfer edilen toplam model parametresi verisi (MB).
"""

from __future__ import annotations

import os
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_model


def measure_model_size_kb(
    model: nn.Module,
    save_path: str | Path | None = None,
) -> float:
    """
    Modelin `state_dict` dosyasını diske kaydeder ve gerçek dosya boyutunu
    Kilobyte (KB) cinsinden döndürür. `save_path` verilmezse geçici dosya kullanır.
    """
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        size_bytes = path.stat().st_size
        return round(size_bytes / 1024.0, 2)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return round(size_bytes / 1024.0, 2)


def measure_inference_latency_ms(
    model: nn.Module,
    input_shape: tuple[int, int, int] = (1, 9, 128),
    warmup_runs: int = 50,
    timed_runs: int = 500,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """
    Tekil örnek (`batch_size=1`) üzerinde gerçek zamanlı çıkarım gecikmesini (ms) ölçer.
    Önce `warmup_runs` kadar ısınma turu koşturur, ardından `timed_runs` turun
    sürelerini yüksek çözünürlüklü sayaçla (`perf_counter`) kaydeder.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    # Isınma (Warm-up) turları
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    latencies_ms = np.zeros(timed_runs, dtype=np.float64)
    with torch.no_grad():
        for i in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms[i] = (t1 - t0) * 1000.0

    return {
        "latency_mean_ms": round(float(np.mean(latencies_ms)), 4),
        "latency_std_ms": round(float(np.std(latencies_ms)), 4),
        "latency_median_ms": round(float(np.median(latencies_ms)), 4),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
    }


def measure_peak_memory_kb(
    model: nn.Module,
    input_shape: tuple[int, int, int] = (1, 9, 128),
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """
    Çıkarım sırasında modelin statik ağırlık/tampon boyutunu ve ileri yayılım
    esnasındaki en yüksek (tepe) aktivasyon + Python bellek kullanımını (KB) ölçer.
    """
    model.eval()
    model.to(device)

    # 1. Modelin diske/belleğe serileştirilmiş gerçek boyutu
    serialized_kb = measure_model_size_kb(model)

    # 2. İleri yayılım sırasındaki en büyük ara katman aktivasyon tensörlerinin toplamı
    activation_bytes = 0
    hooks = []

    def _hook_fn(_module: nn.Module, _inp: Any, out: Any) -> None:
        nonlocal activation_bytes
        if isinstance(out, torch.Tensor):
            # INT8 kuantize tensörlerde element_size=1, FP32'de element_size=4
            activation_bytes += out.numel() * out.element_size()

    for m in model.modules():
        if m is not model:
            hooks.append(m.register_forward_hook(_hook_fn))

    dummy_input = torch.randn(*input_shape, device=device)
    tracemalloc.start()
    with torch.no_grad():
        _ = model(dummy_input)
    _, trace_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for h in hooks:
        h.remove()

    activation_kb = activation_bytes / 1024.0
    total_footprint_kb = serialized_kb + activation_kb

    return {
        "model_serialized_kb": round(serialized_kb, 2),
        "activation_Buffer_kb": round(activation_kb, 2),
        "peak_inference_footprint_kb": round(total_footprint_kb, 2),
        "python_tracemalloc_peak_kb": round(trace_peak_bytes / 1024.0, 2),
    }


def compute_fl_communication_cost_mb(
    num_rounds: int,
    num_clients_per_round: int,
    model_or_size_kb: nn.Module | float,
) -> Dict[str, float]:
    """
    Federatif Öğrenme (FedAvg) iletişim yükünü hesaplar.
    Her turda sunucu $K$ istemciye küresel modeli gönderir (downlink) ve
    $K$ istemciden güncellenmiş yerel modeli alır (uplink):
        Toplam İletişim = num_rounds * num_clients_per_round * 2 * model_boyutu
    """
    if isinstance(model_or_size_kb, nn.Module):
        param_bytes = sum(p.numel() * p.element_size() for p in model_or_size_kb.parameters())
        single_transfer_kb = param_bytes / 1024.0
    else:
        single_transfer_kb = float(model_or_size_kb)

    per_round_kb = num_clients_per_round * 2.0 * single_transfer_kb
    total_kb = num_rounds * per_round_kb
    total_mb = total_kb / 1024.0

    return {
        "single_model_payload_kb": round(single_transfer_kb, 2),
        "per_round_comm_mb": round(per_round_kb / 1024.0, 3),
        "total_comm_mb": round(total_mb, 2),
    }


def run_full_benchmark(
    model: nn.Module,
    test_loader: DataLoader,
    save_path: str | Path | None = None,
    warmup_runs: int = 50,
    timed_runs: int = 500,
    fl_rounds: int = 0,
    fl_clients_per_round: int = 0,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Bir model için sınıflandırma doğruluğu (Accuracy, Macro-F1, Confusion Matrix)
    ve sistem metriklerini (Boyut KB, Gecikme ms, Tepe Bellek KB, FL İletişim MB)
    tek seferde ölçüp birleştirilmiş sözlük olarak döndürür.
    """
    cls_metrics = evaluate_model(model, test_loader, device=device)
    size_kb = measure_model_size_kb(model, save_path=save_path)
    lat_metrics = measure_inference_latency_ms(
        model, warmup_runs=warmup_runs, timed_runs=timed_runs, device=device
    )
    mem_metrics = measure_peak_memory_kb(model, device=device)

    if fl_rounds > 0 and fl_clients_per_round > 0:
        comm_metrics = compute_fl_communication_cost_mb(fl_rounds, fl_clients_per_round, model)
    else:
        comm_metrics = {
            "single_model_payload_kb": 0.0,
            "per_round_comm_mb": 0.0,
            "total_comm_mb": 0.0,
        }

    return {
        **cls_metrics,
        "model_size_kb": size_kb,
        **lat_metrics,
        **mem_metrics,
        **comm_metrics,
    }
