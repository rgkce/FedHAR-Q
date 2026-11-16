r"""
FedHAR-Q Hafif 1D-CNN Mimarisi (`FedHAR1DCNN`).

Tasarım İlkeleri:
1. **Diferansiyel Gizlilik (Opacus DP-SGD) Uyumluluğu:**
   Standart `BatchNorm1d` katmanı yığın (batch) içindeki farklı örneklerin istatistiklerini
   karıştırdığı için örnek-bazlı gradyan (per-sample gradient) hesabını bozar ve DP garantisini
   ihlal eder. Bu nedenle mimaride örnek-içi (per-sample) normalizasyon yapan `GroupNorm`
   kullanılmıştır. Böylece E1–E7 arasındaki tüm koşullarda (DP'li ve DP'siz) **birebir aynı**
   mimari kullanılır.
2. **DP Gürültü Dayanıklılığı ve Hafiflik:**
   DP-SGD'de eklenen Gauss gürültüsünün toplam etkisi parametre sayısı ($P$) arttıkça büyür.
   3 evrişim bloğu ve 2 tam bağlantılı katmandan oluşan bu hafif mimari (~41,286 parametre,
   ~161 KB FP32), hem UCI HAR üzerinde yüksek ayırt edicilik sağlar hem de DP gürültüsünden
   minimum düzeyde etkilenir.
3. **INT8 Kuantizasyon Uyumluluğu:**
   Giriş ve çıkışta `QuantStub` / `DeQuantStub` kapıları içerir; hem PyTorch statik/dinamik
   INT8 kuantizasyonu hem de ONNX Runtime INT8 dönüşümü ile tam uyumludur.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub


class FedHAR1DCNN(nn.Module):
    """
    UCI HAR (9 kanal, 128 zaman adımı, 6 sınıf) için Opacus (DP) ve INT8 Kuantizasyon
    uyumlu hafif 1D Evrişimli Sinir Ağı (1D-CNN).
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        conv_channels: Sequence[int] = (32, 64, 64),
        kernel_size: int = 5,
        num_groups: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        pool_output_len: int = 4,
    ) -> None:
        super().__init__()
        c1, c2, c3 = (
            conv_channels[0],
            conv_channels[1],
            conv_channels[2] if len(conv_channels) > 2 else conv_channels[1],
        )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool_output_len = pool_output_len

        # INT8 Statik Kuantizasyon için giriş/çıkış kapıları (FP32 modunda birim işlemdir)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Evrişim Bloğu 1: (B, 9, 128) -> (B, 32, 64)
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                c1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Evrişim Bloğu 2: (B, 32, 64) -> (B, 64, 32)
        self.block2 = nn.Sequential(
            nn.Conv1d(
                c1,
                c2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=c2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Evrişim Bloğu 3: (B, 64, 32) -> (B, 64, 4)
        self.block3 = nn.Sequential(
            nn.Conv1d(
                c2,
                c3,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=pool_output_len),
        )

        flattened_dim = c3 * pool_output_len  # 64 * 4 = 256

        # Sınıflandırıcı Başlık (Classifier Head): (B, 256) -> (B, 6)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri yayılım (Forward pass).
        Girdi x: (Batch, 9, 128) -> Çıktı logits: (Batch, 6)
        """
        x = self.quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def build_model_from_config(cfg: Dict[str, Any] | None = None) -> FedHAR1DCNN:
    """Konfigürasyon sözlüğünden (`default.yaml`) `FedHAR1DCNN` modeli oluşturur."""
    if cfg is None:
        return FedHAR1DCNN()

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    conv_ch = list(model_cfg.get("conv_channels", [32, 64, 64]))
    if len(conv_ch) == 2:
        conv_ch.append(conv_ch[-1])

    return FedHAR1DCNN(
        in_channels=int(data_cfg.get("in_channels", 9)),
        num_classes=int(data_cfg.get("num_classes", 6)),
        conv_channels=conv_ch,
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        num_groups=int(model_cfg.get("num_groups", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Modelin toplam/eğitilebilir parametre sayısını ve teorik FP32 parametre belleğini (KB) döndürür.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes_fp32 = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "architecture": model.__class__.__name__,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "fp32_param_size_kb": round(param_bytes_fp32 / 1024.0, 2),
        "theoretical_int8_param_size_kb": round((total_params * 1) / 1024.0, 2),
    }
