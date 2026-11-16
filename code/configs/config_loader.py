"""
FedHAR-Q Konfigürasyon Yükleyici ve Tekrarlanabilirlik (Seed) Modülü.

Bu modül:
1. `code/configs/default.yaml` dosyasını okur ve tüm modüllere standart bir sözlük olarak sunar.
2. Proje kök dizinini (`PROJECT_ROOT`) işletim sisteminden bağımsız olarak çözer.
3. Deneylerin tekrarlanabilirliği için Python, NumPy ve PyTorch rastgele tohumlarını sabitler.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Proje kök dizini: .../FedHAR-Q
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "code" / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """YAML konfigürasyon dosyasını yükler ve sözlük (dict) olarak döndürür."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config


def set_seed(seed: int) -> None:
    """
    Tüm rastgele sayı üreteçlerini (Python, NumPy, PyTorch) verilen tohuma sabitler.
    En az 3 farklı tohumla (42, 123, 999) tekrarlanabilir deneyler için zorunludur.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_path(relative_path: str | Path) -> Path:
    """Proje kök dizinine göre tam dosya yolunu döndürür."""
    return PROJECT_ROOT / Path(relative_path)
