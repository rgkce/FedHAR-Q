"""
UCI HAR Veri Kümesi İndirme ve Doğrulama Modülü.

Resmî UCI Machine Learning Repository kaynağından "UCI HAR Dataset.zip"
dosyasını `dataset/raw/` dizinine indirir, çıkartır ve gerekli tüm
Inertial Signals (eylemsizlik sinyalleri) ile kişi (subject) dosyalarının
varlığını doğrular.
"""

from __future__ import annotations

import ssl
import urllib.request
import zipfile
from pathlib import Path

from configs.config_loader import get_path

UCI_HAR_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
]

SIGNAL_PREFIXES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def verify_dataset_files(dataset_dir: Path) -> bool:
    """UCI HAR veri kümesinin tüm zorunlu alt dosyalarının mevcut olduğunu kontrol eder."""
    if not dataset_dir.exists():
        return False

    for split in ("train", "test"):
        if not (dataset_dir / split / f"subject_{split}.txt").exists():
            return False
        if not (dataset_dir / split / f"y_{split}.txt").exists():
            return False
        for sig in SIGNAL_PREFIXES:
            sig_file = dataset_dir / split / "Inertial Signals" / f"{sig}_{split}.txt"
            if not sig_file.exists():
                return False
    return True


def download_and_extract_uci_har(raw_dir: str | Path = "dataset/raw") -> Path:
    """
    UCI HAR veri kümesi henüz indirilmemişse indirir ve `dataset/raw/UCI HAR Dataset`
    altına çıkartır. Veri kümesi klasörünün tam yolunu (`Path`) döndürür.
    """
    raw_path = get_path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    extracted_dir = raw_path / "UCI HAR Dataset"

    if verify_dataset_files(extracted_dir):
        return extracted_dir

    zip_path = raw_path / "UCI_HAR_Dataset.zip"

    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        downloaded = False
        for url in UCI_HAR_URLS:
            try:
                print(f"[UCI HAR] İndiriliyor: {url} ...")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                    data = resp.read()
                with open(zip_path, "wb") as f:
                    f.write(data)
                print(f"[UCI HAR] İndirme tamamlandı ({len(data) / (1024 * 1024):.1f} MB).")
                downloaded = True
                break
            except Exception as err:
                print(f"[UCI HAR] Uyarı - {url} indirilemedi: {err}")

        if not downloaded:
            raise RuntimeError("UCI HAR veri kümesi hiçbir kaynaktan indirilemedi.")

    print(f"[UCI HAR] Arşiv çıkartılıyor: {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_path)

    # Eğer iç içe zip varsa ("UCI HAR Dataset.zip" -> human+activity+recognition.zip içinde)
    inner_zip = raw_path / "UCI HAR Dataset.zip"
    if not verify_dataset_files(extracted_dir) and inner_zip.exists():
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(raw_path)

    if not verify_dataset_files(extracted_dir):
        raise FileNotFoundError(
            f"UCI HAR dosyaları çıkartıldıktan sonra doğrulanamadı: {extracted_dir}"
        )

    print(f"[UCI HAR] Veri kümesi hazır ve doğrulandı: {extracted_dir}")
    return extracted_dir


if __name__ == "__main__":
    download_and_extract_uci_har()
