# FedHAR-Q — Kurulum, Parametreler ve Çalıştırma Yönergesi

Bu belge, `Bildiri_Gorev_Dokumani.docx` içinde teslim çıktısı olarak istenen **"Çalıştırılabilir kaynak kod, kurulum yönergesi, parametreler ve deney konfigürasyonları"** rehberidir.

---

## 1. Sistem ve Yazılım Gereksinimleri

Deneylerin tekrarlanabilirliği için doğrulanan yazılım ve kütüphane sürümleri:
- **İşletim Sistemi:** Windows 10/11 (x86_64) veya Linux / Raspberry Pi OS (64-bit `aarch64`)
- **Python Sürümü:** `Python 3.10+` (Test edilen sürüm: `3.13.7`)
- **Temel Kütüphaneler:**
  - `torch == 2.12.0+cpu`
  - `flwr == 1.43.0` (Flower Federated Learning Framework)
  - `opacus == 1.6.0` (PyTorch Differential Privacy Engine)
  - `numpy == 2.2.6`, `scikit-learn == 1.6.1`, `pandas >= 2.0.0`, `matplotlib >= 3.7.0`, `seaborn >= 0.12.2`, `pyyaml >= 6.0.1`, `pytest >= 7.4.0`

---

## 2. Adım Adım Kurulum

### Adım 2.1: Proje Klasörüne Geçiş ve Sanal Ortam Kurulumu
```powershell
cd FedHAR-Q
python -m venv .venv
.\.venv\Scripts\activate
```

### Adım 2.2: Bağımlılıkların Yüklenmesi
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Adım 2.3: Kurulumun ve Birim Testlerinin Doğrulanması (17 Test)
Veri sızıntısı kontrolü, 1D-CNN model uyumluluğu, Opacus DP-SGD, INT8 kuantizasyon ve Flower istemci-sunucu yapısını doğrulamak için:
```powershell
python -m pytest -v
```
*(Beklenen çıktı: `17 passed`)*

---

## 3. Deney Konfigürasyon Dosyaları ve Parametreler

Tüm deney konfigürasyonları `code/configs/` dizininde ve deney matrisi tanımı `code/experiments/experiment_matrix.csv` dosyasında yer almaktadır:

| Konfigürasyon Dosyası | Kapsadığı Deneyler | Temel Parametreler |
| :--- | :---: | :--- |
| **[code/configs/default.yaml](../code/configs/default.yaml)** | `E1` – `E7` (Tümü) | `seeds: [42, 123, 999]`, `in_channels: 9`, `seq_len: 128`, `num_classes: 6`, `val_subjects_count: 4` |
| **[code/configs/baseline.yaml](../code/configs/baseline.yaml)** | `E1` (Merkezi FP32) | `epochs: 30`, `batch_size: 64`, `optimizer: Adam`, `lr: 0.001`, `weight_decay: 1e-4` |
| **[code/configs/federated.yaml](../code/configs/federated.yaml)** | `E2` (Federatif FP32) | `strategy: FedAvg`, `num_clients: 17`, `num_rounds: 30`, `local_epochs: 2`, `lr: 0.002` |
| **[code/configs/privacy.yaml](../code/configs/privacy.yaml)** | `E4`, `E6` (DP-SGD) | `accountant: RDPAccountant`, `delta: 1e-4`, `max_grad_norm: 2.0`, $\varepsilon_1 = 3.0$ ($\sigma=4.39$), $\varepsilon_2 = 8.0$ ($\sigma=2.03$) |
| **[code/configs/quantization.yaml](../code/configs/quantization.yaml)** | `E3`, `E5`, `E7` (INT8) | `method: PTQ Static`, `dtype: int8`, `calibration_batches: 20` (`1,280` eğitim penceresi) |
| **[code/experiments/experiment_matrix.csv](../code/experiments/experiment_matrix.csv)** | `E1` – `E7` Matrisi | Tüm 7 deney koşulunun satır bazlı parametre özeti |

---

## 4. Deneylerin Çalıştırılması

### 4.1. UCI HAR Veri Kümesinin İndirilmesi ve Doğrulanması
```powershell
python code/data/download_uci_har.py
```

### 4.2. Tekil Deneylerin Çalıştırılması
```powershell
# Sadece E1 (Merkezi FP32 Baseline, Seed=42)
python code/experiments/train_centralized.py

# Sadece E2 (Flower FedAvg FP32 Baseline, Seed=42)
python code/federated/server.py
```

### 4.3. Tüm Deney Matrisinin (E1–E7 × 3 Tohum = 21 Koşu) Çalıştırılması
```powershell
python code/experiments/run_matrix.py
```
- Bu komut `seeds = [42, 123, 999]` için `E1`–`E7` koşullarını sırasıyla çalıştırır.
- Model ağırlıklarını `result/models/E{1..7}_seed{42,123,999}.pt` altına kaydeder.
- Detaylı eğitim ve DP bütçe kayıtlarını `result/logs/E{1..7}_seed{42,123,999}.json` altına kaydeder.
- Ham sonuç tablosunu `result/tables/raw_all_21_runs.csv` olarak üretir.

### 4.4. Sonuç Tablolarının ve Akademik Grafiklerin Üretilmesi
```powershell
python code/experiments/export_results.py
```
- **Üretilen Tablolar (`result/tables/`):**
  - `summary_mean_std.csv`
  - `summary_table_markdown.md`
  - `ieee_results_table.tex`
- **Üretilen Grafikler (`result/plots/` ve `paper/figures/`):**
  - `fig1_accuracy_f1_comparison.png / .pdf`
  - `fig2_dp_quantization_interaction.png / .pdf`
  - `fig3_convergence_and_epsilon_curves.png / .pdf`
  - `fig4_size_tradeoff_and_per_class_f1.png / .pdf`
