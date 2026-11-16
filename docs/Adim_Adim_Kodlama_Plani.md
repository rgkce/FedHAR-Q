# FedHAR-Q — Adım Adım Uygulamalı Kodlama Planı

Bu doküman, `FedHAR-Q` projesinin kodlama sürecini **7 ardışık faz (Adım 1 – Adım 7)** halinde, her adımda **hangi dosyanın yazılacağını**, **içinde ne olacağını** ve **o adımın çalıştığını nasıl test edeceğimizi (doğrulama kriteri)** gösterecek şekilde hazırlanmıştır.

Her adımı tek tek kodlayıp doğrulama testini geçtikten sonra bir sonraki adıma geçeceğiz.

---

## Adım 1: Ortam Kurulumu ve Konfigürasyon Altyapısı (`code/configs/`)
Kodun içine sabit sayı (hardcoded parametre) gömmemek ve tüm deneyleri (E1–E7) aynı ayarlarla tekrarlayabilmek için önce merkezi konfigürasyon yapısını kuracağız.

- **Yazılacak Dosyalar:**
  1. `code/configs/default.yaml`:
     - Rastgele tohumlar: `seeds: [42, 123, 999]`
     - Veri ayarları: `num_classes: 6`, `in_channels: 9`, `seq_len: 128`, `val_subjects_count: 4`
     - Model ayarları: `conv_channels: [32, 64]`, `kernel_size: 5`, `hidden_dim: 64`, `dropout: 0.2`
     - Merkezi ve FL eğitim ayarları: `batch_size: 64`, `lr: 0.001`, `centralized_epochs: 30`, `fl_rounds: 30`, `local_epochs: 2`, `num_clients: 17` (her eğitim katılımcısı 1 istemci)
     - DP ayarları: `delta: 1e-4`, `max_grad_norm: 1.0`, `epsilon_1: 3.0` (sıkı gizlilik), `epsilon_2: 8.0` (daha esnek gizlilik)
  2. `code/configs/config_loader.py`:
     - YAML dosyasını okuyan ve tüm sistemde tohumları (`random`, `numpy`, `torch`) sabitleyen `set_seed(seed)` yardımcı fonksiyonunu içeren modül.
- **Doğrulama Testi:** Python ortamındaki paketlerin (`torch`, `flwr`, `opacus`, `onnxruntime`, `sklearn`) kontrolü ve konfigürasyon dosyasının hatasız okunması.

---

## Adım 2: Veri Hattı ve Kişi Bazlı (Subject-Wise) Ayrım (`code/data/`)
Görev dokümanındaki en kritik kural olan **"kişi bazlı ayrım ve sıfır veri sızıntısı (zero data leakage)"** bu adımda kurulur.

- **Yazılacak Dosyalar:**
  1. `code/data/download_uci_har.py`:
     - UCI HAR veri kümesini (`UCI HAR Dataset.zip`) `dataset/raw/` klasörüne otomatik indirir ve açar.
  2. `code/data/preprocessing.py`:
     - `Inertial Signals` altındaki 9 sensör kanalını (`body_acc_x/y/z`, `body_gyro_x/y/z`, `total_acc_x/y/z`) `(N, 9, 128)` boyutlu FP32 tensörlere dönüştürür.
     - `subject_train.txt` (21 kişi) ve `subject_test.txt` (9 kişi) kimliklerini okur.
     - Eğitimdeki 21 kişiyi **kişi bazında** `17 Eğitim Kişisi` ve `4 Doğrulama (Validation) Kişisi` olarak ayırır.
     - Kanal bazlı ortalama ve standart sapmayı ($\mu, \sigma$) **sadece 17 eğitim kişisinin verisinden** hesaplayıp train/val/test kümelerini normalize eder.
  3. `code/data/dataset.py` & `code/data/partition.py`:
     - PyTorch `UCIHARDataset` sınıfını ve Federatif Öğrenme için her eğitim katılımcısını (veya istemci grubunu) kendi yerel `DataLoader` nesnesine dönüştüren istemci bölünmesi fonksiyonlarını içerir.
  4. `tests/test_data_leakage.py`:
     - Eğitim, doğrulama ve test kümelerindeki kişi ID'lerinin kesişiminin boş küme ($\emptyset$) olduğunu ve veri boyutlarının `(N, 9, 128)` olduğunu otomatik doğrular.
- **Doğrulama Testi:** `pytest tests/test_data_leakage.py` komutunun %100 başarılı geçmesi.

---

## Adım 3: Hafif 1D-CNN Modeli (`code/models/`)
Hem **Opacus (DP)** hem de **INT8 Kuantizasyon** ile tam uyumlu tek bir hafif 1D-CNN mimarisi kurulur.

- **Yazılacak Dosyalar:**
  1. `code/models/cnn1d.py`:
     - `BatchNorm1d` yerine Opacus uyumlu `GroupNorm` kullanan 1D-CNN sınıfı (`FedHAR1DCNN`).
     - Katman yapısı:
       - `Conv1d(9 -> 32, k=5, pad=2) -> GroupNorm -> ReLU -> MaxPool1d(2)`
       - `Conv1d(32 -> 64, k=5, pad=2) -> GroupNorm -> ReLU -> MaxPool1d(2)`
       - `AdaptiveAvgPool1d(1) -> Flatten -> Linear(64 -> 64) -> ReLU -> Dropout -> Linear(64 -> 6)`
     - Parametre sayısını ve teorik FP32 bellek boyutunu hesaplayan yardımcı metotlar.
  2. `tests/test_model.py`:
     - Modelin `(Batch, 9, 128)` girdiyle `(Batch, 6)` çıktı ürettiğini ve `opacus.validators.ModuleValidator.validate(model)` testinden **0 hata** ile geçtiğini doğrular.
- **Doğrulama Testi:** `pytest tests/test_model.py` testinin geçmesi.

---

## Adım 4: Değerlendirme ve Sistem Ölçüm Modülü (`code/evaluation/`)
Tüm deney koşullarında (E1–E7) aynı standart protokolle doğruluk ve kaynak tüketimi ölçen modüldür.

- **Yazılacak Dosyalar:**
  1. `code/evaluation/metrics.py`:
     - Test kümesi üzerinde **Accuracy (%)**, **Macro-F1 (%)**, sınıf bazlı F1 ve **Confusion Matrix** hesaplar.
  2. `code/evaluation/benchmark.py`:
     - **Model Boyutu (KB):** Kaydedilen model dosyasının disk boyutu.
     - **Çıkarım Gecikmesi (Latency - ms):** Isınma (warm-up) turları sonrası tekil örnek (`batch_size=1`) ve yığın (`batch_size=64`) için ortalama/standart sapma çıkarım süresi.
     - **Tepe Bellek (Peak RAM - KB/MB):** Çıkarım sırasında bellek tüketimi.
     - **FL İletişim Yükü (MB):** `Tur Sayısı × Katılan İstemci Sayısı × 2 (İndirme + Gönderme) × Model Ağırlık Boyutu` hesabı.
- **Doğrulama Testi:** Rastgele başlatılmış bir 1D-CNN üzerinde tüm ölçüm fonksiyonlarının çalışıp örnek bir JSON/sözlük döndürmesi.

---

## Adım 5: Merkezi (E1) ve Federatif (E2) Baseline Eğitimi (`code/federated/` & `code/experiments/`)
Gizlilik ve kuantizasyon eklenmeden önceki referans (baseline) modellerin eğitilmesi.

- **Yazılacak Dosyalar:**
  1. `code/experiments/train_centralized.py` (**E1 Koşulu**):
     - Merkezi FP32 1D-CNN modelini eğitir, doğrulama (validation) Macro-F1'i en yüksek ağırlığı `result/models/E1_seed{s}.pt` olarak kaydeder ve test metriklerini üretir.
  2. `code/federated/client.py` & `code/federated/server.py` (**E2 Koşulu**):
     - Flower `NumPyClient` istemcisi ve `FedAvg` sunucu agregasyonu.
     - Her FL turunda küresel modeli doğrulama kümesinde değerlendirir, en iyi küresel modeli `result/models/E2_seed{s}.pt` olarak kaydeder ve toplam iletişim yükünü (MB) hesaplar.
- **Doğrulama Testi:** Tek bir tohum (`seed=42`) ile E1 ve E2'nin uçtan uca çalıştırılıp test Accuracy ve Macro-F1 değerlerinin (%85–%92 bandında) görülmesi.

---

## Adım 6: Diferansiyel Gizlilik (E4, E6) ve INT8 Kuantizasyon (E3, E5, E7)
Araştırma sorularımızın (RQ1, RQ2, RQ3) kalbi olan DP ve INT8 modüllerinin entegrasyonu.

- **Yazılacak Dosyalar:**
  1. `code/privacy/dp_engine.py` (**E4: $\varepsilon_1$ ve E6: $\varepsilon_2$ Koşulları**):
     - Opacus `PrivacyEngine` ve `RDPAccountant` kullanarak istemci yerel eğitimini diferansiyel gizli (`DP-SGD`) hale getirir.
     - Hedef $\varepsilon_1$ (ör. `3.0`) ve $\varepsilon_2$ (ör. `8.0`), $\delta = 10^{-4}$ ve `max_grad_norm = 1.0` parametreleriyle gürültü çarpanını (`noise_multiplier`) kalibre eder ve harcanan gerçek $\varepsilon$ bütçesini loglar.
  2. `code/quantization/quantize.py` (**E3, E5, E7 Koşulları**):
     - Eğitilmiş FP32 modelleri (E2 $\rightarrow$ E3, E4 $\rightarrow$ E5, E6 $\rightarrow$ E7) alır.
     - Eğitim kümesinden ayrılan kalibrasyon verisiyle **INT8 Post-Training Static Quantization (ONNX Runtime INT8 & PyTorch Quantization)** uygular, `.onnx` / `.pt` olarak kaydeder ve aynı test kümesinde Accuracy, Macro-F1, Boyut (KB) ve Gecikme (ms) değerlerini ölçer.
- **Doğrulama Testi:** `tests/test_dp_and_quant.py` ile DP bütçe hesabının ve INT8 model boyut küçülmesinin (~3–4× küçülme) doğrulanması.

---

## Adım 7: Tüm Deney Matrisinin (3 Tohum × E1–E7) Koşturulması ve Tablo/Grafik Üretimi
Bildiriye doğrudan konulacak nihai istatistiksel sonuçların üretilmesi.

- **Yazılacak Dosyalar:**
  1. `code/experiments/run_matrix.py`:
     - `seeds = [42, 123, 999]` için **E1, E2, E3, E4, E5, E6, E7** (toplam 21 deney koşusu) matrisini sırasıyla çalıştırır ve ham sonuçları `result/logs/raw_results.csv` dosyasına yazar.
  2. `code/experiments/export_results.py`:
     - 3 tohumun **Ortalama ± Standart Sapma (`Mean ± Std`)** özet tablosunu hem `.csv` hem de doğrudan IEEE bildirisine yapıştırılacak LaTeX (`.tex`) tablosu olarak `result/tables/` altına kaydeder.
     - Doğruluk vs. Gizlilik ($\varepsilon$), FP32 vs. INT8 karşılaştırması ve Boyut-Gecikme-Doğruluk ödünleşim grafiklerini `result/plots/` ve `paper/figures/` altına yüksek çözünürlüklü (`.png` ve `.pdf`) olarak çizer.

---

## Takip Durumu (Neredeyiz?)

- [x] **Adım 1:** Ortam Kurulumu ve Konfigürasyon (`code/configs/`) — *Tamamlandı & 3/3 Test Geçti*
- [x] **Adım 2:** Veri Hattı ve Kişi Bazlı Ayrım (`code/data/` + `tests/test_data_leakage.py`) — *Tamamlandı & 4/4 Test Geçti*
- [x] **Adım 3:** Opacus & INT8 Uyumlu 1D-CNN Modeli (`code/models/` + `tests/test_model.py`) — *Tamamlandı & 3/3 Test Geçti*
- [x] **Adım 4:** Değerlendirme ve Sistem Ölçümleri (`code/evaluation/`) — *Tamamlandı & 3/3 Test Geçti*
- [x] **Adım 5:** Merkezi (E1) ve Federatif (E2) Baseline Eğitimi (`code/federated/`) — *Tamamlandı (E1 Acc: %93.69, E2 Acc: %90.70)*
- [x] **Adım 6:** Diferansiyel Gizlilik (E4, E6) ve INT8 Kuantizasyon (E3, E5, E7) (`code/privacy/` & `code/quantization/`) — *Tamamlandı & 2/2 Test Geçti*
- [x] **Adım 7:** 3 Tohumlu Tam Deney Matrisi (E1–E7) ve Bildiri Tablo/Grafiklerinin Üretimi (`code/experiments/`) — *Tamamlandı (21/21 Koşu, 3 Tablo, 4 Grafik)*
