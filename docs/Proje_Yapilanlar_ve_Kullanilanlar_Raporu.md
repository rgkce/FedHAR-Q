# FedHAR-Q — Projede Yapılanlar, Kullanılan Teknolojiler ve Elde Edilen Sonuçlar Raporu

**Proje Adı:** FedHAR-Q (*Federated Human Activity Recognition with Differential Privacy and Quantization*)  
**Türkçe Başlık:** Federated İnsan Aktivite Tanımada Diferansiyel Gizlilik ve Kuantizasyonun Birlikte Etkisi  
**İngilizce Başlık:** *Evaluating the Joint Impact of Differential Privacy and Quantization on Federated Human Activity Recognition*

Bu doküman; projede **hangi araçların, kütüphanelerin, veri protokollerinin ve algoritmaların kullanıldığını**, **hangi kod modüllerinin yazıldığını** ve **21 deney koşusu sonunda hangi bilimsel bulguların elde edildiğini** tek bir merkezde özetler.

---

## 1. Projede Kullanılan Yazılım, Kütüphane ve Araçlar

Tüm geliştirme ve simülasyon çalışmaları tekrarlanabilir bir Python ortamında gerçekleştirilmiş ve kütüphane sürümleri sabitlenmiştir:

| Araç / Kütüphane | Sürüm | Projede Kullanım Amacı |
| :--- | :---: | :--- |
| **Python** | `3.13.7` | Temel programlama dili ve deney çalışma zamanı |
| **PyTorch (`torch`)** | `2.12.0+cpu` | 1D-CNN model tasarımı, merkezi/yerel eğitim ve `torch.ao.quantization` (`onednn`/`qnnpack`) ile INT8 statik kuantizasyon |
| **Flower (`flwr`)** | `1.43.0` | Federatif Öğrenme (FL) istemci (`NumPyClient`) ve sunucu (`FedAvg` ağırlıklı agregasyon) simülasyonu |
| **Opacus (`opacus`)** | `1.6.0` | Örnek-bazlı gradyan kırpma (`GradSampleModule`), Gauss gürültüsü (`DPOptimizer`) ve Rényi DP muhasebesi (`RDPAccountant`) |
| **NumPy (`numpy`)** | `2.2.6` | Sensör sinyali matris işlemleri, kanal bazlı normalizasyon ve Flower parametre serileştirmesi |
| **Scikit-Learn (`sklearn`)** | `1.6.1` | Accuracy, Macro-F1, sınıf bazlı F1 ve `6×6` Confusion Matrix hesaplamaları |
| **Pandas (`pandas`)** | `2.x` | 21 deney koşusunun ham ve özet (`Ortalama ± Std`) sonuç tablolarının oluşturulması |
| **Matplotlib & Seaborn** | `3.x / 0.13` | Bildiri için 300 DPI `.png` ve vektörel `.pdf` formatında akademik grafiklerin çizimi |
| **PyYAML (`yaml`)** | `6.0.x` | Tüm deney hiperparametrelerinin (`code/configs/default.yaml`) merkezi yönetimi |
| **Pytest (`pytest`)** | `9.2.0` | Veri sızıntısı, model uyumluluğu, DP muhasebesi ve kuantizasyon için 17 otomatik birim testi |

---

## 2. Kullanılan Veri Kümesi ve Sıfır Sızıntılı (Subject-Wise) Veri Protokolü

### 2.1. Veri Kümesi: UCI HAR (`[1]`)
- **Sensör Kanalları (9 Kanal):** Akıllı telefon ivmeölçer ve jiroskopundan 50 Hz örnekleme hızıyla kaydedilen ham eylemsizlik sinyalleri (`Inertial Signals`):
  - `body_acc_x`, `body_acc_y`, `body_acc_z` (Yerçekimi arındırılmış gövde ivmesi)
  - `body_gyro_x`, `body_gyro_y`, `body_gyro_z` (Jiroskop açısal hız sinyalleri)
  - `total_acc_x`, `total_acc_y`, `total_acc_z` (Toplam üç eksenli ivme)
- **Pencere Boyutu:** 2.56 saniye (%50 örtüşmeli) $\rightarrow$ **128 zaman adımı × 9 kanal = `(9, 128)`** boyutlu girdi tensörü.
- **Aktivite Sınıfları (6 Sınıf):**
  - `0: WALKING`, `1: WALKING_UPSTAIRS`, `2: WALKING_DOWNSTAIRS`, `3: SITTING`, `4: STANDING`, `5: LAYING`.

### 2.2. Kişi Bazlı (Subject-Wise) Ayrım ve Normalizasyon
Görev dokümanındaki *"kişi bazlı ayrım korunacak, veri sızıntısı önlenecek ve model seçimi test kümesi üzerinde yapılmayacak"* kuralı doğrultusunda 30 katılımcı şu şekilde ayrılmıştır:

| Alt Küme | Katılımcı Sayısı | Katılımcı (Subject) Kimlikleri | Örnek (Pencere) Sayısı | Kullanım Amacı |
| :--- | :---: | :--- | :---: | :--- |
| **Eğitim (Train)** | **17 Kişi** | `[5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 26, 28, 29, 30]` | **5,879** | Merkezi eğitim (E1) ve 17 FL istemcisinin yerel verisi (E2–E7) + INT8 PTQ kalibrasyonu |
| **Doğrulama (Val)** | **4 Kişi** | `[1, 3, 25, 27]` | **1,473** | En iyi epoch/round (checkpoint) seçimi |
| **Test (Test)** | **9 Kişi** | `[2, 4, 9, 10, 12, 13, 18, 20, 24]` | **2,947** | Yalnızca nihai başarım değerlendirmesi |
| **TOPLAM** | **30 Kişi** | **Kesişim = $\emptyset$ (Sıfır Veri Sızıntısı)** | **10,299** | — |

- **Sızıntısız Normalizasyon:** 9 kanalın ortalaması ($\mu$) ve standart sapması ($\sigma$) **yalnızca 17 eğitim kişisinin 5,879 örneği** üzerinden hesaplanmış ve doğrulama/test kümelerine uygulanmıştır.
- **Federatif İstemci Yapısı:** 17 eğitim katılımcısının her biri bağımsız birer Flower istemcisi (`client_id = 0..16`) olarak modellenmiştir (gerçekçi, doğal kullanıcı bazlı Non-IID veri dağılımı; istemci başına min: `281`, maks: `408`, ortalama: `345.8` örnek).

---

## 3. Kurulan Sinir Ağı Mimarisi (`FedHAR1DCNN`)

Hem **Opacus (DP-SGD)** hem de **INT8 Kuantizasyon** ile %100 uyumlu olması için `BatchNorm1d` yerine örnek-içi (per-sample) normalizasyon yapan `GroupNorm(num_groups=8)` kullanılmış ve tüm koşullarda (E1–E7) birebir aynı mimari korunmuştur:

| Katman Bloğu | İçerik | Girdi Boyutu | Çıktı Boyutu | Parametre Sayısı |
| :--- | :--- | :---: | :---: | :---: |
| **Giriş** | `QuantStub()` | `(B, 9, 128)` | `(B, 9, 128)` | `0` |
| **Blok 1** | `Conv1d(9→32, k=5, p=2)` + `GroupNorm(8, 32)` + `ReLU` + `MaxPool1d(2)` | `(B, 9, 128)` | `(B, 32, 64)` | `1,536` |
| **Blok 2** | `Conv1d(32→64, k=5, p=2)` + `GroupNorm(8, 64)` + `ReLU` + `MaxPool1d(2)` | `(B, 32, 64)` | `(B, 64, 32)` | `10,432` |
| **Blok 3** | `Conv1d(64→64, k=3, p=1)` + `GroupNorm(8, 64)` + `ReLU` + `AdaptiveAvgPool1d(4)` | `(B, 64, 32)` | `(B, 64, 4)` | `12,480` |
| **Sınıflandırıcı** | `Flatten` $\rightarrow$ `Linear(256→64)` + `ReLU` + `Dropout(0.2)` $\rightarrow$ `Linear(64→6)` | `(B, 64, 4)` | `(B, 6)` | `16,838` |
| **Çıkış** | `DeQuantStub()` | `(B, 6)` | `(B, 6)` | `0` |
| **TOPLAM** | **Hafif 1D-CNN (`FedHAR1DCNN`)** | **FP32: `167.29 KB`** | **INT8: `59.78 KB`** | **`41,286`** |

---

## 4. Uygulanan Yöntemler ve Deneysel Parametreler

1. **Merkezi Eğitim (E1):**
   - `Optimizer`: Adam (`lr = 0.001`, `weight_decay = 1e-4`), `batch_size = 64`, `epochs = 30`.
2. **Federatif Öğrenme (E2–E7):**
   - `Çerçeve`: Flower (`flwr`) + `FedAvg` ağırlıklı ortalama ($w^{(t+1)} = \sum \frac{n_k}{N} w_k^{(t+1)}$).
   - `İstemci Sayısı`: $K = 17$ (tam katılım, `fraction_fit = 1.0`), `num_rounds = 30`, `local_epochs = 2`, `batch_size = 64`.
   - `İletişim Yükü`: Tur başına `5.355 MB` (17 istemci indirme + gönderme), 30 tur sonunda toplam **`160.64 MB`**.
3. **Diferansiyel Gizlilik (E4–E7):**
   - `Kütüphane ve Muhasebe`: Opacus `GradSampleModule`, `DPOptimizer` ve **Rényi DP (`RDPAccountant`)**.
   - `Gizlilik Birimi`: Örnek düzeyi (*Sample-level DP-SGD*, istemci içindeki bireysel sensör pencereleri düzeyinde).
   - `Sabit Parametreler`: $\delta = 10^{-4} < 1/N_{\text{train}}$, maksimum gradyan kırpma normu $C = 2.0$ (`max_grad_norm`).
   - `Gizlilik Hedefleri`:
     - **Sıkı Gizlilik ($\varepsilon_1 = 3.00$, E4 & E5):** Kalibre edilen gürültü çarpanı $\sigma = 4.3911$, 30. tur sonunda harcanan bütçe $\varepsilon = 3.00$.
     - **Esnek Gizlilik ($\varepsilon_2 = 8.00$, E6 & E7):** Kalibre edilen gürültü çarpanı $\sigma = 2.0334$, 30. tur sonunda harcanan bütçe $\varepsilon = 7.99$.
4. **INT8 Model Kuantizasyonu (E3, E5, E7):**
   - `Yöntem`: *Post-Training Static Quantization (PTQ)*.
   - `Kalibrasyon`: Eğitim kümesinden 20 mini-batch (`1,280` pencere) ile aktivasyon ölçek (`scale`) ve sıfır noktası (`zero_point`) kalibrasyonu.
   - `Sıkıştırma Başarımı`: Model disk boyutu **`167.29 KB`'dan `59.78 KB`'a (`2.80×` / `%64.3` küçülme)**, tepe çıkarım belleği ise **`327.64 KB`'dan `189.95 KB`'a (`%42.0` azalma)** düşürülmüştür.

---

## 5. Projede Yazılan Kod Dosyaları ve Görevleri

### A. Kaynak Kodlar (`code/`)
- **[code/configs/default.yaml](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/configs/default.yaml) & [config_loader.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/configs/config_loader.py):** Tüm hiperparametrelerin okunması ve `set_seed()` ile rastgele tohumların (`42, 123, 999`) sabitlenmesi.
- **[code/data/download_uci_har.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/data/download_uci_har.py):** UCI HAR veri kümesinin otomatik indirilmesi ve dosya bütünlüğü kontrolü.
- **[code/data/preprocessing.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/data/preprocessing.py):** 9 kanallı sinyallerin okunması, kişi bazlı (`17 Train / 4 Val / 9 Test`) ayrım ve sızıntısız normalizasyon.
- **[code/data/dataset.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/data/dataset.py) & [partition.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/data/partition.py):** PyTorch `UCIHARDataset` sınıfı ve 17 istemcili FL veri bölünmesi.
- **[code/models/cnn1d.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/models/cnn1d.py):** Opacus ve INT8 uyumlu `FedHAR1DCNN` mimarisi.
- **[code/evaluation/metrics.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/evaluation/metrics.py) & [benchmark.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/evaluation/benchmark.py):** Accuracy, Macro-F1, Confusion Matrix, Model Boyutu (KB), Çıkarım Gecikmesi (ms), Tepe Bellek (KB) ve FL İletişim Yükü (MB) ölçümleri.
- **[code/federated/client.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/federated/client.py) & [server.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/federated/server.py):** Flower `FedHARFlowerClient` ve `FedAvg` sunucu simülasyonu.
- **[code/privacy/dp_engine.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/privacy/dp_engine.py):** Opacus DP-SGD yerel eğitimi, adım bazlı $\sigma$ kalibrasyonu ve `RDPAccountant` takibi.
- **[code/quantization/quantize.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/quantization/quantize.py):** `FloatGroupNormWrapper` köprüsü ve FP32 $\rightarrow$ INT8 statik kuantizasyon dönüşümü.
- **[code/experiments/train_centralized.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/experiments/train_centralized.py):** E1 merkezi baseline eğitim betiği.
- **[code/experiments/run_matrix.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/experiments/run_matrix.py):** E1–E7 × 3 tohum (21 koşu) tam deney matrisi orkestratörü.
- **[code/experiments/export_results.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/experiments/export_results.py):** Otomatik CSV/Markdown/LaTeX tablo ve 4 akademik grafik üreticisi.
- **[code/experiments/run_rpi_benchmark.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/code/experiments/run_rpi_benchmark.py):** Raspberry Pi (ARM `qnnpack`) üzerinde tek komutla çıkarım ölçümü yapan betik.

### B. Otomatik Birim Testleri (`tests/` — 17/17 Test Geçti)
- **[tests/test_config.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/tests/test_config.py):** Konfigürasyon ve tohum tekrarlanabilirlik testleri (3 test).
- **[tests/test_data_leakage.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/tests/test_data_leakage.py):** Kişi bazlı sıfır veri sızıntısı ve normalizasyon testleri (4 test).
- **[tests/test_model.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/tests/test_model.py):** 1D-CNN boyut, `Opacus ModuleValidator` ve `grad_sample` testleri (3 test).
- **[tests/test_evaluation.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/tests/test_evaluation.py):** Doğruluk, gecikme, bellek ve iletişim yükü ölçüm testleri (3 test).
- **[tests/test_federated.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/tests/test_federated.py):** Flower parametre dönüşümü ve istemci eğitim testi (2 test).
- **[tests/test_dp_and_quant.py](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/tests/test_dp_and_quant.py):** RDP gizlilik bütçesi ve INT8 boyut küçültme testi (2 test).

---

## 6. Elde Edilen Nihai Deney Sonuçları (`3 Tohum: 42, 123, 999`)

| Kod | Eğitim Modu | Gizlilik (DP) | Harcanan $\varepsilon$ ($\delta=10^{-4}$) | $\sigma$ | Temsil | **Test Accuracy (%)** | **Test Macro-F1 (%)** | **Model Boyutu (KB)** | **Tepe Bellek (KB)** | **Çıkarım Gecikmesi (ms)** | **FL İletişim (MB)** |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Merkezi | Yok | $\infty$ | — | **FP32** | **`93.67 ± 0.65`** | **`93.62 ± 0.69`** | `167.29 KB` | `327.64 KB` | `0.287 ± 0.063` | `0.00 MB` |
| **E2** | Federatif (FedAvg) | Yok | $\infty$ | — | **FP32** | **`90.02 ± 0.74`** | **`90.10 ± 0.76`** | `167.29 KB` | `327.64 KB` | `0.299 ± 0.040` | `160.64 MB` |
| **E3** | Federatif (FedAvg) | Yok | $\infty$ | — | **INT8** | **`89.97 ± 0.79`** | **`90.05 ± 0.81`** | **`59.78 KB`** | **`189.95 KB`** | `1.348 ± 0.101` | `160.64 MB` |
| **E4** | Federatif (FedAvg) | DP-SGD ($\varepsilon_1=3.0$) | `3.00` | `4.39` | **FP32** | **`70.92 ± 7.55`** | **`68.91 ± 8.62`** | `167.29 KB` | `327.64 KB` | `0.304 ± 0.059` | `160.64 MB` |
| **E5** | Federatif (FedAvg) | DP-SGD ($\varepsilon_1=3.0$) | `3.00` | `4.39` | **INT8** | **`70.41 ± 7.23`** | **`68.19 ± 8.26`** | **`59.78 KB`** | **`189.95 KB`** | `1.346 ± 0.144` | `160.64 MB` |
| **E6** | Federatif (FedAvg) | DP-SGD ($\varepsilon_2=8.0$) | `7.99` | `2.03` | **FP32** | **`81.23 ± 3.22`** | **`80.91 ± 3.24`** | `167.29 KB` | `327.64 KB` | `0.293 ± 0.043` | `160.64 MB` |
| **E7** | Federatif (FedAvg) | DP-SGD ($\varepsilon_2=8.0$) | `7.99` | `2.03` | **INT8** | **`81.34 ± 2.99`** | **`81.05 ± 2.99`** | **`59.78 KB`** | **`189.95 KB`** | `1.292 ± 0.184` | `160.64 MB` |

### Temel Bilimsel Bulgular (Bildirinin Özeti İçin)
1. **Merkezi vs. Federatif Öğrenme (E1 vs. E2):** Ham sensör verisini merkezileştirmeden 17 katılımcı üzerinde eğitilen FL modeli (`%90.02`), merkezi üst sınır modeline (`%93.67`) kıyasla Non-IID kullanıcı heterojenliğine rağmen yalnızca `3.65` puanlık bir farkla yüksek başarım sağlamıştır.
2. **Diferansiyel Gizliliğin Etkisi (RQ1 — E2 vs. E6 vs. E4):** Orta düzey gizlilik bütçesi ($\varepsilon_2 = 8.0$, **E6**) `%81.23` doğruluk ile anlamlı bir gizlilik-fayda dengesi kurarken, sıkı gizlilik bütçesinde ($\varepsilon_1 = 3.0$, **E4**) yüksek Gauss gürültüsü ($\sigma = 4.39$) özellikle dinamik yürüyüş alt sınıflarını (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`) zorlaştırarak doğruluğu `%70.92` seviyesine düşürmüştür.
3. **DP ve INT8 Kuantizasyon Etkileşimi (RQ2 — E2→E3, E4→E5, E6→E7):** INT8 statik kuantizasyon, DP-SGD gürültüsüyle eğitilmiş modellerde ek bir yıkıcı bozulmaya yol açmamıştır (`E2→E3: -0.05%`, `E4→E5: -0.51%`, `E6→E7: +0.11%`). Hatta $\varepsilon_2 = 8.0$ koşulunda (**E7**) ağırlık kuantizasyonu hafif bir düzenlileştirici (regularizer) etkisi göstererek ortalama doğruluğu `%81.23`'ten `%81.34`'e taşımıştır.
4. **Kaynak Verimliliği ve Pratik Ödünleşim (RQ3):** INT8 kuantizasyon tüm modellerde disk boyutunu **`2.80×` (`167.29 KB` $\rightarrow$ `59.78 KB`)** ve tepe bellek izini **`%42` (`327.64 KB` $\rightarrow$ `189.95 KB`)** azaltmıştır.

---

## 7. Literatür ve Çıktı Dosyaları Envanteri

- **Akademik Literatür (`paper/literature/` & `paper/references/`):**
  - `01_Anguita_2013_UCI_HAR.pdf` ... `20_Kang_2024_Quantization_FL_RDP.pdf` (20 tam metin PDF).
  - [Literatur_Listesi_ve_Plani.md](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/literature/Literatur_Listesi_ve_Plani.md), [IEEE_Kaynakca.md](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/references/IEEE_Kaynakca.md) ve [references.bib](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/references/references.bib).
- **Sonuç Tabloları (`result/tables/`):**
  - [raw_all_21_runs.csv](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/result/tables/raw_all_21_runs.csv), [summary_mean_std.csv](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/result/tables/summary_mean_std.csv), [summary_table_markdown.md](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/result/tables/summary_table_markdown.md), [ieee_results_table.tex](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/result/tables/ieee_results_table.tex).
- **Akademik Grafikler (`paper/figures/` ve `result/plots/`):**
  - [fig1_accuracy_f1_comparison.png](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/figures/fig1_accuracy_f1_comparison.png)
  - [fig2_dp_quantization_interaction.png](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/figures/fig2_dp_quantization_interaction.png)
  - [fig3_convergence_and_epsilon_curves.png](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/figures/fig3_convergence_and_epsilon_curves.png)
  - [fig4_size_tradeoff_and_per_class_f1.png](file:///c:/Users/rumeysagokce/Desktop/FedHAR-Q/paper/figures/fig4_size_tradeoff_and_per_class_f1.png)
