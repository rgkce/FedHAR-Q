# FedHAR-Q — Kodlama Yol Haritası, Mimari ve Takip Dokümanı

**Proje Adı:** FedHAR-Q (*Federated Human Activity Recognition with Differential Privacy and Quantization*)  
**Türkçe Başlık:** Federated İnsan Aktivite Tanımada Diferansiyel Gizlilik ve Kuantizasyonun Birlikte Etkisi  
**İngilizce Başlık:** *Evaluating the Joint Impact of Differential Privacy and Quantization on Federated Human Activity Recognition*

Bu doküman, `Bildiri_Gorev_Dokumani.docx` ve `Federated_HAR_1_Aylik_Calisma_Plani.md` dosyalarındaki tüm teknik isterleri karşılamak üzere oluşturulmuş **uygulama mimarisini**, **kodlama adımlarını** ve **takip kontrol listesini (checklist)** içerir.

---

## 1. Araştırma Soruları ve Zorunlu Kapsam Özeti

### 1.1. Araştırma Soruları (RQ)
1. **RQ1 (Gizlilik Etkisi):** FL tabanlı bir HAR modeline Diferansiyel Gizlilik (DP) eklenmesi tanıma başarısını nasıl etkiler?
2. **RQ2 (Kuantizasyon × DP Etkileşimi):** INT8 kuantizasyonun etkisi, DP bulunan ($\varepsilon_1, \varepsilon_2$) ve bulunmayan modellerde farklı mıdır?
3. **RQ3 (Pratik Ödünleşim / Trade-off):** Doğruluk (Accuracy / Macro-F1), model boyutu (KB) ve çıkarım gecikmesi (ms) arasında nasıl bir pratik denge oluşur?

### 1.2. Zorunlu Teknik Kurallar
- **Veri Sızıntısını Önleme:** UCI HAR veri kümesinde **kişi (subject) bazlı** eğitim/doğrulama/test ayrımı korunacaktır. Model veya hiperparametre seçimi kesinlikle test kümesi üzerinde yapılmayacaktır.
- **Model:** Hem Opacus (DP) hem de INT8 kuantizasyon ile uyumlu **tek bir hafif 1D-CNN** mimarisi kullanılacaktır.
- **Gizlilik Muhasebesi:** DP-SGD uygulamasında gizlilik birimi (örnek düzeyi vs. istemci düzeyi), $\varepsilon_1, \varepsilon_2$ hedefleri, $\delta$ değeri ve kullanılan *privacy accountant* (ör. RDPAccountant) açıkça belgelenecektir.
- **Tekrarlanabilirlik:** Tüm deney matrisi (E1–E7) aynı test protokolüyle ve **en az 3 farklı rastgele tohumla (seed: ör. 42, 123, 999)** tekrarlanacaktır.

---

## 2. Güncellenmiş Proje Klasör Yapısı

Klasör yapısı **Kod (`code/`)**, **Veri (`dataset/`)**, **Makale (`paper/`)**, **Sonuçlar (`result/`)** ve **Testler (`tests/`)** olarak modüler biçimde düzenlenmiştir:

```text
FedHAR-Q/
├── code/                           # Tüm çalıştırılabilir Python kaynak kodları
│   ├── __init__.py
│   ├── configs/                    # Deney ve hiperparametre YAML konfigürasyonları
│   │   ├── default.yaml            # Ortak veri, model ve tohum (seed) ayarları
│   │   ├── centralized.yaml        # E1 merkezi eğitim ayarları
│   │   ├── federated.yaml          # E2-E7 Flower FedAvg ayarları
│   │   ├── privacy.yaml            # Opacus DP (epsilon_1, epsilon_2, delta, max_grad_norm)
│   │   └── quantization.yaml       # INT8 kuantizasyon ayarları
│   ├── data/                       # Veri indirme, ön işleme ve istemci bölünmesi
│   │   ├── __init__.py
│   │   ├── download_uci_har.py     # UCI HAR veri kümesini otomatik indirme ve doğrulama
│   │   ├── preprocessing.py        # Kanal normalizasyonu (yalnızca train üzerinden) ve pencereleme
│   │   ├── dataset.py              # PyTorch Dataset ve DataLoader sınıfları
│   │   └── partition.py            # Kişi (subject) bazlı train/val ayrımı ve FL istemci ataması
│   ├── models/                     # Sinir ağı mimarisi
│   │   ├── __init__.py
│   │   └── cnn1d.py                # Opacus ve Quantization uyumlu hafif 1D-CNN modeli
│   ├── federated/                  # Flower FL istemci ve sunucu yapısı
│   │   ├── __init__.py
│   │   ├── client.py               # Flower NumPyClient (yerel eğitim ve DP entegrasyonu)
│   │   ├── server.py               # FedAvg stratejisi, küresel değerlendirme ve iletişim yükü hesabı
│   │   └── simulation.py           # FL simülasyon başlatıcı
│   ├── privacy/                    # Diferansiyel gizlilik modülü
│   │   ├── __init__.py
│   │   └── dp_engine.py            # Opacus PrivacyEngine sarmalayıcı, RDP muhasebesi ve doğrulama
│   ├── quantization/               # Model sıkıştırma ve INT8 dönüşümü
│   │   ├── __init__.py
│   │   └── quantize.py             # FP32 -> INT8 Post-Training Static Quantization / ONNX Runtime
│   ├── evaluation/                 # Doğruluk ve sistem/kaynak metrikleri
│   │   ├── __init__.py
│   │   ├── metrics.py              # Accuracy, Macro-F1, Confusion Matrix
│   │   └── benchmark.py            # Model boyutu (KB), çıkarım gecikmesi (ms), tepe bellek (MB)
│   └── experiments/                # Deney orkestrasyonu ve rapor üretimi
│       ├── __init__.py
│       ├── train_centralized.py    # Merkezi eğitim (E1) koşucusu
│       ├── run_matrix.py           # E1-E7 matrisini 3 tohumla çalıştıran ana betik
│       └── export_results.py       # Sonuçları tablo (CSV/LaTeX) ve grafiklere dönüştüren betik
├── dataset/                        # Veri dosyaları (Git'e dahil edilmez)
│   ├── raw/                        # Ham UCI HAR zip ve açılmış klasörü
│   └── processed/                  # Kişi bazlı ayrılmış ve normalize edilmiş tensörler
├── paper/                          # IEEE Konferans Bildirisi dosyaları
│   ├── figures/                    # Bildiriye eklenecek mimari ve sonuç şekilleri
│   ├── literature/                 # İncelenen 15-20 makale ve literatür matrisi
│   ├── manuscript/                 # Bildiri metni (.tex veya .docx)
│   └── references/                 # Kaynakça dosyası (.bib)
├── result/                         # Deney çıktıları
│   ├── logs/                       # Eğitim ve DP epsilon takip logları (.json / .csv)
│   ├── models/                     # Kaydedilen FP32 ve INT8 model ağırlıkları (.pt / .onnx)
│   ├── plots/                      # Otomatik üretilen karşılaştırma grafikleri (.png / .pdf)
│   └── tables/                     # Ortalama ± standart sapma sonuç tabloları (.csv / .tex)
├── docs/                           # Görev, çalışma planı ve yol haritası dokümanları
│   ├── Bildiri_Gorev_Dokumani.docx
│   ├── Federated_HAR_1_Aylik_Calisma_Plani.md
│   └── Kodlama_Yol_Haritasi_ve_Mimari.md
├── tests/                          # Birim ve entegrasyon testleri (veri sızıntısı, DP, INT8 testi)
├── requirements.txt
└── .gitignore
```

---

## 3. Deney Matrisi (Çekirdek Koşullar)

Tüm koşullar en az **3 farklı rastgele tohumla (`seeds: [42, 123, 999]`)** çalıştırılacak ve aynı kişi-bazlı test kümesi üzerinde değerlendirilecektir:

| Deney Kodu | Eğitim Türü | Diferansiyel Gizlilik (DP) | Model Temsili | Açıklama |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Merkezi (Centralized) | Yok ($\infty$) | FP32 | Üst sınır (upper-bound) referans modeli |
| **E2** | Federatif (FL - FedAvg) | Yok ($\infty$) | FP32 | Gizliliksiz federatif baseline |
| **E3** | Federatif (FL - FedAvg) | Yok ($\infty$) | **INT8** | E2 modelinin INT8 kuantize edilmiş hali |
| **E4** | Federatif (FL - FedAvg) | **$\varepsilon_1$ Hedefi** (Yüksek/Orta Gizlilik) | FP32 | DP-SGD ile eğitilmiş federatif FP32 model |
| **E5** | Federatif (FL - FedAvg) | **$\varepsilon_1$ Hedefi** (Yüksek/Orta Gizlilik) | **INT8** | E4 modelinin INT8 kuantize edilmiş hali |
| **E6** | Federatif (FL - FedAvg) | **$\varepsilon_2$ Hedefi** (Daha Gevşek Gizlilik) | FP32 | DP-SGD ile eğitilmiş federatif FP32 model |
| **E7** | Federatif (FL - FedAvg) | **$\varepsilon_2$ Hedefi** (Daha Gevşek Gizlilik) | **INT8** | E6 modelinin INT8 kuantize edilmiş hali |
| *E1-Q (Ops.)* | *Merkezi (Centralized)* | *Yok ($\infty$)* | *INT8* | *Merkezi modelde kuantizasyon etkisi kontrolü* |

---

## 4. Modül Bazlı Kodlama Detayları (Neler Yapacağız?)

### Aşama 1: Veri Hattı ve Kişi Bazlı Ayrım (`code/data/`)
1. **Veri Formatı:** UCI HAR veri kümesindeki 9 kanallı ham eylemsizlik sinyalleri (`body_acc_x/y/z`, `body_gyro_x/y/z`, `total_acc_x/y/z`) kullanılacaktır. Her örnek `(9, 128)` boyutundadır (9 kanal, 128 zaman adımı) ve 6 aktivite sınıfından (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) birine aittir.
2. **Kişi Bazlı Ayrım (Zero Data Leakage):**
   - UCI HAR resmî ayrımında 30 katılımcının **21'i eğitim**, **9'u test** kümesindedir.
   - Eğitimdeki 21 katılımcının örneğin 17'si eğitim, 4'ü doğrulama (validation) olarak kişi bazında ayrılacaktır.
   - Kanal bazlı ortalama ve standart sapma ($\mu, \sigma$) **yalnızca eğitim katılımcılarının verisinden** hesaplanacaktır.
3. **Federatif İstemci Dağılımı:**
   - Doğal Non-IID senaryo: Her katılımcı (subject) kendi verisine sahip 1 istemci (client) olarak modellenir (veya yapılandırılabilir istemci sayısına kişi bazlı atanır).

### Aşama 2: Hafif 1D-CNN Mimarisi (`code/models/cnn1d.py`)
1. **Mimari Tasarım:**
   - Girdi: `(Batch, 9, 128)`
   - Evrişim Blokları: `Conv1d -> GroupNorm (veya normsuz) -> ReLU -> MaxPool1d`
   - Sınıflandırıcı: `AdaptiveAvgPool1d -> Flatten -> Linear -> ReLU -> Dropout -> Linear(num_classes=6)`
2. **Neden `BatchNorm1d` Kullanmıyoruz?**
   - Opacus ile Diferansiyel Gizlilik uygulanırken örnek başına gradyan (per-sample gradient) hesaplanır. `BatchNorm` aynı yığındaki (batch) örnekleri birbirine bağladığı için DP garantisini bozar. Bu nedenle en baştan `GroupNorm` kullanarak tüm koşullarda (E1–E7) **birebir aynı mimariyi** koruyacağız.
3. **Kuantizasyon Uyumluluğu:**
   - PyTorch statik kuantizasyonu ve ONNX dışa aktarımı için modelde `QuantStub` ve `DeQuantStub` giriş/çıkış kapıları hazır bulunacaktır.

### Aşama 3: Merkezi ve Federatif Baseline (`code/federated/` & `code/experiments/`)
1. **Merkezi Eğitim (`train_centralized.py` - E1):**
   - 3 tohum için modeli eğitir, en iyi doğrulama (validation) modelini seçer ve test kümesi üzerinde değerlendirir.
2. **Flower Federatif Eğitim (`client.py`, `server.py` - E2):**
   - `FedAvg` algoritması ile $R$ tur (round) boyunca istemci-sunucu simülasyonu yürütülür.
   - Her turda istemciden sunucuya ve sunucudan istemciye aktarılan parametre boyutu (MB cinsinden iletişim yükü) kaydedilir.

### Aşama 4: Diferansiyel Gizlilik Entegrasyonu (`code/privacy/dp_engine.py`)
1. **Opacus Entegrasyonu (E4, E6):**
   - İstemci yerel eğitiminde `PrivacyEngine` devreye girer.
   - Gradyan kırpma eşiği (`max_grad_norm`, ör. `1.0`) ve gürültü çarpanı (`noise_multiplier`) veya hedef bütçe (`make_private_with_epsilon`) kullanılır.
2. **Belgelenecek Parametreler:**
   - Gizlilik birimi (örnek düzeyi / sample-level DP-SGD),
   - Hedef $\varepsilon_1$ (ör. `3.0`) ve $\varepsilon_2$ (ör. `8.0`),
   - $\delta$ değeri ($\delta < 1 / N_{\text{train}}$ prensibiyle ör. `1e-4`),
   - *Accountant* mekanizması (`RDPAccountant`).

### Aşama 5: INT8 Kuantizasyon (`code/quantization/quantize.py`)
1. **Post-Training Static Quantization (PTQ) (E3, E5, E7):**
   - Eğitilmiş FP32 modelleri (E2, E4, E6) alınır.
   - Eğitim kümesinden küçük bir kalibrasyon alt kümesi ile aktivasyon istatistikleri toplanır ve ağırlıklar + aktivasyonlar **INT8** formatına dönüştürülür (PyTorch `qnnpack`/`fbgemm` veya `ONNX Runtime` INT8).
2. **Karşılaştırma:**
   - Kuantizasyon öncesi (FP32) ve sonrası (INT8) modeller aynı test protokolüyle değerlendirilerek DP'li modellerin kuantizasyon gürültüsüne daha duyarlı olup olmadığı (RQ2) analiz edilir.

### Aşama 6: Ölçümler ve Raporlama (`code/evaluation/`)
Her deney koşusu sonunda şu metrikler otomatik olarak `result/tables/` altına kaydedilir:
- **Sınıflandırma:** Accuracy (%), Macro-F1 (%), Sınıf bazlı F1 ve Confusion Matrix.
- **Model Boyutu:** Disk üzerindeki serileştirilmiş model boyutu (KB).
- **Çıkarım Gecikmesi (Latency):** Tekil örnek (`batch_size=1`) ve yığın (`batch_size=64`) için ısınma (warm-up) sonrası ortalama ve medyan çıkarım süresi (ms).
- **Tepe Bellek (Peak Memory):** Çıkarım sırasında kullanılan bellek (KB/MB).
- **İletişim Yükü:** FL eğitimi boyunca toplam veri transferi (MB).

---

## 5. Haftalık İlerleme ve Takip Kontrol Listesi (Checklist)

Çalışma boyunca tamamlanan adımları buradan işaretleyerek (`[x]`) takip edeceğiz:

### 1. Hafta: Ortam, Veri Hattı ve Model İskeleti
- [x] Proje klasör yapısının oluşturulması ve düzenlenmesi (`code`, `dataset`, `paper`, `result`, `tests`).
- [x] `requirements.txt` ve `.gitignore` dosyalarının hazırlanması.
- [x] Python sanal ortamının kurulması ve paketlerin (`torch`, `flwr`, `opacus`, `onnxruntime`) doğrulanması.
- [x] `code/data/download_uci_har.py`: UCI HAR veri kümesinin indirilmesi.
- [x] `code/data/partition.py` & `dataset.py`: Kişi bazlı (subject-wise) eğitim/doğrulama/test ayrımının kodlanması ve sızıntı olmadığının birim testiyle (`tests/test_data_leakage.py`) doğrulanması.
- [x] `code/models/cnn1d.py`: Opacus ve INT8 uyumlu hafif 1D-CNN modelinin yazılması ve ileri yayılım (forward pass) testi.

### 2. Hafta: Merkezi (E1) ve Federatif (E2) Baseline Modelleri
- [x] `code/configs/`: Deney konfigürasyon dosyalarının hazırlanması.
- [x] `code/evaluation/metrics.py` & `benchmark.py`: Accuracy, Macro-F1, model boyutu ve gecikme ölçüm fonksiyonlarının yazılması.
- [x] `code/experiments/train_centralized.py`: Merkezi FP32 baseline (E1) eğitiminin tamamlanması.
- [x] `code/federated/client.py` & `server.py`: Flower FedAvg FP32 baseline (E2) simülasyonunun kurulması ve iletişim yükünün ölçülmesi.

### 3. Hafta: Diferansiyel Gizlilik (E4, E6) ve INT8 Kuantizasyon (E3, E5, E7)
- [x] `code/privacy/dp_engine.py`: Opacus DP-SGD entegrasyonu ve $\varepsilon_1, \varepsilon_2, \delta$ muhasebesinin doğrulanması.
- [x] E4 ($\varepsilon_1$, FP32) ve E6 ($\varepsilon_2$, FP32) federatif eğitimlerinin çalıştırılması.
- [x] `code/quantization/quantize.py`: FP32 modellerin INT8 formatına kuantize edilmesi (E3, E5, E7).
- [x] DP × INT8 birleşik koşullarının ilk doğruluk ve gecikme testlerinin yapılması.

### 4. Hafta: Çoklu Tohum (3 Seed) Tekrarları, Tablolar ve Bildiri Entegrasyonu
- [x] `code/experiments/run_matrix.py`: E1–E7 tüm deney matrisinin 3 farklı rastgele tohumla (`42, 123, 999`) koşturulması.
- [x] `code/experiments/export_results.py`: Ortalama ± standart sapma tablolarının (`result/tables/`) ve karşılaştırma grafiklerinin (`result/plots/`) üretilmesi.
- [ ] Sonuçların `paper/manuscript/` altındaki IEEE bildiri taslağına aktarılması.
