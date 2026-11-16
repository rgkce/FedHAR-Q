# FedHAR-Q: Federated İnsan Aktivite Tanımada Diferansiyel Gizlilik ve Kuantizasyonun Birlikte Etkisi

> **İngilizce Başlık:** *Evaluating the Joint Impact of Differential Privacy and Quantization on Federated Human Activity Recognition*

---

## 1. Çalışmanın Tanımı ve Motivasyonu

Akıllı telefonlar ve giyilebilir sensörlerle yürütülen **İnsan Aktivite Tanıma (Human Activity Recognition — HAR)** sistemleri; sağlık takibi, yaşlı bakımı ve akıllı ortam uygulamalarının temel bileşenlerinden biridir. Ancak ivmeölçer ve jiroskop gibi eylemsizlik sensörlerinden sürekli toplanan zaman serisi verileri; kullanıcının günlük rutinlerini, fiziksel alışkanlıklarını ve biyometrik yürüyüş imzasını barındırdığı için ciddi **kişisel veri gizliliği** riskleri taşır.

Bu problemi çözmek ve modelleri kaynak kısıtlı uç cihazlarda (akıllı telefonlar, gömülü sistemler, Raspberry Pi vb.) verimli biçimde çalıştırabilmek için üç temel yaklaşım bir arada ele alınmalıdır:
1. **Federated Learning (FL):** Ham sensör verisini merkezi bir sunucuya göndermeden, her kullanıcının kendi cihazında yerel eğitim yapması ve yalnızca model ağırlık güncellemelerinin (`FedAvg`) paylaşılması.
2. **Differential Privacy (DP):** Paylaşılan ağırlık güncellemelerinden veya eğitilmiş modelden bireysel sensör pencerelerinin çıkarımlanmasını önlemek amacıyla yerel gradyanların kırpılması ve kontrollü Gauss gürültüsü eklenmesi (`DP-SGD`).
3. **Model Kuantizasyonu (INT8 Quantization):** Eğitilen 32-bit kayan noktalı (`FP32`) küresel modelin uç cihazlarda düşük bellek izi ve hızlı çıkarım (inference) sağlaması için 8-bit tam sayı (`INT8`) temsile sıkıştırılması.

**FedHAR-Q** projesi; yeni bir algoritma önermek yerine, **Federated Learning, Diferansiyel Gizlilik ve INT8 Model Kuantizasyonu** üçlüsünün birlikte kullanıldığında birbirleriyle nasıl etkileşime girdiğini kontrollü, sıfır veri sızıntılı ve tekrarlanabilir deneylerle ortaya koyan bilimsel bir inceleme çalışmasıdır.

---

## 2. Araştırma Soruları

Çalışma üç temel araştırma sorusuna (RQ) odaklanmaktadır:

* **RQ1 (Diferansiyel Gizliliğin Etkisi):** Federatif öğrenme tabanlı bir HAR modeline farklı gizlilik bütçelerinde ($\varepsilon_1 = 3.0$ sıkı gizlilik ve $\varepsilon_2 = 8.0$ esnek gizlilik) Diferansiyel Gizlilik eklenmesi aktivite tanıma başarımını nasıl etkiler?
* **RQ2 (Gizlilik × Kuantizasyon Etkileşimi):** Eğitim sonrası uygulanan `FP32` $\rightarrow$ `INT8` statik kuantizasyonun doğruluk üzerindeki etkisi, DP gürültüsüyle eğitilmiş modeller ile gizlilik uygulanmayan modeller arasında farklılık gösterir mi?
* **RQ3 (Doğruluk–Kaynak Ödünleşimi):** Uç cihaz dağıtımı perspektifinden sınıflandırma doğruluğu (*Accuracy / Macro-F1*), model boyutu (*KB*), tepe bellek izi (*KB*) ve çıkarım gecikmesi (*ms*) arasında nasıl bir pratik denge oluşur?

---

## 3. Yöntem ve Teknik Tasarım

### 3.1. Veri Kümesi ve Sıfır Sızıntılı Kişi Bazlı (Subject-Wise) Protokol
Çalışmada 30 katılımcının 6 farklı günlük aktivitesini (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) içeren **UCI HAR** veri kümesinin 9 kanallı ham eylemsizlik sinyalleri (`(9, 128)` boyutlu pencereler, 50 Hz, 2.56 sn) kullanılmıştır:
* **Kişi Bazlı Ayrım:** Aynı kişiye ait pencerelerin hem eğitim hem test kümesine düşerek yapay doğruluk artışına (veri sızıntısına) yol açmasını engellemek için 30 katılımcı kişi bazında ayrılmıştır:
  * **Eğitim Kümesi (17 Kişi — 5,879 pencere):** Merkezi eğitimde (E1) eğitim verisi olarak; federatif öğrenmede (E2–E7) ise her katılımcı bağımsız birer istemci (`client_id = 0..16`, doğal kullanıcı bazlı Non-IID dağılım) olacak şekilde kullanılır.
  * **Doğrulama Kümesi (4 Kişi — 1,473 pencere):** Eğitim sırasında en iyi model ağırlığının (epoch/round) seçilmesi için kullanılır.
  * **Test Kümesi (9 Kişi — 2,947 pencere):** Eğitim veya model seçimi aşamalarında hiçbir şekilde dokunulmaz; yalnızca nihai başarım ölçümünde kullanılır.
* **Normalizasyon:** Kanal bazlı ortalama ve standart sapma ($\mu, \sigma$) yalnızca 17 eğitim katılımcısının verisinden hesaplanmıştır.

### 3.2. Opacus ve Kuantizasyon Uyumlu Hafif 1D-CNN Mimarisi (`FedHAR1DCNN`)
Derin öğrenme modeli olarak 3 evrişim bloğu ve 2 tam bağlantılı katmandan oluşan toplam **`41,286` parametreli** (`167.29 KB` FP32) hafif bir 1D-CNN tasarlanmıştır:
* **Neden `GroupNorm`?** Standart `BatchNorm1d` katmanı bir mini-batch içindeki farklı örneklerin istatistiklerini birbirine karıştırdığı için Diferansiyel Gizlilikte örnek-bazlı gradyan (`per-sample gradient`) hesabını bozar. Bu nedenle mimaride örnek-içi normalizasyon yapan `GroupNorm(num_groups=8)` tercih edilmiş; böylece gizliliksiz ve gizlilikli tüm deney koşullarında (E1–E7) **birebir aynı model mimarisi** kullanılmıştır.

### 3.3. Federatif Öğrenme, Rényi DP Muhasebesi ve INT8 Statik Kuantizasyon
* **Federatif Öğrenme (`Flower + FedAvg`):** 17 istemci üzerinde 30 iletişim turu (`round`) ve tur başına 2 yerel epoch ile yürütülür.
* **Diferansiyel Gizlilik (`Opacus DP-SGD`):** İstemci yerel eğitiminde örnek düzeyi (*sample-level*) gradyan kırpma ($C = 2.0$) ve Gauss gürültüsü uygulanır. Gizlilik bütçesi **Rényi Diferansiyel Gizlilik (`RDPAccountant`, $\delta = 10^{-4}$)** ile 30 tur boyunca kümülatif olarak izlenir.
* **INT8 Kuantizasyon (`Post-Training Static Quantization - PTQ`):** Eğitilmiş `FP32` modeller, yalnızca eğitim kümesinden alınan kalibrasyon verisiyle `INT8` tam sayı formatına dönüştürülür (`167.29 KB` $\rightarrow$ `59.78 KB`, **2.80× sıkıştırma**).

---

## 4. Deney Matrisi (E1–E7)

Tüm deney koşulları **3 farklı rastgele tohumla (`seeds: [42, 123, 999]`, toplam 21 deney koşusu)** çalıştırılmış ve aynı 9 kişilik test kümesi üzerinde değerlendirilmiştir:

| Deney Kodu | Eğitim Türü | Diferansiyel Gizlilik | Model Temsili | Deneyin Amacı |
| :---: | :--- | :---: | :---: | :--- |
| **E1** | Merkezi (Centralized) | Yok ($\varepsilon = \infty$) | `FP32` | Merkezi üst sınır (upper-bound) referans modeli |
| **E2** | Federatif (`FedAvg`) | Yok ($\varepsilon = \infty$) | `FP32` | Gizliliksiz federatif baseline modeli |
| **E3** | Federatif (`FedAvg`) | Yok ($\varepsilon = \infty$) | **`INT8`** | Gizliliksiz federatif modelde INT8 kuantizasyon etkisi |
| **E4** | Federatif (`FedAvg`) | Sıkı DP ($\varepsilon_1 = 3.0, \delta=10^{-4}$) | `FP32` | Sıkı gizlilik bütçesinin federatif HAR doğruluğuna etkisi |
| **E5** | Federatif (`FedAvg`) | Sıkı DP ($\varepsilon_1 = 3.0, \delta=10^{-4}$) | **`INT8`** | Sıkı DP altında INT8 kuantizasyon × gizlilik etkileşimi |
| **E6** | Federatif (`FedAvg`) | Esnek DP ($\varepsilon_2 = 8.0, \delta=10^{-4}$) | `FP32` | Orta/esnek gizlilik bütçesinin federatif HAR doğruluğuna etkisi |
| **E7** | Federatif (`FedAvg`) | Esnek DP ($\varepsilon_2 = 8.0, \delta=10^{-4}$) | **`INT8`** | Esnek DP altında INT8 kuantizasyon × gizlilik etkileşimi |

---

## 5. Proje Klasör Yapısı

Proje; **Kaynak Kodlar (`code/`)**, **Veri Kümesi (`dataset/`)**, **Yönlendirme/Rapor Dokümanları (`docs/`)**, **Bildiri (`paper/`)**, **Deney Çıktıları (`result/`)** ve **Birim Testleri (`tests/`)** olmak üzere 6 ana dizinden oluşmaktadır:

```text
FedHAR-Q/
├── code/                           # Çalıştırılabilir modüler Python kaynak kodları
│   ├── configs/                    # Deney ve hiperparametre konfigürasyonu (default.yaml)
│   ├── data/                       # UCI HAR indirme, kişi bazlı ayrım ve FL istemci bölünmesi
│   ├── models/                     # Opacus ve INT8 uyumlu hafif 1D-CNN mimarisi (cnn1d.py)
│   ├── federated/                  # Flower NumPyClient istemcisi ve FedAvg sunucu simülasyonu
│   ├── privacy/                    # Opacus DP-SGD yerel eğitimi ve RDPAccountant muhasebesi
│   ├── quantization/               # FP32 -> INT8 Post-Training Static Quantization modülü
│   ├── evaluation/                 # Accuracy, Macro-F1, boyut, gecikme, bellek ve iletişim ölçümleri
│   └── experiments/                # E1-E7 matris koşucusu, Raspberry Pi ölçümü ve tablo/grafik üretimi
├── dataset/                        # Ham ve işlenmiş UCI HAR veri dosyaları
├── docs/                           # Proje planları, teknik raporlar ve uygulama rehberleri
│   ├── Bildiri_Gorev_Dokumani.docx
│   ├── Federated_HAR_1_Aylik_Calisma_Plani.md
│   ├── Kodlama_Yol_Haritasi_ve_Mimari.md
│   ├── Adim_Adim_Kodlama_Plani.md
│   ├── Proje_Yapilanlar_ve_Kullanilanlar_Raporu.md
│   ├── Raspberry_Pi_Calistirma_Plani.md
│   └── Hizli_Komutlar_ve_Tablo_Ozeti.md
├── paper/                          # IEEE Konferans Bildirisi dosyaları
│   ├── figures/                    # Bildiri için üretilen 300 DPI PNG ve vektörel PDF grafikler
│   ├── literature/                 # İncelenen 20 akademik yayın (PDF) ve literatür planı
│   ├── manuscript/                 # Konferans bildirisi taslağı
│   └── references/                 # IEEE formatında kaynakça (.md ve .bib)
├── result/                         # Deney çıktıları
│   ├── logs/                       # 21 deney koşusunun detaylı JSON eğitim ve DP logları
│   ├── models/                     # Kaydedilen 21 adet FP32 ve INT8 model ağırlığı (.pt)
│   ├── plots/                      # Karşılaştırma ve etkileşim grafikleri
│   └── tables/                     # CSV, Markdown ve IEEE LaTeX sonuç tabloları
├── tests/                          # 17 adet otomatik doğrulama ve birim testi (pytest)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 6. Detaylı Dokümantasyon Bağlantıları

Projenin alt detaylarına, çalıştırma komutlarına ve deneysel bulgularına `docs/` klasöründeki şu belgelerden ulaşabilirsiniz:

* **[Hızlı Komutlar ve Tablo Özeti](docs/Hizli_Komutlar_ve_Tablo_Ozeti.md):** Testleri, deney matrisini ve tablo/grafik üretimini çalıştırma komutları ile özet sonuç tablosu.
