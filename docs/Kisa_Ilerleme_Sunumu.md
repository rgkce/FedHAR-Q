# FedHAR-Q — Kısa İlerleme Sunumu (Temel Deneyler ve Bulgular)

**Çalışma Başlığı:** Federated İnsan Aktivite Tanımada Diferansiyel Gizlilik ve Kuantizasyonun Birlikte Etkisi  
**İngilizce Başlık:** *Evaluating the Joint Impact of Differential Privacy and Quantization on Federated Human Activity Recognition*

*(Not: Bu sunumun hem Markdown versiyonu hem de grafik ve tabloların doğrudan gömülü olduğu PowerPoint **`docs/FedHAR_Q_Kisa_Ilerleme_Sunumu.pptx`** dosyası hazırlanmıştır.)*

---

## Slayt 1: Problem, Amaç ve Araştırma Soruları
- **Problem:** Akıllı telefon sensörleriyle İnsan Aktivite Tanıma (HAR) yüksek kişisel gizlilik riski taşır; ayrıca uç cihazlarda (edge) çalışacak modellerin düşük bellek ve hızlı çıkarım sunması gerekir.
- **Amaç:** Yeni bir algoritma önermek yerine; **Federated Learning (FL)**, **Differential Privacy (DP-SGD)** ve **INT8 Model Kuantizasyonu** yöntemlerinin birlikte kullanımının başarım ve kaynak tüketimi üzerindeki etkileşimini kontrollü deneylerle ölçmek.
- **Araştırma Soruları:**
  1. **RQ1:** FL tabanlı bir HAR modelinde DP ($\varepsilon_1=3.0, \varepsilon_2=8.0$) eklenmesi tanıma başarısını nasıl etkiler?
  2. **RQ2:** `FP32` $\rightarrow$ `INT8` kuantizasyonun etkisi, DP bulunan ve bulunmayan modellerde farklı mıdır?
  3. **RQ3:** Doğruluk, model boyutu (KB) ve çıkarım gecikmesi (ms) arasında nasıl bir pratik denge oluşur?

---

## Slayt 2: Veri Protokolü (Sıfır Veri Sızıntısı) ve 1D-CNN Mimarisi
- **Veri Kümesi (UCI HAR):** 9 kanallı ham eylemsizlik sinyalleri (`body_acc`, `body_gyro`, `total_acc`), 128 zaman adımı (2.56 sn @ 50 Hz), 6 aktivite sınıfı, toplam 30 katılımcı (10,299 pencere).
- **Kişi Bazlı (Subject-Wise) Ayrım ($\text{Kesişim} = \emptyset$):**
  - **Eğitim:** **17 Kişi** (`5,879` pencere) $\rightarrow$ Federatif öğrenmede her kişi bağımsız 1 istemci (`17 Client`, doğal Non-IID).
  - **Doğrulama (Validation):** **4 Kişi** (`1,473` pencere) $\rightarrow$ Model/tur seçimi yalnızca doğrulama kümesinde yapılır.
  - **Test:** **9 Kişi** (`2,947` pencere) $\rightarrow$ Yalnızca nihai değerlendirmede kullanılır.
- **Hafif 1D-CNN Mimarisi (`FedHAR1DCNN`):**
  - Opacus (DP-SGD) uyumluluğu için `BatchNorm` yerine örnek-içi `GroupNorm(8, C)` kullanılmıştır (tüm koşullarda birebir aynı mimari).
  - 3 Evrişim Bloğu (`32 -> 64 -> 64`) + 2 Tam Bağlantılı Katman (`256 -> 64 -> 6`).
  - Toplam **`41,286` parametre** (`167.29 KB` FP32).

---

## Slayt 3: Deneysel Kurulum ve E1–E7 Deney Matrisi
- **Yazılım Altyapısı:** `PyTorch 2.12`, `Flower (flwr) 1.43`, `Opacus 1.6`, `RDPAccountant` ($\delta = 10^{-4}$, $C = 2.0$).
- **FL Protokolü:** `FedAvg`, $K=17$ istemci, $R=30$ iletişim turu, tur başına $E=2$ yerel epoch.
- **Tekrarlanabilirlik:** Tüm 7 deney koşulu **3 farklı rastgele tohumla (`seeds: [42, 123, 999]`, toplam 21 deney koşusu)** çalıştırılmıştır:
  - **E1:** Merkezi Eğitim | DP Yok | `FP32`
  - **E2 / E3:** Federatif Eğitim | DP Yok | `FP32` (**E2**) ve `INT8` (**E3**)
  - **E4 / E5:** Federatif Eğitim | Sıkı DP ($\varepsilon_1 = 3.00, \sigma = 4.39$) | `FP32` (**E4**) ve `INT8` (**E5**)
  - **E6 / E7:** Federatif Eğitim | Esnek DP ($\varepsilon_2 = 8.00, \sigma = 2.03$) | `FP32` (**E6**) ve `INT8` (**E7**)

---

## Slayt 4: Temel Deneylerin Tamamlandığını Gösteren Ana Sonuç Tablosu (`3 Tohum Ortalama ± Std`)

| Kod | Eğitim Modu | Gizlilik (DP) | Harcanan $\varepsilon$ ($\delta=10^{-4}$) | $\sigma$ | Temsil | **Test Accuracy (%)** | **Test Macro-F1 (%)** | **Model Boyutu (KB)** | **Tepe Bellek (KB)** | **Gecikme (ms)** | **FL İletişim (MB)** |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Merkezi | Yok | $\infty$ | — | **FP32** | **`93.67 ± 0.65`** | **`93.62 ± 0.69`** | `167.29 KB` | `327.64 KB` | `0.287 ± 0.063` | `0.00 MB` |
| **E2** | Federatif | Yok | $\infty$ | — | **FP32** | **`90.02 ± 0.74`** | **`90.10 ± 0.76`** | `167.29 KB` | `327.64 KB` | `0.299 ± 0.040` | `160.64 MB` |
| **E3** | Federatif | Yok | $\infty$ | — | **INT8** | **`89.97 ± 0.79`** | **`90.05 ± 0.81`** | **`59.78 KB`** | **`189.95 KB`** | `1.348 ± 0.101` | `160.64 MB` |
| **E4** | Federatif | $\varepsilon_1 = 3.0$ | `3.00` | `4.39` | **FP32** | **`70.92 ± 7.55`** | **`68.91 ± 8.62`** | `167.29 KB` | `327.64 KB` | `0.304 ± 0.059` | `160.64 MB` |
| **E5** | Federatif | $\varepsilon_1 = 3.0$ | `3.00` | `4.39` | **INT8** | **`70.41 ± 7.23`** | **`68.19 ± 8.26`** | **`59.78 KB`** | **`189.95 KB`** | `1.346 ± 0.144` | `160.64 MB` |
| **E6** | Federatif | $\varepsilon_2 = 8.0$ | `7.99` | `2.03` | **FP32** | **`81.23 ± 3.22`** | **`80.91 ± 3.24`** | `167.29 KB` | `327.64 KB` | `0.293 ± 0.043` | `160.64 MB` |
| **E7** | Federatif | $\varepsilon_2 = 8.0$ | `7.99` | `2.03` | **INT8** | **`81.34 ± 2.99`** | **`81.05 ± 2.99`** | **`59.78 KB`** | **`189.95 KB`** | `1.292 ± 0.184` | `160.64 MB` |

---

## Slayt 5: Deney Grafikleri — Başarım ve DP × INT8 Etkileşimi
- **Şekil 1:** [E1–E7 Test Accuracy ve Macro-F1 Karşılaştırması](../paper/figures/fig1_accuracy_f1_comparison.png)
- **Şekil 2:** [Gizlilik Rejimi × INT8 Kuantizasyon Etkileşim Analizi](../paper/figures/fig2_dp_quantization_interaction.png)
- **Temel Bulgu (RQ1 & RQ2):**
  - `FP32` ve `INT8` eğrileri tüm gizlilik seviyelerinde (`DP Yok`, $\varepsilon_2=8.0$, $\varepsilon_1=3.0$) üst üste binmektedir (`ΔAcc < %0.55`).
  - DP-SGD gürültüsü ile eğitilmiş modellerde INT8 statik kuantizasyon ek bir doğruluk çöküşüne yol açmamakta; hatta $\varepsilon_2=8.0$ (**E7**) koşulunda hafif düzenlileştirici (regularization) etkisiyle doğruluğu `%81.23`'ten `%81.34`'e taşımaktadır.

---

## Slayt 6: Deney Grafikleri — Yakınsama, Gizlilik Bütçesi ve Kaynak Ödünleşimi
- **Şekil 3:** [Federatif Yakınsama ve RDPAccountant Bütçe Harcama Eğrileri](../paper/figures/fig3_convergence_and_epsilon_curves.png)
- **Şekil 4:** [Model Boyutu–Doğruluk Ödünleşimi ve Sınıf Bazlı F1 Isı Haritası](../paper/figures/fig4_size_tradeoff_and_per_class_f1.png)
- **Temel Bulgu (RQ3):**
  - INT8 kuantizasyon tüm modellerde disk boyutunu **`167.29 KB`'dan `59.78 KB`'a (`2.80×` küçülme)** ve tepe çıkarım belleğini **`327.64 KB`'dan `189.95 KB`'a (`%42` azalma)** indirmektedir.
  - **E7 ($\varepsilon_2=8.0$ + INT8)** koşulu, formal diferansiyel gizlilik garantisi altında `%81.34` doğruluk ve `59.78 KB` model boyutuyla en dengeli uç cihaz profili olarak öne çıkmaktadır.
