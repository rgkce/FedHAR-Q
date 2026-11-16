# FedHAR-Q: Hızlı Kurulum Komutları ve Sonuç Tablosu Özeti

**Türkçe Başlık:** Federated İnsan Aktivite Tanımada Diferansiyel Gizlilik ve Kuantizasyonun Birlikte Etkisi  
**İngilizce Başlık:** *Evaluating the Joint Impact of Differential Privacy and Quantization on Federated Human Activity Recognition*

Bu doküman, önceki `README.md` içeriğinin taşınmış halidir ve deneyleri hızlıca çalıştırmak için gereken komutlar ile özet sonuç tablosunu içerir.

---

## 1. Hızlı Kurulum ve Çalıştırma

### Bağımlılıkların Kurulumu
```bash
pip install -r requirements.txt
```

### Birim ve Doğrulama Testlerinin Çalıştırılması (17 Test)
Veri sızıntısı kontrolü (subject-wise split), Opacus DP uyumluluğu, INT8 sıkıştırma ve Flower istemci-sunucu testlerini çalıştırmak için:
```bash
python -m pytest -v
```

### Tüm Deney Matrisinin (E1–E7 × 3 Tohum = 21 Koşu) Çalıştırılması
```bash
python -c "import sys; sys.path.insert(0, 'code'); from experiments.run_matrix import run_full_experiment_matrix; run_full_experiment_matrix()"
```

### Tablo ve Grafiklerin Otomatik Üretilmesi
```bash
python -c "import sys; sys.path.insert(0, 'code'); from experiments.export_results import export_all; export_all()"
```

---

## 2. Deney Matrisi Sonuç Özeti (`Seeds: [42, 123, 999]`, Ortalama ± Std)

| Kod | Eğitim Modu | Gizlilik (DP) | Harcanan $\varepsilon$ ($\delta=10^{-4}$) | $\sigma$ | Temsil | Test Accuracy (%) | Test Macro-F1 (%) | Model Boyutu (KB) | Tepe Bellek (KB) | FL İletişim (MB) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Merkezi | Yok | $\infty$ | — | **FP32** | **93.67 ± 0.65** | **93.62 ± 0.69** | `167.29 KB` | `327.64 KB` | `0.00 MB` |
| **E2** | Federatif (FedAvg) | Yok | $\infty$ | — | **FP32** | **90.02 ± 0.74** | **90.10 ± 0.76** | `167.29 KB` | `327.64 KB` | `160.64 MB` |
| **E3** | Federatif (FedAvg) | Yok | $\infty$ | — | **INT8** | **89.97 ± 0.79** | **90.05 ± 0.81** | **`59.78 KB`** | **`189.95 KB`** | `160.64 MB` |
| **E4** | Federatif (FedAvg) | DP-SGD ($\varepsilon_1=3.0$) | `3.00` | `4.39` | **FP32** | **70.92 ± 7.55** | **68.91 ± 8.62** | `167.29 KB` | `327.64 KB` | `160.64 MB` |
| **E5** | Federatif (FedAvg) | DP-SGD ($\varepsilon_1=3.0$) | `3.00` | `4.39` | **INT8** | **70.41 ± 7.23** | **68.19 ± 8.26** | **`59.78 KB`** | **`189.95 KB`** | `160.64 MB` |
| **E6** | Federatif (FedAvg) | DP-SGD ($\varepsilon_2=8.0$) | `7.99` | `2.03` | **FP32** | **81.23 ± 3.22** | **80.91 ± 3.24** | `167.29 KB` | `327.64 KB` | `160.64 MB` |
| **E7** | Federatif (FedAvg) | DP-SGD ($\varepsilon_2=8.0$) | `7.99` | `2.03` | **INT8** | **81.34 ± 2.99** | **81.05 ± 2.99** | **`59.78 KB`** | **`189.95 KB`** | `160.64 MB` |

---

## 3. Üretilen Çıktıların Konumları

- **Sonuç Tabloları (`result/tables/`):**
  - `raw_all_21_runs.csv`: 21 deney koşusunun tüm ham verileri.
  - `summary_mean_std.csv`: 3 tohum üzerinden hesaplanan Ortalama ± Standart Sapma tablosu.
  - `summary_table_markdown.md`: Ana sonuç tablosu ve 6 aktivite sınıfı bazında F1 tablosu.
  - `ieee_results_table.tex`: IEEE konferans bildirisine doğrudan yapıştırılabilir LaTeX tablosu.
- **Grafikler (`result/plots/` ve `paper/figures/`):**
  - `fig1_accuracy_f1_comparison.png / .pdf`
  - `fig2_dp_quantization_interaction.png / .pdf`
  - `fig3_convergence_and_epsilon_curves.png / .pdf`
  - `fig4_size_tradeoff_and_per_class_f1.png / .pdf`
- **Kaydedilen Modeller (`result/models/`):**
  - `E1_seed{42,123,999}.pt` ... `E7_seed{42,123,999}.pt` (toplam 21 model).
