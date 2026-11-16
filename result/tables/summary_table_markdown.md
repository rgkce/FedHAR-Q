# FedHAR-Q — Deney Matrisi Sonuç Tabloları (3 Tohum Ortalama ± Standart Sapma)

## Tablo 1: Ana Karşılaştırma Tablosu (E1–E7)

| Kod | Eğitim Modu | Gizlilik (DP) | Harcanan $\varepsilon$ ($\delta=10^{-4}$) | $\sigma$ | Temsil | Accuracy (%) | Macro-F1 (%) | Model Boyutu (KB) | Çıkarım Gecikmesi (ms) | Tepe Bellek (KB) | FL İletişim (MB) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Centralized | None | ∞ (Yok) | - | **FP32** | **93.67 ± 0.65** | **93.62 ± 0.69** | 167.29 | 0.287 ± 0.063 | 327.64 | 0.00 |
| **E2** | Federated (FedAvg) | None | ∞ (Yok) | - | **FP32** | **90.02 ± 0.74** | **90.10 ± 0.76** | 167.29 | 0.299 ± 0.040 | 327.64 | 160.64 |
| **E3** | Federated (FedAvg) | None | ∞ (Yok) | - | **INT8** | **89.97 ± 0.79** | **90.05 ± 0.81** | 59.78 | 1.348 ± 0.101 | 189.95 | 160.64 |
| **E4** | Federated (FedAvg) | DP-SGD (eps=3.0) | 3.00 | 4.39 | **FP32** | **70.92 ± 7.55** | **68.91 ± 8.62** | 167.29 | 0.304 ± 0.059 | 327.64 | 160.64 |
| **E5** | Federated (FedAvg) | DP-SGD (eps=3.0) | 3.00 | 4.39 | **INT8** | **70.41 ± 7.23** | **68.19 ± 8.26** | 59.78 | 1.346 ± 0.144 | 189.95 | 160.64 |
| **E6** | Federated (FedAvg) | DP-SGD (eps=8.0) | 7.99 | 2.03 | **FP32** | **81.23 ± 3.22** | **80.91 ± 3.24** | 167.29 | 0.293 ± 0.043 | 327.64 | 160.64 |
| **E7** | Federated (FedAvg) | DP-SGD (eps=8.0) | 7.99 | 2.03 | **INT8** | **81.34 ± 2.99** | **81.05 ± 2.99** | 59.78 | 1.292 ± 0.184 | 189.95 | 160.64 |

## Tablo 2: Aktivite Sınıfı Bazında F1 Skorları (Ortalama %)

| Kod | Koşul | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | None (FP32) | 97.83 | 95.33 | 94.49 | 85.21 | 89.00 | 99.85 |
| **E2** | None (FP32) | 97.97 | 92.72 | 93.35 | 77.42 | 80.63 | 98.48 |
| **E3** | None (INT8) | 97.98 | 92.69 | 93.38 | 77.34 | 80.45 | 98.45 |
| **E4** | DP-SGD (eps=3.0) (FP32) | 60.47 | 51.08 | 50.94 | 72.93 | 80.38 | 97.63 |
| **E5** | DP-SGD (eps=3.0) (INT8) | 59.95 | 50.27 | 47.97 | 72.94 | 80.39 | 97.63 |
| **E6** | DP-SGD (eps=8.0) (FP32) | 71.41 | 77.56 | 81.35 | 76.45 | 80.66 | 98.04 |
| **E7** | DP-SGD (eps=8.0) (INT8) | 72.18 | 77.79 | 81.57 | 76.36 | 80.36 | 98.03 |
