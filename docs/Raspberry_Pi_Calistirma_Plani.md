# FedHAR-Q — Raspberry Pi Üzerinde Çalıştırma ve Ölçüm Planı (Adım Adım Rehber)

Bu rehber, bilgisayarda eğittiğimiz 21 modelin (`E1`–`E7` × `3 tohum`) **Raspberry Pi** üzerinde **çıkarım gecikmesi (inference latency - ms)**, **tepe bellek kullanımı (KB)** ve **INT8 ARM (`qnnpack`) doğruluğunu** ölçmek için izlenecek en hızlı ve sorunsuz yolu anlatır.

---

## 💡 Temel Strateji: Eğitimi Bilgisayarda Tut, Çıkarımı (Inference) Raspberry Pi'de Ölç!

Gerçek dünyadaki Federatif Öğrenme + Uç Cihaz Dağıtımı (Edge Deployment) mimarisinde ve çalışma planımızın **Seçenek B** maddesinde belirtildiği gibi:
- **Ağır Federatif Eğitim ve Opacus DP-SGD (E1, E2, E4, E6):** Bilgisayarımızda zaten tamamlandı ve ağırlıklar `result/models/` klasörüne kaydedildi (`E1_seed42.pt`, `E2_seed42.pt`, `E4_seed42.pt`, `E6_seed42.pt` vb.). Raspberry Pi üzerinde saatlerce federatif eğitim veya Opacus kurulumu yapmana **gerek yoktur!**
- **Raspberry Pi'de Yapılacak İş (Sadece ~3 Dakika Sürer):** Bilgisayardaki eğitilmiş model ağırlıklarını Raspberry Pi'ye aktarıp, senin için hazırladığım **`code/experiments/run_rpi_benchmark.py`** betiğini tek komutla çalıştırmaktır.

---

## Adım 1: Raspberry Pi Ön Gereksinimleri (Donanım & İşletim Sistemi)

1. **Desteklenen Modeller:** Raspberry Pi 3B+, Raspberry Pi 4 (2GB/4GB/8GB) veya Raspberry Pi 5.
2. **İşletim Sistemi (Önemli):** **64-bit Raspberry Pi OS (`aarch64` / Debian Bookworm veya Bullseye)** kurulu olmalıdır. *(PyTorch'un güncel ARM tekerlekleri ve `qnnpack` INT8 motoru 64-bit işletim sistemi gerektirir).*
   - Kontrol etmek için Raspberry Pi terminalinde şu komutu yaz:
     ```bash
     uname -m
     ```
   - Çıktı `aarch64` ise 64-bit işletim sistemin hazırdır!

---

## Adım 2: Proje Dosyalarını Bilgisayardan Raspberry Pi'ye Aktarma

Bilgisayarındaki `FedHAR-Q` klasörünü Raspberry Pi'ye aktarmak için iki yöntemden birini seçebilirsin:

### Yöntem A: USB Bellek ile (En Kolay)
1. Bilgisayarındaki `FedHAR-Q` klasörünü (içindeki `code/`, `dataset/`, `result/models/` ile birlikte) USB belleğe kopyala.
2. USB belleği Raspberry Pi'ye tak ve `/home/pi/FedHAR-Q` (veya Masaüstü) dizinine yapıştır.

### Yöntem B: Aynı Wi-Fi Ağında `scp` Komutu ile (Windows PowerShell'den)
Bilgisayarında PowerShell'i açıp şu komutu çalıştır (`pi` ve `192.168.1.XX` yerine kendi Raspberry Pi kullanıcı adını ve IP adresini yaz):
```powershell
scp -r C:\Users\rumeysagokce\Desktop\FedHAR-Q pi@192.168.1.XX:~/FedHAR-Q
```

---

## Adım 3: Raspberry Pi Üzerinde Python Sanal Ortamı ve Hafif Paketlerin Kurulumu

Raspberry Pi üzerinde sadece **çıkarım ve ölçüm** yapacağımız için `flwr` (Flower) veya `opacus` gibi ağır eğitim paketlerini kurmamıza bile gerek yoktur! Sadece `torch`, `numpy`, `pandas`, `scikit-learn` ve `pyyaml` yeterlidir:

Raspberry Pi terminalinde sırasıyla şu komutları çalıştır:

```bash
cd ~/FedHAR-Q

# 1. Sanal ortam oluştur ve etkinleştir
python3 -m venv .venv
source .venv/bin/activate

# 2. Sadece çıkarım ve ölçüm için gerekli hafif paketleri kur
pip install --upgrade pip
pip install torch numpy pandas scikit-learn pyyaml
```

*(Kontrol: `python -c "import torch; print(torch.__version__, torch.backends.quantized.supported_engines)"` yazdığında listede `'qnnpack'` göreceksin; bu ARM işlemcilerin yerel INT8 hızlandırıcısıdır!)*

---

## Adım 4: Tek Komutla Raspberry Pi Ölçümlerini Çalıştırma

Senin için `code/experiments/run_rpi_benchmark.py` adında özel bir betik hazırladım. Bu betik:
1. `result/models/` içindeki hazır eğitilmiş FP32 modelleri (`E1, E2, E4, E6`) yükler,
2. Raspberry Pi'nin ARM işlemcisi üzerinde `qnnpack` motoruyla INT8'e (`E3, E5, E7`) kuantize eder,
3. 3 tohum (`42, 123, 999`) × 7 koşul (`E1–E7`) = **21 modelin tamamı** için Raspberry Pi üzerindeki gerçek **Çıkarım Gecikmesini (ms)**, **Tepe Belleği (KB)**, **Model Boyutunu (KB)** ve **Test Accuracy / Macro-F1 (%)** değerlerini ölçer.

Raspberry Pi terminalinde sadece şu komutu çalıştır:

```bash
python code/experiments/run_rpi_benchmark.py
```

Yaklaşık **2–3 dakika** içinde ekranda tüm E1–E7 ölçümleri akacak ve işlem bittiğinde şu iki dosya otomatik oluşacaktır:
- `result/tables/rpi_edge_benchmark_summary.csv`
- `result/tables/rpi_edge_benchmark_markdown.md`

---

## Adım 5: Sonuçları Bilgisayara Geri Getirme ve Bildiriye Ekleme

1. Raspberry Pi'de oluşan `result/tables/rpi_edge_benchmark_markdown.md` ve `rpi_edge_benchmark_summary.csv` dosyalarını bilgisayarına geri kopyala (veya bana buradan içeriğini yapıştır).
2. Bildirinin **Bölüm 5 (Sonuçlar)** kısmına hem **Bilgisayar (x86 CPU)** hem de **Raspberry Pi (ARM Cortex-A Uç Cihaz)** gecikme karşılaştırmasını birlikte koyabiliriz!

---

## ⚠️ Önemli Hatırlatma (Görev Dokümanı Kuralı)
- Görev dokümanında belirtildiği gibi, Raspberry Pi üzerinde yazılım tabanlı gecikme (`ms`) ve bellek (`KB`) ölçümü yaptığımızda bildiride **"gerçek uç cihaz çıkarım gecikmesi ve bellek kullanımı"** raporlayabiliriz; ancak fiziksel bir USB güç/ampermetre cihazı bağlamadığımız sürece **enerji tüketimi (Joule / mW) iddiasında bulunmayacağız**.
