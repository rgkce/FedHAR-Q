# Federated HAR Çalışması --- 1 Aylık Paralel Çalışma Planı

Proje Adı: FedHAR-Q
Proje adı açıklaması: Federated Human Activity Recognition with Differential Privacy and Quantization

**Çalışma başlığı:** Federated İnsan Aktivite Tanımada Diferansiyel
Gizlilik ve Kuantizasyonun Birlikte Etkisi\
**İngilizce başlık:** *Evaluating the Joint Impact of Differential
Privacy and Quantization on Federated Human Activity Recognition*\
**Plan süresi:** 4 hafta (1 ay)\
**Hedef:** Bir ay sonunda konferans bildirisi ve kodlama/deney
çalışmalarının her birini yaklaşık **%80 olgunluğa** ulaştırmak.

------------------------------------------------------------------------

## 1. Çalışmanın amacı ve kapsamı

Bu çalışma, UCI HAR veri kümesi üzerinde insan aktivite tanıma (Human
Activity Recognition --- HAR) problemini ele alır. Amaç yeni bir
algoritma önermek değil; Federated Learning (FL), Differential Privacy
(DP) ve INT8 kuantizasyonunun birlikte kullanımının model performansı ve
kaynak kullanımı üzerindeki etkisini kontrollü, tekrarlanabilir
deneylerle incelemektir.

Araştırmanın temel soruları:

1.  FL tabanlı bir HAR modeline DP eklenmesi tanıma başarısını nasıl
    etkiler?
2.  INT8 kuantizasyonun etkisi, DP bulunan ve bulunmayan modellerde
    farklı mıdır?
3.  Doğruluk, model boyutu ve çıkarım gecikmesi arasında nasıl bir
    pratik denge oluşur?

### Zorunlu teknik kapsam

-   UCI HAR veri kümesi kullanılacak.
-   Kişi/katılımcı temelli eğitim-test ayrımı korunacak ve veri
    sızıntısı önlenecek.
-   Tek bir hafif 1D-CNN modeli kurulacak.
-   Merkezi FP32 ve FL FP32 temel sonuçları elde edilecek.
-   FL simülasyonu Flower ile yapılacak.
-   Gizlilik uygulanmayan koşul ve iki DP hedefi (ε₁, ε₂)
    karşılaştırılacak.
-   DP'nin gizlilik birimi/düzeyi, δ değeri ve privacy accountant açıkça
    belgelenecek.
-   FP32 ve INT8 koşulları karşılaştırılacak.
-   DP × kuantizasyon koşulları aynı test protokolüyle
    değerlendirilecek.
-   En az üç farklı rastgele tohumla deney tekrarlanacak.
-   Accuracy, macro-F1, model boyutu, çıkarım gecikmesi; mümkünse tepe
    bellek ve FL iletişim yükü raporlanacak.

### Henüz karara bağlanmamış konu

Deneylerin nerede çalıştırılacağı başlangıçta açık bırakılacaktır:

-   **Seçenek A --- Bilgisayar ortamında simülasyon:** Flower
    istemcileri aynı bilgisayarda simüle edilir; model ve ölçümler
    bilgisayar üzerinde çalıştırılır.
-   **Seçenek B --- Raspberry Pi ortamı:** Uygun donanım ve yazılım
    uyumluluğu doğrulanırsa çıkarım/dağıtım ölçümleri Raspberry Pi
    üzerinde yapılır.
-   **Karar tarihi:** 1. haftanın sonunda teknik fizibilite
    değerlendirmesi yapılarak seçilir. Raspberry Pi mümkün olmazsa
    çalışma simülasyonla devam eder. Donanımda ölçüm yapılmadıysa enerji
    tasarrufu veya gerçek cihaz performansı iddiasında bulunulmaz.

Bu karar, araştırmanın temel deney sorularını değiştirmemelidir.
Raspberry Pi seçeneği esas olarak dağıtım ve performans ölçüm ortamını
etkiler.

------------------------------------------------------------------------

# 2. Plan A --- Konferans bildirisi

## 2.1. Bildirinin hedefi

Bir konferans bildirisi için yaklaşık **15--20 akademik kaynak** içeren,
IEEE biçimine uygun, yöntem ve deney sonuçlarıyla desteklenen tam
bildiri hazırlanacaktır.

Bir aylık hedef, bildirinin yaklaşık %80'inin tamamlanmasıdır. Bu hedef;
literatürün büyük ölçüde taranması, bölüm taslaklarının yazılması,
yöntem ve deney düzeninin açıklanması, elde edilen ilk sonuçların
tablo/grafiklere aktarılması ve tartışma taslağının hazırlanması
anlamına gelir. Son deney sonuçları henüz oluşmadıysa sonuçlar
uydurulmayacak; ilgili kısımlar açıkça bekleyen iş olarak
işaretlenecektir.

## 2.2. Kaynak tarama stratejisi

15--20 kaynak aşağıdaki konu gruplarını kapsayacak şekilde seçilecektir.
Bu dağılım bir hedef dağılımdır; aynı makale birden fazla temayı
kapsayabilir.

  ------------------------------------------------------------------------
  Literatür grubu                Hedef kaynak sayısı İncelenecek konular
  --------------------- ---------------------------- ---------------------
  Human Activity                                2--3 Sensör tabanlı
  Recognition ve UCI                                 aktivite tanıma, veri
  HAR                                                kümesi, 1D-CNN

  Federated Learning                            3--4 FL temelleri, FedAvg,
                                                     HAR uygulamaları,
                                                     istemci/veri dağılımı

  Differential Privacy                          3--4 DP tanımı, DP-SGD,
                                                     ε/δ, privacy
                                                     accountant, FL'de DP

  Model quantization                            2--3 FP32/INT8,
                                                     post-training
                                                     quantization,
                                                     quantization-aware
                                                     training

  DP ve quantization                            3--4 Birleşik etkiler,
  birlikteliği / edge                                kaynak kullanımı, uç
  deployment                                         cihazlarda çıkarım

  **Toplam hedef**                        **15--20** 
  ------------------------------------------------------------------------

### Her kaynak için tutulacak notlar

Her makale için aşağıdaki bilgileri tek bir literatür tablosuna kaydet:

-   Tam künye: yazar, yıl, başlık, yayın/konferans, DOI veya kalıcı
    bağlantı.
-   Çalışmanın problemi ve amacı.
-   Kullanılan veri kümesi.
-   Kullanılan model/yöntem.
-   DP kullanılıyorsa gizlilik düzeyi, ε ve δ bilgisi.
-   Kuantizasyon kullanılıyorsa türü ve bit genişliği.
-   Raporlanan metrikler.
-   Çalışmanın bulgusu.
-   Bu bildiride hangi bölüm veya argüman için kullanılacağı.
-   Sınırlılık veya bu çalışmadan ayrışan nokta.

Kaynaklar doğrulanabilir akademik yayınlardan seçilmeli; yalnızca başlık
veya özet üzerinden teknik ayrıntı aktarılmamalıdır. DOI ve
bibliyografik bilgiler kontrol edilmelidir.

## 2.3. Bildiri bölüm planı

  -----------------------------------------------------------------------
  Bölüm                   İçerik                  Tamamlanma ölçütü
  ----------------------- ----------------------- -----------------------
  Özet                    Problem, yöntem, deney  Sonuçlar elde edilince
                          kapsamı ve temel        son kez yazılıp
                          bulgular                güncellenmiş

  1\. Giriş               HAR, kişisel            Araştırma boşluğu ve
                          veri/gizlilik           sorular açık
                          motivasyonu, FL ve      
                          kaynak kısıtları,       
                          araştırma soruları ve   
                          katkı                   

  2\. İlgili Çalışmalar   HAR, FL, DP,            15--20 kaynakla tematik
                          quantization ve kesişim sentez
                          çalışmaları             

  3\. Yöntem              UCI HAR, veri           Başka bir
                          protokolü, 1D-CNN, FL,  araştırmacının
                          DP ve INT8 yaklaşımı    izleyebileceği
                                                  açıklıkta

  4\. Deneysel Kurulum    Yazılım/donanım ortamı, Tüm deney koşulları
                          istemciler,             belgelenmiş
                          parametreler, ε/δ,      
                          tohumlar ve ölçüm       
                          yöntemi                 

  5\. Sonuçlar            Accuracy, macro-F1,     Gerçek deney
                          model boyutu, gecikme,  verileriyle
                          iletişim yükü ve bellek tablolar/grafikler
                          sonuçları               

  6\. Tartışma            DP etkisi, quantization Araştırma sorularına
                          etkisi, etkileşim ve    kanıta dayalı yanıt
                          pratik ödünleşimler     

  7\. Sınırlılıklar       Simülasyon/donanım      İddialar ölçümlerin
                          sınırları, gizlilik     sınırını aşmıyor
                          kapsamı, veri ve ölçüm  
                          sınırlılıkları          

  8\. Sonuç ve Gelecek    Temel bulgular ve       Sonuçlarla tutarlı
  Çalışmalar              sonraki adımlar         

  Kaynakça                IEEE biçiminde 15--20   Metin içi atıflar ve
                          kaynak                  kaynakça eşleşiyor
  -----------------------------------------------------------------------

## 2.4. Dört haftalık bildiri iş planı

### 1. Hafta --- Araştırma çerçevesi ve literatür başlangıcı

-   Araştırma sorularını ve çalışmanın katkısını netleştir.
-   Literatür tarama anahtar kelimelerini belirle.
-   15--20 kaynaklık aday liste oluştur; en az 8--10 kaynağı ayrıntılı
    incele.
-   Literatür matrisi oluştur.
-   Bildiri şablonunu ve bölüm başlıklarını hazırla.
-   Giriş bölümünün ilk taslağını yaz.
-   Yöntem bölümünün alt başlıklarını aç.

**Hafta sonu çıktısı:** Bildiri iskeleti, araştırma soruları, kaynak
havuzu ve giriş taslağı.

### 2. Hafta --- Literatür sentezi ve yöntem yazımı

-   Kaynak sayısını 15--20'ye tamamla ve künyeleri doğrula.
-   İlgili çalışmalar bölümünü tematik biçimde yaz.
-   UCI HAR veri kümesi ve kişi bazlı ayrım protokolünü açıkla.
-   1D-CNN, Flower/FedAvg, DP ve INT8 yöntemlerinin taslak
    açıklamalarını hazırla.
-   Deney matrisi ve raporlanacak metrikleri bildiride gösterilecek
    biçimde tasarla.
-   Deney ortamı seçimi için bilgisayar/Raspberry Pi fizibilite sonucunu
    yöntem notlarına ekle.

**Hafta sonu çıktısı:** Giriş ve ilgili çalışmalar büyük ölçüde
tamamlanmış; yöntem bölümü taslak hâlinde.

### 3. Hafta --- Deney düzeni ve ilk sonuçların metne aktarılması

-   Kodlama tarafındaki gerçek deney ayarlarını yöntem ve deneysel
    kurulum bölümlerine işle.
-   Çalıştırılabilen deneylerden gelen ilk sonuç tablolarını oluştur.
-   Sonuç grafikleri için şablonları hazırla; yalnızca gerçek sonuçları
    doldur.
-   Sonuçlar bölümünün mevcut verilerle yazılabilen kısımlarını yaz.
-   Tartışma bölümünde yanıtlanacak soruları ve karşılaştırma mantığını
    belirle.
-   Sınırlılıklar taslağını oluştur.

**Hafta sonu çıktısı:** Yöntem ve deneysel kurulum bölümü büyük ölçüde
tamamlanmış; ilk sonuçlar metne aktarılmış.

### 4. Hafta --- Bildiriyi %80 seviyesine getirme

-   Elde edilen deney sonuçlarını güncelle ve doğrula.
-   Sonuç tablolarını ve grafiklerini düzenle.
-   Tartışma ve sınırlılıklar bölümlerini mevcut kanıtlarla yaz.
-   Sonuç bölümünün taslağını hazırla.
-   Özet bölümünü mevcut gerçek bulgulara göre yaz.
-   IEEE biçim kontrolü, atıf-kaynakça eşleştirmesi ve dil kontrolü yap.
-   Eksik deneyleri ve tamamlanması gereken yazım işlerini ayrı bir
    listeye al.

**Ay sonu hedefi:** 15--20 kaynak doğrulanmış; giriş ve ilgili
çalışmalar tamamlanmış; yöntem ve deney düzeni yazılmış; ilk sonuçlar ve
tartışma taslağı mevcut; bildiri yaklaşık %80 tamamlanmış.

## 2.5. Bildiri tarafının teslim çıktıları

-   IEEE biçiminde düzenlenmiş bildiri taslağı (`.docx` veya `.tex`).
-   15--20 kaynaklık doğrulanmış kaynakça.
-   Literatür karşılaştırma tablosu.
-   Deney matrisi ve metrik tanımları.
-   Gerçek sonuçlardan üretilen tablo ve grafikler.
-   Eksik kalan işler ve revizyon listesi.

------------------------------------------------------------------------

# 3. Plan B --- Kodlama ve deney çalışması

## 3.1. Kodlama tarafının hedefi

Bir ay sonunda deney altyapısının ve temel modellerin çalışır durumda
olması; DP ve INT8 koşullarının önemli bölümünün uygulanması; ölçümlerin
ve tekrarların büyük ölçüde tamamlanması hedeflenir.

Kodlama planı, deneylerin bilgisayar simülasyonunda mı yoksa Raspberry
Pi üzerinde mi yürütüleceği kesinleşmeden başlayabilecek şekilde
tasarlanmıştır.

## 3.2. Teknik kararlar ve başlangıç varsayımları

  -----------------------------------------------------------------------
  Karar                   Başlangıç yaklaşımı     Netleştirilecek nokta
  ----------------------- ----------------------- -----------------------
  Veri kümesi             UCI HAR                 Resmî veri ayrımının ve
                                                  kişi kimliklerinin
                                                  kontrolü

  Model                   Hafif 1D-CNN            Girdi boyutu, katmanlar
                                                  ve hiperparametreler

  Merkezi baseline        PyTorch FP32            Eğitim/test protokolü

  FL                      Flower + FedAvg         İstemci sayısı, round
                                                  sayısı ve veri dağılımı

  DP                      Opacus veya uyumlu DP   Örnek/istemci düzeyi,
                          uygulaması              ε₁, ε₂, δ ve accountant

  Quantization            INT8                    Post-training veya
                                                  quantization-aware
                                                  yaklaşım

  Çalıştırma ortamı       Önce bilgisayar         Raspberry Pi
                                                  fizibilitesi ve ölçüm
                                                  kapsamı

  Tekrar                  En az 3 rastgele tohum  Aynı tohumların tüm
                                                  koşullarda kullanılması
  -----------------------------------------------------------------------

**Önemli uygulama notu:** Flower ve Opacus entegrasyonunda gizlilik
garantisinin hangi birim için hesaplandığı açıkça tanımlanmalıdır. Örnek
düzeyinde DP-SGD kullanımı, tek başına istemci düzeyinde DP garantisi
vermez. Kullanılan privacy accountant ve gizlilik varsayımları kod ve
raporda belgelenmelidir.

## 3.3. Kod mimarisi önerisi

``` text
federated-har/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train_centralized.py
│   ├── federated_client.py
│   ├── federated_server.py
│   ├── privacy.py
│   ├── quantization.py
│   ├── evaluation.py
│   └── measurements.py
├── configs/
│   ├── baseline.yaml
│   ├── federated.yaml
│   ├── privacy.yaml
│   └── quantization.yaml
├── experiments/
│   └── experiment_matrix.csv
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
├── tests/
├── requirements.txt
└── README.md
```

Bu mimari öneridir; deneyler ilerledikçe dosya yapısı sadeleştirilebilir
veya genişletilebilir.

## 3.4. Deney matrisi

Temel karşılaştırma aşağıdaki koşulları kapsamalıdır:

  Kod   Eğitim koşulu      Gizlilik    Temsil
  ----- ------------------ ----------- --------
  E1    Merkezi eğitim     DP yok      FP32
  E2    Federatif eğitim   DP yok      FP32
  E3    Federatif eğitim   DP yok      INT8
  E4    Federatif eğitim   ε₁ hedefi   FP32
  E5    Federatif eğitim   ε₁ hedefi   INT8
  E6    Federatif eğitim   ε₂ hedefi   FP32
  E7    Federatif eğitim   ε₂ hedefi   INT8

Merkezi INT8 ek bir karşılaştırma olarak yapılabilir; zorunlu çekirdek
matrise eklenmesi zaman ve hesaplama bütçesine göre kararlaştırılır.

Her koşul aynı test protokolüyle değerlendirilmelidir. En az üç farklı
rastgele tohumla tekrar yapılmalı; ortalama ve standart sapma
raporlanmalıdır.

## 3.5. Dört haftalık kodlama iş planı

### 1. Hafta --- Ortam, veri ve deney protokolü

-   Proje deposu ve sanal Python ortamı oluştur.
-   PyTorch ve temel bağımlılıkları kur; sürümleri kaydet.
-   UCI HAR veri kümesini indir ve yapısını incele.
-   Katılımcı bazlı eğitim/test ayrımını doğrula.
-   Ön işleme ve veri yükleme kodunu oluştur.
-   1D-CNN mimarisini kur.
-   Bilgisayar simülasyonu ve Raspberry Pi için fizibilite kontrolü yap:
    -   Raspberry Pi mevcut mu, erişilebilir mi?
    -   PyTorch/Flower/INT8 çalışma ortamı uyumlu mu?
    -   Gecikme ve bellek ölçümü yapılabilecek mi?
-   Hafta sonunda çalışma ortamı kararını ver veya Raspberry Pi
    seçeneğini sonraki aşamaya bırak.

**Hafta sonu çıktısı:** Veri yükleme hattı, doğrulanmış veri ayrımı, ilk
model taslağı ve çalışma ortamı kararı.

### 2. Hafta --- Merkezi ve federatif baseline

-   Merkezi FP32 1D-CNN modelini eğit.
-   Accuracy, macro-F1 ve confusion matrix üret.
-   Flower ile istemci-sunucu simülasyonunu kur.
-   FedAvg ile federatif FP32 modelini eğit.
-   İstemci sayısı, veri dağılımı, round sayısı ve yerel epoch
    bilgilerini kaydet.
-   Merkezi ve federatif sonuçları aynı test protokolünde karşılaştır.
-   Model ağırlığı/dosyası boyutunu ölç.

**Hafta sonu çıktısı:** Merkezi FP32 ve FL FP32 baseline modelleri
çalışıyor; ilk metrikler kaydedilmiş.

### 3. Hafta --- DP ve INT8 uygulamaları

-   DP mekanizmasını küçük bir testte entegre et.
-   DP'nin örnek/istemci düzeyini ve gizlilik muhasebesini doğrula.
-   ε₁ ve ε₂ hedefleri için yapılandırmaları oluştur.
-   DP'li federatif modelleri eğit.
-   INT8 kuantizasyon yöntemini seç ve uygulamayı doğrula.
-   FP32 ve INT8 modellerini aynı test kümesinde değerlendir.
-   Model boyutu ve çıkarım gecikmesi ölçüm kodunu hazırla.
-   Çalışmayan entegrasyonları ve uyumluluk sorunlarını belgele.

**Hafta sonu çıktısı:** DP ve INT8 koşullarının ilk çalışır sürümleri;
temel deney matrisi büyük ölçüde uygulanmış.

### 4. Hafta --- Tekrarlar, ölçümler ve sonuçların sabitlenmesi

-   Deney koşullarını en az üç farklı rastgele tohumla çalıştır.
-   Eksik kalan koşulları tamamla.
-   Ortalama ve standart sapmaları hesapla.
-   Accuracy ve macro-F1 tablolarını oluştur.
-   Model boyutu, çıkarım gecikmesi, mümkünse tepe bellek ölçümlerini
    tamamla.
-   FL iletişim yükünü kaydet.
-   DP için ε, δ, accountant ve gizlilik varsayımlarını doğrula.
-   DP × INT8 etkileşimini karşılaştır.
-   Kodları, parametreleri ve deney konfigürasyonlarını düzenle.
-   README ve kurulum yönergesini yaz.
-   Sonuç tablolarını bildiri çalışmasıyla paylaşılabilecek biçimde dışa
    aktar.

**Ay sonu hedefi:** Temel deneylerin tamamına yakını çalışır; en az üç
tohumla sonuçların büyük bölümü elde edilmiş; kod, konfigürasyon ve
ölçüm kayıtları düzenlenmiş; kodlama/deney tarafı yaklaşık %80
tamamlanmış.

## 3.6. Kodlama tarafının teslim çıktıları

-   Çalıştırılabilir Python kaynak kodu.
-   Bağımlılık ve sürüm listesi.
-   Kurulum ve çalıştırma yönergesi.
-   Veri hazırlama ve deney konfigürasyonları.
-   Merkezi ve federatif baseline sonuçları.
-   DP ve INT8 deney çıktıları.
-   Tekrarlı deneylerin ham sonuçları.
-   Ortalama/standart sapma tabloları.
-   Model boyutu, gecikme ve iletişim yükü ölçümleri.
-   Grafikler ve kısa ilerleme sunumu.

------------------------------------------------------------------------

# 4. İki planın eş zamanlı yürütülmesi

Kodlama ve bildiri çalışmaları birbirinden bağımsız iki iş paketi olarak
takip edilir; ancak aynı araştırma kararlarını ve deney çıktılarını
paylaşırlar.

**Temel çalışma ilkesi:** Kodlama tarafında alınan her teknik karar,
bildiri tarafındaki yöntem/deneysel kurulum bölümüne işlenir. Her deney
tamamlandığında sonuçlar aynı gün sonuç tablosuna ve ilgili bildiri
notlarına aktarılır.

## 4.1. Haftalık paralel çalışma takvimi

  -----------------------------------------------------------------------
  Hafta             Kodlama tarafı    Bildiri tarafı    Ortak kontrol
                                                        noktası
  ----------------- ----------------- ----------------- -----------------
  1                 Ortam, veri,      Literatür,        Veri protokolü ve
                    1D-CNN,           araştırma         deney kapsamı net
                    fizibilite        soruları, giriş   

  2                 Merkezi ve FL     İlgili            Model ve FL
                    baseline          çalışmalar,       ayarları belgeli
                                      yöntem taslağı    

  3                 DP, INT8, ilk     Yöntem ve         DP/quantization
                    deneyler          deneysel kurulum, koşulları belgeli
                                      ilk tablolar      

  4                 Tekrarlar,        Sonuçlar,         Bildiri ve kod
                    metrikler,        tartışma,         çıktıları tutarlı
                    grafikler, README sınırlılıklar,    
                                      IEEE kontrolü     
  -----------------------------------------------------------------------

## 4.2. Örnek haftalık zaman dağılımı

Haftada 6 gün çalışma ve günde yaklaşık 4--5 saat ayırma varsayımıyla:

  -----------------------------------------------------------------------
  Gün                     Ana odak                İkincil odak
  ----------------------- ----------------------- -----------------------
  Pazartesi               Kodlama                 Bildiri/literatür

  Salı                    Kodlama                 Bildiri yazımı

  Çarşamba                Bildiri/literatür       Kodlama

  Perşembe                Kodlama                 Bildiri yazımı

  Cuma                    Kodlama ve deney        Sonuçları metne aktarma

  Cumartesi               Haftalık entegrasyon ve İlerleme
                          eksik kapatma           değerlendirmesi

  Pazar                   Dinlenme veya isteğe    ---
                          bağlı hafif okuma       
  -----------------------------------------------------------------------

Zamanı iki tarafa her gün tam eşit bölmek zorunlu değildir. Önemli olan
her hafta iki iş paketinin de ilerlemesi ve birinin diğerini bekleterek
durdurmamasıdır.

## 4.3. Haftalık ortak toplantı/kontrol

Her hafta sonunda aşağıdaki sorular cevaplanır:

1.  Bu hafta kod tarafında hangi somut çıktı tamamlandı?
2.  Hangi deneyler gerçekten çalıştırıldı?
3.  Bildiride hangi bölüm veya kaynak tamamlandı?
4.  Kod ile bildirideki yöntem açıklamaları tutarlı mı?
5.  Hangi teknik karar hâlâ belirsiz?
6.  Gelecek haftanın en önemli üç işi nedir?
7.  Takvim gerideyse hangi isteğe bağlı iş ertelenecek?

------------------------------------------------------------------------

# 5. İlerleme takibi ve %80 tamamlanma ölçütü

İlerlemeyi yalnızca harcanan saatle değil, doğrulanabilir çıktılarla
ölç.

## 5.1. Bildiri iş paketi ağırlıkları

  İş kalemi                                      Ağırlık
  ------------------------------------------- ----------
  Araştırma soruları ve bildiri iskeleti             %10
  15--20 kaynağın bulunması ve doğrulanması          %20
  Giriş ve ilgili çalışmalar                         %20
  Yöntem ve deneysel kurulum                         %20
  Sonuçlar, tablolar ve grafikler                    %15
  Tartışma, sınırlılıklar ve sonuç                   %10
  IEEE biçim ve son kontrol                           %5
  **Toplam**                                    **%100**

## 5.2. Kodlama iş paketi ağırlıkları

  İş kalemi                                           Ağırlık
  ------------------------------------------------ ----------
  Ortam ve veri hazırlama                                 %10
  Merkezi 1D-CNN baseline                                 %15
  Flower FL FP32 baseline                                 %15
  DP entegrasyonu ve gizlilik muhasebesi                  %20
  INT8 kuantizasyon                                       %15
  Tekrarlı birleşik deneyler                              %15
  Ölçümler, dokümantasyon ve sonuç dışa aktarımı          %10
  **Toplam**                                         **%100**

## 5.3. Ay sonu %80 hedefinin yorumu

Bir ay sonunda her iki iş paketi için ayrı ayrı yaklaşık %80 ilerleme
hedeflenir:

**Bildiri tarafında %80:** - 15--20 kaynak doğrulanmış ve literatür
sentezi yazılmış. - Giriş, ilgili çalışmalar, yöntem ve deneysel kurulum
bölümleri büyük ölçüde tamamlanmış. - Gerçek sonuçların önemli bölümü
tablo ve grafiklere aktarılmış. - Tartışma ve sonuç taslakları
hazırlanmış. - Son deneyler ve son dil/IEEE revizyonu kalmış olabilir.

**Kodlama tarafında %80:** - Veri hattı ve merkezi/FL baseline
çalışıyor. - DP ve INT8 uygulamaları doğrulanmış. - Temel deney matrisi
büyük ölçüde çalıştırılmış. - Tekrarlı deneylerin ve ölçümlerin büyük
bölümü elde edilmiş. - Kod ve konfigürasyonlar düzenlenmiş. - Kalan
işler eksik deneyler, son doğrulama ve dokümantasyon iyileştirmeleri
olabilir.

Yüzde değerleri ilerleme tahminidir; tamamlanmış sayılmaları için ilgili
çıktının gerçekten mevcut ve kontrol edilmiş olması gerekir. Özellikle
DP garantisi doğrulanmadan veya deneyler çalıştırılmadan ilgili iş
kalemi tamamlandı kabul edilmemelidir.

------------------------------------------------------------------------

# 6. Riskler ve B planları

  -----------------------------------------------------------------------
  Risk                    Etkisi                  Önlem / B planı
  ----------------------- ----------------------- -----------------------
  Flower--Opacus          DP deneyleri            Önce küçük entegrasyon
  uyumsuzluğu             gecikebilir             testi; sürümleri
                                                  sabitle; gizlilik
                                                  birimini ve yöntemi
                                                  yeniden değerlendir

  Raspberry Pi erişimi    Donanım ölçümleri       Bilgisayar
  veya uyumluluğu         gecikebilir             simülasyonuyla devam
                                                  et; Pi ölçümlerini ek
                                                  çalışma olarak bırak

  INT8 operatör desteği   Model kuantize olsa da  Desteklenen backend ve
                          çalışmayabilir          katmanları doğrula;
                                                  uygun kuantizasyon
                                                  yolunu seç

  Deneylerin uzun sürmesi Üç tohum hedefi         Önce küçük doğrulama
                          gecikebilir             koşusu; sonra tam
                                                  deneyleri sırayla
                                                  çalıştır

  Literatürde doğrudan    İlgili çalışmalar       FL, DP ve quantization
  kesişim kaynağı azlığı  bölümü zorlaşabilir     literatürünü ayrı
                                                  temalarda sentezle;
                                                  kesişim çalışmalarını
                                                  ayrıca belirt

  Sonuçların beklenenden  Olumsuz gibi            Sonuçları değiştirmeye
  düşük olması            algılanabilir           çalışma; nedenleri ve
                                                  sınırlılıkları bilimsel
                                                  biçimde tartış

  Veri sızıntısı veya     Sonuçların geçerliliği  Kişi bazlı ayrımı ve
  tutarsız test protokolü bozulabilir             aynı test protokolünü
                                                  baştan doğrula
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 7. İlk gün yapılacaklar

-   [ ] Proje klasörünü ve Git deposunu oluştur.
-   [ ] Bildiri için IEEE şablonunu hazırla.
-   [ ] Literatür takip tablosunu oluştur.
-   [ ] UCI HAR veri kümesini indir ve incele.
-   [ ] Python ortamını ve PyTorch kurulumunu doğrula.
-   [ ] Kişi bazlı veri ayrımı planını yaz.
-   [ ] Flower ve Opacus sürüm/uyumluluk araştırmasına başla.
-   [ ] Raspberry Pi erişimi ve teknik gereksinimleri kontrol et.
-   [ ] Hocaya sorulacak açık kararları listele.

## 8. Hocayla netleştirilecek kararlar

Aşağıdaki konular kesinleşmeden nihai deney konfigürasyonları
sabitlenmemelidir:

-   DP'nin örnek düzeyinde mi, istemci düzeyinde mi uygulanacağı.
-   ε₁, ε₂ ve δ değerlerinin seçimi.
-   Privacy accountant ve gizlilik varsayımları.
-   İstemci sayısı, yerel epoch ve federatif round sayısı.
-   IID veya Non-IID istemci veri dağılımı.
-   INT8 için kullanılacak kuantizasyon yöntemi.
-   Bilgisayar simülasyonu ve Raspberry Pi arasında nihai ölçüm ortamı.
-   Konferansın şablonu, sayfa sınırı ve teslim tarihi.

------------------------------------------------------------------------

## 9. Bir ayın sonunda beklenen genel durum

  -----------------------------------------------------------------------
  Çıktı                               1 aylık hedef
  ----------------------------------- -----------------------------------
  Konferans bildirisi                 Yaklaşık %80 tamamlanmış taslak

  Literatür                           15--20 doğrulanmış akademik kaynak

  Merkezi model                       FP32 baseline çalışır

  Federatif model                     Flower + FP32 baseline çalışır

  DP                                  İki gizlilik hedefi için uygulama
                                      ve ilk/temel sonuçlar

  Quantization                        FP32--INT8 karşılaştırması

  Deneyler                            Temel deney matrisi ve en az üç
                                      tohum hedefi

  Ölçümler                            Accuracy, macro-F1, boyut, gecikme;
                                      mümkünse bellek ve iletişim

  Kod deposu                          Kurulum, yapılandırma ve çalıştırma
                                      yönergesiyle düzenli

  Sonraki adım                        Eksik deneyleri tamamlama, sonuç
                                      doğrulama ve bildiri revizyonu
  -----------------------------------------------------------------------

**Başarı ölçütü:** Bir ayın sonunda yalnızca okunmuş makaleler veya
yazılmış kodlar değil; çalışan bir deney hattı, doğrulanabilir ilk
sonuçlar ve bu sonuçlarla tutarlı bir konferans bildirisi taslağı
bulunmalıdır.
