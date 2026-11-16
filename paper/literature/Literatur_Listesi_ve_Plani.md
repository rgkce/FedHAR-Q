# FedHAR-Q — 20 Kaynaklık Nihai Literatür Listesi ve Yazım Sıralaması

Bu dosya, IEEE konferans bildirisinde kullanılacak **20 doğrulanmış akademik kaynağı**, bildiri yazımındaki atıf sırasına (`[1]`–`[20]`) ve **5 tematik gruba** göre listeler. Makaleleri bilgisayarınıza indirirken dosya adlarını `01_Anguita_2013_UCI_HAR.pdf` ... `20_Kang_2024_RDP_Quantization.pdf` şeklinde numaralandırarak kaydedebilirsiniz.

---

## Grup 1: Human Activity Recognition (HAR) ve UCI HAR Veri Kümesi (`[1]`–`[4]`)
*Bildiride Kullanım Yeri: Bölüm 1 (Giriş), Bölüm 2.1 (Sensör Tabanlı HAR) ve Bölüm 3.1 (UCI HAR Veri Kümesi)*

1. **`[1]` Anguita et al. (2013) — [Zorunlu Veri Kümesi Referansı]**
   - **Başlık:** *A Public Domain Dataset for Human Activity Recognition Using Smartphones*
   - **Yazarlar:** Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz
   - **Yayın:** 21st European Symposium on Artificial Neural Networks (ESANN 2013), Bruges, Belgium
   - **Önerilen Dosya Adı:** `01_Anguita_2013_UCI_HAR.pdf`
   - **İndirme Linki:** [Ücretsiz PDF – ESANN](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-84.pdf)
   - **Neden Kullanıyoruz?** Kullandığımız UCI HAR veri kümesinin (30 katılımcı, 9 kanallı 128 örnekli eylemsizlik sinyalleri, 6 aktivite sınıfı) orijinal kurucu makalesidir.

2. **`[2]` Sarkar, Hossain & Sarkar (2023)**
   - **Başlık:** *Human Activity Recognition from Sensor Data Using Spatial Attention-Aided CNN with Genetic Algorithm*
   - **Yazarlar:** Apu Sarkar, S. K. Sabbir Hossain, Ram Sarkar
   - **Yayın:** Neural Computing and Applications, 35, 5165–5191 (2023)
   - **Önerilen Dosya Adı:** `02_Sarkar_2023_CNN_HAR.pdf`
   - **İndirme Linki:** [Ücretsiz Tam Metin – PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9596348/) | [Doğrudan PDF](https://pmc.ncbi.nlm.nih.gov/articles/PMC9596348/pdf/521_2022_Article_7911.pdf)
   - **Neden Kullanıyoruz?** Giyilebilir sensör verileri ve UCI HAR üzerinde CNN mimarilerinin etkinliğini göstermek için.

3. **`[3]` Raja Sekaran et al. (2024)**
   - **Başlık:** *A Hybrid TCN-GRU Model for Classifying Human Activities Using Smartphone Inertial Signals*
   - **Yazarlar:** Sarmela Raja Sekaran, Ying Han Pang, Lim Zheng You, Ooi Shih Yin
   - **Yayın:** PLOS ONE, 19(8), e0304655 (2024) — DOI: `10.1371/journal.pone.0304655`
   - **Önerilen Dosya Adı:** `03_Sekaran_2024_Inertial_HAR.pdf`
   - **İndirme Linki:** [Ücretsiz Tam Metin – PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11321576/) | [Doğrudan PDF – PLOS ONE](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0304655&type=printable)
   - **Neden Kullanıyoruz?** Doğrudan UCI HAR ham eylemsizlik sinyalleri (inertial signals) üzerinde derin öğrenme sınıflandırmasını inceleyen güncel çalışma.

4. **`[4]` Aouedi et al. (2024) — [HAR → FL Geçiş Köprüsü]**
   - **Başlık:** *Federated Learning for Human Activity Recognition: Overview, Advances, and Challenges*
   - **Yazarlar:** Ons Aouedi, Alessio Sacco, Latif U. Khan, Dinh C. Nguyen, Mohsen Guizani
   - **Yayın:** IEEE (2024)
   - **Önerilen Dosya Adı:** `04_Aouedi_2024_FL_HAR_Survey.pdf`
   - **İndirme Linki:** [Ücretsiz PDF – IRIS/ORBilu](https://iris.polito.it/retrieve/bd47a2e8-7071-467d-9c8a-f600f62c994f/Federated_Learning_for_Human_Activity_Recognition_Overview_Advances_and_Challenges.pdf)
   - **Neden Kullanıyoruz?** Merkezi HAR modellerinin gizlilik sorunlarını ve neden Federated HAR'a ihtiyaç duyulduğunu açıklayan temel tarama makalesi.

---

## Grup 2: Federated Learning (FL) Temelleri ve HAR Uygulamaları (`[5]`–`[8]`)
*Bildiride Kullanım Yeri: Bölüm 1 (Giriş), Bölüm 2.2 (Federated Learning), Bölüm 3.3 (FedAvg) ve Bölüm 4 (Flower Kurulumu)*

5. **`[5]` McMahan et al. (2017) — [FedAvg Kurucu Makalesi]**
   - **Başlık:** *Communication-Efficient Learning of Deep Networks from Decentralized Data*
   - **Yazarlar:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas
   - **Yayın:** Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 54:1273–1282 (2017)
   - **Önerilen Dosya Adı:** `05_McMahan_2017_FedAvg.pdf`
   - **İndirme Linki:** [Ücretsiz PDF – PMLR](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
   - **Neden Kullanıyoruz?** Çalışmamızda kullandığımız `FedAvg` algoritmasının kurucu makalesidir.

6. **`[6]` Kairouz, McMahan et al. (2021)**
   - **Başlık:** *Advances and Open Problems in Federated Learning*
   - **Yazarlar:** Peter Kairouz, H. Brendan McMahan, et al.
   - **Yayın:** Foundations and Trends® in Machine Learning, 14(1–2), 1–210 (2021)
   - **Önerilen Dosya Adı:** `06_Kairouz_2021_FL_Open_Problems.pdf`
   - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/1912.04977) | [DOI](https://doi.org/10.1561/2200000083)
   - **Neden Kullanıyoruz?** FL'de kullanıcı bazlı Non-IID veri dağılımı, gizlilik tehditleri ve iletişim maliyeti arasındaki temel ödünleşimleri referans göstermek için.

7. **`[7]` Albogamy (2025)**
   - **Başlık:** *Federated Learning for IoMT-Enhanced Human Activity Recognition with Hybrid LSTM-GRU Networks*
   - **Yazarlar:** Fahad R. Albogamy
   - **Yayın:** Sensors, 25(3), 907 (2025)
   - **Önerilen Dosya Adı:** `07_Albogamy_2025_FL_IoMT_HAR.pdf`
   - **İndirme Linki:** [Ücretsiz Tam Metin – MDPI](https://www.mdpi.com/1424-8220/25/3/907) | [Doğrudan PDF – MDPI](https://www.mdpi.com/1424-8220/25/3/907/pdf)
   - **Neden Kullanıyoruz?** Giyilebilir/IoMT sensörlerle federatif aktivite tanıma üzerine en güncel (2025) çalışmalardan biri olarak ilgili çalışmalarda karşılaştırma yapmak için.

8. **`[8]` Beutel et al. (2020) — [Flower Çerçevesi Referansı]**
   - **Başlık:** *Flower: A Friendly Federated Learning Research Framework*
   - **Yazarlar:** Daniel J. Beutel, Taner Topal, Akhil Mathur, Pedro Porto Buarque de Gusmão, Nicholas D. Lane, et al.
   - **Yayın:** arXiv:2007.14390 (2020)
   - **Önerilen Dosya Adı:** `08_Beutel_2020_Flower.pdf`
   - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2007.14390)
   - **Neden Kullanıyoruz?** Kodlama ve simülasyon altyapımızda kullandığımız **Flower** çerçevesinin resmî akademik referansıdır.

---

## Grup 3: Differential Privacy (DP) ve Gizlilik Muhasebesi (`[9]`–`[13]`)
*Bildiride Kullanım Yeri: Bölüm 2.3 (Derin Öğrenme ve FL'de DP), Bölüm 3.4 (DP-SGD ve RDP Muhasebesi), Bölüm 4 (Opacus)*

9. **`[9]` Abadi et al. (2016) — [DP-SGD Kurucu Makalesi]**
   - **Başlık:** *Deep Learning with Differential Privacy*
   - **Yazarlar:** Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang
   - **Yayın:** ACM SIGSAC Conference on Computer and Communications Security (CCS), 308–318 (2016)
   - **Önerilen Dosya Adı:** `09_Abadi_2016_DP_SGD.pdf`
   - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/1607.00133)
   - **Neden Kullanıyoruz?** Örnek bazlı gradyan kırpma ($C$) ve Gauss gürültüsü ekleme (`DP-SGD`) mekanizmasının temel makalesidir.

10. **`[10]` Mironov (2017) — [RDPAccountant Teorik Referansı]**
    - **Başlık:** *Rényi Differential Privacy*
    - **Yazarlar:** Ilya Mironov
    - **Yayın:** IEEE 30th Computer Security Foundations Symposium (CSF), 263–275 (2017)
    - **Önerilen Dosya Adı:** `10_Mironov_2017_Renyi_DP.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/1702.07476)
    - **Neden Kullanıyoruz?** Görev dokümanında zorunlu tutulan *privacy accountant* açıklamasında Opacus'un kullandığı **Rényi DP (RDP)** muhasebesini temellendirmek için.

11. **`[11]` McMahan et al. (2018) — [Örnek vs. İstemci Düzeyi DP Referansı]**
    - **Başlık:** *Learning Differentially Private Recurrent Language Models*
    - **Yazarlar:** H. Brendan McMahan, Daniel Ramage, Kunal Talwar, Li Zhang
    - **Yayın:** International Conference on Learning Representations (ICLR 2018)
    - **Önerilen Dosya Adı:** `11_McMahan_2018_Client_Level_DP.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – OpenReview](https://openreview.net/pdf?id=BJ0hF1Z0b)
    - **Neden Kullanıyoruz?** Görev dokümanında şart koşulan *"Örnek düzeyi (sample-level) DP-SGD ile İstemci düzeyi (user/client-level) DP arasındaki teorik sınır ve farkı"* açıklamak için.

12. **`[12]` Yousefpour et al. (2021) — [Opacus Kütüphanesi Referansı]**
    - **Başlık:** *Opacus: User-Friendly Differential Privacy Library in PyTorch*
    - **Yazarlar:** Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, et al.
    - **Yayın:** NeurIPS 2021 Workshop on Privacy in Machine Learning / arXiv:2109.12298
    - **Önerilen Dosya Adı:** `12_Yousefpour_2021_Opacus.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2109.12298)
    - **Neden Kullanıyoruz?** Kodlamada DP-SGD ve RDP bütçe takibi ($\varepsilon_1, \varepsilon_2, \delta$) için kullandığımız **Opacus** kütüphanesinin resmî makalesidir.

13. **`[13]` Saha et al. (2024)**
    - **Başlık:** *A Multifaceted Survey on Privacy Preservation of Federated Learning: Progress, Challenges, and Opportunities*
    - **Yazarlar:** Sanchita Saha, Ashlesha Hota, Arup Kumar Chattopadhyay, Amitava Nag, Sukumar Nandi
    - **Yayın:** Artificial Intelligence Review, 57, 184 (2024)
    - **Önerilen Dosya Adı:** `13_Saha_2024_Privacy_FL_Survey.pdf`
    - **İndirme Linki:** [Ücretsiz Tam Metin – Springer](https://link.springer.com/article/10.1007/s10462-024-10766-7) | [Doğrudan PDF – Springer](https://link.springer.com/content/pdf/10.1007/s10462-024-10766-7.pdf)
    - **Neden Kullanıyoruz?** FL'de DP eklendiğinde oluşan gizlilik-doğruluk (privacy-utility) kaybını güncel literatürle desteklemek için.

---

## Grup 4: Model Kuantizasyonu (FP32 → INT8 ve Çıkarım Verimliliği) (`[14]`–`[16]`)
*Bildiride Kullanım Yeri: Bölüm 2.4 (Model Kuantizasyonu), Bölüm 3.5 (FP32 → INT8 PTQ) ve Bölüm 5 (Boyut/Gecikme Analizi)*

14. **`[14]` Jacob et al. (2018) — [INT8 Çıkarım Kuantizasyonu Kurucu Makalesi]**
    - **Başlık:** *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*
    - **Yazarlar:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko
    - **Yayın:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2704–2713 (2018)
    - **Önerilen Dosya Adı:** `14_Jacob_2018_INT8_Quantization.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – CVF OpenAccess](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)
    - **Neden Kullanıyoruz?** FP32 ağırlık ve aktivasyonların ölçek (scale) ve sıfır noktası (zero-point) ile 8-bit tam sayıya (INT8) dönüştürülmesinin temel matematiğini verir.

15. **`[15]` Nagel et al. (2021) — [PTQ vs. QAT Temel Rehberi]**
    - **Başlık:** *A White Paper on Neural Network Quantization*
    - **Yazarlar:** Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort
    - **Yayın:** arXiv:2106.08295 (2021)
    - **Önerilen Dosya Adı:** `15_Nagel_2021_Quantization_WhitePaper.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2106.08295)
    - **Neden Kullanıyoruz?** Uyguladığımız *Post-Training Static Quantization (PTQ)* kalibrasyon sürecini ve katman hassasiyetini açıklamak için.

16. **`[16]` Gholami et al. (2022) — [Uç Cihazlarda Bellek ve Gecikme Referansı]**
    - **Başlık:** *A Survey of Quantization Methods for Efficient Neural Network Inference*
    - **Yazarlar:** Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer
    - **Yayın:** Low-Power Computer Vision, Chapman and Hall/CRC / arXiv:2103.13630 (2022)
    - **Önerilen Dosya Adı:** `16_Gholami_2022_Quantization_Inference_Survey.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2103.13630)
    - **Neden Kullanıyoruz?** INT8 kuantizasyonun model boyutunda (~4× küçülme) ve CPU çıkarım gecikmesinde (latency) sağladığı kazanımları teorik olarak desteklemek için.

---

## Grup 5: FL'de DP ve Kuantizasyon Kesişimi / Araştırma Boşluğu (`[17]`–`[20]`)
*Bildiride Kullanım Yeri: Bölüm 1 (Araştırma Boşluğu ve Katkı), Bölüm 2.5 (DP ve Kuantizasyon Birlikteliği), Bölüm 6 (Tartışma)*

17. **`[17]` Youn et al. (2023)**
    - **Başlık:** *Randomized Quantization is All You Need for Differential Privacy in Federated Learning*
    - **Yazarlar:** Yeojoon Youn, Zihao Hu, Juba Ziani, Jacob Abernethy
    - **Yayın:** arXiv:2306.11913 (2023)
    - **Önerilen Dosya Adı:** `17_Youn_2023_Randomized_Quantization_DP_FL.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2306.11913)
    - **Neden Kullanıyoruz?** FL'de kuantizasyon ve DP'nin birlikte incelendiği çalışmaları göstermek ve bizim çalışmamızın (uç cihazda INT8 model çıkarımı) bunlardan nasıl ayrıştığını vurgulamak için.

18. **`[18]` Guo et al. (2023)**
    - **Başlık:** *Privacy-Aware Compression for Federated Learning Through Numerical Mechanism Design*
    - **Yazarlar:** Chuan Guo, Kamalika Chaudhuri, Pierre Stock, Michael Rabbat
    - **Yayın:** International Conference on Machine Learning (ICML), PMLR 202:12086–12105 (2023)
    - **Önerilen Dosya Adı:** `18_Guo_2023_Privacy_Aware_Compression_FL.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – PMLR](https://proceedings.mlr.press/v202/guo23a/guo23a.pdf)
    - **Neden Kullanıyoruz?** Gizlilik gürültüsü ile sıkıştırma/kuantizasyon hatasının birleşik etkisini (joint noise effect) tartışmak için.

19. **`[19]` Hasırcıoğlu & Gündüz (2023)**
    - **Başlık:** *Communication Efficient Private Federated Learning Using Dithering*
    - **Yazarlar:** Burak Hasırcıoğlu, Deniz Gündüz
    - **Yayın:** arXiv:2309.07809 (2023)
    - **Önerilen Dosya Adı:** `19_Hasircioglu_2023_Private_FL_Dithering.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2309.07809)
    - **Neden Kullanıyoruz?** İletişim verimliliği ve diferansiyel gizlilik arasındaki ödünleşimi (trade-off) analiz etmek için.

20. **`[20]` Kang et al. (2024)**
    - **Başlık:** *The Effect of Quantization in Federated Learning: A Rényi Differential Privacy Perspective*
    - **Yazarlar:** Tianqu Kang, Lumin Liu, Hengtao He, Jun Zhang, S. H. Song, Khaled B. Letaief
    - **Yayın:** arXiv:2405.10096 (2024)
    - **Önerilen Dosya Adı:** `20_Kang_2024_Quantization_FL_RDP.pdf`
    - **İndirme Linki:** [Ücretsiz PDF – arXiv](https://arxiv.org/pdf/2405.10096)
    - **Neden Kullanıyoruz?** Rényi DP (RDP) çerçevesinde kuantizasyonun gizlilik ve yakınsama üzerindeki etkisini ele alan en güncel teorik çalışma olduğu için.

---

### 💡 Yedek (Bonus) Kaynak
Eğer dergi/konferans yazımında iletişim sıkıştırması için ek bir referans daha gerekirse elindeki şu makaleyi `[21]` olarak tutabilirsin:
- **Hegazy et al. (2024):** *Compression with Exact Error Distribution for Federated Learning*, AISTATS 2024 ([PDF – PMLR](https://proceedings.mlr.press/v238/hegazy24a/hegazy24a.pdf)).
