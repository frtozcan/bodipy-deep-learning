# BODIPY Floroforlarının Fotofiziksel Özelliklerinin Yorumlanabilir Çok-Görevli Derin Öğrenme ile Tahmini

**Taslak v0.1** — çalışma metni. Tüm sayılar `results/` klasöründen alınmıştır (5 tohumlu
topluluk ortalamaları). Doldurulacak yerler `[TODO]` ile işaretlidir.

**Yazarlar:** Fırat Özcan [+ diğer yazarlar TODO]
**Kurum:** Kırklareli Üniversitesi [TODO]

---

## Özet

BODIPY (4,4-difloro-4-bora-3a,4a-diaza-*s*-indasen) boyaları floresan problar, lazer boyaları
ve fotoduyarlaştırıcılar olarak yaygın biçimde kullanılmakta olup fotofizikleri büyük ölçüde
*mezo* ve 3,5 konumlarındaki sübstitüsyonla belirlenir. Bu özelliklerin yapıdan tahmin
edilebilmesi boya tasarımını hızlandıracaktır; ancak mevcut makine öğrenmesi çalışmalarının
çoğu yalnızca absorpsiyon veya emisyon maksimumlarını raporlamakta ve öğrenilen modellerin
bilinen yapı–özellik kurallarını içselleştirip içselleştirmediğini nadiren sınamaktadır. Bu
çalışmada dört mimariyi — parmak izi/tanımlayıcı temelli çok katmanlı algılayıcı (MLP), tek
boyutlu evrişimli ağ (1B-ESA), SMILES tabanlı dönüştürücü (Transformer) ve çizge dikkat ağı
(GATv2) — açık bir deneysel veri tabanından çıkarılan 607 BODIPY kromoforu (1.853
kromofor–çözücü kaydı) üzerinde, absorpsiyon maksimumu, emisyon maksimumu, floresans kuantum
verimi (Φ_F) ve molar soğurganlığı (log ε) **birlikte** tahmin eden çok-görevli regresyon
modelleri olarak eğittik. Kromofor temelli bölme ve 5 rastgele tohum kullanıldığında dalga
boyları yüksek doğrulukla tahmin edilmektedir (MLP: absorpsiyon için R² = 0,912 ± 0,007;
emisyon için 0,931 ± 0,009); buna karşılık Φ_F ve log ε belirgin biçimde daha zordur
(R² ≈ 0,21–0,54). Modeller daha sonra, eğitimde hiç görülmemiş olan ve bağımsız bir sentez
çalışmasından alınan sekiz adet 3,5-dialkil BODIPY üzerinde sınanmış; absorpsiyonda ~9,5–11,6
nm, Φ_F'de 0,126 ± 0,025 ortalama mutlak hata elde edilmiştir. Son olarak, *in-silico*
sübstitüent taraması modellerin kendilerine hiç öğretilmemiş fotofiziksel kuralları yeniden
ürettiğini göstermektedir: tüm modellerde *mezo*-alkil sübstitüsyonu en yüksek Φ_F değerini
vermekte, çizge ağı ise ağır atom serisinin tamamını (F > Cl > Br > I; iyot için Φ_F 0,080'e
kadar düşmektedir) yeniden üretmektedir. Dikkat (attention) haritaları ve SHAP katkı analizi,
model çıkarımını birbirinden bağımsız olarak BF₂ çekirdeği ve *mezo* bağlantısı üzerinde
konumlandırmaktadır. Dikkat çekici biçimde, en yüksek R² değerine sahip mimari kimyasal olarak
en tutarlı mimari değildir; bunun boya tasarımına yönelik model seçiminde önemli bir uyarı
olduğunu savunuyoruz.

**Anahtar kelimeler:** BODIPY; çok-görevli öğrenme; çizge sinir ağları; floresans kuantum
verimi; yorumlanabilirlik; SHAP

---

## 1. Giriş

Bor-dipirometen (4,4-difloro-4-bora-3a,4a-diaza-*s*-indasen, BODIPY) boyaları sentetik
floroforlar arasında merkezi bir yer tutar. Bu ilginin temelinde ender rastlanan bir özellik
birleşimi yatar: kimyasal ve fotokimyasal dayanıklılık, görünür bölgede yüksek molar
soğurganlık, küçük Stokes kaymalarıyla dar absorpsiyon ve emisyon bantları, bire yaklaşabilen
floresans kuantum verimleri ve çözücü polaritesine görece düşük duyarlılık [1–3]. Bunun kadar
önemli olarak BODIPY çekirdeği sentetik açıdan esnektir: *mezo* (8) ve 3,5 konumlarındaki
sübstitüsyon ya da π sisteminin genişletilmesi, spektrumları kaydırmakta ve emisyon verimini
denetimli biçimde değiştirmektedir. Bu ayarlanabilirlik; floresan problar ve biyo-işaretleme
belirteçleri, lazer boyaları, fotodinamik terapi için fotoduyarlaştırıcılar ve güneş
hücrelerinde ışık toplayıcı bileşenler gibi uygulamaların temelini oluşturur [1–3].

Bu olgunluğa karşın boya tasarımı büyük ölçüde deneysel kalmaktadır. Uygulamada en çok önem
taşıyan dört niceliğin — absorpsiyon ve emisyon maksimumları (λ_abs, λ_em), floresans kuantum
verimi (Φ_F) ve molar soğurganlık (ε) — belirlenmesi hâlâ sentez ve ardından spektroskopik
karakterizasyon gerektirir; dolayısıyla her yeni sübstitüsyon deseni laboratuvar zamanına mal
olur. Kuantum kimyasal hesaplamalar kısmi bir çözüm sunar: zamana bağlı yoğunluk fonksiyoneli
teorisi dikey uyarılma enerjilerini makul doğrulukla öngörür; ancak geniş tarama çalışmaları
için hesaplama maliyeti yüksektir ve özellikle Φ_F, ışımalı ve ışımasız kanallar (sistemler
arası geçiş ve sübstitüent dönmesi dâhil) arasındaki rekabetle belirlendiğinden rutin
hesaplamalarla doğrudan erişilebilir değildir. Niteliksel kurallar sentetik kimyacılarca iyi
bilinmektedir — elektron veren ve elektron çeken gruplar sınır orbital aralığını zıt yönlerde
kaydırır, brom veya iyot gibi ağır atomlar ise artan spin–yörünge çiftlenimi yoluyla
floresansı söndürür [4] — ancak bu kurallar tek başlarına niceliksel öngörü sağlamaz.


Veriye dayalı modelleme son yıllarda bu tabloyu önemli ölçüde değiştirmiştir. Absorpsiyon ve
emisyon maksimumları, bant genişlikleri, sönme katsayıları, kuantum verimleri ve ömürleri
kapsayan 20.000'den fazla kromofor–çözücü kaydından oluşan derlenmiş deneysel veri
tabanlarının yayımlanması [5], boya fotofiziği için denetimli öğrenmeyi uygulanabilir kılmış;
bu veriler üzerinde eğitilen derin modeller, kromofor–çözücü etkileşimini açıkça hesaba
katarak organik kromoforların absorpsiyon ve emisyon maksimumlarını birkaç on nanometrelik
karekök ortalama hatalarla öngörebilmektedir [6]. Bununla birlikte raporlanan doğruluk
özellikler arasında çarpıcı biçimde dengesizdir: spektral konumlar iyi öğrenilirken kuantum
verimi ve molar soğurganlık belirgin şekilde daha zor kalmakta ve bu örüntü farklı
çalışmalarda ve model ailelerinde yinelenmektedir.

Bu çalışmayı iki eksik motive etmektedir. Birincisi, söz konusu dört özellik aynı elektronik
yapıdan kaynaklanmasına ve aynı örnekler üzerinde ölçülmesine karşın genellikle **birbirinden
bağımsız** olarak modellenmektedir; çok-görevli bir kurgu ise gösterimi hedefler arasında
paylaştırabilir ve daha da önemlisi, yalnızca bazı özelliklerin raporlandığı kayıtlardan da
yararlanabilir. İkincisi ve daha temel olarak, modellerin **bilinen kimyayı** içselleştirip
içselleştirmediği nadiren sorgulanmaktadır. Raporlanan başarım ölçütleri ayrılmış bir test
kümesi üzerindeki aradeğerleme kalitesini ortaya koyar; ancak modelin, *mezo*-aril grubunun
*mezo*-alkil gruba kıyasla emisyonu söndürdüğünü ya da daha ağır halojenlerin Φ_F'yi
düşürdüğünü öğrenip öğrenmediğini göstermez. Bu tür modeller, eğitim derlemesinin dışında
sentezlenip karakterize edilmiş bileşiklerden oluşan tümüyle bağımsız bir sentez çalışmasıyla
— ileriye dönük doğrulamanın en yakın karşılığıyla — de çoğu zaman sınanmamaktadır.

Bu çalışmada her iki eksiği özel olarak BODIPY ailesi için ele alıyoruz. Açık deneysel veri
tabanından 607 benzersiz BODIPY kromoforu (1.853 kromofor–çözücü kaydı) çıkarıyor ve dört
mimari ailesini — parmak izi/tanımlayıcı temelli çok katmanlı algılayıcı, SMILES üzerinde tek
boyutlu evrişimli ağ ve dönüştürücü, ve çizge dikkat ağı (GATv2 [7]) — λ_abs, λ_em, Φ_F ve
log ε'yi birlikte tahmin eden, çözücü-farkındalıklı **çok-görevli** regresyon modelleri olarak
eğitiyoruz; maskeli bir yitim işlevi sayesinde kısmen etiketlenmiş kayıtlar da katkı
vermektedir. Tüm modeller özdeş, kromofor düzeyinde bölmelerle ve beş rastgele tohumla
karşılaştırılmakta, böylece bildirilen farklar tohumlar arası değişkenlikleriyle birlikte
sunulmaktadır. Ardından eğitilmiş modelleri, hiçbir değişiklik yapmadan, eğitimde bulunmayan
ve bağımsız bir sentez ile fotofizik çalışmasından [4] alınan sekiz adet 3,5-dialkil BODIPY
üzerinde uyguluyoruz. Son olarak modellerin ne öğrendiğini sorguluyoruz: in-siliko bir
sübstitüent taraması üç açık fotofiziksel hipotezi sınarken, dikkat çözümlemesi ve SHAP
katkı analizi [8] tahminleri yönlendiren öznitelikleri konumlandırmaktadır. Yinelenen bir
sonuç — R² bakımından en iyi sıralanan mimarinin, yerleşik kimyayla en tutarlı mimari
olmaması — modeller yalnızca aradeğerleme için değil boya tasarımına yön vermek için
kullanılacaksa model seçiminin nasıl yapılması gerektiği konusunda doğrudan sonuçlar
doğurmaktadır.

---

## 2. Yöntem

### 2.1 Veri kümesi

Deneysel fotofiziksel veriler, kromofor ve çözücü için SMILES gösterimleri içeren açık
Deep4Chem veri tabanından (`DB for chromophore`, figshare DOI 10.6084/m9.figshare.12045567)
[5,9] alınmıştır; veri tabanı 20.836 kromofor–çözücü kaydı, 6.865 benzersiz kromofor ve
1.363 çözücü içermektedir.

BODIPY kayıtları, BF₂–dipirometen çekirdeğine alt yapı eşleşmesiyle (SMARTS
`[F][B]([F])([#7])[#7]`) belirlenmiş ve **1.853 kromofor–çözücü kaydında 607 benzersiz
BODIPY kromoforu** elde edilmiştir. Hedef doluluk sayıları: λ_abs 1.803; λ_em 1.799;
Φ_F 1.692; log ε 1.130 kayıt. Molar soğurganlık boyunca log₁₀ε olarak modellenmiştir.

### 2.2 Veri bölme

Her kromofor ortalama ~3 çözücüde yer aldığından, kayıtlar satır bazında değil **kromofor
bazında** bölünmüştür; böylece aynı boyanın farklı çözücülerde hem eğitim hem test kümesinde
yer alması (veri sızıntısı) önlenmiştir. 70/15/15 oranındaki bölme (tohum 42) 424/91/92
molekül ve 1.307/284/262 kayıt vermiştir. Tüm BODIPY'ler aynı çekirdek iskeleti paylaştığı
için standart Bemis–Murcko iskelet bölmesi burada bilgi taşımaz [17]; molekül düzeyinde bölme
bunun uygun karşılığıdır.

### 2.3 Moleküler gösterimler

Aynı kayıtlardan dört gösterim türetilmiştir:

- **Çizge** (GATv2 için): atomlar 17 boyutlu vektörler (C/N/O/F/B/S/Cl/Br + diğer için
  element one-hot; derece; biçimsel yük; toplam H sayısı; SP/SP2/SP3 + diğer için
  melezleşme one-hot; aromatiklik bayrağı), bağlar ise 7 boyutlu vektörler (tekli/ikili/
  üçlü/aromatik bağ türü one-hot; konjugasyon; halka üyeliği; stereo bayrağı) olarak
  kodlanmış, kenarlar çift yönlü tanımlanmıştır.
- **Parmak izi/tanımlayıcı** (MLP için): 2.048 bitlik Morgan (ECFP benzeri) parmak izi [11]
  (yarıçap 2) ve on adet RDKit [12] tanımlayıcısı (MolWt, MolLogP, TPSA, NumHAcceptors,
  NumHDonors, NumRotatableBonds, NumAromaticRings, FractionCSP3, RingCount, NumHeteroatoms).
- **Dizi** (1B-ESA ve dönüştürücü için): `kromofor [SEP] çözücü` dizgesinin atom düzeyinde
  düzenli ifade ile belirteçlenmesi. **SMILES gösterimleri belirteçleme öncesinde RDKit ile
  kanonikleştirilmiştir**; bunun kritik olduğu görülmüştür (Bölüm 3.3).
- **Çözücü**: aynı on tanımlayıcı çözücü SMILES'i üzerinde hesaplanarak MLP girdisine ve
  GATv2'nin çizge düzeyindeki okuma katmanına eklenmiştir.

Tüm standartlaştırma istatistikleri (hedefler ve tanımlayıcılar) yalnızca eğitim kümesinden
hesaplanmıştır.

### 2.4 Mimariler

| Model | Girdi | Çekirdek |
|---|---|---|
| MLP | 2.068 boyut (Morgan + 10 + 10) | 512→256, BatchNorm, ReLU, dropout 0,2 |
| 1B-ESA | belirteç kimlikleri (≤256) | gömme 64; paralel evrişimler k = 3/5/7, 128 kanal; global maks-havuzlama |
| Dönüştürücü | belirteç kimlikleri (≤256) | gömme 128, 4 başlık, 3 kodlayıcı katmanı, maskeli ortalama havuzlama |
| GATv2 | çizge + çözücü vektörü | 3 × GATv2Conv (128, 4 başlık, kenar öznitelikli), ortalama havuzlama |

Her modelde paylaşılan bir gövde ve **dört çıkışlı bir regresyon başlığı** (çok-görevli)
bulunmaktadır.

### 2.5 Eğitim ve değerlendirme

Hedefler eğitim kümesi istatistikleriyle standartlaştırılmıştır. Kayıtların çoğunda en az bir
özellik eksik olduğundan **maskeli ortalama karesel hata** kullanılmıştır: yitim yalnızca
gözlenen hedefler üzerinden ortalanır, böylece kısmen etiketlenmiş kayıtlar da katkı verir.
Eniyileme Adam [15] ile (öğrenme oranı 1e-3, ağırlık sönümü 1e-5), yığın boyutu 256 (MLP) ya
da 128 olacak şekilde, en çok 300 dönem boyunca ve doğrulama yitimi üzerinden erken durdurma
(sabır 30) ile yapılmıştır. Modeller PyTorch [13] ile, çizge modeli için PyTorch Geometric
[14] kullanılarak gerçeklenmiştir. Başarım ölçütleri (R², RMSE, MAE) standartlaştırma geri
alındıktan sonra özgün birimlerde ve hedef bazında hesaplanmıştır. Tüm deneyler veri bölmesi
sabit tutularak **5 tohumla (0–4)** yinelenmiştir; dolayısıyla bildirilen değişkenlik
başlangıç değerleri ve eğitim rastgeleliğini yansıtır ve sonuçlar ortalama ± standart sapma
olarak verilmiştir.

### 2.6 Harici doğrulama kümesi

Sekiz adet 3,5-dialkil BODIPY (**1A–4B**) bağımsız bir sentez ve fotofizik çalışmasından [4]
alınmıştır; bileşikler *mezo* konumunda etil (1), fenil (2), 4-metoksifenil (3) ve
4-bromofenil (4) gruplarını, 3,5 konumunda ise dimetil (A) veya dietil (B) sübstitüsyonunu
taşımaktadır. SMILES gösterimleri bildirilen yapılardan oluşturulmuş ve RDKit ile hesaplanan
moleküler formüllerin yayımlanmış formüllerle eşleşmesiyle doğrulanmıştır (8/8 tam eşleşme).
CHCl₃ içinde ölçülen λ_abs, λ_em, Φ_F ve ε değerleri gerçek değer olarak kullanılmıştır.

Eğitim verisiyle örtüşme kanonik SMILES üzerinden denetlenmiştir: **sekiz bileşiğin hiçbiri
eğitim kümesinde bulunmamaktadır**; altısı Deep4Chem'de hiç yer almamakta, ikisi (2A, 3A) ise
yalnızca ayrılmış bölümlerde görülmektedir. Dolayısıyla sekiz bileşiğin tamamı modeller
açısından gerçekten görülmemiştir.

### 2.7 Yorumlanabilirlik

**In-siliko sübstitüent taraması.** 3,5-dialkil BODIPY iskeleti sabit tutularak *mezo*
sübstitüenti; alkil (metil, etil, n-propil), sübstitüe olmamış fenil, elektron veren aril
(4-NMe₂, 4-OMe, 4-Me), halojen serisi (4-F, 4-Cl, 4-Br, 4-I) ve elektron çeken aril
(4-CN, 4-CF₃, 4-NO₂) olmak üzere 14 grup üzerinde, her biri 3,5-Me ve 3,5-Et ile
(28 yapı) değiştirilmiş ve Φ_F, CHCl₃ içinde 5 tohumlu topluluklarla tahmin edilmiştir.
Yerleşik BODIPY fotofiziğinden üç hipotez sınanmıştır:

- **H1** *mezo*-alkil, *mezo*-arile kıyasla daha yüksek Φ_F verir;
- **H2** elektron veren aril > fenil > elektron çeken aril;
- **H3** Φ_F, F > Cl > Br > I boyunca tekdüze azalır (ağır atom etkisi).

**Dikkat haritaları.** GATv2 için atom başına önem, üç evrişim katmanı boyunca, her atomun
komşularından *aldığı* dikkat (kaynak yönü) olarak biriktirilmiş; başlıklar ve tohumlar
üzerinden ortalanmıştır. GATv2 dikkati hedef düğüm başına softmax ile normalleştirdiğinden,
gelen dikkatin toplanmasının bilgi taşımadığına dikkat edilmelidir.

**SHAP.** Φ_F için öznitelik katkıları MLP üzerinde GradientExplainer ile (200 eğitim arka
plan örneği, test kümesinde değerlendirme, 5 tohum üzerinden ortalama) hesaplanmıştır. Yüksek
katkılı Morgan bitleri, RDKit bit bilgisi aracılığıyla alt yapılara geri eşlenmiştir.

---

## 3. Bulgular ve tartışma

### 3.1 Ayrılmış BODIPY kümesinde çok-görevli karşılaştırma

Tablo 1, dört mimari için test kümesi R² değerlerini (5 tohum üzerinden ortalama ± standart
sapma) vermektedir. Absorpsiyon ve emisyon maksimumları, dönüştürücü dışındaki tüm modeller
tarafından iyi tahmin edilmekte; MLP hem en doğru hem de açık ara en kararlı model olarak öne
çıkmaktadır (λ_abs R² = 0,912 ± 0,007; λ_em 0,931 ± 0,009). Kuantum verimi ve log ε ise tüm
modeller için belirgin biçimde daha zordur (R² ≈ 0,2–0,55). Hiçbir mimari her özellikte üstün
değildir: MLP dalga boylarında, 1B-ESA Φ_F'de, GATv2 ise log ε'de öndedir. Dolayısıyla mimari
seçimi tek bir başlık değere göre değil, ilgilenilen hedefe göre yapılmalıdır (Şekil 1).

**Tablo 1.** İç test kümesi R² değerleri (ortalama ± s.s., 5 tohum; kromofor temelli bölme).

| Model | λ_abs | λ_em | Φ_F | log ε |
|---|---|---|---|---|
| MLP | **0,912 ± 0,007** | **0,931 ± 0,009** | 0,508 ± 0,037 | 0,517 ± 0,029 |
| 1B-ESA | 0,885 ± 0,016 | 0,899 ± 0,005 | **0,543 ± 0,046** | 0,327 ± 0,054 |
| GATv2 | 0,865 ± 0,025 | 0,888 ± 0,039 | 0,435 ± 0,049 | **0,527 ± 0,068** |
| Dönüştürücü | 0,671 ± 0,042 | 0,644 ± 0,049 | 0,213 ± 0,084 | 0,296 ± 0,052 |

Dönüştürücünün göreli zayıflığı veri gereksinimiyle tutarlıdır: yaklaşık 1.300 eğitim kaydıyla
tanımlayıcı ve çizge temelli modellerin gerisinde kalması, küçük veri rejiminde beklenen bir
davranıştır. Φ_F ve log ε'nin zor hedefler olması da kimyasal olarak anlamlıdır: BODIPY Φ_F
dağılımı belirgin biçimde iki tepelidir (sıfıra yakın geniş bir ışımasız topluluk ve bire
yakın ışımalı bir topluluk) ve bu alt kümede log ε aralığı dardır; dolayısıyla bir regresyon
modelinin yararlanabileceği değişkenlik sınırlıdır.

### 3.2 Bağımsız bileşiklerde harici doğrulama

Eğitilmiş toplulukları, hiçbir değişiklik yapmadan, bağımsız sentez çalışmasındaki sekiz
3,5-dialkil BODIPY'ye uyguladık (Bölüm 2.6). Bu bileşikler bilinçli olarak dar bir optik
pencereye yayıldığından (λ_abs 508–515 nm; λ_em 514–529 nm), söz konusu sınama esas olarak
dalga boyları için mutlak kalibrasyonun, ölçülen aralığı geniş olan Φ_F (0,24–1,00) için ise
sıralamanın testidir. Ortalama mutlak hatalar Tablo 2'de toplanmıştır (Şekil 2).

**Tablo 2.** Harici küme ortalama mutlak hataları (ortalama ± s.s., 5 tohum; 8 bileşik).

| Model | λ_abs (nm) | λ_em (nm) | Φ_F | log ε |
|---|---|---|---|---|
| MLP | **9,6 ± 2,6** | 20,7 ± 4,5 | 0,241 ± 0,022 | 0,069 ± 0,017 |
| 1B-ESA | 11,6 ± 3,6 | 19,3 ± 3,1 | **0,126 ± 0,025** | **0,058 ± 0,008** |
| GATv2 | 10,4 ± 2,4 | 20,6 ± 4,2 | 0,171 ± 0,022 | 0,063 ± 0,027 |
| Dönüştürücü | 9,5 ± 3,6 | **10,4 ± 3,2** | 0,204 ± 0,030 | 0,097 ± 0,014 |

Tümüyle görülmemiş bu boyaların absorpsiyon maksimumları ~9–12 nm doğrulukla yeniden
üretilmektedir; bu değer laboratuvarlar arası spektroskopik değişkenlik ölçeğine
yaklaşmaktadır. İç testte en zayıf model olan dönüştürücünün en düşük emisyon hatasını
(10,4 ± 3,2 nm) vermesi, iç ve dış sıralamaların ayrışabileceğini hatırlatmaktadır. Sekiz
boyanın Φ_F'ye göre sıralanmasında Spearman ρ değeri en yüksek ve en kararlı biçimde 1B-ESA
(0,814 ± 0,080) ve GATv2 (0,719 ± 0,191) için elde edilmiştir; buna karşılık dalga boyu ve
log ε sıralamalarına ait ρ değerleri tohumlar arasında çok büyük yayılım göstermekte ve işaret
değiştirmektedir. Yalnızca sekiz bileşik üzerinden bu değerlerin aşırı yorumlanmaması gerekir
(Bölüm 3.7).


### 3.3 Dizi modelleri için SMILES kanonikleştirmesi zorunludur

İlk harici değerlendirme, gösterime bağlı büyük bir başarısızlığı açığa çıkarmıştır. Sekiz
harici bileşiğin SMILES gösterimleri yüklü Kekulé biçiminde ([N⁺]/[B⁻]) yazılmıştı; bu
dizgiler üzerinde 1B-ESA, iç testteki güçlü başarımına karşın yaklaşık 104 nm'lik bir
absorpsiyon ortalama mutlak hatası üretmiştir. Buna karşılık gösterimden bağımsız
özniteliklerle çalışan tanımlayıcı temelli (MLP) ve çizge temelli (GATv2) modeller
etkilenmemiştir. Tüm SMILES gösterimlerinin hem eğitim hem çıkarım aşamasında, jetonlaştırma
öncesinde RDKit ile kanonikleştirilmesi, 1B-ESA hatasını yaklaşık 9 nm'ye düşürmüştür. Bu
nedenle dizi modelleri, çizge ve tanımlayıcı temelli modellerin olmadığı ölçüde keyfî SMILES
yazım geleneklerine duyarlıdır; tutarlı kanonikleştirme hem adil değerlendirmenin hem de
dışarıdan sağlanan yapılar üzerinde kullanımın ön koşuludur.

### 3.4 Modeller bilinen yapı–özellik kurallarını yeniden üretmektedir

Bu çalışmanın merkezî sorusu, modellerin yalnızca aradeğerleme yapmak yerine BODIPY
fotofiziğini içselleştirip içselleştirmediğidir. Bilgisayar ortamındaki *mezo*-sübstitüent
taraması (Şekil 3, Tablo 3) üç yerleşik kuralı doğrudan sınamaktadır. **H1** (Φ_F'de
*mezo*-alkil ≫ *mezo*-aril) üç topluluğun tamamında yeniden üretilmiştir; bu, harici kümedeki
alkil-*mezo* referans boyalarının olağanüstü yüksek Φ_F değerlerinin altında yatan etkinin ta
kendisidir. Ağır atom serisi olan **H3** (F > Cl > Br > I) 1B-ESA tarafından ve en belirgin
biçimde GATv2 tarafından yeniden üretilmiştir; GATv2'nin öngördüğü Φ_F değeri 0,304'ten (F)
0,080'e (I) düşmektedir. Modellere hangi atomların ağır olduğu hiçbir biçimde öğretilmediğinden
bu tekdüze azalış yalnızca veriden çıkarılmıştır ve kaynak çalışmada belirtilen brom kaynaklı
ağır atom etkisini iyoda doğru genişletmektedir. **H2** (elektron veren aril > fenil > elektron
çeken aril) yalnızca GATv2'de tümüyle sağlanmaktadır; diğer modellerde verici/fenil sıralaması
zayıftır; bununla birlikte elektron çeken aril grubu her modelde en düşük Φ_F değerini
vermektedir.

**Tablo 3.** *Mezo* sınıfına göre topluluk ortalaması öngörülen Φ_F değerleri ve hipotez
sonuçları (✓ sağlandı, ✗ sağlanmadı). Halojen sütunları 4-halofenil serisine aittir.

| Model | alkil | EVG | fenil | EÇG | F | Cl | Br | I | H1 | H2 | H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1B-ESA | 0,642 | 0,292 | 0,374 | 0,207 | 0,297 | 0,287 | 0,214 | 0,199 | ✓ | ✗ | ✓ |
| GATv2 | 0,584 | 0,331 | 0,319 | 0,258 | 0,304 | 0,240 | 0,213 | **0,080** | ✓ | ✓ | ✓ |
| MLP | 0,448 | 0,299 | 0,403 | 0,229 | 0,209 | 0,201 | 0,236 | 0,110 | ✓ | ✗ | ✗ |

(EVG: elektron veren grup; EÇG: elektron çeken grup.)

### 3.5 Dikkat ve SHAP analizleri BF₂ çekirdeği ile *mezo* konumunu işaret etmektedir

Birbirinden bağımsız iki katkı yöntemi, modellerin nereye baktığı konusunda uyuşmaktadır.
GATv2 dikkati (Şekil 4) BF₂ şelatı üzerinde yoğunlaşmakta (ortalama atom önemi çevresel
atomlar için 0,45 iken bu bölge için 0,55; bor atomu her bileşikte ilk üç sırada yer
almaktadır) ve aril-*mezo* boyalarında *mezo* köprü karbonuna odaklanmaktadır;
4-metoksi türevlerinde metoksi oksijeni de öne çıkmaktadır. MLP'nin Φ_F çıktısı için yapılan
SHAP analizi (Şekil 5) ise **çözücü tanımlayıcılarının baskın olduğunu** göstermektedir:
çözücü TPSA değeri, herhangi bir kromofor özniteliğinin önünde, tek başına en önemli
özniteliktir (ortalama |SHAP| 0,113 ± 0,018). Bu sonuç hem BODIPY Φ_F'sinin bilinen çözücü
duyarlılığını yansıtmakta hem de çözücü-farkındalıklı girdi tasarımını gerekçelendirmektedir.
En önemli Morgan bitlerinin alt yapılara geri eşlenmesi, BODIPY çekirdeğinin kendi
parçalarını ortaya çıkarmaktadır: *mezo* köprüsü/dipirometen motifi
`cc(c)C(=C(C)[N⁺])c(c)n`, BF₂–N bağlantısı `[B⁻]n(c)c` ve 3,5-alkil-pirol birimi `cc(C)n`.
Bir çizge modelinden gelen dikkat analizi ile bir tanımlayıcı modelinden gelen SHAP
analizinin aynı BF₂/*mezo* bölgesinde buluşması, mekanistik yorumu güçlendirmektedir.

### 3.6 Doğruluk ve kimyasal tutarlılık birbirinden ayrı eksenlerdir

Çalışma boyunca yinelenen bir bulgu, en yüksek R² değerine sahip modelin kimyasal açıdan en
sadık model olmadığıdır. GATv2, üç sübstitüent hipotezinin tamamını sağlayan tek mimaridir ve
en temiz ağır atom eğilimi ile dikkat haritalarını üretmektedir; buna karşılık dalga boyu
R² değeri MLP'de daha yüksektir. Modelin tasarım üzerine *akıl yürütmek* — yeni sübstitüentlere
dışdeğerleme yapmak — amacıyla kullanıldığı uygulamalarda kimyasal tutarlılık, marjinal bir
R² kazancından daha önemli olabilir; bu nedenle her iki ölçütün birlikte raporlanmasını
öneriyoruz.

### 3.7 Sınırlılıklar

Birkaç sınırlılığın açıkça belirtilmesi gerekir. (i) Harici küme yalnızca sekiz bileşik
içermekte ve dar bir optik aralığa yayılmaktadır; bu küme üzerinde dalga boyu ve log ε
sıralamalarına ait korelasyonlar tohumlar arasında çok büyük yayılım göstermekte ve işaret
değiştirmektedir. Dolayısıyla yalnızca Φ_F sıralaması ve dalga boyu ortalama mutlak hataları
sağlam sonuçlar olarak değerlendirilmelidir. (ii) Bilgisayar ortamındaki tarama, model
*öngörülerini*, yani hipotezleri raporlamaktadır; iyot ve elektron çeken grup durumlarına
ilişkin dışdeğerlemeler kimyasal olarak makul olmakla birlikte bu çalışmada deneysel olarak
doğrulanmamıştır. (iii) Φ_F ve log ε zor hedefler olmayı sürdürmektedir (R² ≲ 0,55);
özellikle Φ_F'nin iki tepeli dağılımı düz bir regresyon modelini sınırlamaktadır ve problemin
iki aşamalı (ışımalı/ışımasız) biçimde modellenmesi umut verici bir yöndür. (iv) Tüm veriler
tek bir veri tabanından türetilmiştir; kaynak çalışmalar arasındaki ölçüm heterojenliği
kaçınılmazdır. (v) Dizi modelleri bu ölçekte veriye açtır; tüm kromofor veri tabanından
transfer öğrenme ve ardından BODIPY üzerinde ince ayar yapılması doğal bir sonraki adımdır ve
en çok dönüştürücüye yarar sağlaması beklenmektedir.

---

## 4. Sonuçlar

Bu çalışmada, yalnızca doğruluk açısından değil kimyasal sadakat açısından da değerlendirilen,
çözücü-farkındalıklı ve çok-görevli bir BODIPY fotofiziği derin öğrenme incelemesi
sunulmuştur. Kromofor temelli katı bölme ve tohum ortalamalı istatistikler altında,
tanımlayıcı ve çizge temelli modeller görülmemiş BODIPY'lerin absorpsiyon ve emisyon
maksimumlarını yaklaşık 6–12 nm doğrulukla tahmin etmekte; kuantum verimi ve molar soğurganlık
ise daha zor kalmaktadır. Doğruluğun ötesinde modeller, yerleşik yapı–özellik kurallarını
gözetimsiz biçimde yeniden üretmektedir: *mezo*-alkil grubunun *mezo*-arile üstünlüğü ve
kendiliğinden ortaya çıkan F > Cl > Br > I ağır atom serisi. Dikkat ve SHAP analizleri, model
çıkarımını birbirinden bağımsız olarak BF₂ çekirdeği ve *mezo* konumunda konumlandırmakta;
çözücü tanımlayıcıları ise Φ_F'nin başat belirleyicisi olarak öne çıkmaktadır. Çizge dikkat
ağı, R² açısından en doğru model olmamakla birlikte kimyasal olarak en tutarlı modeldir; boya
tasarımına yön vermesi amaçlanan modellerde her iki eksenin de raporlanması gerektiğini
savunuyoruz. Yakın vadeli uzantılar, daha geniş kromofor uzayından transfer öğrenme ve iki
tepeli kuantum verimi dağılımının açıkça ele alınmasıdır.

---

## Veri ve kod erişilebilirliği

Deneysel veriler açık Deep4Chem veri tabanından alınmıştır (figshare
10.6084/m9.figshare.12045567). Tüm kodlar (veri hazırlama, öznitelik çıkarımı, dört model,
çok tohumlu eğitim, harici doğrulama ve yorumlanabilirlik analizleri) ile türetilen sonuç
tabloları ve şekiller [depo adresi — TODO] adresinde erişime açıktır.

## Yazar katkıları

[TODO]

## Teşekkür

[TODO — destekleyen kurum, TÜBİTAK/BAP varsa]

---

## Kaynaklar

*Genel bir biçemde düzenlenmiştir; gönderim öncesinde hedef derginin biçemine dönüştürülmelidir.*

1. Loudet, A.; Burgess, K. BODIPY dyes and their derivatives: syntheses and spectroscopic properties. *Chem. Rev.* **2007**, *107*, 4891–4932.
2. Ulrich, G.; Ziessel, R.; Harriman, A. The chemistry of fluorescent bodipy dyes: versatility unsurpassed. *Angew. Chem. Int. Ed.* **2008**, *47*, 1184–1201.
3. Boens, N.; Leen, V.; Dehaen, W. Fluorescent indicators based on BODIPY. *Chem. Soc. Rev.* **2012**, *41*, 1130–1172.
4. Derin, Y.; Yılmaz, R. F.; Baydilek, İ. H.; Enisoğlu Atalay, V.; Özdemir, A.; Tutar, A. Synthesis, electrochemical/photophysical properties and computational investigation of 3,5-dialkyl BODIPY fluorophores. *Inorg. Chim. Acta* **2018**, *482*, 130–135. DOI: 10.1016/j.ica.2018.06.006.
5. Joung, J. F.; Han, M.; Jeong, M.; Park, S. Experimental database of optical properties of organic compounds. *Sci. Data* **2020**, *7*, 295. DOI: 10.1038/s41597-020-00634-8.
6. Joung, J. F.; Han, M.; Hwang, J.; Jeong, M.; Choi, D. H.; Park, S. Deep learning optical spectroscopy based on experimental database: potential applications to molecular design. *JACS Au* **2021**, *1*, 427–438. DOI: 10.1021/jacsau.1c00035.
7. Brody, S.; Alon, U.; Yahav, E. How attentive are graph attention networks? *International Conference on Learning Representations (ICLR)*, **2022**. arXiv:2105.14491.
8. Lundberg, S. M.; Lee, S.-I. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*, **2017**, 4765–4774.
9. Joung, J. F.; Han, M.; Jeong, M.; Park, S. DB for chromophore. *figshare* **2020**. DOI: 10.6084/m9.figshare.12045567.
10. Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. *J. Chem. Inf. Comput. Sci.* **1988**, *28*, 31–36.
11. Rogers, D.; Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **2010**, *50*, 742–754.
12. Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org (erişim 2026).
13. Paszke, A.; Gross, S.; Massa, F.; ve ark. PyTorch: an imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems 32 (NeurIPS)*, **2019**, 8024–8035.
14. Fey, M.; Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*, **2019**. arXiv:1903.02428.
15. Kingma, D. P.; Ba, J. Adam: a method for stochastic optimization. *International Conference on Learning Representations (ICLR)*, **2015**. arXiv:1412.6980.
16. Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. Graph attention networks. *International Conference on Learning Representations (ICLR)*, **2018**. arXiv:1710.10903.
17. Bemis, G. W.; Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* **1996**, *39*, 2887–2893.
18. Vaswani, A.; Shazeer, N.; Parmar, N.; ve ark. Attention is all you need. *Advances in Neural Information Processing Systems 30 (NeurIPS)*, **2017**, 5998–6008.

---

## Şekiller ve tablolar

- **Şekil 1** — Mimarilerin karşılaştırması, iç test R² (ortalama ± s.s.). `results/multiseed_R2.png`
- **Şekil 2** — Harici doğrulama: sekiz BODIPY için tahmin–ölçüm karşılaştırması. `results/external_parity.png`
  (sıra korelasyonu özeti: `results/multiseed_external_rho.png`)
- **Şekil 3** — In-siliko *mezo*-sübstitüent taraması: sübstitüente göre tahmin edilen Φ_F
  (`results/probe_QY.png`) ve halojen serisi (`results/probe_heavy_atom.png`)
- **Şekil 4** — GATv2 dikkat temelli atom önem haritaları. `results/attention_maps.png`
- **Şekil 5** — Φ_F için SHAP öznitelik katkıları (MLP). `results/shap_summary.png`
- **Şekil E1** — BODIPY alt kümesinin hedef dağılımları. `results/eda_bodipy_dist.png`
- **Tablo 1** — İç test R² değerleri. `results/multiseed_internal_summary.csv`
- **Tablo 2** — Harici küme ortalama mutlak hataları. `results/multiseed_external_summary.csv`
- **Tablo 3** — Sübstitüent taraması hipotez sonuçları. `results/probe_summary.csv`
- **Tablo E1** — Bileşik bazında harici tahminler. `results/external_predictions.csv`
- **Tablo E2** — En yüksek SHAP katkılı Morgan bitleri → alt yapılar. `results/shap_bits.csv`
