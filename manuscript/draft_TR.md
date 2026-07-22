# BODIPY Floroforlarının Fotofiziksel Özelliklerinin Yorumlanabilir Çok-Görevli Derin Öğrenme ile Tahmini

**Taslak v0.1 (Türkçe)** — çalışma metni. Tüm sayısal değerler `results/` klasöründen
(5-seed toplulukları). Doldurulacak yerler `[TODO]` ile işaretlidir.

**Yazarlar:** Fırat Özcan [+ diğer yazarlar TODO]
**Kurum:** Kırklareli Üniversitesi [TODO]

---

## Özet

BODIPY (4,4-difloro-4-bora-3a,4a-diaza-*s*-indasen) boyaları floresan problar, lazer
boyaları ve fotoduyarlaştırıcılar olarak yaygın biçimde kullanılmakta olup fotofizikleri
büyük ölçüde *mezo* ve 3,5 konumlarındaki sübstitüsyonla belirlenir. Bu özelliklerin yapıdan
tahmin edilmesi boya tasarımını hızlandırabilir; ancak mevcut makine öğrenmesi çalışmalarının
çoğu yalnızca absorpsiyon veya emisyon maksimumunu raporlamakta ve öğrenilen modellerin
bilinen yapı–özellik kurallarını kodlayıp kodlamadığını nadiren sınamaktadır. Bu çalışmada
dört mimariyi — parmak izi/tanımlayıcı tabanlı çok katmanlı algılayıcı (MLP), tek boyutlu
evrişimli ağ (1D-CNN), SMILES tabanlı transformer ve çizge dikkat ağı (GATv2) — açık bir
deneysel veri tabanından çıkarılan **607 BODIPY kromoforu** (1.853 kromofor–çözücü çifti)
üzerinde, absorpsiyon maksimumu, emisyon maksimumu, floresans kuantum verimi (Φ_F) ve molar
absorptiviteyi (log ε) **birlikte tahmin eden çok-görevli regresyon modelleri** olarak
eğittik. Kromofor temelli bölme ve 5 rastgele tohum kullanıldığında dalga boyları yüksek
doğrulukla tahmin edilmektedir (MLP: absorpsiyon için R² = 0,912 ± 0,007; emisyon için
0,931 ± 0,009); buna karşılık Φ_F ve log ε belirgin biçimde daha zordur (R² ≈ 0,21–0,54).
Modelleri, eğitimde hiç görülmemiş, bağımsız bir sentez çalışmasından alınan sekiz
3,5-dialkil BODIPY üzerinde doğruladık ve absorpsiyonda ~9,5–11,6 nm, Φ_F'de 0,126 ± 0,025
ortalama mutlak hata elde ettik. Son olarak, in-siliko sübstitüent taraması modellerin
yerleşik fotofiziksel kuralları kendilerine öğretilmeksizin yeniden ürettiğini
göstermektedir: *mezo*-alkil sübstitüsyonu tüm modellerde en yüksek Φ_F'yi vermekte, çizge
ağı ise ağır atom serisinin tamamını (F > Cl > Br > I; iyot için Φ_F 0,080'e düşerek) yeniden
üretmektedir. Dikkat haritaları ve SHAP atıfları, model çıkarımını birbirinden bağımsız
olarak BF₂ çekirdeği ve *mezo* bağlantı noktasında yoğunlaştırmaktadır. Dikkat çekici
biçimde, en yüksek R²'ye sahip mimari kimyasal olarak en tutarlı olan değildir; bunun boya
tasarımına yönelik model seçiminde önemli bir uyarı olduğunu savunuyoruz.

**Anahtar kelimeler:** BODIPY; çok-görevli öğrenme; çizge sinir ağları; floresans kuantum
verimi; yorumlanabilirlik; SHAP

---

## 1. Giriş

Bor-dipirometen (4,4-difloro-4-bora-3a,4a-diaza-*s*-indasen, BODIPY) boyaları sentetik
floroforlar arasında merkezi bir yer tutar. Bu ilgi, ender rastlanan bir özellik
birleşiminden kaynaklanır: kimyasal ve fotokimyasal dayanıklılık, görünür bölgede yüksek
molar absorptivite, küçük Stokes kaymalarıyla dar absorpsiyon ve emisyon bantları, birime
yaklaşabilen floresans kuantum verimleri ve çözücü polaritesine görece düşük duyarlılık
[1–3]. Aynı ölçüde önemli olarak, BODIPY çekirdeği sentetik açıdan esnektir: *mezo* (8)
konumundaki ve 3,5 konumlarındaki sübstitüsyon ya da π sisteminin genişletilmesi, spektrumu
kaydırır ve emisyon verimliliğini denetimli biçimde ayarlar. Bu ayarlanabilirlik, floresan
prob ve biyo-işaretleme reaktifi, lazer boyası, fotodinamik terapi için fotoduyarlaştırıcı
ve güneş hücrelerinde ışık toplayıcı bileşen olarak kullanımların temelini oluşturur [1–3].

Bu olgunluğa karşın boya tasarımı büyük ölçüde deneysel kalmıştır. Uygulamada en çok önem
taşıyan dört büyüklüğün — absorpsiyon ve emisyon maksimumları (λ_abs, λ_em), floresans
kuantum verimi (Φ_F) ve molar absorptivite (ε) — belirlenmesi hâlâ sentez ve ardından
spektroskopik karakterizasyon gerektirir; dolayısıyla her yeni sübstitüsyon deseni
laboratuvar zamanına mal olur. Kuantum kimyasal hesaplama kısmi bir çözüm sunar: zamana
bağlı yoğunluk fonksiyoneli teorisi dikey uyarılma enerjilerini makul doğrulukla öngörür,
ancak geniş ölçekli tarama çalışmaları için hesaplama maliyeti yüksektir ve özellikle Φ_F
rutin hesaplamalardan doğrudan elde edilemez; çünkü Φ_F, sistemler arası geçiş ve sübstitüent
dönmesi gibi ışımasız kanallarla ışımalı kanal arasındaki rekabetle belirlenir. Nitel
kurallar sentetik kimyacılarca iyi bilinir — elektron veren ve elektron çeken gruplar sınır
orbital aralığını zıt yönlerde kaydırır, brom veya iyot gibi ağır atomlar ise güçlenen
spin–yörünge etkileşimi yoluyla floresansı söndürür [4] — ancak bu kurallar tek başlarına
nicel öngörü sağlamaz.

Veriye dayalı modelleme son yıllarda bu tabloyu belirgin biçimde değiştirmiştir. Absorpsiyon
ve emisyon maksimumlarını, bant genişliklerini, ekstinksiyon katsayılarını, kuantum
verimlerini ve ömürleri kapsayan 20.000'den fazla kromofor–çözücü kaydından oluşan derlemeler
[5] başta olmak üzere büyük ölçekli deneysel veri tabanlarının yayımlanması, boya fotofiziği
için gözetimli öğrenmeyi uygulanabilir kılmıştır; bu veriler üzerinde eğitilen derin modeller
organik kromoforların absorpsiyon ve emisyon maksimumlarını, kromofor–çözücü etkileşimini de
açıkça hesaba katarak, birkaç on nanometrelik kök ortalama kare hatasıyla tahmin edebilmektedir
[6]. Bununla birlikte raporlanan doğruluk özellikler arasında çarpıcı biçimde dengesizdir:
spektral konumlar iyi öğrenilirken kuantum verimi ve molar absorptivite belirgin ölçüde daha
zor kalmaktadır; bu örüntü farklı çalışmalarda ve model ailelerinde yinelenmektedir.

Bu çalışmayı iki boşluk güdülemektedir. Birincisi, söz konusu dört özellik aynı elektronik
yapıdan kaynaklanmasına ve aynı numuneler üzerinde ölçülmesine karşın genellikle
**birbirinden bağımsız** olarak modellenmektedir; çok-görevli bir formülasyon ise temsili
hedefler arasında paylaştırabilir ve önemli biçimde, yalnızca bazı özelliklerin raporlandığı
kayıtlardan da yararlanabilir. İkincisi ve daha temel olarak, modellerin **bilinen kimyayı**
içselleştirip içselleştirmediği nadiren sorgulanmaktadır. Raporlanan başarım ölçütleri
ayrılmış bir test kümesindeki aradeğerleme kalitesini ortaya koyar; ancak modelin, bir
*mezo*-aril grubunun *mezo*-alkil gruba kıyasla emisyonu söndürdüğünü ya da daha ağır
halojenlerin Φ_F'yi düşürdüğünü öğrenip öğrenmediğini göstermez. Ayrıca bu modeller,
ileriye dönük doğrulamanın mevcut en yakın karşılığı olan bağımsız bir sentez çalışmasıyla —
eğitim derlemi dışında hazırlanmış ve karakterize edilmiş bileşiklerle — çoğu zaman
sınanmamaktadır.

Bu çalışmada her iki boşluğu özellikle BODIPY ailesi için ele alıyoruz. Açık deneysel veri
tabanından 607 benzersiz BODIPY kromoforu (1.853 kromofor–çözücü kaydı) çıkararak dört mimari
ailesini — parmak izi/tanımlayıcı tabanlı çok katmanlı algılayıcı, SMILES üzerinde tek
boyutlu evrişimli ağ ve transformer, ve çizge dikkat ağı (GATv2 [7]) — λ_abs, λ_em, Φ_F ve
log ε'yi birlikte tahmin eden, **çözücü-farkındalıklı çok-görevli** regresyon modelleri
olarak eğittik; maskeli bir yitim işlevi sayesinde kısmen etiketli kayıtlar da eğitime
katkı vermektedir. Tüm modeller aynı kromofor düzeyli bölmeler altında ve beş rastgele
tohumla karşılaştırılmış, böylece raporlanan farklar tohumlar arası değişkenlikle birlikte
sunulmuştur. Ardından eğitilmiş modelleri, hiçbir değişiklik yapmadan, eğitimde bulunmayan
sekiz 3,5-dialkil BODIPY bileşiğine [4] uyguladık. Son olarak modellerin ne öğrendiğini
sınadık: in-siliko sübstitüent taraması üç açık fotofiziksel hipotezi test ederken, dikkat
çözümlemesi ve SHAP atfı [8] tahminleri yönlendiren öznitelikleri konumlandırmaktadır.
Yinelenen bir bulgu — R² bakımından en iyi sıralanan mimarinin yerleşik kimyayla en tutarlı
olan olmaması — modellerin yalnızca aradeğerleme için değil boya tasarımına yön vermek üzere
kullanıldığı durumlarda model seçiminin nasıl yapılması gerektiği açısından doğrudan sonuçlar
doğurmaktadır.


---

## 2. Yöntem

### 2.1 Veri kümesi

Deneysel fotofiziksel veriler, açık erişimli Deep4Chem veri tabanından (`DB for chromophore`,
figshare DOI 10.6084/m9.figshare.12045567) alınmıştır [5,9]. Veri tabanı, kromofor ve çözücü
için SMILES gösterimleri içeren 20.836 kromofor–çözücü kaydından, 6.865 benzersiz kromofordan
ve 1.363 çözücüden oluşmaktadır.

BODIPY kayıtları, BF₂–dipirometen çekirdeğine altyapı eşleşmesi ile (SMARTS
`[F][B]([F])([#7])[#7]`) belirlenmiş ve **607 benzersiz BODIPY kromoforu** ile **1.853
kromofor–çözücü kaydı** elde edilmiştir. Hedef değişkenlerin doluluk sayıları: λ_abs 1.803;
λ_em 1.799; Φ_F 1.692; log ε 1.130 kayıt. Molar absorptivite çalışma boyunca log₁₀ε olarak
modellenmiştir.

### 2.2 Veri bölme

Her kromofor ortalama ~3 çözücüde yer aldığından, kayıtlar satır bazında değil **kromofor
bazında** bölünmüştür; böylece aynı boyanın farklı çözücülerdeki kayıtlarıyla hem eğitim hem
test kümesinde bulunması (veri sızıntısı) engellenmiştir. 70/15/15 oranındaki bölme
(tohum 42) 424/91/92 molekül ve 1.307/284/262 kayıt vermiştir. Bütün BODIPY'ler aynı çekirdek
iskeleti paylaştığı için standart Bemis–Murcko iskelet bölmesi burada bilgi taşımaz [17];
molekül düzeyinde bölme uygun karşılıktır.

### 2.3 Moleküler temsiller

Aynı kayıtlardan dört farklı temsil türetilmiştir:

- **Çizge** (GATv2 için): atomlar 17 boyutlu vektörlerle (C/N/O/F/B/S/Cl/Br + diğer için
  element one-hot; derece; formal yük; toplam H sayısı; SP/SP2/SP3 + diğer için hibritleşme
  one-hot; aromatiklik bayrağı), bağlar 7 boyutlu vektörlerle (tekli/ikili/üçlü/aromatik bağ
  türü one-hot; konjugasyon; halka üyeliği; stereo bayrağı) ve çift yönlü kenarlarla temsil
  edilmiştir.
- **Parmak izi/tanımlayıcı** (MLP için): 2.048 bitlik Morgan (ECFP benzeri) parmak izi [11]
  (yarıçap 2) ve on adet RDKit [12] tanımlayıcısı (MolWt, MolLogP, TPSA, NumHAcceptors,
  NumHDonors, NumRotatableBonds, NumAromaticRings, FractionCSP3, RingCount, NumHeteroatoms).
- **Dizi** (1D-CNN ve transformer için): `kromofor [SEP] çözücü` dizgisinin atom düzeyinde
  düzenli ifade ile belirteçlenmesi. **SMILES gösterimleri belirteçleme öncesinde RDKit ile
  kanonikleştirilmiştir**; bunun kritik olduğu görülmüştür (Bölüm 3.3).
- **Çözücü**: aynı on tanımlayıcı çözücü SMILES'i üzerinde hesaplanarak MLP girdisine ve
  GATv2'nin çizge düzeyindeki okuma katmanına eklenmiştir.

Tüm standartlaştırma istatistikleri (hedefler ve tanımlayıcılar) yalnızca eğitim kümesi
üzerinden hesaplanmıştır.


### 2.4 Mimariler

| Model | Girdi | Çekirdek |
|---|---|---|
| MLP | 2.068 boyut (Morgan + 10 + 10) | 512→256, BatchNorm, ReLU, seyreltme 0,2 |
| 1D-CNN | belirteç kimlikleri (≤256) | gömme 64; paralel evrişimler k = 3/5/7, 128 kanal; global maks-havuzlama |
| Transformer | belirteç kimlikleri (≤256) | gömme 128, 4 başlık, 3 kodlayıcı katmanı, maskeli ortalama havuzlama |
| GATv2 | çizge + çözücü vektörü | 3 × GATv2Conv (128, 4 başlık, kenar öznitelikleri), ortalama havuzlama |

Her modelde paylaşılan bir gövde ve **dört çıkışlı bir regresyon başlığı** (çok-görevli)
bulunmaktadır.

### 2.5 Eğitim ve değerlendirme

Hedefler eğitim kümesi istatistikleriyle standartlaştırılmıştır. Kayıtların çoğunda en az bir
özellik eksik olduğundan **maskeli ortalama kare hata** kullanılmıştır: yitim yalnızca
gözlenen hedefler üzerinden ortalanır, böylece kısmen etiketli kayıtlar da eğitime katkı
verir. Eniyileme Adam ile (öğrenme oranı 1e-3, ağırlık sönümü 1e-5), yığın boyutu 256 (MLP)
veya 128 olacak şekilde, en fazla 300 dönem boyunca ve doğrulama yitimi üzerinden erken
durdurma (sabır 30) ile yapılmıştır. Modeller PyTorch [13] ve çizge modeli için PyTorch
Geometric [14] kullanılarak gerçeklenmiş, Adam [15] ile eniyilenmiştir. Başarım ölçütleri
(R², RMSE, MAE) standartlaştırma geri alındıktan sonra her hedef için özgün birimlerinde
hesaplanmıştır. Tüm deneyler, veri bölmesi sabit tutularak **5 tohumla (0–4)** yinelenmiştir;
dolayısıyla raporlanan değişkenlik başlangıç değerleri ve eğitim rastgeleliğini yansıtır.
Sonuçlar ortalama ± standart sapma olarak verilmiştir.

### 2.6 Harici doğrulama kümesi

Sekiz adet 3,5-dialkil BODIPY (**1A–4B**) bağımsız bir sentez ve fotofizik çalışmasından [4]
alınmıştır; bileşikler *mezo* konumunda etil (1), fenil (2), 4-metoksifenil (3) ve
4-bromofenil (4) gruplarını, 3,5 konumlarında ise dimetil (A) veya dietil (B) sübstitüsyonunu
taşımaktadır. SMILES gösterimleri raporlanan yapılardan oluşturulmuş ve RDKit ile hesaplanan
moleküler formüllerin yayımlanmış formüllerle eşleşmesiyle doğrulanmıştır (8/8 tam eşleşme).
CHCl₃ içinde ölçülen λ_abs, λ_em, Φ_F ve ε değerleri gerçek referans olarak kullanılmıştır.

Eğitim verisiyle örtüşme kanonik SMILES üzerinden denetlenmiştir: **sekiz bileşiğin hiçbiri
eğitim kümesinde yer almamaktadır**; altısı Deep4Chem'de hiç bulunmamakta, ikisi (2A, 3A) ise
yalnızca ayrılmış bölümlerde geçmektedir. Dolayısıyla sekiz bileşiğin tamamı modeller
açısından gerçekten görülmemiştir.

### 2.7 Yorumlanabilirlik

**In-siliko sübstitüent taraması.** 3,5-dialkil BODIPY iskeleti sabit tutularak *mezo*
sübstitüenti 14 grup üzerinden değiştirilmiştir: alkil (metil, etil, n-propil),
sübstitüe olmamış fenil, elektron veren aril (4-NMe₂, 4-OMe, 4-Me), halojen serisi (4-F,
4-Cl, 4-Br, 4-I) ve elektron çeken aril (4-CN, 4-CF₃, 4-NO₂). Her biri 3,5-Me ve 3,5-Et ile
birlikte ele alınarak 28 yapı oluşturulmuş ve Φ_F, CHCl₃ içinde 5-tohumlu topluluklarla
tahmin edilmiştir. Yerleşik BODIPY fotofiziğinden üç hipotez sınanmıştır:

- **H1** *mezo*-alkil, *mezo*-arile kıyasla daha yüksek Φ_F verir;
- **H2** elektron veren aril > fenil > elektron çeken aril;
- **H3** Φ_F, F > Cl > Br > I boyunca tekdüze azalır (ağır atom etkisi).

**Dikkat haritaları.** GATv2 için atom başına önem, üç evrişim katmanı boyunca her atomun
komşularından *aldığı* dikkat (kaynak yönü) toplanarak, başlıklar ve tohumlar üzerinden
ortalanarak hesaplanmıştır. GATv2 dikkati hedef düğüm başına softmax ile normalleştirdiğinden
gelen dikkatin toplanmasının bilgi taşımadığı not edilmelidir.

**SHAP.** Φ_F için öznitelik atıfları MLP üzerinde GradientExplainer ile (200 eğitim arka
plan örneği, test kümesi üzerinde değerlendirme, 5 tohum ortalaması) hesaplanmıştır. Yüksek
atıflı Morgan bitleri RDKit bit bilgisi aracılığıyla altyapılara geri eşlenmiştir.
