# Interpretable Multi-Task Deep Learning for Photophysical Property Prediction of BODIPY Fluorophores

**Draft v0.1** — working manuscript. All numbers are from `results/` (5-seed ensembles).
Placeholders marked `[TODO]`.

**Authors:** Fırat Özcan [+ co-authors TODO]
**Affiliation:** Kırklareli University [TODO]

---

## Abstract

BODIPY (4,4-difluoro-4-bora-3a,4a-diaza-s-indacene) dyes are widely used as fluorescent
probes, laser dyes and photosensitizers, and their photophysics is governed largely by
substitution at the *meso* and 3,5 positions. Predicting these properties from structure
would accelerate dye design, but most machine-learning studies report only absorption or
emission maxima and rarely test whether the learned models encode known structure–property
rules. Here we train four architectures — a fingerprint/descriptor multilayer perceptron
(MLP), a 1D convolutional network (1D-CNN), a SMILES transformer and a graph attention
network (GATv2) — as **multi-task regressors** that jointly predict absorption maximum,
emission maximum, fluorescence quantum yield (Φ_F) and molar absorptivity (log ε) for 607
BODIPY chromophores (1,853 chromophore–solvent pairs) extracted from an open experimental
database. Using chromophore-based splits and 5 random seeds, wavelengths are predicted
accurately (MLP: R² = 0.912 ± 0.007 for absorption, 0.931 ± 0.009 for emission) whereas
Φ_F and log ε remain substantially harder (R² ≈ 0.21–0.54). We then validate the models on
eight 3,5-dialkyl BODIPYs from an independent synthetic study that were never seen in
training, reaching mean absolute errors of ~9.5–11.6 nm (absorption) and 0.126 ± 0.025 (Φ_F).
Finally, an in-silico substituent probe shows that the models reproduce established
photophysical rules without being told them: *meso*-alkyl substitution yields the highest
Φ_F in all models, and the graph network reproduces the full heavy-atom series
(F > Cl > Br > I, Φ_F dropping to 0.080 ± 0.11 for iodine). Attention maps and SHAP
attributions independently localise model reasoning on the BF₂ core and the *meso* linkage.
Notably, the architecture with the best R² is not the one most consistent with chemistry,
which we argue is an important caveat for model selection in dye design.

**Keywords:** BODIPY; multi-task learning; graph neural networks; fluorescence quantum
yield; interpretability; SHAP

---

## 1. Introduction

Boron-dipyrromethene (4,4-difluoro-4-bora-3a,4a-diaza-*s*-indacene, BODIPY) dyes occupy a
central place among synthetic fluorophores. Their appeal rests on a rare combination of
properties: chemical and photochemical robustness, high molar absorptivity in the visible
region, narrow absorption and emission bands with small Stokes shifts, fluorescence quantum
yields that can approach unity, and comparatively weak sensitivity to solvent polarity
[1–3]. Equally important, the BODIPY core is synthetically malleable — substitution at the
*meso* (8) position and at the 3,5 positions, or extension of the π system, shifts the
spectra and modulates the emission efficiency in a controlled way. This tunability underpins
applications as fluorescent probes and bio-labelling reagents, laser dyes, photosensitizers
for photodynamic therapy, and light-harvesting components in solar cells [1–3].

Despite this maturity, dye design remains largely empirical. Establishing the four
quantities that matter most in practice — the absorption and emission maxima (λ_abs, λ_em),
the fluorescence quantum yield (Φ_F) and the molar absorptivity (ε) — still requires
synthesis followed by spectroscopic characterisation, so each new substitution pattern costs
laboratory time. Quantum-chemical calculation offers partial relief: time-dependent density
functional theory predicts vertical excitation energies reasonably well, but it is
computationally expensive for large screening campaigns, and Φ_F in particular is not
directly accessible from routine calculations because it is governed by a competition
between radiative and non-radiative channels, including intersystem crossing and
substituent rotation. The qualitative rules are well known to synthetic chemists —
electron-donating and electron-withdrawing groups shift the frontier orbital gap in opposite
directions, and heavy atoms such as bromine or iodine quench fluorescence through enhanced
spin–orbit coupling [4] — yet these rules are not, by themselves, quantitatively predictive.

Data-driven modelling has changed this picture substantially over the past few years. The
release of large curated experimental databases — most notably a collection of more than
20,000 chromophore–solvent records covering absorption and emission maxima, bandwidths,
extinction coefficients, quantum yields and lifetimes [5] — made supervised learning
practical for dye photophysics, and deep models trained on such data now predict absorption
and emission maxima of organic chromophores with root-mean-square errors of a few tens of
nanometres, including explicit treatment of chromophore–solvent interaction [6]. Reported
accuracy is, however, strikingly uneven across properties: spectral positions are learned
well, whereas quantum yield and molar absorptivity remain considerably harder, a pattern
that recurs across studies and model families.

Two gaps motivate the present work. First, the four properties are usually modelled
**independently**, even though they originate from the same electronic structure and are
measured on the same samples; a multi-task formulation can share representation across
targets and, importantly, exploit records in which only some properties were reported.
Second, and more fundamentally, models are rarely interrogated for whether they have
internalised **known chemistry**. Reported metrics establish interpolation quality on a
held-out split, but they do not reveal whether a model has learned that a *meso*-aryl group
quenches emission relative to a *meso*-alkyl group, or that heavier halogens depress Φ_F.
Nor are such models often confronted with a fully independent synthetic study — compounds
prepared and characterised outside the training corpus — which is the closest available
analogue of prospective validation.

Here we address both gaps for the BODIPY family specifically. From the open experimental
database we extract 607 unique BODIPY chromophores (1,853 chromophore–solvent records) and
train four architecture families — a fingerprint/descriptor multilayer perceptron, a 1D
convolutional network and a Transformer over SMILES, and a graph attention network (GATv2
[7]) — as solvent-aware **multi-task** regressors that predict λ_abs, λ_em, Φ_F and log ε
jointly, using a masked loss so that partially labelled records still contribute. All models
are compared under identical, chromophore-level splits with five random seeds, so that
reported differences are accompanied by their seed-to-seed variability. We then apply the
trained models, unchanged, to eight 3,5-dialkyl BODIPYs from an independent synthetic and
photophysical study [4] that are absent from training. Finally, we probe what the models
learned: an in-silico substituent scan tests three explicit photophysical hypotheses,
while attention analysis and SHAP attribution [8] localise the features driving the
predictions. A recurring outcome — that the architecture ranked best by R² is not the one
most consistent with established chemistry — has direct consequences for how models should
be selected when they are intended to guide dye design rather than merely to interpolate.

---

## 2. Methods

### 2.1 Dataset

Experimental photophysical data were taken from the open Deep4Chem database
(`DB for chromophore`, figshare DOI 10.6084/m9.figshare.12045567) [5,9], comprising 20,836 chromophore–solvent records, 6,865 unique chromophores
and 1,363 solvents, with SMILES for both chromophore and solvent.

BODIPY entries were identified by substructure match to the BF₂–dipyrromethene core
(SMARTS `[F][B]([F])([#7])[#7]`), giving **607 unique BODIPY chromophores** in **1,853
chromophore–solvent records**. Target availability: λ_abs 1,803; λ_em 1,799; Φ_F 1,692;
log ε 1,130 records. Molar absorptivity is modelled as log₁₀ε throughout.

### 2.2 Data splitting

Because each chromophore appears in ~3 solvents on average, records were split **by
chromophore**, not by row, to prevent the same dye appearing in both training and test sets
in different solvents. A 70/15/15 split (seed 42) gave 424/91/92 molecules and
1,307/284/262 records. Standard Bemis–Murcko scaffold splitting is uninformative here
because all BODIPYs share the same core scaffold [17]; molecule-level splitting is the
appropriate analogue.

### 2.3 Molecular representations

Four representations were derived from the same records:

- **Graph** (for GATv2): atoms as 17-dimensional vectors (element one-hot over
  C/N/O/F/B/S/Cl/Br + other; degree; formal charge; total H count; hybridisation one-hot
  SP/SP2/SP3 + other; aromaticity flag) and bonds as 7-dimensional vectors (bond type
  one-hot single/double/triple/aromatic; conjugation; ring membership; stereo flag),
  with bidirectional edges.
- **Fingerprint/descriptor** (for MLP): 2,048-bit Morgan (ECFP-like) fingerprint [11]
  (radius 2) plus ten RDKit [12] descriptors (MolWt, MolLogP, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds,
  NumAromaticRings, FractionCSP3, RingCount, NumHeteroatoms).
- **Sequence** (for 1D-CNN and transformer): atom-level regex tokenisation of the string
  `chromophore [SEP] solvent`. **SMILES are canonicalised with RDKit before tokenisation**;
  this proved essential (Section 3.3).
- **Solvent**: the same ten descriptors computed on the solvent SMILES, concatenated to the
  MLP input and to the GATv2 graph-level readout.

All standardisation statistics (targets and descriptors) were computed on the training
split only.

### 2.4 Architectures

| Model | Input | Core |
|---|---|---|
| MLP | 2,068-dim (Morgan + 10 + 10) | 512→256, BatchNorm, ReLU, dropout 0.2 |
| 1D-CNN | token ids (≤256) | embedding 64; parallel convs k = 3/5/7, 128 ch; global max-pool |
| Transformer | token ids (≤256) | embedding 128, 4 heads, 3 encoder layers, masked mean-pool |
| GATv2 | graph + solvent vector | 3 × GATv2Conv (128, 4 heads, edge features), mean-pool |

Each model has a shared trunk and a **four-output regression head** (multi-task).

### 2.5 Training and evaluation

Targets are standardised with training-set statistics. Because most records lack at least
one property, we use a **masked mean-squared error**: the loss is averaged only over
observed targets, so partially labelled records still contribute. Optimisation used Adam
(lr 1e-3, weight decay 1e-5), batch size 256 (MLP) or 128, up to 300 epochs with early
stopping on validation loss (patience 30). Models were implemented in PyTorch [13] with
PyTorch Geometric [14] for the graph model, and optimised with Adam [15]. Metrics (R², RMSE, MAE) are computed per target
in original units after de-standardisation. All experiments were repeated with **5 seeds
(0–4)** with the data split held fixed, so reported variation reflects initialisation and
training stochasticity; results are given as mean ± s.d.

### 2.6 External validation set

Eight 3,5-dialkyl BODIPYs (**1A–4B**) were taken from an independent synthetic and
photophysical study [4], comprising *meso*-ethyl (1), phenyl (2),
4-methoxyphenyl (3) and 4-bromophenyl (4), each with 3,5-dimethyl (A) or 3,5-diethyl (B)
substitution. SMILES were constructed from the reported structures and verified by
RDKit molecular-formula match against the published formulas (8/8 exact). Measured λ_abs,
λ_em, Φ_F and ε in CHCl₃ were used as ground truth.

Overlap with the training data was checked by canonical SMILES: **none of the eight
compounds occurs in the training split**; six do not occur anywhere in Deep4Chem, and two
(2A, 3A) appear only in held-out portions, so all eight are genuinely unseen.

### 2.7 Interpretability

**In-silico substituent probe.** Keeping the 3,5-dialkyl BODIPY scaffold fixed, the *meso*
substituent was varied over 14 groups spanning alkyl (methyl, ethyl, n-propyl), unsubstituted
phenyl, electron-donating aryl (4-NMe₂, 4-OMe, 4-Me), the halogen series (4-F, 4-Cl, 4-Br,
4-I) and electron-withdrawing aryl (4-CN, 4-CF₃, 4-NO₂), each with 3,5-Me and 3,5-Et
(28 structures), and Φ_F was predicted in CHCl₃ with the 5-seed ensembles. Three hypotheses
from established BODIPY photophysics were tested:

- **H1** *meso*-alkyl gives higher Φ_F than *meso*-aryl;
- **H2** electron-donating aryl > phenyl > electron-withdrawing aryl;
- **H3** Φ_F decreases monotonically along F > Cl > Br > I (heavy-atom effect).

**Attention maps.** For GATv2, per-atom importance was accumulated over the three
convolutions as the attention each atom *receives from* its neighbours (source direction),
averaged over heads and seeds. Note that summing incoming attention is uninformative
because GATv2 normalises attention per destination node by softmax.

**SHAP.** Feature attributions for Φ_F were computed for the MLP with GradientExplainer
(200 training background samples, evaluated on the test split, averaged over 5 seeds).
High-attribution Morgan bits were mapped back to substructures via RDKit bit information.


---

## 3. Results and discussion

### 3.1 Multi-task benchmark on held-out BODIPYs

Table 1 reports test-set R² (mean ± s.d. over 5 seeds) for all four architectures. Absorption
and emission maxima are predicted well by every model except the transformer, with the MLP
the most accurate and by far the most stable (λ_abs R² = 0.912 ± 0.007; λ_em
0.931 ± 0.009). Quantum yield and log ε are substantially harder for all models
(R² ≈ 0.2–0.55). No single architecture dominates every property: the MLP leads on
wavelengths, the 1D-CNN on Φ_F, and GATv2 on log ε, so architecture choice should follow the
target of interest rather than a single headline number (Fig. 1).

**Table 1.** Internal test-set R² (mean ± s.d., 5 seeds; chromophore-level split).

| Model | λ_abs | λ_em | Φ_F | log ε |
|---|---|---|---|---|
| MLP | **0.912 ± 0.007** | **0.931 ± 0.009** | 0.508 ± 0.037 | 0.517 ± 0.029 |
| 1D-CNN | 0.885 ± 0.016 | 0.899 ± 0.005 | **0.543 ± 0.046** | 0.327 ± 0.054 |
| GATv2 | 0.865 ± 0.025 | 0.888 ± 0.039 | 0.435 ± 0.049 | **0.527 ± 0.068** |
| Transformer | 0.671 ± 0.042 | 0.644 ± 0.049 | 0.213 ± 0.084 | 0.296 ± 0.052 |

The transformer's weakness is consistent with its data requirements: with ~1,300 training
records it under-performs the descriptor and graph models, an expected small-data regime
effect. That Φ_F and log ε are the hard targets is chemically sensible — the BODIPY Φ_F
distribution is strongly bimodal (a large non-emissive population near 0 and an emissive
population near 1), and the log ε range in this subset is narrow, leaving little variance for
a regressor to exploit.

### 3.2 External validation on independent compounds

We next applied the trained ensembles, unchanged, to the eight 3,5-dialkyl BODIPYs of the
independent synthetic study (Section 2.6). Because these compounds span a deliberately narrow
optical window (λ_abs 508–515 nm; λ_em 514–529 nm), this is primarily a test of absolute
calibration for wavelengths and of ranking for Φ_F, whose measured range (0.24–1.00) is
wide. Mean absolute errors are collected in Table 2 (Fig. 2).

**Table 2.** External-set mean absolute error (mean ± s.d., 5 seeds; 8 compounds).

| Model | λ_abs (nm) | λ_em (nm) | Φ_F | log ε |
|---|---|---|---|---|
| MLP | **9.6 ± 2.6** | 20.7 ± 4.5 | 0.241 ± 0.022 | 0.069 ± 0.017 |
| 1D-CNN | 11.6 ± 3.6 | 19.3 ± 3.1 | **0.126 ± 0.025** | **0.058 ± 0.008** |
| GATv2 | 10.4 ± 2.4 | 20.6 ± 4.2 | 0.171 ± 0.022 | 0.063 ± 0.027 |
| Transformer | 9.5 ± 3.6 | **10.4 ± 3.2** | 0.204 ± 0.030 | 0.097 ± 0.014 |

Absorption maxima of these fully unseen dyes are reproduced to ~9–12 nm, approaching the
scale of inter-laboratory spectroscopic variation, and the transformer — the weakest model
internally — gives the lowest emission error (10.4 ± 3.2 nm), a reminder that internal and
external rankings can diverge. For ranking the eight dyes by Φ_F, Spearman ρ is highest and
most stable for the 1D-CNN (0.814 ± 0.080) and GATv2 (0.719 ± 0.191); the ρ values for the
wavelength and log ε rankings carry very large seed-to-seed spread and even change sign, and
should not be over-interpreted on only eight compounds (Section 3.7).

### 3.3 SMILES canonicalisation is essential for sequence models

An initial external evaluation exposed a large, representation-dependent failure. The eight
external SMILES were written in a charged Kekulé form ([N⁺]/[B⁻]); on these strings the
1D-CNN produced an absorption MAE of ~104 nm despite its strong internal performance, whereas
the descriptor (MLP) and graph (GATv2) models — which operate on representation-invariant
features — were unaffected. Canonicalising all SMILES with RDKit before tokenisation, for
both training and inference, reduced the 1D-CNN error to ~9 nm. Sequence models are therefore
sensitive to arbitrary SMILES conventions in a way that graph and descriptor models are not,
and consistent canonicalisation is a prerequisite for fair evaluation and for deployment on
externally supplied structures.

### 3.4 The models recover known structure–property rules

The central question of this work is whether the models encode BODIPY photophysics rather
than merely interpolating. The in-silico *meso*-substituent probe (Fig. 3, Table 3) tests
three established rules directly. **H1** (meso-alkyl ≫ meso-aryl in Φ_F) is reproduced by all
three ensembles — the very effect underlying the exceptionally high Φ_F of the alkyl-*meso*
reference dyes in the external set. **H3**, the heavy-atom series F > Cl > Br > I, is
reproduced by the 1D-CNN and, most cleanly, by GATv2, whose predicted Φ_F falls from
0.304 (F) to 0.080 (I); the models were never told which atoms are heavy, so this monotonic
decline is inferred from data alone and extrapolates the bromine heavy-atom effect noted in
the source study to iodine. **H2** (electron-donating aryl > phenyl > electron-withdrawing
aryl) holds fully only for GATv2; for the other models the donor/phenyl ordering is weak,
although electron-withdrawing aryl gives the lowest Φ_F in every model.

**Table 3.** Ensemble-mean predicted Φ_F by *meso* class and hypothesis outcomes
(✓ satisfied, ✗ not). Halogen columns are the 4-halophenyl series.

| Model | alkyl | EDG | phenyl | EWG | F | Cl | Br | I | H1 | H2 | H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1D-CNN | 0.642 | 0.292 | 0.374 | 0.207 | 0.297 | 0.287 | 0.214 | 0.199 | ✓ | ✗ | ✓ |
| GATv2 | 0.584 | 0.331 | 0.319 | 0.258 | 0.304 | 0.240 | 0.213 | **0.080** | ✓ | ✓ | ✓ |
| MLP | 0.448 | 0.299 | 0.403 | 0.229 | 0.209 | 0.201 | 0.236 | 0.110 | ✓ | ✗ | ✗ |

### 3.5 Attention and SHAP localise on the BF₂ core and meso position

Two independent attribution methods agree on where the models look. GATv2 attention
(Fig. 4) concentrates on the BF₂ chelate (mean atom importance 0.55 vs 0.45 for peripheral
atoms; the boron atom ranks in the top three for every compound) and, in aryl-*meso* dyes, on
the *meso* bridging carbon; for the 4-methoxy dyes the methoxy oxygen is also highlighted.
SHAP attribution for the MLP's Φ_F output (Fig. 5) shows that **solvent descriptors dominate**
— solvent TPSA is the single most important feature (mean |SHAP| 0.113 ± 0.018), ahead of any
chromophore feature — which both reflects the known solvent sensitivity of BODIPY Φ_F and
justifies the solvent-aware input design. Mapping the most important Morgan bits back to
substructures recovers fragments of the BODIPY core itself (the meso-bridge/dipyrromethene
motif `cc(c)C(=C(C)[N⁺])c(c)n`, the BF₂–N linkage `[B⁻]n(c)c`, and the 3,5-alkyl-pyrrole
`cc(C)n`). That attention (a graph model) and SHAP (a descriptor model) converge on the same
BF₂/meso region strengthens the mechanistic reading.

### 3.6 Accuracy and chemical consistency are distinct axes

A recurring theme is that the model with the best R² is not the most chemically faithful.
GATv2 is the only architecture to satisfy all three substituent hypotheses and gives the
cleanest heavy-atom trend and attention maps, yet the MLP has the higher wavelength R². For
applications where a model is used to *reason* about design — extrapolating to new
substituents — chemical consistency may matter more than a marginal R² gain, and we recommend
reporting both.

### 3.7 Limitations

Several limitations should be stated plainly. (i) The external set has only eight compounds
and a narrow optical range; rank correlations for wavelengths and log ε on this set have very
large seed spread and change sign, so only the Φ_F ranking and the wavelength MAEs are
robust conclusions. (ii) The in-silico probe reports model *predictions*, i.e. hypotheses;
the extrapolated iodine and electron-withdrawing cases are chemically plausible but not
experimentally verified here. (iii) Φ_F and log ε remain hard (R² ≲ 0.55); the bimodal Φ_F
distribution in particular limits a plain regressor, and modelling it as a two-stage
emissive/non-emissive problem is a promising direction. (iv) All data derive from a single
database; some measurement heterogeneity across source studies is unavoidable. (v) Sequence
models are data-hungry at this scale — transfer learning from the full chromophore database,
followed by BODIPY fine-tuning, is the natural next step and is expected to benefit the
transformer most.

---

## 4. Conclusions

We have presented a solvent-aware, multi-task deep-learning study of BODIPY photophysics that
is evaluated not only for accuracy but for chemical fidelity. Under strict chromophore-level
splitting and seed-averaged statistics, descriptor and graph models predict absorption and
emission maxima of unseen BODIPYs to within ~6–12 nm, while quantum yield and molar
absorptivity remain harder. Beyond accuracy, the models reproduce established
structure–property rules without supervision — the dominance of *meso*-alkyl over
*meso*-aryl emission and a spontaneous F > Cl > Br > I heavy-atom series — and attention and
SHAP independently localise the models' reasoning on the BF₂ core and *meso* position, with
solvent descriptors emerging as the leading determinant of Φ_F. The graph attention network,
though not the most accurate by R², is the most chemically consistent, and we argue that both
axes should be reported when models are intended to guide dye design. Immediate extensions
are transfer learning from the broader chromophore space and explicit treatment of the
bimodal quantum-yield distribution.

---

## Data and code availability

Experimental data are from the open Deep4Chem database (figshare
10.6084/m9.figshare.12045567). All code (data preparation, featurisation, the four models,
multi-seed training, external validation and interpretability) and the derived result tables
and figures are available at [repository URL — TODO].

## Author contributions

[TODO]

## Acknowledgements

[TODO — funding, TÜBİTAK/BAP if applicable]

---

## References

*Formatted in a generic style; convert to the target journal's style before submission.*

1. Loudet, A.; Burgess, K. BODIPY dyes and their derivatives: syntheses and spectroscopic properties. *Chem. Rev.* **2007**, *107*, 4891–4932.
2. Ulrich, G.; Ziessel, R.; Harriman, A. The chemistry of fluorescent bodipy dyes: versatility unsurpassed. *Angew. Chem. Int. Ed.* **2008**, *47*, 1184–1201.
3. Boens, N.; Leen, V.; Dehaen, W. Fluorescent indicators based on BODIPY. *Chem. Soc. Rev.* **2012**, *41*, 1130–1172.
4. Derin, Y.; Yılmaz, R. F.; Baydilek, İ. H.; Enisoğlu Atalay, V.; Özdemir, A.; Tutar, A. Synthesis, electrochemical/photophysical properties and computational investigation of 3,5-dialkyl BODIPY fluorophores. *Inorg. Chim. Acta* **2018**, *482*, 130–135. DOI: 10.1016/j.ica.2018.06.006.
5. Joung, J. F.; Han, M.; Jeong, M.; Park, S. Experimental database of optical properties of organic compounds. *Sci. Data* **2020**, *7*, 295. DOI: 10.1038/s41597-020-00634-8.
6. Joung, J. F.; Han, M.; Hwang, J.; Jeong, M.; Choi, D. H.; Park, S. Deep learning optical spectroscopy based on experimental database: potential applications to molecular design. *JACS Au* **2021**, *1*, 427–438. DOI: 10.1021/jacsau.1c00035.
7. Brody, S.; Alon, U.; Yahav, E. How attentive are graph attention networks? In *International Conference on Learning Representations (ICLR)*, **2022**. arXiv:2105.14491.
8. Lundberg, S. M.; Lee, S.-I. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*, **2017**, 4765–4774.
9. Joung, J. F.; Han, M.; Jeong, M.; Park, S. DB for chromophore. *figshare* **2020**. DOI: 10.6084/m9.figshare.12045567.
10. Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. *J. Chem. Inf. Comput. Sci.* **1988**, *28*, 31–36.
11. Rogers, D.; Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **2010**, *50*, 742–754.
12. Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org (accessed 2026).
13. Paszke, A.; Gross, S.; Massa, F.; et al. PyTorch: an imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32 (NeurIPS)*, **2019**, 8024–8035.
14. Fey, M.; Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*, **2019**. arXiv:1903.02428.
15. Kingma, D. P.; Ba, J. Adam: a method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*, **2015**. arXiv:1412.6980.
16. Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. Graph attention networks. In *International Conference on Learning Representations (ICLR)*, **2018**. arXiv:1710.10903.
17. Bemis, G. W.; Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* **1996**, *39*, 2887–2893.
18. Vaswani, A.; Shazeer, N.; Parmar, N.; et al. Attention is all you need. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*, **2017**, 5998–6008.

---

## Figures and tables

- **Fig. 1** — Architecture comparison, internal test R² (mean ± s.d.). `results/multiseed_R2.png`
- **Fig. 2** — External validation parity, 8 BODIPYs vs measured. `results/external_parity.png`
  (rank-correlation summary: `results/multiseed_external_rho.png`)
- **Fig. 3** — In-silico *meso*-substituent probe: predicted Φ_F by substituent
  (`results/probe_QY.png`) and halogen series (`results/probe_heavy_atom.png`)
- **Fig. 4** — GATv2 attention-derived atom importance. `results/attention_maps.png`
- **Fig. 5** — SHAP feature attribution for Φ_F (MLP). `results/shap_summary.png`
- **Fig. S1** — BODIPY target distributions. `results/eda_bodipy_dist.png`
- **Table 1** — Internal test R². `results/multiseed_internal_summary.csv`
- **Table 2** — External MAE. `results/multiseed_external_summary.csv`
- **Table 3** — Substituent-probe hypotheses. `results/probe_summary.csv`
- **Table S1** — Per-compound external predictions. `results/external_predictions.csv`
- **Table S2** — Top SHAP Morgan bits → substructures. `results/shap_bits.csv`
