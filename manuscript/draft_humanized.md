# Interpretable multi-task deep learning for photophysical property prediction of BODIPY fluorophores

Draft v0.2 (working manuscript). All numbers come from `results/` and are 5-seed ensembles.
Placeholders are marked `[TODO]`.

**Author:** Fırat Özcan¹,*

¹ Department of Mechatronics Engineering, Faculty of Technology, Kırklareli University,
Kayalı Campus, 39100 Kırklareli, Turkey

\* Correspondence: firatozcan@klu.edu.tr

---

## Abstract

BODIPY (4,4-difluoro-4-bora-3a,4a-diaza-s-indacene) dyes are widely used as fluorescent
probes, laser dyes and photosensitizers, and their photophysics depends mainly on
substitution at the *meso* and 3,5 positions. Predicting these properties from structure
would shorten the design cycle, but most machine-learning studies report only absorption or
emission maxima, and few ask whether the trained model has actually captured known
structure-property rules. We trained four architectures on 607 BODIPY chromophores (1,853
chromophore-solvent records) drawn from an open experimental database: a
fingerprint/descriptor multilayer perceptron (MLP), a 1D convolutional network (1D-CNN), a
SMILES transformer, and a graph attention network (GATv2). Each model predicts absorption
maximum, emission maximum, fluorescence quantum yield (Φ_F) and molar absorptivity (log ε)
together rather than separately. With chromophore-based splits and five random seeds the
wavelengths are predicted accurately (MLP: R² = 0.912 ± 0.007 for absorption, 0.931 ± 0.009
for emission), while Φ_F and log ε are harder, with R² between 0.21 and 0.54. We then held
out twelve entire source publications, 190 BODIPY dyes in all, and retrained without them.
On this pooled external set the absorption maximum is predicted with a mean absolute error
of 30.8 ± 0.7 nm and R² = 0.563 ± 0.040, against 18.5 nm under the random split, which shows
how much a conventional split flatters the task. On a further eight 3,5-dialkyl BODIPYs from
a study outside the database the error drops to 7.2 ± 2.0 nm, because those dyes sit close to
the training distribution. Quantum yield resists every remedy we tried: a logit transform
improves it internally but not across studies, a two-stage dark/bright model does not improve
it at all, and richer physical solvent descriptors help only marginally, which points to
inter-laboratory measurement variation rather than to model design. An in-silico substituent
scan shows that the models
recover photophysical rules nobody supplied to them: *meso*-alkyl substitution gives the
highest Φ_F in every model, and the graph network reproduces the whole heavy-atom series
(F > Cl > Br > I), with Φ_F falling to 0.080 for iodine. Attention maps and SHAP attributions
point independently to the BF₂ core and the *meso* linkage. The architecture with the best R²
is not the one that agrees best with chemistry, which we think matters for how models are
chosen in dye design.

**Keywords:** BODIPY; multi-task learning; graph neural networks; fluorescence quantum
yield; interpretability; SHAP

---

## 1. Introduction

Boron-dipyrromethene (4,4-difluoro-4-bora-3a,4a-diaza-*s*-indacene, BODIPY) dyes are among
the most heavily used synthetic fluorophores. They combine chemical and photochemical
robustness with high molar absorptivity in the visible region, narrow absorption and
emission bands, small Stokes shifts, quantum yields that can approach unity, and fairly
weak sensitivity to solvent polarity [1-3]. The core is also easy to modify. Substitution at
the *meso* (8) position and at the 3,5 positions, or extension of the π system, shifts the
spectra and changes the emission efficiency in a controlled way. Because of this, BODIPYs
appear as fluorescent probes and bio-labelling reagents, laser dyes, photosensitizers for
photodynamic therapy, and light-harvesting components in solar cells [1-3].

Dye design nevertheless remains largely empirical. The four quantities that matter most in
practice are the absorption and emission maxima (λ_abs, λ_em), the fluorescence quantum
yield (Φ_F) and the molar absorptivity (ε), and getting them still means synthesising the
compound and measuring it. Every new substitution pattern therefore costs laboratory time.
Quantum-chemical calculation helps only partly. Time-dependent density functional theory
predicts vertical excitation energies reasonably well, but it is expensive for large
screening campaigns, and Φ_F is not directly accessible from routine calculations at all,
since it depends on a competition between radiative and non-radiative channels such as
intersystem crossing and substituent rotation. Synthetic chemists know the qualitative rules
perfectly well. Electron-donating and electron-withdrawing groups move the frontier orbital
gap in opposite directions, and heavy atoms such as bromine or iodine quench fluorescence
through stronger spin-orbit coupling [4]. On their own, though, these rules do not predict
numbers.

Data-driven modelling has changed the picture over the past few years. Large curated
experimental databases became available, among them a collection of more than 20,000
chromophore-solvent records with absorption and emission maxima, bandwidths, extinction
coefficients, quantum yields and lifetimes [5]. Deep models trained on such data now predict
absorption and emission maxima of organic chromophores to within a few tens of nanometres
and can account explicitly for chromophore-solvent interaction [6]. The accuracy is very
uneven across properties, however. Spectral positions are learned well; quantum yield and
molar absorptivity are much harder, and the same pattern shows up across studies and model
families.

Work aimed specifically at BODIPYs has followed a different route, relying on hand-crafted
descriptors and classical regression on small curated sets. Casanola-Martin and co-workers
combined DFT and TD-DFT optimisation with a genetic-algorithm descriptor selection and
multi-linear regression on 131 BODIPYs, reaching a test-set R² of 0.734 for the absorption
maximum [19]. Buglak and co-workers built QSPR models for singlet oxygen generation by
heavy-atom-free BODIPY photosensitizers [20], and the same group later predicted the ratio
of fluorescence to singlet oxygen quantum yield for BODIPY dyes with QSPR and machine
learning [21]. These studies establish that BODIPY photophysics is learnable from structure,
and the descriptor analyses in them agree with the chemical intuition we test later in this
paper. They also share three limits that we set out to address: the datasets are small, each
property is modelled on its own, and the solvent is either fixed or absent from the model.

Two gaps motivated this work. The first is that the four properties are usually modelled
one at a time, even though they come from the same electronic structure and are measured on
the same samples. A multi-task model can share a representation across targets and, just as
usefully, make use of records where only some of the properties were reported. The second
gap is that models are rarely asked whether they have internalised any chemistry. Standard
metrics tell you how well a model interpolates on a held-out split. They do not tell you
whether it has learned that a *meso*-aryl group quenches emission relative to a *meso*-alkyl
group, or that heavier halogens depress Φ_F. Models are also seldom confronted with a
separate synthetic study, that is, compounds prepared and characterised outside the training
corpus, which is the closest thing to prospective validation that is available offline.

We address both gaps for BODIPYs. From the open experimental database we extracted 607
unique BODIPY chromophores in 1,853 chromophore-solvent records and trained four
architecture families: a fingerprint/descriptor multilayer perceptron, a 1D convolutional
network and a transformer over SMILES, and a graph attention network (GATv2 [7]). All of
them are solvent-aware multi-task regressors that predict λ_abs, λ_em, Φ_F and log ε
jointly, with a masked loss so that partially labelled records still contribute. The models
are compared under identical chromophore-level splits with five random seeds, so every
reported difference comes with its seed-to-seed variability. We then applied the trained
models, unchanged, to eight 3,5-dialkyl BODIPYs from an independent synthesis and
photophysics study [4] that are absent from training. Finally we probed what the models
learned. An in-silico substituent scan tests three explicit photophysical hypotheses, and
attention analysis together with SHAP attribution [8] shows which features drive the
predictions. One result came up repeatedly: the architecture that ranks best by R² is not
the one that agrees best with established chemistry. That has practical consequences when a
model is meant to guide dye design rather than only to interpolate.

---

## 2. Methods

### 2.1 Dataset

Experimental photophysical data came from the open Deep4Chem database (`DB for chromophore`,
figshare DOI 10.6084/m9.figshare.12045567) [5,9], which holds 20,836 chromophore-solvent
records covering 6,865 unique chromophores and 1,363 solvents, with SMILES for both the
chromophore and the solvent.

BODIPY entries were identified by substructure match to the BF₂-N₂ chelate motif that defines
the family (SMARTS `[F][B]([F])([#7])[#7]`, in which `[#7]` matches nitrogen in any
aromaticity or charge state, since the source SMILES are written inconsistently across the
contributing publications). This gave 607 unique chromophores in 1,853 chromophore-solvent
records. Of these, 550 also satisfy the stricter condition that the boron sits in a
six-membered chelate ring closed by two five-membered nitrogen heterocycles, which is the
classical 4,4-difluoro-4-bora-3a,4a-diaza-*s*-indacene skeleton; the remaining 57 are close
relatives in which a nitrogen donor belongs to a six-membered heterocycle. We kept the latter,
since they carry the same BF₂ chromophore and are governed by the same photophysical
mechanisms, but the distinction should be borne in mind when comparing with studies restricted
to the classical core, and the subset membership is provided with the released data. The
targets are unevenly populated: λ_abs 1,803 records, λ_em 1,799, Φ_F 1,692 and log ε 1,130.
Molar absorptivity is modelled as log₁₀ε throughout.

### 2.2 Data splitting

Each chromophore appears in about three solvents on average, so records were split by
chromophore rather than by row. Splitting by row would let the same dye appear in training
and test in different solvents. A 70/15/15 split (seed 42) gave 424/91/92 molecules and
1,307/284/262 records. Bemis-Murcko scaffold splitting tells you nothing here, since every
BODIPY shares the same core scaffold [17], so molecule-level splitting takes its place.

### 2.3 Molecular representations

Four representations were derived from the same records.

For the graph model, atoms are 17-dimensional vectors (element one-hot over C/N/O/F/B/S/Cl/Br
plus other, degree, formal charge, total H count, hybridisation one-hot over SP/SP2/SP3 plus
other, and an aromaticity flag) and bonds are 7-dimensional vectors (bond type one-hot over
single/double/triple/aromatic, conjugation, ring membership, stereo flag), with bidirectional
edges. For the MLP, each chromophore is a 2,048-bit Morgan fingerprint [11] of radius 2
together with ten RDKit [12] descriptors: MolWt, MolLogP, TPSA, NumHAcceptors, NumHDonors,
NumRotatableBonds, NumAromaticRings, FractionCSP3, RingCount and NumHeteroatoms. The two
sequence models read an atom-level regex tokenisation of the string
`chromophore [SEP] solvent`; SMILES are canonicalised with RDKit before tokenisation, which
turned out to matter a great deal (Section 3.3). Solvents are represented by the same ten
descriptors, computed on the solvent SMILES and concatenated to the MLP input and to the
GATv2 graph-level readout.

All standardisation statistics, for targets and descriptors alike, were computed on the
training split only.

### 2.4 Architectures

| Model | Input | Core |
|---|---|---|
| MLP | 2,068-dim (Morgan + 10 + 10) | 512 to 256, BatchNorm, ReLU, dropout 0.2 |
| 1D-CNN | token ids (≤256) | embedding 64; parallel convs k = 3/5/7, 128 ch; global max-pool |
| Transformer | token ids (≤256) | embedding 128, 4 heads, 3 encoder layers, masked mean-pool |
| GATv2 | graph + solvent vector | 3 × GATv2Conv (128, 4 heads, edge features), mean-pool |

Every model has a shared trunk and a four-output regression head.

### 2.5 Training and evaluation

Targets are standardised with training-set statistics. Most records lack at least one
property, so the loss is a masked mean-squared error, averaged only over the targets that
were actually observed; partially labelled records still contribute. Training used Adam [15]
(learning rate 1e-3, weight decay 1e-5) with a batch size of 256 for the MLP and 128
otherwise, for up to 300 epochs, with early stopping on validation loss after 30 epochs
without improvement. The models were implemented in PyTorch [13], with PyTorch Geometric [14]
for the graph model. R², RMSE and MAE are computed per target in the original units after
de-standardisation. Every experiment was repeated with five seeds (0 to 4) while the data
split stayed fixed, so the spread we report reflects initialisation and training
stochasticity. Results are given as mean ± s.d.

### 2.6 External validation sets

Validation was carried out at two levels of difficulty.

*Pooled publication holdout.* Deep4Chem records the source publication of every measurement,
which allows a much stricter test than a random split. We selected the twelve source
publications contributing at least five BODIPY molecules and at least five Φ_F values, giving
190 molecules in 790 records, and removed them entirely from training. A review compilation
(10.1039/c4cs00030g) was excluded from the candidate list, since it aggregates dyes from many
laboratories and is not a single study. Fourteen molecules occur in more than one candidate
publication, so removal was done by molecule rather than by record: every record of a held-out
molecule was withheld, whatever publication it came from. Standardisation statistics were
recomputed from the reduced training set. The remaining 417 molecules (1,063 records) were
split into training and validation by chromophore, and the whole procedure was repeated over
five seeds.

*Independent synthetic study.* Eight 3,5-dialkyl BODIPYs (1A to 4B) were taken from a
synthesis and photophysics paper outside the database [4]. They carry *meso*-ethyl (1),
phenyl (2), 4-methoxyphenyl (3) and 4-bromophenyl (4), each with either 3,5-dimethyl (A) or
3,5-diethyl (B) substitution. SMILES were built from the reported structures and checked by
matching RDKit molecular formulas against the published ones, which agreed in all eight cases.
The measured λ_abs, λ_em, Φ_F and ε in CHCl₃ served as ground truth. Overlap was checked on
canonical SMILES: none of the eight occurs in training, six do not occur anywhere in
Deep4Chem, and the remaining two (2A and 3A) appear only in held-out portions.

We report mean absolute error and Spearman rank correlation for both sets. Coefficients of
determination are reported only where the measured range is wide enough for R² to be
meaningful; this point is taken up in Section 3.7.

### 2.7 Interpretability

*In-silico substituent probe.* With the 3,5-dialkyl BODIPY scaffold held fixed, the *meso*
substituent was varied over 14 groups: alkyl (methyl, ethyl, n-propyl), unsubstituted phenyl,
electron-donating aryl (4-NMe₂, 4-OMe, 4-Me), the halogen series (4-F, 4-Cl, 4-Br, 4-I) and
electron-withdrawing aryl (4-CN, 4-CF₃, 4-NO₂). Each was combined with 3,5-Me and 3,5-Et,
giving 28 structures, and Φ_F was predicted in CHCl₃ with the 5-seed ensembles. Three
hypotheses taken from established BODIPY photophysics were tested. H1: *meso*-alkyl gives a
higher Φ_F than *meso*-aryl. H2: electron-donating aryl beats phenyl, which beats
electron-withdrawing aryl. H3: Φ_F falls monotonically along F, Cl, Br, I, the heavy-atom
effect.

*Attention maps.* For GATv2, per-atom importance was accumulated over the three convolutions
as the attention each atom receives from its neighbours, that is, the source direction,
averaged over heads and seeds. Summing incoming attention instead is uninformative, because
GATv2 normalises attention per destination node with a softmax and the incoming weights of
any node therefore sum to one.

*SHAP.* Attributions for Φ_F were computed for the MLP with GradientExplainer, using 200
training samples as background, evaluated on the test split and averaged over five seeds.
Morgan bits with high attribution were mapped back to substructures using RDKit bit
information.

### 2.8 Quantum-yield parametrisations and solvent ablation

Two further comparisons use a single MLP body, five seeds and both evaluation levels, so that
the effect of one design choice is isolated at a time.

*Φ_F parametrisation.* Four ways of writing the same target were compared: the standardised
regression used throughout, a bounded variant with a sigmoid output, a logit transform with
values clipped to [10⁻³, 1 − 10⁻³], and a two-stage model with a classification head for
Φ_F ≥ 0.05 and a regression head fitted only on the emissive records, combined at prediction
time as p·E[Φ_F | emissive] + (1 − p)·E[Φ_F | dark], the last term estimated from the
training set. The dark/bright threshold of 0.05 leaves 18.5 per cent of measurements in the
dark class.

*Solvent representation.* Physical constants were compiled for the 22 solvents that account
for 97 per cent of the records: static dielectric constant, refractive index, ET(30) polarity,
dynamic viscosity and the Kamlet-Taft parameters π*, α and β [22,23]. Entries we could not
source reliably were left empty, flagged with a missing-value indicator and filled with the
training median. Four settings were compared: no solvent input, the ten generic RDKit
descriptors used elsewhere in this work, the physical constants, and the physical constants
applied through FiLM conditioning, in which a small network maps the solvent vector to a
scale and a shift applied to the chromophore representation rather than concatenating the two.

---

## 3. Results and discussion

### 3.1 Multi-task benchmark on held-out BODIPYs

Table 1 gives test-set R² for all four architectures, averaged over five seeds. Every model
except the transformer predicts absorption and emission maxima well, and the MLP is both the
most accurate and by some margin the most stable (λ_abs R² = 0.912 ± 0.007, λ_em
0.931 ± 0.009). Quantum yield and log ε are considerably harder for all of them, with R²
between roughly 0.2 and 0.55. No architecture wins across the board: the MLP leads on
wavelengths, the 1D-CNN on Φ_F and GATv2 on log ε (Fig. 1). Architecture choice should
therefore follow the property of interest rather than a single headline number.

It is worth putting these numbers next to the earlier BODIPY-specific literature, with the
caveat that the comparison is not like for like. The descriptor and multi-linear-regression
model of Casanola-Martin and co-workers reached a test R² of 0.734 for the absorption
maximum on 131 BODIPYs [19], against 0.912 ± 0.007 here. Their features come from DFT
optimisation and ours do not, their split and test set differ from ours, and their dataset is
roughly a fifth the size of the BODIPY subset used here, so the gap should not be read as a
head-to-head win. What it does suggest is that a larger dataset, a learned representation and
explicit solvent input together buy a useful margin over hand-crafted descriptors on this
particular property.

**Table 1.** Internal test-set R² (mean ± s.d., 5 seeds; chromophore-level split).

| Model | λ_abs | λ_em | Φ_F | log ε |
|---|---|---|---|---|
| MLP | 0.912 ± 0.007 | 0.931 ± 0.009 | 0.508 ± 0.037 | 0.517 ± 0.029 |
| 1D-CNN | 0.885 ± 0.016 | 0.899 ± 0.005 | 0.543 ± 0.046 | 0.327 ± 0.054 |
| GATv2 | 0.865 ± 0.025 | 0.888 ± 0.039 | 0.435 ± 0.049 | 0.527 ± 0.068 |
| Transformer | 0.671 ± 0.042 | 0.644 ± 0.049 | 0.213 ± 0.084 | 0.296 ± 0.052 |

The transformer's weakness fits its appetite for data. With roughly 1,300 training records
it falls behind the descriptor and graph models, which is what one expects in a small-data
regime. That Φ_F and log ε are the hard targets also makes chemical sense. The BODIPY Φ_F
distribution is strongly bimodal, with a large non-emissive population near zero and an
emissive one near unity, and the log ε range in this subset is narrow, so there is little
variance left for a regressor to work with.

### 3.2 How far does this generalise? Two levels of external validation

A random chromophore split is the standard evaluation in this field, but it may still be
optimistic. Dyes reported in the same paper tend to share a scaffold family, a synthetic
route and a measurement protocol, and a random split scatters such relatives across training
and test. To find out how much this matters we held out twelve entire source publications,
190 molecules in all, and retrained from scratch without them (Section 2.6). Table 2 gives
the result.

**Table 2.** Pooled external validation on twelve held-out publications (190 dyes, 790
records; mean ± s.d. over 5 seeds).

| Model | λ_abs MAE (nm) | λ_em MAE (nm) | Φ_F MAE | λ_abs R² | λ_abs ρ | Φ_F ρ |
|---|---|---|---|---|---|---|
| MLP | 30.8 ± 0.7 | 34.5 ± 1.2 | 0.298 ± 0.010 | 0.563 ± 0.040 | 0.744 ± 0.017 | 0.335 ± 0.056 |
| 1D-CNN | 35.2 ± 2.2 | 44.2 ± 2.1 | 0.338 ± 0.004 | 0.440 ± 0.074 | 0.749 ± 0.027 | 0.056 ± 0.038 |
| GATv2 | 36.7 ± 3.6 | 43.3 ± 4.9 | 0.327 ± 0.011 | 0.342 ± 0.141 | 0.651 ± 0.076 | 0.075 ± 0.104 |
| Transformer | 46.4 ± 2.2 | 46.8 ± 1.9 | 0.322 ± 0.010 | 0.034 ± 0.098 | 0.671 ± 0.023 | 0.140 ± 0.044 |

The absorption error roughly doubles, from 18.5 ± 1.0 nm under the random split to
30.8 ± 0.7 nm here, and emission behaves the same way. We take this as the more honest
estimate of what these models do when they meet a study they have never seen, and as evidence
that the usual random-split figures in this literature are systematically flattering. The
ranking of architectures survives the change, with the MLP first and the transformer last, so
the conclusions of Section 3.1 are not an artefact of the split. Rank correlations are now
stable as well, at 0.744 ± 0.017 for absorption, where the earlier eight-compound estimates
swung wildly between seeds.

Quantum yield is where the picture is bleakest. Absolute Φ_F prediction essentially fails on
independent studies (R² = 0.048 ± 0.051 for the MLP, negative for the rest), and only a
modest ranking signal survives (ρ = 0.335 ± 0.056). Whatever the models have learned about
Φ_F does not transfer as a calibrated number across laboratories. Section 3.4 asks whether a
different treatment of the bimodal Φ_F distribution repairs this, and Section 3.5 whether a
richer solvent representation does.

The eight compounds of the independent synthetic study behave very differently (Table 3).
Trained without the twelve publications as well, the models reach 7.2 ± 2.0 nm on absorption,
four times better than on the pooled set, and the Φ_F ranking is recovered well
(ρ = 0.690 ± 0.127, R² = 0.658 ± 0.133 for the 1D-CNN, whose Φ_F range of 0.24 to 1.00 makes
R² meaningful here). These dyes are classical 3,5-dialkyl BODIPYs sitting in a
well-populated part of the training distribution, so the contrast with Table 2 is itself the
finding: accuracy on a new study depends less on the model than on how far that study sits
from what the model has already seen. Any practical use of such models should come with that
caveat attached.

**Table 3.** Independent synthetic study [4] (8 dyes, CHCl₃; mean ± s.d. over 5 seeds).
R² is given for Φ_F only, since the wavelength range spans just 7 nm.

| Model | λ_abs MAE (nm) | λ_em MAE (nm) | Φ_F MAE | Φ_F R² | Φ_F ρ |
|---|---|---|---|---|---|
| 1D-CNN | 7.2 ± 2.0 | 12.4 ± 3.3 | 0.127 ± 0.026 | 0.658 ± 0.133 | 0.690 ± 0.127 |
| MLP | 8.1 ± 1.4 | 20.2 ± 3.1 | 0.179 ± 0.007 | 0.315 ± 0.056 | 0.524 ± 0.121 |
| Transformer | 14.0 ± 4.8 | 16.3 ± 5.1 | 0.164 ± 0.017 | 0.505 ± 0.067 | 0.471 ± 0.181 |
| GATv2 | 15.9 ± 7.7 | 20.5 ± 6.5 | 0.168 ± 0.034 | 0.398 ± 0.246 | 0.452 ± 0.138 |

### 3.3 SMILES canonicalisation is essential for sequence models

The first external evaluation exposed a failure that had nothing to do with chemistry and
everything to do with representation. The eight external SMILES were written in a charged
Kekulé form with [N⁺] and [B⁻]. On those strings the 1D-CNN produced an absorption MAE of
about 104 nm, despite its strong internal performance, while the MLP and GATv2 were
unaffected, since descriptors and graphs do not care how the string was written.
Canonicalising every SMILES with RDKit before tokenisation, in training and at inference
alike, brought the 1D-CNN error down to about 9 nm. Sequence models are sensitive to
arbitrary SMILES conventions in a way that graph and descriptor models are not, so
consistent canonicalisation is a precondition for fair comparison and for any use on
structures supplied from outside.

### 3.4 Does a different parametrisation rescue the quantum yield?

Section 3.2 left Φ_F prediction in poor shape on independent studies, and the shape of the
data offers an obvious suspect. Quantum yields in this subset are strongly bimodal: 18.5 per
cent of the measurements fall below 0.05, essentially dark dyes, while 10.7 per cent lie
above 0.9. A regressor trained with a plain squared error on such a distribution spends much
of its capacity in a region where few points live. We therefore compared four
parametrisations of the same target, using one MLP body, five seeds and both evaluation
levels: the standardised regression used so far, a bounded variant with a sigmoid output, a
logit transform of Φ_F, and the two-stage scheme of a dark/bright classifier multiplied by a
regressor fitted on the emissive subset. Table 5 collects the results.

**Table 5.** Φ_F parametrisations (MLP, mean ± s.d. over 5 seeds). AUC is for the
dark/bright classification at Φ_F = 0.05, available only for the two-stage model.

| Parametrisation | internal MAE | internal R² | external MAE | external R² | dark/bright AUC |
|---|---|---|---|---|---|
| direct (standardised) | 0.204 ± 0.013 | 0.536 ± 0.035 | 0.299 ± 0.007 | 0.070 ± 0.019 | - |
| bounded (sigmoid) | 0.204 ± 0.009 | 0.531 ± 0.030 | 0.301 ± 0.007 | -0.017 ± 0.038 | - |
| logit | 0.172 ± 0.003 | 0.582 ± 0.021 | 0.304 ± 0.004 | -0.097 ± 0.068 | - |
| two-stage | 0.221 ± 0.003 | 0.432 ± 0.024 | 0.296 ± 0.007 | 0.053 ± 0.040 | 0.781 / 0.626 |

The logit transform helps where the distribution is the problem: it cuts the internal error
by about a sixth, from 0.204 to 0.172, and lifts internal R² from 0.536 to 0.582. None of
that survives the move to held-out publications, where its R² is the worst of the four. The
two-stage scheme, the remedy one would reach for first, does not improve regression at all;
it is the weakest internally. Its classifier, on the other hand, is the one component that
does something useful, separating dark from bright dyes with an AUC of 0.781 internally and
0.626 across studies.

The negative result is worth stating clearly, because it locates the problem. Four different
ways of writing the same target run into the same wall externally, so the failure is not an
artefact of the loss function or of the boundary pile-up; it is a genuine shift between
laboratories, and quantum yield is the property most exposed to it, depending as it does on
reference standards and excitation conditions that vary from paper to paper. The practical
consequence is a change of framing rather than of architecture. These models should be used
to rank candidate dyes by expected emissiveness, and to answer the coarser question of
whether a dye will emit at all, which is measurably more learnable than its exact Φ_F.

### 3.5 What does the solvent actually contribute?

SHAP put solvent descriptors at the top of the Φ_F attribution (Section 3.7), which sits
awkwardly with the fact that we had fed the solvent to the models in the crudest possible
way, as ten generic RDKit descriptors concatenated to the chromophore vector. Two questions
follow. Do descriptors with physical meaning do better, and does the way the two are combined
matter? We built a table of physical constants for the 22 solvents covering 97 per cent of
the records, taking dielectric constant, refractive index, the ET(30) polarity scale,
viscosity and the Kamlet-Taft parameters π*, α and β from standard compilations [22,23],
with a missing-value indicator and median imputation for the few gaps. We then compared four
settings: no solvent input at all, the generic descriptors used so far, the physical
descriptors, and the physical descriptors injected by FiLM conditioning, where the solvent
vector scales and shifts the chromophore representation instead of being pasted onto it.

**Table 6.** Solvent representation ablation (MLP, multi-task, mean ± s.d. over 5 seeds).

| Solvent input | internal λ_abs MAE | internal Φ_F MAE | external λ_abs MAE | external Φ_F MAE |
|---|---|---|---|---|
| none | 18.33 ± 1.12 | 0.225 ± 0.004 | 30.00 ± 1.53 | 0.303 ± 0.005 |
| generic descriptors | 18.09 ± 0.71 | 0.197 ± 0.004 | 30.09 ± 0.35 | 0.298 ± 0.008 |
| physical descriptors | 18.74 ± 0.69 | 0.192 ± 0.004 | 29.74 ± 1.31 | 0.300 ± 0.007 |
| physical + FiLM | 18.58 ± 1.30 | 0.192 ± 0.004 | 29.92 ± 2.33 | 0.301 ± 0.008 |

The solvent earns its place, but only for one property. Removing it entirely costs 12 per
cent of the internal Φ_F accuracy, 0.197 rising to 0.225, while the absorption maximum barely
notices, which is what one would expect for a dye whose spectral position is set mainly by
the chromophore and whose emission efficiency is set partly by its surroundings. This agrees
with the SHAP ranking from a different model and a different method. Physical constants beat
generic descriptors for Φ_F by a small but consistent margin, 0.192 against 0.197, so the
chemistry in those constants is being used. FiLM conditioning, however, matches plain
concatenation exactly. Whatever limits the models here, it is not the crudeness of the fusion
mechanism, and the simpler design is enough. As in Section 3.4, none of these gains reach the
held-out publications, where all four settings land within noise of each other.

### 3.6 The models recover known structure-property rules

The central question here is whether the models encode BODIPY photophysics or merely
interpolate. The in-silico *meso*-substituent probe (Fig. 3, Table 4) tests three established
rules directly. H1, that *meso*-alkyl beats *meso*-aryl in Φ_F, is reproduced by all three
ensembles; this is the effect behind the unusually high Φ_F of the alkyl-*meso* reference
dyes in the external set. H3, the heavy-atom series F > Cl > Br > I, is reproduced by the
1D-CNN and most cleanly by GATv2, whose predicted Φ_F falls from 0.304 for fluorine to 0.080
for iodine. Nobody told the models which atoms are heavy, so that monotonic decline comes
from the data alone, and it extends the bromine heavy-atom effect noted in the source study
to iodine. H2, that electron-donating aryl beats phenyl which beats electron-withdrawing
aryl, holds fully only for GATv2. In the other two models the donor and phenyl ordering is
weak and inverted, though electron-withdrawing aryl gives the lowest Φ_F everywhere. It is
worth noting that these rules are recovered even though Section 3.2 showed that absolute Φ_F
does not transfer across studies: the models capture the direction of substituent effects
more reliably than their magnitude.

**Table 4.** Ensemble-mean predicted Φ_F by *meso* class, with hypothesis outcomes
(✓ satisfied, ✗ not). The halogen columns are the 4-halophenyl series.

| Model | alkyl | EDG | phenyl | EWG | F | Cl | Br | I | H1 | H2 | H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1D-CNN | 0.642 | 0.292 | 0.374 | 0.207 | 0.297 | 0.287 | 0.214 | 0.199 | ✓ | ✗ | ✓ |
| GATv2 | 0.584 | 0.331 | 0.319 | 0.258 | 0.304 | 0.240 | 0.213 | 0.080 | ✓ | ✓ | ✓ |
| MLP | 0.448 | 0.299 | 0.403 | 0.229 | 0.209 | 0.201 | 0.236 | 0.110 | ✓ | ✗ | ✗ |

(EDG: electron-donating group. EWG: electron-withdrawing group.)

### 3.7 Attention and SHAP point to the BF₂ core and the meso position

Two attribution methods, applied to different models, agree on where the models look. GATv2
attention (Fig. 4) concentrates on the BF₂ chelate, with a mean atom importance of 0.55
there against 0.45 for peripheral atoms, and the boron atom lands in the top three for every
compound. In the aryl-*meso* dyes the attention also settles on the *meso* bridging carbon,
and for the 4-methoxy dyes the methoxy oxygen stands out. SHAP attribution for the MLP's Φ_F
output (Fig. 5) tells a different but compatible story: solvent descriptors dominate, and
solvent TPSA is the single most important feature at a mean |SHAP| of 0.113 ± 0.018, ahead
of anything belonging to the chromophore. That fits what is known about the solvent
sensitivity of BODIPY quantum yields and justifies feeding solvent information to the models
in the first place. Mapping the most important Morgan bits back to substructures returns
fragments of the BODIPY core itself: the meso-bridge and dipyrromethene motif
`cc(c)C(=C(C)[N⁺])c(c)n`, the BF₂-N linkage `[B⁻]n(c)c`, and the 3,5-alkyl-pyrrole `cc(C)n`.
Attention comes from a graph model and SHAP from a descriptor model, so their agreement on
the same region is worth something.

### 3.8 Accuracy and chemical consistency are separate axes

One theme runs through these results: the model with the best R² is not the most chemically
faithful one. GATv2 is the only architecture that satisfies all three substituent hypotheses,
and it gives the cleanest heavy-atom trend and the clearest attention maps, yet the MLP beats
it on wavelength R². If a model is going to be used to reason about design, and in particular
to extrapolate to substituents nobody has made yet, chemical consistency may be worth more
than a small gain in R². We would report both.

### 3.9 Limitations

Some limitations should be stated plainly. Absolute Φ_F does not transfer across independent
studies: on the pooled external set R² is 0.070 ± 0.019 at best and negative for three of the
four models, so these models should be used to rank candidate dyes by expected emissiveness,
not to quote a quantum yield. Section 3.4 shows that this is not a matter of parametrisation,
since four different ways of writing the target fail in the same way, and Section 3.5 shows
it is not a matter of solvent representation either. The most likely cause is inter-laboratory
variation in how quantum yields are referenced and measured, which no amount of model
engineering on this dataset will remove; a curated multi-laboratory Φ_F benchmark would.

Care is also needed with R² on narrow-range subsets. Within a single publication the spread
of λ_abs is often only a few tens of nanometres, which makes the denominator of R² small and
drives the coefficient sharply negative even when the mean absolute error is unremarkable.
For that reason we report R² only on the pooled external set, where the range is wide, and on
Φ_F for the eight-compound study, whose values span 0.24 to 1.00; per-publication R² is not
informative and we do not quote it. Mean absolute error and rank correlation are the safer
metrics at small sample size.

The physical solvent constants of Section 3.5 are compiled values rather than measurements
made here, and a few entries were unavailable and imputed; the ablation should be read as
evidence that physically meaningful solvent descriptors help Φ_F, not as a calibrated
solvent model. The in-silico probe likewise reports predictions, which are hypotheses rather
than measurements: the iodine and electron-withdrawing cases are chemically plausible but
were not verified experimentally here.

Finally, sequence models are data-hungry at this scale. Transfer learning from the full
chromophore database followed by BODIPY fine-tuning is the obvious next step, with the
transformer likely to gain most.

---

## 4. Conclusions

We have presented a solvent-aware multi-task deep-learning study of BODIPY photophysics that
is judged on chemical fidelity and on generalisation as well as on accuracy. Under a
conventional chromophore-level split the descriptor and graph models place absorption and
emission maxima to within about 15 to 19 nm. Holding out twelve entire source publications
roughly doubles that error, to 30.8 ± 0.7 nm for absorption, and we regard this as the
realistic figure for a study the model has not seen; the widely used random split is
measurably optimistic for this task. On eight dyes from an independent synthesis paper the
error falls back to 7.2 ± 2.0 nm, so what governs accuracy in practice is less the choice of
architecture than the distance between the target dyes and the training distribution.
Quantum yield is the weak point: its absolute value does not transfer across laboratories,
and only a partial ranking signal survives. We tried the obvious repairs and report that they
do not work. A logit transform of Φ_F improves the internal error by a sixth but not the
external one, a two-stage dark/bright model does not improve the regression at all, and
physical solvent constants in place of generic descriptors buy only a small internal gain.
Four parametrisations failing in the same way externally points away from model design and
towards how quantum yields are referenced and measured from one laboratory to the next. The
useful residue is that the coarse question, whether a dye emits at all, is measurably more
learnable than its exact Φ_F, with a dark/bright AUC of 0.781 internally and 0.626 across
studies.

Against that, the models reproduce established structure-property rules without supervision.
*Meso*-alkyl dyes are predicted to emit more efficiently than *meso*-aryl ones, and an
F > Cl > Br > I heavy-atom series emerges on its own, so the direction of substituent effects
is learned even where the magnitude is not. Attention and SHAP localise the models' reasoning
on the BF₂ core and the *meso* position, and solvent descriptors turn out to be the leading
determinant of Φ_F. The graph attention network, which is not the most accurate model by R²,
is the most chemically consistent one, and we argue that both axes belong in any report of a
model meant to guide dye design. The immediate extensions are transfer learning from the
broader chromophore space, which should benefit the sequence models most, and a curated
multi-laboratory quantum-yield benchmark, without which the Φ_F ceiling reported here is
unlikely to move.

---

## Data and code availability

The experimental data come from the open Deep4Chem database (figshare
10.6084/m9.figshare.12045567). All code for data preparation, featurisation, the four models,
multi-seed training, external validation and interpretability, along with the derived result
tables and figures, is available at [repository URL, TODO].

## Author contributions

F.Ö. is the sole author: he designed the study, curated the data, implemented and trained the
models, carried out the analyses and wrote the manuscript.

## Acknowledgements

[TODO: funding, TÜBİTAK/BAP project number if applicable]

## Declaration of generative AI use

[TODO: most publishers now require a statement here. State which tool was used and for what,
for example manuscript drafting and code assistance, and confirm that the author reviewed and
takes responsibility for the content.]

---

## References

Formatted in a generic style; convert to the target journal's style before submission.

1. Loudet, A.; Burgess, K. BODIPY dyes and their derivatives: syntheses and spectroscopic properties. *Chem. Rev.* 2007, 107, 4891-4932.
2. Ulrich, G.; Ziessel, R.; Harriman, A. The chemistry of fluorescent bodipy dyes: versatility unsurpassed. *Angew. Chem. Int. Ed.* 2008, 47, 1184-1201.
3. Boens, N.; Leen, V.; Dehaen, W. Fluorescent indicators based on BODIPY. *Chem. Soc. Rev.* 2012, 41, 1130-1172.
4. Derin, Y.; Yılmaz, R. F.; Baydilek, İ. H.; Enisoğlu Atalay, V.; Özdemir, A.; Tutar, A. Synthesis, electrochemical/photophysical properties and computational investigation of 3,5-dialkyl BODIPY fluorophores. *Inorg. Chim. Acta* 2018, 482, 130-135. DOI: 10.1016/j.ica.2018.06.006.
5. Joung, J. F.; Han, M.; Jeong, M.; Park, S. Experimental database of optical properties of organic compounds. *Sci. Data* 2020, 7, 295. DOI: 10.1038/s41597-020-00634-8.
6. Joung, J. F.; Han, M.; Hwang, J.; Jeong, M.; Choi, D. H.; Park, S. Deep learning optical spectroscopy based on experimental database: potential applications to molecular design. *JACS Au* 2021, 1, 427-438. DOI: 10.1021/jacsau.1c00035.
7. Brody, S.; Alon, U.; Yahav, E. How attentive are graph attention networks? *International Conference on Learning Representations (ICLR)*, 2022. arXiv:2105.14491.
8. Lundberg, S. M.; Lee, S.-I. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, 4765-4774.
9. Joung, J. F.; Han, M.; Jeong, M.; Park, S. DB for chromophore. *figshare* 2020. DOI: 10.6084/m9.figshare.12045567.
10. Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. *J. Chem. Inf. Comput. Sci.* 1988, 28, 31-36.
11. Rogers, D.; Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* 2010, 50, 742-754.
12. Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org (accessed 2026).
13. Paszke, A.; Gross, S.; Massa, F.; et al. PyTorch: an imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, 8024-8035.
14. Fey, M.; Lenssen, J. E. Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019. arXiv:1903.02428.
15. Kingma, D. P.; Ba, J. Adam: a method for stochastic optimization. *International Conference on Learning Representations (ICLR)*, 2015. arXiv:1412.6980.
16. Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. Graph attention networks. *International Conference on Learning Representations (ICLR)*, 2018. arXiv:1710.10903.
17. Bemis, G. W.; Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* 1996, 39, 2887-2893.
18. Vaswani, A.; Shazeer, N.; Parmar, N.; et al. Attention is all you need. *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, 5998-6008.
19. Casanola-Martin, G. M.; Wang, J.; Zhou, J.-G.; Rasulev, B.; Leszczynski, J. Chemical feature-based machine learning model for predicting photophysical properties of BODIPY compounds: density functional theory and quantitative structure-property relationship modeling. *J. Mol. Model.* 2025, 31, 18. DOI: 10.1007/s00894-024-06240-4.
20. Buglak, A. A.; Charisiadis, A.; Sheehan, A.; Kingsbury, C. J.; Senge, M. O.; Filatov, M. A. Quantitative structure-property relationship modelling for the prediction of singlet oxygen generation by heavy-atom-free BODIPY photosensitizers. *Chem. Eur. J.* 2021, 27, 9934-9947. DOI: 10.1002/chem.202100922.
21. Chebotaev, P. P.; Buglak, A. A.; Sheehan, A.; Filatov, M. A. Predicting fluorescence to singlet oxygen generation quantum yield ratio for BODIPY dyes using QSPR and machine learning. *Phys. Chem. Chem. Phys.* 2024, 26, 25131-25142. DOI: 10.1039/D4CP02471K.
22. Reichardt, C. Solvatochromic dyes as solvent polarity indicators. *Chem. Rev.* 1994, 94, 2319-2358.
23. Kamlet, M. J.; Abboud, J.-L. M.; Abraham, M. H.; Taft, R. W. Linear solvation energy relationships. 23. A comprehensive collection of the solvatochromic parameters π*, α and β, and some methods for simplifying the generalized solvatochromic equation. *J. Org. Chem.* 1983, 48, 2877-2887.

---

## Figures and tables

- Fig. 1. Architecture comparison, internal test R² (mean ± s.d.). `results/multiseed_R2.png`
- Fig. 2. Pooled external validation, twelve held-out publications (190 dyes). `results/extpool_parity.png`
- Fig. 3. In-silico *meso*-substituent probe: predicted Φ_F by substituent (`results/probe_QY.png`) and the halogen series (`results/probe_heavy_atom.png`)
- Fig. 4. GATv2 attention-derived atom importance. `results/attention_maps.png`
- Fig. 5. SHAP feature attribution for Φ_F (MLP). `results/shap_summary.png`
- Fig. 6. Φ_F parametrisations compared at both evaluation levels. `results/qy_bimodal.png`
- Fig. 7. Solvent representation ablation. `results/solvent_ablation.png`
- Fig. S1. BODIPY target distributions. `results/eda_bodipy_dist.png`
- Fig. S2. Error distribution across individual held-out publications. `results/lopo_box.png`
- Fig. S3. Parity plots on the eight-dye independent study. `results/external_parity.png`
- Table 1. Internal test R². `results/multiseed_internal_summary.csv`
- Table 2. Pooled external validation. `results/extpool_summary.csv`
- Table 3. Independent synthetic study. `results/extpool_summary.csv`
- Table 4. Substituent-probe hypotheses. `results/probe_summary.csv`
- Table 5. Φ_F parametrisations. `results/qy_bimodal_summary.csv`
- Table 6. Solvent representation ablation. `results/solvent_ablation_summary.csv`
- Table S1. Per-publication breakdown of external errors. `results/extpool_perpub.csv`
- Table S2. Per-compound predictions for the eight-dye study. `results/external_predictions.csv`
- Table S3. Top SHAP Morgan bits mapped to substructures. `results/shap_bits.csv`
