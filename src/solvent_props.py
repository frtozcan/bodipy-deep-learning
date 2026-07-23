"""solvent_props.py - Fiziksel cozucu tanimlayicilari (hakem M3).
Sutunlar: dielektrik sabiti (eps_r), kirilma indisi (n_D), ET(30) polarite (kcal/mol),
viskozite (eta, mPa.s), Kamlet-Taft pi*, alpha, beta.

KAMLET-TAFT (pi*, alpha, beta): Kamlet, Abboud, Abraham & Taft, J. Org. Chem. 1983, 48,
2877-2887, Tablo I'den DOGRULANDI (ref [23]). Kaynakta parantezli (tahmini) degerler '~'
ile isaretli. Kamlet'te OLMAYAN: 1-oktanol, 2-MeTHF -> None. Pinakolon = Tablo I no. 65
"methyl tert-butyl ketone": yalnizca beta verilmis.

ET(30): Reichardt, Chem. Rev. 1994, 94, 2319-2358, Tablo 2'den DOGRULANDI (ref [22]).
22 cozucunun 20'si birebir; 1-oktanol 48.3->48.1, pinakolon 39.0 eklendi.

eps_r ve n_D(20 C): Reichardt & Welton, "Solvents and Solvent Effects in Organic Chemistry",
4. baski, Wiley-VCH 2011, Ek A, Tablo A-1'den DOGRULANDI. 10 cozucu birebir tutti;
8 duzeltme yapildi (aseton eps 20.70->20.56, CHCl3 4.81->4.89, Et2O 4.27->4.20,
THF nD 1.4050->1.4072, toluen 1.4961->1.4969, 2-PrOH 1.3776->1.3772, MeCN 1.3442->1.3441,
EtOAc 1.3723->1.3724); pinakolon eps 12.60 / nD 1.3952 EKLENDI (Tablo A-1 no. 60).
Tablo A-1'de BULUNMAYAN (deger baska derlemelerden, DOGRULANMADI): TFE, 1-oktanol, 2-MeTHF.

eta (viskozite): standart derlemelerden; HENUZ DOGRULANMADI (Reichardt Ek A'da viskozite yok).
Kapsam: 22 cozucu = BODIPY kayitlarinin %97'si. Listede olmayan cozucu -> tamamen medyan.
"""
import numpy as np

FIELDS = ['eps_r', 'n_D', 'ET30', 'eta', 'pi_star', 'alpha', 'beta']

# SMILES: (ad, eps_r, n_D, ET30, eta, pi*, alpha, beta)   [# = dogrulanmamis alan]
SOLVENTS = {
    'ClCCl':            ('dichloromethane',  8.93, 1.4242, 40.7, 0.413,  0.81, 0.30, 0.00),  # alpha ~
    'CC#N':             ('acetonitrile',    35.94, 1.3441, 45.6, 0.369,  0.75, 0.19, 0.31),
    'Cc1ccccc1':        ('toluene',          2.38, 1.4969, 33.9, 0.560,  0.54, 0.00, 0.11),
    'C1CCOC1':          ('THF',              7.58, 1.4072, 37.4, 0.456,  0.58, 0.00, 0.55),
    'CO':               ('methanol',        32.66, 1.3284, 55.4, 0.544,  0.60, 0.93, 0.62),  # beta ~
    'ClC(Cl)Cl':        ('chloroform',       4.89, 1.4459, 39.1, 0.537,  0.58, 0.44, 0.00),
    'CCCCCC':           ('n-hexane',         1.88, 1.3749, 31.0, 0.300, -0.08, 0.00, 0.00),
    'CCOC(C)=O':        ('ethyl acetate',    6.02, 1.3724, 38.1, 0.423,  0.55, 0.00, 0.45),
    'C1CCCCC1':         ('cyclohexane',      2.02, 1.4262, 30.9, 0.898,  0.00, 0.00, 0.00),
    'CCO':              ('ethanol',         24.55, 1.3614, 51.9, 1.074,  0.54, 0.83, 0.77),  # beta ~
    'CN(C)C=O':         ('DMF',             36.71, 1.4305, 43.2, 0.802,  0.88, 0.00, 0.69),
    'CS(C)=O':          ('DMSO',            46.45, 1.4793, 45.1, 1.996,  1.00, 0.00, 0.76),
    'CC(C)=O':          ('acetone',         20.56, 1.3587, 42.2, 0.306,  0.71, 0.08, 0.48),
    'OCC(F)(F)F':       ('TFE',             26.67, 1.2907, 59.8, None,   0.73, 1.51, 0.00),  # eps/nD dogrulanmadi
    'C1COCCO1':         ('1,4-dioxane',      2.21, 1.4224, 36.0, 1.177,  0.55, 0.00, 0.37),
    'CC(=O)C(C)(C)C':   ('pinacolone',      12.60, 1.3952, 39.0, None,   None, None, 0.48),
    'O':                ('water',           78.36, 1.3330, 63.1, 0.890,  1.09, 1.17, 0.18),  # beta ~
    'CCOCC':            ('diethyl ether',    4.20, 1.3524, 34.5, 0.224,  0.27, 0.00, 0.47),
    'CCCCCCCCO':        ('1-octanol',       10.30, 1.4295, 48.1, 7.288,  None, None, None),  # eps/nD dogrulanmadi
    'CCCCO':            ('1-butanol',       17.51, 1.3993, 49.7, 2.544,  0.47, 0.79, 0.88),  # beta ~
    'CC1CCCO1':         ('2-MeTHF',          6.97, 1.4059, 36.5, None,   None, None, None),  # eps/nD dogrulanmadi
    'CC(C)O':           ('2-propanol',      19.92, 1.3772, 48.4, 2.038,  0.48, 0.76, 0.95),  # beta ~
}

def raw_vector(smiles):
    """[7 deger] + eksik gostergesi. Bilinmeyen cozucu -> hepsi NaN."""
    rec = SOLVENTS.get(smiles)
    if rec is None:
        return np.full(len(FIELDS), np.nan, dtype=np.float32)
    return np.array([np.nan if v is None else float(v) for v in rec[1:]], dtype=np.float32)
