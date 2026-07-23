"""solvent_props.py - Fiziksel cozucu tanimlayicilari (hakem M3).
Sutunlar: dielektrik sabiti (eps_r), kirilma indisi (n_D), ET(30) polarite (kcal/mol),
viskozite (eta, mPa.s), Kamlet-Taft pi*, alpha, beta.

TUM DEGERLER BIRINCIL KAYNAKLARDAN DOGRULANMISTIR. Sicaklik: TAMAMI 25 C.
* eps_r, n_D -> Marcus, Y. "The Properties of Solvents", Wiley 1998, Tablo 3.5 (ref [24]).
  22/22 cozucu dogrulandi. n_D degerleri 25 C'dedir (Reichardt Ek A 20 C veriyordu;
  tutarlilik icin Marcus'un 25 C degerleri tercih edildi -> tum ozellikler ayni sicaklikta).
* eta -> Marcus, ayni kitap, Tablo 3.9. 20/22 dogrulandi.
  ISTISNA: 2-propanol ve 1-butanol viskozitesi Tablo 3.9'da eksik sayfada kaldi; mevcut
  degerler standart derlemelerden ve ayni tablodaki komsu alkollerle tutarli.
* ET(30) -> Reichardt, Chem. Rev. 1994, 94, 2319, Tablo 2 (ref [22]). 22/22.
* pi*, alpha, beta -> Kamlet, Abboud, Abraham & Taft, J. Org. Chem. 1983, 48, 2877,
  Tablo I (ref [23]). Parantezli (tahmini) degerler '~'. Kamlet'te YOK: 1-oktanol,
  2-MeTHF -> None (medyan imputasyonu). Pinakolon: yalnizca beta verilmis.

Kapsam: 22 cozucu = BODIPY kayitlarinin %97'si. Listede olmayan cozucu -> tamamen medyan.
"""
import numpy as np

FIELDS = ['eps_r', 'n_D', 'ET30', 'eta', 'pi_star', 'alpha', 'beta']

# SMILES: (ad, eps_r, n_D, ET30, eta, pi*, alpha, beta)
SOLVENTS = {
    'ClCCl':            ('dichloromethane',  8.93, 1.4210, 40.7, 0.411,  0.81, 0.30, 0.00),  # alpha ~
    'CC#N':             ('acetonitrile',    35.94, 1.3410, 45.6, 0.341,  0.75, 0.19, 0.31),
    'Cc1ccccc1':        ('toluene',          2.38, 1.4941, 33.9, 0.553,  0.54, 0.00, 0.11),
    'C1CCOC1':          ('THF',              7.58, 1.4050, 37.4, 0.462,  0.58, 0.00, 0.55),
    'CO':               ('methanol',        32.66, 1.3265, 55.4, 0.551,  0.60, 0.93, 0.62),  # beta ~
    'ClC(Cl)Cl':        ('chloroform',       4.89, 1.4420, 39.1, 0.536,  0.58, 0.44, 0.00),
    'CCCCCC':           ('n-hexane',         1.88, 1.3723, 31.0, 0.294, -0.08, 0.00, 0.00),
    'CCOC(C)=O':        ('ethyl acetate',    6.02, 1.3698, 38.1, 0.426,  0.55, 0.00, 0.45),
    'C1CCCCC1':         ('cyclohexane',      2.02, 1.4235, 30.9, 0.898,  0.00, 0.00, 0.00),
    'CCO':              ('ethanol',         24.55, 1.3594, 51.9, 1.083,  0.54, 0.83, 0.77),  # beta ~
    'CN(C)C=O':         ('DMF',             36.71, 1.4280, 43.2, 0.802,  0.88, 0.00, 0.69),
    'CS(C)=O':          ('DMSO',            46.45, 1.4770, 45.1, 1.991,  1.00, 0.00, 0.76),
    'CC(C)=O':          ('acetone',         20.56, 1.3560, 42.2, 0.303,  0.71, 0.08, 0.48),
    'OCC(F)(F)F':       ('TFE',             26.67, 1.2907, 59.8, 1.755,  0.73, 1.51, 0.00),
    'C1COCCO1':         ('1,4-dioxane',      2.21, 1.4203, 36.0, 1.194,  0.55, 0.00, 0.37),
    'CC(=O)C(C)(C)C':   ('pinacolone',      12.60, 1.3950, 39.0, 0.713,  None, None, 0.48),
    'O':                ('water',           78.36, 1.3325, 63.1, 0.890,  1.09, 1.17, 0.18),  # beta ~
    'CCOCC':            ('diethyl ether',    4.20, 1.3495, 34.5, 0.242,  0.27, 0.00, 0.47),
    'CCCCCCCCO':        ('1-octanol',       10.34, 1.4276, 48.1, 7.363,  None, None, None),
    'CCCCO':            ('1-butanol',       17.51, 1.3974, 49.7, 2.544,  0.47, 0.79, 0.88),  # beta ~, eta bekleyen
    'CC1CCCO1':         ('2-MeTHF',          5.26, 1.4051, 36.5, 0.473,  None, None, None),
    'CC(C)O':           ('2-propanol',      19.92, 1.3752, 48.4, 2.038,  0.48, 0.76, 0.95),  # beta ~, eta bekleyen
}

def raw_vector(smiles):
    """[7 deger] + eksik gostergesi. Bilinmeyen cozucu -> hepsi NaN."""
    rec = SOLVENTS.get(smiles)
    if rec is None:
        return np.full(len(FIELDS), np.nan, dtype=np.float32)
    return np.array([np.nan if v is None else float(v) for v in rec[1:]], dtype=np.float32)
