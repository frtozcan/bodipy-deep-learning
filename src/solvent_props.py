"""solvent_props.py - Fiziksel cozucu tanimlayicilari (hakem M3).
Sutunlar: dielektrik sabiti (eps_r), kirilma indisi (n_D), ET(30) polarite (kcal/mol),
viskozite (eta, mPa.s), Kamlet-Taft pi*, alpha, beta.

!! ONEMLI: Degerler standart derlemelerden (Reichardt; Marcus) alinmis TAHMINI degerlerdir.
   GONDERIM ONCESI birincil kaynaktan DOGRULANMALIDIR. Emin olunmayan alanlar None
   birakilmistir; egitim medyani ile doldurulur ve ayrica 'eksik' gostergesi eklenir.
Kapsam: 22 cozucu = BODIPY kayitlarinin %97'si. Listede olmayan cozucu -> tamamen medyan.
"""
import numpy as np

FIELDS = ['eps_r', 'n_D', 'ET30', 'eta', 'pi_star', 'alpha', 'beta']

# SMILES: (ad, eps_r, n_D, ET30, eta, pi*, alpha, beta)
SOLVENTS = {
    'ClCCl':            ('dichloromethane',  8.93, 1.4242, 40.7, 0.413,  0.82, 0.13, 0.10),
    'CC#N':             ('acetonitrile',    35.94, 1.3442, 45.6, 0.369,  0.75, 0.19, 0.31),
    'Cc1ccccc1':        ('toluene',          2.38, 1.4961, 33.9, 0.560,  0.54, 0.00, 0.11),
    'C1CCOC1':          ('THF',              7.58, 1.4050, 37.4, 0.456,  0.58, 0.00, 0.55),
    'CO':               ('methanol',        32.66, 1.3284, 55.4, 0.544,  0.60, 0.98, 0.66),
    'ClC(Cl)Cl':        ('chloroform',       4.81, 1.4459, 39.1, 0.537,  0.58, 0.44, 0.10),
    'CCCCCC':           ('n-hexane',         1.88, 1.3749, 31.0, 0.300, -0.04, 0.00, 0.00),
    'CCOC(C)=O':        ('ethyl acetate',    6.02, 1.3723, 38.1, 0.423,  0.55, 0.00, 0.45),
    'C1CCCCC1':         ('cyclohexane',      2.02, 1.4262, 30.9, 0.898,  0.00, 0.00, 0.00),
    'CCO':              ('ethanol',         24.55, 1.3614, 51.9, 1.074,  0.54, 0.86, 0.75),
    'CN(C)C=O':         ('DMF',             36.71, 1.4305, 43.2, 0.802,  0.88, 0.00, 0.69),
    'CS(C)=O':          ('DMSO',            46.45, 1.4793, 45.1, 1.996,  1.00, 0.00, 0.76),
    'CC(C)=O':          ('acetone',         20.70, 1.3588, 42.2, 0.306,  0.71, 0.08, 0.43),
    'OCC(F)(F)F':       ('TFE',             26.67, 1.2907, 59.8, None,   0.73, 1.51, 0.00),
    'C1COCCO1':         ('1,4-dioxane',      2.21, 1.4224, 36.0, 1.177,  0.55, 0.00, 0.37),
    'CC(=O)C(C)(C)C':   ('pinacolone',       None, None,   None, None,   None, None, None),
    'O':                ('water',           78.36, 1.3330, 63.1, 0.890,  1.09, 1.17, 0.47),
    'CCOCC':            ('diethyl ether',    4.27, 1.3524, 34.5, 0.224,  0.27, 0.00, 0.47),
    'CCCCCCCCO':        ('1-octanol',       10.30, 1.4295, 48.3, 7.288,  0.40, 0.77, 0.81),
    'CCCCO':            ('1-butanol',       17.51, 1.3993, 49.7, 2.544,  0.47, 0.84, 0.84),
    'CC1CCCO1':         ('2-MeTHF',          6.97, 1.4059, 36.5, None,   0.53, 0.00, 0.58),
    'CC(C)O':           ('2-propanol',      19.92, 1.3776, 48.4, 2.038,  0.48, 0.76, 0.84),
}

def raw_vector(smiles):
    """[7 deger] + eksik gostergesi. Bilinmeyen cozucu -> hepsi NaN."""
    rec = SOLVENTS.get(smiles)
    if rec is None:
        return np.full(len(FIELDS), np.nan, dtype=np.float32)
    return np.array([np.nan if v is None else float(v) for v in rec[1:]], dtype=np.float32)
