"""external_build.py - Derin et al. 2018 (Inorg. Chim. Acta 482, 130-135) 8 BODIPY'si.
SMILES (yapıdan) + ölçülen değerler -> data/external/derin2018.csv. RDKit ile formül doğrular.
Çözücü: CHCl3 (Tablo 1 / Şekil 2). Çalıştır: python src/external_build.py
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from data_utils import ROOT, is_bodipy

OUT = os.path.join(ROOT, 'data', 'external', 'derin2018.csv')
SOLVENT = 'ClC(Cl)Cl'   # CHCl3

def core(meso, r35):
    # meso-substitue, 3,5-dialkil BODIPY (1,2,6,7 = H); yüklü Kekulé: [B-] + [N+]
    return f"C({meso})(=C2C=CC({r35})=[N+]2[B-]3(F)F)C4=CC=C({r35})N34"

# id, meso, R(3,5), abs(nm), em(nm), QY, eps(x1e4), beklenen formül
rows = [
    ('1A', 'CC',            'C',  508, 514, 1.00, 6.8, 'C13H15BF2N2'),
    ('1B', 'CC',            'CC', 509, 515, 0.93, 7.3, 'C15H19BF2N2'),
    ('2A', 'c1ccccc1',      'C',  512, 525, 0.36, 5.7, 'C17H15BF2N2'),
    ('2B', 'c1ccccc1',      'CC', 514, 526, 0.37, 5.8, 'C19H19BF2N2'),
    ('3A', 'c1ccc(OC)cc1',  'C',  511, 522, 0.45, 7.1, 'C18H17BF2N2O'),
    ('3B', 'c1ccc(OC)cc1',  'CC', 512, 523, 0.51, 7.0, 'C20H21BF2N2O'),
    ('4A', 'c1ccc(Br)cc1',  'C',  514, 529, 0.24, 6.6, 'C17H14BBrF2N2'),
    ('4B', 'c1ccc(Br)cc1',  'CC', 515, 528, 0.25, 5.6, 'C19H18BBrF2N2'),
]

recs, allok = [], True
for cid, meso, r35, ab, em, qy, eps, formula in rows:
    smi = core(meso, r35)
    m = Chem.MolFromSmiles(smi)
    ok = m is not None
    f = rdMolDescriptors.CalcMolFormula(m) if ok else None
    bod = is_bodipy(smi) if ok else False
    match = (f == formula)
    allok = allok and ok and bod and match
    print(f"{cid}: parse={ok} bodipy={bod} formula={f} (beklenen {formula}) eslesme={match}")
    recs.append({'id': cid, 'Chromophore': smi, 'Solvent': SOLVENT,
                 'lambda_abs': ab, 'lambda_em': em, 'QY': qy,
                 'log_eps': round(float(np.log10(eps * 1e4)), 4)})

df = pd.DataFrame(recs)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print("\nTUMU GECERLI:", allok)
print("Kayit:", OUT)
print(df.to_string(index=False))
