"""build_dataset.py - BODIPY alt-kumesini suz, KROMOFOR-bazli split ekle,
data/processed/bodipy.csv yaz + hedef dagilim grafigini results/'a kaydet.
Calistir: python src/build_dataset.py
"""
import os
import pandas as pd
from data_utils import (load_raw, filter_bodipy, TARGETS,
                        SMILES_COL, SOLVENT_COL, ROOT)

SEED = 42
OUT = os.path.join(ROOT, 'data', 'processed', 'bodipy.csv')
PNG = os.path.join(ROOT, 'results', 'eda_bodipy_dist.png')

df = load_raw()
bod = filter_bodipy(df)

# Kolonlari sadelestir + kisa hedef adlari
extra = ['Lifetime (ns)', 'Molecular weight (g mol-1)', 'Reference']
keep = [SMILES_COL, SOLVENT_COL] + list(TARGETS.values()) + extra
keep = [c for c in keep if c in bod.columns]
bod = bod[keep].rename(columns={v: k for k, v in TARGETS.items()})
for k in TARGETS:
    bod[k] = pd.to_numeric(bod[k], errors='coerce')

# --- KROMOFOR-bazli split 70/15/15 (sizinti onleme) ---
# Not: tum BODIPY'ler ayni cekirdek scaffold'u paylasir; bu yuzden Bemis-Murcko
# scaffold-split anlamsiz -> molekul (Chromophore) bazli rastgele bolme dogru secim.
mols = bod[SMILES_COL].drop_duplicates().sample(frac=1, random_state=SEED).tolist()
n = len(mols); n_tr = int(0.70 * n); n_va = int(0.15 * n)
sp = {m: ('train' if i < n_tr else 'val' if i < n_tr + n_va else 'test')
      for i, m in enumerate(mols)}
bod['split'] = bod[SMILES_COL].map(sp)

# Klasik dipirometen iskeleti mi, yakin akraba mi (yalnizca ETIKET; filtre DEGIL)
from data_utils import is_strict_core
_sc = {s: is_strict_core(s) for s in bod[SMILES_COL].unique()}
bod['strict_core'] = bod[SMILES_COL].map(_sc)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
bod.to_csv(OUT, index=False)

# --- Ozet ---
print("BODIPY satir:", len(bod), "| benzersiz mol:", bod[SMILES_COL].nunique())
print("Split (satir) :", bod['split'].value_counts().to_dict())
print("Split (mol)   :", bod.groupby('split')[SMILES_COL].nunique().to_dict())
print("Her split'te hedef doluluk:")
for s in ['train', 'val', 'test']:
    sub = bod[bod['split'] == s]
    print(f"  {s:5s}:", {k: int(sub[k].notna().sum()) for k in TARGETS})
print("Kayit:", OUT)

# --- Hizli dagilim grafigi ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    disp = {'lambda_abs': 'Abs. λmax (nm)', 'lambda_em': 'Em. λmax (nm)',
            'QY': 'ΦF (quantum yield)', 'log_eps': 'log ε'}
    for ax, k in zip(axes.ravel(), TARGETS):
        bod[k].dropna().hist(bins=40, ax=ax, color='#3b7a57', edgecolor='white')
        ax.set_title(f"{disp.get(k, k)} (n={int(bod[k].notna().sum())})")
    fig.suptitle("BODIPY subset — target distributions")
    fig.tight_layout()
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.savefig(PNG, dpi=120)
    print("Grafik:", PNG)
except Exception as e:
    print("Grafik atlandi:", repr(e))
