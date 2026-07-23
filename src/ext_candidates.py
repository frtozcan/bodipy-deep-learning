"""ext_candidates.py - Harici set genisletme icin aday analizi.
Deep4Chem'deki BODIPY kayitlarini KAYNAK YAYIN (Reference DOI) bazinda gruplar.
Amac: rastgele bolme yerine "yayin bazinda disarida birakma" ile 20-30+ bilesikli,
gercekten bagimsiz bir harici dogrulama seti kurmak.
Calistir: python src/ext_candidates.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from data_utils import load_raw, filter_bodipy, SMILES_COL, TARGETS, ROOT

df = filter_bodipy(load_raw())
for k, col in TARGETS.items():
    df[k] = pd.to_numeric(df[col], errors='coerce')

REF = 'Reference'
print("BODIPY kaydi:", len(df), "| benzersiz molekul:", df[SMILES_COL].nunique())
print("Benzersiz kaynak yayin:", df[REF].nunique())

g = df.groupby(REF).agg(
    mol=(SMILES_COL, 'nunique'), kayit=(SMILES_COL, 'size'),
    abs_n=('lambda_abs', 'count'), em_n=('lambda_em', 'count'),
    qy_n=('QY', 'count'), eps_n=('log_eps', 'count'),
    qy_min=('QY', 'min'), qy_max=('QY', 'max'),
    abs_min=('lambda_abs', 'min'), abs_max=('lambda_abs', 'max'),
).sort_values('mol', ascending=False)

print("\n=== EN COK BODIPY ICEREN 15 YAYIN ===")
print(g.head(15).to_string())

# Harici set icin ideal adaylar: cok molekul + QY dolu + genis spektral aralik
cand = g[(g['mol'] >= 5) & (g['qy_n'] >= 5)].copy()
cand['abs_span'] = cand['abs_max'] - cand['abs_min']
cand['qy_span'] = cand['qy_max'] - cand['qy_min']
print(f"\n>=5 molekul VE >=5 QY olcumu olan yayin sayisi: {len(cand)}")
print("Bu yayinlarin toplam molekul sayisi:", int(cand['mol'].sum()))
print("\n=== HARICI SET ADAYLARI (spektral genislige gore) ===")
print(cand.sort_values('abs_span', ascending=False).head(12).to_string())
