"""data_utils.py - Ortak veri yardimcilari (BODIPY tanimi, kolon adlari, filtre).
feature_extraction / data_loader / build_dataset hepsi buradan import etsin ki
BODIPY tanimi ve hedef kolonlar TEK yerde tanimli olsun.
"""
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, 'data', 'raw', 'DB for chromophore_Sci_Data_rev03.csv')

# Kisa ad -> ham CSV kolon adi
TARGETS = {
    'lambda_abs': 'Absorption max (nm)',
    'lambda_em' : 'Emission max (nm)',
    'QY'        : 'Quantum yield',
    'log_eps'   : 'log(e/mol-1 dm3 cm-1)',
}
SMILES_COL = 'Chromophore'
SOLVENT_COL = 'Solvent'

# BODIPY ailesi filtresi: BF2-N2 selat motifi. [#7] = atom no 7 (azot); aromatiklik/yuk
# durumundan bagimsiz eslesir, cunku kaynak SMILES'ler tek bicimli yazilmamis.
BODIPY_SMARTS = Chem.MolFromSmarts('[F][B]([F])([#7])[#7]')

# Klasik 4,4-difloro-4-bora-3a,4a-diaza-s-indasen iskeleti: bor 6-uyeli selat halkasinda,
# her iki azot 5-uyeli heterosiklde. Ailenin geri kalani (6-uyeli N-heterosikl iceren
# akrabalar) filtreye takilmaz ama veri setinde TUTULUR; ayrim strict_core sutununda.
BODIPY_STRICT_SMARTS = Chem.MolFromSmarts('[F][B;r6]([F])([#7;r5])[#7;r5]')

def is_bodipy(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(BODIPY_SMARTS)

def is_strict_core(smi):
    """Klasik dipirometen iskeleti mi (True) yoksa yakin akraba mi (False)."""
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(BODIPY_STRICT_SMARTS)

def load_raw():
    return pd.read_csv(RAW_CSV)

def filter_bodipy(df):
    uniq = df.drop_duplicates(SMILES_COL)[[SMILES_COL]].copy()
    uniq['is_bodipy'] = uniq[SMILES_COL].apply(is_bodipy)
    keep = set(uniq.loc[uniq['is_bodipy'], SMILES_COL])
    return df[df[SMILES_COL].isin(keep)].copy()
