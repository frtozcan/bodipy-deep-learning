"""feature_extraction.py - BODIPY çok-görevli regresyon için featurizer'lar (saf RDKit/numpy).
Üretilen temsiller (her satır için):
  - graph : kromofor moleküler grafı (atom 17-dim, bağ 7-dim) -> GNN/GATv2
  - morgan: kromofor Morgan FP (2048) + kromofor global descriptor (10) -> MLP
  - solv  : çözücü global descriptor (10)
  - tokens: "kromofor [SEP] çözücü" SMILES token id dizisi -> 1D-CNN / Transformer
  - targets(4) + mask(4): [lambda_abs, lambda_em, QY, log_eps]
Standartlaştırma istatistikleri YALNIZCA train split'ten hesaplanır (sızıntı yok).
Çıktı: data/processed/features.pkl + data/processed/meta.json
Çalıştır: python src/feature_extraction.py
"""
import os, re, json, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from data_utils import TARGETS, SMILES_COL, SOLVENT_COL, ROOT

TARGET_NAMES = list(TARGETS)            # ['lambda_abs','lambda_em','QY','log_eps']
PROC = os.path.join(ROOT, 'data', 'processed')
CSV = os.path.join(PROC, 'bodipy.csv')

# ---------- Graf featurizer (atom 17 / bağ 7) ----------
SYMBOLS = ['C', 'N', 'O', 'F', 'B', 'S', 'Cl', 'Br']   # +other = 9
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3]                      # +other = 4

def _onehot(x, choices):
    v = [1.0 if x == c else 0.0 for c in choices]
    v.append(1.0 if x not in choices else 0.0)          # 'other' biti
    return v

def atom_features(a):                                   # 9+3+4+1 = 17
    f = _onehot(a.GetSymbol(), SYMBOLS)
    f += [float(a.GetDegree()), float(a.GetFormalCharge()), float(a.GetTotalNumHs())]
    f += _onehot(a.GetHybridization(), HYB)
    f += [1.0 if a.GetIsAromatic() else 0.0]
    return f

def bond_features(b):                                    # 4+1+1+1 = 7
    bt = b.GetBondType()
    types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    f = [1.0 if bt == t else 0.0 for t in types]
    f += [1.0 if b.GetIsConjugated() else 0.0,
          1.0 if b.IsInRing() else 0.0,
          1.0 if b.GetStereo() != Chem.BondStereo.STEREONONE else 0.0]
    return f

def mol_to_graph(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    x = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, eattr = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]; dst += [j, i]; eattr += [bf, bf]   # çift yönlü
    if len(src) == 0:                                     # tek atom (ör. çözücü 'O')
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 7), dtype=np.float32)
    else:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(eattr, dtype=np.float32)
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}

# ---------- Morgan FP + global descriptor ----------
DESC_FUNCS = [
    ('MolWt', Descriptors.MolWt), ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA), ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('NumHDonors', Descriptors.NumHDonors), ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('NumAromaticRings', Descriptors.NumAromaticRings), ('FractionCSP3', Descriptors.FractionCSP3),
    ('RingCount', Descriptors.RingCount), ('NumHeteroatoms', Descriptors.NumHeteroatoms),
]
DESC_DIM = len(DESC_FUNCS)   # 10
MORGAN_BITS = 2048

def morgan_fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(MORGAN_BITS, dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=MORGAN_BITS)
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr

def global_desc(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(DESC_DIM, dtype=np.float32)
    out = []
    for _, fn in DESC_FUNCS:
        try:
            v = fn(m)
        except Exception:
            v = 0.0
        out.append(0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
    return np.array(out, dtype=np.float32)

# ---------- SMILES tokenizer (regex, atom-seviyesi) ----------
SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
SPECIALS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[SEP]']

def tokenize(smi):
    return SMI_REGEX.findall(smi)

def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi

def build_vocab(smiles_iter):
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for smi in smiles_iter:
        for t in tokenize(_canon(smi)):
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

def encode_pair(chrom, solv, vocab):
    unk = vocab['[UNK]']
    ids = [vocab['[BOS]']]
    ids += [vocab.get(t, unk) for t in tokenize(_canon(chrom))]
    ids += [vocab['[SEP]']]
    ids += [vocab.get(t, unk) for t in tokenize(_canon(solv))]
    ids += [vocab['[EOS]']]
    return ids

# ---------- Ana akış ----------
def main():
    df = pd.read_csv(CSV)
    for k in TARGET_NAMES:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    train = df[df['split'] == 'train']

    # Vocab: yalnızca train SMILES (kromofor + çözücü) -> sızıntı yok
    vocab = build_vocab(list(train[SMILES_COL]) + list(train[SOLVENT_COL]))

    # Standartlaştırma istatistikleri (yalnızca train)
    tgt_mean = {k: float(train[k].mean()) for k in TARGET_NAMES}
    tgt_std = {k: float(train[k].std() if train[k].std() > 1e-6 else 1.0) for k in TARGET_NAMES}
    chrom_desc_tr = np.stack([global_desc(s) for s in train[SMILES_COL]])
    solv_desc_tr = np.stack([global_desc(s) for s in train[SOLVENT_COL]])
    cd_mean, cd_std = chrom_desc_tr.mean(0), chrom_desc_tr.std(0) + 1e-8
    sd_mean, sd_std = solv_desc_tr.mean(0), solv_desc_tr.std(0) + 1e-8

    samples, max_len = [], 0
    gcache = {}
    for _, r in df.iterrows():
        chrom, solv = r[SMILES_COL], r[SOLVENT_COL]
        if chrom not in gcache:
            gcache[chrom] = (mol_to_graph(chrom), morgan_fp(chrom), global_desc(chrom))
        graph, fp, cdesc = gcache[chrom]
        ids = encode_pair(chrom, solv, vocab)
        max_len = max(max_len, len(ids))
        tgt = np.array([r[k] for k in TARGET_NAMES], dtype=np.float32)
        mask = (~np.isnan(tgt)).astype(np.float32)
        samples.append({
            'chromophore': chrom, 'solvent': solv, 'split': r['split'],
            'graph': graph, 'morgan': fp, 'chrom_desc': cdesc.astype(np.float32),
            'solv_desc': global_desc(solv).astype(np.float32),
            'token_ids': ids, 'targets': tgt, 'mask': mask,
        })

    meta = {
        'target_names': TARGET_NAMES, 'target_mean': tgt_mean, 'target_std': tgt_std,
        'chrom_desc_mean': cd_mean.tolist(), 'chrom_desc_std': cd_std.tolist(),
        'solv_desc_mean': sd_mean.tolist(), 'solv_desc_std': sd_std.tolist(),
        'desc_names': [n for n, _ in DESC_FUNCS], 'morgan_bits': MORGAN_BITS,
        'atom_feat_dim': 17, 'bond_feat_dim': 7, 'vocab': vocab,
        'vocab_size': len(vocab), 'max_seq_len': max_len, 'pad_id': vocab['[PAD]'],
    }
    with open(os.path.join(PROC, 'features.pkl'), 'wb') as f:
        pickle.dump(samples, f)
    with open(os.path.join(PROC, 'meta.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Özet
    g0 = samples[0]['graph']
    print("Örnek sayısı        :", len(samples))
    print("Atom feat dim        :", g0['x'].shape[1], "| Bağ feat dim:", samples[0]['graph']['edge_attr'].shape[1] if g0['edge_attr'].size else 7)
    print("Morgan + desc (MLP)  :", MORGAN_BITS, "+", DESC_DIM, "| çözücü desc:", DESC_DIM)
    print("Vocab boyutu         :", len(vocab), "| max seq len:", max_len)
    print("Hedef ort (train)    :", {k: round(v, 2) for k, v in tgt_mean.items()})
    print("Hedef std (train)    :", {k: round(v, 2) for k, v in tgt_std.items()})
    print("Kayıt: data/processed/features.pkl + meta.json")

if __name__ == '__main__':
    main()
