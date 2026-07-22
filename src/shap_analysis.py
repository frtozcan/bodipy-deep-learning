"""shap_analysis.py - MLP için SHAP (ΦF hedefi), 5-seed ensemble.
Girdi 2068 = Morgan(2048) + kromofor descriptor(10) + çözücü descriptor(10).
Önemli Morgan bitleri RDKit bitInfo ile substruktüre eşlenir.
Çıktı: results/shap_top_features.csv, shap_bits.csv, shap_summary.png
Çalıştır: python src/shap_analysis.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

from data_utils import ROOT
from data_loader import load_features, MLPDataset, _split
from model import MLP
from train import DEVICE, RESULTS
from feature_extraction import DESC_FUNCS, MORGAN_BITS, TARGET_NAMES

SEEDS = [0, 1, 2, 3, 4]
QY = TARGET_NAMES.index('QY')
DESC_NAMES = [n for n, _ in DESC_FUNCS]

def feature_names():
    return ([f'MorganBit_{i}' for i in range(MORGAN_BITS)]
            + [f'chrom:{n}' for n in DESC_NAMES]
            + [f'solvent:{n}' for n in DESC_NAMES])

def bit_to_substructure(samples, top_bits):
    """Önemli Morgan bitlerini örnek substruktür SMILES'ine eşle."""
    out = {b: None for b in top_bits}
    want = set(top_bits)
    for s in samples:
        if not want:
            break
        mol = Chem.MolFromSmiles(s['chromophore'])
        if mol is None:
            continue
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS, bitInfo=info)
        for b in list(want):
            if b in info:
                atom, rad = info[b][0]
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom) if rad > 0 else []
                amap = {}
                sub = Chem.PathToSubmol(mol, env, atomMap=amap) if rad > 0 else None
                frag = Chem.MolToSmiles(sub) if sub and sub.GetNumAtoms() else \
                    mol.GetAtomWithIdx(atom).GetSymbol()
                out[b] = frag
                want.discard(b)
    return out

def main():
    samples, meta = load_features()
    tr = _split(samples, 'train'); te = _split(samples, 'test')
    ds_tr = MLPDataset(tr, meta); ds_te = MLPDataset(te, meta)
    in_dim = ds_tr.X.shape[1]
    rng = np.random.default_rng(0)
    bg = torch.tensor(ds_tr.X[rng.choice(len(ds_tr.X), 200, replace=False)]).to(DEVICE)
    X = torch.tensor(ds_te.X).to(DEVICE)

    per_seed = []
    for s in SEEDS:
        ck = os.path.join(RESULTS, f'ckpt_MLP_s{s}.pt')
        if not os.path.exists(ck):
            continue
        m = MLP(in_dim); m.load_state_dict(torch.load(ck, map_location=DEVICE))
        m.to(DEVICE).eval()
        ex = shap.GradientExplainer(m, bg)
        sv = ex.shap_values(X)                      # list veya [n,d,out]
        sv = np.array(sv[QY]) if isinstance(sv, list) else np.array(sv)[..., QY]
        per_seed.append(np.abs(sv).mean(axis=0))    # ortalama |SHAP| (global önem)
        print(f"  seed {s} tamam")
    return np.mean(per_seed, axis=0), np.std(per_seed, axis=0), samples, meta

if __name__ == '__main__':
    imp, sd, samples, meta = main()
    names = feature_names()
    df = pd.DataFrame({'feature': names, 'mean_abs_shap': imp, 'sd': sd})
    df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    df.head(40).round(5).to_csv(os.path.join(RESULTS, 'shap_top_features.csv'), index=False)

    # Descriptor'ların (adlandırılmış) sıralaması
    desc = df[df.feature.str.contains(':')].head(20)
    print("\n=== EN ÖNEMLİ ADLANDIRILMIŞ DESCRIPTOR'LAR (ΦF) ===")
    print(desc.round(4).to_string(index=False))

    # Morgan bitlerini substruktüre eşle
    top_bits = [int(f.split('_')[1]) for f in df.feature.head(60) if f.startswith('MorganBit_')][:15]
    mapping = bit_to_substructure(samples, top_bits)
    bits = pd.DataFrame([{'bit': b, 'substructure': mapping.get(b),
                          'mean_abs_shap': float(df.loc[df.feature == f'MorganBit_{b}', 'mean_abs_shap'].iloc[0])}
                         for b in top_bits]).sort_values('mean_abs_shap', ascending=False)
    bits.round(5).to_csv(os.path.join(RESULTS, 'shap_bits.csv'), index=False)
    print("\n=== EN ÖNEMLİ MORGAN BİTLERİ → SUBSTRUKTÜR ===")
    print(bits.to_string(index=False))

    # Figür: top-20 özellik
    top = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.feature, top.mean_abs_shap, xerr=top.sd, color='#2c6e8f', capsize=2)
    ax.set_xlabel('mean |SHAP| (ΦF, 5-seed ensemble)')
    ax.set_title('MLP feature attribution for ΦF — top 20')
    fig.tight_layout()
    out = os.path.join(RESULTS, 'shap_summary.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print("\nFigür:", out)
