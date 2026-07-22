"""interpret.py - Yorumlanabilirlik: in-silico substitüent probu (5-seed ensemble).
Çekirdek sabit, meso substitüenti sistematik değiştirilir; modelin ΦF/λ tahminleri
bilinen BODIPY kimyasını yeniden üretiyor mu?
Test edilen hipotezler:
  H1 meso-alkil >> meso-aril (ΦF)
  H2 EDG (OMe, NMe2) > fenil > EWG (CN, NO2)
  H3 ağır atom serisi F > Cl > Br > I (ΦF azalır)
Çıktı: results/probe_predictions.csv, probe_summary.csv, probe_QY.png, probe_heavy_atom.png
Çalıştır: python src/interpret.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

from data_utils import ROOT
from data_loader import (MLPDataset, SeqDataset, _mlp_collate, _seq_collate,
                         _to_data, _tgt_stats, load_features, destandardize)
from model import MLP, CNN1D, SMILESTransformer, GATv2Net
from train import predict, fwd_mlp, fwd_seq, fwd_graph, DEVICE, RESULTS
from external_test import featurize_external
from feature_extraction import TARGET_NAMES

SEEDS = [0, 1, 2, 3, 4]
SOLVENT = 'ClC(Cl)Cl'          # CHCl3 (Derin 2018 ile aynı)
QY_IDX = TARGET_NAMES.index('QY')

def core(meso, r35):
    """meso-substitue 3,5-dialkil BODIPY (Derin 2018 ile aynı iskelet)."""
    return f"C({meso})(=C2C=CC({r35})=[N+]2[B-]3(F)F)C4=CC=C({r35})N34"

# (etiket, meso SMILES, sınıf)  — sınıf: alkyl / EDG / neutral / EWG / halogen
MESO = [
    ('methyl',        'C',                  'alkyl'),
    ('ethyl',         'CC',                 'alkyl'),
    ('n-propyl',      'CCC',                'alkyl'),
    ('phenyl',        'c1ccccc1',           'neutral'),
    ('4-NMe2-phenyl', 'c1ccc(N(C)C)cc1',    'EDG'),
    ('4-OMe-phenyl',  'c1ccc(OC)cc1',       'EDG'),
    ('4-Me-phenyl',   'c1ccc(C)cc1',        'EDG'),
    ('4-F-phenyl',    'c1ccc(F)cc1',        'halogen'),
    ('4-Cl-phenyl',   'c1ccc(Cl)cc1',       'halogen'),
    ('4-Br-phenyl',   'c1ccc(Br)cc1',       'halogen'),
    ('4-I-phenyl',    'c1ccc(I)cc1',        'halogen'),
    ('4-CN-phenyl',   'c1ccc(C#N)cc1',      'EWG'),
    ('4-CF3-phenyl',  'c1ccc(C(F)(F)F)cc1', 'EWG'),
    ('4-NO2-phenyl',  'c1ccc([N+](=O)[O-])cc1', 'EWG'),
]
R35 = [('Me', 'C'), ('Et', 'CC')]

def build_probe_df():
    rows = []
    for lab, meso, cls in MESO:
        for r35lab, r35 in R35:
            rows.append({'label': lab, 'meso_class': cls, 'r35': r35lab,
                         'Chromophore': core(meso, r35), 'Solvent': SOLVENT,
                         **{t: np.nan for t in TARGET_NAMES}})
    return pd.DataFrame(rows)

def make_loaders(df, meta):
    samples = featurize_external(df, meta['vocab'])
    sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
    sds = np.array(meta['solv_desc_std'], dtype=np.float32)
    tm, ts = _tgt_stats(meta)
    n = len(df)
    return {
        'mlp': DataLoader(MLPDataset(samples, meta), batch_size=n, shuffle=False, collate_fn=_mlp_collate),
        'seq': DataLoader(SeqDataset(samples, meta), batch_size=n, shuffle=False, collate_fn=_seq_collate),
        'graph': GeoLoader([_to_data(s, sdm, sds, tm, ts) for s in samples], batch_size=n, shuffle=False),
    }, MLPDataset(samples, meta).X.shape[1]

def ensemble_predict(df, meta):
    """5 seed × 3 mimari; her mimari için seed-ortalaması ve std döndürür."""
    loaders, in_dim = make_loaders(df, meta)
    specs = [
        ('1D-CNN', 'seq', fwd_seq, lambda: CNN1D(meta['vocab_size'])),
        ('GATv2', 'graph', fwd_graph, lambda: GATv2Net()),
        ('MLP', 'mlp', fwd_mlp, lambda: MLP(in_dim)),
    ]
    out = {}
    for name, key, fwd, build in specs:
        per_seed = []
        for s in SEEDS:
            ck = os.path.join(RESULTS, f'ckpt_{name}_s{s}.pt')
            if not os.path.exists(ck):
                continue
            m = build(); m.load_state_dict(torch.load(ck, map_location=DEVICE)); m.to(DEVICE)
            ps, _, _ = predict(m, loaders[key], fwd)
            per_seed.append(destandardize(ps, meta))
        if per_seed:
            arr = np.stack(per_seed)                    # [seed, n, 4]
            out[name] = (arr.mean(0), arr.std(0), len(per_seed))
    return out

CLS_COLOR = {'alkyl': '#2ca02c', 'EDG': '#1f77b4', 'neutral': '#7f7f7f',
             'halogen': '#ff7f0e', 'EWG': '#d62728'}

def plot_qy(df, preds, fname='probe_QY.png'):
    models = list(preds)
    sub = df[df.r35 == 'Me'].reset_index()
    idx = sub['index'].values
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 3.4 * len(models)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        mu, sd, ns = preds[name]
        y = mu[idx, QY_IDX]; e = sd[idx, QY_IDX]
        colors = [CLS_COLOR[c] for c in sub['meso_class']]
        ax.bar(np.arange(len(sub)), y, yerr=e, capsize=3, color=colors)
        ax.set_ylabel('Predicted ΦF'); ax.set_title(f'{name} (mean ± s.d., {ns} seeds)')
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub['label'], rotation=35, ha='right')
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in CLS_COLOR.values()]
    axes[0].legend(handles, list(CLS_COLOR), ncol=5, fontsize=8)
    fig.suptitle('In-silico meso-substituent probe — predicted ΦF (3,5-dimethyl BODIPY, CHCl₃)')
    fig.tight_layout()
    out = os.path.join(RESULTS, fname); fig.savefig(out, dpi=130); plt.close(fig)
    return out

def plot_heavy_atom(df, preds, fname='probe_heavy_atom.png'):
    order = ['4-F-phenyl', '4-Cl-phenyl', '4-Br-phenyl', '4-I-phenyl']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in preds:
        mu, sd, ns = preds[name]
        ys, es = [], []
        for lab in order:
            i = df.index[(df.label == lab) & (df.r35 == 'Me')][0]
            ys.append(mu[i, QY_IDX]); es.append(sd[i, QY_IDX])
        ax.errorbar(range(len(order)), ys, yerr=es, marker='o', capsize=4, label=name)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(['F', 'Cl', 'Br', 'I'])
    ax.set_xlabel('para-halogen on meso-phenyl'); ax.set_ylabel('Predicted ΦF')
    ax.set_title('Heavy-atom effect probe (predicted, 5-seed ensemble)'); ax.legend()
    fig.tight_layout(); out = os.path.join(RESULTS, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    return out

def hypothesis_tests(df, preds):
    """H1 alkil>aril, H2 EDG>fenil>EWG, H3 F>Cl>Br>I (ΦF)."""
    lines = []
    for name in preds:
        mu, _, _ = preds[name]
        d = df.copy(); d['QY_pred'] = mu[:, QY_IDX]
        g = d.groupby('meso_class')['QY_pred'].mean()
        alkyl = g.get('alkyl', np.nan); neutral = g.get('neutral', np.nan)
        edg = g.get('EDG', np.nan); ewg = g.get('EWG', np.nan)
        hv = [d.loc[(d.label == f'4-{x}-phenyl') & (d.r35 == 'Me'), 'QY_pred'].values[0]
              for x in ['F', 'Cl', 'Br', 'I']]
        h1 = alkyl > max(neutral, edg, ewg)
        h2 = (edg > neutral) and (neutral > ewg)
        h3 = all(hv[i] >= hv[i + 1] for i in range(3))
        lines.append({'model': name,
                      'alkyl': round(alkyl, 3), 'EDG': round(edg, 3),
                      'neutral': round(neutral, 3), 'EWG': round(ewg, 3),
                      'F': round(hv[0], 3), 'Cl': round(hv[1], 3),
                      'Br': round(hv[2], 3), 'I': round(hv[3], 3),
                      'H1_alkyl>aryl': bool(h1), 'H2_EDG>Ph>EWG': bool(h2),
                      'H3_F>Cl>Br>I': bool(h3)})
    return pd.DataFrame(lines)

if __name__ == '__main__':
    _, meta = load_features()
    df = build_probe_df()
    print(f"Prob seti: {len(df)} yapı ({len(MESO)} meso × {len(R35)} 3,5-alkil)")
    preds = ensemble_predict(df, meta)
    print("Yüklenen modeller:", {k: v[2] for k, v in preds.items()})

    rec = df[['label', 'meso_class', 'r35', 'Chromophore']].copy()
    for name, (mu, sd, _) in preds.items():
        for j, t in enumerate(TARGET_NAMES):
            rec[f'{name}_{t}'] = mu[:, j].round(3)
            rec[f'{name}_{t}_sd'] = sd[:, j].round(3)
    rec.to_csv(os.path.join(RESULTS, 'probe_predictions.csv'), index=False)

    tests = hypothesis_tests(df, preds)
    tests.to_csv(os.path.join(RESULTS, 'probe_summary.csv'), index=False)
    print("\n=== HİPOTEZ TESTLERİ (ortalama tahmin ΦF) ===")
    print(tests.to_string(index=False))
    print("\nFigürler:", plot_qy(df, preds), plot_heavy_atom(df, preds))
