"""lopo.py - Leave-One-Publication-Out harici dogrulama.
Bir yayin tamamen disarida birakilir; o yayinin MOLEKULLERI (baska yayinlarda gecse bile)
egitimden cikarilir. Hedef/tanimlayici standartlastirmasi HER KATMAN icin yeniden hesaplanir.
Cikti: results/lopo_raw.csv, lopo_summary.csv, lopo_box.png
Calistir: python src/lopo.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

from data_utils import ROOT, SMILES_COL
from data_loader import (MLPDataset, SeqDataset, _mlp_collate, _seq_collate,
                         _to_data, load_features, destandardize)
from model import MLP, CNN1D, SMILESTransformer, GATv2Net
from train import fit, predict, fwd_mlp, fwd_seq, fwd_graph, DEVICE, DISPLAY, RESULTS
from feature_extraction import TARGET_NAMES

SEED = 0
MIN_MOL, MIN_QY = 5, 5
EXCLUDE = {'10.1039/c4cs00030g'}   # Chem Soc Rev DERLEMESI: tek bir calisma degil
N_FOLDS = 12
PROC = os.path.join(ROOT, 'data', 'processed')

def load_with_refs():
    """features.pkl ornekleri bodipy.csv satir sirasiyla ayni; Reference'i eslestir."""
    samples, meta = load_features()
    df = pd.read_csv(os.path.join(PROC, 'bodipy.csv'))
    assert len(df) == len(samples), 'satir sayisi uyusmuyor'
    bad = sum(1 for i, s in enumerate(samples) if s['chromophore'] != df[SMILES_COL].iloc[i])
    assert bad == 0, f'{bad} satirda SMILES uyusmuyor'
    for i, s in enumerate(samples):
        s['ref'] = df['Reference'].iloc[i]
    return samples, meta, df

def fold_meta(base, train_samples):
    """Standartlastirma istatistiklerini SADECE bu katmanin egitim setinden hesapla."""
    m = dict(base)
    T = np.stack([s['targets'] for s in train_samples])
    M = np.stack([s['mask'] for s in train_samples])
    mean, std = {}, {}
    for j, name in enumerate(base['target_names']):
        v = T[:, j][M[:, j] > 0]
        mean[name] = float(np.mean(v)) if len(v) else 0.0
        s = float(np.std(v)) if len(v) > 1 else 1.0
        std[name] = s if s > 1e-6 else 1.0
    m['target_mean'], m['target_std'] = mean, std
    cd = np.stack([s['chrom_desc'] for s in train_samples])
    sd = np.stack([s['solv_desc'] for s in train_samples])
    m['chrom_desc_mean'] = cd.mean(0).tolist()
    m['chrom_desc_std'] = (cd.std(0) + 1e-8).tolist()
    m['solv_desc_mean'] = sd.mean(0).tolist()
    m['solv_desc_std'] = (sd.std(0) + 1e-8).tolist()
    return m

def build_loaders(tr, va, te, meta):
    n = max(len(te), 1)
    sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
    sds = np.array(meta['solv_desc_std'], dtype=np.float32)
    tm = np.array([meta['target_mean'][k] for k in meta['target_names']], dtype=np.float32)
    ts = np.array([meta['target_std'][k] for k in meta['target_names']], dtype=np.float32)
    out = {}
    out['mlp'] = {k: DataLoader(MLPDataset(v, meta), batch_size=256 if k == 'train' else n,
                                shuffle=(k == 'train'), collate_fn=_mlp_collate)
                  for k, v in [('train', tr), ('val', va), ('test', te)]}
    out['seq'] = {k: DataLoader(SeqDataset(v, meta), batch_size=128 if k == 'train' else n,
                                shuffle=(k == 'train'), collate_fn=_seq_collate)
                  for k, v in [('train', tr), ('val', va), ('test', te)]}
    out['graph'] = {k: GeoLoader([_to_data(s, sdm, sds, tm, ts) for s in v],
                                 batch_size=128 if k == 'train' else n, shuffle=(k == 'train'))
                    for k, v in [('train', tr), ('val', va), ('test', te)]}
    out['in_dim'] = MLPDataset(tr, meta).X.shape[1]
    return out

def score(pred_std, y_std, mask, meta, fold, model, n_mol):
    pred = destandardize(pred_std, meta); y = destandardize(y_std, meta)
    rows = []
    for j, t in enumerate(TARGET_NAMES):
        mk = mask[:, j] > 0
        if mk.sum() < 3:
            continue
        yt, yp = y[mk, j], pred[mk, j]
        ss_res = float(np.sum((yt - yp) ** 2)); ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        rho = spearmanr(yt, yp).statistic if len(yt) > 2 else np.nan
        rows.append({'fold': fold, 'model': model, 'n_mol': n_mol,
                     'target': DISPLAY.get(t, t), 'n': int(mk.sum()),
                     'MAE': round(float(np.mean(np.abs(yt - yp))), 4),
                     'RMSE': round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
                     'R2': round(1 - ss_res / ss_tot, 4) if ss_tot > 1e-9 else np.nan,
                     'rho': round(float(rho), 4) if rho == rho else np.nan})
    return rows

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    samples, base_meta, df = load_with_refs()

    # Aday yayinlar: >=5 molekul ve >=5 QY, derleme haric
    g = df.groupby('Reference').agg(mol=(SMILES_COL, 'nunique'),
                                    qy=('QY', 'count')).sort_values('mol', ascending=False)
    cand = [r for r, row in g.iterrows()
            if row['mol'] >= MIN_MOL and row['qy'] >= MIN_QY and r not in EXCLUDE][:N_FOLDS]
    print(f"Aday yayin: {len(cand)} | cihaz: {DEVICE}")

    # Sizinti kontrolu: yayinlar arasi molekul ortusmesi
    ref2mols = {r: set(df.loc[df['Reference'] == r, SMILES_COL]) for r in cand}
    overlap = sum(len(ref2mols[a] & ref2mols[b]) for i, a in enumerate(cand) for b in cand[i+1:])
    print(f"Adaylar arasi ortusen molekul cifti: {overlap}")

    rows = []
    for fi, ref in enumerate(cand, 1):
        test_mols = set(df.loc[df['Reference'] == ref, SMILES_COL])
        te = [s for s in samples if s['ref'] == ref]
        # SIZINTI ONLEME: test molekullerinin TUM kayitlari egitimden cikar
        pool = [s for s in samples if s['chromophore'] not in test_mols]
        pool_mols = sorted({s['chromophore'] for s in pool})
        rng = np.random.default_rng(SEED); rng.shuffle(pool_mols)
        n_val = max(int(0.12 * len(pool_mols)), 1)
        val_mols = set(pool_mols[:n_val])
        tr = [s for s in pool if s['chromophore'] not in val_mols]
        va = [s for s in pool if s['chromophore'] in val_mols]
        meta = fold_meta(base_meta, tr)
        n_mol = len(test_mols)
        print(f"\n[{fi}/{len(cand)}] {ref} | test {len(te)} kayit / {n_mol} mol "
              f"| train {len(tr)} | val {len(va)}")
        L = build_loaders(tr, va, te, meta)

        cfg = [('MLP', 'mlp', fwd_mlp, lambda: MLP(L['in_dim'])),
               ('1D-CNN', 'seq', fwd_seq, lambda: CNN1D(meta['vocab_size'])),
               ('Transformer', 'seq', fwd_seq, lambda: SMILESTransformer(meta['vocab_size'])),
               ('GATv2', 'graph', fwd_graph, lambda: GATv2Net())]
        for name, key, fwd, build in cfg:
            torch.manual_seed(SEED); np.random.seed(SEED)
            m = fit(build(), L[key], fwd, name, patience=20)
            ps, ys, ms = predict(m, L[key]['test'], fwd)
            rows += score(ps, ys, ms, meta, ref, name, n_mol)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, 'lopo_raw.csv'), index=False)

    # Ozet: katmanlar arasi ortalama +- s.s.
    summ = raw.groupby(['model', 'target'])[['MAE', 'R2', 'rho']].agg(['mean', 'std', 'count'])
    out = pd.DataFrame(index=summ.index)
    for met in ['MAE', 'R2', 'rho']:
        out[met] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(summ[(met, 'mean')], summ[(met, 'std')])]
    out['n_fold'] = summ[('MAE', 'count')].values
    out = out.reset_index()
    out.to_csv(os.path.join(RESULTS, 'lopo_summary.csv'), index=False)

    # Kutu grafigi: hedef basina MAE dagilimi (katmanlar uzerinden)
    targets = [DISPLAY.get(t, t) for t in TARGET_NAMES]
    models = sorted(raw['model'].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.1 * len(targets), 4.6))
    for ax, t in zip(np.atleast_1d(axes), targets):
        data = [raw[(raw.model == m) & (raw.target == t)]['MAE'].dropna().values for m in models]
        ax.boxplot(data, tick_labels=models, showfliers=False)
        ax.set_title(t, fontsize=10); ax.set_ylabel('MAE per publication')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('Leave-one-publication-out validation (MAE across held-out studies)')
    fig.tight_layout()
    p = os.path.join(RESULTS, 'lopo_box.png'); fig.savefig(p, dpi=130); plt.close(fig)

    print("\n=== LOPO OZET (katmanlar arasi ort ± s.s.) ===")
    print(out.pivot(index='model', columns='target', values='MAE').to_string())
    print("\n=== Spearman rho ===")
    print(out.pivot(index='model', columns='target', values='rho').to_string())
    print("\nFigur:", p)

if __name__ == '__main__':
    main()
