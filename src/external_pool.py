"""external_pool.py - BUYUK harici dogrulama seti (hakem M1'e nihai cevap).
12 bagimsiz yayindaki TUM BODIPY'ler + Derin 2018'in 8 bilesigi tek bir harici sette
havuzlanir; model bunlarin HICBIRINI gormeden egitilir.
Avantaj: (i) ~200+ molekul -> istatistiksel guc, (ii) genis spektral aralik -> R2 anlamli
(LOPO'daki katman-ici negatif R2 artefakti kalkar), (iii) yayin bazinda kirilim da verilir.
Cikti: results/extpool_summary.csv, extpool_perpub.csv, extpool_parity.png, extpool_log
Calistir: python src/external_pool.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader

from data_utils import ROOT, SMILES_COL
from data_loader import (MLPDataset, SeqDataset, _mlp_collate, _seq_collate,
                         _to_data, destandardize)
from model import MLP, CNN1D, SMILESTransformer, GATv2Net
from train import fit, predict, fwd_mlp, fwd_seq, fwd_graph, DEVICE, DISPLAY, RESULTS
from feature_extraction import TARGET_NAMES
from external_test import featurize_external
from lopo import load_with_refs, fold_meta, build_loaders, MIN_MOL, MIN_QY, EXCLUDE, N_FOLDS

SEEDS = [0, 1, 2, 3, 4]
EXT_CSV = os.path.join(ROOT, 'data', 'external', 'derin2018.csv')

def metrics(pred, y, mask, tag, model, seed, extra=None):
    rows = []
    for j, t in enumerate(TARGET_NAMES):
        mk = mask[:, j] > 0
        if mk.sum() < 3:
            continue
        yt, yp = y[mk, j], pred[mk, j]
        ss_res = float(np.sum((yt - yp) ** 2)); ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r = {'set': tag, 'model': model, 'seed': seed, 'target': DISPLAY.get(t, t),
             'n': int(mk.sum()),
             'MAE': round(float(np.mean(np.abs(yt - yp))), 4),
             'RMSE': round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
             'R2': round(1 - ss_res / ss_tot, 4) if ss_tot > 1e-9 else np.nan,
             'rho': round(float(spearmanr(yt, yp).statistic), 4) if len(yt) > 2 else np.nan}
        if extra:
            r.update(extra)
        rows.append(r)
    return rows

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    samples, base_meta, df = load_with_refs()

    g = df.groupby('Reference').agg(mol=(SMILES_COL, 'nunique'),
                                    qy=('QY', 'count')).sort_values('mol', ascending=False)
    pubs = [r for r, row in g.iterrows()
            if row['mol'] >= MIN_MOL and row['qy'] >= MIN_QY and r not in EXCLUDE][:N_FOLDS]
    ext_mols = set(df.loc[df['Reference'].isin(pubs), SMILES_COL])

    te = [s for s in samples if s['chromophore'] in ext_mols]
    pool = [s for s in samples if s['chromophore'] not in ext_mols]
    print(f"Harici havuz: {len(pubs)} yayin | {len(ext_mols)} molekul | {len(te)} kayit")
    print(f"Egitim havuzu: {len({s['chromophore'] for s in pool})} molekul | {len(pool)} kayit")

    # Derin 2018 (veri tabani disi) ayri set
    dfd = pd.read_csv(EXT_CSV)
    derin = featurize_external(dfd, base_meta['vocab'])

    rows, perpub, preds_store = [], [], {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mols = sorted({s['chromophore'] for s in pool}); rng.shuffle(mols)
        val_mols = set(mols[:max(int(0.12 * len(mols)), 1)])
        tr = [s for s in pool if s['chromophore'] not in val_mols]
        va = [s for s in pool if s['chromophore'] in val_mols]
        meta = fold_meta(base_meta, tr)
        L = build_loaders(tr, va, te, meta)
        sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
        sds = np.array(meta['solv_desc_std'], dtype=np.float32)
        tm = np.array([meta['target_mean'][k] for k in meta['target_names']], dtype=np.float32)
        ts = np.array([meta['target_std'][k] for k in meta['target_names']], dtype=np.float32)
        dl = {'mlp': DataLoader(MLPDataset(derin, meta), batch_size=8, shuffle=False, collate_fn=_mlp_collate),
              'seq': DataLoader(SeqDataset(derin, meta), batch_size=8, shuffle=False, collate_fn=_seq_collate),
              'graph': GeoLoader([_to_data(s, sdm, sds, tm, ts) for s in derin], batch_size=8, shuffle=False)}

        cfg = [('MLP', 'mlp', fwd_mlp, lambda: MLP(L['in_dim'])),
               ('1D-CNN', 'seq', fwd_seq, lambda: CNN1D(meta['vocab_size'])),
               ('Transformer', 'seq', fwd_seq, lambda: SMILESTransformer(meta['vocab_size'])),
               ('GATv2', 'graph', fwd_graph, lambda: GATv2Net())]
        for name, key, fwd, build in cfg:
            torch.manual_seed(seed); np.random.seed(seed)
            print(f"\n--- seed {seed} | {name} ---")
            m = fit(build(), L[key], fwd, name, patience=25)
            ps, ys, ms = predict(m, L[key]['test'], fwd)
            P, Y = destandardize(ps, meta), destandardize(ys, meta)
            rows += metrics(P, Y, ms, 'pooled_external', name, seed)
            preds_store.setdefault(name, []).append((P, Y, ms))
            # yayin bazinda kirilim
            refs = np.array([s['ref'] for s in te])
            for pub in pubs:
                sel = refs == pub
                if sel.sum() >= 3:
                    perpub += metrics(P[sel], Y[sel], ms[sel], 'per_pub', name, seed,
                                      extra={'publication': pub})
            # Derin 2018
            dp, dy, dm = predict(m, dl[key], fwd)
            rows += metrics(destandardize(dp, meta), destandardize(dy, meta), dm,
                            'derin2018', name, seed)
    return pd.DataFrame(rows), pd.DataFrame(perpub), preds_store, base_meta

if __name__ == '__main__':
    raw, perpub, preds, meta = main()
    raw.to_csv(os.path.join(RESULTS, 'extpool_raw.csv'), index=False)
    perpub.to_csv(os.path.join(RESULTS, 'extpool_perpub.csv'), index=False)

    summ = raw.groupby(['set', 'model', 'target'])[['MAE', 'R2', 'rho']].agg(['mean', 'std'])
    out = pd.DataFrame(index=summ.index)
    for met in ['MAE', 'R2', 'rho']:
        out[met] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(summ[(met, 'mean')], summ[(met, 'std')])]
    out = out.reset_index()
    out.to_csv(os.path.join(RESULTS, 'extpool_summary.csv'), index=False)

    for tag in ['pooled_external', 'derin2018']:
        s = out[out['set'] == tag]
        print(f"\n=== {tag.upper()} — MAE (5 seed ort ± s.s.) ===")
        print(s.pivot(index='model', columns='target', values='MAE').to_string())
        print(f"--- R2 ---")
        print(s.pivot(index='model', columns='target', values='R2').to_string())
        print(f"--- Spearman rho ---")
        print(s.pivot(index='model', columns='target', values='rho').to_string())

    # Parity: havuzlanmis harici set (seed 0 tahminleri)
    targets = [DISPLAY.get(t, t) for t in TARGET_NAMES]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for ax, j in zip(axes, range(4)):
        for name in preds:
            P, Y, M = preds[name][0]
            mk = M[:, j] > 0
            ax.scatter(Y[mk, j], P[mk, j], s=11, alpha=.5, label=name)
        lim = ax.get_xlim()
        ax.plot(lim, lim, '--', color='gray', lw=1)
        ax.set_title(targets[j], fontsize=10)
        ax.set_xlabel('Measured'); ax.set_ylabel('Predicted')
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Pooled external validation: 12 held-out publications (~200 BODIPY dyes)')
    fig.tight_layout()
    p = os.path.join(RESULTS, 'extpool_parity.png'); fig.savefig(p, dpi=130); plt.close(fig)
    print('\nFigur:', p)
