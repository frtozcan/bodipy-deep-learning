"""multiseed.py - 4 mimariyi N seed ile eğit; iç test + harici test için mean±std üret.
Veri bölmesi SABİT (bodipy.csv, split seed=42); değişen: model init + eğitim stokastikliği.
Çıktı: results/multiseed_{internal,external}_{raw,summary}.csv,
       results/multiseed_R2.png, results/multiseed_external_MAE.png
Çalıştır: python src/multiseed.py
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
from data_loader import (make_mlp_loaders, make_seq_loaders, make_graph_loaders,
                         destandardize, MLPDataset, SeqDataset, _mlp_collate,
                         _seq_collate, _to_data, _tgt_stats, load_features)
from model import MLP, CNN1D, SMILESTransformer, GATv2Net
from train import (fit, predict, evaluate, fwd_mlp, fwd_seq, fwd_graph,
                   DEVICE, DISPLAY, RESULTS)
from external_test import featurize_external
from feature_extraction import TARGET_NAMES

SEEDS = [0, 1, 2, 3, 4]
EXT_CSV = os.path.join(ROOT, 'data', 'external', 'derin2018.csv')

def build_external_loaders(meta):
    df = pd.read_csv(EXT_CSV)
    ext = featurize_external(df, meta['vocab'])
    sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
    sds = np.array(meta['solv_desc_std'], dtype=np.float32)
    tm, ts = _tgt_stats(meta)
    return {
        'mlp': DataLoader(MLPDataset(ext, meta), batch_size=8, shuffle=False, collate_fn=_mlp_collate),
        'seq': DataLoader(SeqDataset(ext, meta), batch_size=8, shuffle=False, collate_fn=_seq_collate),
        'graph': GeoLoader([_to_data(s, sdm, sds, tm, ts) for s in ext], batch_size=8, shuffle=False),
    }, df[TARGET_NAMES].values.astype(float)

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Device:", DEVICE, "| seeds:", SEEDS)
    _, meta = load_features()
    ext_loaders, measured = build_external_loaders(meta)

    int_rows, ext_rows = [], []
    for seed in SEEDS:
        print(f"\n{'='*20} SEED {seed} {'='*20}")
        torch.manual_seed(seed); np.random.seed(seed)
        ml, _, in_dim = make_mlp_loaders(256)
        sl, _ = make_seq_loaders(128)
        gl, _ = make_graph_loaders(128)
        configs = [
            ('MLP', ml, fwd_mlp, lambda: MLP(in_dim), 'mlp'),
            ('1D-CNN', sl, fwd_seq, lambda: CNN1D(meta['vocab_size']), 'seq'),
            ('Transformer', sl, fwd_seq, lambda: SMILESTransformer(meta['vocab_size']), 'seq'),
            ('GATv2', gl, fwd_graph, lambda: GATv2Net(), 'graph'),
        ]

        for name, loaders, fwd, build, ekey in configs:
            torch.manual_seed(seed)          # her mimari için aynı başlangıç tohumu
            print(f"--- seed {seed} | {name} ---")
            model = fit(build(), loaders, fwd, name)
            torch.save(model.state_dict(),
                       os.path.join(RESULTS, f'ckpt_{name}_s{seed}.pt'))

            # iç test
            ps, ys, ms = predict(model, loaders['test'], fwd)
            tbl, _, _ = evaluate(ps, ys, ms, meta)
            for _, r in tbl.iterrows():
                int_rows.append({'seed': seed, 'model': name,
                                 'target': DISPLAY.get(r['target'], r['target']),
                                 'R2': r['R2'], 'RMSE': r['RMSE'], 'MAE': r['MAE']})

            # harici test (Derin 2018)
            eps, _, _ = predict(model, ext_loaders[ekey], fwd)
            epred = destandardize(eps, meta)
            for j, t in enumerate(TARGET_NAMES):
                e = epred[:, j] - measured[:, j]
                ext_rows.append({'seed': seed, 'model': name, 'target': DISPLAY.get(t, t),
                                 'MAE': round(float(np.mean(np.abs(e))), 4),
                                 'RMSE': round(float(np.sqrt(np.mean(e**2))), 4),
                                 'Spearman_rho': round(float(spearmanr(measured[:, j], epred[:, j]).statistic), 4)})

    int_df = pd.DataFrame(int_rows); ext_df = pd.DataFrame(ext_rows)
    int_df.to_csv(os.path.join(RESULTS, 'multiseed_internal_raw.csv'), index=False)
    ext_df.to_csv(os.path.join(RESULTS, 'multiseed_external_raw.csv'), index=False)
    return int_df, ext_df, meta

def summarize(df, metrics, fname):
    g = df.groupby(['model', 'target'])[metrics].agg(['mean', 'std'])
    out = pd.DataFrame(index=g.index)
    for m in metrics:
        out[m] = [f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(g[(m, 'mean')], g[(m, 'std')])]
    out = out.reset_index()
    out.to_csv(os.path.join(RESULTS, fname), index=False)
    return out, g

def barplot(g, metric, ylabel, title, fname, order, ylim=None):
    mu = g[(metric, 'mean')].unstack(); sd = g[(metric, 'std')].unstack()
    cols = [c for c in order if c in mu.columns]
    mu, sd = mu[cols], sd[cols]
    models = list(mu.index); x = np.arange(len(cols)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, m in enumerate(models):
        ax.bar(x + i * w, mu.loc[m].values, w, yerr=sd.loc[m].values,
               capsize=3, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(cols)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout(); out = os.path.join(RESULTS, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    return out

if __name__ == '__main__':
    int_df, ext_df, meta = main()
    order = [DISPLAY.get(t, t) for t in TARGET_NAMES]
    isum, ig = summarize(int_df, ['R2', 'RMSE', 'MAE'], 'multiseed_internal_summary.csv')
    esum, eg = summarize(ext_df, ['MAE', 'RMSE', 'Spearman_rho'], 'multiseed_external_summary.csv')
    p1 = barplot(ig, 'R2', 'Test R² (mean ± s.d., 5 seeds)',
                 'Architecture comparison — internal test (BODIPY)', 'multiseed_R2.png',
                 order, ylim=(0, 1))
    p2 = barplot(eg, 'Spearman_rho', 'Spearman ρ (mean ± s.d., 5 seeds)',
                 'External validation — rank correlation (Derin et al. 2018)',
                 'multiseed_external_rho.png', order)
    print("\n=== INTERNAL TEST (mean ± s.d., 5 seeds) ===")
    print(isum.pivot(index='model', columns='target', values='R2').to_string())
    print("\n=== EXTERNAL (Derin 2018) — MAE ===")
    print(esum.pivot(index='model', columns='target', values='MAE').to_string())
    print("\n=== EXTERNAL — Spearman ρ ===")
    print(esum.pivot(index='model', columns='target', values='Spearman_rho').to_string())
    print("\nFigürler:", p1, p2)
