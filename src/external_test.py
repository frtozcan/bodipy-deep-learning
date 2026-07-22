"""external_test.py - Derin et al. 2018'in 8 BODIPY'sinde 4 modeli harici doğrulama.
results/ckpt_*.pt yükler, ölçülen değerlerle kıyaslar, sızıntı (eğitimde var mı) kontrol eder.
Çıktı: results/external_predictions.csv, external_metrics.csv, external_parity.png
Çalıştır: python src/external_test.py
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from data_utils import ROOT, SMILES_COL, SOLVENT_COL
from feature_extraction import mol_to_graph, morgan_fp, global_desc, encode_pair, TARGET_NAMES
from data_loader import (MLPDataset, SeqDataset, _mlp_collate, _seq_collate,
                         _to_data, _tgt_stats, load_features, destandardize)
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoLoader
from model import MLP, CNN1D, SMILESTransformer, GATv2Net
from train import fwd_mlp, fwd_seq, fwd_graph, predict, DEVICE, DISPLAY, UNITS

RESULTS = os.path.join(ROOT, 'results')
EXT = os.path.join(ROOT, 'data', 'external', 'derin2018.csv')
COLORS = {'MLP': '#2ca02c', '1D-CNN': '#1f77b4', 'Transformer': '#d62728', 'GATv2': '#ff7f0e'}

def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s

def featurize_external(df, vocab):
    out = []
    for _, r in df.iterrows():
        chrom, solv = r[SMILES_COL], r[SOLVENT_COL]
        out.append({
            'chromophore': chrom, 'solvent': solv, 'split': 'ext',
            'graph': mol_to_graph(chrom), 'morgan': morgan_fp(chrom),
            'chrom_desc': global_desc(chrom).astype(np.float32),
            'solv_desc': global_desc(solv).astype(np.float32),
            'token_ids': encode_pair(chrom, solv, vocab),
            'targets': np.array([r[k] for k in TARGET_NAMES], dtype=np.float32),
            'mask': np.ones(len(TARGET_NAMES), dtype=np.float32),
        })
    return out

def main():
    train_samples, meta = load_features()
    vocab = meta['vocab']
    df = pd.read_csv(EXT)

    # --- Sızıntı kontrolü: bu bileşikler Deep4Chem'de var mı? ---
    train_canon = {canon(s['chromophore']) for s in train_samples if s['split'] == 'train'}
    all_canon = {canon(s['chromophore']) for s in train_samples}
    print("=== SIZINTI KONTROLÜ (Deep4Chem'de var mı?) ===")
    n_overlap = 0
    for _, r in df.iterrows():
        c = canon(r[SMILES_COL])
        where = 'TRAIN (sızıntı!)' if c in train_canon else ('val/test' if c in all_canon else 'yok')
        if c in all_canon:
            n_overlap += 1
        print(f"  {r['id']}: {where}")
    print(f"  -> {len(df) - n_overlap}/{len(df)} bileşik tamamen harici (Deep4Chem'de yok).")

    # --- Featurize + yükleyiciler ---
    ext = featurize_external(df, vocab)
    sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
    sds = np.array(meta['solv_desc_std'], dtype=np.float32)
    tm, ts = _tgt_stats(meta)
    mlp_loader = DataLoader(MLPDataset(ext, meta), batch_size=8, shuffle=False, collate_fn=_mlp_collate)
    seq_loader = DataLoader(SeqDataset(ext, meta), batch_size=8, shuffle=False, collate_fn=_seq_collate)
    graph_loader = GeoLoader([_to_data(s, sdm, sds, tm, ts) for s in ext], batch_size=8, shuffle=False)
    in_dim = MLPDataset(ext, meta).X.shape[1]

    specs = [
        ('MLP', mlp_loader, fwd_mlp, MLP(in_dim)),
        ('1D-CNN', seq_loader, fwd_seq, CNN1D(meta['vocab_size'])),
        ('Transformer', seq_loader, fwd_seq, SMILESTransformer(meta['vocab_size'])),
        ('GATv2', graph_loader, fwd_graph, GATv2Net()),
    ]
    preds = {}
    for name, loader, fwd, model in specs:
        model.load_state_dict(torch.load(os.path.join(RESULTS, f'ckpt_{name}.pt'), map_location=DEVICE))
        model.to(DEVICE)
        ps, _, _ = predict(model, loader, fwd)
        preds[name] = destandardize(ps, meta)        # [8,4] gerçek birimlerde

    measured = df[TARGET_NAMES].values.astype(float)  # [8,4]

    # --- Tablolar ---
    recs = []
    for i, row in df.iterrows():
        for j, t in enumerate(TARGET_NAMES):
            rec = {'id': row['id'], 'target': DISPLAY.get(t, t),
                   'measured': round(float(measured[i, j]), 3)}
            for name in preds:
                rec[name] = round(float(preds[name][i, j]), 3)
            recs.append(rec)
    pd.DataFrame(recs).to_csv(os.path.join(RESULTS, 'external_predictions.csv'), index=False)

    mrecs = []
    for name in preds:
        for j, t in enumerate(TARGET_NAMES):
            e = preds[name][:, j] - measured[:, j]
            mrecs.append({'model': name, 'target': DISPLAY.get(t, t),
                          'MAE': round(float(np.mean(np.abs(e))), 3),
                          'RMSE': round(float(np.sqrt(np.mean(e ** 2))), 3)})
    metr = pd.DataFrame(mrecs)
    metr.to_csv(os.path.join(RESULTS, 'external_metrics.csv'), index=False)

    # --- Sıra korelasyonu (Spearman): kimyasal trend yakalanmış mı? ---
    from scipy.stats import spearmanr
    srecs = []
    for name in preds:
        for j, t in enumerate(TARGET_NAMES):
            rho = spearmanr(measured[:, j], preds[name][:, j]).statistic
            srecs.append({'model': name, 'target': DISPLAY.get(t, t),
                          'Spearman_rho': round(float(rho), 3)})
    sp = pd.DataFrame(srecs)
    sp.to_csv(os.path.join(RESULTS, 'external_spearman.csv'), index=False)

    # --- Figür: ölçülen vs tahmin (8 bileşik, 4 model) ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, j in zip(axes.ravel(), range(len(TARGET_NAMES))):
        t = TARGET_NAMES[j]; u = UNITS.get(t, ''); uu = f' ({u})' if u else ''
        mvals = measured[:, j]; allv = [mvals]
        for name in preds:
            ax.scatter(mvals, preds[name][:, j], label=name, color=COLORS[name],
                       s=48, alpha=0.85, edgecolor='white')
            allv.append(preds[name][:, j])
        allv = np.concatenate(allv); lo, hi = float(allv.min()), float(allv.max())
        pad = (hi - lo) * 0.08 + 1e-6
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--', color='gray', lw=1)
        ax.set_title(DISPLAY.get(t, t)); ax.set_xlabel(f'Measured{uu}'); ax.set_ylabel(f'Predicted{uu}')
        ax.legend(fontsize=8)
    fig.suptitle('External validation — Derin et al. 2018 (8 BODIPY dyes, CHCl₃)', fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, 'external_parity.png')
    fig.savefig(out, dpi=130); plt.close(fig)

    # --- Konsol özeti ---
    print("\n=== HARİCİ TEST — MAE (8 bileşik) ===")
    piv = metr.pivot(index='model', columns='target', values='MAE')
    print(piv.to_string())
    print("\n=== SPEARMAN ρ (sıralama / kimyasal trend) ===")
    print(sp.pivot(index='model', columns='target', values='Spearman_rho').to_string())
    print("\nTablo: results/external_predictions.csv + external_metrics.csv + external_spearman.csv")
    print("Figür:", out)

if __name__ == '__main__':
    main()
