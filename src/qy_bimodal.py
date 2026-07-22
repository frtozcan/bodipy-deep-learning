"""qy_bimodal.py - Hakem M2: bimodal ΦF icin modelleme stratejileri karsilastirmasi.
Dort varyant, ayni MLP govdesi, tek gorev (ΦF):
  A direct     : standartlastirilmis ΦF uzerinde MSE (mevcut yaklasim)
  B bounded    : sigmoid cikis, ham [0,1] uzerinde MSE
  C logit      : logit(ΦF) uzerinde MSE, sigmoid ile geri cevrilir
  D two_stage  : simif basi P(isimali) + isimali altkumede regresyon
                 tahmin = p*mu + (1-p)*E[ΦF|karanlik]
Iki bolme: (i) ic test (bodipy.csv split), (ii) YAYIN HOLDOUT (havuzlanmis harici).
Cikti: results/qy_bimodal_raw.csv, qy_bimodal_summary.csv, qy_bimodal.png
Calistir: python src/qy_bimodal.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data_utils import ROOT, SMILES_COL
from data_loader import load_features
from train import DEVICE, RESULTS
from lopo import load_with_refs, MIN_MOL, MIN_QY, EXCLUDE, N_FOLDS
from feature_extraction import TARGET_NAMES

QY = TARGET_NAMES.index('QY')
THRESH = 0.05          # ΦF < 0.05 -> "karanlik" (%18.5)
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE = 300, 30

class QYNet(nn.Module):
    """Ortak govde; two_stage'de iki bas (sinif + regresyon)."""
    def __init__(self, in_dim, variant, hidden=(512, 256), p=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.variant = variant
        self.reg = nn.Linear(d, 1)
        self.cls = nn.Linear(d, 1) if variant == 'two_stage' else None

    def forward(self, x):
        h = self.trunk(x)
        r = self.reg(h).squeeze(-1)
        if self.variant == 'two_stage':
            return r, self.cls(h).squeeze(-1)
        return r, None

def build_xy(samples, meta_stats):
    cdm, cds, sdm, sds = meta_stats
    X, Y = [], []
    for s in samples:
        cd = (s['chrom_desc'] - cdm) / cds
        sd = (s['solv_desc'] - sdm) / sds
        X.append(np.concatenate([s['morgan'].astype(np.float32), cd, sd]))
        Y.append(s['targets'][QY])
    X = np.stack(X).astype(np.float32); Y = np.array(Y, dtype=np.float32)
    ok = ~np.isnan(Y)
    return X[ok], Y[ok]

def stats_from(samples):
    cd = np.stack([s['chrom_desc'] for s in samples])
    sd = np.stack([s['solv_desc'] for s in samples])
    return cd.mean(0), cd.std(0) + 1e-8, sd.mean(0), sd.std(0) + 1e-8

def make_target(y, variant, mu, sd):
    if variant == 'direct':
        return (y - mu) / sd
    if variant in ('bounded', 'two_stage'):
        return y
    if variant == 'logit':
        yc = np.clip(y, 1e-3, 1 - 1e-3)
        return np.log(yc / (1 - yc))
    raise ValueError(variant)

def invert(pred, variant, mu, sd):
    if variant == 'direct':
        return pred * sd + mu
    if variant == 'logit':
        return 1 / (1 + np.exp(-pred))
    return pred                              # bounded / two_stage zaten [0,1]

def run(Xtr, ytr, Xva, yva, Xte, variant, seed, dark_mean):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-8)
    net = QYNet(Xtr.shape[1], variant).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()

    def pack(X, y):
        t = torch.tensor(make_target(y, variant, mu, sd)).to(DEVICE)
        return torch.tensor(X).to(DEVICE), t, torch.tensor((y >= THRESH).astype(np.float32)).to(DEVICE)

    Xt, Tt, Ct = pack(Xtr, ytr); Xv, Tv, Cv = pack(Xva, yva)
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); opt.zero_grad()
        idx = torch.randperm(len(Xt), device=DEVICE)
        r, c = net(Xt[idx])
        if variant == 'two_stage':
            m = Ct[idx] > 0
            loss = bce(c, Ct[idx]) + (mse(torch.sigmoid(r[m]), Tt[idx][m]) if m.any() else 0)
        elif variant == 'bounded':
            loss = mse(torch.sigmoid(r), Tt[idx])
        else:
            loss = mse(r, Tt[idx])
        loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            rv, cv = net(Xv)
            if variant == 'two_stage':
                m = Cv > 0
                v = bce(cv, Cv).item() + (mse(torch.sigmoid(rv[m]), Tv[m]).item() if m.any() else 0)
            elif variant == 'bounded':
                v = mse(torch.sigmoid(rv), Tv).item()
            else:
                v = mse(rv, Tv).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.cpu().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        r, c = net(torch.tensor(Xte).to(DEVICE))
        r = r.cpu().numpy()
        if variant == 'two_stage':
            p = 1 / (1 + np.exp(-c.cpu().numpy()))
            mu_e = 1 / (1 + np.exp(-r))                     # sigmoid: E[ΦF|isimali]
            pred = p * mu_e + (1 - p) * dark_mean
            return np.clip(pred, 0, 1), p
        if variant == 'bounded':
            r = 1 / (1 + np.exp(-r))
    return np.clip(invert(r, variant, mu, sd), 0, 1), None

def score(y, pred, p_em, variant, split, seed):
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    row = {'split': split, 'variant': variant, 'seed': seed, 'n': len(y),
           'MAE': round(float(np.mean(np.abs(y - pred))), 4),
           'RMSE': round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
           'R2': round(1 - ss_res / ss_tot, 4) if ss_tot > 1e-9 else np.nan,
           'rho': round(float(spearmanr(y, pred).statistic), 4)}
    lab = (y >= THRESH).astype(int)
    if p_em is not None and 0 < lab.sum() < len(lab):
        row['dark_AUC'] = round(float(roc_auc_score(lab, p_em)), 4)
    return row

def main():
    samples, base_meta, df = load_with_refs()
    # (i) ic bolme
    splits = {}
    splits['internal'] = ([s for s in samples if s['split'] == 'train'],
                          [s for s in samples if s['split'] == 'val'],
                          [s for s in samples if s['split'] == 'test'])
    # (ii) yayin holdout (havuzlanmis harici)
    g = df.groupby('Reference').agg(mol=(SMILES_COL, 'nunique'), qy=('QY', 'count')) \
          .sort_values('mol', ascending=False)
    pubs = [r for r, row in g.iterrows()
            if row['mol'] >= MIN_MOL and row['qy'] >= MIN_QY and r not in EXCLUDE][:N_FOLDS]
    ext_mols = set(df.loc[df['Reference'].isin(pubs), SMILES_COL])
    te_p = [s for s in samples if s['chromophore'] in ext_mols]
    pool = [s for s in samples if s['chromophore'] not in ext_mols]

    rows = []
    for split_name in ['internal', 'pooled_external']:
        for seed in SEEDS:
            if split_name == 'internal':
                tr, va, te = splits['internal']
            else:
                rng = np.random.default_rng(seed)
                mols = sorted({s['chromophore'] for s in pool}); rng.shuffle(mols)
                vm = set(mols[:max(int(0.12 * len(mols)), 1)])
                tr = [s for s in pool if s['chromophore'] not in vm]
                va = [s for s in pool if s['chromophore'] in vm]
                te = te_p
            st = stats_from(tr)
            Xtr, ytr = build_xy(tr, st); Xva, yva = build_xy(va, st); Xte, yte = build_xy(te, st)
            dark_mean = float(ytr[ytr < THRESH].mean()) if (ytr < THRESH).any() else 0.0
            for variant in ['direct', 'bounded', 'logit', 'two_stage']:
                pred, p_em = run(Xtr, ytr, Xva, yva, Xte, variant, seed, dark_mean)
                rows.append(score(yte, pred, p_em, variant, split_name, seed))
            print(f"{split_name} seed{seed}: train {len(ytr)} / test {len(yte)} "
                  f"(karanlik %{100*(ytr<THRESH).mean():.1f})")
    return pd.DataFrame(rows)

if __name__ == '__main__':
    raw = main()
    raw.to_csv(os.path.join(RESULTS, 'qy_bimodal_raw.csv'), index=False)
    cols = [c for c in ['MAE', 'RMSE', 'R2', 'rho', 'dark_AUC'] if c in raw.columns]
    g = raw.groupby(['split', 'variant'])[cols].agg(['mean', 'std'])
    out = pd.DataFrame(index=g.index)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" if a == a else "-"
                  for a, b in zip(g[(c, 'mean')], g[(c, 'std')])]
    out = out.reset_index()
    out.to_csv(os.path.join(RESULTS, 'qy_bimodal_summary.csv'), index=False)
    order = ['direct', 'bounded', 'logit', 'two_stage']
    for s in ['internal', 'pooled_external']:
        sub = out[out['split'] == s].set_index('variant').reindex(order)
        print(f"\n=== {s.upper()} (5 seed ort ± s.s.) ===")
        print(sub.drop(columns=['split']).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, met, ttl in zip(axes, ['MAE', 'R2', 'rho'],
                            ['MAE (lower better)', 'R²', 'Spearman ρ']):
        w, x = 0.35, np.arange(len(order))
        for i, s in enumerate(['internal', 'pooled_external']):
            m = [g.loc[(s, v), (met, 'mean')] for v in order]
            e = [g.loc[(s, v), (met, 'std')] for v in order]
            ax.bar(x + i * w, m, w, yerr=e, capsize=3,
                   label='internal test' if i == 0 else 'pooled external')
        ax.set_xticks(x + w / 2); ax.set_xticklabels(order, rotation=15)
        ax.set_title(ttl); ax.axhline(0, color='gray', lw=.8)
        if met == 'MAE':
            ax.legend(fontsize=8)
    fig.suptitle('Modelling the bimodal quantum-yield distribution (MLP, 5 seeds)')
    fig.tight_layout()
    p = os.path.join(RESULTS, 'qy_bimodal.png'); fig.savefig(p, dpi=130); plt.close(fig)
    print('\nFigur:', p)
