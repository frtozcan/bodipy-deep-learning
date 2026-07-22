"""solvent_ablation.py - Hakem M3: cozucu temsili ve entegrasyon mekanizmasi ablasyonu.
Varyantlar (ayni MLP govdesi, cok gorevli, maskeli MSE):
  none     : cozucu girdisi YOK          (cozucu gercekten gerekli mi?)
  generic  : 10 RDKit tanimlayici, concat (MEVCUT yaklasim = temel)
  physical : 7 fiziksel tanimlayici (eps_r, n_D, ET30, eta, pi*, alpha, beta) + eksik
             gostergesi, concat
  film     : ayni fiziksel vektor, ama concat yerine FiLM kosullama
             (cozucu -> gamma,beta ile kromofor temsili olceklenir/kaydirilir)
Iki bolme: ic test + havuzlanmis harici (yayin holdout). 5 seed.
Cikti: results/solvent_ablation_{raw,summary}.csv, solvent_ablation.png
Calistir: python src/solvent_ablation.py
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

from data_utils import ROOT, SMILES_COL
from train import DEVICE, RESULTS, DISPLAY
from lopo import load_with_refs, MIN_MOL, MIN_QY, EXCLUDE, N_FOLDS
from feature_extraction import TARGET_NAMES
from solvent_props import raw_vector, FIELDS

SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE = 300, 30
VARIANTS = ['none', 'generic', 'physical', 'film']

class FiLMNet(nn.Module):
    """Kromofor govdesi + cozucu kosullamasi.
    concat=True  -> cozucu vektoru girdiye eklenir (generic/physical)
    concat=False -> FiLM: cozucu -> (gamma, beta), h <- gamma*h + beta
    """
    def __init__(self, chrom_dim, solv_dim, n_targets=4, hidden=(512, 256),
                 p=0.2, mode='concat'):
        super().__init__()
        self.mode = mode
        in_dim = chrom_dim + (solv_dim if mode == 'concat' else 0)
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        if mode == 'film':
            self.film = nn.Sequential(nn.Linear(solv_dim, 64), nn.ReLU(), nn.Linear(64, 2 * d))
        self.head = nn.Linear(d, n_targets)

    def forward(self, xc, xs):
        if self.mode == 'concat':
            h = self.trunk(torch.cat([xc, xs], dim=1) if xs is not None else xc)
        else:
            h = self.trunk(xc)
            g, b = self.film(xs).chunk(2, dim=1)
            h = (1 + g) * h + b               # 1+gamma: kimlik baslangici
        return self.head(h)

def masked_mse(pred, y, m):
    return (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)

def solvent_matrix(samples, variant):
    if variant == 'none':
        return None
    if variant == 'generic':
        return np.stack([s['solv_desc'] for s in samples]).astype(np.float32)
    X = np.stack([raw_vector(s['solvent']) for s in samples]).astype(np.float32)
    miss = np.isnan(X).any(axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([X, miss], axis=1)      # NaN'lar sonra doldurulur

def prep(tr, va, te, variant):
    """Kromofor girdisi (Morgan+desc) + cozucu matrisi; istatistikler SADECE train'den.
    Fiziksel varyantta NaN'lar train MEDYANI ile doldurulur."""
    def chrom(ss):
        return (np.stack([s['morgan'] for s in ss]).astype(np.float32),
                np.stack([s['chrom_desc'] for s in ss]).astype(np.float32))
    fp, de = {}, {}
    for k, v in [('tr', tr), ('va', va), ('te', te)]:
        fp[k], de[k] = chrom(v)
    # Morgan bitleri ikili -> OLCEKLENMEZ (makalenin geri kalaniyla tutarli);
    # yalnizca surekli tanimlayicilar standartlastirilir
    cm, cs = de['tr'].mean(0), de['tr'].std(0) + 1e-8
    Xc = {k: np.concatenate([fp[k], (de[k] - cm) / cs], axis=1).astype(np.float32)
          for k in fp}

    Xs = {k: solvent_matrix(v, variant) for k, v in [('tr', tr), ('va', va), ('te', te)]}
    if Xs['tr'] is not None:
        med = np.nanmedian(Xs['tr'], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        for k in Xs:
            Xs[k] = np.where(np.isnan(Xs[k]), med, Xs[k])
        sm, ss_ = Xs['tr'].mean(0), Xs['tr'].std(0) + 1e-8
        for k in Xs:
            Xs[k] = ((Xs[k] - sm) / ss_).astype(np.float32)

    def yy(ss):
        Y = np.stack([s['targets'] for s in ss]).astype(np.float32)
        M = np.stack([s['mask'] for s in ss]).astype(np.float32)
        return Y, M
    Y, M = {}, {}
    for k, v in [('tr', tr), ('va', va), ('te', te)]:
        Y[k], M[k] = yy(v)
    tm = np.array([np.nanmean(np.where(M['tr'][:, j] > 0, Y['tr'][:, j], np.nan))
                   for j in range(4)], dtype=np.float32)
    tsd = np.array([max(np.nanstd(np.where(M['tr'][:, j] > 0, Y['tr'][:, j], np.nan)), 1e-6)
                    for j in range(4)], dtype=np.float32)
    for k in Y:
        Y[k] = np.where(M[k] > 0, (Y[k] - tm) / tsd, 0.0).astype(np.float32)
    return Xc, Xs, Y, M, tm, tsd

def run(tr, va, te, variant, seed, split_name):
    torch.manual_seed(seed); np.random.seed(seed)
    Xc, Xs, Y, M, tm, tsd = prep(tr, va, te, variant)
    mode = 'film' if variant == 'film' else 'concat'
    solv_dim = 0 if Xs['tr'] is None else Xs['tr'].shape[1]
    net = FiLMNet(Xc['tr'].shape[1], solv_dim, mode=mode).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    T = lambda a: None if a is None else torch.tensor(a).to(DEVICE)
    xc, xs = {k: T(Xc[k]) for k in Xc}, {k: T(Xs[k]) for k in Xs}
    y, m = {k: T(Y[k]) for k in Y}, {k: T(M[k]) for k in M}

    best, state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); opt.zero_grad()
        idx = torch.randperm(len(xc['tr']), device=DEVICE)
        pred = net(xc['tr'][idx], None if xs['tr'] is None else xs['tr'][idx])
        loss = masked_mse(pred, y['tr'][idx], m['tr'][idx])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = masked_mse(net(xc['va'], xs['va']), y['va'], m['va']).item()
        if not np.isfinite(v):
            raise RuntimeError(f"{variant}/{split_name}/seed{seed}: dogrulama kaybi NaN (ep{ep})")
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.cpu().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        P = net(xc['te'], xs['te']).cpu().numpy()
    P = P * tsd + tm
    Yt = np.stack([s['targets'] for s in te]).astype(np.float32)
    Mt = np.stack([s['mask'] for s in te]).astype(np.float32)

    rows = []
    for j, t in enumerate(TARGET_NAMES):
        k = Mt[:, j] > 0
        if k.sum() < 3:
            continue
        a, b = Yt[k, j], P[k, j]
        ss_res = float(np.sum((a - b) ** 2)); ss_tot = float(np.sum((a - a.mean()) ** 2))
        rows.append({'split': split_name, 'variant': variant, 'seed': seed,
                     'target': DISPLAY.get(t, t), 'n': int(k.sum()),
                     'MAE': round(float(np.mean(np.abs(a - b))), 4),
                     'R2': round(1 - ss_res / ss_tot, 4) if ss_tot > 1e-9 else np.nan,
                     'rho': round(float(spearmanr(a, b).statistic), 4)})
    return rows

if __name__ == '__main__':
    samples, base_meta, df = load_with_refs()
    g = df.groupby('Reference').agg(mol=(SMILES_COL, 'nunique'), qy=('QY', 'count')) \
          .sort_values('mol', ascending=False)
    pubs = [r for r, row in g.iterrows()
            if row['mol'] >= MIN_MOL and row['qy'] >= MIN_QY and r not in EXCLUDE][:N_FOLDS]
    ext_mols = set(df.loc[df['Reference'].isin(pubs), SMILES_COL])
    te_pool = [s for s in samples if s['chromophore'] in ext_mols]
    pool = [s for s in samples if s['chromophore'] not in ext_mols]
    cov = 100 * np.mean([s['solvent'] in __import__('solvent_props').SOLVENTS for s in samples])
    print(f"Fiziksel tablo kapsami: %{cov:.1f} kayit")

    rows = []
    for split_name in ['internal', 'pooled_external']:
        for seed in SEEDS:
            if split_name == 'internal':
                tr = [s for s in samples if s['split'] == 'train']
                va = [s for s in samples if s['split'] == 'val']
                te = [s for s in samples if s['split'] == 'test']
            else:
                rng = np.random.default_rng(seed)
                mols = sorted({s['chromophore'] for s in pool}); rng.shuffle(mols)
                vm = set(mols[:max(int(0.12 * len(mols)), 1)])
                tr = [s for s in pool if s['chromophore'] not in vm]
                va = [s for s in pool if s['chromophore'] in vm]
                te = te_pool
            for v in VARIANTS:
                rows += run(tr, va, te, v, seed, split_name)
            print(f"  {split_name} seed{seed} tamam")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, 'solvent_ablation_raw.csv'), index=False)
    gg = raw.groupby(['split', 'variant', 'target'])[['MAE', 'R2', 'rho']].agg(['mean', 'std'])
    out = pd.DataFrame(index=gg.index)
    for c in ['MAE', 'R2', 'rho']:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(gg[(c, 'mean')], gg[(c, 'std')])]
    out = out.reset_index()
    out.to_csv(os.path.join(RESULTS, 'solvent_ablation_summary.csv'), index=False)
    for s in ['internal', 'pooled_external']:
        print(f"\n=== {s.upper()} — MAE ===")
        print(out[out.split == s].pivot(index='variant', columns='target',
                                        values='MAE').reindex(VARIANTS).to_string())
    print("\nTablo: results/solvent_ablation_summary.csv")
