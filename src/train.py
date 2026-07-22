"""train.py - Train 4 architectures (MLP/1D-CNN/SMILES-Transformer/GATv2), multi-task regression.
Produces (paper-ready, English labels): results/metrics_<model>.csv, all_metrics.csv,
parity_<model>.png, preds_<model>.csv, comparison_table.csv, comparison_R2.csv, compare_R2.png
Run: python src/train.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import ROOT
from data_loader import (make_mlp_loaders, make_seq_loaders, make_graph_loaders, destandardize)
from model import MLP, CNN1D, SMILESTransformer, GATv2Net

RESULTS = os.path.join(ROOT, 'results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UNITS = {'lambda_abs': 'nm', 'lambda_em': 'nm', 'QY': '', 'log_eps': ''}
DISPLAY = {'lambda_abs': 'Abs. λmax (nm)', 'lambda_em': 'Em. λmax (nm)',
           'QY': 'ΦF (quantum yield)', 'log_eps': 'log ε'}

def fwd_mlp(model, batch, dev):
    X, Y, M = batch
    return model(X.to(dev)), Y.to(dev), M.to(dev)

def fwd_seq(model, batch, dev):
    tok, pm, Y, M = batch
    return model(tok.to(dev), pm.to(dev)), Y.to(dev), M.to(dev)

def fwd_graph(model, batch, dev):
    batch = batch.to(dev)
    return model(batch), batch.y, batch.mask

def masked_mse(pred, y, m):
    se = ((pred - y) ** 2) * m
    return se.sum() / m.sum().clamp(min=1.0)

def run_epoch(model, loader, fwd, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        if train:
            opt.zero_grad()
        pred, Y, M = fwd(model, batch, DEVICE)
        loss = masked_mse(pred, Y, M)
        if train:
            loss.backward(); opt.step()
        bs = Y.size(0); tot += loss.item() * bs; n += bs
    return tot / max(n, 1)

@torch.no_grad()
def predict(model, loader, fwd):
    model.eval(); P, Yt, Mt = [], [], []
    for batch in loader:
        pred, Y, M = fwd(model, batch, DEVICE)
        P.append(pred.cpu().numpy()); Yt.append(Y.cpu().numpy()); Mt.append(M.cpu().numpy())
    return np.concatenate(P), np.concatenate(Yt), np.concatenate(Mt)

def evaluate(pred_std, y_std, mask, meta):
    pred = destandardize(pred_std, meta); y = destandardize(y_std, meta)
    rows = []
    for j, name in enumerate(meta['target_names']):
        mk = mask[:, j] > 0
        yt, yp = y[mk, j], pred[mk, j]
        if len(yt) < 2:
            continue
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        ss_res = float(np.sum((yt - yp) ** 2)); ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rows.append({'target': name, 'n': int(mk.sum()),
                     'R2': round(r2, 4), 'RMSE': round(rmse, 3), 'MAE': round(mae, 3)})
    return pd.DataFrame(rows), pred, y

def parity_plot(pred, y, mask, meta, name):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, j in zip(axes.ravel(), range(len(meta['target_names']))):
        nm = meta['target_names'][j]
        mk = mask[:, j] > 0; yt, yp = y[mk, j], pred[mk, j]
        ax.scatter(yt, yp, s=12, alpha=0.45, color='#2c6e8f', edgecolor='none')
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1)
        ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rmse = np.sqrt(np.mean((yt - yp) ** 2)); u = UNITS.get(nm, ''); uu = f' ({u})' if u else ''
        ax.set_title(f"{DISPLAY.get(nm, nm)}  (R²={r2:.3f}, RMSE={rmse:.2f} {u})".rstrip())
        ax.set_xlabel(f'Measured{uu}'); ax.set_ylabel(f'Predicted{uu}')
    fig.suptitle(f"{name} — test-set parity (BODIPY dyes)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, f'parity_{name}.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    return out

def save_metrics(tbl, name):
    os.makedirs(RESULTS, exist_ok=True)
    disp = tbl.copy(); disp['target'] = disp['target'].map(lambda t: DISPLAY.get(t, t))
    disp.to_csv(os.path.join(RESULTS, f'metrics_{name}.csv'), index=False)
    t2 = tbl.copy(); t2.insert(0, 'model', name)            # all_metrics keeps raw keys
    master = os.path.join(RESULTS, 'all_metrics.csv')
    if os.path.exists(master):
        prev = pd.read_csv(master); prev = prev[prev['model'] != name]
        t2 = pd.concat([prev, t2], ignore_index=True)
    t2.to_csv(master, index=False)

def fit(model, loaders, fwd, name, epochs=300, lr=1e-3, wd=1e-5, patience=30):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e9, None, 0
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, loaders['train'], fwd, opt)
        va = run_epoch(model, loaders['val'], fwd)
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 25 == 0:
            print(f"  ep{ep:3d}  train {tr:.4f}  val {va:.4f}  (best {best:.4f})")
        if bad >= patience:
            print(f"  early stop @ ep{ep} (best val {best:.4f})")
            break
    model.load_state_dict(best_state)
    return model

def build_comparison(meta):
    df = pd.read_csv(os.path.join(RESULTS, 'all_metrics.csv'))
    order = list(meta['target_names'])
    disp = df.copy(); disp['target'] = disp['target'].map(lambda t: DISPLAY.get(t, t))
    disp.to_csv(os.path.join(RESULTS, 'comparison_table.csv'), index=False)
    piv = df.pivot_table(index='model', columns='target', values='R2')
    piv = piv[[t for t in order if t in piv.columns]]
    targets = list(piv.columns)
    piv.rename(columns={t: DISPLAY.get(t, t) for t in targets}).to_csv(
        os.path.join(RESULTS, 'comparison_R2.csv'))
    models = list(piv.index)
    x = np.arange(len(targets)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        ax.bar(x + i * w, piv.loc[m].values, w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels([DISPLAY.get(t, t) for t in targets])
    ax.set_ylabel('Test R²'); ax.set_ylim(0, 1); ax.legend()
    ax.set_title('Architecture comparison — test R² (BODIPY dyes)')
    fig.tight_layout(); out = os.path.join(RESULTS, 'compare_R2.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print("\n=== COMPARISON (test R²) ===")
    print(piv.rename(columns={t: DISPLAY.get(t, t) for t in targets}).round(3).to_string())
    print("Figure:", out)

def main():
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Device:", DEVICE)
    ml, meta, in_dim = make_mlp_loaders(256)
    sl, _ = make_seq_loaders(128)
    gl, _ = make_graph_loaders(128)
    configs = [
        ('MLP', ml, fwd_mlp, lambda: MLP(in_dim)),
        ('1D-CNN', sl, fwd_seq, lambda: CNN1D(meta['vocab_size'])),
        ('Transformer', sl, fwd_seq, lambda: SMILESTransformer(meta['vocab_size'])),
        ('GATv2', gl, fwd_graph, lambda: GATv2Net()),
    ]
    for name, loaders, fwd, build in configs:
        print(f"\n##### {name} #####")
        model = fit(build(), loaders, fwd, name)
        torch.save(model.state_dict(), os.path.join(RESULTS, f'ckpt_{name}.pt'))
        ps, ys, ms = predict(model, loaders['test'], fwd)
        tbl, pred, y = evaluate(ps, ys, ms, meta)
        save_metrics(tbl, name); parity_plot(pred, y, ms, meta, name)
        pd.DataFrame(pred, columns=[f'pred_{n}' for n in meta['target_names']]).to_csv(
            os.path.join(RESULTS, f'preds_{name}.csv'), index=False)
        print(tbl.to_string(index=False))
    build_comparison(meta)

if __name__ == '__main__':
    main()
