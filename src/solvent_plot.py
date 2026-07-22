"""solvent_plot.py - Cozucu ablasyonu figuru (kayitli CSV'den, egitim tekrarlanmaz)."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import ROOT

R = os.path.join(ROOT, 'results')
raw = pd.read_csv(os.path.join(R, 'solvent_ablation_raw.csv'))
order = ['none', 'generic', 'physical', 'film']
labels = ['no solvent', 'generic\ndescriptors', 'physical\ndescriptors', 'physical\n+ FiLM']
targets = ['ΦF (quantum yield)', 'Abs. λmax (nm)']

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, t in zip(axes, targets):
    w, x = 0.36, np.arange(len(order))
    for i, sp in enumerate(['internal', 'pooled_external']):
        sub = raw[(raw.split == sp) & (raw.target == t)]
        m = [sub[sub.variant == v]['MAE'].mean() for v in order]
        e = [sub[sub.variant == v]['MAE'].std() for v in order]
        ax.bar(x + i * w, m, w, yerr=e, capsize=3,
               label='internal test' if i == 0 else 'pooled external')
    ax.set_xticks(x + w / 2); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MAE'); ax.set_title(t, fontsize=11)
axes[0].legend(fontsize=8)
fig.suptitle('Solvent representation ablation (MLP, multi-task, 5 seeds)')
fig.tight_layout()
p = os.path.join(R, 'solvent_ablation.png'); fig.savefig(p, dpi=130); plt.close(fig)
print('Figur:', p)
