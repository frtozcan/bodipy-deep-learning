"""lopo_plot.py - lopo_raw.csv'den kutu grafigi (egitim tekrarlanmaz)."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import ROOT

R = os.path.join(ROOT, 'results')
raw = pd.read_csv(os.path.join(R, 'lopo_raw.csv'))
targets = ['Abs. λmax (nm)', 'Em. λmax (nm)', 'ΦF (quantum yield)', 'log ε']
models = ['MLP', '1D-CNN', 'GATv2', 'Transformer']

fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
for ax, t in zip(axes, targets):
    data = [raw[(raw.model == m) & (raw.target == t)]['MAE'].dropna().values for m in models]
    bp = ax.boxplot(data, tick_labels=models, showfliers=True, patch_artist=True)
    for p in bp['boxes']:
        p.set_facecolor('#cfe3ee')
    for i, d in enumerate(data, 1):                       # katman noktalari
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.08, .08, len(d)), d,
                   s=14, color='#2c6e8f', alpha=.75, zorder=3)
    ax.set_title(t, fontsize=10); ax.set_ylabel('MAE per held-out publication')
    ax.tick_params(axis='x', rotation=25)
fig.suptitle('Leave-one-publication-out validation: error distribution across 12 held-out studies')
fig.tight_layout()
p = os.path.join(R, 'lopo_box.png'); fig.savefig(p, dpi=130); plt.close(fig)
print('Figur:', p)
