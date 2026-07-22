"""attention_maps.py - GATv2 attention → atom-düzeyi önem haritaları (5-seed ensemble).
Derin 2018'in 8 bileşiği + 4-I-fenil analoğu için hangi atomların ΦF tahminini
sürüklediğini gösterir. Çıktı: results/attention_maps.png, attention_atoms.csv
Çalıştır: python src/attention_maps.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from data_utils import ROOT
from data_loader import load_features, _to_data, _tgt_stats
from model import GATv2Net
from train import DEVICE, RESULTS
from feature_extraction import mol_to_graph, global_desc, TARGET_NAMES

SEEDS = [0, 1, 2, 3, 4]
EXT_CSV = os.path.join(ROOT, 'data', 'external', 'derin2018.csv')

@torch.no_grad()
def atom_importance(model, data):
    """Atom önemi = DİĞER atomların bu atoma verdiği attention (giden/source yönü).
    NOT: gelen (dst) attention softmax ile normalize olduğu için düğüm başına
    daima 1.0 toplar ve bilgi taşımaz; bu yüzden source yönü kullanılır."""
    x, ei, ea = data.x, data.edge_index, data.edge_attr
    imp = np.zeros(x.size(0), dtype=np.float64)
    for conv in [model.g1, model.g2, model.g3]:
        out, (ei_a, alpha) = conv(x, ei, ea, return_attention_weights=True)
        a = alpha.mean(dim=1).cpu().numpy()          # kafalar üzerinden ortalama
        src = ei_a[0].cpu().numpy()
        np.add.at(imp, src, a)                        # bu atoma verilen dikkat
        x = torch.relu(out)
    return imp / (imp.max() + 1e-12)

def ensemble_importance(smiles, meta):
    """5 seed ortalaması: atom önem vektörü."""
    sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
    sds = np.array(meta['solv_desc_std'], dtype=np.float32)
    tm, ts = _tgt_stats(meta)
    sample = {'graph': mol_to_graph(smiles),
              'solv_desc': global_desc('ClC(Cl)Cl').astype(np.float32),
              'targets': np.zeros(4, dtype=np.float32),
              'mask': np.zeros(4, dtype=np.float32)}
    data = _to_data(sample, sdm, sds, tm, ts).to(DEVICE)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
    acc = []
    for s in SEEDS:
        ck = os.path.join(RESULTS, f'ckpt_GATv2_s{s}.pt')
        if not os.path.exists(ck):
            continue
        m = GATv2Net(); m.load_state_dict(torch.load(ck, map_location=DEVICE))
        m.to(DEVICE).eval()
        acc.append(atom_importance(m, data))
    return np.mean(acc, axis=0), np.std(acc, axis=0)

def draw_mol(smiles, imp, title, size=(420, 340)):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    cmap = plt.get_cmap('OrRd')
    norm = mcolors.Normalize(vmin=float(imp.min()), vmax=float(imp.max()))
    hl = {i: [tuple(cmap(norm(imp[i]))[:3])] for i in range(mol.GetNumAtoms())}
    rad = {i: 0.35 for i in range(mol.GetNumAtoms())}
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors={k: v[0] for k, v in hl.items()},
        highlightAtomRadii=rad, highlightBonds=[])
    d.FinishDrawing()
    import io
    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))

def region_of(mol, idx):
    """Atomu kabaca sınıflandır: BF2 / pyrrole-N / core / meso-substituent."""
    a = mol.GetAtomWithIdx(idx)
    s = a.GetSymbol()
    if s in ('B', 'F'):
        return 'BF2'
    if s == 'N':
        return 'pyrrole N'
    return 'other'

if __name__ == '__main__':
    _, meta = load_features()
    df = pd.read_csv(EXT_CSV)
    # + iyot analoğu (ağır atom etkisinin uç örneği)
    core_i = "C(c1ccc(I)cc1)(=C2C=CC(C)=[N+]2[B-]3(F)F)C4=CC=C(C)N34"
    extra = pd.DataFrame([{'id': '4A-I*', 'Chromophore': core_i, 'Solvent': 'ClC(Cl)Cl'}])
    df = pd.concat([df[['id', 'Chromophore']], extra[['id', 'Chromophore']]], ignore_index=True)

    rows, images, titles = [], [], []
    for _, r in df.iterrows():
        imp, sd = ensemble_importance(r['Chromophore'], meta)
        mol = Chem.MolFromSmiles(r['Chromophore'])
        for i in range(mol.GetNumAtoms()):
            rows.append({'id': r['id'], 'atom_idx': i,
                         'symbol': mol.GetAtomWithIdx(i).GetSymbol(),
                         'region': region_of(mol, i),
                         'importance': round(float(imp[i]), 4),
                         'sd': round(float(sd[i]), 4)})
        images.append(draw_mol(r['Chromophore'], imp, r['id']))
        titles.append(r['id'])
        print(f"{r['id']}: en yüksek 3 atom ->",
              [(mol.GetAtomWithIdx(int(i)).GetSymbol(), round(float(imp[i]), 3))
               for i in np.argsort(-imp)[:3]])

    at = pd.DataFrame(rows)
    at.to_csv(os.path.join(RESULTS, 'attention_atoms.csv'), index=False)
    print("\nBölge ortalama önem:")
    print(at.groupby('region')['importance'].agg(['mean', 'count']).round(3).to_string())

    n = len(images); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow))
    for ax, img, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img); ax.set_title(t, fontsize=11); ax.axis('off')
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis('off')
    fig.suptitle('GATv2 attention-derived atom importance (5-seed ensemble)', fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, 'attention_maps.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print("Figür:", out)
