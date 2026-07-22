"""data_loader.py - torch veri yükleyicileri + standartlaştırma yardımcıları.
Standartlaştırma istatistikleri meta.json'dan (yalnızca train) gelir.
Şimdilik MLP yükleyicisi; GNN/sekans yükleyicileri sonra eklenecek.
"""
import os, json, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_utils import ROOT

PROC = os.path.join(ROOT, 'data', 'processed')

def load_features():
    with open(os.path.join(PROC, 'features.pkl'), 'rb') as f:
        samples = pickle.load(f)
    with open(os.path.join(PROC, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return samples, meta

def _split(samples, sp):
    return [s for s in samples if s['split'] == sp]

def _tgt_stats(meta):
    m = np.array([meta['target_mean'][k] for k in meta['target_names']], dtype=np.float32)
    s = np.array([meta['target_std'][k] for k in meta['target_names']], dtype=np.float32)
    return m, s

def destandardize(arr, meta):
    m, s = _tgt_stats(meta)
    return arr * s + m

# ---------- MLP yükleyici ----------
class MLPDataset(Dataset):
    """X = [Morgan(2048), kromofor_desc(10, std), çözücü_desc(10, std)] = 2068"""
    def __init__(self, samples, meta):
        cdm = np.array(meta['chrom_desc_mean'], dtype=np.float32)
        cds = np.array(meta['chrom_desc_std'], dtype=np.float32)
        sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
        sds = np.array(meta['solv_desc_std'], dtype=np.float32)
        tm, ts = _tgt_stats(meta)
        X, Y, M = [], [], []
        for s in samples:
            cd = (s['chrom_desc'] - cdm) / cds
            sd = (s['solv_desc'] - sdm) / sds
            X.append(np.concatenate([s['morgan'].astype(np.float32), cd, sd]))
            t = s['targets'].astype(np.float32); m = s['mask'].astype(np.float32)
            t = np.where(m > 0, (t - tm) / ts, 0.0).astype(np.float32)
            Y.append(t); M.append(m)
        self.X = np.stack(X); self.Y = np.stack(Y); self.M = np.stack(M)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i]

def _mlp_collate(batch):
    X = torch.tensor(np.stack([b[0] for b in batch]))
    Y = torch.tensor(np.stack([b[1] for b in batch]))
    M = torch.tensor(np.stack([b[2] for b in batch]))
    return X, Y, M

def make_mlp_loaders(batch_size=256):
    samples, meta = load_features()
    dss = {sp: MLPDataset(_split(samples, sp), meta) for sp in ['train', 'val', 'test']}
    loaders = {sp: DataLoader(dss[sp], batch_size=batch_size, shuffle=(sp == 'train'),
                              collate_fn=_mlp_collate) for sp in dss}
    return loaders, meta, dss['train'].X.shape[1]

# ---------- Sekans yükleyici (1D-CNN / Transformer) ----------
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader

MAX_LEN = 256

class SeqDataset(Dataset):
    def __init__(self, samples, meta):
        tm, ts = _tgt_stats(meta)
        self.tok, self.Y, self.M = [], [], []
        for s in samples:
            self.tok.append(np.array(s['token_ids'][:MAX_LEN], dtype=np.int64))
            t = s['targets'].astype(np.float32); m = s['mask'].astype(np.float32)
            t = np.where(m > 0, (t - tm) / ts, 0.0).astype(np.float32)
            self.Y.append(t); self.M.append(m)
    def __len__(self): return len(self.tok)
    def __getitem__(self, i): return self.tok[i], self.Y[i], self.M[i]

def _seq_collate(batch):
    L = max(len(b[0]) for b in batch); B = len(batch)
    tok = np.zeros((B, L), dtype=np.int64)   # PAD=0
    for i, b in enumerate(batch):
        tok[i, :len(b[0])] = b[0]
    tok = torch.tensor(tok)
    padmask = (tok == 0)                       # True = pad
    Y = torch.tensor(np.stack([b[1] for b in batch]))
    M = torch.tensor(np.stack([b[2] for b in batch]))
    return tok, padmask, Y, M

def make_seq_loaders(batch_size=128):
    samples, meta = load_features()
    dss = {sp: SeqDataset(_split(samples, sp), meta) for sp in ['train', 'val', 'test']}
    loaders = {sp: DataLoader(dss[sp], batch_size=batch_size, shuffle=(sp == 'train'),
                              collate_fn=_seq_collate) for sp in dss}
    return loaders, meta

# ---------- Graf yükleyici (GATv2-GNN, PyG) ----------
def _to_data(s, sdm, sds, tm, ts):
    g = s['graph']
    sd = (s['solv_desc'] - sdm) / sds
    t = s['targets'].astype(np.float32); m = s['mask'].astype(np.float32)
    t = np.where(m > 0, (t - tm) / ts, 0.0).astype(np.float32)
    d = Data(x=torch.tensor(g['x']),
             edge_index=torch.tensor(g['edge_index']),
             edge_attr=torch.tensor(g['edge_attr']))
    d.solv = torch.tensor(sd, dtype=torch.float32).view(1, -1)
    d.y = torch.tensor(t).view(1, -1)
    d.mask = torch.tensor(m).view(1, -1)
    return d

def make_graph_loaders(batch_size=128):
    samples, meta = load_features()
    sdm = np.array(meta['solv_desc_mean'], dtype=np.float32)
    sds = np.array(meta['solv_desc_std'], dtype=np.float32)
    tm, ts = _tgt_stats(meta)
    loaders = {}
    for sp in ['train', 'val', 'test']:
        ds = [_to_data(s, sdm, sds, tm, ts) for s in _split(samples, sp)]
        loaders[sp] = GeoLoader(ds, batch_size=batch_size, shuffle=(sp == 'train'))
    return loaders, meta
