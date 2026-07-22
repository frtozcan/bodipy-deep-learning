"""model.py - mimari tanımları (4-başlı çok-görevli regresyon).
Şimdilik MLP; 1D-CNN / GATv2-GNN / SMILES-Transformer sonra eklenecek.
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, n_targets=4, hidden=(512, 256), p=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_targets)
    def forward(self, x):
        return self.head(self.trunk(x))


# ---------- 1D-CNN (sekans) ----------
class CNN1D(nn.Module):
    def __init__(self, vocab, n_targets=4, emb=64, ch=128, p=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, ch, k, padding=k // 2) for k in (3, 5, 7)])
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(ch * 3, 256), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(256, n_targets))
    def forward(self, tok, padmask=None):
        e = self.emb(tok).transpose(1, 2)                      # [B,emb,L]
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.head(self.drop(torch.cat(feats, dim=1)))

# ---------- SMILES Transformer (sekans) ----------
class SMILESTransformer(nn.Module):
    def __init__(self, vocab, n_targets=4, emb=128, nhead=4, layers=3, p=0.1, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, emb) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=emb, nhead=nhead, dim_feedforward=emb * 4,
                                         dropout=p, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(emb, 128), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(128, n_targets))
    def forward(self, tok, padmask):
        L = tok.size(1)
        h = self.emb(tok) + self.pos[:, :L, :]
        h = self.tr(h, src_key_padding_mask=padmask)
        valid = (~padmask).unsqueeze(-1).float()
        h = (h * valid).sum(1) / valid.sum(1).clamp(min=1)     # masked mean pool
        return self.head(h)

# ---------- GATv2 GNN (graf) ----------
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2Net(nn.Module):
    def __init__(self, atom_dim=17, edge_dim=7, solv_dim=10, n_targets=4,
                 hidden=128, heads=4, p=0.2):
        super().__init__()
        self.g1 = GATv2Conv(atom_dim, hidden, heads=heads, edge_dim=edge_dim)
        self.g2 = GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim)
        self.g3 = GATv2Conv(hidden * heads, hidden, heads=1, edge_dim=edge_dim)
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(hidden + solv_dim, 128), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(128, n_targets))
    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.g1(x, ei, ea))
        x = torch.relu(self.g2(x, ei, ea))
        x = torch.relu(self.g3(x, ei, ea))
        g = global_mean_pool(x, data.batch)
        return self.head(self.drop(torch.cat([g, data.solv], dim=1)))
