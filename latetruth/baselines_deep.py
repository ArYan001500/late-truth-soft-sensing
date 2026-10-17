"""Protocol 05_Protocols/PUBLISHED_BASELINES_v001.md (Amendment 4) — deep baselines, optional GPU support.
B4 Fu 2019 CNN-LSTM, B7 Yan 2023 hybrid-attention BiConvLSTM, B8 Dai 2025 GCN-TCN-LSTM, B9 Grataloup 2025 LSTM fleet NBM (local variant).
All four are target-free as published (no oil-temperature history), so lag 3 vs 1 does not apply. Per turbine; same monthly starts as ours:
first window trained from scratch (<= 30 epochs, early stopping patience 3 on the last 10 % of released rows), later windows fine-tuned
from the previous weights for 5 epochs on all released rows (documented compute deviation). Inputs causally forward-filled, standardised on
the first training window. usage: python baselines_deep.py <B4|B7|B8|B9> <turbine> <dataset>
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import sys, json, hashlib, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(min(2, JOBS)); torch.manual_seed(0); np.random.seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = OUT_DIR/'published_baselines_2026-10-14_v001'; EV0 = 4032; DELAY = 3
POWER, WIND, ROTOR, GENRPM, AMB, NAC, CUR, GENB, INLET, FRONT, REAR, ROTB, GBS, VANE, PPUMP, PINL = range(16); WDIR, STATOR = 16, 17
# channel maps (paper variable -> closest Kelmarsh channel; see protocol Amendment 4)
CH = {'B4': [REAR, FRONT, INLET, GENB, NAC, AMB, ROTB, STATOR, WIND, POWER],            # Fu Table 1 (T1..T8, V, P)
      'B7': [POWER, ROTOR, AMB, CUR, FRONT, INLET, PPUMP, REAR, GENRPM, STATOR, ROTB, GBS, WIND],  # Yan (13 inputs)
      'B8': [WIND, POWER, ROTOR, CUR, NAC, STATOR, FRONT, REAR, GENB, AMB],             # Dai (10 inputs)
      'B9': [WIND, AMB, 'sin', 'cos']}                                                  # Grataloup
L = {'B4': 10, 'B7': 6, 'B8': 12, 'B9': 144}
LR = {'B4': 1e-3, 'B7': 1e-3, 'B8': 1e-3, 'B9': 8e-4}; BS = {'B4': 100, 'B7': 64, 'B8': 64, 'B9': 256}; STRIDE = {'B9': 3}
BUDGET = os.environ.get('DEEP_BUDGET', 'v1')  # v1: 30 ep (+patience 3) then 5; v2 (Amendment 5): 100 ep (+patience 10) then 10
E0, PAT, E1 = (30, 3, 5) if BUDGET == 'v1' else (100, 10, 10)


def load(ds, t):
    z = np.load(OUT/f'inputs_{ds}.npz', allow_pickle=False); z2 = np.load(OUT/f'inputs2_{ds}.npz', allow_pickle=False)
    X = np.column_stack([z[f'X_{t}'], z2[f'X_{t}']]); return z['month'], X, z[f'Y_{t}'], z[f'OK_{t}']


def channels(name, X):
    cols = []
    for c in CH[name]:
        if c == 'sin': cols.append(np.sin(np.deg2rad(X[:, WDIR])))
        elif c == 'cos': cols.append(np.cos(np.deg2rad(X[:, WDIR])))
        else: cols.append(X[:, c])
    A = np.column_stack(cols)
    for j in range(A.shape[1]):  # causal forward fill
        v = A[:, j]; idx = np.where(np.isfinite(v), np.arange(len(v)), 0); np.maximum.accumulate(idx, out=idx); A[:, j] = v[idx]
    return A


class Fu(nn.Module):        # conv on the 10x10 (time x variable) image -> FC -> LSTM -> output
    def __init__(s, k, l):
        super().__init__(); s.c = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1, 2)))
        s.fc = nn.Linear(16 * (k // 2), 100); s.lstm = nn.LSTM(100, 32, batch_first=True); s.o = nn.Linear(32, 1)
    def forward(s, x):
        b, l, k = x.shape; h = s.c(x[:, None]).permute(0, 2, 1, 3).reshape(b, l, -1); h, _ = s.lstm(torch.relu(s.fc(h))); return s.o(h[:, -1])[:, 0]


class Yan(nn.Module):       # feature attention x time attention, then bidirectional conv-LSTM (1-D conv over variables in the gates)
    def __init__(s, k, l, hid=32):
        super().__init__(); s.fa = nn.Linear(k, k); s.ta = nn.Linear(k, 1); s.conv = nn.Conv1d(1, 4, 3, padding=1)
        s.f = nn.LSTM(4 * k, hid, batch_first=True); s.b = nn.LSTM(4 * k, hid, batch_first=True); s.o = nn.Linear(2 * hid, 1)
    def forward(s, x):
        b, l, k = x.shape; x = x * torch.softmax(s.fa(x), -1) * k; x = x * torch.softmax(s.ta(x), 1) * l
        h = torch.relu(s.conv(x.reshape(b * l, 1, k))).reshape(b, l, -1)
        hf, _ = s.f(h); hb, _ = s.b(torch.flip(h, [1])); return s.o(torch.cat([hf[:, -1], hb[:, -1]], -1))[:, 0]


class Dai(nn.Module):       # GCN (64, 32) over variables with a data adjacency, TCN (64,64; kernel 2; dilation 1,2; causal; dropout .05), LSTM 64, dense 64->1
    def __init__(s, k, l, A):
        super().__init__(); s.register_buffer('A', A); s.g1 = nn.Linear(1, 64); s.g2 = nn.Linear(64, 32)
        s.t1 = nn.Conv1d(32 * k, 64, 2, dilation=1); s.t2 = nn.Conv1d(64, 64, 2, dilation=2); s.dp = nn.Dropout(0.05)
        s.lstm = nn.LSTM(64, 64, batch_first=True); s.d = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x):
        b, l, k = x.shape; h = torch.relu(s.A @ s.g1(x[..., None])); h = torch.relu(s.A @ s.g2(h)); h = h.reshape(b, l, -1).transpose(1, 2)
        h = s.dp(torch.relu(s.t1(nn.functional.pad(h, (1, 0))))); h = s.dp(torch.relu(s.t2(nn.functional.pad(h, (2, 0)))))
        h, _ = s.lstm(h.transpose(1, 2)); return s.d(h[:, -1])[:, 0]


class Grat(nn.Module):      # LSTM 16 -> LSTM 64 -> FC 64 -> FC 32 -> 1, ReLU
    def __init__(s, k, l):
        super().__init__(); s.l1 = nn.LSTM(k, 16, batch_first=True); s.l2 = nn.LSTM(16, 64, batch_first=True)
        s.d = nn.Sequential(nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): h, _ = s.l1(x); h, _ = s.l2(h); return s.d(h[:, -1])[:, 0]


def windows(month, n):
    s = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]
    return [(a, s[j + 1] if j + 1 < len(s) else n) for j, a in enumerate(s)]


def seq(A, rows, l): return np.stack([A[r - l + 1:r + 1] for r in rows]).astype(np.float32)


def train(model, Xs, ys, lr, bs, epochs, patience=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr); n = len(ys)
    if patience: cut = int(n * 0.9); Xv, yv = torch.tensor(Xs[cut:], device=DEV), torch.tensor(ys[cut:], device=DEV); Xs, ys = Xs[:cut], ys[:cut]; best, bad, keep = 1e9, 0, None
    Xt, yt = torch.tensor(Xs, device=DEV), torch.tensor(ys, device=DEV)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(yt), bs):
            j = perm[i:i + bs]; opt.zero_grad(); loss = nn.functional.mse_loss(model(Xt[j]), yt[j]); loss.backward(); opt.step()
        if os.environ.get('EPOCH_LOG'): print(f'  epoch {ep + 1}/{epochs}  last batch loss {float(loss):.4f}', flush=True)   # progress only
        if patience:
            model.eval()
            with torch.no_grad(): v = float(nn.functional.mse_loss(model(Xv), yv))
            if v < best - 1e-6: best, bad, keep = v, 0, {k: x.clone() for k, x in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if patience and keep: model.load_state_dict(keep)
    return model


def main(name, t, ds):
    f = OUT/(f'deep_{name}_{ds}_T{t}.npz' if BUDGET == 'v1' else f'deep{BUDGET}_{name}_{ds}_T{t}.npz')
    if f.exists(): raise FileExistsError(f)
    month, X, y, OK = load(ds, t); n = len(month); A = channels(name, X); l = L[name]; k = A.shape[1]
    first = np.arange(l, EV0 - DELAY); first = first[OK[first] & np.isfinite(y[first])]
    mu, sd = np.nanmean(A[first], 0), np.nanstd(A[first], 0) + 1e-6; ym, ys_ = y[first].mean(), y[first].std()
    An = np.nan_to_num((A - mu) / sd)
    if name == 'B8':
        C = np.nan_to_num(np.corrcoef(An[first].T)); Adj = (np.abs(C) > 0.3).astype(np.float32); Dg = Adj.sum(1); Adj = Adj / np.sqrt(np.outer(Dg, Dg))
        model = Dai(k, l, torch.tensor(Adj, dtype=torch.float32))
    else:
        model = {'B4': Fu, 'B7': Yan, 'B9': Grat}[name](k, l)
    model = model.to(DEV); P = np.full(n, np.nan); st = STRIDE.get(name, 1)
    for j, (s, e) in enumerate(windows(month, n)):
        tr = np.arange(l, s - DELAY); tr = tr[OK[tr] & np.isfinite(y[tr])][::st]
        model = train(model, seq(An, tr, l), ((y[tr] - ym) / ys_).astype(np.float32), LR[name], BS[name], E0 if j == 0 else E1, PAT if j == 0 else None)
        r = np.arange(s, e); r = r[OK[r]]; model.eval()
        if len(r) == 0: continue   # a turbine-window without valid rows (HoT 2022 T6, February): nothing to predict, model kept
        with torch.no_grad(): P[r] = np.concatenate([model(torch.tensor(seq(An, r[i:i + 4096], l), device=DEV)).cpu().numpy() for i in range(0, len(r), 4096)]) * ys_ + ym
        print(name, ds, 'T', t, 'window', j, 'done', flush=True)
    np.savez_compressed(f, P=P, code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest())


if __name__ == '__main__': main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
