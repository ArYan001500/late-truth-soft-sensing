"""Amendment 11 S6 — Cao, Chang, Qian, Meng, Li 2026 (MSSP 260, 115021): RESCINet (SCINet with residual-enhanced SCI blocks and hierarchical
residual correction over interaction depths). Lite re-implementation from the paper text (authors' code not available):
- per-variable causal 1-D Kalman filter (A = H = 1, P0 = 1, Q = 0.01, R = 1), Pearson screening (|r| >= 0.4 on the first training window);
- input length 128 (paper), the target's own history included as a channel but only up to t-3 (receipt-causal; values after t-3 are held at
  the last released value); nowcast of y(t) (paper: 6-step forecast);
- 3 interaction levels, hidden 16, one stack; SCI interaction with a gated residual path; depth candidates fused by learned weights into a
  base prediction, the remaining depths give a weighted residual correction; final = base + correction;
- Adam lr 1e-5, batch 32, 100 epochs without early stopping (all as in the paper) on the first window; later monthly windows 10
  fine-tuning epochs (the monthly retraining protocol shared by every method in this comparison).
MC-dropout / conformal thresholds are for alarms and are not used (we compare point accuracy). Kelmarsh 2018 only (cost).
usage: python baselines_cao.py <turbine> [dataset]"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from latetruth import baselines_deep as BD
DEV = BD.DEV; L = 128; DELAY = 3
import os
V2 = os.environ.get('CAO_BUDGET') == 'v2'; LR = 1e-3 if V2 else 1e-5; PRE = 'caov2' if V2 else 'cao'   # Amendment 12


def kalman(v):
    out = np.empty_like(v); x, p = (v[np.isfinite(v)][0] if np.isfinite(v).any() else 0.0), 1.0
    for i, z in enumerate(v):
        p += 0.01
        if np.isfinite(z): k = p / (p + 1.0); x += k * (z - x); p *= (1 - k)
        out[i] = x
    return out


class SCI(nn.Module):
    def __init__(s, c, h=16):
        super().__init__()
        conv = lambda: nn.Sequential(nn.Conv1d(c, h, 3, padding=1), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Conv1d(h, c, 3, padding=1), nn.Tanh())
        s.phi, s.psi, s.rho, s.eta = conv(), conv(), conv(), conv(); s.gate = nn.Conv1d(2 * c, c, 1)
    def forward(s, x):          # x: B, C, T (T even)
        e, o = x[..., ::2], x[..., 1::2]
        o1 = o * torch.exp(s.phi(e)); e1 = e * torch.exp(s.psi(o))
        o2 = o1 + s.rho(e1); e2 = e1 - s.eta(o1)
        g = torch.sigmoid(s.gate(torch.cat([e, o], 1)))          # gated residual path (residual-enhanced SCI)
        return e2 + g * e, o2 + g * o


class RESCINet(nn.Module):
    def __init__(s, c, levels=3):
        super().__init__(); s.levels = levels
        s.blocks = nn.ModuleList([nn.ModuleList([SCI(c) for _ in range(2 ** l)]) for l in range(levels)])
        s.proj = nn.ModuleList([nn.Linear(c * L, 32) for _ in range(levels)])
        s.wb = nn.Linear(32, 1); s.wr = nn.Linear(32, 1); s.base = nn.Linear(32, 1); s.res = nn.Linear(32, 1)
    @staticmethod
    def merge(parts):           # interleave the even/odd subsequences back into time order
        while len(parts) > 1:
            nxt = []
            for a, b in zip(parts[::2], parts[1::2]):
                z = torch.stack([a, b], -1).flatten(-2); nxt.append(z)
            parts = nxt
        return parts[0]
    def forward(s, x):          # x: B, T, C
        x = x.transpose(1, 2); parts = [x]; H = []
        for l in range(s.levels):
            nxt = []
            for blk, p in zip(s.blocks[l], parts): e, o = blk(p); nxt += [e, o]
            parts = nxt; z = s.merge(list(parts)) + x                   # realigned depth-l candidate (+ input residual)
            H.append(torch.tanh(s.proj[l](z.flatten(1))))
        Hs = torch.stack(H, 1)                                         # B, J, 32
        wb = torch.softmax(s.wb(Hs)[..., 0], 1); base = s.base((wb[..., None] * Hs).sum(1))[:, 0]
        dom = wb.argmax(1); mask = torch.ones_like(wb).scatter_(1, dom[:, None], 0.0)
        wr = torch.softmax(s.wr(Hs)[..., 0].masked_fill(mask == 0, -1e9), 1); corr = (wr[..., None] * s.res(Hs)).sum(1)[:, 0]
        return base + corr


def main(t, ds='kelmarsh2018'):
    # Resume support (2026-10-15): after each monthly window the model weights and the predictions so far are saved to a
    # checkpoint; a restarted run continues after the last finished window. Test-only overrides: CAO_OUT (output folder),
    # CAO_EPOCHS (epochs per window), CAO_MAXWIN (stop after this many windows). None of them is set in the real runs.
    out = Path(os.environ.get('CAO_OUT', BD.OUT)); f = out/f'{PRE}_{ds}_T{t}.npz'; ck = out/f'{PRE}_{ds}_T{t}.ckpt.pt'
    ep_test = int(os.environ['CAO_EPOCHS']) if 'CAO_EPOCHS' in os.environ else None; maxwin = int(os.environ.get('CAO_MAXWIN', 99))
    if f.exists(): raise FileExistsError(f)
    month, X, y, OK = BD.load(ds, t); n = len(month)
    first = np.arange(L, BD.EV0 - DELAY); first = first[OK[first] & np.isfinite(y[first])]
    rr = [abs(np.corrcoef(X[first, c][np.isfinite(X[first, c])], y[first][np.isfinite(X[first, c])])[0, 1]) if np.isfinite(X[first, c]).mean() > 0.95 and np.nanstd(X[first, c]) > 0 else 0 for c in range(X.shape[1])]
    ch = [c for c in range(X.shape[1]) if rr[c] >= 0.4]
    A = np.column_stack([kalman(X[:, c]) for c in ch]); yk = kalman(y)
    mu, sd = A[first].mean(0), A[first].std(0) + 1e-6; ym, ys = y[first].mean(), y[first].std()
    An = ((A - mu) / sd).astype(np.float32); yn = ((yk - ym) / ys).astype(np.float32)
    def windows(rows):          # inputs + target history visible only up to t-3
        Xs = np.stack([An[r - L + 1:r + 1] for r in rows]); Yh = np.stack([yn[r - L + 1:r + 1] for r in rows]).copy()
        Yh[:, -DELAY:] = Yh[:, [-DELAY - 1]]; return np.concatenate([Xs, Yh[..., None]], -1).astype(np.float32)
    model = RESCINet(len(ch) + 1).to(DEV); P = np.full(n, np.nan); start = 0
    if ck.exists():
        c = torch.load(ck, map_location=DEV, weights_only=True); model.load_state_dict(c['model']); P = c['P'].cpu().numpy(); start = int(c['next'])
        print('cao', ds, 'T', t, 'resumed at window', start, flush=True)
    for j, (s, e) in enumerate(BD.windows(month, n)):
        if j < start: continue
        if j >= maxwin: return
        tr = np.arange(L, s - DELAY); tr = tr[OK[tr] & np.isfinite(y[tr])]
        epochs = ep_test if ep_test is not None else (100 if j == 0 else 10)
        model = BD.train(model, windows(tr), ((y[tr] - ym) / ys).astype(np.float32), LR, 32, epochs, None)   # paper: 100 epochs, no early stopping; v1 lr 1e-5 (paper), v2 lr 1e-3
        r = np.arange(s, e); r = r[OK[r]]; model.eval()
        with torch.no_grad(): P[r] = np.concatenate([model(torch.tensor(windows(r[i:i + 2048]), device=DEV)).cpu().numpy() for i in range(0, len(r), 2048)]) * ys + ym
        tmp = ck.with_suffix('.tmp'); torch.save({'model': model.state_dict(), 'P': torch.from_numpy(P), 'next': j + 1}, tmp); os.replace(tmp, ck)
        print('cao', ds, 'T', t, 'window', j, 'done', flush=True)
    np.savez_compressed(f, P=P, channels=np.array(ch)); ck.unlink(missing_ok=True)


if __name__ == '__main__': main(int(sys.argv[1]), *(sys.argv[2:3] or []))
