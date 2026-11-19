from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part G, Amendment G-b: architecture (v), a GRU on the six most recent input vectors.
Uses the functions of revision_controlled.py unchanged. usage: [HOT_YEAR=2019] python revision_controlled_gru.py <ds>"""
import sys, json, hashlib, time
import numpy as np
import torch
from latetruth import baselines_common as BC
from latetruth import revision_turbine_id as RT
from latetruth import revision_controlled as RC

OUT = RT.OUT; EV0 = BC.EV0; D = 3; W = 6; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'   # the published runs used a GPU


class GRUNet(torch.nn.Module):
    def __init__(self, f):
        super().__init__(); self.g = torch.nn.GRU(f, 64, batch_first=True); self.o = torch.nn.Linear(64, 1)

    def forward(self, x): return self.o(self.g(x)[0][:, -1])[:, 0]


def fit_predict(F, Y, OK, T, month, n, pooled):
    P = {t: np.full(n, np.nan) for t in T}
    for s, e in BC.windows(month, n):
        for G in ([T] if pooled else [[t] for t in T]):
            Dm = {t: RC.design(F[t], T, t, n, 'mlp', pooled) for t in G}
            tr = {t: (lambda r: r[OK[t][r]])(np.arange(7, s - D)) for t in G}
            Xtr = np.vstack([Dm[t][tr[t]] for t in G]); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xtr.T])
            prep = RC.Model('mlp'); prep._prep(Xtr[:, keep], True)
            Z = np.vstack([prep._prep(Dm[t][:, keep], False) for t in G]).astype(np.float32)     # all rows of all turbines of G
            off = {t: j * n for j, t in enumerate(G)}
            itr = np.concatenate([tr[t] + off[t] for t in G]); ytr = np.concatenate([Y[t][tr[t]] for t in G])
            ym, ys = float(ytr.mean()), float(ytr.std() + 1e-9)
            Zg = torch.tensor(Z, device=DEV); yg = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV)
            back = torch.arange(-(W - 1), 1, device=DEV)
            torch.manual_seed(20261117); net = GRUNet(Z.shape[1]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            it = torch.tensor(itr, device=DEV); g = torch.Generator(device='cpu').manual_seed(20261117)
            for _ in range(20):
                for b in torch.randperm(len(it), generator=g).split(1024):
                    b = b.to(DEV); x = Zg[it[b][:, None] + back]
                    opt.zero_grad(); loss = torch.mean((net(x) - yg[b]) ** 2); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                for t in G:
                    r = np.arange(s, e); r = r[OK[t][r]]
                    if not len(r): continue
                    ii = torch.tensor(r + off[t], device=DEV); out = []
                    for c in ii.split(65536): out.append(net(Zg[c[:, None] + back]).cpu().numpy())
                    P[t][r] = np.concatenate(out) * ys + ym
            del Zg, yg, net, opt; torch.cuda.empty_cache()
    return P


def main(ds):
    t0 = time.time(); Dd, Y, OK, Ffull, hc, month, n = RT.data(ds); T = list(Dd); Fb = RC.base_feats(ds, Dd, n); S = {}; info = {}
    for inp, F in [('full', Ffull), ('base', Fb)]:
        for pool in ('fleet', 'pt'):
            t1 = time.time(); P = fit_predict(F, Y, OK, T, month, n, pool == 'fleet'); PR = RC.correct(P, Y, OK, Dd, hc, n)
            for t in T: S[f'{inp}|{pool}|none_{t}'] = P[t]; S[f'{inp}|{pool}|corr_{t}'] = PR[t]
            ev = np.arange(EV0, n)
            m0 = np.mean([np.abs(Y[t][ev] - P[t][ev])[OK[t][ev] & np.isfinite(P[t][ev])].mean() for t in T])
            m1 = np.mean([np.abs(Y[t][ev] - PR[t][ev])[OK[t][ev] & np.isfinite(PR[t][ev])].mean() for t in T])
            info[f'{inp}|{pool}'] = dict(own_rows_mae_none=float(m0), own_rows_mae_corr=float(m1), seconds=round(time.time() - t1, 1))
            print(ds, 'gru', inp, pool, round(m0, 4), round(m1, 4), round(time.time() - t1), 's', flush=True)
    np.savez_compressed(OUT/f'controlled_{ds}_gru.npz', **S)
    (OUT/f'controlled_{ds}_gru.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part G, Amendment G-b', ds=ds, arch='gru', info=info,
        seconds=round(time.time() - t0, 1), code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__': main(sys.argv[1])
