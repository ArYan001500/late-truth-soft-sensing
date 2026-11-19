from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part G: controlled comparison (architecture x inputs x pooling x correction).
usage: [HOT_YEAR=2019] python revision_controlled.py <ds> <arch>      arch in hgb xgb ridge mlp"""
import os, sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth import residual_references as R
from latetruth import revision_turbine_id as RT
from latetruth.run_local_late_truth_long import PARAMS, local, HistGradientBoostingRegressor, threadpool_limits

OUT = RT.OUT; EV0 = BC.EV0; D = 3; TH = int(os.environ.get('THREADS', '2'))


def base_feats(ds, Dd, n):
    if ds.startswith('hot'):
        from latetruth import run_hill_of_towie_replication as H
        return {t: H.feats(df, 'base') for t, df in Dd.items()}
    from latetruth import run_h29_h30_physics as B
    rows = np.arange(7, n); F = {}
    for t, df in Dd.items():
        f = B.feats(df, rows, 'base'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    return F


class Model:
    """fit(X, y) / predict(X) for the four architectures; X may contain NaN."""
    def __init__(self, arch): self.arch = arch

    def _prep(self, X, fit):
        if fit:
            self.mu = np.nanmean(X, 0); self.sd = np.nanstd(X, 0); self.sd[~np.isfinite(self.sd) | (self.sd < 1e-9)] = 1.0
            self.mu[~np.isfinite(self.mu)] = 0.0; self.miss = np.flatnonzero(np.isnan(X).any(0))
        Z = (X - self.mu) / self.sd; M = np.isnan(Z[:, self.miss]).astype(float); Z = np.nan_to_num(Z, nan=0.0)
        return np.column_stack([Z, M])

    def fit(self, X, y):
        a = self.arch
        if a == 'hgb':
            with threadpool_limits(limits=TH): self.m = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
        elif a == 'xgb':
            import xgboost
            self.m = xgboost.XGBRegressor(n_estimators=400, learning_rate=0.1, max_leaves=63, grow_policy='lossguide', max_depth=0,
                                          min_child_weight=20, reg_lambda=10, max_bin=255, tree_method='hist', device='cuda',
                                          random_state=20261012).fit(X, y)
        elif a == 'ridge':
            from sklearn.linear_model import Ridge
            self.m = Ridge(alpha=1.0).fit(self._prep(X, True), y)
        elif a == 'mlp':
            import torch
            Z = self._prep(X, True); self.ym, self.ys = float(y.mean()), float(y.std() + 1e-9)
            torch.manual_seed(20261117); dev = 'cuda'
            net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3); Xt = torch.tensor(Z, dtype=torch.float32, device=dev)
            yt = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32, device=dev)[:, None]
            g = torch.Generator(device='cpu').manual_seed(20261117)
            for _ in range(20):
                for b in torch.randperm(len(Xt), generator=g).split(1024):
                    opt.zero_grad(); loss = torch.mean((net(Xt[b.to(dev)]) - yt[b.to(dev)]) ** 2); loss.backward(); opt.step()
            self.m = net.eval()
        return self

    def predict(self, X):
        if self.arch in ('hgb', 'xgb'): return self.m.predict(X)
        if self.arch == 'ridge': return self.m.predict(self._prep(X, False))
        import torch
        with torch.no_grad():
            out = self.m(torch.tensor(self._prep(X, False), dtype=torch.float32, device='cuda')).cpu().numpy()[:, 0]
        return out * self.ys + self.ym


def design(F, T, t, k, arch, pooled):
    """Feature matrix of turbine t (k rows); turbine index numeric for trees, one-hot otherwise; none for per-turbine models."""
    if not pooled: return F
    if arch in ('hgb', 'xgb'): return np.column_stack([F, np.full(k, t)])
    oh = np.zeros((k, len(T))); oh[:, T.index(t)] = 1.0; return np.column_stack([F, oh])


def fleet(arch, F, Y, OK, T, month, n, pooled):
    P = {t: np.full(n, np.nan) for t in T}
    for s, e in BC.windows(month, n):
        groups = [T] if pooled else [[t] for t in T]
        for G in groups:
            XX, yy = [], []
            for t in G:
                r = np.arange(7, s - D); r = r[OK[t][r]]; XX.append(design(F[t][r], T, t, len(r), arch, pooled)); yy.append(Y[t][r])
            Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
            if pooled and arch in ('hgb', 'xgb'): keep[-1] = True
            m = Model(arch).fit(Xa[:, keep], np.concatenate(yy))
            for t in G:
                r = np.arange(s, e); r = r[OK[t][r]]
                if len(r): P[t][r] = m.predict(design(F[t][r], T, t, len(r), arch, pooled)[:, keep])
    return P


def correct(P, Y, OK, Dd, hc, n):
    ev = np.arange(EV0, n); PR = {}
    for t in Dd:
        ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        Hh = np.nan_to_num(Dd[t][hc].to_numpy(float)[ev]); Hf = Dd[t][hc].to_numpy(float)[4:EV0]
        Z = (Hh - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL = P[t][ev] + local(Z, r, ok, 8640)
        ok2 = OK[t][ev] & np.isfinite(PL); r2 = np.where(ok2, Y[t][ev] - PL, 0.0)
        PR[t] = np.full(n, np.nan); PR[t][ev] = PL + 0.5 * R.ewma(r2, ok2, 18)
    return PR


def main(ds, arch):
    t0 = time.time(); Dd, Y, OK, Ffull, hc, month, n = RT.data(ds); T = list(Dd); Fb = base_feats(ds, Dd, n); S = {}; info = {}
    for inp, F in [('full', Ffull), ('base', Fb)]:
        for pool in ('fleet', 'pt'):
            t1 = time.time(); P = fleet(arch, F, Y, OK, T, month, n, pool == 'fleet'); PR = correct(P, Y, OK, Dd, hc, n)
            for t in T: S[f'{inp}|{pool}|none_{t}'] = P[t]; S[f'{inp}|{pool}|corr_{t}'] = PR[t]
            m0 = np.mean([np.abs(Y[t] - P[t])[np.arange(n) >= EV0][(OK[t] & np.isfinite(P[t]))[np.arange(n) >= EV0]].mean() for t in T])
            m1 = np.mean([np.abs(Y[t] - PR[t])[np.arange(n) >= EV0][(OK[t] & np.isfinite(PR[t]))[np.arange(n) >= EV0]].mean() for t in T])
            info[f'{inp}|{pool}'] = dict(own_rows_mae_none=float(m0), own_rows_mae_corr=float(m1), seconds=round(time.time() - t1, 1))
            print(ds, arch, inp, pool, round(m0, 4), round(m1, 4), round(time.time() - t1), 's', flush=True)
            if arch == 'hgb' and inp == 'full' and pool == 'fleet':
                info['repro_max_abs_fleet'], info['repro_max_abs_proposed'] = RT.reproduce(ds, Y, OK, P, PR)
                print(ds, 'reproduction', info['repro_max_abs_fleet'], info['repro_max_abs_proposed'], flush=True)
    np.savez_compressed(OUT/f'controlled_{ds}_{arch}.npz', **S)
    (OUT/f'controlled_{ds}_{arch}.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part G', ds=ds, arch=arch, info=info,
        seconds=round(time.time() - t0, 1), code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__': main(sys.argv[1], sys.argv[2])
