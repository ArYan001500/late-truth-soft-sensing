"""B6 of protocols/PUBLISHED_BASELINES_v001.md — Kim, Okajima, Kano, Hasebe 2013 (Chemom. Intell. Lab. Syst. 124, 43-49):
just-in-time LW-PLS with the adaptive weighted-Euclidean similarity (their eqs. 3-16, 20-23).
Database for a query at decision time t = released rows of the same turbine in [t-3-8640, t-3) (same memory as our local correction),
so this is the JIT competitor that also learns from late truth. Inputs: the 16 raw channels, autoscaled on the training window, plus the
latest released target y(t-lag) when lag is given ('lag' variants, as for the other baselines).
Offline part (theta_m = V_m^gamma, I = 5 iterations) on 1000 random released rows before each monthly start (N LW-PLS models on the full
database are infeasible at 10-min rates; documented deviation). Online part J = 3 iterations, theta = [V^gamma + theta_prev] / 2.
Hyper-parameters R, phi, gamma chosen on Kelmarsh 2017 March (every 6th row), then frozen.  usage: python baselines_kim.py <lag 3|1|0>
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib, itertools
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
MEM, DELAY = 8640, 3


def lwpls(X, y, om, xq, R, coef=False):
    """Single-output LW-PLS (Kim et al. eqs. 4-16). X: N x M, y: N, om: N similarities, xq: M."""
    s = om.sum(); xm = om @ X / s; ym = om @ y / s
    Xr, yr, xr = X - xm, y - ym, xq - xm; yhat = ym; W, P, Q = [], [], []
    for _ in range(R):
        w = Xr.T @ (om * yr); nw = np.linalg.norm(w)
        if nw < 1e-12: break
        w /= nw; t = Xr @ w; tt = (om * t) @ t
        if tt < 1e-12: break
        p = Xr.T @ (om * t) / tt; q = (om * yr) @ t / tt; tq = xr @ w
        yhat += q * tq; W.append(w); P.append(p); Q.append(q)
        Xr = Xr - np.outer(t, p); yr = yr - t * q; xr = xr - tq * p
    if not coef: return yhat
    if not W: return yhat, np.zeros(X.shape[1])
    W, P = np.array(W).T, np.array(P).T
    return yhat, W @ np.linalg.solve(P.T @ W, np.array(Q))


def sim(Xd, xq, theta, phi):
    d = np.sqrt(((Xd - xq) ** 2) @ theta); sd = d.std() + 1e-12
    return np.exp(-phi * d / sd)


def offline(Xd, yd, R, phi, gamma, rng, n=1000, I=5):
    idx = rng.choice(len(Xd), min(n, len(Xd)), replace=False); Xo, yo = Xd[idx], yd[idx]; th = np.ones(Xd.shape[1]); A = None
    for _ in range(I):
        A = np.array([lwpls(Xo, yo, sim(Xo, Xo[k], th, phi), Xo[k], R, coef=True)[1] for k in range(len(Xo))])
        new = np.var(A, 0) ** gamma if gamma > 0 else np.ones_like(th)
        new = new / (new.mean() + 1e-12)
        if np.all(np.abs(new - th) <= 1e-3 * np.abs(th) + 1e-12): th = new; break
        th = new
    return Xo, A, th


def online_theta(Xo, A, th0, xq, phi, gamma, J=3):
    th = th0.copy()
    if gamma == 0: return th
    for _ in range(J):
        om = sim(Xo, xq, th, phi); am = om @ A / om.sum(); V = om @ (A - am) ** 2 / om.sum()
        new = (V ** gamma / ((V ** gamma).mean() + 1e-12) + th) / 2
        if np.all(np.abs(new - th) <= 1e-3 * np.abs(th) + 1e-12): return new
        th = new
    return th


def design(X, y, lag):
    return np.column_stack([X, y[np.arange(len(y)) - lag]]) if lag else X.copy()


def run(hp, lag, ds, val=False, step=1):
    month, X, Y, OK = C.load(ds); n = len(month); P = {t: np.full(n, np.nan) for t in Y}; rng = np.random.default_rng(0)
    wins = [(int(np.flatnonzero(month == 3)[0]), int(np.flatnonzero(month == 4)[0]))] if val else C.windows(month, n)
    for t in Y:
        Z0 = design(X[t], Y[t], lag)
        for s, e in wins:
            tr0 = np.arange(7, s - DELAY); tr0 = tr0[OK[t][tr0] & np.isfinite(Y[t][tr0])]
            keep = (np.isfinite(Z0[tr0]).mean(0) >= 0.95) & (np.nanstd(Z0[tr0], 0) > 0)   # channels present in this training window
            Z = Z0[:, keep]; rel = OK[t] & np.isfinite(Y[t]) & np.isfinite(Z).all(1)
            tr = tr0[rel[tr0]]; mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9; Zs = (Z - mu) / sd
            Xo, A, th0 = offline(Zs[tr[-MEM:]], Y[t][tr[-MEM:]], hp['R'], hp['phi'], hp['gamma'], rng)
            for q in range(s, e, step):
                if not (OK[t][q] and np.isfinite(Zs[q]).all()): continue
                db = np.arange(max(7, q - DELAY - MEM), q - DELAY); db = db[rel[db]]
                if len(db) < 50: continue
                th = online_theta(Xo, A, th0, Zs[q], hp['phi'], hp['gamma'])
                P[t][q] = lwpls(Zs[db], Y[t][db], sim(Zs[db], Zs[q], th, hp['phi']), Zs[q], hp['R'])
    if val:
        a, b = wins[0]; ev = np.arange(a, b)
        return float(np.mean([np.abs(Y[t][ev] - P[t][ev])[OK[t][ev] & np.isfinite(P[t][ev])].mean() for t in Y]))
    return C.score(Y, P, OK, month), P


GRID = [{'R': R, 'phi': phi, 'gamma': g} for R, phi, g in itertools.product((2, 4, 6), (0.5, 1.0, 2.0, 4.0), (0.0, 0.5, 1.0))]

if __name__ == '__main__':
    lag = int(sys.argv[1]); tag = f'B6_lag{lag}'; f = C.OUT/f'{tag}.json'
    if f.exists(): raise FileExistsError(f)
    v = []
    for hp in GRID:
        v.append(run(hp, lag, 'kelmarsh2017', val=True, step=6)); print('tune', hp, round(v[-1], 4), flush=True)
    hp = GRID[int(np.argmin(v))]; res = {'hp': hp, 'val_mae_2017Mar_every6': float(min(v))}; print('chosen', hp, flush=True)
    for ds in ['kelmarsh2017', 'kelmarsh2018']:
        fz = C.OUT/f'{tag}_{ds}.npz'
        if fz.exists():
            month, X, Y, OK = C.load(ds); z = np.load(fz, allow_pickle=False); P = {t: z[f'P_{t}'] for t in Y}; sc = C.score(Y, P, OK, month)
        else:
            sc, P = run(hp, lag, ds); np.savez_compressed(fz, **{f'P_{t}': p for t, p in P.items()})
        res[ds] = sc; print(tag, ds, round(sc['all'], 4), flush=True)
    res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); f.write_text(json.dumps(res, indent=2))
