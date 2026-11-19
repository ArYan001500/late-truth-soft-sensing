from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR, RAW_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part H: irregular delays, target outages, input gaps, and link outages.
usage: [HOT_YEAR=2019] python revision_irregular.py <ds> <scenario 0-5>"""
import os, sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth.run_local_late_truth_long import PARAMS, HistGradientBoostingRegressor, threadpool_limits

OUT = RAW_DIR/'revision_checks_2026-11-17_v001'; OUT.mkdir(parents=True, exist_ok=True); EV0 = BC.EV0; TH = int(os.environ.get('THREADS', '2')); K = 20


def blocks(rng, n, cover, lo, hi):
    gap = (lo + hi) / 2 * (1 - cover) / cover; m = np.zeros(n, bool); t = int(rng.exponential(gap))
    while t < n:
        e = min(n, t + int(rng.integers(lo, hi + 1))); m[t:e] = True; t = e + 1 + int(rng.exponential(gap))
    return m


def runs(m):
    """(start, end) of the True runs of m."""
    d = np.diff(np.r_[0, m.astype(int), 0]); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def release(sc, n, rng):
    s = np.arange(n); inmask = np.zeros(n, bool); blk = np.zeros(n, bool)
    if sc == 0: rel = s + 3
    elif sc == 1: rel = np.maximum.accumulate(s + rng.integers(1, 7, n))
    elif sc == 2: rel = 24 * ((s + 24) // 24)
    elif sc in (3, 5):
        rel = s + 3 if sc == 3 else np.maximum.accumulate(s + rng.integers(1, 7, n))
        blk = blocks(rng, n, 0.10, 36, 432)
        for b, e in runs(blk): rel[(rel >= b) & (rel < e)] = e
        rel = np.maximum.accumulate(rel)
        if sc == 5: inmask = blk
    elif sc == 4:
        rel = s + 3; inmask = blocks(rng, n, 0.05, 6, 36); blk = inmask
    assert np.all(rel >= s + 1) and np.all(np.diff(rel) >= 0)
    return rel, inmask, blk


def setup(ds):
    if ds.startswith('hot'):
        from latetruth import run_hill_of_towie_replication as H
        assert ds == f'hot{H.YEAR}', 'set HOT_YEAR to match ' + ds
        Dd = H.read_all(); n = H.n; month = H.month; tcol = H.TARGET; hc = H.HELP
        ff = lambda df: H.feats(df, 'base_x')
    else:
        from latetruth import run_extra_heat_paths as X
        from latetruth.run_local_late_truth_long import HELP as hc, TARGET as tcol
        DD, n, month, _, _ = BC.load(ds); Dd = {t: df for t, (df, _) in DD.items()}; rows = np.arange(7, n)
        ff = lambda df: (lambda f: np.vstack([np.full((7, f.shape[1]), np.nan), f]))(X.feats(df, rows, 'h30_x'))
    return Dd, n, month, tcol, hc, ff


def features(df, tcol, ff, age, rowsel, inmask):
    """F_full (inputs complete, target lags at the decision-time age) and F_pred (inputs masked where unavailable)."""
    n = len(df); Ff = Fp = None; inp = [c for c in df.columns if c != tcol]
    for a in np.unique(age[rowsel]):
        for msk in ([False, True] if inmask.any() else [False]):
            sel = rowsel & (age == a) & (inmask == msk) if msk else rowsel & (age == a)
            if not sel.any(): continue
            d = df.copy(); d[tcol] = d[tcol].shift(-(3 - int(a)))
            if msk: d.loc[inmask, inp] = np.nan
            f = ff(d)
            if Ff is None: Ff = np.full(f.shape, np.nan); Fp = np.full(f.shape, np.nan)
            if not msk: Ff[sel] = f[sel]; Fp[sel] = f[sel]
            else: Fp[sel] = f[sel]
    return Ff, Fp


def local_g(Z, r, ok, mem, Srel, k=20, k0=20):
    out = np.zeros(len(r)); idx = np.flatnonzero(ok)
    for i in range(len(r)):
        hi = Srel[i]
        if hi <= 0: continue
        a, b = np.searchsorted(idx, max(hi - mem, 0)), np.searchsorted(idx, hi); cand = idx[a:b]
        if len(cand) <= k: continue
        d = np.sum((Z[cand] - Z[i]) ** 2, 1); nn = cand[np.argpartition(d, k)[:k]]
        out[i] = r[nn].mean() * k / (k + k0)
    return out


def ewma_g(r, ok, h, Srel):
    idx = np.flatnonzero(ok); out = np.zeros(len(r))
    if len(idx) == 0: return out
    lam = 0.5 ** (1.0 / h); E = np.empty(len(idx)); S = W = 0.0; prev = idx[0]
    for j, s in enumerate(idx):
        dec = lam ** (s - prev); S = S * dec + r[s]; W = W * dec + 1.0; E[j] = S / W; prev = s
    pos = np.searchsorted(idx, Srel); m = pos > K; out[m] = E[pos[m] - 1]
    return out


def main(ds, sc):
    t0 = time.time(); Dd, n, month, tcol, hc, ff = setup(ds); T = list(Dd); tt = np.arange(n)
    Y, OK, F, Fp, REL, INM, BLK, AGE, S1 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for t in T:
        rng = np.random.default_rng(20261118 + 1000 * sc + t); rel, inm, blk = release(sc, n, rng)
        y = Dd[t][tcol].to_numpy(float); s1 = np.searchsorted(rel, tt, 'right') - 1; age = tt - s1
        ok = (tt >= 7) & (s1 >= 1) & np.isfinite(y)
        ok[ok] &= np.isfinite(y[s1[ok]]) & np.isfinite(y[s1[ok] - 1])
        F[t], Fp[t] = features(Dd[t], tcol, ff, age, (tt >= 7) & (s1 >= 1), inm)
        Y[t], OK[t], REL[t], INM[t], BLK[t], AGE[t], S1[t] = y, ok, rel, inm, blk, age, s1
    P = {t: np.full(n, np.nan) for t in T}
    for s0, e in BC.windows(month, n):
        XX, yy = [], []
        for t in T:
            r = np.arange(7, np.searchsorted(REL[t], s0, 'left')); r = r[OK[t][r]]
            XX.append(np.column_stack([F[t][r], np.full(len(r), t)])); yy.append(Y[t][r])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T]); keep[-1] = True
        with threadpool_limits(limits=TH): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in T:
            r = np.arange(s0, e); r = r[OK[t][r]]
            if len(r): P[t][r] = m.predict(np.column_stack([Fp[t][r], np.full(len(r), t)])[:, keep])
    ev = np.arange(EV0, n); PR, PE, PP = {}, {}, {}
    for t in T:
        Srel = np.searchsorted(REL[t], ev, 'left') - EV0
        ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        Hraw = Dd[t][hc].to_numpy(float).copy(); Hraw[INM[t]] = np.nan
        Hh = np.nan_to_num(Hraw[ev]); Hf = Dd[t][hc].to_numpy(float)[4:EV0]
        Z = (Hh - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL = P[t][ev] + local_g(Z, r, ok, 8640, Srel)
        ok2 = OK[t][ev] & np.isfinite(PL); r2 = np.where(ok2, Y[t][ev] - PL, 0.0)
        PR[t] = np.full(n, np.nan); PR[t][ev] = PL + 0.5 * ewma_g(r2, ok2, 18, Srel)
        PE[t] = np.full(n, np.nan); PE[t][ev] = P[t][ev] + 0.75 * ewma_g(r, ok, 18, Srel)
        PP[t] = np.full(n, np.nan); PP[t][ev] = np.where(S1[t][ev] >= 0, Y[t][np.maximum(S1[t][ev], 0)], np.nan)
    M = {'fleet': P, 'proposed': PR, 'ewma_reference': PE, 'persistence': PP}

    def mae(Pm, sel):
        v = []
        for t in T:
            k = OK[t][ev] & np.isfinite(Pm[t][ev]) & sel(t)
            if k.sum(): v.append(float(np.abs(Y[t][ev] - Pm[t][ev])[k].mean()))
        return float(np.mean(v)) if v else None
    sets = {'all': lambda t: np.ones(len(ev), bool), 'inside_blocks': lambda t: BLK[t][ev], 'outside_blocks': lambda t: ~BLK[t][ev]}
    for lo, hi in [(1, 3), (4, 6), (7, 12), (13, 24), (25, 10 ** 6)]:
        sets[f'age_{lo}_{hi}'] = (lambda lo, hi: lambda t: (AGE[t][ev] >= lo) & (AGE[t][ev] <= hi))(lo, hi)
    res = {nm: {s: mae(Pm, f) for s, f in sets.items()} for nm, Pm in M.items()}
    share = dict(rows_in_blocks=float(np.mean([BLK[t][ev].mean() for t in T])), rows_inputs_missing=float(np.mean([INM[t][ev].mean() for t in T])),
                 mean_age_rows=float(np.mean([AGE[t][ev][OK[t][ev]].mean() for t in T])))
    out = dict(protocol='REVISION_CHECKS_v001 Part H', ds=ds, scenario=sc, scores=res, share=share)
    if sc == 0:
        z = np.load(BC.OUT/f'ours_{ds}.npz', allow_pickle=False); zr = np.load(BC.OUT/f'rec_{ds}.npz', allow_pickle=False)
        out['repro_max_abs_fleet'] = max(float(np.nanmax(np.abs(z[f'P_fleet_{t}'] - P[t]))) for t in T)
        out['repro_max_abs_proposed'] = max(float(np.nanmax(np.abs(zr[f'P_{t}'] - PR[t]))) for t in T)
        for t in T: assert np.array_equal(np.isfinite(zr[f'P_{t}']), np.isfinite(PR[t])), f'row mask differs, turbine {t}'
    out['seconds'] = round(time.time() - t0, 1); out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (OUT/f'irregular_{ds}_s{sc}.json').write_text(json.dumps(out, indent=1))
    print(ds, 's', sc, {k: {s: (round(v, 4) if v is not None else None) for s, v in d.items() if s in ('all', 'inside_blocks', 'outside_blocks')} for k, d in res.items()},
          share, {k: out[k] for k in out if k.startswith('repro')}, flush=True)


if __name__ == '__main__': main(sys.argv[1], int(sys.argv[2]))
