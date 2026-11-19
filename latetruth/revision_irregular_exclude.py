from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part H, Amendment H-b: residuals of rows with missing inputs excluded (exploratory).
usage: [HOT_YEAR=2019] python revision_irregular_exclude.py <ds> <scenario 4|5>"""
import sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth.run_local_late_truth_long import PARAMS, HistGradientBoostingRegressor, threadpool_limits
from latetruth.revision_irregular import OUT, EV0, TH, K, release, setup, features, local_g, ewma_g


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
    ev = np.arange(EV0, n); PR, PE, PP, PRX = {}, {}, {}, {}
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
        okx = ok & ~INM[t][ev]; PLx = P[t][ev] + local_g(Z, r, okx, 8640, Srel)
        ok2x = OK[t][ev] & np.isfinite(PLx) & ~INM[t][ev]; r2x = np.where(ok2x, Y[t][ev] - PLx, 0.0)
        PRX[t] = np.full(n, np.nan); PRX[t][ev] = PLx + 0.5 * ewma_g(r2x, ok2x, 18, Srel)
    PFB = {t: np.where(INM[t], PP[t], P[t]) for t in T}; PRB = {t: np.where(INM[t], PP[t], PR[t]) for t in T}
    PRXB = {t: np.where(INM[t], PP[t], PRX[t]) for t in T}
    M = {'fleet': P, 'proposed': PR, 'persistence': PP, 'fleet_fallback': PFB, 'proposed_fallback': PRB, 'proposed_excl': PRX, 'proposed_excl_fallback': PRXB}

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
    out = dict(protocol='REVISION_CHECKS_v001 Part H, Amendment H-b', ds=ds, scenario=sc, scores=res, share=share)
    if sc == 0:
        z = np.load(BC.OUT/f'ours_{ds}.npz', allow_pickle=False); zr = np.load(BC.OUT/f'rec_{ds}.npz', allow_pickle=False)
        out['repro_max_abs_fleet'] = max(float(np.nanmax(np.abs(z[f'P_fleet_{t}'] - P[t]))) for t in T)
        out['repro_max_abs_proposed'] = max(float(np.nanmax(np.abs(zr[f'P_{t}'] - PR[t]))) for t in T)
        for t in T: assert np.array_equal(np.isfinite(zr[f'P_{t}']), np.isfinite(PR[t])), f'row mask differs, turbine {t}'
    out['seconds'] = round(time.time() - t0, 1); out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (OUT/f'irregular_exclude_{ds}_s{sc}.json').write_text(json.dumps(out, indent=1))
    print(ds, 's', sc, {k: {s: (round(v, 4) if v is not None else None) for s, v in d.items() if s in ('all', 'inside_blocks', 'outside_blocks')} for k, d in res.items()},
          share, {k: out[k] for k in out if k.startswith('repro')}, flush=True)


if __name__ == '__main__': main(sys.argv[1], int(sys.argv[2]))
