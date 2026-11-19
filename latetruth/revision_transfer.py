from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part F: transfer to a wind farm never seen by the model, without the initial adaptation.
usage: python revision_transfer.py <src> <tgt>     (kelmarsh2017 penmanshiel2019 | penmanshiel2019 kelmarsh2018)"""
import os, sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth import residual_references as R
from latetruth import revision_turbine_id as RT
from latetruth.run_local_late_truth_long import PARAMS, local, HistGradientBoostingRegressor, threadpool_limits

OUT = RT.OUT; EV0 = BC.EV0; D = 3; TH = int(os.environ.get('THREADS', '2'))


def rows_xy(F, Y, OK, T, hi):
    XX, yy = [], []
    for t in T:
        r = np.arange(7, hi[t] if isinstance(hi, dict) else hi); r = r[OK[t][r]]; XX.append(F[t][r]); yy.append(Y[t][r])
    return np.vstack(XX), np.concatenate(yy)


def fit(X, y):
    keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in X.T])
    with threadpool_limits(limits=TH): m = HistGradientBoostingRegressor(**PARAMS).fit(X[:, keep], y)
    return m, keep


def correct(P, Y, OK, Dd, hc, n, mean, std):
    """Similarity + recency terms from day 1 (row 7) on the released residuals of the target farm; source standardization."""
    ev = np.arange(7, n); PL, PR = {}, {}
    for t in Dd:
        ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        Z = (np.nan_to_num(Dd[t][hc].to_numpy(float)[ev]) - mean) / (std + 1e-9)
        PL[t] = np.full(n, np.nan); PL[t][ev] = P[t][ev] + local(Z, r, ok, 8640)
        ok2 = OK[t][ev] & np.isfinite(PL[t][ev]); r2 = np.where(ok2, Y[t][ev] - PL[t][ev], 0.0)
        PR[t] = np.full(n, np.nan); PR[t][ev] = PL[t][ev] + 0.5 * R.ewma(r2, ok2, 18)
    return PR


def mae(Y, P, OK, T, lo, hi):
    v = []
    for t in T:
        r = np.arange(lo, hi); ok = OK[t][r] & np.isfinite(P[t][r]); v.append(float(np.abs(Y[t][r] - P[t][r])[ok].mean()))
    return float(np.mean(v))


def wins(Y, A, B, OK, T, month, n):
    """Turbine-months (rows >= EV0) in which A has the lower MAE than B, on rows where both are finite."""
    ev = np.arange(EV0, n); w = k = 0
    for t in T:
        ok = OK[t][ev] & np.isfinite(A[t][ev]) & np.isfinite(B[t][ev])
        for m in range(2, 13):
            s = ok & (month[ev] == m)
            if s.sum() > 100: k += 1; w += int(np.abs(Y[t][ev] - A[t][ev])[s].mean() < np.abs(Y[t][ev] - B[t][ev])[s].mean())
    return f'{w}/{k}'


def main(src, tgt):
    t0 = time.time()
    Ds, Ys, OKs, Fs, hc, ms, ns = RT.data(src); Dt, Yt, OKt, Ft, _, mt, nt = RT.data(tgt); Ts, Tt = list(Ds), list(Dt)
    Hs = np.vstack([Ds[t][hc].to_numpy(float)[4:EV0] for t in Ts]); mean, std = np.nanmean(Hs, 0), np.nanstd(Hs, 0)
    Xs, ys = rows_xy(Fs, Ys, OKs, Ts, ns); m0, k0 = fit(Xs, ys)
    pred = lambda m, k, t, r: m.predict(Ft[t][r][:, k])
    PA = {t: np.full(nt, np.nan) for t in Tt}
    for t in Tt:
        r = np.arange(7, nt); r = r[OKt[t][r]]; PA[t][r] = pred(m0, k0, t, r)
    PB = correct(PA, Yt, OKt, Dt, hc, nt, mean, std)
    PC0 = {t: np.full(nt, np.nan) for t in Tt}
    for t in Tt:
        r = np.arange(7, EV0); r = r[OKt[t][r]]; PC0[t][r] = pred(m0, k0, t, r)
    for s, e in BC.windows(mt, nt):
        Xt_, yt_ = rows_xy(Ft, Yt, OKt, Tt, s - D); m, k = fit(np.vstack([Xs, Xt_]), np.concatenate([ys, yt_]))
        for t in Tt:
            r = np.arange(s, e); r = r[OKt[t][r]]
            if len(r): PC0[t][r] = pred(m, k, t, r)
    PC = correct(PC0, Yt, OKt, Dt, hc, nt, mean, std)
    _, PN = RT.run(tgt, Dt, Yt, OKt, Ft, hc, mt, nt, mode='none')
    PO = {t: np.load(BC.OUT/f'rec_{tgt}.npz', allow_pickle=False)[f'P_{t}'] for t in Tt}
    PP = {t: np.r_[np.full(D, np.nan), Yt[t][:-D]] for t in Tt}
    V = {'a_zero_shot': PA, 'b_source_plus_correction': PB, 'c_start_day1': PC, 'noid': PN, 'orig': PO, 'persistence': PP}
    res = {k: dict(days_1_28=(mae(Yt, P, OKt, Tt, 7, EV0) if k not in ('noid', 'orig') else None), from_ev0=mae(Yt, P, OKt, Tt, EV0, nt)) for k, P in V.items()}
    out = dict(protocol='REVISION_CHECKS_v001 Part F', src=src, tgt=tgt, scores=res,
               wins={f'{a}_vs_{b}': wins(Yt, V[a], V[b], OKt, Tt, mt, nt) for a in ('b_source_plus_correction', 'c_start_day1') for b in ('noid', 'orig')},
               seconds=round(time.time() - t0, 1), code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest())
    (OUT/f'transfer_{src}_to_{tgt}.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT/f'transfer_{src}_to_{tgt}.npz', **{f'{k}_{t}': P[t] for k, P in V.items() if k in ('a_zero_shot', 'b_source_plus_correction', 'c_start_day1', 'noid') for t in Tt})
    print(src, '->', tgt, json.dumps(res), out['wins'], flush=True)


if __name__ == '__main__': main(sys.argv[1], sys.argv[2])
