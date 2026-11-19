from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR, RAW_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part E: release delays d on Penmanshiel 2019 and Hill of Towie 2019.
Same procedure as baselines_common.ours_sync (Kelmarsh delay study), generalized to the Siemens features.
usage: [HOT_YEAR=2019] python revision_delay.py <ds> <d>"""
import os, sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth import residual_references as R
from latetruth.run_local_late_truth_long import PARAMS, local, HistGradientBoostingRegressor, threadpool_limits

OUT = RAW_DIR/'revision_checks_2026-11-17_v001'; OUT.mkdir(parents=True, exist_ok=True); EV0 = BC.EV0; TH = int(os.environ.get('THREADS', '2'))


def data(ds, d):
    sh = 3 - d
    if ds.startswith('hot'):
        from latetruth import run_hill_of_towie_replication as H
        assert ds == f'hot{H.YEAR}', 'set HOT_YEAR to match ' + ds
        Dd = H.read_all(); n = H.n; month = H.month; Y = {t: df[H.TARGET].to_numpy(float) for t, df in Dd.items()}
        OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in Dd}
        F = {}
        for t, df in Dd.items():
            c = df.copy(); c[H.TARGET] = c[H.TARGET].shift(-sh); F[t] = H.feats(c, 'base_x')
        help_cols = H.HELP
    else:
        from latetruth import run_extra_heat_paths as X
        from latetruth.run_local_late_truth_long import HELP as help_cols, TARGET
        DD, n, month, Y, OK = BC.load(ds); Dd = {t: df for t, (df, _) in DD.items()}; rows = np.arange(7, n); F = {}
        for t, df in Dd.items():
            c = df.copy(); c[TARGET] = c[TARGET].shift(-sh); f = X.feats(c, rows, 'h30_x'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    OK = {t: OK[t] & np.r_[np.zeros(d, bool), np.isfinite(Y[t][:-d])] for t in Dd}   # rule of ours_sync
    return Dd, Y, OK, F, help_cols, month, n


def run(Dd, Y, OK, F, help_cols, month, n, d):
    T = list(Dd); P = {t: np.full(n, np.nan) for t in T}
    for s, e in BC.windows(month, n):
        XX, yy = [], []
        for t in T:
            r = np.arange(7, s - d); r = r[OK[t][r]]; XX.append(np.column_stack([F[t][r], np.full(len(r), t)])); yy.append(Y[t][r])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=TH): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in T:
            r = np.arange(s, e); r = r[OK[t][r]]
            if len(r): P[t][r] = m.predict(np.column_stack([F[t][r], np.full(len(r), t)])[:, keep])
    ev = np.arange(EV0, n); PL, PR = {}, {}; R.D = d
    for t in T:
        ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        Hh = np.nan_to_num(Dd[t][help_cols].to_numpy(float)[ev]); Hf = Dd[t][help_cols].to_numpy(float)[4:EV0]
        Z = (Hh - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL[t] = np.full(n, np.nan); PL[t][ev] = P[t][ev] + local(Z, r, ok, 8640, delay=d)
        ok2 = OK[t][ev] & np.isfinite(PL[t][ev]); r2 = np.where(ok2, Y[t][ev] - PL[t][ev], 0.0)
        PR[t] = np.full(n, np.nan); PR[t][ev] = PL[t][ev] + 0.5 * R.ewma(r2, ok2, 18)
    return P, PL, PR


def main(ds, d):
    t0 = time.time(); Dd, Y, OK, F, hc, month, n = data(ds, d); P, PL, PR = run(Dd, Y, OK, F, hc, month, n, d)
    res = {k: BC.score(Y, M, OK, month, n) for k, M in [('fleet', P), ('similarity', PL), ('proposed', PR)]}
    out = dict(protocol='REVISION_CHECKS_v001 Part E', ds=ds, d=d, scores=res,
               similarity_change_pct=100 * (res['similarity']['all'] / res['fleet']['all'] - 1),
               recency_change_pct=100 * (res['proposed']['all'] / res['similarity']['all'] - 1))
    if d == 3:
        z = np.load(BC.OUT/f'ours_{ds}.npz', allow_pickle=False); zr = np.load(BC.OUT/f'rec_{ds}.npz', allow_pickle=False)
        out['repro_max_abs_fleet'] = max(float(np.nanmax(np.abs(z[f'P_fleet_{t}'] - P[t]))) for t in Y)
        out['repro_max_abs_proposed'] = max(float(np.nanmax(np.abs(zr[f'P_{t}'] - PR[t]))) for t in Y)
        for t in Y: assert np.array_equal(np.isfinite(zr[f'P_{t}']), np.isfinite(PR[t])), f'row mask differs, turbine {t}'
    out['seconds'] = round(time.time() - t0, 1); out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    np.savez_compressed(OUT/f'delay_{ds}_d{d}.npz', **{f'{k}_{t}': M[t] for t in Y for k, M in [('P_fleet', P), ('P_sim', PL), ('P_prop', PR)]})
    (OUT/f'delay_{ds}_d{d}.json').write_text(json.dumps(out, indent=1))
    print(ds, 'd', d, {k: round(v['all'], 4) for k, v in res.items()}, 'sim %', round(out['similarity_change_pct'], 2),
          'rec %', round(out['recency_change_pct'], 2), {k: out[k] for k in out if k.startswith('repro')}, flush=True)


if __name__ == '__main__': main(sys.argv[1], int(sys.argv[2]))
