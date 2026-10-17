"""PUBLISHED_BASELINES_v001 Amendment 21: matched-input per-turbine control. Same HGB (PARAMS), features, windows, released-row rule and
late-truth terms as the proposed estimator, but one model per turbine fitted on its own released rows (no turbine index).
Writes ptfleet_<ds>.npz and ptrec_<ds>.npz (P_<t>). usage (Python): [HOT_YEAR=2022] python per_turbine_matched.py <ds>"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import os, sys, json
import numpy as np
ds = sys.argv[1]
if ds.startswith('hot'): os.environ['HOT_YEAR'] = ds[3:]
from latetruth import baselines_common as BC
from latetruth import residual_references as R
from latetruth.run_local_late_truth_long import HELP, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
O = BC.OUT; EV0 = BC.EV0; TH = min(JOBS, int(os.environ.get('PT_THREADS', '4')))
if (O/f'ptrec_{ds}.npz').exists(): raise FileExistsError(ds)

if ds.startswith('hot'):
    from latetruth import run_hill_of_towie_replication as H
    D = H.read_all(); n = H.n; month = H.month; Y = {t: df[H.TARGET].to_numpy(float) for t, df in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    F = {t: H.feats(D[t], 'base_x') for t in D}; HV = {t: D[t][H.HELP].to_numpy(float) for t in D}
else:
    from latetruth import run_extra_heat_paths as X
    Dd, n, month, Y, OK = BC.load(ds); rows = np.arange(7, n); F = {}
    for t in Dd:
        f = X.feats(Dd[t][0], rows, 'h30_x'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    HV = {t: Dd[t][0][HELP].to_numpy(float) for t in Dd}; D = Dd

z = np.load(O/f'ours_{ds}.npz', allow_pickle=False)
for t in D: assert np.allclose(z[f'Y_{t}'], Y[t], equal_nan=True) and np.array_equal(z[f'OK_{t}'], OK[t]), t   # same rows as the proposed run
P = {t: np.full(n, np.nan) for t in D}
for s, e in BC.windows(month, n):
    for t in D:
        r = np.arange(7, s - 3); r = r[OK[t][r]]; Xa = F[t][r]
        keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=TH): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], Y[t][r])
        q = np.arange(s, e); q = q[OK[t][q]]
        if len(q): P[t][q] = m.predict(F[t][q][:, keep])
    print(ds, 'window', s, 'done', flush=True)
PR = {}
for t in D:
    ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
    Hh = np.nan_to_num(HV[t][ev]); Hf = HV[t][4:EV0]; Z = (Hh - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
    sim = np.full(n, np.nan); sim[ev] = P[t][ev] + local(Z, r, ok, 8640)
    ok2 = OK[t][ev] & np.isfinite(sim[ev]); r2 = np.where(ok2, Y[t][ev] - sim[ev], 0.0)
    PR[t] = sim.copy(); PR[t][ev] = sim[ev] + 0.5 * R.ewma(r2, ok2, 18)
np.savez_compressed(O/f'ptfleet_{ds}.npz', **{f'P_{t}': P[t] for t in D}); np.savez_compressed(O/f'ptrec_{ds}.npz', **{f'P_{t}': PR[t] for t in D})
s1, s2 = BC.score(Y, P, OK, month, n), BC.score(Y, PR, OK, month, n)
print(ds, 'per-turbine alone', round(s1['all'], 4), '| per-turbine + both terms', round(s2['all'], 4), flush=True)
