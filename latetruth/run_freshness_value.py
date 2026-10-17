"""Protocol 05_Protocols/FRESHNESS_VALUE_v001.md: leave-one-sensor-stale (3 steps) value of freshness."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib
import numpy as np
from latetruth import run_extra_heat_paths as X
from latetruth import run_h29_h30_physics as B
from latetruth.run_local_late_truth_long import ROOT, TARGET, HELP, PARAMS, HistGradientBoostingRegressor, threadpool_limits
OUT = OUT_DIR/'freshness_value_2026-10-14_v001'; EV0 = 4032
SENSORS = HELP + X.EXTRA


def stale_df(df, s):
    d = df.copy()
    if s is not None: d[s] = d[s].shift(3)   # value at t becomes value at t-3 (and its changes end at t-3)
    return d


def main():
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    OUT.mkdir(parents=True, exist_ok=True); D = {t: X.read(t) for t in range(1, 7)}; n = len(D[1][0]); month = D[1][1]
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    starts = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]; out = {}
    for s in [None] + SENSORS:
        F = {}
        for t, (df, _) in D.items():
            d = stale_df(df, s); f = X.feats(d, np.arange(7, n), 'h30_x'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
        P = {t: np.full(n, np.nan) for t in D}
        for j, st in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else n; XX, yy = [], []
            for t in D:
                rows = np.arange(7, st - 3); rows = rows[OK[t][rows]]; XX.append(np.column_stack([F[t][rows], np.full(len(rows), t)])); yy.append(Y[t][rows])
            Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
            with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
            for t in D:
                rows = np.arange(st, e); rows = rows[OK[t][rows]]; P[t][rows] = m.predict(np.column_stack([F[t][rows], np.full(len(rows), t)])[:, keep])
        a, tr = [], []
        for t in D:
            ev = np.arange(EV0, n); ok = OK[t][ev]; err = np.abs(Y[t][ev] - P[t][ev]); move = np.abs(Y[t][ev] - Y[t][ev - 3]); q90 = np.quantile(move[ok], .9)
            a.append(err[ok].mean()); tr.append(err[ok & (move >= q90)].mean())
        key = 'all_fresh' if s is None else s; out[key] = dict(all=float(np.mean(a)), transient=float(np.mean(tr)))
        base = out['all_fresh']
        print(f"{key[:40]:40s} all {out[key]['all']:.4f} (+{100*(out[key]['all']/base['all']-1):.1f}%) transient {out[key]['transient']:.4f} (+{100*(out[key]['transient']/base['transient']-1):.1f}%)", flush=True)
    (OUT/'results.json').write_text(json.dumps({'results': out, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
