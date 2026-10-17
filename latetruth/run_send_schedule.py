"""Protocol protocols/SEND_SCHEDULE_v001.md."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib
import numpy as np
from latetruth import run_extra_heat_paths as X
from latetruth.run_local_late_truth_long import ROOT, TARGET, PARAMS, HistGradientBoostingRegressor, threadpool_limits
OUT = OUT_DIR/'send_schedule_2026-10-14_v001'; EV0 = 4032
INL = 'Gear oil inlet temperature (°C)'; BEAR = ['Rear bearing temperature (°C)', 'Front bearing temperature (°C)']


def apply_policy(df, policy):
    n = len(df); I = df[INL].to_numpy(float); Bm = df[BEAR].to_numpy(float)
    if policy == 'always_bearing': fresh_b = np.ones(n, bool)
    elif policy == 'always_inlet': fresh_b = np.zeros(n, bool)
    elif policy == 'alternate': fresh_b = (np.arange(n) // 3) % 2 == 0
    else:
        sd_i = np.nanstd(np.diff(I[:EV0])); sd_b = np.nanstd(np.diff(Bm[:EV0, 0]))
        ci = np.abs(np.r_[0, np.diff(I)]) / sd_i; cb = np.abs(np.r_[0, np.diff(Bm[:, 0])]) / sd_b
        fresh_b = np.nan_to_num(cb) >= np.nan_to_num(ci)
    # latest available value: fresh at t if chosen; otherwise value of the most recent step s<=t with (fresh at s) or s<=t-3
    def latest(v, fresh):
        out = np.full(n, np.nan); age = np.full(n, 3.0)
        for t in range(n):
            best = t - 3;
            for s in range(t, t - 3, -1):
                if s >= 0 and fresh[s]: best = s; break
            if best >= 0: out[t] = v[best]; age[t] = t - best
        return out, age
    d = df.copy(); bi, inlet_age = latest(I, ~fresh_b); d[INL] = bi; ages = [inlet_age]
    for k, b in enumerate(BEAR):
        bv, ab = latest(Bm[:, k], fresh_b); d[b] = bv
        if k == 0: ages.append(ab)
    return d, np.column_stack(ages)


def main():
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    OUT.mkdir(parents=True, exist_ok=True); D = {t: X.read(t) for t in range(1, 7)}; n = len(D[1][0]); month = D[1][1]
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    starts = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]; out = {}
    for pol in ['always_bearing', 'always_inlet', 'alternate', 'event']:
        F = {}; share = []
        for t, (df, _) in D.items():
            d, ages = apply_policy(df, pol); f = np.column_stack([X.feats(d, np.arange(7, n), 'h30_x'), ages[7:]])
            F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f]); share.append(float(np.mean(ages[EV0:, 1] == 0)))
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
            ev = np.arange(EV0, n); ok = OK[t][ev]; err = np.abs(Y[t][ev] - P[t][ev]); mv = np.abs(Y[t][ev] - Y[t][ev - 3]); q = np.quantile(mv[ok], .9)
            a.append(err[ok].mean()); tr.append(err[ok & (mv >= q)].mean())
        out[pol] = dict(all=float(np.mean(a)), transient=float(np.mean(tr)), bearing_fresh_share=float(np.mean(share)))
        print(pol, {k: round(v, 4) for k, v in out[pol].items()}, flush=True)
    (OUT/'results.json').write_text(json.dumps({'results': out, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
