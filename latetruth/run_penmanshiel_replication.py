"""Protocol protocols/PENMANSHIEL_REPLICATION_v001.md — reuses the Kelmarsh feature code unchanged."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
from latetruth.data import require_archive
import sys
import io, csv, zipfile, json, hashlib, sys
import numpy as np, pandas as pd
from latetruth import run_extra_heat_paths as X
from latetruth import run_h29_h30_physics as B
from latetruth import run_send_schedule as SS
from latetruth.run_local_late_truth_long import ROOT, TARGET, HELP, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
DIR = DATA_DIR/'Penmanshiel/Version 0.0.2'
OUT = OUT_DIR/'penmanshiel_replication_2019_2026-10-14_v001'; EV0 = 4032
TURB = [1, 2] + list(range(4, 16))


def read(t):
    zp = DIR/('Penmanshiel_SCADA_2019_WT01-10_3112.zip' if t <= 10 else 'Penmanshiel_SCADA_2019_WT11-15_3117.zip')
    with zipfile.ZipFile(require_archive(zp)) as z:
        name = next(n for n in z.namelist() if n.startswith(f'Turbine_Data_Penmanshiel_{t:02d}_'))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            df = pd.read_csv(f, names=h, header=None, usecols=[h[0], TARGET, B.VANE] + HELP + X.EXTRA)
    return df, pd.to_datetime(df[h[0]], utc=True).dt.month.to_numpy()


def run(D, featfn, label, use_local, n, month, OK, Y):
    starts = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]
    F = {t: featfn(t) for t in D}; P = {t: np.full(n, np.nan) for t in D}
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else n; XX, yy = [], []
        for t in D:
            rows = np.arange(7, s - 3); rows = rows[OK[t][rows]]; XX.append(np.column_stack([F[t][rows], np.full(len(rows), t)])); yy.append(Y[t][rows])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in D:
            rows = np.arange(s, e); rows = rows[OK[t][rows]]; P[t][rows] = m.predict(np.column_stack([F[t][rows], np.full(len(rows), t)])[:, keep])
    out = {}
    for lab in [label] + ([label + '+local'] if use_local else []):
        a, ca, tr, mo = [], [], [], []
        for t, (df, mon) in D.items():
            ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
            if lab.endswith('+local'):
                H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:EV0]
                Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9); r = r - local(Z, r, ok, 8640)
            err = np.abs(r); mv = np.abs(Y[t][ev] - Y[t][ev - 3]); q50, q90 = np.quantile(mv[ok], [.5, .9])
            a.append(err[ok].mean()); ca.append(err[ok & (mv < q50)].mean()); tr.append(err[ok & (mv >= q90)].mean())
            mo.append([err[ok & (mon[ev] == m_)].mean() for m_ in range(2, 13)])
        out[lab] = dict(all=float(np.mean(a)), calm=float(np.mean(ca)), transient=float(np.mean(tr)), months=np.nanmean(mo, 0).round(4).tolist())
        print(f"{lab:28s} all {out[lab]['all']:.4f} calm {out[lab]['calm']:.4f} transient {out[lab]['transient']:.4f}", flush=True)
    return out


def pad(f): return np.vstack([np.full((7, f.shape[1]), np.nan), f])


def main():
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    OUT.mkdir(parents=True, exist_ok=True); D = {t: read(t) for t in TURB}; n = len(D[1][0]); month = D[1][1]
    assert all(len(D[t][0]) == n for t in D)
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    res = {}
    for kind in ['base', 'h29', 'h30', 'h30_x']:
        fn = (lambda t, k=kind: pad(X.feats(D[t][0], np.arange(7, n), k)) if k.endswith('_x') else pad(B.feats(D[t][0], np.arange(7, n), k)))
        res.update(run(D, fn, kind, True, n, month, OK, Y))
    for s in HELP + X.EXTRA:   # R4 single-stale
        def fn(t, s=s):
            d = D[t][0].copy(); d[s] = d[s].shift(3); return pad(X.feats(d, np.arange(7, n), 'h30_x'))
        res.update(run(D, fn, 'stale:' + s, False, n, month, OK, Y))
    def fn_both(t):
        d = D[t][0].copy()
        for s in [SS.INL] + SS.BEAR: d[s] = d[s].shift(3)
        return pad(X.feats(d, np.arange(7, n), 'h30_x'))
    res.update(run(D, fn_both, 'stale:inlet+bearings', False, n, month, OK, Y))
    for pol in ['always_bearing', 'always_inlet']:   # R5
        def fn(t, pol=pol):
            d, ages = SS.apply_policy(D[t][0], pol); return pad(np.column_stack([X.feats(d, np.arange(7, n), 'h30_x'), ages[7:]]))
        res.update(run(D, fn, 'policy:' + pol, False, n, month, OK, Y))
    (OUT/'results.json').write_text(json.dumps({'results': res, 'turbines': TURB, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
