"""Protocol 05_Protocols/PUBLISHED_BASELINES_v001.md — shared data, schedule, metric, and saved predictions of our pipeline.

`python baselines_common.py kelmarsh2017|kelmarsh2018` writes 06_Experiments/published_baselines_2026-10-14_v001/ours_<ds>.npz
with, per turbine: Y, OK, month, P_fleet (h30_x fleet HGB, no local), P_ours (h30_x + local late-truth correction).
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
import numpy as np
from latetruth import run_extra_heat_paths as X
from latetruth import run_kelmarsh_2018_unseen as K18
from latetruth.run_local_late_truth_long import ROOT, TARGET, HELP, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
OUT = OUT_DIR/'published_baselines_2026-10-14_v001'; EV0 = 4032; DELAY = 3


def load(ds):
    if ds == 'kelmarsh2017':
        D = {t: X.read(t) for t in range(1, 7)}
        D = {t: (df, mon) for t, (df, mon) in D.items()}
    elif ds == 'kelmarsh2018':
        D = {t: K18.read(t) for t in range(1, 7)}
    elif ds == 'penmanshiel2019':
        from latetruth import run_penmanshiel_replication as PEN
        D = {t: PEN.read(t) for t in PEN.TURB}
    elif ds in ('kelmarsh2024', 'penmanshiel2022', 'kelmarsh2020', 'penmanshiel2020'):   # later held-out years, 05_Protocols/LATER_YEARS_v001.md
        from latetruth import later_years as LY; from latetruth import run_h29_h30_physics as B; from latetruth import run_penmanshiel_replication as PEN
        cols = [TARGET, B.VANE] + HELP + X.EXTRA + K18.PRES
        D = {t: LY.read(ds, t, cols) for t in (range(1, 7) if ds.startswith('kelmarsh') else PEN.TURB)}
    else:
        raise ValueError(ds)
    n = len(D[1][0]); month = D[1][1]; assert all(len(D[t][0]) == n for t in D)
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    return D, n, month, Y, OK


def starts(month):
    """First row of each retraining window: EV0 (late January), then the first row of March..December."""
    return [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]


def windows(month, n):
    s = starts(month); return [(a, s[j + 1] if j + 1 < len(s) else n) for j, a in enumerate(s)]


def score(Y, P, OK, month, n):
    """Same metric as all earlier reports: MAE over Feb–Dec rows (row >= EV0), mean over turbines; calm/transient by 3-step target move."""
    a, ca, tr, mo = [], [], [], []
    for t in Y:
        ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); err = np.abs(Y[t][ev] - P[t][ev])
        mv = np.abs(Y[t][ev] - Y[t][ev - 3]); q50, q90 = np.quantile(mv[ok], [.5, .9])
        a.append(err[ok].mean()); ca.append(err[ok & (mv < q50)].mean()); tr.append(err[ok & (mv >= q90)].mean())
        mo.append([err[ok & (month[ev] == m_)].mean() for m_ in range(2, 13)])
    return dict(all=float(np.mean(a)), calm=float(np.mean(ca)), transient=float(np.mean(tr)), months=np.nanmean(mo, 0).round(4).tolist())


def ours(ds):
    D, n, month, Y, OK = load(ds); rows = np.arange(7, n)
    F = {}
    for t in D:
        f = X.feats(D[t][0], rows, 'h30_x'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    P = {t: np.full(n, np.nan) for t in D}
    for s, e in windows(month, n):
        XX, yy = [], []
        for t in D:
            r = np.arange(7, s - DELAY); r = r[OK[t][r]]; XX.append(np.column_stack([F[t][r], np.full(len(r), t)])); yy.append(Y[t][r])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in D:
            r = np.arange(s, e); r = r[OK[t][r]]; P[t][r] = m.predict(np.column_stack([F[t][r], np.full(len(r), t)])[:, keep])
    PL = {}
    for t, (df, _) in D.items():
        ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:EV0]
        Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL[t] = np.full(n, np.nan); PL[t][ev] = P[t][ev] + local(Z, r, ok, 8640)
    return D, n, month, Y, OK, P, PL


def senvion_raw(ds, t, cols):
    """Raw columns of one Senvion turbine-year (Kelmarsh or Penmanshiel), same row order as load()."""
    import io, csv, zipfile, pandas as pd
    if ds in ('kelmarsh2024', 'penmanshiel2022', 'kelmarsh2020', 'penmanshiel2020'):
        from latetruth import later_years as LY
        return LY.read(ds, t, [TARGET] + list(cols))[0]
    if ds.startswith('kelmarsh'):
        zp = next((DATA_DIR/'Kelmarsh').glob(f'*{ds[-4:]}*.zip')); pre = f'Turbine_Data_Kelmarsh_{t}_'
    else:
        d = DATA_DIR/'Penmanshiel/Version 0.0.2'
        zp = d/('Penmanshiel_SCADA_2019_WT01-10_3112.zip' if t <= 10 else 'Penmanshiel_SCADA_2019_WT11-15_3117.zip'); pre = f'Turbine_Data_Penmanshiel_{t:02d}_'
    with zipfile.ZipFile(zp) as z:
        name = next(nm for nm in z.namelist() if nm.startswith(pre))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            return pd.read_csv(f, names=h, header=None, usecols=[h[0], TARGET] + list(cols))


def export_inputs(ds):
    """Raw channels for the published baselines (pure-numpy consumers in Python): inputs_<ds> (16-slot layout), inputs2_<ds>
    (wind direction, stator temp), inputs3_<ds> (pitch A, nacelle position). Same rows/OK as load()."""
    from latetruth import run_h29_h30_physics as B
    D, n, month, Y, OK = load(ds); cols = HELP + X.EXTRA + [B.VANE] + K18.PRES
    ex2 = ['Wind direction (°)', 'Stator temperature 1 (°C)']; ex3 = ['Blade angle (pitch position) A (°)', 'Nacelle position (°)']; XS, X2, X3 = {}, {}, {}
    for t in D:
        df = senvion_raw(ds, t, cols + ex2 + ex3)
        assert len(df) == n and np.allclose(df[TARGET].to_numpy(float), Y[t], equal_nan=True)
        XS[t], X2[t], X3[t] = df[cols].to_numpy(float), df[ex2].to_numpy(float), df[ex3].to_numpy(float)
    np.savez_compressed(OUT/f'inputs_{ds}.npz', month=month, cols=np.array(cols), **{f'X_{t}': XS[t] for t in D}, **{f'Y_{t}': Y[t] for t in D}, **{f'OK_{t}': OK[t] for t in D})
    np.savez_compressed(OUT/f'inputs2_{ds}.npz', cols=np.array(ex2), **{f'X_{t}': X2[t] for t in D})
    np.savez_compressed(OUT/f'inputs3_{ds}.npz', cols=np.array(ex3), **{f'X_{t}': X3[t] for t in D})
    print(ds, 'exported', len(cols), '+', len(ex2), '+', len(ex3), 'channels,', len(D), 'turbines', flush=True)


def ours_sync(ds, delay=1):
    """Claim C3: our pipeline when truth is released `delay` steps late (synchronous regime: delay 1). Feature code is unchanged; it reads
    oil(t-3), oil(t-4), so the oil column handed to it is advanced by (3 - delay) steps -> oil(t-delay), oil(t-delay-1). Training rows < start - delay;
    local correction uses residuals released `delay` steps late."""
    D, n, month, Y, OK = load(ds); rows = np.arange(7, n); sh = DELAY - delay; F = {}
    for t in D:
        d = D[t][0].copy(); d[TARGET] = d[TARGET].shift(-sh)
        f = X.feats(d, rows, 'h30_x'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    # Amendment 16: the newest released target y(t-delay) must be finite (was Y[t+delay], a future-missingness mask)
    OKs = {t: OK[t] & np.r_[np.zeros(delay, bool), np.isfinite(Y[t][:-delay])] for t in D}
    P = {t: np.full(n, np.nan) for t in D}
    for s, e in windows(month, n):
        XX, yy = [], []
        for t in D:
            r = np.arange(7, s - delay); r = r[OKs[t][r]]; XX.append(np.column_stack([F[t][r], np.full(len(r), t)])); yy.append(Y[t][r])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in D:
            r = np.arange(s, e); r = r[OKs[t][r]]; P[t][r] = m.predict(np.column_stack([F[t][r], np.full(len(r), t)])[:, keep])
    PL = {}
    for t, (df, _) in D.items():
        ev = np.arange(EV0, n); ok = OKs[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:EV0]
        Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL[t] = np.full(n, np.nan); PL[t][ev] = P[t][ev] + local(Z, r, ok, 8640, delay=delay)
    return Y, OK, month, n, P, PL


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[2] == 'sync':
    ds = sys.argv[1]; f = OUT/f'ours_sync_{ds}.npz'
    if f.exists(): raise FileExistsError(f)
    Y, OK, month, n, P, PL = ours_sync(ds)
    s1, s2 = score(Y, P, OK, month, n), score(Y, PL, OK, month, n)
    print(ds, 'SYNC fleet h30_x', round(s1['all'], 4), '| SYNC ours h30_x+local', round(s2['all'], 4), flush=True)
    np.savez_compressed(f, month=month, **{f'{k}_{t}': v for t in Y for k, v in [('P_fleet', P[t]), ('P_ours', PL[t])]})
    (OUT/f'ours_sync_{ds}.json').write_text(json.dumps({'fleet_h30_x_sync': s1, 'ours_sync': s2, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))
elif __name__ == '__main__':
    ds = sys.argv[1]; OUT.mkdir(exist_ok=True); f = OUT/f'ours_{ds}.npz'
    if f.exists(): raise FileExistsError(f)
    D, n, month, Y, OK, P, PL = ours(ds)
    s1, s2 = score(Y, P, OK, month, n), score(Y, PL, OK, month, n)
    print(ds, 'fleet h30_x', round(s1['all'], 4), '| ours h30_x+local', round(s2['all'], 4), flush=True)
    np.savez_compressed(f, month=month, **{f'{k}_{t}': v for t in D for k, v in [('Y', Y[t]), ('OK', OK[t]), ('P_fleet', P[t]), ('P_ours', PL[t])]})
    (OUT/f'ours_{ds}.json').write_text(json.dumps({'fleet_h30_x': s1, 'ours_h30_x_local': s2, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))
