"""Protocol 05_Protocols/HILL_OF_TOWIE_REPLICATION_v001.md — same pipeline as Kelmarsh/Penmanshiel on Siemens SWT-2.3."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import zipfile, json, hashlib, sys, os
import numpy as np, pandas as pd
from latetruth.run_local_late_truth_long import ROOT, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
YEAR = int(os.environ.get('HOT_YEAR', '2019'))   # later held-out year: HOT_YEAR=2025 (05_Protocols/LATER_YEARS_v001.md)
ZP = DATA_DIR/f'HillOfTowie_04/{YEAR}.zip'
OUT = OUT_DIR/('hill_of_towie_replication_2019_2026-10-13_v001' if YEAR == 2019 else f'later_years_hot{YEAR}_2026-10-15_v001')
TARGET = 'wtc_GeOilTmp_mean'
HELP = ['wtc_ActPower_mean', 'wtc_AcWindSp_mean', 'wtc_MainSRpm_mean', 'wtc_GenRpm_mean', 'wtc_AmbieTmp_mean', 'wtc_NacelTmp_mean', 'wtc_AmpPhR_mean', 'wtc_GenBeGTm_mean']
GBB = ['wtc_HSGenTmp_mean', 'wtc_HSRotTmp_mean', 'wtc_IMSGenTm_mean', 'wtc_IMSRotTm_mean']
EXTRA = GBB + ['wtc_MainBTmp_mean']; PRES = 'wtc_GearPres_mean'
TABLES = {'tblSCTurTemp': [TARGET, 'wtc_AmbieTmp_mean', 'wtc_NacelTmp_mean', 'wtc_GenBeGTm_mean'] + EXTRA, 'tblSCTurPress': [PRES],
          'tblSCTurGrid': ['wtc_ActPower_mean', 'wtc_AmpPhR_mean'], 'tblSCTurbine': ['wtc_AcWindSp_mean', 'wtc_MainSRpm_mean', 'wtc_GenRpm_mean']}
GRID = pd.date_range(f'{YEAR}-01-01 00:00', f'{YEAR}-12-31 23:50', freq='10min'); n = len(GRID); month = GRID.month.to_numpy()
EV0 = int(np.flatnonzero(month == 2)[0]); STATIONS = list(range(2304510, 2304531))


def read_all():
    parts = {s: [] for s in STATIONS}
    with zipfile.ZipFile(ZP) as z:
        for m in range(1, 13):
            frames = []
            for tb, cols in TABLES.items():
                d = pd.read_csv(z.open(f'{tb}_{YEAR}_{m:02d}.csv'), usecols=['TimeStamp', 'StationId'] + cols)
                d['TimeStamp'] = pd.to_datetime(d['TimeStamp'], format='ISO8601'); frames.append(d.drop_duplicates(['TimeStamp', 'StationId']).set_index(['TimeStamp', 'StationId']))
            d = pd.concat(frames, axis=1)
            have = set(d.index.get_level_values('StationId'))   # a turbine absent from a monthly file (HoT 2022) -> rows missing (NaN)
            for s in STATIONS:
                if s in have: parts[s].append(d.xs(s, level='StationId'))
    return {i + 1: pd.concat(parts[s]).sort_index().loc[lambda x: ~x.index.duplicated()].reindex(GRID) for i, s in enumerate(STATIONS)}


def feats(df, kind):
    rows = np.arange(7, n); V = df[HELP].to_numpy(float); oil = df[TARGET].to_numpy(float)
    f = [V[rows]] + [V[rows] - V[rows - k] for k in (1, 2, 3)] + [np.column_stack([oil[rows - 3], oil[rows - 3] - oil[rows - 4]])]
    if kind.startswith('base_x'):
        E = df[EXTRA].to_numpy(float); f += [E[rows]] + [E[rows] - E[rows - k] for k in (1, 2, 3)]
    if kind.endswith('_p'):
        p = df[PRES].to_numpy(float); f += [np.column_stack([p[rows]] + [p[rows] - p[rows - k] for k in (1, 2, 3)])]
    f = np.column_stack(f); return np.vstack([np.full((7, f.shape[1]), np.nan), f])


def run(D, featfn, label, use_local, OK, Y):
    starts = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]
    F = {t: featfn(t) for t in D}; P = {t: np.full(n, np.nan) for t in D}
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else n; XX, yy = [], []
        for t in D:
            rows = np.arange(7, s - 3); rows = rows[OK[t][rows]]; XX.append(np.column_stack([F[t][rows], np.full(len(rows), t)])); yy.append(Y[t][rows])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=4): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in D:
            rows = np.arange(s, e); rows = rows[OK[t][rows]]; P[t][rows] = m.predict(np.column_stack([F[t][rows], np.full(len(rows), t)])[:, keep])
    out = {}
    for lab in [label] + ([label + '+local'] if use_local else []):
        a, ca, tr, mo = [], [], [], []
        for t, df in D.items():
            ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
            if lab.endswith('+local'):
                H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:EV0]
                Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9); r = r - local(Z, r, ok, 8640)
            err = np.abs(r); mv = np.abs(Y[t][ev] - Y[t][ev - 3]); q50, q90 = np.quantile(mv[ok], [.5, .9])
            a.append(err[ok].mean()); ca.append(err[ok & (mv < q50)].mean()); tr.append(err[ok & (mv >= q90)].mean())
            mo.append([err[ok & (month[ev] == m_)].mean() for m_ in range(2, 13)])
        out[lab] = dict(all=float(np.mean(a)), calm=float(np.mean(ca)), transient=float(np.mean(tr)), months=np.nanmean(mo, 0).round(4).tolist())
        print(f"{lab:34s} all {out[lab]['all']:.4f} calm {out[lab]['calm']:.4f} transient {out[lab]['transient']:.4f}", flush=True)
    return out


def stale(D, cols, kind='base_x'):
    def fn(t):
        d = D[t].copy()
        for c in cols: d[c] = d[c].shift(3)
        return feats(d, kind)
    return fn


def main(part):
    out = OUT/f'results_{part}.json'
    if out.exists(): raise FileExistsError(out)
    OUT.mkdir(exist_ok=True); D = read_all()
    Y = {t: df[TARGET].to_numpy(float) for t, df in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    print('rows', n, 'EV0', EV0, 'ok share', round(float(np.mean([OK[t].mean() for t in D])), 4), flush=True); res = {}
    if part == 'A':
        for kind in ['base', 'base_x']: res.update(run(D, lambda t, k=kind: feats(D[t], k), kind, True, OK, Y))
        res.update(run(D, stale(D, GBB), 'stale:all_gearbox_bearings', False, OK, Y))
        res.update(run(D, stale(D, GBB, 'base_x_p'), 'stale:all_gearbox_bearings+pressure', False, OK, Y))
    else:
        cols = (HELP + EXTRA)[:7] if part == 'B' else (HELP + EXTRA)[7:]
        for c in cols: res.update(run(D, stale(D, [c]), 'stale:' + c, False, OK, Y))
    out.write_text(json.dumps({'results': res, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main(sys.argv[1])
