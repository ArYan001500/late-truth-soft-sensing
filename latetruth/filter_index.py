"""Exploratory filter-condition index (Amendment 13), Hill of Towie 2019, 21 Siemens turbines.
Laminar Darcy law with Walther viscosity: dP_f = R_i(t) * mu(T) * Q  ->  D_i(t) = ln dP_f - ln Q - ln nu(T) - ref_i,
ref_i = median of the January values of turbine i (kinematic viscosity; oil density treated as constant).
Rows: one pump stage on for the whole 10-min interval (Q = 70 or 142 L/min), 20 < T < 55 degC, dP_f > 0.02 bar (as the physics check).
Weekly medians per turbine; for each turbine: week and size of the largest weekly drop, and the change of the index from the
January reference to the week before that drop. No maintenance log is available, so drops cannot be matched to filter changes.
usage: python filter_index.py <HillOfTowie 2019.zip>"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import zipfile, sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
z = zipfile.ZipFile(sys.argv[1]); K = ['TimeStamp', 'StationId']; F = []
for m in range(1, 13):
    t = pd.read_csv(z.open(f'tblSCTurTemp_2019_{m:02d}.csv'), usecols=K + ['wtc_GeOilTmp_mean'])
    p = pd.read_csv(z.open(f'tblSCTurPress_2019_{m:02d}.csv'), usecols=K + ['wtc_InlPrAft_mean', 'wtc_InlPrBef_mean'])
    o = pd.read_csv(z.open(f'tblSCTurDigiOut_2019_{m:02d}.csv'), usecols=K + ['wtc_GearPuHS_timeon', 'wtc_GearPuLS_timeon'])
    F.append(t.merge(p, on=K).merge(o, on=K))
d = pd.concat(F).dropna(); d['TimeStamp'] = pd.to_datetime(d.TimeStamp)
d['dP'] = d.wtc_InlPrBef_mean - d.wtc_InlPrAft_mean; d['T'] = d.wtc_GeOilTmp_mean
d['Q'] = np.select([d.wtc_GearPuHS_timeon >= 590, d.wtc_GearPuLS_timeon >= 590], [142.0, 70.0], np.nan)
d = d[np.isfinite(d.Q) & (d.dP > 0.02) & (d['T'] > 20) & (d['T'] < 55)]
T1, T2 = np.log10(313.15), np.log10(373.15); Y1, Y2 = np.log10(np.log10(320.7)), np.log10(np.log10(24.7)); B = (Y1 - Y2) / (T2 - T1); A = Y1 + B * T1
nu = 10 ** (10 ** (A - B * np.log10(d['T'] + 273.15))) - 0.7
d['D'] = np.log(d.dP) - np.log(d.Q) - np.log(nu)
res = {'n_rows': int(len(d)), 'turbines': {}}
for s, g in d.groupby('StationId'):
    ref = g.D[g.TimeStamp.dt.month == 1].median(); w = (g.set_index('TimeStamp').D - ref).resample('7D').median()
    j = w.diff(); k = j.idxmin(); before = w[:k].iloc[-2] if len(w[:k]) >= 2 else np.nan
    res['turbines'][f'T{s - 2304509:02d}'] = dict(largest_drop_week=str(k.date()), drop=float(j.min()), index_before_drop=float(before),
                                                  index_end_of_year=float(w.iloc[-1]), weekly=[None if np.isnan(x) else round(float(x), 4) for x in w.values])
dates = pd.to_datetime([v['largest_drop_week'] for v in res['turbines'].values()])
best = max(((dates >= a) & (dates < a + pd.Timedelta(days=42))).sum() for a in pd.date_range('2019-01-01', '2019-12-31', freq='7D'))
res['summary'] = dict(first_drop_week=str(dates.min().date()), last_drop_week=str(dates.max().date()),
                      max_turbines_within_any_6_weeks=int(best), n_turbines=int(len(dates)),
                      drops_median=float(np.median([v['drop'] for v in res['turbines'].values()])),
                      index_before_drop_positive=int(sum(v['index_before_drop'] > 0 for v in res['turbines'].values())))
res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
out = OUT_DIR/'filter_index_2026-10-15_v001'; out.mkdir(exist_ok=True)
(out/'results.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res['summary'], indent=1))
for k, v in res['turbines'].items(): print(k, v['largest_drop_week'], round(v['drop'], 3), 'before', round(v['index_before_drop'], 3), 'end', round(v['index_end_of_year'], 3))
