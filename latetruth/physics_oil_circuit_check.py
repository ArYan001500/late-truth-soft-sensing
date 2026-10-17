from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
from latetruth.data import require_archive
import sys
# Physics check on Hill of Towie January 2019 (training window only): does the filter pressure drop follow
# dP = R * mu(T) * Q with Q set by the pump stage (70 / 142 l/min) and mu(T) from the Walther equation of an ISO VG 320 oil?
import zipfile, sys, numpy as np, pandas as pd
z = zipfile.ZipFile(require_archive(sys.argv[1])); K = ['TimeStamp', 'StationId']
t = pd.read_csv(z.open('tblSCTurTemp_2019_01.csv'), usecols=K + ['wtc_GeOilTmp_mean'])
p = pd.read_csv(z.open('tblSCTurPress_2019_01.csv'), usecols=K + ['wtc_InlPrAft_mean', 'wtc_InlPrBef_mean', 'wtc_GearPres_mean'])
o = pd.read_csv(z.open('tblSCTurDigiOut_2019_01.csv'), usecols=K + ['wtc_GearPuHS_timeon', 'wtc_GearPuLS_timeon'])
d = t.merge(p, on=K).merge(o, on=K).dropna()
d['dP'] = d.wtc_InlPrBef_mean - d.wtc_InlPrAft_mean; d['T'] = d.wtc_GeOilTmp_mean
d['stage'] = np.select([d.wtc_GearPuHS_timeon >= 590, d.wtc_GearPuLS_timeon >= 590], [142.0, 70.0], np.nan)
d = d[np.isfinite(d.stage) & (d.dP > 0.02) & (d['T'] > 20) & (d['T'] < 55)]
# Walther (ASTM D341): log10 log10(nu + 0.7) = A - B log10(T_K); ISO VG 320 nominal nu40 = 320 cSt, nu100 assumed 24 cSt (typical VG320 mineral gear oil, VI ~ 95)
def walther(nu40=320.0, nu100=24.0):
    T1, T2 = np.log10(313.15), np.log10(373.15); Y1, Y2 = np.log10(np.log10(nu40 + .7)), np.log10(np.log10(nu100 + .7))
    B = (Y1 - Y2) / (T2 - T1); A = Y1 + B * T1
    return lambda Tc: 10 ** (10 ** (A - B * np.log10(Tc + 273.15))) - 0.7
nu = walther(); d['nu'] = nu(d['T'])
# model 1: log dP = c + a*log Q + b*log nu   (physics predicts a = 1, b = 1 for laminar flow through the filter)
X = np.column_stack([np.ones(len(d)), np.log(d.stage), np.log(d.nu)]); yv = np.log(d.dP)
coef, *_ = np.linalg.lstsq(X, yv, rcond=None); res = yv - X @ coef; r2 = 1 - res.var() / yv.var()
print('n', len(d), 'turbines', d.StationId.nunique(), '| fit log dP = c + a log Q + b log nu: a=%.2f b=%.2f R2=%.3f' % (coef[1], coef[2], r2))
# per-turbine intercept (filter resistance differs per turbine)
Xt = pd.get_dummies(d.StationId).to_numpy(float); X2 = np.column_stack([Xt, np.log(d.stage), np.log(d.nu)])
c2, *_ = np.linalg.lstsq(X2, yv, rcond=None); r22 = 1 - (yv - X2 @ c2).var() / yv.var()
print('with per-turbine resistance: a=%.2f b=%.2f R2=%.3f' % (c2[-2], c2[-1], r22))
for lo, hi in [(30, 35), (35, 40), (40, 45), (45, 50)]:
    m = (d['T'] >= lo) & (d['T'] < hi) & (d.stage == 142)
    print(f'HS stage T {lo}-{hi}: median dP {d.dP[m].median():.3f} bar, Walther nu {nu((lo+hi)/2):.0f} cSt, n {m.sum()}')
print('ratio median dP 30-35 / 45-50 (HS):', round(d.dP[(d["T"]>=30)&(d["T"]<35)&(d.stage==142)].median()/d.dP[(d["T"]>=45)&(d["T"]<50)&(d.stage==142)].median(),2), '| Walther nu ratio:', round(nu(32.5)/nu(47.5),2))

import json, hashlib
from pathlib import Path
out = OUT_DIR/'physics_oil_circuit_2026-10-15_v001/results.json'
bins = {}
for lo, hi in [(30, 35), (35, 40), (40, 45), (45, 50)]:
    m = (d['T'] >= lo) & (d['T'] < hi) & (d.stage == 142)
    bins[f'{lo}-{hi}'] = dict(median_dP_bar=float(d.dP[m].median()), walther_nu_cSt=float(nu((lo + hi) / 2)), n=int(m.sum()))
out.write_text(json.dumps(dict(data='Hill of Towie 2019, January (initial training window), pump stage HS/LS rows, 20<T<55 C, dP>0.02 bar',
    n_rows=int(len(d)), n_turbines=int(d.StationId.nunique()), oil='ISO VG 320, nu40=320 cSt, nu100=24 cSt (assumed)',
    pooled=dict(a_flow=float(coef[1]), b_viscosity=float(coef[2]), r2=float(r2)),
    per_turbine_resistance=dict(a_flow=float(c2[-2]), b_viscosity=float(c2[-1]), r2=float(r22)), hs_stage_bins=bins,
    code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=2))
print('saved', out)
