"""Figure data for the oil-circuit figure: filter pressure drop against oil temperature on Hill of Towie, January 2019 (the training
window), with the same row filter as physics_oil_circuit_check.py. Per pump stage, 1-degC bins: median, quartiles and count, plus the
Walther viscosity of ISO VG 320 at the bin centre.   usage: python figure_data_oil.py <path to HillOfTowie 2019.zip>"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import zipfile, sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
z = zipfile.ZipFile(sys.argv[1]); K = ['TimeStamp', 'StationId']
t = pd.read_csv(z.open('tblSCTurTemp_2019_01.csv'), usecols=K + ['wtc_GeOilTmp_mean'])
p = pd.read_csv(z.open('tblSCTurPress_2019_01.csv'), usecols=K + ['wtc_InlPrAft_mean', 'wtc_InlPrBef_mean'])
o = pd.read_csv(z.open('tblSCTurDigiOut_2019_01.csv'), usecols=K + ['wtc_GearPuHS_timeon', 'wtc_GearPuLS_timeon'])
d = t.merge(p, on=K).merge(o, on=K).dropna()
d['dP'] = d.wtc_InlPrBef_mean - d.wtc_InlPrAft_mean; d['T'] = d.wtc_GeOilTmp_mean
d['stage'] = np.select([d.wtc_GearPuHS_timeon >= 590, d.wtc_GearPuLS_timeon >= 590], [142.0, 70.0], np.nan)
d = d[np.isfinite(d.stage) & (d.dP > 0.02) & (d['T'] > 20) & (d['T'] < 55)]
T1, T2 = np.log10(313.15), np.log10(373.15); Y1, Y2 = np.log10(np.log10(320.7)), np.log10(np.log10(24.7)); B = (Y1 - Y2) / (T2 - T1); A = Y1 + B * T1
nu = lambda c: 10 ** (10 ** (A - B * np.log10(c + 273.15))) - 0.7
res = {'n_rows': int(len(d)), 'stages': {}}
for q, g in d.groupby('stage'):
    b = np.floor(g['T']).astype(int); rows = []
    for c, h in g.groupby(b):
        if len(h) >= 50: rows.append(dict(T=float(c + 0.5), median=float(h.dP.median()), q25=float(h.dP.quantile(.25)), q75=float(h.dP.quantile(.75)), n=int(len(h)), nu=float(nu(c + 0.5))))
    res['stages'][str(int(q))] = rows
res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
out = OUT_DIR/'physics_oil_circuit_2026-10-15_v001/figdata_dp_vs_T.json'
out.write_text(json.dumps(res, indent=1)); print('saved', out, res['n_rows'], {k: len(v) for k, v in res['stages'].items()})
