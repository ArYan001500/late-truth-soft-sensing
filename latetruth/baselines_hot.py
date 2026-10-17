"""PUBLISHED_BASELINES_v001 Amendment 8 — Hill of Towie 2019 (Siemens SWT-2.3, 21 turbines) adapter.
(1) our pipeline on Hill of Towie = base_x + local late-truth correction (as in HILL_OF_TOWIE_REPLICATION_v001; Siemens has no inlet/vane),
    but with the same evaluation start as every other comparison here (EV0 = 4032, then 1 March .. 1 December monthly starts);
(2) baseline inputs in the Senvion 16-slot layout used by baselines_classic/deep/kim/bai. Slot map (Siemens -> slot):
    POWER ActPower, WIND AcWindSp, ROTOR MainSRpm, GENRPM GenRpm, AMB AmbieTmp, NAC NacelTmp, CUR AmpPhR, GENB GenBeGTm, INLET none,
    FRONT IMSRotTm, REAR HSGenTmp, ROTB MainBTmp, GBS GenRpm, VANE none, PPUMP GearPres, PINL none; inputs2: wind direction none,
    stator Gen1U1Tm; inputs3: pitch PitcPosA, nacelle position NacelPos.  usage: python baselines_hot.py
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib
import numpy as np
from latetruth import run_hill_of_towie_replication as H
from latetruth import baselines_common as BC
from latetruth.run_local_late_truth_long import PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
DS = f'hot{H.YEAR}'; EV0 = 4032
H.TABLES['tblSCTurbine'] = H.TABLES['tblSCTurbine'] + ['wtc_PitcPosA_mean', 'wtc_NacelPos_mean']
H.TABLES['tblSCTurTemp'] = H.TABLES['tblSCTurTemp'] + ['wtc_Gen1U1Tm_mean']


def main():
    f = BC.OUT/f'ours_{DS}.npz'
    if f.exists(): raise FileExistsError(f)
    D = H.read_all(); n = H.n; month = H.month; Y = {t: df[H.TARGET].to_numpy(float) for t, df in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    F = {t: H.feats(D[t], 'base_x') for t in D}; P = {t: np.full(n, np.nan) for t in D}
    for s, e in BC.windows(month, n):
        XX, yy = [], []
        for t in D:
            r = np.arange(7, s - 3); r = r[OK[t][r]]; XX.append(np.column_stack([F[t][r], np.full(len(r), t)])); yy.append(Y[t][r])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
        for t in D:
            r = np.arange(s, e); r = r[OK[t][r]]
            if len(r): P[t][r] = m.predict(np.column_stack([F[t][r], np.full(len(r), t)])[:, keep])
    PL = {}
    for t, df in D.items():
        ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        Hh = np.nan_to_num(df[H.HELP].to_numpy(float)[ev]); Hf = df[H.HELP].to_numpy(float)[4:EV0]
        Z = (Hh - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9); PL[t] = np.full(n, np.nan); PL[t][ev] = P[t][ev] + local(Z, r, ok, 8640)
    s1, s2 = BC.score(Y, P, OK, month, n), BC.score(Y, PL, OK, month, n)
    print(DS, 'fleet base_x', round(s1['all'], 4), '| ours base_x+local', round(s2['all'], 4), flush=True)
    np.savez_compressed(f, month=month, **{f'{k}_{t}': v for t in D for k, v in [('Y', Y[t]), ('OK', OK[t]), ('P_fleet', P[t]), ('P_ours', PL[t])]})
    (BC.OUT/f'ours_{DS}.json').write_text(json.dumps({'fleet_base_x': s1, 'ours_base_x_local': s2, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))
    nanc = lambda: np.full(n, np.nan)
    slots = lambda d: np.column_stack([d['wtc_ActPower_mean'], d['wtc_AcWindSp_mean'], d['wtc_MainSRpm_mean'], d['wtc_GenRpm_mean'], d['wtc_AmbieTmp_mean'],
                                       d['wtc_NacelTmp_mean'], d['wtc_AmpPhR_mean'], d['wtc_GenBeGTm_mean'], nanc(), d['wtc_IMSRotTm_mean'], d['wtc_HSGenTmp_mean'],
                                       d['wtc_MainBTmp_mean'], d['wtc_GenRpm_mean'], nanc(), d['wtc_GearPres_mean'], nanc()]).astype(float)
    np.savez_compressed(BC.OUT/f'inputs_{DS}.npz', month=month, cols=np.array(['siemens 16-slot map, see baselines_hot.py']),
                        **{f'X_{t}': slots(D[t]) for t in D}, **{f'Y_{t}': Y[t] for t in D}, **{f'OK_{t}': OK[t] for t in D})
    np.savez_compressed(BC.OUT/f'inputs2_{DS}.npz', **{f'X_{t}': np.column_stack([nanc(), D[t]['wtc_Gen1U1Tm_mean']]).astype(float) for t in D})
    np.savez_compressed(BC.OUT/f'inputs3_{DS}.npz', **{f'X_{t}': D[t][['wtc_PitcPosA_mean', 'wtc_NacelPos_mean']].to_numpy(float) for t in D})
    print(DS, 'inputs exported', len(D), 'turbines', flush=True)


if __name__ == '__main__': main()
