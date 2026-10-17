"""B11 of protocols/PUBLISHED_BASELINES_v001.md — Bai, Han, Kang, Tao, Pang, Dai, Liu 2024 (Applied Energy 371, 123687):
energy-flow physical features for gear-oil temperature. Their final feature set (Sec. 3, end): gearbox component temperatures and their
rates of change, nacelle temperature, gear and bearing rotational speeds, wind speed, pitch angle, yaw error, tip-speed ratio; no oil
history. Kelmarsh mapping: front/rear/rotor-bearing temps (+1-step changes), nacelle temp, rotor speed, gearbox speed, generator rpm,
wind speed, pitch A, yaw error = wrap(wind direction - nacelle position), tip-speed ratio = omega_rotor * 46 m / wind speed (MM92 radius).
Model: XGBoost (one of their four models; hyper-parameters as B3 grid, chosen on Kelmarsh 2017 March). Per turbine, monthly retraining.
usage: python baselines_bai.py
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C


def features(ds):
    month, X, Y, OK = C.load(ds); z2 = np.load(C.OUT/f'inputs2_{ds}.npz', allow_pickle=False); z3 = np.load(C.OUT/f'inputs3_{ds}.npz', allow_pickle=False); F = {}
    for t in X:
        x = X[t]; wdir = z2[f'X_{t}'][:, 0]; pitch, npos = z3[f'X_{t}'][:, 0], z3[f'X_{t}'][:, 1]
        temps = x[:, [C.FRONT, C.REAR, C.ROTB]]; rate = np.vstack([np.full((1, 3), np.nan), np.diff(temps, axis=0)])
        yaw = (wdir - npos + 180) % 360 - 180; tsr = x[:, C.ROTOR] * 2 * np.pi / 60 * 46.0 / np.maximum(x[:, C.WIND], 0.5)
        F[t] = np.column_stack([temps, rate, x[:, [C.NAC, C.ROTOR, C.GBS, C.GENRPM, C.WIND]], pitch, yaw, tsr])
    return month, F, Y, OK


class B11(C.B3):
    def prepare(self, X, y, tr): self.ch = [c for c in range(X.shape[1]) if np.isfinite(X[tr, c]).mean() >= 0.95]   # drops channels a farm does not record (Hill of Towie: wind direction -> yaw error)
    def design(self, X, y, r): return X[r][:, self.ch]


def run(hp, ds, val=False):
    month, F, Y, OK = features(ds); n = len(month); P = {t: np.full(n, np.nan) for t in Y}
    wins = [(int(np.flatnonzero(month == 3)[0]), int(np.flatnonzero(month == 4)[0]))] if val else C.windows(month, n)
    for t in Y:
        for s, e in wins:
            m = B11(hp, 0); m.fit(F[t], Y[t], C.released(OK[t], Y[t], s)); r = np.arange(s, e); P[t][r] = m.predict(F[t], Y[t], r)
    if val:
        a, b = wins[0]; ev = np.arange(a, b)
        return float(np.mean([np.abs(Y[t][ev] - P[t][ev])[OK[t][ev] & np.isfinite(P[t][ev])].mean() for t in Y]))
    return C.score(Y, P, OK, month), P


if __name__ == '__main__':
    f = C.OUT/'B11_bai_xgb.json'
    if f.exists(): raise FileExistsError(f)
    v = [run(hp, 'kelmarsh2017', val=True) for hp in C.GRIDS['B3']]
    for hp, m in zip(C.GRIDS['B3'], v): print('tune', hp, round(m, 4), flush=True)
    hp = C.GRIDS['B3'][int(np.argmin(v))]; res = {'hp': hp, 'val_mae_2017Mar': float(min(v))}; print('chosen', hp, flush=True)
    for ds in ['kelmarsh2017', 'kelmarsh2018']:
        sc, P = run(hp, ds); res[ds] = sc; print('B11', ds, round(sc['all'], 4), flush=True)
        np.savez_compressed(C.OUT/f'B11_bai_xgb_{ds}.npz', **{f'P_{t}': p for t, p in P.items()})
    res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); f.write_text(json.dumps(res, indent=2))
