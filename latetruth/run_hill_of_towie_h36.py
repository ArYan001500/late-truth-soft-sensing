"""Protocol 05_Protocols/HILL_OF_TOWIE_REPLICATION_v001.md, Amendment 1 (R7/R8)."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import json, hashlib, sys
import numpy as np, pandas as pd
from latetruth import run_hill_of_towie_replication as H
PUMP = ['wtc_GearPuHS_timeon', 'wtc_GearPuLS_timeon']; FILT = ['wtc_InlPrBef_mean', 'wtc_InlPrAft_mean']
CTRL = ['wtc_GearCoHS_timeon', 'wtc_GearCoLS_timeon', 'wtc_GearHeat_timeon']
H.TABLES['tblSCTurPress'] = H.TABLES['tblSCTurPress'] + FILT; H.TABLES['tblSCTurDigiOut'] = PUMP + CTRL


def feats(df, kind):
    f = H.feats(df, 'base_x_p'); d = df.copy(); d['dP'] = d[FILT[0]] - d[FILT[1]]
    cols = PUMP + FILT + ['dP'] + (CTRL if kind == 'r8' else []); A = d[cols].to_numpy(float); rows = np.arange(7, H.n)
    g = np.column_stack([A[rows]] + [A[rows] - A[rows - k] for k in (1, 2, 3)])
    return np.column_stack([f, np.vstack([np.full((7, g.shape[1]), np.nan), g])])


def main():
    out = H.OUT/'results_h36.json'
    if out.exists(): raise FileExistsError(out)
    D = H.read_all(); Y = {t: df[H.TARGET].to_numpy(float) for t, df in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    print('missing share pump/filter/ctrl', {c: round(float(np.mean([D[t][c].isna().mean() for t in D])), 4) for c in PUMP + FILT + CTRL}, flush=True)
    res = {}
    for kind in ['r7', 'r8']:
        def fn(t, k=kind):
            d = D[t].copy()
            for c in H.GBB: d[c] = d[c].shift(3)
            return feats(d, k)
        res.update(H.run(D, fn, 'stale:all_gearbox_bearings+pressure+' + kind, False, OK, Y))
    out.write_text(json.dumps({'results': res, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
