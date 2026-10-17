"""Saves the frozen Amendment 17/18 variants for the confirmation years (CONFIRM_TWO_TIMESCALE_v001), with the same code as
confirm_two_timescale.py: ref_mean20_, ref_ewma_, rec_, recg_, gate_ <ds>.npz (P_<t>). The Table-7 years already have them."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import residual_references as R
from latetruth import standstill_gate as G
O = C.OUT; EV0 = C.EV0
sel = json.load(open(O/'GATE.json'))['selection']; form, h, a = sel['chosen']['form'], sel['chosen']['h'], sel['a']
for ds in BT.CONFIRM:
    z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); month, Y, OK, T, yr, s, nac = G.prep(ds); n = len(month); ev = np.arange(EV0, n); P = {}
    for t in T:
        PF, PO = z[f'P_fleet_{t}'], z[f'P_ours_{t}']; ok = OK[t][ev] & np.isfinite(PF[ev]); r = np.where(ok, Y[t][ev] - PF[ev], 0.0)
        ok2 = OK[t][ev] & np.isfinite(PO[ev]); r2 = np.where(ok2, Y[t][ev] - PO[ev], 0.0); v = {}
        for nm, base, cor in [('ref_mean20', PF, R.mean20(r, ok)), ('ref_ewma', PF, 0.75 * R.ewma(r, ok, 18)), ('rec', PO, 0.5 * R.ewma(r2, ok2, 18))]:
            q = np.full(n, np.nan); q[ev] = base[ev] + cor; v[nm] = q
        v['gate'] = G.gated(PO, yr[t], s[t], nac[t], form, h, a); v['recg'] = np.where(np.abs(v['gate'] - PO) > 1e-12, v['gate'], v['rec'])
        for k, q in v.items(): P.setdefault(k, {})[t] = q
    for k in P: np.savez_compressed(O/f'{k}_{ds}.npz', **{f'P_{t}': P[k][t] for t in T})
    print(ds, 'saved', sorted(P), flush=True)
