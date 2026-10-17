"""Dependence-aware companion to S1 (added after the external review, 2026-10-16): the turbine is the unit. For every held-out
farm-year and baseline, count the turbines on which the proposed estimator has the lower common-row MAE, with an exact
two-sided sign test. Writes TURBINE_LEVEL.json. Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from math import comb
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import significance as S


def sign_p(k, n):
    m = min(k, n - k); return min(1.0, 2 * sum(comb(n, j) for j in range(m + 1)) / 2 ** n)


out = {}
for ds in ['kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER:
    month, X, Y, OK = C.load(ds); T = sorted(Y); ev = np.arange(C.EV0, len(month)); M = BT.load_methods(ds, T)
    keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
    com = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
    tb = lambda k: np.array([np.abs(Y[t][ev] - M[k][t][ev])[com[t]].mean() for t in T])
    o = tb(S.OURS); rows = {}
    for k in keys:
        if k == S.OURS or '[v1]' in k: continue
        w = int((tb(k) > o).sum()); rows[k] = dict(wins=w, n=len(T), p_sign=sign_p(w, len(T)))
    out[ds] = rows
    print(ds, len(T), 'turbines; fewest wins:', min((v['wins'], k) for k, v in rows.items()))
(C.OUT/'TURBINE_LEVEL.json').write_text(json.dumps(out, indent=1))
