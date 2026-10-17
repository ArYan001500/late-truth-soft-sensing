"""Discussion diagnostic (exploratory, post hoc; registered in protocols/PUBLISHED_BASELINES_v001.md as Amendment 13 before running).
Question: does the local correction gain come from choosing released residuals by state similarity, or would the same correction
with residuals chosen by recency do as well? Same saved fleet predictions (ours_<ds>.npz), same k = 20, shrink k0 = 20, memory 8640,
delay 3, residuals collected from EV0. Variants:
  fleet      : fleet model alone
  last1      : fleet + 1/2 * newest released residual r(t-3)            (recency, one value, same 1/2 shrink as k = k0)
  last20     : fleet + 20/40 * mean of the 20 newest released residuals (recency, same k and shrink as ours)
  similar20  : the proposed estimator (saved P_ours: 20 most similar released states)
MAE over rows >= EV0 with valid target, mean over turbines (same metric as baselines_common.score).  usage: python <this> (numpy only)"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib
from pathlib import Path
import numpy as np
O = OUT_DIR/'published_baselines_2026-10-14_v001'; EV0, D, K, K0, MEM = 4032, 3, 20, 20, 8640


def recency(r, ok, k):
    out = np.zeros(len(r)); idx = np.flatnonzero(ok)
    for i in range(len(r)):
        hi = i - D
        if hi <= 0: continue
        a, b = np.searchsorted(idx, max(hi - MEM, 0)), np.searchsorted(idx, hi); cand = idx[a:b]
        if len(cand) <= (k if k > 1 else 0): continue
        out[i] = r[cand[-k:]].mean() * (k / (k + K0) if k > 1 else 0.5)
    return out


res = {}
for ds in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019']:
    z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); acc = {v: [] for v in ['fleet', 'last1', 'last20', 'similar20']}
    for t in T:
        Y, OK, PF, PO = z[f'Y_{t}'], z[f'OK_{t}'], z[f'P_fleet_{t}'], z[f'P_ours_{t}']; n = len(Y); ev = np.arange(EV0, n)
        ok = OK[ev] & np.isfinite(PF[ev]); r = np.where(ok, Y[ev] - PF[ev], 0.0)
        P = {'fleet': PF[ev], 'last1': PF[ev] + recency(r, ok, 1), 'last20': PF[ev] + recency(r, ok, K), 'similar20': PO[ev]}
        for v, p in P.items(): m = ok & np.isfinite(p); acc[v].append(float(np.abs(Y[ev] - p)[m].mean()))
    res[ds] = {v: float(np.mean(a)) for v, a in acc.items()}
    print(ds, {v: round(x, 4) for v, x in res[ds].items()}, flush=True)
res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
(O/'DISCUSSION_RECENCY_VS_SIMILARITY.json').write_text(json.dumps(res, indent=2))
