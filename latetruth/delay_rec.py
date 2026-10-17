"""Delay study (S3) for the proposed estimator after the revision: the frozen recency term of Amendment 18 (EWMA, half-life 18 rows,
weight 1/2, released rule s < t - d) on top of the saved similarity-corrected estimates of each delay run (delay_ours_d<d>.npz).
Same metric and rows as DELAY_SENSITIVITY (own rows of each run). Writes DELAY_SENSITIVITY_rec.json and delay_rec_d<d>.npz."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib
import numpy as np
from latetruth import baselines_classic as C
from latetruth import residual_references as R
O = C.OUT; out = {}; DL = {**json.load(open(O/'DELAY_SENSITIVITY_part1.json')), **json.load(open(O/'DELAY_SENSITIVITY_part2.json'))}
for d in [1, 3, 6, 12, 24]:
    z = np.load(O/f'delay_ours_d{d}.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); month = z['month']; n = len(month)
    Y = {t: z[f'Y_{t}'] for t in T}; OK = {t: z[f'OK_{t}'] for t in T}; PO = {t: z[f'P_ours_{t}'] for t in T}; R.D = d; P = {}
    for t in T:
        ev = np.arange(C.EV0, n); ok = OK[t][ev] & np.isfinite(PO[t][ev]); r = np.where(ok, Y[t][ev] - PO[t][ev], 0.0)
        q = PO[t].copy(); q[ev] = PO[t][ev] + 0.5 * R.ewma(r, ok, 18); P[t] = q
    chk = C.score(Y, PO, OK, month)['all']; assert abs(chk - DL['ours'][str(d)]['all']) < 1e-9, (d, chk, DL['ours'][str(d)]['all'])
    out[str(d)] = C.score(Y, P, OK, month); np.savez_compressed(O/f'delay_rec_d{d}.npz', **{f'P_{t}': P[t] for t in T})
    print('d', d, 'similarity only', round(chk, 4), 'proposed', round(out[str(d)]['all'], 4), 'Wang', round(DL['Wang 2022 (frozen NN)'][str(d)]['all'], 4), flush=True)
out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
(O/'DELAY_SENSITIVITY_rec.json').write_text(json.dumps({'proposed': out}, indent=2))
