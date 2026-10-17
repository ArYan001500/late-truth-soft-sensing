"""Companion of discussion_recency_vs_similarity.py (same Amendment 13): turbine-month comparison of similar20 (proposed) against
last20 (20 most recent released residuals). Counts turbine-months where similar20 has the lower MAE, with an exact two-sided sign
test, and the per-farm MAE difference. Writes DISCUSSION_RECENCY_VS_SIMILARITY_WINS.json.   usage: python <this> (numpy only)"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib, math
import numpy as np
from latetruth import discussion_recency_vs_similarity as R   # re-runs the main diagnostic (identical numbers) and exposes recency()


def sign_test(w, n):
    k = min(w, n - w); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


out = {}
for ds in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019']:
    z = np.load(R.O/f'ours_{ds}.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); month = z['month'] if 'month' in z.files else None
    wins = n = 0
    for t in T:
        Y, OK, PF, PO = z[f'Y_{t}'], z[f'OK_{t}'], z[f'P_fleet_{t}'], z[f'P_ours_{t}']; ev = np.arange(R.EV0, len(Y))
        ok = OK[ev] & np.isfinite(PF[ev]); r = np.where(ok, Y[ev] - PF[ev], 0.0); P20 = PF[ev] + R.recency(r, ok, R.K)
        mon = month[ev] if month is not None else (np.arange(len(ev)) // 4380)
        for m in range(2, 13):   # February-December, the months of the paired tests in the paper
            s = ok & (mon == m) & np.isfinite(PO[ev])
            if s.sum() < 100: continue
            a, b = np.abs(Y[ev] - PO[ev])[s].mean(), np.abs(Y[ev] - P20)[s].mean(); n += 1; wins += int(a < b)
    out[ds] = dict(wins_similar_over_last20=wins, turbine_months=n, sign_test_p=sign_test(wins, n), month_source='file' if month is not None else 'approx')
    print(ds, out[ds], flush=True)
out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
(R.O/'DISCUSSION_RECENCY_VS_SIMILARITY_WINS.json').write_text(json.dumps(out, indent=2))
