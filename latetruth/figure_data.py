"""Figure data for the paper (run with Python): monthly MAE of every turbine on the common receipt-causal rows for the proposed
estimator, the fleet model alone, the Kalman filter with fixed networks (Wang 2022), Yan 2023 and Kim 2013, on the three held-out
turbine-years. Same row set and scoring as baselines_table.py / significance.py. Writes FIGDATA_turbine_month.json."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
KEYS = {'ours': BT.PROPOSED, 'similarity': 'ours (fleet + local late-truth correction)', 'fleet': 'our fleet model (no late truth)', 'wang': 'B10f Wang 2022 (frozen NN)',
        'yan': 'B7 Yan 2023 [v2]', 'kim': 'B6 Kim 2013 [lag 3]'}
out = {}
for ds in ['kelmarsh2018', 'penmanshiel2019', 'hot2019']:
    month, X, Y, OK = C.load(ds); T = sorted(Y); n = len(month); ev = np.arange(C.EV0, n); M = BT.load_methods(ds, T, extras=True)
    keys = [k for k in M if '[lag 1]' not in k and 'online' not in k and k not in BT.EXTRA]
    common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}; mon = month[ev]
    out[ds] = {s: [[float(np.abs(Y[t][ev] - M[k][t][ev])[common[t] & (mon == m)].mean()) for m in range(2, 13)] for t in T] for s, k in KEYS.items()}
    out[ds]['turbines'] = T
    print(ds, {s: round(float(np.mean(v)), 4) for s, v in out[ds].items() if s != 'turbines'}, flush=True)
out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
(C.OUT/'FIGDATA_turbine_month.json').write_text(json.dumps(out, indent=1))
