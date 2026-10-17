"""Table I statistics for every turbine-year (reproduces the hand-entered values of the first four exactly):
usable evaluation rows (OK, i.e. valid current and lagged target), oil mean (sd) on those rows, and the median / 90th percentile of
|y(t) - y(t-3)|. Writes DATASET_STATS.json. Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
out = {}
for ds in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019', 'kelmarsh2024', 'penmanshiel2022', 'hot2025', 'kelmarsh2020', 'penmanshiel2020', 'hot2022']:
    month, X, Y, OK = C.load(ds); T = sorted(Y); ev = np.arange(C.EV0, len(month))
    y = np.concatenate([Y[t][ev][OK[t][ev]] for t in T])
    d = np.concatenate([np.abs(Y[t][ev] - Y[t][ev - 3])[OK[t][ev]] for t in T]); d = d[np.isfinite(d)]
    out[ds] = dict(turbines=len(T), rows=len(month), usable_pct=100 * float(np.mean([OK[t][ev].mean() for t in T])), oil_mean=float(np.nanmean(y)),
                   oil_sd=float(np.nanstd(y)), d30_median=float(np.median(d)), d30_p90=float(np.percentile(d, 90)))
    print(ds, {k: round(v, 2) for k, v in out[ds].items()})
(C.OUT/'DATASET_STATS.json').write_text(json.dumps(out, indent=1))
# common rows of all receipt-causal methods (Section V-D claim: at least 99.5 % of the evaluation rows of every turbine)
from latetruth import baselines_table as BT
for ds in out:
    month, X, Y, OK = C.load(ds); T = sorted(Y); ev = np.arange(C.EV0, len(month)); M = BT.load_methods(ds, T)
    keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
    sh = [float((OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0)).sum() / OK[t][ev].sum()) for t in T]
    out[ds]['common_min_turbine_pct'] = 100 * min(sh); print(ds, 'common min per turbine %.2f %%' % (100 * min(sh)))
(C.OUT/'DATASET_STATS.json').write_text(json.dumps(out, indent=1))
