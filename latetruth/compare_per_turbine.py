"""Amendment 21 scoring: proposed vs per-turbine + late-truth correction, and fleet model alone vs per-turbine model alone, on the
Table-7 common rows; S1 rule (turbine-month Wilcoxon, Holm over the two comparisons, turbine bootstrap) and 7/28-day block bootstrap.
Writes PER_TURBINE_CONTROL.json. Run after per_turbine_matched.py for all farm-years."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth.significance import holm
O = C.OUT; EV0 = C.EV0; PAIRS = [(BT.PROPOSED, 'per-turbine model + late-truth correction'), ('our fleet model (no late truth)', 'per-turbine model, same inputs')]
out = {}; rng = np.random.default_rng(20261021)
for ds in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER + BT.CONFIRM:
    month, X, Y, OK = C.load(ds); T = sorted(Y); ev = np.arange(EV0, len(month)); M = BT.load_methods(ds, T, extras=True)
    keys = [k for k in M if '[lag 1]' not in k and 'online' not in k and k not in BT.EXTRA]
    common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}; mon = month[ev]
    for a, b in PAIRS: assert all(np.array_equal(common[t], common[t] & np.isfinite(M[b][t][ev])) for t in T), (ds, b)
    tb = lambda k: np.array([np.abs(Y[t][ev] - M[k][t][ev])[common[t]].mean() for t in T])
    tm = lambda k: np.array([[np.abs(Y[t][ev] - M[k][t][ev])[common[t] & (mon == m)].mean() if (common[t] & (mon == m)).any() else np.nan for m in range(2, 13)] for t in T])
    rows, ps = [], []
    for a, b in PAIRS:
        At, Bt = tb(a), tb(b); d = (tm(b) - tm(a)).ravel(); d = d[np.isfinite(d)]
        boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
        row = dict(pooled=a, per_turbine=b, mae_pooled=float(At.mean()), mae_per_turbine=float(Bt.mean()), diff=float(Bt.mean() - At.mean()),
                   pct=float(100 * (1 - At.mean() / Bt.mean())), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])], wins=int((d > 0).sum()),
                   n_pairs=int(d.size), turbines_pooled_lower=int((Bt > At).sum()), n_turbines=len(T), p=float(wilcoxon(d).pvalue))
        for days in (7, 28):
            blk = np.arange(len(ev)) // (144 * days); nb = blk.max() + 1; S = {}
            for k in (a, b):
                SS = np.zeros((len(T), nb)); NN = np.zeros((len(T), nb))
                for i, t in enumerate(T):
                    m = common[t]; np.add.at(SS[i], blk[m], np.abs(Y[t][ev] - M[k][t][ev])[m]); np.add.at(NN[i], blk[m], 1)
                S[k] = SS
            idx = rng.integers(0, nb, (2000, nb)); cnt = np.stack([NN[:, j].sum(1) for j in idx])
            dd = np.nanmean(np.stack([S[b][:, j].sum(1) for j in idx]) / cnt - np.stack([S[a][:, j].sum(1) for j in idx]) / cnt, 1)
            row[f'block{days}d_ci95'] = [float(x) for x in np.percentile(dd, [2.5, 97.5])]
        rows.append(row); ps.append(row['p'])
    for r_, h in zip(rows, holm(ps)): r_['p_holm'] = float(h); r_['pooled_better'] = bool(h < 0.05 and r_['ci95'][0] > 0)
    out[ds] = rows
    print(ds, [(r_['per_turbine'][:22], round(r_['mae_pooled'], 4), round(r_['mae_per_turbine'], 4), f"{r_['pct']:.1f}%", [round(x, 4) for x in r_['ci95']],
               [round(x, 4) for x in r_['block7d_ci95']], f"{r_['wins']}/{r_['n_pairs']}", r_['pooled_better']) for r_ in rows], flush=True)
out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
(O/'PER_TURBINE_CONTROL.json').write_text(json.dumps(out, indent=2))
