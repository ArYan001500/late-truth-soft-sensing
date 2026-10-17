"""Amendment 11 S1: ours vs every baseline, turbine x month MAE pairs on the common rows, Wilcoxon signed-rank + Holm, and a
turbine-cluster bootstrap CI of the overall MAE difference. Datasets: kelmarsh2018, penmanshiel2019, hot2019. Run with Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
OURS = 'ours (fleet + local late-truth correction)'


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


def main():
    out = {}; rng = np.random.default_rng(20261015)
    for ds in ['kelmarsh2018', 'penmanshiel2019', 'hot2019'] + [d for d in BT.LATER if (C.OUT/f'ours_{d}.npz').exists()]:
        month, X, Y, OK = C.load(ds); T = sorted(Y); n = len(month); ev = np.arange(C.EV0, n); M = BT.load_methods(ds, T)
        keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        mon = month[ev]
        def tm(k):   # turbine x month MAE on common rows
            return np.array([[np.abs(Y[t][ev] - M[k][t][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T])
        def turb(k): return np.array([np.abs(Y[t][ev] - M[k][t][ev])[common[t]].mean() for t in T])
        A, At = tm(OURS), turb(OURS); rows, ps = [], []
        for k in keys:
            if k == OURS: continue
            B, Bt = tm(k), turb(k); d = (B - A).ravel(); d = d[np.isfinite(d)]
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(baseline=k, mae_diff=float(Bt.mean() - At.mean()), ci95=[float(lo), float(hi)], wins_turbine_month=int((d > 0).sum()), n_pairs=int(d.size), p=p)); ps.append(p)
        for r, a in zip(rows, holm(ps)): r['p_holm'] = float(a); r['ours_better'] = bool(a < 0.05 and r['ci95'][0] > 0)
        out[ds] = sorted(rows, key=lambda r: r['mae_diff'])
        print(f'\n{ds}  ({len(T)} turbines)')
        for r in out[ds]: print(f"  {r['baseline']:42s} diff {r['mae_diff']:+.4f}  CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  wins {r['wins_turbine_month']}/{r['n_pairs']}  p_holm {r['p_holm']:.1e}  {'BETTER' if r['ours_better'] else 'not shown'}")
    (C.OUT/'SIGNIFICANCE.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
