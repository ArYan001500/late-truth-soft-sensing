"""Amendment 12 reporting: Cao et al. 2026 (v1 = paper settings, v2 = lr 1e-3) on the turbines where it was run
(Kelmarsh 2018: all 6; Penmanshiel 2019: 1, 2, 4, 5, 6, 7), against ours and every baseline ON THAT SAME TURBINE SUBSET and the
common rows of that subset (Cao included). S1 rule (Wilcoxon on turbine x month pairs + Holm, turbine bootstrap CI, own seed).
Writes CAO_TABLE.json. Pure numpy/scipy; run with Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import significance as S
EST = BT.PROPOSED if 'proposed' in sys.argv else S.OURS   # 'proposed': the estimator after the 2026-10-17 revision -> CAO_TABLE_PROPOSED.json
O = C.OUT; SUBSET = {'kelmarsh2018': [1, 2, 3, 4, 5, 6], 'penmanshiel2019': [1, 2, 4, 5, 6, 7]}


def main():
    out = {}; rng = np.random.default_rng(20261017)
    for ds, T in SUBSET.items():
        month, X, Y, OK = C.load(ds); n = len(month); ev = np.arange(C.EV0, n); M = BT.load_methods(ds, sorted(Y), extras=EST != S.OURS)
        M = {k: {t: v[t] for t in T} for k, v in M.items() if '[lag 1]' not in k and 'online' not in k and (k not in BT.EXTRA or k == EST)}
        for v, pre in [('v1', 'cao'), ('v2', 'caov2')]:
            M[f'B12 Cao 2026 [{v}]'] = {t: np.load(O/f'{pre}_{ds}_T{t}.npz', allow_pickle=False)['P'] for t in T}
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in M], 0) for t in T}
        share = float(np.mean([common[t].sum() / OK[t][ev].sum() for t in T])); mon = month[ev]
        tm = lambda P: np.array([[np.abs(Y[t][ev] - P[t][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T])
        tb = lambda P: np.array([np.abs(Y[t][ev] - P[t][ev])[common[t]].mean() for t in T])
        A, At = tm(M[EST]), tb(M[EST]); rows, ps = [], []
        for k in M:
            if k == EST: continue
            B, Bt = tm(M[k]), tb(M[k]); d = (B - A).ravel(); d = d[np.isfinite(d)]
            p = float(S.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(method=k, mae_common=float(Bt.mean()), mae_diff=float(Bt.mean() - At.mean()), ci95=[float(lo), float(hi)],
                             wins=int((d > 0).sum()), n_pairs=int(d.size), p=p, cao_months=[round(float(x), 4) for x in np.nanmean(B, 0)] if 'Cao' in k else None)); ps.append(p)
        for r, adj in zip(rows, S.holm(ps)): r['p_holm'] = float(adj); r['ours_better'] = bool(adj < 0.05 and r['ci95'][0] > 0)
        out[ds] = dict(turbines=T, common_share=share, ours_mae_common=float(At.mean()), vs=sorted(rows, key=lambda r: r['mae_common']))
        print(f'\n{ds} turbines {T}  common share {share:.4f}  ours {At.mean():.4f}')
        for r in out[ds]['vs']: print(f"  {r['method']:42s} {r['mae_common']:.4f}  diff {r['mae_diff']:+.4f} CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] wins {r['wins']}/{r['n_pairs']} {'BETTER' if r['ours_better'] else 'not shown'}")
    (O/('CAO_TABLE.json' if EST == S.OURS else 'CAO_TABLE_PROPOSED.json')).write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
