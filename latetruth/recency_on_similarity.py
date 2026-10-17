"""Amendment 18: ours+rec = P_ours + 0.5 * EWMA(h = 18 rows) of the released residuals of P_ours (frozen on Kelmarsh 2017), and the same
with the Amendment 14 gate on hold rows. Common rows of Table 7; S1 statistics vs ours, ref ewma, gate. Writes rec_<ds>.npz, recg_<ds>.npz,
RECENCY_ON_SIMILARITY.json. Run after residual_references.py."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import residual_references as R
O = C.OUT; EV0 = C.EV0; H, CC = 18, 0.5


def main():
    out = {'h': H, 'c': CC}; rng = np.random.default_rng(20261018)
    for ds in ['kelmarsh2017'] + R.EVAL:
        z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); g = np.load(O/f'gate_{ds}.npz', allow_pickle=False); e = np.load(O/f'ref_ewma_{ds}.npz', allow_pickle=False)
        T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); P = {'ours': {}, 'rec': {}, 'gate': {}, 'recg': {}, 'ewma': {}}
        for t in T:
            Y, OK, PO = z[f'Y_{t}'], z[f'OK_{t}'], z[f'P_ours_{t}']; ev = np.arange(EV0, len(Y)); ok = OK[ev] & np.isfinite(PO[ev])
            r = np.where(ok, Y[ev] - PO[ev], 0.0); rec = PO.copy(); rec[ev] = PO[ev] + CC * R.ewma(r, ok, H)
            G = g[f'P_{t}']; hold = np.abs(G - PO) > 1e-12
            P['ours'][t], P['rec'][t], P['gate'][t], P['recg'][t], P['ewma'][t] = PO, rec, G, np.where(hold, G, rec), e[f'P_{t}']
        np.savez_compressed(O/f'rec_{ds}.npz', **{f'P_{t}': P['rec'][t] for t in T}); np.savez_compressed(O/f'recg_{ds}.npz', **{f'P_{t}': P['recg'][t] for t in T})
        month, X, Y, OK = C.load(ds); ev = np.arange(EV0, len(month)); mon = month[ev]
        if ds == 'kelmarsh2017': common = {t: OK[t][ev] & np.isfinite(P['ours'][t][ev]) for t in T}
        else:
            M = BT.load_methods(ds, T); keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
            common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        turb = lambda k: np.array([np.abs(Y[t][ev] - P[k][t][ev])[common[t]].mean() for t in T])
        tm = lambda k: np.array([[np.abs(Y[t][ev] - P[k][t][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T])
        res = {k: float(turb(k).mean()) for k in P}; cmp = {}
        for a, b in [('rec', 'ours'), ('rec', 'ewma'), ('recg', 'gate'), ('recg', 'ewma')]:
            At, Bt = turb(a), turb(b); d = (tm(b) - tm(a)).ravel(); d = d[np.isfinite(d)]
            boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            cmp[f'{a}_vs_{b}'] = dict(diff=float(Bt.mean() - At.mean()), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                                      wins=f'{int((d > 0).sum())}/{d.size}', p=float(wilcoxon(d).pvalue))
        out[ds] = dict(mae=res, comparisons=cmp)
        print(ds, {k: round(v, 4) for k, v in res.items()}, {k: (round(v['diff'], 4), [round(x, 4) for x in v['ci95']], v['wins']) for k, v in cmp.items()}, flush=True)
    out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (O/'RECENCY_ON_SIMILARITY.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
