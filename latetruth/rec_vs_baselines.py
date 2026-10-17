"""Amendment 19: ours, ours+rec and ours+rec+gate vs every Table-8 baseline, S1 rule (turbine-month Wilcoxon + Holm within farm-year
and estimator, turbine bootstrap 2000) and 7/28-day block bootstrap (same blocks for all turbines), on the Table-7 common rows.
usage: python rec_vs_baselines.py [ds ...]  (default: the six farm-years of Table 7). Writes REC_VS_BASELINES_<tag>.json."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import residual_references as R
from latetruth import standstill_gate as G
from latetruth.significance import holm, OURS
O = C.OUT; EV0 = C.EV0


def variants(ds, T, M):
    """rec / recg predictions: saved files for the Table-7 years, recomputed with the frozen rule otherwise (same code as confirm)."""
    if (O/f'rec_{ds}.npz').exists():
        a, b = np.load(O/f'rec_{ds}.npz', allow_pickle=False), np.load(O/f'recg_{ds}.npz', allow_pickle=False); return {t: a[f'P_{t}'] for t in T}, {t: b[f'P_{t}'] for t in T}
    sel = json.load(open(O/'GATE.json'))['selection']; month, Y, OK, _, yr, s, nac = G.prep(ds); ev = np.arange(EV0, len(month)); rec, recg = {}, {}
    for t in T:
        PO = M[OURS][t]; ok = OK[t][ev] & np.isfinite(PO[ev]); r = np.where(ok, Y[t][ev] - PO[ev], 0.0)
        q = np.full(len(month), np.nan); q[ev] = PO[ev] + 0.5 * R.ewma(r, ok, 18); rec[t] = q
        g = G.gated(PO, yr[t], s[t], nac[t], sel['chosen']['form'], sel['chosen']['h'], sel['a']); recg[t] = np.where(np.abs(g - PO) > 1e-12, g, q)
    return rec, recg


def main():
    DS = sys.argv[1:] or ['kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER; out = {}
    if DS == ['all9']: DS = ['kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER + BT.CONFIRM; rng = np.random.default_rng(20261019)
    for ds in DS:
        month, X, Y, OK = C.load(ds); T = sorted(Y); n = len(month); ev = np.arange(EV0, n); M = BT.load_methods(ds, T)
        keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        M['ours+rec'], M['ours+rec+gate'] = variants(ds, T, M); mon = month[ev]
        tm = lambda k: np.array([[np.abs(Y[t][ev] - M[k][t][ev])[common[t] & (mon == m)].mean() if (common[t] & (mon == m)).any() else np.nan for m in range(2, 13)] for t in T])
        turb = lambda k: np.array([np.abs(Y[t][ev] - M[k][t][ev])[common[t]].mean() for t in T])
        base = [k for k in keys if k not in (OURS, 'our fleet model (no late truth)')]
        blocks = {d: (np.arange(len(ev)) // (144 * d)) for d in (7, 28)}
        def bsum(k, blk, nb):
            S = np.zeros((len(T), nb)); N = np.zeros((len(T), nb))
            for i, t in enumerate(T):
                m = common[t]; np.add.at(S[i], blk[m], np.abs(Y[t][ev] - M[k][t][ev])[m]); np.add.at(N[i], blk[m], 1)
            return S, N
        out[ds] = {}
        for est in [OURS, 'ours+rec', 'ours+rec+gate']:
            A, At = tm(est), turb(est); rows, ps = [], []
            for k in base + ([OURS] if est != OURS else []):
                B, Bt = tm(k), turb(k); d = (B - A).ravel(); d = d[np.isfinite(d)]
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
                row = dict(baseline=k, mae=float(Bt.mean()), diff=float(Bt.mean() - At.mean()), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                           wins=int((d > 0).sum()), n_pairs=int(d.size), turbines_lower=int((Bt > At).sum()), n_turbines=len(T), p=p)
                for days, blk in blocks.items():
                    nb = blk.max() + 1; SA, N = bsum(est, blk, nb); SB, _ = bsum(k, blk, nb); idx = rng.integers(0, nb, (2000, nb))
                    cnt = np.stack([N[:, j].sum(1) for j in idx]); dd = np.nanmean(np.stack([SB[:, j].sum(1) for j in idx]) / cnt - np.stack([SA[:, j].sum(1) for j in idx]) / cnt, 1)
                    row[f'block{days}d_ci95'] = [float(x) for x in np.percentile(dd, [2.5, 97.5])]
                rows.append(row); ps.append(p)
            for r_, a in zip(rows, holm(ps)): r_['p_holm'] = float(a); r_['better'] = bool(a < 0.05 and r_['ci95'][0] > 0); r_['better_block7'] = bool(a < 0.05 and r_['block7d_ci95'][0] > 0)
            out[ds][est] = dict(mae=float(At.mean()), vs=sorted(rows, key=lambda r_: r_['diff']))
            nb_ = [r_['baseline'][:20] for r_ in rows if not r_['better']]; nb7 = [r_['baseline'][:20] for r_ in rows if not r_['better_block7']]
            print(f"{ds:16s} {est[:14]:14s} MAE {At.mean():.4f}  better than {sum(r_['better'] for r_ in rows)}/{len(rows)}  not: {nb_}  | block7 not: {nb7}", flush=True)
    out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    tag = 'table7' if not sys.argv[1:] else ('all9' if sys.argv[1:] == ['all9'] else '_'.join(DS))
    (O/f'REC_VS_BASELINES_{tag}.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
