"""05_Protocols/CONFIRM_TWO_TIMESCALE_v001.md: frozen variants on untouched years (Kelmarsh 2020, Penmanshiel 2020, Hill of Towie 2022).
fleet, ours, ref mean20, ref ewma (h 18, c 0.75), ours+rec (h 18, c 0.5), gate (Am. 14, frozen: hold, h 72), ours+rec+gate.
Common rows = valid target and every variant finite. Primary: ours+rec vs ours and vs ref ewma, turbine-month Wilcoxon (Holm over the two)
and turbine bootstrap 2000; block bootstrap (7, 28 days) as sensitivity. Writes CONFIRM_TWO_TIMESCALE.json. Pure numpy/scipy."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from latetruth import baselines_classic as C
from latetruth import residual_references as R
from latetruth import standstill_gate as G
from latetruth.significance import holm
O = C.OUT; EV0 = C.EV0
DS = [d for d in ['kelmarsh2020', 'penmanshiel2020', 'hot2022'] if (O/f'ours_{d}.npz').exists() and (O/f'inputs_{d}.npz').exists()]


def main():
    sel = json.load(open(O/'GATE.json'))['selection']; form, h, a = sel['chosen']['form'], sel['chosen']['h'], sel['a']
    out = {'datasets': DS}; rng = np.random.default_rng(20261018)
    for ds in DS:
        z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); month, Y, OK, T, yr, s, nac = G.prep(ds); n = len(month); ev = np.arange(EV0, n); P = {}
        for t in T:
            assert np.allclose(z[f'Y_{t}'], Y[t], equal_nan=True)
            PF, PO = z[f'P_fleet_{t}'], z[f'P_ours_{t}']; ok = OK[t][ev] & np.isfinite(PF[ev])
            r = np.where(ok, Y[t][ev] - PF[ev], 0.0); ok2 = OK[t][ev] & np.isfinite(PO[ev]); r2 = np.where(ok2, Y[t][ev] - PO[ev], 0.0)
            v = {'fleet': PF, 'ours': PO}
            for nm, base, cor in [('mean20', PF, R.mean20(r, ok)), ('ewma', PF, 0.75 * R.ewma(r, ok, 18)), ('rec', PO, 0.5 * R.ewma(r2, ok2, 18))]:
                q = np.full(n, np.nan); q[ev] = base[ev] + cor; v[nm] = q
            v['gate'] = G.gated(PO, yr[t], s[t], nac[t], form, h, a)
            hold = np.abs(v['gate'] - PO) > 1e-12; v['rec+gate'] = np.where(hold, v['gate'], v['rec'])
            for k, q in v.items(): P.setdefault(k, {})[t] = q
        common = {t: OK[t][ev] & np.all([np.isfinite(P[k][t][ev]) for k in P], 0) for t in T}; mon = month[ev]
        turb = lambda k: np.array([np.abs(Y[t][ev] - P[k][t][ev])[common[t]].mean() for t in T])
        tm = lambda k: np.array([[np.abs(Y[t][ev] - P[k][t][ev])[common[t] & (mon == m)].mean() if (common[t] & (mon == m)).any() else np.nan for m in range(2, 13)] for t in T])
        res = {k: float(turb(k).mean()) for k in P}; cmp = {}
        for b in ['ours', 'ewma']:
            At, Bt = turb('rec'), turb(b); d = (tm(b) - tm('rec')).ravel(); d = d[np.isfinite(d)]
            boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            cmp[b] = dict(diff=float(Bt.mean() - At.mean()), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])], wins=int((d > 0).sum()),
                          n_pairs=int(d.size), turbines_rec_lower=int((Bt > At).sum()), n_turbines=len(T), p=float(wilcoxon(d).pvalue))
            for days in (7, 28):   # block bootstrap, same blocks for all turbines
                blk = np.arange(len(ev)) // (144 * days); nb = blk.max() + 1; S = {}
                for k in ('rec', b):
                    SS = np.zeros((len(T), nb)); NN = np.zeros((len(T), nb))
                    for i, t in enumerate(T):
                        m = common[t]; np.add.at(SS[i], blk[m], np.abs(Y[t][ev] - P[k][t][ev])[m]); np.add.at(NN[i], blk[m], 1)
                    S[k] = SS
                idx = rng.integers(0, nb, (2000, nb)); cnt = np.stack([NN[:, j].sum(1) for j in idx])
                dd = (np.stack([S[b][:, j].sum(1) for j in idx]) / cnt - np.stack([S['rec'][:, j].sum(1) for j in idx]) / cnt)
                dd = np.nanmean(dd, 1); cmp[b][f'block{days}d_ci95'] = [float(x) for x in np.percentile(dd, [2.5, 97.5])]
        for b, ph in zip(['ours', 'ewma'], holm([cmp['ours']['p'], cmp['ewma']['p']])):
            cmp[b]['p_holm'] = float(ph); cmp[b]['better'] = bool(ph < 0.05 and cmp[b]['ci95'][0] > 0)
        out[ds] = dict(mae=res, rec_vs=cmp, confirmed=bool(cmp['ours']['better'] and cmp['ewma']['better']),
                       common_share=float(np.mean([common[t].sum() / OK[t][ev].sum() for t in T])))
        print(ds, {k: round(v, 4) for k, v in res.items()}, flush=True)
        for b in cmp: print('   rec vs', b, round(cmp[b]['diff'], 4), [round(x, 4) for x in cmp[b]['ci95']], f"{cmp[b]['wins']}/{cmp[b]['n_pairs']}",
                            f"p_holm {cmp[b]['p_holm']:.1e}", 'block7', [round(x, 4) for x in cmp[b]['block7d_ci95']], 'BETTER' if cmp[b]['better'] else 'not shown', flush=True)
        print('   CONFIRMED' if out[ds]['confirmed'] else '   NOT confirmed', flush=True)
    out['n_confirmed'] = int(sum(out[d]['confirmed'] for d in DS))
    out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (O/'CONFIRM_TWO_TIMESCALE.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
