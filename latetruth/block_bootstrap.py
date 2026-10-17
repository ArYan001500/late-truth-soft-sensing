"""Amendment 17 (b): dependence-aware sensitivity. Paired bootstrap of whole calendar blocks of the evaluation period (the same
blocks for all turbines), blocks of 7 days (primary) and 28 days; 2000 resamples; 95 % percentile CI of the fleet-mean MAE
difference (baseline - ours) on the Table-7 common rows. Conditional on the fitted models. Includes the gate and the Amendment 17
references. Writes BLOCK_BOOTSTRAP.json. Run after residual_references.py."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth.significance import OURS
O = C.OUT; EV0 = C.EV0; B = 2000


def main():
    out = {}; rng = np.random.default_rng(20261017)
    for ds in ['kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER:
        month, X, Y, OK = C.load(ds); T = sorted(Y); n = len(month); ev = np.arange(EV0, n); M = BT.load_methods(ds, T)
        keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        for nm in ['gate', 'ref_mean20', 'ref_ewma']:
            z = np.load(O/f'{nm}_{ds}.npz', allow_pickle=False); M[nm] = {t: z[f'P_{t}'] for t in T}; keys.append(nm)
        out[ds] = {}
        for days in (7, 28):
            blk = (np.arange(len(ev)) // (144 * days)); nb = blk.max() + 1
            def sums(k):   # turbine x block: sum of |error| and count on common rows
                S = np.zeros((len(T), nb)); N = np.zeros((len(T), nb))
                for i, t in enumerate(T):
                    m = common[t]; np.add.at(S[i], blk[m], np.abs(Y[t][ev] - M[k][t][ev])[m]); np.add.at(N[i], blk[m], 1)
                return S, N
            SA, N = sums(OURS); idx = rng.integers(0, nb, (B, nb)); res = []
            cnt = np.stack([N[:, j].sum(1) for j in idx])            # B x turbines
            A = np.stack([SA[:, j].sum(1) for j in idx]) / cnt
            for k in keys:
                if k == OURS: continue
                SB, _ = sums(k); Bm = np.stack([SB[:, j].sum(1) for j in idx]) / cnt
                d = (Bm - A).mean(1); lo, hi = np.percentile(d, [2.5, 97.5])
                res.append(dict(method=k, mae_diff=float((SB.sum(1) / N.sum(1) - SA.sum(1) / N.sum(1)).mean()), ci95=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0)))
            out[ds][f'{days}d'] = dict(n_blocks=int(nb), rows=sorted(res, key=lambda r: r['mae_diff']))
            print(ds, f'{days}d', nb, 'blocks;', ' | '.join(f"{r['method'][:14]} {r['mae_diff']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]" for r in out[ds][f'{days}d']['rows'][:4]), flush=True)
    out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (O/'BLOCK_BOOTSTRAP.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
