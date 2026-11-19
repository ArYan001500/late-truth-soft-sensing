from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part G (+ G-a, G-b): scores on the rows where all 40 estimators of a farm-year return
an estimate; decomposition per architecture. usage: python revision_controlled_summary.py"""
import json, hashlib, itertools
import numpy as np
from latetruth import baselines_common as BC
from latetruth import revision_turbine_id as RT

OUT = RT.OUT; EV0 = BC.EV0; ARCH = ['hgb', 'xgb', 'ridge', 'mlp', 'gru']; DS = ['kelmarsh2018', 'penmanshiel2019', 'hot2019']
CELLS = [f'{i}|{p}|{c}' for i, p, c in itertools.product(['full', 'base'], ['fleet', 'pt'], ['none', 'corr'])]


def main():
    out = {}
    for ds in DS:
        if ds.startswith('hot'): import os; os.environ['HOT_YEAR'] = '2019'
        z0 = np.load(BC.OUT/f'ours_{ds}.npz', allow_pickle=False); month = z0['month']; n = len(month); ev = np.arange(EV0, n)
        T = sorted(int(k.split('_')[1]) for k in z0.files if k.startswith('Y_')); Y = {t: z0[f'Y_{t}'] for t in T}; OK = {t: z0[f'OK_{t}'] for t in T}
        Z = {a: np.load(OUT/f'controlled_{ds}_{a}.npz', allow_pickle=False) for a in ARCH}
        info = {a: json.loads((OUT/f'controlled_{ds}_{a}.json').read_text())['info'] for a in ARCH}
        assert info['hgb']['repro_max_abs_fleet'] <= 1e-9 and info['hgb']['repro_max_abs_proposed'] <= 1e-9
        common = {t: OK[t][ev] & np.all([np.isfinite(Z[a][f'{c}_{t}'][ev]) for a in ARCH for c in CELLS], 0) for t in T}
        mae = {a: {c: float(np.mean([np.abs(Y[t][ev] - Z[a][f'{c}_{t}'][ev])[common[t]].mean() for t in T])) for c in CELLS} for a in ARCH}
        mon = month[ev]
        tm = lambda a, c: np.array([[np.abs(Y[t][ev] - Z[a][f'{c}_{t}'][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T]).ravel()
        dec = {}
        for a in ARCH:
            ref = 'full|fleet|corr'; r = mae[a][ref]
            d = {'inputs_pct': 100 * (mae[a]['base|fleet|corr'] / r - 1), 'pooling_pct': 100 * (mae[a]['full|pt|corr'] / r - 1),
                 'correction_pct': 100 * (mae[a]['full|fleet|none'] / r - 1)}
            for f, (x, y) in {'inputs': ('full', 'base'), 'pooling': ('fleet', 'pt'), 'correction': ('corr', 'none')}.items():
                pos = {'inputs': 0, 'pooling': 1, 'correction': 2}[f]
                good = [c for c in CELLS if c.split('|')[pos] == x]; bad = ['|'.join(y if k == pos else s for k, s in enumerate(c.split('|'))) for c in good]
                d[f'{f}_main_effect_pct'] = float(np.mean([100 * (mae[a][b] / mae[a][g] - 1) for g, b in zip(good, bad)]))
            for f, b in [('inputs', 'base|fleet|corr'), ('pooling', 'full|pt|corr'), ('correction', 'full|fleet|none')]:
                dd = tm(a, b) - tm(a, ref); dd = dd[np.isfinite(dd)]; d[f'{f}_wins'] = f'{int((dd > 0).sum())}/{dd.size}'
            dec[a] = d
        best = min(((a, c) for a in ARCH for c in CELLS), key=lambda k: mae[k[0]][k[1]])
        out[ds] = dict(mae=mae, decomposition=dec, best=list(best), rows_common_share=float(np.mean([common[t].mean() for t in T])))
        print(ds, 'best', best, round(mae[best[0]][best[1]], 4), 'common share', round(out[ds]['rows_common_share'], 3), flush=True)
        for a in ARCH:
            print(f'  {a:5s} ref {mae[a]["full|fleet|corr"]:.4f} | ' + ' '.join(f'{c}:{mae[a][c]:.4f}' for c in CELLS if c != 'full|fleet|corr'))
            print(f'        dec ' + ' '.join(f'{k}={v:.1f}' if isinstance(v, float) else f'{k}={v}' for k, v in dec[a].items()))
    (OUT/'controlled_summary.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part G summary', **out,
        code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__': main()
