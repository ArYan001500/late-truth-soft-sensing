"""B10f — Wang et al. 2022 with neural compensations fitted offline on January and frozen (pre-declared fallback, Amendment 2),
because the online-augmented version diverges over the year (B10_wang2022.json). Same synchronization and fusion code (baseline_wang2022.run)."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import json, hashlib, itertools
import numpy as np
from latetruth import baseline_wang2022 as W
from latetruth.baselines_common import OUT, EV0, DELAY, score

GRID = list(itertools.product([0.01, 0.05, 0.2, 1.0, 5.0, 25.0], [0.001, 0.01, 0.05, 0.2], [1e-4, 1e-3, 1e-2]))


def _turbine(args):
    yf, y, prm = args; rel = np.isfinite(y); mu, sd = np.nanmean(y[:EV0 - DELAY]), np.nanstd(y[:EV0 - DELAY])
    th = W.offline_nn(yf, y, mu, sd)
    return W.run(yf, y, rel, (*prm, 0.0), mu, sd, th_init=th, frozen=True)


def _val(prm):
    month, Y, OK, PF = W.load_npz('kelmarsh2017'); ve = int(np.flatnonzero(month == 4)[0]); e = []
    for t in Y:
        p = _turbine((PF[t][:ve], Y[t][:ve], prm)); ev = np.arange(EV0, ve); ok = OK[t][ev] & np.isfinite(p[ev])
        e.append(np.abs(Y[t][ev] - p[ev])[ok].mean())
    return float(np.mean(e))


if __name__ == '__main__':
    from multiprocessing import Pool
    f = OUT/'B10f_wang2022_frozen.json'
    if f.exists(): raise FileExistsError(f)
    with Pool(min(8, JOBS)) as pool: v = pool.map(_val, GRID)
    for prm, m in zip(GRID, v): print('tune', prm, round(m, 4), flush=True)
    i = int(np.argmin(v)); best = GRID[i]; print('chosen', best, round(v[i], 4), flush=True)
    res = {'tuning_val_mae_2017FebMar': v[i], 'params(q,rf,ri)': best}
    for ds in ['kelmarsh2017', 'kelmarsh2018']:
        month, Y, OK, PF = W.load_npz(ds)
        with Pool(min(6, JOBS)) as pool: out = pool.map(_turbine, [(PF[t], Y[t], best) for t in Y])
        P = dict(zip(Y, out)); sc = score(Y, P, OK, month, len(month)); res[ds] = sc
        print(ds, 'B10f Wang2022 frozen', round(sc['all'], 4), sc['months'], flush=True)
        np.savez_compressed(OUT/f'B10f_wang2022_frozen_{ds}.npz', **{f'P_{t}': p for t, p in P.items()})
    res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    res['dep_sha256'] = hashlib.sha256(open(W.__file__, 'rb').read()).hexdigest(); f.write_text(json.dumps(res, indent=2))
