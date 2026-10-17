"""Amendment 11 S4: sensitivity of the local late-truth correction (Kelmarsh 2018, saved fleet predictions from ours_kelmarsh2018.npz).
k in {5,10,20,40,80} x memory in {1440,4320,8640,17280}; shrink k0 = 20, delay 3. The pre-chosen setting (k 20, memory 8640) is not re-selected."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import json, itertools
from multiprocessing import Pool
import numpy as np
from latetruth import baselines_common as BC
from latetruth.run_local_late_truth_long import HELP, local
DS = 'kelmarsh2018'


def one(cfg):
    k, mem = cfg; D, n, month, Y, OK = BC.load(DS); z = np.load(BC.OUT/f'ours_{DS}.npz', allow_pickle=False); P = {}
    for t, (df, _) in D.items():
        pf = z[f'P_fleet_{t}']; ev = np.arange(BC.EV0, n); ok = OK[t][ev] & np.isfinite(pf[ev]); r = np.where(ok, Y[t][ev] - pf[ev], 0.0)
        H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:BC.EV0]; Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        P[t] = np.full(n, np.nan); P[t][ev] = pf[ev] + local(Z, r, ok, mem, k=k)
    return k, mem, BC.score(Y, P, OK, month, n)['all']


if __name__ == '__main__':
    grid = list(itertools.product([5, 10, 20, 40, 80], [1440, 4320, 8640, 17280]))
    with Pool(min(5, JOBS)) as pool: res = pool.map(one, grid)
    for k, mem, m in res: print(f'k {k:3d} memory {mem:6d} MAE {m:.4f}', flush=True)
    (BC.OUT/'LOCAL_GRID_kelmarsh2018.json').write_text(json.dumps([dict(k=k, memory=mem, mae=m) for k, mem, m in res], indent=2))
