"""Independent recomputation of the recursive least-squares reference row of the comparison table (REVISION_CHECKS_v001
Part C) from its saved estimates, with the loading, common rows, and MAE code of verify_final_table.py (no import of
baselines_table / baselines_common / revision_rls). Pass = every |diff| < 1e-9."""
from latetruth.config import RAW_DIR
import json
import numpy as np
from latetruth import verify_final_table as V

R = RAW_DIR/'revision_checks_2026-11-17_v001'
KEY = 'reference: recursive least squares'


def main():
    ref = json.loads((R/'rls_reference.json').read_text()); table = json.loads((V.O/'FINAL_TABLE_all.json').read_text())
    worst, n = 0.0, 0
    for ds in table:
        if ds not in ref or KEY not in ref[ds]:
            continue
        z = np.load(V.O/f'inputs_{ds}.npz', allow_pickle=False); T = sorted(int(k[2:]) for k in z.files if k.startswith('X_'))
        Y = [z[f'Y_{t}'] for t in T]; OK = [z[f'OK_{t}'] for t in T]; P = V.preds(ds, T)
        elig = [k for k in P if '[lag 1]' not in k and 'online' not in k and k not in V.EXTRA.values()]
        common = [OK[i][V.EV0:] & np.logical_and.reduce([np.isfinite(P[k][i][V.EV0:]) for k in elig]) for i in range(len(T))]
        r = np.load(R/f'rls_{ds}.npz', allow_pickle=False); PR = [r[f'P_{t}'] for t in T]
        mine = float(np.mean([np.mean(np.abs(Y[i][V.EV0:] - PR[i][V.EV0:])[common[i]]) for i in range(len(T))]))
        d = abs(mine - ref[ds][KEY]); worst = max(worst, d); n += 1
        print(ds, f'{mine:.6f}', f'{ref[ds][KEY]:.6f}', 'MISMATCH' if d >= 1e-9 else 'ok', flush=True)
    out = {'entries': n, 'max_abs_diff': worst, 'pass': bool(n == 9 and worst < 1e-9)}
    (R/'VERIFY_RLS_REFERENCE.json').write_text(json.dumps(out, indent=1))
    print(f'checked {n} entries, max |diff| = {worst:.2e} ->', 'PASS' if out['pass'] else 'FAIL')


if __name__ == '__main__': main()
