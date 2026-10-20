"""Selection of the recency term on the development year (Kelmarsh 2017 only), as recorded in Amendment 18 of the published-baselines
protocol: EWMA of the released residuals of the similarity-corrected estimate, half-life h in {6, ..., 2304} rows, weight c in
{0.25, 0.5, 0.75, 1}; MAE over rows >= EV0 with a valid target. Writes RECENCY_TUNING_kelmarsh2017.json."""
import json, hashlib
import numpy as np
from latetruth import residual_references as R

CS = [0.25, 0.5, 0.75, 1.0]


def main():
    z = np.load(R.O/'ours_kelmarsh2017.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); grid = {}
    base = []
    for t in T:
        Y, OK, PO = z[f'Y_{t}'], z[f'OK_{t}'], z[f'P_ours_{t}']; ev = np.arange(R.EV0, len(Y)); ok = OK[ev] & np.isfinite(PO[ev])
        r = np.where(ok, Y[ev] - PO[ev], 0.0); base.append(float(np.abs(r[ok]).mean()))
        for h in R.HL:
            e = R.ewma(r, ok, h)
            for c in CS: grid.setdefault(f'h{h}_c{c}', []).append(float(np.abs(r - c * e)[ok].mean()))
    grid = {k: float(np.mean(v)) for k, v in grid.items()}; best = min(grid, key=grid.get)
    out = dict(similarity_only_mae=float(np.mean(base)), grid=grid, best=best, best_mae=grid[best],
               code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest())
    (R.O/'RECENCY_TUNING_kelmarsh2017.json').write_text(json.dumps(out, indent=2))
    print('similarity only', round(out['similarity_only_mae'], 5), '| best', best, round(grid[best], 5))


if __name__ == '__main__': main()
