"""Amendment 11 S2: independent recomputation of FINAL_TABLE_all.json from the saved prediction files.
Own loading and own MAE code (no import of baselines_classic / baselines_table / baselines_common). Pass = every |diff| < 1e-9."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, glob, re
from pathlib import Path
import numpy as np
O = OUT_DIR/'published_baselines_2026-10-14_v001'; EV0 = 4032
NAMES = {'B0': 'B0 last released value', 'B1': 'B1 Bangalore 2015', 'B2': 'B2 Schlechtingen 2011 AR', 'B2f': 'B2f Schlechtingen 2011 FSRC',
         'B3': 'B3 Liu 2020', 'B5': 'B5 Yang 2021', 'B6': 'B6 Kim 2013', 'B4': 'B4 Fu 2019', 'B7': 'B7 Yan 2023', 'B8': 'B8 Dai 2025', 'B9': 'B9 Grataloup 2025'}


EXTRA = {'rec': 'proposed (similarity + recency)', 'recg': 'proposed + standstill gate', 'ref_mean20': 'reference: mean of 20 newest residuals',
         'ref_ewma': 'reference: EWMA of residuals', 'ptfleet': 'per-turbine model, same inputs', 'ptrec': 'per-turbine model + late-truth correction'}   # revision of 2026-10-17; they do not shape the common rows


def rec_check(ds, T, OK):
    """Own re-implementation of the recency term (EWMA of the released residuals of the similarity-corrected estimate, half-life
    18 rows, weight 1/2, residual of row s usable at row i only if s < i - 3, more than 20 released residuals) against rec_<ds>.npz."""
    z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); r = np.load(O/f'rec_{ds}.npz', allow_pickle=False); worst = 0.0; lam = 0.5 ** (1 / 18)
    for i, t in enumerate(T):
        y, po = z[f'Y_{t}'], z[f'P_ours_{t}']; n = len(y); out = np.full(n, np.nan); S = W = 0.0; cnt = 0; last = None; q = []
        rel = [s for s in range(EV0, n) if OK[i][s] and np.isfinite(po[s])]; j = 0
        for k in range(EV0, n):
            while j < len(rel) and rel[j] < k - 3:
                s = rel[j]; dec = lam ** (s - last) if last is not None else 1.0; S = S * dec + (y[s] - po[s]); W = W * dec + 1.0; last = s; cnt += 1; j += 1
            out[k] = po[k] + (0.5 * S / W if cnt > 20 else 0.0)
        m = np.isfinite(out) & np.isfinite(r[f'P_{t}']); worst = max(worst, float(np.max(np.abs(out[m] - r[f'P_{t}'][m]))))
        assert np.array_equal(np.isfinite(out[EV0:]), np.isfinite(r[f'P_{t}'][EV0:]))
    return worst


def preds(ds, T):
    P = {}; z = np.load(O/f'ours_{ds}.npz', allow_pickle=False)
    P['ours (fleet + local late-truth correction)'] = [z[f'P_ours_{t}'] for t in T]; P['our fleet model (no late truth)'] = [z[f'P_fleet_{t}'] for t in T]
    for f in O.glob(f'*_{ds}.npz'):
        nm = f.name[:-len(f'_{ds}.npz')]
        if m := re.fullmatch(r'(B\d+f?)_lag(\d)', nm): P[f'{NAMES[m[1]]} [lag {m[2]}]'] = [np.load(f, allow_pickle=False)[f'P_{t}'] for t in T]
        elif nm == 'B11_bai_xgb': P['B11 Bai 2024'] = [np.load(f, allow_pickle=False)[f'P_{t}'] for t in T]
        elif nm == 'B10f_wang2022_frozen': P['B10f Wang 2022 (frozen NN)'] = [np.load(f, allow_pickle=False)[f'P_{t}'] for t in T]
        elif nm == 'B10_wang2022': P['B10 Wang 2022 (online, as published)'] = [np.load(f, allow_pickle=False)[f'P_{t}'] for t in T]
        elif nm in EXTRA: P[EXTRA[nm]] = [np.load(f, allow_pickle=False)[f'P_{t}'] for t in T]
    for b in ['B4', 'B7', 'B8', 'B9']:
        for v, pre in [('v1', 'deep_'), ('v2', 'deepv2_')]:
            fs = [O/f'{pre}{b}_{ds}_T{t}.npz' for t in T]
            if all(f.exists() for f in fs): P[f'{NAMES[b]} [{v}]'] = [np.load(f, allow_pickle=False)['P'] for f in fs]
    return P


def main():
    table = json.loads((O/'FINAL_TABLE_all.json').read_text()); worst, n = 0.0, 0; recw = 0.0
    for ds, rows in table.items():
        z = np.load(O/f'inputs_{ds}.npz', allow_pickle=False); T = sorted(int(k[2:]) for k in z.files if k.startswith('X_'))
        Y = [z[f'Y_{t}'] for t in T]; OK = [z[f'OK_{t}'] for t in T]; P = preds(ds, T)
        elig = [k for k in P if '[lag 1]' not in k and 'online' not in k and k not in EXTRA.values()]
        common = [OK[i][EV0:] & np.logical_and.reduce([np.isfinite(P[k][i][EV0:]) for k in elig]) for i in range(len(T))]
        if (O/f'rec_{ds}.npz').exists(): recw = max(recw, rec_check(ds, T, OK)); print(ds, 'recency term re-implemented, max |diff| so far', f'{recw:.2e}', flush=True)
        for r in rows:
            k = r['method']; mine = np.mean([np.mean(np.abs(Y[i][EV0:] - P[k][i][EV0:])[common[i]]) for i in range(len(T))])
            d = abs(mine - r['mae_common_rows']); worst = max(worst, d); n += 1
            if d >= 1e-9: print('MISMATCH', ds, k, mine, r['mae_common_rows'])
    print(f'checked {n} entries, max |diff| = {worst:.2e}; recency term max |diff| = {recw:.2e} ->', 'PASS' if worst < 1e-9 and recw < 1e-9 else 'FAIL')
    (O/'VERIFY_FINAL_TABLE.json').write_text(json.dumps({'entries': n, 'max_abs_diff': worst, 'recency_term_max_abs_diff': recw, 'pass': bool(worst < 1e-9 and recw < 1e-9)}, indent=2))


if __name__ == '__main__': main()
