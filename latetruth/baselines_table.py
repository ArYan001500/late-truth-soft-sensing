"""Final comparison table (PUBLISHED_BASELINES_v001): every method scored on its own rows AND on the common rows where all methods
have a finite prediction (pre-declared in Amendment 4). Kelmarsh 2017 and 2018. Pure numpy; reads saved npz only."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
O = C.OUT
LATER = ['kelmarsh2024', 'penmanshiel2022', 'hot2025']   # LATER_YEARS_v001, appended after the original datasets
CONFIRM = ['kelmarsh2020', 'penmanshiel2020', 'hot2022']   # CONFIRM_TWO_TIMESCALE_v001 (+ Amendment 19 baselines)
PROPOSED = 'proposed (similarity + recency)'   # Amendment 18 ours+rec; the paper's proposed estimator after the 2026-10-17 revision
EXTRA = {PROPOSED: 'rec', 'proposed + standstill gate': 'recg', 'reference: mean of 20 newest residuals': 'ref_mean20',
         'reference: EWMA of residuals': 'ref_ewma', 'per-turbine model, same inputs': 'ptfleet',
         'per-turbine model + late-truth correction': 'ptrec'}   # ptfleet/ptrec: Amendment 21   # loaded only with extras=True, so that older scripts see the old method set


def load_methods(ds, T, extras=False):
    z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); M = {'ours (fleet + local late-truth correction)': {t: z[f'P_ours_{t}'] for t in T}, 'our fleet model (no late truth)': {t: z[f'P_fleet_{t}'] for t in T}}
    def npz(name, f, key='P_{t}'):
        if (O/f).exists(): zz = np.load(O/f, allow_pickle=False); M[name] = {t: zz[key.format(t=t)] for t in T}
    npz('B10f Wang 2022 (frozen NN)', f'B10f_wang2022_frozen_{ds}.npz'); npz('B10 Wang 2022 (online, as published)', f'B10_wang2022_{ds}.npz')
    for b, nm in [('B0', 'B0 last released value'), ('B1', 'B1 Bangalore 2015'), ('B2', 'B2 Schlechtingen 2011 AR'), ('B2f', 'B2f Schlechtingen 2011 FSRC'),
                  ('B3', 'B3 Liu 2020'), ('B5', 'B5 Yang 2021'), ('B6', 'B6 Kim 2013')]:
        for lag in (3, 1): npz(f'{nm} [lag {lag}]', f'{b}_lag{lag}_{ds}.npz')
    npz('B11 Bai 2024', f'B11_bai_xgb_{ds}.npz')
    for b, nm in [('B4', 'B4 Fu 2019'), ('B7', 'B7 Yan 2023'), ('B8', 'B8 Dai 2025'), ('B9', 'B9 Grataloup 2025')]:
        for v, pre in [('v1', 'deep_'), ('v2', 'deepv2_')]:
            if all((O/f'{pre}{b}_{ds}_T{t}.npz').exists() for t in T): M[f'{nm} [{v}]'] = {t: np.load(O/f'{pre}{b}_{ds}_T{t}.npz', allow_pickle=False)['P'] for t in T}
    if extras:
        for name, pre in EXTRA.items():
            if (O/f'{pre}_{ds}.npz').exists(): zz = np.load(O/f'{pre}_{ds}.npz', allow_pickle=False); M[name] = {t: zz[f'P_{t}'] for t in T}
    return M


def main():
    out = {}
    for ds in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019'] + [d for d in LATER + CONFIRM if (O/f'ours_{d}.npz').exists()]:
        month, X, Y, OK = C.load(ds); T = sorted(Y); n = len(month); ev = np.arange(C.EV0, n); M = load_methods(ds, T, extras=True)
        lag1 = [k for k in M if '[lag 1]' in k]   # synchronous-truth variants are not eligible for the common receipt-causal row set
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in M if k not in lag1 and 'online' not in k and k not in EXTRA], 0) for t in T}   # extras do not shape the row set
        rows = []
        for k, P in M.items():
            own = C.score(Y, P, OK, month)['all']
            com = float(np.mean([np.abs(Y[t][ev] - P[t][ev])[common[t]].mean() for t in T]))
            cov = float(np.mean([(OK[t][ev] & np.isfinite(P[t][ev])).sum() / OK[t][ev].sum() for t in T]))
            rows.append((k, own, com, cov))
        rows.sort(key=lambda r: r[2]); out[ds] = [dict(method=k, mae_own_rows=o, mae_common_rows=c, coverage=v) for k, o, c, v in rows]
        print(f'\n{ds}: common rows share = {np.mean([common[t].sum() / OK[t][ev].sum() for t in T]):.3f}')
        for k, o, c, v in rows: print(f'  {k:42s} common {c:.4f}  own {o:.4f}  coverage {v:.3f}')
    (O/'FINAL_TABLE_all.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
