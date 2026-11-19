from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part G, Amendment G-c: window versus recurrence, scored on the common rows of Part G
(rows where all 40 Part G estimators and the 6 G-c estimators exist). usage: python revision_controlled_window_summary.py"""
import json, hashlib, itertools
import numpy as np
from latetruth import baselines_common as BC
from latetruth import revision_turbine_id as RT

OUT = RT.OUT; EV0 = BC.EV0; ARCH = ['hgb', 'xgb', 'ridge', 'mlp', 'gru']; VAR = ['hgb_win', 'mlp_win', 'gru_w1']
CELLS = [f'{i}|{p}|{c}' for i, p, c in itertools.product(['full', 'base'], ['fleet', 'pt'], ['none', 'corr'])]
out = {}
for ds in ['kelmarsh2018', 'penmanshiel2019', 'hot2019']:
    z0 = np.load(BC.OUT/f'ours_{ds}.npz', allow_pickle=False); n = len(z0['month']); ev = np.arange(EV0, n); month = z0['month'][ev]
    T = sorted(int(k.split('_')[1]) for k in z0.files if k.startswith('Y_')); Y = {t: z0[f'Y_{t}'] for t in T}; OK = {t: z0[f'OK_{t}'] for t in T}
    Z = {a: np.load(OUT/f'controlled_{ds}_{a}.npz', allow_pickle=False) for a in ARCH}; W = {v: np.load(OUT/f'controlled_window_{ds}_{v}.npz', allow_pickle=False) for v in VAR}
    E = {f'{a}|{c}': {t: Z[a][f'{c}_{t}'] for t in T} for a in ARCH for c in ['full|fleet|none', 'full|fleet|corr']}
    for v in VAR:
        for c in ['none', 'corr']: E[f'{v}|{c}'] = {t: W[v][f'{c}_{t}'] for t in T}
    common = {t: OK[t][ev] & np.all([np.isfinite(Z[a][f'{c}_{t}'][ev]) for a in ARCH for c in CELLS] + [np.isfinite(W[v][f'{c}_{t}'][ev]) for v in VAR for c in ['none', 'corr']], 0) for t in T}
    mae = {k: float(np.mean([np.abs(Y[t][ev] - P[t][ev])[common[t]].mean() for t in T])) for k, P in E.items()}
    tm = lambda k: np.array([[np.abs(Y[t][ev] - E[k][t][ev])[common[t] & (month == m)].mean() for m in range(2, 13)] for t in T]).ravel()
    g = tm('gru|full|fleet|corr'); wins = {}
    for k in ['hgb|full|fleet|corr', 'hgb_win|corr', 'mlp|full|fleet|corr', 'mlp_win|corr', 'gru_w1|corr']:
        d = tm(k) - g; d = d[np.isfinite(d)]; wins[k] = f'{int((d > 0).sum())}/{d.size}'
    out[ds] = dict(mae=mae, gru_wins_against=wins, rows_common_share=float(np.mean([common[t].mean() for t in T])))
    print(ds, 'common', round(out[ds]['rows_common_share'], 3))
    for k in ['hgb|full|fleet|corr', 'hgb_win|corr', 'mlp|full|fleet|corr', 'mlp_win|corr', 'gru_w1|corr', 'gru|full|fleet|corr']:
        print(f'   {k:22s} corr {mae[k]:.4f}  none {mae[k.replace("corr", "none")]:.4f}  GRU better in {wins.get(k, "-")}')
(OUT/'controlled_window_summary.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part G, Amendment G-c summary', **out,
    code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))
