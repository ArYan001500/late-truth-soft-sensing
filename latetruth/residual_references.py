"""Amendment 17 (a): residual references on all six held-out farm-years (plus the development year for tuning).
  mean20 : fleet + 20/40 * mean of the 20 newest released residuals (= last20 of Amendment 13, nothing tuned)
  ewma   : fleet + c * time-decayed mean of all released residuals (half-life h rows); (h, c) chosen on Kelmarsh 2017 only
Released rule: residual of row s usable at row i only if s < i - d (d = 3), residuals collected from EV0, >20 released before use.
Writes ref_mean20_<ds>.npz, ref_ewma_<ds>.npz (P_<t>, NaN before EV0) and RESIDUAL_REFERENCES.json (MAE on Table-7 common rows,
S1 statistics of ours vs each reference, Holm over the Table-8 baselines plus the two references). Run with Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth.significance import holm, OURS
O = C.OUT; EV0, D, K, K0, MEM = C.EV0, 3, 20, 20, 8640
HL = [6, 18, 36, 72, 144, 288, 576, 1152, 2304]; CS = [0.5, 0.75, 1.0]
EVAL = ['kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER


def mean20(r, ok):
    out = np.zeros(len(r)); idx = np.flatnonzero(ok)
    for i in range(len(r)):
        hi = i - D
        if hi <= 0: continue
        a, b = np.searchsorted(idx, max(hi - MEM, 0)), np.searchsorted(idx, hi); cand = idx[a:b]
        if len(cand) <= K: continue
        out[i] = r[cand[-K:]].mean() * (K / (K + K0))
    return out


def ewma(r, ok, h):
    idx = np.flatnonzero(ok); out = np.zeros(len(r))
    if len(idx) == 0: return out
    lam = 0.5 ** (1.0 / h); E = np.empty(len(idx)); S = W = 0.0; prev = idx[0]
    for j, s in enumerate(idx):
        dec = lam ** (s - prev); S = S * dec + r[s]; W = W * dec + 1.0; E[j] = S / W; prev = s
    pos = np.searchsorted(idx, np.arange(len(r)) - D)   # number of released residuals (s < i - d)
    m = pos > K; out[m] = E[pos[m] - 1]
    return out


def residuals(z, t):
    Y, OK, PF = z[f'Y_{t}'], z[f'OK_{t}'], z[f'P_fleet_{t}']; ev = np.arange(EV0, len(Y))
    ok = OK[ev] & np.isfinite(PF[ev]); return Y, PF, ev, ok, np.where(ok, Y[ev] - PF[ev], 0.0)


def tune():
    z = np.load(O/'ours_kelmarsh2017.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); grid = {}
    for h in HL:
        acc = {c: [] for c in CS}
        for t in T:
            Y, PF, ev, ok, r = residuals(z, t); e = ewma(r, ok, h)
            for c in CS: acc[c].append(float(np.abs(Y[ev] - PF[ev] - c * e)[ok].mean()))
        for c in CS: grid[f'h{h}_c{c}'] = float(np.mean(acc[c]))
    best = min(grid, key=grid.get); h, c = best[1:].split('_c'); return int(h), float(c), grid


def main():
    h, c, grid = tune(); print('EWMA tuned on Kelmarsh 2017: half-life', h, 'rows, c =', c, flush=True)
    out = {'ewma_half_life_rows': h, 'ewma_c': c, 'tuning_grid_kelmarsh2017': grid}; rng = np.random.default_rng(20261017)
    for ds in ['kelmarsh2017'] + EVAL:
        z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); R = {'mean20': {}, 'ewma': {}}
        for t in T:
            Y, PF, ev, ok, r = residuals(z, t)
            for nm, cor in [('mean20', mean20(r, ok)), ('ewma', c * ewma(r, ok, h))]:
                P = np.full(len(Y), np.nan); P[ev] = PF[ev] + cor; R[nm][t] = P
        for nm in R: np.savez_compressed(O/f'ref_{nm}_{ds}.npz', **{f'P_{t}': R[nm][t] for t in T})
        if ds == 'kelmarsh2017':
            ev = np.arange(EV0, len(z['month'])); f = lambda P: float(np.mean([np.abs(z[f'Y_{t}'][ev] - P[t][ev])[z[f'OK_{t}'][ev] & np.isfinite(P[t][ev])].mean() for t in T]))
            out[ds] = {'fleet': f({t: z[f'P_fleet_{t}'] for t in T}), 'mean20': f(R['mean20']), 'ewma': f(R['ewma']), 'ours': f({t: z[f'P_ours_{t}'] for t in T})}
            print(ds, out[ds], flush=True); continue
        month, X, Y, OK = C.load(ds); n = len(month); ev = np.arange(EV0, n); M = BT.load_methods(ds, T)
        keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        M['ref mean20'] = R['mean20']; M['ref ewma'] = R['ewma']; keys += ['ref mean20', 'ref ewma']
        mon = month[ev]
        tm = lambda k: np.array([[np.abs(Y[t][ev] - M[k][t][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T])
        turb = lambda k: np.array([np.abs(Y[t][ev] - M[k][t][ev])[common[t]].mean() for t in T])
        A, At = tm(OURS), turb(OURS); rows, ps = [], []
        for k in keys:
            if k == OURS: continue
            B, Bt = tm(k), turb(k); d = (B - A).ravel(); d = d[np.isfinite(d)]
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(method=k, mae=float(Bt.mean()), mae_diff=float(Bt.mean() - At.mean()), ci95=[float(lo), float(hi)],
                             wins_turbine_month=int((d > 0).sum()), n_pairs=int(d.size), turbines_ours_lower=int((Bt > At).sum()), n_turbines=len(T), p=p)); ps.append(p)
        for r_, a in zip(rows, holm(ps)): r_['p_holm'] = float(a); r_['ours_better'] = bool(a < 0.05 and r_['ci95'][0] > 0)
        fl = turb('our fleet model (no late truth)').mean()
        out[ds] = dict(ours=float(At.mean()), fleet=float(fl), refs=[r_ for r_ in rows if r_['method'].startswith('ref ')],
                       share_of_gain={k: float((fl - turb(k).mean()) / (fl - At.mean())) for k in ['ref mean20', 'ref ewma']})
        print(ds, 'ours', round(At.mean(), 4), 'fleet', round(fl, 4), [(r_['method'], round(r_['mae'], 4), round(r_['mae_diff'], 4), [round(x, 4) for x in r_['ci95']], f"{r_['wins_turbine_month']}/{r_['n_pairs']}", f"{r_['p_holm']:.1e}", r_['ours_better']) for r_ in out[ds]['refs']], flush=True)
    out['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    (O/'RESIDUAL_REFERENCES.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
