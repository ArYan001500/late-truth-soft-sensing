"""PUBLISHED_BASELINES_v001 Amendment 14: standstill gate. Form (hold / Newton) and threshold h are chosen on Kelmarsh 2017 only,
then frozen and applied to every other dataset. Saves gate_<ds>.npz (P_<t>) and GATE.json (selection + per-dataset table vs ours, fleet,
Kim, Wang on the common rows, and the S1 test of the gated estimate vs each baseline, own seed). Pure numpy/scipy; run with Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import significance as S
O = C.OUT; OURS = BT.OURS if hasattr(BT, 'OURS') else S.OURS; D = C.DELAY
ROTOR, NAC = 2, 5   # 16-slot layout: slot 2 rotor speed, slot 5 nacelle temperature (Senvion HELP order; Siemens map in baselines_hot.py)
HS = [3, 6, 12, 18, 36, 72, 144]
DATASETS = ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019'] + BT.LATER


def run_len(stop):
    """s(t): consecutive rows up to and including t with rotor speed < 1 RPM."""
    s = np.zeros(len(stop), int); c = 0
    for i, v in enumerate(stop): c = c + 1 if v else 0; s[i] = c
    return s


def prep(ds):
    z = np.load(O/f'inputs_{ds}.npz', allow_pickle=False); month, X, Y, OK = C.load(ds); T = sorted(Y)
    yr = {t: np.r_[np.full(D, np.nan), Y[t][:-D]] for t in T}   # last released value y(t-3)
    s = {t: run_len(np.nan_to_num(z[f'X_{t}'][:, ROTOR], nan=np.inf) < 1) for t in T}
    nac = {t: z[f'X_{t}'][:, NAC] for t in T}
    return month, Y, OK, T, yr, s, nac


def fit_tau(month, Y, T, yr, s, nac):
    """a = exp(-3/tau) by least squares of (y(t)-Tn) on (y(t-3)-Tn), January standstill rows (s >= 3)."""
    u, v = [], []
    for t in T:
        i = (month == 1) & (s[t] >= 3) & np.isfinite(Y[t]) & np.isfinite(yr[t]) & np.isfinite(nac[t])
        u.append(yr[t][i] - nac[t][i]); v.append(Y[t][i] - nac[t][i])
    u, v = np.concatenate(u), np.concatenate(v); a = float(u @ v / (u @ u))
    return a, len(u)


def gated(P, yr, s, nac, form, h, a):
    g = s >= h; alt = yr if form == 'hold' else nac + (yr - nac) * a
    out = P.copy(); m = g & np.isfinite(alt); out[m] = alt[m]; return out


def main():
    month, Y, OK, T, yr, s, nac = prep('kelmarsh2017'); M = BT.load_methods('kelmarsh2017', T)
    a, na = fit_tau(month, Y, T, yr, s, nac); tau = -D / np.log(a)
    base = C.score(Y, M[OURS], OK, month)['all']; grid = []
    for form in ['hold', 'newton']:
        for h in HS:
            sc = C.score(Y, {t: gated(M[OURS][t], yr[t], s[t], nac[t], form, h, a) for t in T}, OK, month)['all']; grid.append((sc, form, h))
    grid.sort(); best = grid[0]
    sel = dict(tau_rows=tau, a=a, n_jan_rows=na, ours_kelmarsh2017=base, grid=[dict(mae=g[0], form=g[1], h=g[2]) for g in grid],
               chosen=dict(form=best[1], h=best[2], mae=best[0]), gate_kept=bool(best[0] < base))
    print(f'tau {tau:.1f} rows (a {a:.4f}, n {na}); kelmarsh2017 ours {base:.4f}; best {best[1]} h={best[2]} {best[0]:.4f}')
    for g in grid: print(f'   {g[1]:6s} h={g[2]:4d}  {g[0]:.4f}')
    out = dict(selection=sel)
    if not sel['gate_kept']: print('GATE DROPPED (no improvement on Kelmarsh 2017)'); (O/'GATE.json').write_text(json.dumps(out, indent=2)); return
    rng = np.random.default_rng(20261016)
    for ds in DATASETS:
        if not (O/f'ours_{ds}.npz').exists(): continue
        month, Y, OK, T, yr, s, nac = prep(ds); M = BT.load_methods(ds, T); n = len(month); ev = np.arange(C.EV0, n)
        G = {t: gated(M[OURS][t], yr[t], s[t], nac[t], best[1], best[2], a) for t in T}
        np.savez_compressed(O/f'gate_{ds}.npz', **{f'P_{t}': G[t] for t in T})
        keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        assert all(np.array_equal(common[t], common[t] & np.isfinite(G[t][ev])) for t in T)   # gate never removes a common row
        mon = month[ev]; share = np.mean([(s[t][ev] >= best[2])[common[t]].mean() for t in T])
        tm = lambda P: np.array([[np.abs(Y[t][ev] - P[t][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T])
        tb = lambda P: np.array([np.abs(Y[t][ev] - P[t][ev])[common[t]].mean() for t in T])
        A, At = tm(G), tb(G); rows, ps = [], []
        for k in keys:
            B, Bt = tm(M[k]), tb(M[k]); d = (B - A).ravel(); d = d[np.isfinite(d)]
            p = float(S.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            boot = [np.mean(Bt[i]) - np.mean(At[i]) for i in (rng.integers(0, len(T), len(T)) for _ in range(2000))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(baseline=k, mae_common=float(Bt.mean()), mae_diff=float(Bt.mean() - At.mean()), ci95=[float(lo), float(hi)], wins=int((d > 0).sum()), n_pairs=int(d.size), p=p)); ps.append(p)
        for r, adj in zip(rows, S.holm(ps)): r['p_holm'] = float(adj); r['gate_better'] = bool(adj < 0.05 and r['ci95'][0] > 0)
        out[ds] = dict(gated_rows_share=float(share), mae_gate_common=float(At.mean()), vs=sorted(rows, key=lambda r: r['mae_diff']))
        print(f'\n{ds}: gate {At.mean():.4f}  (gated share {share:.4f})')
        for r in out[ds]['vs'][:6]: print(f"  vs {r['baseline']:42s} {r['mae_common']:.4f} diff {r['mae_diff']:+.4f} CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] wins {r['wins']}/{r['n_pairs']} {'BETTER' if r['gate_better'] else 'not shown'}")
    (O/'GATE.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__': main()
