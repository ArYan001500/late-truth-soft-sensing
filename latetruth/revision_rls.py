from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR, RAW_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part C: recursive least squares (forgetting factor) on the released residuals of the
fleet model, as a parameter-update reference. Tuned on Kelmarsh 2017 only; scored on the common rows of baselines_table.py.
usage: python revision_rls.py"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import baselines_common as BC
O = C.OUT; EV0, D, K = C.EV0, 3, 20
OUTDIR = RAW_DIR/'revision_checks_2026-11-17_v001'; OUTDIR.mkdir(parents=True, exist_ok=True)
HL = [6, 18, 36, 72, 144, 288, 576, 1152, 2304]; CS = [0.5, 0.75, 1.0]
EVAL = ['kelmarsh2018', 'penmanshiel2019', 'hot2019', 'kelmarsh2020', 'penmanshiel2020', 'hot2022'] + BT.LATER


def rls(f, mu, r, ok, h):
    """Correction b + a (f - mu) with (b, a) from RLS on released pairs s < i - d; zero until more than K pairs are released."""
    idx = np.flatnonzero(ok); out = np.zeros(len(r))
    if len(idx) == 0: return out
    lam = 0.5 ** (1.0 / h); b = a = 0.0; p11, p12, p22 = 100.0, 0.0, 100.0; TB = np.empty(len(idx)); TA = np.empty(len(idx))
    for j, s in enumerate(idx):
        x2 = f[s] - mu; e = r[s] - (b + a * x2)
        q1 = p11 + p12 * x2; q2 = p12 + p22 * x2; den = lam + q1 + x2 * q2
        k1, k2 = q1 / den, q2 / den; b += k1 * e; a += k2 * e
        p11, p12, p22 = (p11 - k1 * q1) / lam, (p12 - k1 * q2) / lam, (p22 - k2 * q2) / lam
        TB[j], TA[j] = b, a
    pos = np.searchsorted(idx, np.arange(len(r)) - D); m = pos > K
    out[m] = TB[pos[m] - 1] + TA[pos[m] - 1] * (f[m] - mu)
    return out


def parts(z, t):
    Y, OK, PF = z[f'Y_{t}'], z[f'OK_{t}'], z[f'P_fleet_{t}']; n = len(Y); ev = np.arange(EV0, n)
    mu = float(np.nanmean(Y[4:EV0])); ok = OK[ev] & np.isfinite(PF[ev]); r = np.where(ok, Y[ev] - PF[ev], 0.0)
    return Y, PF, ev, ok, r, mu   # mu: Amendment C1


def turbines(z): return sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_'))


def tune():
    z = np.load(O/'ours_kelmarsh2017.npz', allow_pickle=False); T = turbines(z); grid = {}
    for h in HL:
        acc = {c: [] for c in CS}
        for t in T:
            Y, PF, ev, ok, r, mu = parts(z, t); e = rls(np.nan_to_num(PF[ev]), mu, r, ok, h)
            for c in CS: acc[c].append(float(np.abs(Y[ev] - PF[ev] - c * e)[ok].mean()))
        for c in CS: grid[f'h{h}_c{c}'] = float(np.mean(acc[c]))
        print('tune h', h, {c: round(np.mean(acc[c]), 5) for c in CS}, flush=True)
    best = min(grid, key=lambda k: grid[k] if np.isfinite(grid[k]) else np.inf); h, c = best[1:].split('_c')
    return int(h), float(c), grid


def main():
    h, c, grid = tune(); print('RLS tuned on Kelmarsh 2017: half-life', h, 'rows, c =', c, flush=True)
    out = {'half_life_rows': h, 'c': c, 'tuning_grid_kelmarsh2017': grid}
    for ds in EVAL:
        z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); T = turbines(z); P = {}
        for t in T:
            Y, PF, ev, ok, r, mu = parts(z, t); Pt = np.full(len(Y), np.nan); Pt[ev] = PF[ev] + c * rls(np.nan_to_num(PF[ev]), mu, r, ok, h); P[t] = Pt
        np.savez_compressed(OUTDIR/f'rls_{ds}.npz', **{f'P_{t}': P[t] for t in T})
        month, X, Y, OK = C.load(ds); n = len(month); ev = np.arange(EV0, n); M = BT.load_methods(ds, T, extras=True)
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in M if '[lag 1]' not in k and 'online' not in k and k not in BT.EXTRA], 0) for t in T}
        M['reference: recursive least squares'] = P
        turb = lambda k: np.array([np.abs(Y[t][ev] - M[k][t][ev])[common[t]].mean() for t in T])
        mon = month[ev]
        tm = lambda k: np.array([[np.abs(Y[t][ev] - M[k][t][ev])[common[t] & (mon == m)].mean() for m in range(2, 13)] for t in T])
        keys = ['our fleet model (no late truth)', BT.PROPOSED, 'reference: EWMA of residuals', 'B10f Wang 2022 (frozen NN)', 'reference: recursive least squares']
        res = {k: float(turb(k).mean()) for k in keys if k in M}
        d = (tm('reference: recursive least squares') - tm(BT.PROPOSED)).ravel(); d = d[np.isfinite(d)]
        res['proposed_wins_turbine_month'] = f'{int((d > 0).sum())}/{d.size}'
        out[ds] = res; print(ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        (OUTDIR/'rls_reference.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part C', **out,
                                                 code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__': main()
