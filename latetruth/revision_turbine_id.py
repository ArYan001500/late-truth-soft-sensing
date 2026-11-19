from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR, RAW_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Parts A and B: encoding of the turbine index and a newly added turbine.

usage:  python revision_turbine_id.py A <ds> [variant ...]      (variants: orig perm1..perm5 cat noid; default all)
        python revision_turbine_id.py B <ds>                    (new-turbine runs; ds in kelmarsh2018 penmanshiel2019 hot2019)
For Hill of Towie set HOT_YEAR before the call (ds hot2019 / hot2025 / hot2022 must match HOT_YEAR).
"""
import os, sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth import residual_references as R
from latetruth.run_local_late_truth_long import PARAMS, local, HistGradientBoostingRegressor, threadpool_limits

OUT = RAW_DIR/'revision_checks_2026-11-17_v001'; OUT.mkdir(parents=True, exist_ok=True)
EV0 = BC.EV0; D = 3; TH = int(os.environ.get('THREADS', '4'))


def data(ds):
    """Turbine frames, target, OK mask, features, driver columns, month, n: exactly as in the submitted pipeline."""
    if ds.startswith('hot'):
        from latetruth import run_hill_of_towie_replication as H
        assert ds == f'hot{H.YEAR}', 'set HOT_YEAR to match ' + ds
        Dd = H.read_all(); n = H.n; month = H.month; Y = {t: df[H.TARGET].to_numpy(float) for t, df in Dd.items()}
        OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in Dd}
        F = {t: H.feats(Dd[t], 'base_x') for t in Dd}; help_cols = H.HELP
    else:
        from latetruth import run_extra_heat_paths as X
        from latetruth.run_local_late_truth_long import HELP as help_cols
        DD, n, month, Y, OK = BC.load(ds); Dd = {t: df for t, (df, _) in DD.items()}; rows = np.arange(7, n); F = {}
        for t in Dd:
            f = X.feats(Dd[t], rows, 'h30_x'); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    return Dd, Y, OK, F, help_cols, month, n


def run(ds, Dd, Y, OK, F, help_cols, month, n, idmap=None, mode='num', new=None):
    """Fleet model + similarity term + recency term. idmap: turbine -> index value; mode: num | cat | none;
    new: turbine treated as newly added at tau_1 (rows before EV0 removed from training, standardization from the others)."""
    T = list(Dd); idmap = idmap or {t: t for t in T}; P = {t: np.full(n, np.nan) for t in T}
    col = lambda t, k: [np.full(k, idmap[t])] if mode != 'none' else []
    for s, e in BC.windows(month, n):
        XX, yy = [], []
        for t in T:
            r = np.arange(7, s - D); r = r[OK[t][r]]
            if t == new: r = r[r >= EV0]
            if not len(r): continue
            XX.append(np.column_stack([F[t][r]] + col(t, len(r)))); yy.append(Y[t][r])
        Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
        if mode != 'none': keep[-1] = True
        kw = dict(PARAMS)
        if mode == 'cat': kw['categorical_features'] = [int(keep.sum()) - 1]
        with threadpool_limits(limits=TH): m = HistGradientBoostingRegressor(**kw).fit(Xa[:, keep], np.concatenate(yy))
        for t in T:
            r = np.arange(s, e); r = r[OK[t][r]]
            if len(r): P[t][r] = m.predict(np.column_stack([F[t][r]] + col(t, len(r)))[:, keep])
    ev = np.arange(EV0, n); PL, PR = {}, {}
    others = np.vstack([Dd[t][help_cols].to_numpy(float)[4:EV0] for t in T if t != new]) if new is not None else None
    for t in T:
        ok = OK[t][ev] & np.isfinite(P[t][ev]); r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
        Hh = np.nan_to_num(Dd[t][help_cols].to_numpy(float)[ev]); Hf = others if t == new else Dd[t][help_cols].to_numpy(float)[4:EV0]
        Z = (Hh - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL[t] = np.full(n, np.nan); PL[t][ev] = P[t][ev] + local(Z, r, ok, 8640)
        ok2 = OK[t][ev] & np.isfinite(PL[t][ev]); r2 = np.where(ok2, Y[t][ev] - PL[t][ev], 0.0)
        PR[t] = np.full(n, np.nan); PR[t][ev] = PL[t][ev] + 0.5 * R.ewma(r2, ok2, 18)
    return P, PR


def mae_t(Y, P, OK, t, n, upto=None):
    ev = np.arange(EV0, n if upto is None else upto); ok = OK[t][ev] & np.isfinite(P[t][ev])
    return float(np.abs(Y[t][ev] - P[t][ev])[ok].mean())


def summary(Y, OK, n, P, PR):
    return dict(fleet=float(np.mean([mae_t(Y, P, OK, t, n) for t in Y])), proposed=float(np.mean([mae_t(Y, PR, OK, t, n) for t in Y])))


def reproduce(ds, Y, OK, P, PR):
    z = np.load(BC.OUT/f'ours_{ds}.npz', allow_pickle=False); zr = np.load(BC.OUT/f'rec_{ds}.npz', allow_pickle=False); d1 = d2 = 0.0
    for t in Y:
        a, b = z[f'P_fleet_{t}'], P[t]; d1 = max(d1, float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0)
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), f'fleet row mask differs, turbine {t}'
        a, b = zr[f'P_{t}'], PR[t]; d2 = max(d2, float(np.nanmax(np.abs(a - b))))
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), f'proposed row mask differs, turbine {t}'
    return d1, d2


def part_a(ds, variants):
    Dd, Y, OK, F, hc, month, n = data(ds); T = list(Dd); out = {}
    f = OUT/f'turbine_id_{ds}.json'
    if f.exists(): out = json.loads(f.read_text())['variants']
    for v in variants:
        t0 = time.time()
        if v == 'orig': P, PR = run(ds, Dd, Y, OK, F, hc, month, n)
        elif v.startswith('perm'):
            lab = np.random.default_rng(int(v[4:])).permutation(T); P, PR = run(ds, Dd, Y, OK, F, hc, month, n, idmap=dict(zip(T, lab.tolist())))
        elif v == 'cat': P, PR = run(ds, Dd, Y, OK, F, hc, month, n, mode='cat')
        elif v == 'noid': P, PR = run(ds, Dd, Y, OK, F, hc, month, n, mode='none')
        else: raise ValueError(v)
        res = summary(Y, OK, n, P, PR)
        if v == 'orig':
            d1, d2 = reproduce(ds, Y, OK, P, PR); res['repro_max_abs_fleet'], res['repro_max_abs_proposed'] = d1, d2
            if max(d1, d2) > 1e-9: print(ds, 'REPRODUCTION FAILED', d1, d2, flush=True)
        if v.startswith('perm'): res['labels'] = dict(zip(map(str, T), lab.tolist()))
        res['seconds'] = round(time.time() - t0, 1); out[v] = res
        print(ds, v, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in res.items() if k != 'labels'}, flush=True)
        f.write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part A', ds=ds, variants=out,
                                     code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


def part_b(ds):
    Dd, Y, OK, F, hc, month, n = data(ds); T = list(Dd); e1 = BC.windows(month, n)[0][1]
    P0, PR0 = run(ds, Dd, Y, OK, F, hc, month, n); out = {}
    f = OUT/f'new_turbine_{ds}.json'
    for j in T:
        t0 = time.time(); P, PR = run(ds, Dd, Y, OK, F, hc, month, n, new=j)
        out[str(j)] = dict(fleet_new=mae_t(Y, P, OK, j, n), fleet_orig=mae_t(Y, P0, OK, j, n),
                           proposed_new=mae_t(Y, PR, OK, j, n), proposed_orig=mae_t(Y, PR0, OK, j, n),
                           proposed_new_w1=mae_t(Y, PR, OK, j, n, e1), proposed_orig_w1=mae_t(Y, PR0, OK, j, n, e1),
                           fleet_new_w1=mae_t(Y, P, OK, j, n, e1), fleet_orig_w1=mae_t(Y, P0, OK, j, n, e1),
                           seconds=round(time.time() - t0, 1))
        print(ds, 'new turbine', j, {k: round(v, 4) for k, v in out[str(j)].items()}, flush=True)
        f.write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part B', ds=ds, turbines=out,
                                     code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__':
    part, ds = sys.argv[1], sys.argv[2]
    if part == 'A': part_a(ds, sys.argv[3:] or ['orig', 'perm1', 'perm2', 'perm3', 'perm4', 'perm5', 'cat', 'noid'])
    elif part == 'B': part_b(ds)
    else: raise ValueError(part)
