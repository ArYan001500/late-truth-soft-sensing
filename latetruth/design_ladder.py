"""Design ladder on the development year (Kelmarsh 2017), descriptive (the development year was used to build the method).
All rungs share the receipt-causal protocol (d = 3, released rows only, the 11 retraining windows, same scoring):
 L0 fleet h30_x trained once on the initial window (no monthly retraining)
 L1 per-turbine models, base features      L2 fleet model, base features
 L3 fleet + heat-transfer/load terms (h30)  L4 fleet + bearing channels (h30_x)   L5 = L4 + local late-truth correction
usage: python design_ladder.py   (Python with ml_runtime on PYTHONPATH)"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, hashlib
import numpy as np
from latetruth import baselines_common as BC
from latetruth import run_extra_heat_paths as X; from latetruth import run_h29_h30_physics as B
from latetruth.run_local_late_truth_long import HELP, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
DS = 'kelmarsh2017'


def feats(D, n, kind):
    rows = np.arange(7, n); F = {}
    for t, (df, _) in D.items():
        f = X.feats(df, rows, kind) if kind.endswith('_x') else B.feats(df, rows, kind); F[t] = np.vstack([np.full((7, f.shape[1]), np.nan), f])
    return F


def fit_predict(F, Y, OK, wins, turbines_per_model):
    n = len(next(iter(Y.values()))); P = {t: np.full(n, np.nan) for t in Y}
    for group in turbines_per_model:
        for s, e in wins:
            XX, yy = [], []
            for t in group:
                r = np.arange(7, s - 3); r = r[OK[t][r]]; XX.append(np.column_stack([F[t][r], np.full(len(r), t)])); yy.append(Y[t][r])
            Xa = np.vstack(XX); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
            with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
            for t in group:
                r = np.arange(s, e); r = r[OK[t][r]]; P[t][r] = m.predict(np.column_stack([F[t][r], np.full(len(r), t)])[:, keep])
    return P


def main():
    D, n, month, Y, OK = BC.load(DS); T = list(D); wins = BC.windows(month, n); res = {}
    def rec(name, P): res[name] = BC.score(Y, P, OK, month, n); print(f"{name:52s} {res[name]['all']:.4f}", flush=True)
    Fx = feats(D, n, 'h30_x')
    rec('L0 fleet h30_x, trained once (no monthly retraining)', fit_predict(Fx, Y, OK, [(BC.EV0, n)], [T]))
    Fb = feats(D, n, 'base')
    rec('L1 per-turbine models, base features', fit_predict(Fb, Y, OK, wins, [[t] for t in T]))
    rec('L2 fleet model, base features', fit_predict(Fb, Y, OK, wins, [T]))
    rec('L3 fleet + heat-transfer and load terms (h30)', fit_predict(feats(D, n, 'h30'), Y, OK, wins, [T]))
    P4 = fit_predict(Fx, Y, OK, wins, [T]); rec('L4 fleet + bearing channels (h30_x)', P4)
    PL = {}
    for t, (df, _) in D.items():
        ev = np.arange(BC.EV0, n); ok = OK[t][ev] & np.isfinite(P4[t][ev]); r = np.where(ok, Y[t][ev] - P4[t][ev], 0.0)
        H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:BC.EV0]; Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
        PL[t] = np.full(n, np.nan); PL[t][ev] = P4[t][ev] + local(Z, r, ok, 8640)
    rec('L5 = L4 + local late-truth correction', PL)
    (BC.OUT/'DESIGN_LADDER_kelmarsh2017.json').write_text(json.dumps({'results': res, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
