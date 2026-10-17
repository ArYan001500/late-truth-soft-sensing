"""Protocol 05_Protocols/H29_H30_PHYSICS_TERMS_v001.md."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import io, csv, zipfile, json, hashlib
import numpy as np, pandas as pd
from latetruth.run_local_late_truth_long import ROOT, TARGET, HELP, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
OUT = OUT_DIR/'h29_h30_physics_2026-10-13_v001'; EV0 = 4032
VANE = 'Vane position 1+2 (°)'


def read(t):
    zp = next((DATA_DIR/'Kelmarsh').glob('*2017*.zip'))
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.startswith(f'Turbine_Data_Kelmarsh_{t}_'))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            df = pd.read_csv(f, names=h, header=None, usecols=[h[0], TARGET, VANE] + HELP)
    return df, pd.to_datetime(df[h[0]], utc=True).dt.month.to_numpy()


def feats(df, rows, kind):
    V = df[HELP].to_numpy(float); oil = df[TARGET].to_numpy(float); inl = V[:, 8]; P = V[:, 0]; rot = V[:, 2]
    base = [V[rows]] + [V[rows] - V[rows - k] for k in (1, 2, 3)] + [np.column_stack([oil[rows - 3], oil[rows - 3] - oil[rows - 4]])]
    if kind == 'base': return np.column_stack(base)
    dT = oil[rows - 3] - (inl[rows - 2] + inl[rows - 1] + inl[rows]) / 3; dinl = inl[rows] - inl[rows - 3]
    h29 = base + [np.column_stack([dT, dinl])]
    if kind == 'h29': return np.column_stack(h29)
    vane = np.abs(df[VANE].to_numpy(float)[rows])
    return np.column_stack(h29 + [np.column_stack([P[rows] * dT, rot[rows] ** 2, vane, P[rows] * dinl])])


def main():
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    OUT.mkdir(parents=True, exist_ok=True); D = {t: read(t) for t in range(1, 7)}; n = len(D[1][0]); month = D[1][1]
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(4, bool), np.isfinite(Y[t][4:]) & np.isfinite(Y[t][1:-3]) & np.isfinite(Y[t][:-4])] for t in D}
    starts = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]
    res = []
    for kind in ['base', 'h29', 'h30']:
        F = {t: feats(df, np.arange(4, n), kind) for t, (df, _) in D.items()}
        P = {t: np.full(n, np.nan) for t in D}
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else n; hi = s - 3; X, yy = [], []
            for t in D:
                rows = np.arange(4, hi); rows = rows[OK[t][rows]]; X.append(np.column_stack([F[t][rows - 4], np.full(len(rows), t)])); yy.append(Y[t][rows])
            Xa = np.vstack(X); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])  # drop all-NaN/constant columns in this window
            if (~keep).any(): print(kind, 'segment', j, 'dropped columns', np.flatnonzero(~keep).tolist(), flush=True)
            with threadpool_limits(limits=6): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
            for t in D:
                rows = np.arange(s, e); rows = rows[OK[t][rows]]; P[t][rows] = m.predict(np.column_stack([F[t][rows - 4], np.full(len(rows), t)])[:, keep])
        print(kind, 'trained', flush=True)
        for t, (df, mon) in D.items():
            ev = np.arange(EV0, n); ok = OK[t][ev]; r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
            move = np.abs(Y[t][ev] - Y[t][ev - 3]); mv = move[ok]; q50, q90 = np.quantile(mv, [.5, .9])
            H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:EV0]; Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
            c = local(Z, r, ok, 8640)
            np.savez_compressed(OUT/f'{kind}_turbine_{t}.npz', pred=P[t][ev], local_corr=c, truth=Y[t][ev], ok=ok, move=move)
            for lab, e in [(kind, np.abs(r)), (kind + '+local', np.abs(r - c))]:
                res.append(dict(turbine=t, method=lab, all=float(e[ok].mean()), calm=float(e[ok & (move < q50)].mean()), transient=float(e[ok & (move >= q90)].mean()),
                                **{f'm{mo}': float(e[ok & (mon[ev] == mo)].mean()) for mo in range(2, 13)}))
    summ = {}
    for lab in sorted({x['method'] for x in res}):
        rr = [x for x in res if x['method'] == lab]; summ[lab] = {k: float(np.mean([x[k] for x in rr])) for k in ['all', 'calm', 'transient'] + [f'm{m}' for m in range(2, 13)]}
        print(f"{lab:12s} all {summ[lab]['all']:.4f} calm {summ[lab]['calm']:.4f} transient {summ[lab]['transient']:.4f}", flush=True)
    (OUT/'results.json').write_text(json.dumps({'summary': summ, 'results': res, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
