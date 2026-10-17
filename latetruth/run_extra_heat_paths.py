"""Protocol 05_Protocols/EXTRA_HEAT_PATHS_v001.md."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import io, csv, zipfile, json, hashlib
import numpy as np, pandas as pd
from latetruth import run_h29_h30_physics as base_mod
from latetruth.run_local_late_truth_long import ROOT, TARGET, HELP, PARAMS, local, HistGradientBoostingRegressor, threadpool_limits
OUT = OUT_DIR/'extra_heat_paths_2026-10-14_v001'; EV0 = 4032
EXTRA = ['Front bearing temperature (°C)', 'Rear bearing temperature (°C)', 'Rotor bearing temp (°C)', 'Gearbox speed (RPM)']


def read(t):
    zp = next((DATA_DIR/'Kelmarsh').glob('*2017*.zip'))
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.startswith(f'Turbine_Data_Kelmarsh_{t}_'))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            df = pd.read_csv(f, names=h, header=None, usecols=[h[0], TARGET, base_mod.VANE] + HELP + EXTRA)
    return df, pd.to_datetime(df[h[0]], utc=True).dt.month.to_numpy()


def feats(df, rows, kind):
    core = base_mod.feats(df, rows, kind.replace('_x', ''))
    E = df[EXTRA].to_numpy(float)
    return np.column_stack([core, E[rows]] + [E[rows] - E[rows - k] for k in (1, 2, 3)])


def main():
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    OUT.mkdir(parents=True, exist_ok=True); D = {t: read(t) for t in range(1, 7)}; n = len(D[1][0]); month = D[1][1]
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(4, bool), np.isfinite(Y[t][4:]) & np.isfinite(Y[t][1:-3]) & np.isfinite(Y[t][:-4])] for t in D}
    starts = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]; res = []
    for kind in ['base_x', 'h30_x']:
        F = {t: feats(df, np.arange(4, n), kind) for t, (df, _) in D.items()}; P = {t: np.full(n, np.nan) for t in D}
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else n; X, yy = [], []
            for t in D:
                rows = np.arange(4, s - 3); rows = rows[OK[t][rows]]; X.append(np.column_stack([F[t][rows - 4], np.full(len(rows), t)])); yy.append(Y[t][rows])
            Xa = np.vstack(X); keep = np.array([len(np.unique(c[np.isfinite(c)])) >= 2 for c in Xa.T])
            if (~keep).any(): print(kind, 'segment', j, 'dropped', np.flatnonzero(~keep).tolist(), flush=True)
            with threadpool_limits(limits=8): m = HistGradientBoostingRegressor(**PARAMS).fit(Xa[:, keep], np.concatenate(yy))
            for t in D:
                rows = np.arange(s, e); rows = rows[OK[t][rows]]; P[t][rows] = m.predict(np.column_stack([F[t][rows - 4], np.full(len(rows), t)])[:, keep])
        print(kind, 'trained', flush=True)
        for t, (df, mon) in D.items():
            ev = np.arange(EV0, n); ok = OK[t][ev]; r = np.where(ok, Y[t][ev] - P[t][ev], 0.0)
            move = np.abs(Y[t][ev] - Y[t][ev - 3]); q50, q90 = np.quantile(move[ok], [.5, .9])
            H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); Hf = df[HELP].to_numpy(float)[4:EV0]; Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
            c = local(Z, r, ok, 8640)
            np.savez_compressed(OUT/f'{kind}_turbine_{t}.npz', pred=P[t][ev], local_corr=c, truth=Y[t][ev], ok=ok, move=move)
            for lab, e in [(kind, np.abs(r)), (kind + '+local', np.abs(r - c))]:
                res.append(dict(turbine=t, method=lab, all=float(e[ok].mean()), calm=float(e[ok & (move < q50)].mean()), transient=float(e[ok & (move >= q90)].mean()),
                                **{f'm{mo}': float(e[ok & (mon[ev] == mo)].mean()) for mo in range(2, 13)}))
    summ = {}
    for lab in sorted({x['method'] for x in res}):
        rr = [x for x in res if x['method'] == lab]; summ[lab] = {k: float(np.mean([x[k] for x in rr])) for k in ['all', 'calm', 'transient'] + [f'm{m}' for m in range(2, 13)]}
        print(f"{lab:14s} all {summ[lab]['all']:.4f} calm {summ[lab]['calm']:.4f} transient {summ[lab]['transient']:.4f}", flush=True)
    (OUT/'results.json').write_text(json.dumps({'summary': summ, 'results': res, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main()

