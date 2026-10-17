"""Protocol 05_Protocols/LOCAL_LATE_TRUTH_LONG_v001.md (H26 + H27 over Feb-Dec 2017)."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, io, csv, zipfile, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = REPO_DIR
from sklearn.ensemble import HistGradientBoostingRegressor
from latetruth.config import threadpool_limits
OUT = OUT_DIR/'local_late_truth_long_2026-10-13_v001'
TARGET = 'Gear oil temperature (°C)'
HELP = ['Power (kW)', 'Wind speed (m/s)', 'Rotor speed (RPM)', 'Generator RPM (RPM)', 'Nacelle ambient temperature (°C)', 'Nacelle temperature (°C)',
        'Grid current (A)', 'Generator bearing rear temperature (°C)', 'Gear oil inlet temperature (°C)']
AMB = 4
PARAMS = dict(max_iter=400, learning_rate=.1, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=10, early_stopping=False, random_state=20261012)


def read(t):
    zp = next((DATA_DIR/'Kelmarsh').glob('*2017*.zip'))
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.startswith(f'Turbine_Data_Kelmarsh_{t}_'))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            df = pd.read_csv(f, names=h, header=None, usecols=[h[0], TARGET] + HELP)
    ts = pd.to_datetime(df[h[0]], utc=True); return df, ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60, ts.dt.month.to_numpy()


def feats(df, rows):
    V = df[HELP].to_numpy(float); oil = df[TARGET].to_numpy(float)
    return np.column_stack([V[rows]] + [V[rows] - V[rows - k] for k in (1, 2, 3)] + [oil[rows - 3], oil[rows - 3] - oil[rows - 4]])


def local(Z, r, ok, mem, k=20, k0=20, delay=3):
    out = np.zeros(len(r)); idx = np.flatnonzero(ok)
    for i in range(len(r)):
        hi = i - delay
        a, b_ = np.searchsorted(idx, max(hi - mem, 0)), np.searchsorted(idx, hi) if hi > 0 else 0
        cand = idx[a:b_] if hi > 0 else idx[:0]
        if len(cand) <= k: continue
        d = np.sum((Z[cand] - Z[i]) ** 2, 1); nn = cand[np.argpartition(d, k)[:k]]
        out[i] = r[nn].mean() * k / (k + k0)
    return out


def main():
    if (OUT/'results.json').exists(): raise FileExistsError(OUT/'results.json')
    OUT.mkdir(parents=True, exist_ok=True); D = {t: read(t) for t in range(1, 7)}
    tr = np.arange(4, 4032); X, Y = [], []
    for t, (df, _, _) in D.items():
        y = df[TARGET].to_numpy(float); r = tr[np.isfinite(y[tr]) & np.isfinite(y[tr - 3]) & np.isfinite(y[tr - 4])]
        X.append(np.column_stack([feats(df, r), np.full(len(r), t)])); Y.append(y[r])
    with threadpool_limits(limits=4): m = HistGradientBoostingRegressor(**PARAMS).fit(np.vstack(X), np.concatenate(Y))
    res = []; months = list(range(2, 13))
    for t, (df, hour, month) in D.items():
        y = df[TARGET].to_numpy(float); ev = np.arange(4032, len(df))
        ok = np.isfinite(y[ev]) & np.isfinite(y[ev - 3]) & np.isfinite(y[ev - 4])
        F = feats(df, ev); p = np.full(len(ev), np.nan); p[ok] = m.predict(np.column_stack([F[ok], np.full(ok.sum(), t)]))
        r = np.where(ok, y[ev] - p, 0.0)
        H = np.nan_to_num(df[HELP].to_numpy(float)[ev]); inc = np.nan_to_num(F[:, 9:18])
        Hf = df[HELP].to_numpy(float)[tr]; mu, sd = np.nanmean(Hf, 0), np.nanstd(Hf, 0) + 1e-9
        base = (H - mu) / sd; hz = np.column_stack([np.sin(2 * np.pi * hour[ev] / 24), np.cos(2 * np.pi * hour[ev] / 24)])
        h27 = np.column_stack([base, 3 * base[:, [AMB]], hz]); dr = np.column_stack([h27, inc / (inc[:2000].std(0) + 1e-9)])
        b = np.zeros(len(r)); s = 0.0
        for i in range(len(r)):
            if i - 3 >= 0 and ok[i - 3]: s += .05 * (r[i - 3] - s)
            b[i] = s
        corr = {'frozen': np.zeros(len(r)), 'bias': b}
        for mem in (2000, 8640):
            for name, Z in [('local_base', base), ('local_h27', h27), ('local_h27_dir', dr)]:
                corr[f'{name}_m{mem}'] = local(Z, r, ok, mem)
        for name, c in corr.items():
            e = np.abs(r - c)
            res.append(dict(turbine=t, method=name, all=float(e[ok].mean()), **{f'm{mo}': float(e[ok & (month[ev] == mo)].mean()) for mo in months}))
        print('turbine', t, {x['method']: round(x['all'], 4) for x in res if x['turbine'] == t}, flush=True)
    summ = {}
    for name in sorted({x['method'] for x in res}):
        rr = [x for x in res if x['method'] == name]
        summ[name] = {'all': float(np.mean([x['all'] for x in rr])), **{f'm{mo}': float(np.mean([x[f'm{mo}'] for x in rr])) for mo in months}}
    (OUT/'results.json').write_text(json.dumps({'summary': summ, 'results': res, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256((REPO_DIR/'protocols/LOCAL_LATE_TRUTH_LONG_v001.md').read_bytes()).hexdigest()}, indent=2))
    for name, v in summ.items(): print(f'{name:22s} all {v["all"]:.4f} | ' + ' '.join(f'{v[f"m{mo}"]:.3f}' for mo in months))


if __name__ == '__main__': main()

