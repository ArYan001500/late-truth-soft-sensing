"""Amendment 11 S3, part 2 (Python): Kim 2013 and Bangalore 2015 on Kelmarsh 2018 for release delays d in {1, 3, 6, 12};
turbines in parallel (6 workers); hyper-parameters frozen at their Kelmarsh-2017 choice. Appends to DELAY_SENSITIVITY.json."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from latetruth import baselines_classic as C
DS = 'kelmarsh2018'; J = C.OUT/'DELAY_SENSITIVITY_part2.json'


def bang_one(a):
    hp, d, t = a; C.DELAY = d; month, X, Y, OK = C.load(DS); n = len(month); P = np.full(n, np.nan)
    for s, e in C.windows(month, n):
        m = C.B1(hp, d); m.fit(X[t], Y[t], C.released(OK[t], Y[t], s)); r = np.arange(s, e); P[r] = m.predict(X[t], Y[t], r)
    return t, P


def kim_one(a):
    from latetruth import baselines_kim as K
    hp, d, t = a; month, X, Y, OK = C.load(DS); n = len(month); P = np.full(n, np.nan); rng = np.random.default_rng(t); Z0 = K.design(X[t], Y[t], d)
    for s, e in C.windows(month, n):
        tr0 = np.arange(7, s - d); tr0 = tr0[OK[t][tr0] & np.isfinite(Y[t][tr0])]
        keep = (np.isfinite(Z0[tr0]).mean(0) >= 0.95) & (np.nanstd(Z0[tr0], 0) > 0); Z = Z0[:, keep]; rel = OK[t] & np.isfinite(Y[t]) & np.isfinite(Z).all(1)
        tr = tr0[rel[tr0]]; mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9; Zs = (Z - mu) / sd
        Xo, A, th0 = K.offline(Zs[tr[-K.MEM:]], Y[t][tr[-K.MEM:]], hp['R'], hp['phi'], hp['gamma'], rng)
        for q in range(s, e):
            if not (OK[t][q] and np.isfinite(Zs[q]).all()): continue
            db = np.arange(max(7, q - d - K.MEM), q - d); db = db[rel[db]]
            if len(db) < 50: continue
            th = K.online_theta(Xo, A, th0, Zs[q], hp['phi'], hp['gamma']); P[q] = K.lwpls(Zs[db], Y[t][db], K.sim(Zs[db], Zs[q], th, hp['phi']), Zs[q], hp['R'])
    return t, P


def put(method, d, sc):
    res = json.loads(J.read_text()) if J.exists() else {}; res.setdefault(method, {})[str(d)] = sc; J.write_text(json.dumps(res, indent=2))
    print(method, 'delay', d, round(sc['all'], 4), flush=True)


def main():
    month, X, Y, OK = C.load(DS); T = sorted(Y)
    hb = json.loads((C.OUT/'B1_lag3.json').read_text())['hp']; hk = json.loads((C.OUT/'B6_lag3.json').read_text())['hp']
    with Pool(min(6, JOBS)) as pool:
        for d in [1, 3, 6, 12, 24]:
            for name, fn, hp in [('Kim 2013', kim_one, hk), ('Bangalore 2015', bang_one, hb)]:
                f = C.OUT/f"delay_{name.split()[0].lower()}_d{d}.npz"
                if f.exists(): z = np.load(f, allow_pickle=False); P = {t: z[f'P_{t}'] for t in T}
                else:
                    P = dict(pool.map(fn, [(hp, d, t) for t in T])); np.savez_compressed(f, **{f'P_{t}': p for t, p in P.items()})
                put(name, d, C.score(Y, P, OK, month))


if __name__ == '__main__': main()
