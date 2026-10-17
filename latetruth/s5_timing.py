"""Amendment 11 S5 (implementation note in PUBLISHED_BASELINES_v001.md): wall-clock fit + predict time, Kelmarsh 2018, June window,
turbine 1, one method at a time. Writes S5_TIMING.json. Run with the Python (needs pandas stack for ours) for part 'cpu',
and with Python for part 'gpu'.  usage: python s5_timing.py cpu|gpu"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import os, sys, json, time, tempfile
from pathlib import Path
here = Path(__file__).resolve().parent
import numpy as np
DS, T = 'kelmarsh2018', 1
J = OUT_DIR/'published_baselines_2026-10-14_v001/S5_TIMING.json'


def put(k, v, what):
    d = json.loads(J.read_text()) if J.exists() else {}; d[k] = dict(seconds=v, what=what); J.write_text(json.dumps(d, indent=2)); print(k, round(v, 2), 's', flush=True)


def timed(f, rep):
    ts = []
    for _ in range(rep):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
        if ts[-1] > 60: break
    return float(np.median(ts))


def june(month): return int(np.flatnonzero(month == 6)[0]), int(np.flatnonzero(month == 7)[0])


def cpu():
    from latetruth import baselines_classic as C
    month, X, Y, OK = C.load(DS); s, e = june(month); load0 = C.load
    C.windows = lambda m, n: [(s, e)]
    C.load = lambda ds: (lambda r: (r[0], {T: r[1][T]}, {T: r[2][T]}, {T: r[3][T]}))(load0(ds))
    for b in ['B0', 'B1', 'B2', 'B2f', 'B3', 'B5']:
        hp = json.loads((C.OUT/f'{b}_lag3.json').read_text())['hp']
        put(b, timed(lambda: C.run(b, hp, 3, DS), 2), 'refit on rows released before 1 June + predict June, turbine 1')
    from latetruth import baselines_kim as K
    hp = json.loads((C.OUT/'B6_lag3.json').read_text())['hp']; put('B6', timed(lambda: K.run(hp, 3, DS), 2), 'offline + online LW-PLS for June, turbine 1')
    from latetruth import baselines_bai as BA
    hp = json.loads((C.OUT/'B11_bai_xgb.json').read_text())['hp']; put('B11', timed(lambda: BA.run(hp, DS), 2), 'features + XGB refit + predict June, turbine 1')


def ours():
    from latetruth import baselines_common as BC
    D, n, month, Y, OK = BC.load(DS); s, e = june(month)
    BC.windows = lambda m, n_: [(s, e)]; loc = BC.local; BC.local = lambda Z, r, ok, mem, **k: np.zeros(len(r)); BC.load = lambda ds: (D, n, month, Y, OK)
    put('ours_fleet', timed(lambda: BC.ours(DS), 2), 'features for 6 turbines + pooled HGB refit on rows before 1 June + predict June (6 turbines)')
    z = np.load(BC.OUT/f'ours_{DS}.npz', allow_pickle=False); P = z[f'P_fleet_{T}']; df = D[T][0]; ev = np.arange(BC.EV0, n)
    ok = OK[T] & np.isfinite(P); r = np.where(ok, Y[T] - P, 0.0)
    H = np.nan_to_num(df[BC.HELP].to_numpy(float)); Hf = df[BC.HELP].to_numpy(float)[4:BC.EV0]; Z = (H - np.nanmean(Hf, 0)) / (np.nanstd(Hf, 0) + 1e-9)
    a = s - 8640 - 3
    t1 = timed(lambda: loc(Z[a:e], r[a:e], ok[a:e], 8640), 2); t0 = timed(lambda: loc(Z[a:s], r[a:s], ok[a:s], 8640), 2)
    put('ours_local', t1 - t0, 'local late-truth correction (k 20, memory 8640) for the June rows of turbine 1')
    from latetruth import baseline_wang2022 as W; from latetruth import baseline_wang2022_frozen as WF
    prm = tuple(json.loads((BC.OUT/'B10f_wang2022_frozen.json').read_text())['params(q,rf,ri)']); month2, Yw, OKw, PF = W.load_npz(DS)
    put('B10f', timed(lambda: WF._turbine((PF[T], Yw[T], prm)), 2), 'frozen-NN EKF fusion over the full year, turbine 1 (no refit)')


def gpu():
    os.environ['DEEP_BUDGET'] = 'v2'
    from latetruth import baselines_deep as BD
    month = BD.load(DS, T)[0]; s, e = june(month); BD.windows = lambda m, n: [(s, e)]
    tmp = Path(tempfile.mkdtemp(prefix='s5_', dir=OUT_DIR)); BD.BUDGET = 's5timing'   # E0/PAT/E1 stay v2; only the output name changes (deleted below)
    done = json.loads(J.read_text()) if J.exists() else {}
    for m in [x for x in ['B4', 'B7', 'B8', 'B9'] if x not in done]:
        f0 = BD.OUT/f'deeps5timing_{m}_{DS}_T{T}.npz'; f0.unlink(missing_ok=True)
        put(m, timed(lambda: BD.main(m, T, DS), 1), 'v2: full training on rows before 1 June (100 epochs, patience 10) + predict June, turbine 1, GPU')
        (BD.OUT/f'deeps5timing_{m}_{DS}_T{T}.npz').unlink()
    os.environ['CAO_BUDGET'] = 'v2'; os.environ['CAO_OUT'] = str(tmp); os.environ['CAO_EPOCHS'] = '5'
    from latetruth import baselines_cao as CA
    put('B12_v2', 20 * timed(lambda: CA.main(T, DS), 1), 'v2: 20 x (5 epochs on rows before 1 June + predict June), i.e. extrapolated to the 100-epoch budget, turbine 1, GPU')


if __name__ == '__main__': {'cpu': cpu, 'ours': ours, 'gpu': gpu}[sys.argv[1]]()
