"""Protocol protocols/KELMARSH_2018_UNSEEN_v001.md — frozen Kelmarsh pipeline on never-seen 2018."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
from latetruth.data import require_archive
import sys
import io, csv, zipfile, json, hashlib, sys
import numpy as np, pandas as pd
from latetruth import run_extra_heat_paths as X
from latetruth import run_h29_h30_physics as B
from latetruth import run_send_schedule as SS
from latetruth import run_penmanshiel_replication as P
from latetruth.run_local_late_truth_long import ROOT, TARGET, HELP
ZP = DATA_DIR/'Kelmarsh/Kelmarsh_SCADA_2018_3084.zip'
OUT = OUT_DIR/'kelmarsh_2018_unseen_2026-10-13_v001'
PRES = ['Gear oil pump pressure (bar)', 'Gear oil inlet pressure (bar)']


def read(t):
    with zipfile.ZipFile(require_archive(ZP)) as z:
        name = next(n for n in z.namelist() if n.startswith(f'Turbine_Data_Kelmarsh_{t}_'))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            df = pd.read_csv(f, names=h, header=None, usecols=[h[0], TARGET, B.VANE] + HELP + X.EXTRA + PRES)
    return df, pd.to_datetime(df[h[0]], utc=True).dt.month.to_numpy()


def main(part):
    out = OUT/f'results_{part}.json'
    if out.exists(): raise FileExistsError(out)
    OUT.mkdir(exist_ok=True); D = {t: read(t) for t in range(1, 7)}; n = len(D[1][0]); month = D[1][1]
    assert all(len(D[t][0]) == n for t in D)
    Y = {t: df[TARGET].to_numpy(float) for t, (df, _) in D.items()}
    OK = {t: np.r_[np.zeros(7, bool), np.isfinite(Y[t][7:]) & np.isfinite(Y[t][4:-3]) & np.isfinite(Y[t][3:-4])] for t in D}
    pad, run, rows = P.pad, P.run, np.arange(7, n); res = {}
    def stale(cols, extra=None):
        def fn(t):
            d = D[t][0].copy()
            for c in cols: d[c] = d[c].shift(3)
            f = X.feats(d, rows, 'h30_x')
            if extra:
                A = d[extra].to_numpy(float); f = np.column_stack([f, A[rows]] + [A[rows] - A[rows - k] for k in (1, 2, 3)])
            return pad(f)
        return fn
    if part == 'A':
        for kind in ['base', 'h29', 'h30', 'h30_x']:
            fn = (lambda t, k=kind: pad(X.feats(D[t][0], rows, k)) if k.endswith('_x') else pad(B.feats(D[t][0], rows, k)))
            res.update(run(D, fn, kind, True, n, month, OK, Y))
    elif part == 'B':
        for s in (HELP + X.EXTRA)[:7]: res.update(run(D, stale([s]), 'stale:' + s, False, n, month, OK, Y))
    elif part == 'C':
        for s in (HELP + X.EXTRA)[7:]: res.update(run(D, stale([s]), 'stale:' + s, False, n, month, OK, Y))
    else:
        both = [SS.INL] + SS.BEAR
        res.update(run(D, stale(both), 'stale:inlet+bearings', False, n, month, OK, Y))
        res.update(run(D, stale(both, PRES), 'stale:inlet+bearings+pressure', False, n, month, OK, Y))
        for pol in ['always_bearing', 'always_inlet']:
            def fn(t, pol=pol):
                d, ages = SS.apply_policy(D[t][0], pol); return pad(np.column_stack([X.feats(d, rows, 'h30_x'), ages[7:]]))
            res.update(run(D, fn, 'policy:' + pol, False, n, month, OK, Y))
    out.write_text(json.dumps({'results': res, 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}, indent=2))


if __name__ == '__main__': main(sys.argv[1])
