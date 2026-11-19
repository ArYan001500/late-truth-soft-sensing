from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part G, Amendment G-c: window versus recurrence.
usage: [HOT_YEAR=2019] python revision_controlled_window.py <ds> <hgb_win|mlp_win|gru_w1>"""
import sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth import revision_turbine_id as RT
from latetruth import revision_controlled as RC

OUT = RT.OUT; EV0 = BC.EV0


def window(F, W=6):
    """Concatenate the W most recent input vectors (rows t-W+1 .. t) of each turbine."""
    out = {}
    for t, f in F.items():
        n = len(f); parts = []
        for k in range(W):
            s = np.full(f.shape, np.nan); s[k:] = f[:n - k]; parts.append(s)
        out[t] = np.hstack(parts)
    return out


def main(ds, var):
    t0 = time.time(); Dd, Y, OK, F, hc, month, n = RT.data(ds); T = list(Dd)
    if var == 'gru_w1':
        from latetruth import revision_controlled_gru as G
        G.W = 1; P = G.fit_predict(F, Y, OK, T, month, n, True)
    else:
        P = RC.fleet(var.split('_')[0], window(F), Y, OK, T, month, n, True)
    PR = RC.correct(P, Y, OK, Dd, hc, n)
    ev = np.arange(EV0, n); m = lambda Q: float(np.mean([np.abs(Y[t][ev] - Q[t][ev])[OK[t][ev] & np.isfinite(Q[t][ev])].mean() for t in T]))
    np.savez_compressed(OUT/f'controlled_window_{ds}_{var}.npz', **{f'{c}_{t}': Q[t] for t in T for c, Q in [('none', P), ('corr', PR)]})
    info = dict(own_rows_mae_none=m(P), own_rows_mae_corr=m(PR), seconds=round(time.time() - t0, 1))
    (OUT/f'controlled_window_{ds}_{var}.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part G, Amendment G-c', ds=ds, variant=var,
        info=info, code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))
    print(ds, var, {k: round(v, 4) for k, v in info.items()}, flush=True)


if __name__ == '__main__': main(sys.argv[1], sys.argv[2])
