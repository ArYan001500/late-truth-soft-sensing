from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR, RAW_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Part D: practical size of the recency gain (descriptive, saved estimates only).
usage: python revision_recency_size.py"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from latetruth import baselines_common as BC
O = BC.OUT; EV0 = BC.EV0
OUTDIR = RAW_DIR/'revision_checks_2026-11-17_v001'; OUTDIR.mkdir(parents=True, exist_ok=True)
DS = ['kelmarsh2018', 'penmanshiel2019', 'hot2019', 'kelmarsh2024', 'penmanshiel2022', 'hot2025', 'kelmarsh2020', 'penmanshiel2020', 'hot2022']
NAMES = ['fleet', 'similarity', 'proposed']


def one(ds):
    z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); zr = np.load(O/f'rec_{ds}.npz', allow_pickle=False); month = z['month']; n = len(month); ev = np.arange(EV0, n)
    T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_'))
    acc = {k: [] for k in ['all', 'calm', 'transient']}; tail = []; tm = []; res = []
    for t in T:
        Y = z[f'Y_{t}']; P = {'fleet': z[f'P_fleet_{t}'], 'similarity': z[f'P_ours_{t}'], 'proposed': zr[f'P_{t}']}
        dl = np.full(n, np.nan); dl[3:] = np.abs(Y[3:] - Y[:-3])
        ok = z[f'OK_{t}'][ev] & np.all([np.isfinite(P[k][ev]) for k in NAMES], 0)
        E = {k: np.abs(Y[ev] - P[k][ev]) for k in NAMES}
        d = dl[ev]; fin = ok & np.isfinite(d); med, p90 = np.median(d[fin]), np.percentile(d[fin], 90)
        sets = {'all': ok, 'calm': fin & (d < med), 'transient': fin & (d >= p90)}
        for s, m in sets.items(): acc[s].append([float(E[k][m].mean()) for k in NAMES])
        tail.append([[float((E[k][ok] > 0.5).mean()), float((E[k][ok] > 1.0).mean()), float(np.percentile(E[k][ok], 95)),
                      float(np.percentile(E[k][ok], 99))] for k in NAMES])
        mon = month[ev]
        for mo in range(2, 13):
            m = ok & (mon == mo)
            if m.sum() > 100: tm.append(float(E['similarity'][m].mean() - E['proposed'][m].mean()))
        y = Y[np.isfinite(Y)]; res.append([float(np.mean(np.isclose(y, np.round(y)))), float(np.median(np.abs(np.diff(y))))])
    out = {s: {k: float(np.mean([a[i] for a in acc[s]])) for i, k in enumerate(NAMES)} for s in acc}
    for s in acc: out[s]['recency_change_pct'] = 100 * (out[s]['proposed'] / out[s]['similarity'] - 1)
    tl = np.mean(np.array(tail), 0)
    out['tail'] = {k: dict(share_above_0p5=tl[i, 0], share_above_1=tl[i, 1], p95=tl[i, 2], p99=tl[i, 3]) for i, k in enumerate(NAMES)}
    tm = np.array(tm); out['turbine_months'] = dict(n=int(tm.size), share_gain=float((tm > 0).mean()), largest_gain=float(tm.max()),
                                                   largest_loss=float(-tm.min()), median_gain=float(np.median(tm)))
    r = np.mean(np.array(res), 0); out['resolution'] = dict(share_whole_degree=float(r[0]), median_abs_10min_change=float(r[1]))
    return out


def main():
    out = {}
    for ds in DS:
        out[ds] = one(ds)
        o = out[ds]; print(ds, 'rec change all/calm/trans %:', [round(o[s]['recency_change_pct'], 2) for s in ['all', 'calm', 'transient']],
                           '| >1C share sim/prop:', round(o['tail']['similarity']['share_above_1'], 4), round(o['tail']['proposed']['share_above_1'], 4),
                           '| p99 sim/prop:', round(o['tail']['similarity']['p99'], 3), round(o['tail']['proposed']['p99'], 3),
                           '| tm:', {k: round(v, 4) for k, v in o['turbine_months'].items()}, '| res:', {k: round(v, 3) for k, v in o['resolution'].items()}, flush=True)
    (OUTDIR/'recency_size.json').write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Part D', **out,
                                            code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__': main()
