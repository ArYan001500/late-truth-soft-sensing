from latetruth.config import DATA_DIR, OUT_DIR, REPO_DIR, PAPER_DIR
import sys
"""protocols/REVISION_CHECKS_v001.md, Amendment B-a: new-turbine diagnostics (categorical index, mid-range numeric index;
exact first-window share). usage: [HOT_YEAR=2019] python revision_newturbine_diag.py <ds> <turbine> [<turbine> ...]"""
import sys, json, hashlib, time
import numpy as np
from latetruth import baselines_common as BC
from latetruth import revision_turbine_id as RT

OUT = RT.OUT; EV0 = BC.EV0


def main(ds, js):
    Dd, Y, OK, F, hc, month, n = RT.data(ds); T = list(Dd); e1 = BC.windows(month, n)[0][1]
    nb = json.loads((OUT/f'new_turbine_{ds}.json').read_text())['turbines']; med = sorted(T)[len(T) // 2]; out = {}
    for j in js:
        t0 = time.time(); r = {}
        n1 = int(OK[j][EV0:e1].sum()); na = int(OK[j][EV0:].sum()); b = nb[str(j)]
        r['first_window_share_exact'] = (b['proposed_new_w1'] - b['proposed_orig_w1']) * n1 / ((b['proposed_new'] - b['proposed_orig']) * na)
        r['rows_first_window'], r['rows_all'] = n1, na
        _, PRc = RT.run(ds, Dd, Y, OK, F, hc, month, n, mode='cat', new=j); r['proposed_new_cat'] = RT.mae_t(Y, PRc, OK, j, n)
        r['proposed_new_cat_w1'] = RT.mae_t(Y, PRc, OK, j, n, e1)
        m = med if med != j else sorted(T)[len(T) // 2 - 1]; idmap = {t: t for t in T}; idmap[j], idmap[m] = m, j
        _, PRm = RT.run(ds, Dd, Y, OK, F, hc, month, n, idmap=idmap, new=j); r['proposed_new_mid'] = RT.mae_t(Y, PRm, OK, j, n)
        r['proposed_new_mid_w1'] = RT.mae_t(Y, PRm, OK, j, n, e1); r['swapped_with'] = int(m)
        r['proposed_orig'] = b['proposed_orig']; r['proposed_new_num'] = b['proposed_new']; r['seconds'] = round(time.time() - t0, 1)
        out[str(j)] = r; f = OUT/f'new_turbine_diag_{ds}_{j}.json'; print(ds, j, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        f.write_text(json.dumps(dict(protocol='REVISION_CHECKS_v001 Amendment B-a', ds=ds, turbines=out,
                                     code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()), indent=1))


if __name__ == '__main__': main(sys.argv[1], [int(x) for x in sys.argv[2:]])
