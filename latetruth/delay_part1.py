"""Amendment 11 S3, part 1 (Python): ours and Wang-2022-frozen on Kelmarsh 2018 for release delays d in {1, 3, 6, 12} steps.
Writes delay_ours_d{d}.npz (Y, OK, P_fleet, P_ours) and delay_wang_d{d}.npz, and appends scores to DELAY_SENSITIVITY.json."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json
import numpy as np
from latetruth import baselines_common as BC
from latetruth import baseline_wang2022 as W; from latetruth import baseline_wang2022_frozen as WF
DS = 'kelmarsh2018'; O = BC.OUT; J = O/'DELAY_SENSITIVITY_part1.json'


def put(method, d, sc):
    res = json.loads(J.read_text()) if J.exists() else {}; res.setdefault(method, {})[str(d)] = sc; J.write_text(json.dumps(res, indent=2))
    print(method, 'delay', d, round(sc['all'], 4), flush=True)


def main():
    prm = tuple(json.loads((O/'B10f_wang2022_frozen.json').read_text())['params(q,rf,ri)'])
    for d in [1, 3, 6, 12, 24]:
        f = O/f'delay_ours_d{d}.npz'
        if not f.exists():
            Y, OK, month, n, P, PL = BC.ours_sync(DS, delay=d)
            np.savez_compressed(f, month=month, **{f'{k}_{t}': v for t in Y for k, v in [('Y', Y[t]), ('OK', OK[t]), ('P_fleet', P[t]), ('P_ours', PL[t])]})
        z = np.load(f, allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_')); month = z['month']; n = len(month)
        Y = {t: z[f'Y_{t}'] for t in T}; OK = {t: z[f'OK_{t}'] for t in T}; PF = {t: z[f'P_fleet_{t}'] for t in T}
        put('ours', d, BC.score(Y, {t: z[f'P_ours_{t}'] for t in T}, OK, month, n)); put('our fleet model (no late truth)', d, BC.score(Y, PF, OK, month, n))
        W.DELAY = d; WF.DELAY = d
        Pw = {t: WF._turbine((PF[t], Y[t], prm)) for t in T}
        np.savez_compressed(O/f'delay_wang_d{d}.npz', **{f'P_{t}': p for t, p in Pw.items()}); put('Wang 2022 (frozen NN)', d, BC.score(Y, Pw, OK, month, n))


if __name__ == '__main__': main()
