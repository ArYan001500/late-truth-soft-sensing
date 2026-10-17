"""Amendment 14 reporting: standstill diagnostics for the paper (from saved files only).
(1) per dataset: share of common evaluation rows inside the gate (rotor stopped >= 12 h) and the MAE of fleet / ours / gated / Kim
    on those rows and on all other rows; (2) the Kelmarsh 2024 turbine 4 case (25 Jan - 15 Feb, hourly means) for the figure;
(3) error by hours since restart after a gated stop (exploratory, for the Discussion). Writes STANDSTILL_DIAG.json. Python."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
import numpy as np
from latetruth import baselines_classic as C
from latetruth import baselines_table as BT
from latetruth import standstill_gate as G
O = C.OUT; KIM = 'B6 Kim 2013 [lag 3]'
H = json.loads((O/'GATE.json').read_text())['selection']['chosen']['h']


def main():
    out = {'h_rows': H}
    for ds in G.DATASETS + BT.CONFIRM:   # + confirmation years; rec / recg = proposed after the revision (Amendment 18) and its gated form
        month, Y, OK, T, yr, s, nac = G.prep(ds); M = BT.load_methods(ds, T); n = len(month); ev = np.arange(C.EV0, n)
        Gt = {t: np.load(O/f'gate_{ds}.npz', allow_pickle=False)[f'P_{t}'] for t in T}
        Rc = {t: np.load(O/f'rec_{ds}.npz', allow_pickle=False)[f'P_{t}'] for t in T}; Rg = {t: np.load(O/f'recg_{ds}.npz', allow_pickle=False)[f'P_{t}'] for t in T}
        keys = [k for k in M if '[lag 1]' not in k and 'online' not in k]
        common = {t: OK[t][ev] & np.all([np.isfinite(M[k][t][ev]) for k in keys], 0) for t in T}
        acc = {w: {m: [0.0, 0] for m in ['fleet', 'ours', 'gate', 'kim', 'rec', 'recg']} for w in ['stop', 'rest']}
        since = {(lo, hi): {m: [0.0, 0] for m in ['gate', 'kim']} for lo, hi in [(0, 6), (6, 18), (18, 36), (36, 72)]}
        for t in T:
            c = common[t]; st = (s[t][ev] >= H)
            P = {'fleet': M['our fleet model (no late truth)'][t][ev],
                 'ours': M[G.OURS][t][ev], 'gate': Gt[t][ev], 'kim': M[KIM][t][ev], 'rec': Rc[t][ev], 'recg': Rg[t][ev]}
            for w, mask in [('stop', c & st), ('rest', c & ~st)]:
                for m, p in P.items(): a = acc[w][m]; a[0] += float(np.abs(Y[t][ev] - p)[mask].sum()); a[1] += int(mask.sum())
            # rows since the end of a gated stop (stop of >= H rows that ended)
            run = s[t]; last = -10 ** 9; sc = np.full(n, 10 ** 9)
            for i in range(1, n):
                if run[i] == 0 and run[i - 1] >= H: last = i
                sc[i] = i - last if run[i] == 0 else 10 ** 9
            for (lo, hi), d in since.items():
                mask = c & (sc[ev] >= lo) & (sc[ev] < hi)
                for m in ['gate', 'kim']: d[m][0] += float(np.abs(Y[t][ev] - P[m])[mask].sum()); d[m][1] += int(mask.sum())
        tot = acc['stop']['ours'][1] + acc['rest']['ours'][1]
        out[ds] = dict(gated_share=acc['stop']['ours'][1] / tot, n_stop=acc['stop']['ours'][1],
                       stop={m: v[0] / max(v[1], 1) for m, v in acc['stop'].items()}, rest={m: v[0] / v[1] for m, v in acc['rest'].items()},
                       since_restart_rows={f'{lo}-{hi}': {m: (v[0] / v[1] if v[1] else None) for m, v in d.items()} | {'n': d['gate'][1]} for (lo, hi), d in since.items()})
        print(ds, f"share {out[ds]['gated_share']:.4f}", 'stop', {m: round(v, 3) for m, v in out[ds]['stop'].items()}, 'rest', {m: round(v, 3) for m, v in out[ds]['rest'].items()}, flush=True)
    # Kelmarsh 2024, turbine 4: hourly means from 25 January to 15 February
    ds, t = 'kelmarsh2024', 4; month, Y, OK, T, yr, s, nac = G.prep(ds); M = BT.load_methods(ds, T); Gt = np.load(O/f'gate_{ds}.npz', allow_pickle=False)[f'P_{t}']
    Rc, Rg = np.load(O/f'rec_{ds}.npz', allow_pickle=False)[f'P_{t}'], np.load(O/f'recg_{ds}.npz', allow_pickle=False)[f'P_{t}']
    a, b = 24 * 144, 46 * 144   # 25 Jan 00:00 .. 16 Feb 00:00 (row 0 = 1 Jan 00:00; 2024 timestamps)
    def hourly(v): v = np.where(np.isfinite(v[a:b]), v[a:b], np.nan).reshape(-1, 6); return [None if np.isnan(x) else round(float(x), 3) for x in np.nanmean(v, 1)]
    with np.errstate(all='ignore'):
        case = dict(start='2024-01-25T00:00', step_h=1, truth=hourly(Y[t]), fleet=hourly(M['our fleet model (no late truth)'][t]), ours=hourly(M[G.OURS][t]),
                    gate=hourly(Gt), kim=hourly(M[KIM][t]), rec=hourly(Rc), recg=hourly(Rg), gated=[bool(x) for x in (s[t][a:b] >= H).reshape(-1, 6).any(1)],
                    stopped=[bool(x) for x in (s[t][a:b] >= 1).reshape(-1, 6).all(1)])
        feb = month == 2; okf = OK[t] & feb
        case['feb_mae'] = {m: float(np.nanmean(np.abs(Y[t] - p)[okf & np.isfinite(M[KIM][t])])) for m, p in
                           [('fleet', M['our fleet model (no late truth)'][t]), ('ours', M[G.OURS][t]), ('gate', Gt), ('kim', M[KIM][t]), ('rec', Rc), ('recg', Rg)]}
        case['oil_min'] = float(np.nanmin(Y[t][feb])); case['stop_days'] = float(s[t][feb].max() / 144)   # longest standstill run reaching into February
    out['case_k24_t4'] = case; print('case', {k: round(v, 3) for k, v in case['feb_mae'].items()}, 'oil min', case['oil_min'], 'longest stop (days)', round(case['stop_days'], 2))
    (O/'STANDSTILL_DIAG.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__': main()
