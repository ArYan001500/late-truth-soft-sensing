"""Claim C2 of PUBLISHED_BASELINES_v001 (Amendment 8): run ours + all baselines on Penmanshiel 2019 and Hill of Towie 2019, strictly one job
at a time (laptop RAM limit), with every hyper-parameter frozen at its Kelmarsh-2017 choice. Receipt-causal regime only (lag 3).
Resumable: a step is skipped when its output file exists.  usage: python run_new_datasets.py
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import os, sys, json, subprocess, time
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent
from latetruth import baselines_classic as C
O = C.OUT; LOG = OUT_DIR/'newds.log'
PYTHON = sys.executable
DLPY = sys.executable
ENV_C = dict(os.environ)
ENV_D = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', DEEP_BUDGET='v2')


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f: f.write(time.strftime('%H:%M ') + msg + '\n')


def sh(args, env, tag):
    log('START ' + tag)
    with open(LOG, 'a', encoding='utf-8') as f: r = subprocess.run(args, cwd=REPO_DIR, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f'END rc={r.returncode} ' + tag); return r.returncode


def save(tag, ds, sc, P):
    np.savez_compressed(O/f'{tag}_{ds}.npz', **{f'P_{t}': p for t, p in P.items()})
    js = O/f'{tag}_newds.json'; d = json.loads(js.read_text()) if js.exists() else {}; d[ds] = sc; js.write_text(json.dumps(d, indent=2))
    log(f'RESULT {tag} {ds} {sc["all"]:.4f}')


def main():
    for ds in ['penmanshiel2019', 'hot2019']:
        if not (O/f'inputs_{ds}.npz').exists():
            if ds == 'hot2019': sh([PYTHON, '-m', 'latetruth.baselines_hot'], ENV_C, f'ours+inputs {ds}')
            else:
                sh([PYTHON, '-m', 'latetruth.baselines_common', ds], ENV_C, f'ours {ds}')
                sh([PYTHON, '-c', f"from latetruth import baselines_common as B; B.export_inputs('{ds}')"], ENV_C, f'inputs {ds}')
        run_cpu_baselines(ds)
        T = sorted(int(k.split('_')[1]) for k in np.load(O/f'inputs_{ds}.npz', allow_pickle=False).files if k.startswith('X_'))
        for m in ['B7', 'B4', 'B8', 'B9']:
            for t in T:
                if not (O/f'deepv2_{m}_{ds}_T{t}.npz').exists(): sh([DLPY, '-m', 'latetruth.baselines_deep', m, str(t), ds], ENV_D, f'deep {m} T{t} {ds}')
            if all((O/f'deepv2_{m}_{ds}_T{t}.npz').exists() for t in T):
                month, X, Y, OK = C.load(ds); P = {t: np.load(O/f'deepv2_{m}_{ds}_T{t}.npz', allow_pickle=False)['P'] for t in T}; log(f'RESULT deepv2 {m} {ds} {C.score(Y, P, OK, month)["all"]:.4f}')
    log('ALL DONE')


def run_cpu_baselines(ds):
    """Our saved results logged, then every CPU baseline with hyper-parameters frozen at the Kelmarsh 2017 choice (resumable)."""
    if True:
        z = np.load(O/f'ours_{ds}.npz', allow_pickle=False); log(f'RESULT ours {ds} ' + json.dumps({k: round(v['all'], 4) for k, v in json.loads((O/f"ours_{ds}.json").read_text()).items() if isinstance(v, dict)}))
        # classic baselines, hp frozen from Kelmarsh
        for b in ['B0', 'B1', 'B2', 'B2f', 'B3', 'B5']:
            tag = f'{b}_lag3'
            if (O/f'{tag}_{ds}.npz').exists(): continue
            hp = json.loads((O/f'{tag}.json').read_text())['hp']; log('START ' + tag + ' ' + ds)
            sc, P = C.run(b, hp, 3, ds); save(tag, ds, sc, P)
        if not (O/f'B6_lag3_{ds}.npz').exists():
            from latetruth import baselines_kim as K
            hp = json.loads((O/'B6_lag3.json').read_text())['hp']; log('START B6 ' + ds); sc, P = K.run(hp, 3, ds); save('B6_lag3', ds, sc, P)
        if not (O/f'B11_bai_xgb_{ds}.npz').exists():
            from latetruth import baselines_bai as BA
            hp = json.loads((O/'B11_bai_xgb.json').read_text())['hp']; log('START B11 ' + ds); sc, P = BA.run(hp, ds); save('B11_bai_xgb', ds, sc, P)
        if not (O/f'B10f_wang2022_frozen_{ds}.npz').exists():   # Wang code imports the pandas/ml_runtime stack -> run it with the Python
            code = "; ".join([
                "import json, numpy as np; from latetruth import baseline_wang2022 as W, baseline_wang2022_frozen as WF",
                f"ds = '{ds}'", "O = W.OUT", "prm = tuple(json.loads((O/'B10f_wang2022_frozen.json').read_text())['params(q,rf,ri)'])",
                "month, Y, OK, PF = W.load_npz(ds)", "P = {t: WF._turbine((PF[t], Y[t], prm)) for t in Y}",
                "np.savez_compressed(O/f'B10f_wang2022_frozen_{ds}.npz', **{f'P_{t}': p for t, p in P.items()})",
                "print('RESULT B10f', ds, round(W.score(Y, P, OK, month, len(month))['all'], 4), flush=True)"])
            sh([PYTHON, '-c', code], ENV_C, f'B10f {ds}')


if __name__ == '__main__': main()
