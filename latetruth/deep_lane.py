"""Claim C2 GPU lane: deep baselines (budget v2) one job at a time, for Penmanshiel 2019 then Hill of Towie 2019; waits until the CPU lane
has written the dataset's inputs. Run with any python.  usage: python deep_lane.py"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import os, subprocess, time
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent; O = OUT_DIR/'published_baselines_2026-10-14_v001'; LOG = OUT_DIR/'newds.log'
DLPY = sys.executable; env = dict(os.environ, DEEP_BUDGET='v2', OMP_NUM_THREADS='2')


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f: f.write(time.strftime('%H:%M ') + 'GPU ' + msg + '\n')


for ds in os.environ.get('LANE_DS', 'penmanshiel2019,hot2019').split(','):
    while not (O/f'inputs2_{ds}.npz').exists(): time.sleep(30)
    time.sleep(10)
    T = sorted(int(k.split('_')[1]) for k in np.load(O/f'inputs_{ds}.npz', allow_pickle=False).files if k.startswith('X_'))
    if os.environ.get('LANE_REVERSE'): T = T[::-1]
    k = int(os.environ.get('LANE_OFFSET', '0')); T = T[k % len(T):] + T[:k % len(T)]
    MS = ['B7', 'B4', 'B8', 'B9'] if not os.environ.get('LANE_REVERSE') else ['B9', 'B8', 'B4', 'B7']; MS = MS[k % 4:] + MS[:k % 4]
    for m in MS:
        for t in T:
            if (O/f'deepv2_{m}_{ds}_T{t}.npz').exists(): continue
            L = O/'locks'; L.mkdir(exist_ok=True)   # one lane per job: atomic create, skip if another lane holds it
            try: os.close(os.open(L/f'{m}_{ds}_T{t}.lock', os.O_CREAT | os.O_EXCL))
            except FileExistsError: continue
            with open(LOG, 'a', encoding='utf-8') as f: r = subprocess.run([DLPY, '-m', 'latetruth.baselines_deep', m, str(t), ds], cwd=REPO_DIR, env=env, stdout=subprocess.DEVNULL, stderr=f)
            if r.returncode: log(f'END rc={r.returncode} {m} T{t} {ds}')
        log(f'DONE {m} {ds}')
log('GPU LANE DONE')
