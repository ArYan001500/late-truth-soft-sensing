"""Run ONE CPU baseline of run_new_datasets.run_cpu_baselines for one dataset (same code, same frozen hp, same output files),
so that several baselines can run in parallel.  usage (Python): python cpu_baseline_one.py <ds> <B0|B1|B2|B2f|B3|B5|B6|B11|B10f>"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import sys, json
from pathlib import Path
here = Path(__file__).resolve().parent
from latetruth import run_new_datasets as N
from latetruth import baselines_classic as C
ds, b = sys.argv[1], sys.argv[2]; O = N.O; N.LOG = OUT_DIR/'later_years.log'
if b in ['B0', 'B1', 'B2', 'B2f', 'B3', 'B5']:
    tag = f'{b}_lag3'
    if not (O/f'{tag}_{ds}.npz').exists():
        hp = json.loads((O/f'{tag}.json').read_text())['hp']; N.log('START ' + tag + ' ' + ds); sc, P = C.run(b, hp, 3, ds); N.save(tag, ds, sc, P)
elif b == 'B6':
    if not (O/f'B6_lag3_{ds}.npz').exists():
        from latetruth import baselines_kim as K
        hp = json.loads((O/'B6_lag3.json').read_text())['hp']; N.log('START B6 ' + ds); sc, P = K.run(hp, 3, ds); N.save('B6_lag3', ds, sc, P)
elif b == 'B11':
    if not (O/f'B11_bai_xgb_{ds}.npz').exists():
        from latetruth import baselines_bai as BA
        hp = json.loads((O/'B11_bai_xgb.json').read_text())['hp']; N.log('START B11 ' + ds); sc, P = BA.run(hp, ds); N.save('B11_bai_xgb', ds, sc, P)
elif b == 'B10f':   # delegate to the exact code path of run_cpu_baselines (it skips every baseline whose file exists)
    import types
    src = Path(N.__file__).read_text(encoding='utf-8'); i = src.index("        if not (O/f'B10f_wang2022_frozen_{ds}.npz').exists():")
    j = src.index("            sh([PYTHON, '-c', code], ENV_C, f'B10f {ds}')") + len("            sh([PYTHON, '-c', code], ENV_C, f'B10f {ds}')")
    block = '\n'.join(l[8:] for l in src[i:j].split('\n'))
    exec(block, dict(vars(N), ds=ds))
