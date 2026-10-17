"""Later held-out years (05_Protocols/LATER_YEARS_v001.md), CPU part, one job at a time; resumable (a step is skipped when
its output exists). For each dataset given: (1) pre-registered rules R1-R6 (Senvion) with the frozen Kelmarsh code,
(2) our pipeline + baseline inputs, (3) the CPU baselines with hyper-parameters frozen at their Kelmarsh 2017 choice.
usage: python run_later_years.py kelmarsh2024 [penmanshiel2022]   (Python; the pandas parts run with the Python)"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import os, sys, json, subprocess, time
from pathlib import Path
here = Path(__file__).resolve().parent
from latetruth import run_new_datasets as N   # reuses its logging, job runner, Python paths and the per-baseline code
O = N.O; LOG = OUT_DIR/'later_years.log'; N.LOG = LOG
ROOT = REPO_DIR

RULES = """
import sys; from latetruth.config import RAW_DIR as OUT_DIR; from latetruth import later_years as LY, run_kelmarsh_2018_unseen as K, run_extra_heat_paths as X, run_h29_h30_physics as B
from latetruth.run_local_late_truth_long import TARGET, HELP
ds, part = sys.argv[1], sys.argv[2]
K.OUT = OUT_DIR/f'later_years_{ds}_2026-10-15_v001'
K.read = lambda t: LY.read(ds, t, [TARGET, B.VANE] + HELP + X.EXTRA + K.PRES)
K.main(part)
"""
RULES_PEN = """
import sys; from latetruth.config import RAW_DIR as OUT_DIR; from latetruth import later_years as LY, run_penmanshiel_replication as P, run_extra_heat_paths as X, run_h29_h30_physics as B
from latetruth.run_local_late_truth_long import TARGET, HELP
ds = sys.argv[1]
P.OUT = OUT_DIR/f'later_years_{ds}_2026-10-15_v001'/'rules'
P.read = lambda t: LY.read(ds, t, [TARGET, B.VANE] + HELP + X.EXTRA)
P.main()
"""


def main():
    for ds in sys.argv[1:]:
        out = OUT_DIR/f'later_years_{ds}_2026-10-15_v001'; out.mkdir(exist_ok=True)
        if ds.startswith('hot'):   # Siemens: HILL_OF_TOWIE_REPLICATION_v001 parts A-C + Amendment 1 (R7/R8), then ours + inputs
            env = dict(N.ENV_C, HOT_YEAR=ds[3:])
            for part in 'ABC':
                if not (out/f'results_{part}.json').exists(): N.sh([N.PYTHON, '-m', 'latetruth.run_hill_of_towie_replication', part], env, f'rules {part} {ds}')
            if not (out/'results_h36.json').exists(): N.sh([N.PYTHON, '-m', 'latetruth.run_hill_of_towie_h36'], env, f'rules h36 {ds}')
            if not (O/f'inputs_{ds}.npz').exists(): N.sh([N.PYTHON, '-m', 'latetruth.baselines_hot'], env, f'ours+inputs {ds}')
            N.run_cpu_baselines(ds); continue
        if ds.startswith('penmanshiel'):   # Penmanshiel protocol (R1-R5) and turbine list, as in PENMANSHIEL_REPLICATION_v001
            if not (out/'rules'/'results.json').exists(): N.sh([N.PYTHON, '-c', RULES_PEN, ds], N.ENV_C, f'rules {ds}')
        else:
            for part in 'ABCD':
                if not (out/f'results_{part}.json').exists(): N.sh([N.PYTHON, '-c', RULES, ds, part], N.ENV_C, f'rules {part} {ds}')
        if not (O/f'ours_{ds}.npz').exists(): N.sh([N.PYTHON, '-m', 'latetruth.baselines_common', ds], N.ENV_C, f'ours {ds}')
        if not (O/f'inputs_{ds}.npz').exists():
            N.sh([N.PYTHON, '-c', f"from latetruth import baselines_common as B; B.export_inputs('{ds}')"], N.ENV_C, f'inputs {ds}')
        N.run_cpu_baselines(ds)
    N.log('LATER YEARS CPU PART DONE ' + ' '.join(sys.argv[1:]))


if __name__ == '__main__': main()
