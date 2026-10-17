"""Portable locations and resource limits."""
import os
from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('DATA_DIR', './data')).resolve()
OUT_DIR = Path(os.environ.get('OUT_DIR', './results')).resolve()
PAPER_DIR = OUT_DIR / 'paper'
RAW_DIR = OUT_DIR / 'raw'
JOBS = max(1, min(8, int(os.environ.get('LATETRUTH_JOBS', '8'))))
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = str(min(JOBS, int(os.environ.get(name, str(JOBS)))))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def threadpool_limits(limits=None, **kwargs):
    """Honor the requested CPU cap without changing estimator settings."""
    from threadpoolctl import threadpool_limits as backend
    return backend(limits=min(JOBS, limits) if isinstance(limits, int) else JOBS, **kwargs)
