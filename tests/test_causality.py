"""Synthetic tests of released-row boundaries and initial-window scaling."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from latetruth.run_local_late_truth_long import local
from latetruth.residual_references import ewma


def test_local_matches_naive_and_excludes_boundary():
    rng = np.random.default_rng(8)
    z = rng.normal(size=(120, 3)); r = rng.normal(size=120)
    ok = rng.random(120) > 0.1; mem = 35; delay = 3; k = 20
    expected = np.zeros(120)
    for t in range(120):
        rows = np.array([s for s in range(120) if ok[s] and max(t-delay-mem,0) <= s < t-delay])
        if len(rows) > k:
            nearest = rows[np.argsort(((z[rows]-z[t])**2).sum(axis=1))[:k]]
            expected[t] = r[nearest].mean() * 0.5
    np.testing.assert_allclose(local(z,r,ok,mem),expected,rtol=0,atol=1e-15)
    changed = r.copy(); changed[57:] += 1000
    np.testing.assert_array_equal(local(z,r,ok,mem)[:61],local(z,changed,ok,mem)[:61])


def test_ewma_naive_release_warmup_and_future_invariance():
    rng = np.random.default_rng(19)
    r = rng.normal(size=150); ok = rng.random(150)>0.2; expected=np.zeros(150)
    for t in range(150):
        rows=np.flatnonzero(ok & (np.arange(150)<t-3))
        if len(rows)>20:
            w=0.5**((rows[-1]-rows)/18)
            expected[t]=np.sum(r[rows]*w)/w.sum()
    np.testing.assert_allclose(ewma(r,ok,18),expected,rtol=0,atol=1e-15)
    changed=r.copy();changed[70:]+=500
    np.testing.assert_array_equal(ewma(r,ok,18)[:74],ewma(changed,ok,18)[:74])
    assert np.all(ewma(r,np.ones(150,bool),18)[:24]==0)


def test_actual_pipeline_training_and_initial_scaling(monkeypatch):
    from latetruth import baselines_common as B
    from latetruth import run_extra_heat_paths as X
    n=100; cutoff=40; index=np.arange(n)
    frame=pd.DataFrame({name:index*(j+1)+j for j,name in enumerate(B.HELP)})
    truth=index.astype(float)*0.1
    month=np.ones(n,int); ok=np.ones(n,bool)
    captures=[]; models=[]
    class Model:
        def __init__(self,**kwargs): pass
        def fit(self,x,y):
            models.append(y.copy());self.mean=y.mean();return self
        def predict(self,x):return np.full(len(x),self.mean)
    def correction(z,r,valid,mem):
        captures.append(z.copy());return local(z,r,valid,mem)
    monkeypatch.setattr(B,'EV0',cutoff)
    monkeypatch.setattr(B,'windows',lambda m,n:[(40,70),(70,100)])
    monkeypatch.setattr(B,'HistGradientBoostingRegressor',Model)
    monkeypatch.setattr(B,'local',correction)
    monkeypatch.setattr(X,'feats',lambda df,rows,kind:df[B.HELP].to_numpy()[rows])
    monkeypatch.setattr(B,'load',lambda ds:({1:(frame,month)},n,month,{1:truth},{1:ok}))
    before=B.ours('synthetic')
    np.testing.assert_array_equal(models[0],truth[7:37])
    np.testing.assert_array_equal(models[1],truth[7:67])
    expected=(frame[B.HELP].to_numpy()[40:]-frame[B.HELP].to_numpy()[4:40].mean(0))/(frame[B.HELP].to_numpy()[4:40].std(0)+1e-9)
    np.testing.assert_allclose(captures[0],expected,rtol=0,atol=0)
    truth[61:]+=1000
    after=B.ours('synthetic')
    np.testing.assert_array_equal(before[-1][1][:61],after[-1][1][:61])
    np.testing.assert_array_equal(captures[0],captures[1])


def test_safe_loads():
    root=Path(__file__).resolve().parents[1]/'latetruth'
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.Call) and ast.unparse(node.func)=='np.load':
                assert any(k.arg=='allow_pickle' and isinstance(k.value,ast.Constant) and k.value.value is False for k in node.keywords),path


def test_per_turbine_scaling_uses_initial_rows(monkeypatch):
    import runpy
    from latetruth import baselines_common as B, run_extra_heat_paths as X
    from latetruth import run_local_late_truth_long as L
    n=100; index=np.arange(n); month=np.ones(n,int)
    frame=pd.DataFrame({name:np.sin(index/(j+1))+index for j,name in enumerate(L.HELP)})
    truth=index.astype(float); ok=np.ones(n,bool); captured=[]
    class Model:
        def __init__(self,**kwargs):pass
        def fit(self,x,y):return self
        def predict(self,x):return np.zeros(len(x))
    class Archive(dict):
        pass
    archive=Archive(Y_1=truth,OK_1=ok)
    monkeypatch.setattr(B,'EV0',40)
    monkeypatch.setattr(B,'OUT',Path(__file__).resolve().parents[1]/'.work/synthetic-control')
    monkeypatch.setattr(B,'score',lambda *args,**kwargs:{'all':0.0})
    monkeypatch.setattr(B,'windows',lambda m,n:[(40,100)])
    monkeypatch.setattr(B,'load',lambda ds:({1:(frame,month)},n,month,{1:truth},{1:ok}))
    monkeypatch.setattr(X,'feats',lambda df,rows,kind:df[L.HELP].to_numpy()[rows])
    monkeypatch.setattr(L,'HistGradientBoostingRegressor',Model)
    monkeypatch.setattr(L,'local',lambda z,r,ok,mem:captured.append(z.copy()) or np.zeros(len(r)))
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:archive)
    monkeypatch.setattr(np,'savez_compressed',lambda *args,**kwargs:None)
    monkeypatch.setattr(sys,'argv',['per_turbine_matched','kelmarsh2018'])
    runpy.run_module('latetruth.per_turbine_matched',run_name='__main__')
    initial=frame[L.HELP].to_numpy()[4:40]
    expected=(frame[L.HELP].to_numpy()[40:]-np.nanmean(initial,0))/(np.nanstd(initial,0)+1e-9)
    np.testing.assert_array_equal(captured[0],expected)


def test_paper_tier_reproduces_bytes():
    root=Path(__file__).resolve().parents[1]
    output=root/'.work/paper-test'
    raw=output/'raw/published_baselines_2026-10-14_v001'
    raw.mkdir(parents=True,exist_ok=True)
    (raw/'FINAL_TABLE_all.json').write_text('Unfinished numerical run',encoding='utf-8')
    env=dict(os.environ,OUT_DIR=str(output),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-m','latetruth','reproduce','--tier','paper'],cwd=root,env=env,capture_output=True,text=True,timeout=100)
    assert result.returncode==0,result.stdout+result.stderr
    assert (output/'paper/generated/numbers.tex').read_bytes()==(root/'generated/numbers.tex').read_bytes()
