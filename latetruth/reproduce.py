"""Sequential reproduction tiers with frozen scientific settings."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def module(name, *args, env=None):
    subprocess.run([sys.executable, '-B', '-m', 'latetruth.' + name, *args], check=True, env=env)


def variants(ds):
    import numpy as np
    from latetruth import baselines_common as B, residual_references as R
    from latetruth import baseline_wang2022_frozen as W
    from latetruth.config import OUT_DIR, REPO_DIR
    out = B.OUT
    with np.load(out / f'ours_{ds}.npz', allow_pickle=False) as z:
        turbines = sorted(int(k[2:]) for k in z.files if k.startswith('Y_'))
        refs = json.loads((REPO_DIR/'results'/out.name/'RESIDUAL_REFERENCES.json').read_text())
        hp = json.loads((REPO_DIR/'results'/out.name/'B10f_wang2022_frozen.json').read_text())['params(q,rf,ri)']
        outputs = {name: {} for name in ['rec', 'ref_ewma', 'ref_mean20', 'B10f_wang2022_frozen']}
        for t in turbines:
            y, ok, pf, po = (z[f'{k}_{t}'] for k in ['Y', 'OK', 'P_fleet', 'P_ours'])
            ev = np.arange(B.EV0, len(y)); valid = ok[ev] & np.isfinite(pf[ev])
            residual = np.where(valid, y[ev] - pf[ev], 0.0)
            valid2 = ok[ev] & np.isfinite(po[ev])
            rec = po.copy(); rec[ev] = po[ev] + 0.5 * R.ewma(np.where(valid2, y[ev] - po[ev], 0.0), valid2, 18)
            ewma = pf.copy(); ewma[ev] = pf[ev] + refs['ewma_c'] * R.ewma(residual, valid, refs['ewma_half_life_rows'])
            mean = pf.copy(); mean[ev] = pf[ev] + R.mean20(residual, valid)
            outputs['rec'][f'P_{t}'] = rec
            outputs['ref_ewma'][f'P_{t}'] = ewma
            outputs['ref_mean20'][f'P_{t}'] = mean
            if not (out/f'B10f_wang2022_frozen_{ds}.npz').exists():
                print(ds, 'fixed Kalman turbine', t, flush=True)
                outputs['B10f_wang2022_frozen'][f'P_{t}'] = W._turbine((pf, y, tuple(hp)))
        for name, arrays in outputs.items():
            if arrays and not (out/f'{name}_{ds}.npz').exists():
                np.savez_compressed(out/f'{name}_{ds}.npz', **arrays)


def gates(datasets):
    """Select the published standstill rule on development data only."""
    import numpy as np
    from latetruth import standstill_gate as G, baselines_classic as C
    from latetruth import baselines_common as B
    month, y, ok, ts, yr, stop, nac = G.prep('kelmarsh2017')
    with np.load(B.OUT/'ours_kelmarsh2017.npz', allow_pickle=False) as z:
        p = {t:z[f'P_ours_{t}'] for t in ts}
    a, na = G.fit_tau(month,y,ts,yr,stop,nac)
    grid=[]
    for form in ['hold','newton']:
        for h in G.HS:
            score=C.score(y,{t:G.gated(p[t],yr[t],stop[t],nac[t],form,h,a) for t in ts},ok,month)['all']
            grid.append((score,form,h))
    grid.sort(); best=grid[0]; base=C.score(y,p,ok,month)['all']
    selection=dict(tau_rows=float(-G.D/np.log(a)),a=a,n_jan_rows=na,ours_kelmarsh2017=base,
                   grid=[dict(mae=x,form=f,h=h) for x,f,h in grid],
                   chosen=dict(form=best[1],h=best[2],mae=best[0]),gate_kept=bool(best[0]<base))
    (B.OUT/'GATE_SELECTION.json').write_text(json.dumps(selection,indent=2))
    for ds in datasets:
        month,y,ok,ts,yr,stop,nac=G.prep(ds)
        with np.load(B.OUT/f'ours_{ds}.npz',allow_pickle=False) as z, np.load(B.OUT/f'rec_{ds}.npz',allow_pickle=False) as rec:
            gated={}; recg={}
            for t in ts:
                po=z[f'P_ours_{t}']; g=G.gated(po,yr[t],stop[t],nac[t],best[1],best[2],a)
                gated[f'P_{t}']=g
                recg[f'P_{t}']=np.where(np.abs(g-po)>1e-12,g,rec[f'P_{t}'])
            for tag,arrays in [('gate',gated),('recg',recg)]:
                if not (B.OUT/f'{tag}_{ds}.npz').exists():
                    np.savez_compressed(B.OUT/f'{tag}_{ds}.npz',**arrays)
    print('Development-only standstill selection:',selection['chosen'],flush=True)


def quick_metrics():
    import numpy as np
    from latetruth import baselines_common as B
    ds = 'kelmarsh2018'; actual = {}
    with np.load(B.OUT/f'ours_{ds}.npz', allow_pickle=False) as z:
        ts = sorted(int(k[2:]) for k in z.files if k.startswith('Y_'))
        y = {t: z[f'Y_{t}'] for t in ts}; ok = {t: z[f'OK_{t}'] for t in ts}
        for key in ['P_fleet', 'P_ours']:
            actual[key] = B.score(y, {t:z[f'{key}_{t}'] for t in ts}, ok, z['month'], len(z['month']))['all']
        for name in ['rec', 'ref_ewma', 'ref_mean20', 'B10f_wang2022_frozen']:
            with np.load(B.OUT/f'{name}_{ds}.npz', allow_pickle=False) as q:
                actual[name] = B.score(y, {t:q[f'P_{t}'] for t in ts}, ok, z['month'], len(z['month']))['all']
    return actual


def run(tier, data_dir, jobs, offline=False):
    os.environ['DATA_DIR'] = str(Path(data_dir).resolve())
    os.environ['OUT_DIR'] = str(Path(os.environ.get('OUT_DIR', './results')).resolve())
    os.environ['LATETRUTH_JOBS'] = str(jobs)
    os.environ.update({k:str(jobs) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    from latetruth.config import OUT_DIR, REPO_DIR, PAPER_DIR, RAW_DIR
    start = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if tier == 'paper':
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        for name in ['make_paper_numbers','make_paper_figures','make_paper_schematics']:
            module(name)
        actual = (PAPER_DIR/'generated/numbers.tex').read_bytes()
        if actual != (REPO_DIR/'generated/numbers.tex').read_bytes():
            raise RuntimeError('Paper numbers differ from shipped reference')
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        from latetruth.data import download, manifest
        datasets = ['kelmarsh2017','kelmarsh2018'] if tier == 'quick' else list(dict.fromkeys(x['dataset'] for x in manifest()))
        if not offline:
            download(data_dir, datasets)
        from latetruth import baselines_common as B
        B.OUT.mkdir(parents=True, exist_ok=True)
        # Frozen tuning records are inputs; computed outcomes stay in this output directory.
        import shutil
        for path in (REPO_DIR/'results'/B.OUT.name).glob('*.json'):
            if ('lag3' in path.name or path.name in ['B11_bai_xgb.json','B10f_wang2022_frozen.json','B10_wang2022.json']) and not (B.OUT/path.name).exists():
                shutil.copyfile(path,B.OUT/path.name)
        for ds in datasets:
            if not (B.OUT/f'ours_{ds}.npz').exists():
                print('Starting fleet and similarity estimator', ds, flush=True)
                if ds.startswith('hot'):
                    module('baselines_hot', env=dict(os.environ, HOT_YEAR=ds[3:]))
                else:
                    module('baselines_common', ds)
            if not ds.startswith('hot') and not (B.OUT/f'inputs_{ds}.npz').exists():
                B.export_inputs(ds)
            variants(ds)
            if tier != 'quick':
                if not (B.OUT/f'ptrec_{ds}.npz').exists():
                    module('per_turbine_matched', ds)
                from latetruth.run_new_datasets import run_cpu_baselines
                run_cpu_baselines(ds)
                online=B.OUT/f'B10_wang2022_{ds}.npz'
                if not online.exists():
                    import numpy as np
                    from latetruth import baseline_wang2022 as W
                    hp=json.loads((B.OUT/'B10_wang2022.json').read_text())['params(q,rf,ri,qth)']
                    _,predictions=W.apply(ds,tuple(hp))
                    np.savez_compressed(online,**{f'P_{t}':p for t,p in predictions.items()})
        gates(datasets)
        if tier == 'quick':
            actual = quick_metrics()
            expected = json.loads((REPO_DIR/'expected/quick.json').read_text())
            print(json.dumps(actual, indent=2))
            for name, value in actual.items():
                if abs(value - expected['mae'][name]) > 1e-6:
                    raise RuntimeError(f'MAE mismatch {name}: {value} vs {expected["mae"][name]}')
        if tier == 'full':
            print('Full tier can take days; deep baselines run sequentially.', flush=True)
            for ds in datasets:
                import numpy as np
                with np.load(B.OUT/f'ours_{ds}.npz', allow_pickle=False) as z:
                    ts = sorted(int(k[2:]) for k in z.files if k.startswith('Y_'))
                for method in ['B7','B4','B8','B9']:
                    for t in ts:
                        if not (B.OUT/f'deepv2_{method}_{ds}_T{t}.npz').exists():
                            module('baselines_deep', method, str(t), ds, env=dict(os.environ, DEEP_BUDGET='v2'))
            subsets=json.loads((REPO_DIR/'results'/B.OUT.name/'CAO_TABLE.json').read_text())
            for ds,record in subsets.items():
                if not isinstance(record,dict) or 'turbines' not in record:
                    continue
                for budget,prefix in [('v1','cao'),('v2','caov2')]:
                    for t in record['turbines']:
                        if not (B.OUT/f'{prefix}_{ds}_T{t}.npz').exists():
                            module('baselines_cao',str(t),ds,env=dict(os.environ,CAO_BUDGET=budget))
            for name in ['delay_part1','delay_part2','delay_rec']:
                module(name)
            # Run each dated rule evaluation only when its outcome is absent.
            for name,folder in [('run_extra_heat_paths','extra_heat_paths_2026-10-14_v001'),('run_send_schedule','send_schedule_2026-10-14_v001')]:
                if not (RAW_DIR/folder/'results.json').exists():
                    module(name)
            for part in 'ABCD':
                if not (RAW_DIR/'kelmarsh_2018_unseen_2026-10-13_v001'/f'results_{part}.json').exists():
                    module('run_kelmarsh_2018_unseen',part)
            if not (RAW_DIR/'penmanshiel_replication_2019_2026-10-14_v001/results.json').exists():
                module('run_penmanshiel_replication')
            for part in 'ABC':
                if not (RAW_DIR/'hill_of_towie_replication_2019_2026-10-13_v001'/f'results_{part}.json').exists():
                    module('run_hill_of_towie_replication',part,env=dict(os.environ,HOT_YEAR='2019'))
            if not (RAW_DIR/'hill_of_towie_replication_2019_2026-10-13_v001/results_h36.json').exists():
                module('run_hill_of_towie_h36',env=dict(os.environ,HOT_YEAR='2019'))
            module('run_later_years','kelmarsh2024','penmanshiel2022','hot2025')
            for name in ['run_freshness_value','run_h29_h30_physics']:
                target='freshness_value_2026-10-14_v001' if name=='run_freshness_value' else 'h29_h30_physics_2026-10-13_v001'
                if not (RAW_DIR/target/'results.json').exists():
                    module(name)
            from latetruth.config import DATA_DIR
            archive=str(DATA_DIR/'HillOfTowie_04/2019.zip')
            for name,folder in [('physics_oil_circuit_check','physics_oil_circuit_2026-10-15_v001'),('filter_index','filter_index_2026-10-15_v001')]:
                (RAW_DIR/folder).mkdir(parents=True,exist_ok=True)
                if not (RAW_DIR/folder/'results.json').exists():
                    module(name,archive)
            module('figure_data_oil',archive)
        if tier in ['main','full']:
            for name in ['standstill_gate','residual_references','recency_on_similarity','confirm_two_timescale','baselines_table','significance','block_bootstrap','turbine_level','compare_per_turbine','verify_final_table']:
                module(name)
            module('rec_vs_baselines','all9')
    print(f'PASS: {tier} ({time.monotonic()-start:.2f} seconds)', flush=True)
