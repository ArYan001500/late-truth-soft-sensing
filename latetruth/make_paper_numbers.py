"""Paper numbers. Writes Paper/generated/numbers.tex (one \\val{key} for every number quoted in the Results text) and the Results
tables Paper/tables/tab_rules.tex, tab_comparison.tex and tab_significance.tex, all read directly from the saved result files,
so that no number in the paper is typed by hand. Standard library only.   usage: python make_paper_numbers.py"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json, re, math, datetime
from pathlib import Path

ROOT = REPO_DIR; E = REPO_DIR/'results'; PB = E/'published_baselines_2026-10-14_v001'
K18 = E/'kelmarsh_2018_unseen_2026-10-13_v001'; P19 = E/'penmanshiel_replication_2019_2026-10-14_v001'
H19 = E/'hill_of_towie_replication_2019_2026-10-13_v001'
PAPER = PAPER_DIR; SRC, V = [], {}
MON = ['February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
DS = [('k18', 'kelmarsh2018'), ('p19', 'penmanshiel2019'), ('h19', 'hot2019')]
OURS, FLEET = 'ours (fleet + local late-truth correction)', 'our fleet model (no late truth)'
# (row label, FINAL_TABLE key) in the order of Table II; Cao is added by baselines_table.py once its runs are complete
PUB = [('Schlechtingen and Santos, AR~\\cite{Schlechtingen2011}', 'B2 Schlechtingen 2011 AR [lag 3]'),
       ('Schlechtingen and Santos, FSRC~\\cite{Schlechtingen2011}', 'B2f Schlechtingen 2011 FSRC [lag 3]'),
       ('Kim et al.~\\cite{Kim2013}', 'B6 Kim 2013 [lag 3]'), ('Bangalore and Tjernberg~\\cite{Bangalore2015}', 'B1 Bangalore 2015 [lag 3]'),
       ('Fu et al.~\\cite{Fu2019}', 'B4 Fu 2019 [v2]'), ('Liu et al.~\\cite{Liu2020}', 'B3 Liu 2020 [lag 3]'),
       ('Yang et al.~\\cite{Yang2021}', 'B5 Yang 2021 [lag 3]'), ('Wang et al.~\\cite{Wang2022}, fixed', 'B10f Wang 2022 (frozen NN)'),
       ('Wang et al.~\\cite{Wang2022}, online', 'B10 Wang 2022 (online, as published)'), ('Yan et al.~\\cite{Yan2023}', 'B7 Yan 2023 [v2]'),
       ('Bai et al.~\\cite{Bai2024}', 'B11 Bai 2024'), ('Dai et al.~\\cite{Dai2025}', 'B8 Dai 2025 [v2]'),
       ('Grataloup et al.~\\cite{Grataloup2025}', 'B9 Grataloup 2025 [v2]'), ('Cao et al.~\\cite{Cao2026}', 'B12 Cao 2026')]
SHORT = {k: re.sub(r'~\\cite\{[^}]*\}', '', lab) for lab, k in PUB}; SHORT[FLEET] = 'Fleet model alone'


def load(p):
    SRC.append('06_Experiments/' + p.relative_to(E).as_posix()); return json.loads(p.read_text(encoding='utf-8', errors='replace'))


def res(p):
    d = load(p); return d.get('results', d)


def pick(r, prefix):
    """The one entry whose key starts with `prefix` (keys with a degree sign were saved with a broken encoding)."""
    k = [k for k in r if k.startswith(prefix)]
    if len(k) != 1: raise KeyError(f'{prefix!r} matches {k}')
    return r[k[0]]


def put(k, v):
    if k in V: raise KeyError('duplicate key ' + k)
    V[k] = v


def f3(x): return f'{x:.3f}'
def f4(x): return f'{x:.4f}'
def p1(x): return f'{x:.1f}'
def red(a, b): return 100 * (1 - b / a)    # % reduction from a to b
def inc(a, b): return 100 * (b / a - 1)    # % increase from a to b
def ptex(p): m, e = f'{p:.1e}'.split('e'); return f'${m}{{\\times}}10^{{{int(e)}}}$'
def signed(x, nd=1): return f'$-${abs(x):.{nd}f}' if round(x, nd) < 0 else f'+{abs(x):.{nd}f}'


# ---------------------------------------------------------------- design ladder (Kelmarsh 2017, development year)
lad = {k.split()[0]: v for k, v in load(PB/'DESIGN_LADDER_kelmarsh2017.json')['results'].items()}
for r, v in lad.items():
    put(f'lad.{r}', f3(v['all'])); put(f'lad.{r}.tr', f3(v['transient'])); put(f'lad.{r}.calm', f3(v['calm']))
for k, a, b in [('pool', 'L1', 'L2'), ('phys', 'L2', 'L3'), ('bear', 'L3', 'L4'), ('loc', 'L4', 'L5'), ('retrain', 'L0', 'L4'), ('total', 'L1', 'L5')]:
    put(f'lad.{k}', p1(red(lad[a]['all'], lad[b]['all']))); put(f'lad.{k}.tr', p1(red(lad[a]['transient'], lad[b]['transient'])))
    put(f'lad.{k}.calm', p1(red(lad[a]['calm'], lad[b]['calm'])))
pool = [red(a, b) for a, b in zip(lad['L1']['months'], lad['L2']['months'])]; o = sorted(range(11), key=lambda i: -pool[i])
put('lad.pool.m1', MON[o[0]]); put('lad.pool.m1.pct', p1(pool[o[0]])); put('lad.pool.m2', MON[o[1]]); put('lad.pool.m2.pct', p1(pool[o[1]]))
put('lad.pool.mmin', MON[o[-1]]); put('lad.pool.mmin.pct', p1(pool[o[-1]]))
hi1 = sorted(range(11), key=lambda i: -lad['L1']['months'][i])[:2]
assert set(hi1) == set(o[:2]), 'text says the two largest pooling gains fall in the two worst per-turbine months'
loc = [red(a, b) for a, b in zip(lad['L4']['months'], lad['L5']['months'])]; b = max(range(11), key=lambda i: loc[i])
put('lad.loc.nmonths', str(sum(x > 0 for x in loc))); put('lad.loc.mbest', MON[b]); put('lad.loc.mbest.pct', p1(loc[b]))

# ---------------------------------------------------------------- final table, significance
FT = load(PB/'FINAL_TABLE_all.json'); SG = load(PB/'SIGNIFICANCE.json')


def ft(ds, key, col='mae_common_rows'):
    r = [x for x in FT.get(ds, []) if x['method'] == key]
    return r[0][col] if r else None


put('k17.ours', f3(ft('kelmarsh2017', OURS, 'mae_own_rows'))); put('k18.ours.own', f3(ft('kelmarsh2018', OURS, 'mae_own_rows')))
put('k18.harder', p1(inc(ft('kelmarsh2017', OURS, 'mae_own_rows'), ft('kelmarsh2018', OURS, 'mae_own_rows'))))
put('k18.fleet.harder', p1(inc(ft('kelmarsh2017', FLEET, 'mae_own_rows'), ft('kelmarsh2018', FLEET, 'mae_own_rows'))))
WANG, KIM, YAN = 'B10f Wang 2022 (frozen NN)', 'B6 Kim 2013 [lag 3]', 'B7 Yan 2023 [v2]'
for s, ds in DS:
    o_, f_, w_ = ft(ds, OURS), ft(ds, FLEET), ft(ds, WANG)
    put(f'{s}.ours', f3(o_)); put(f'{s}.fleet', f3(f_)); put(f'{s}.wang', f3(w_))
    put(f'{s}.vswang', p1(red(w_, o_)))                  # ours below the Kalman filter, % of its error
    put(f'{s}.loc', p1(red(f_, o_)))                     # local correction, % of the fleet error
    put(f'{s}.kal', signed(-red(f_, w_)))                # Kalman update: change of the fleet error (signed %)
    pub = [(ft(ds, k), k) for _, k in PUB if ft(ds, k) is not None and k != WANG and 'online' not in k]
    m, k = min(pub); put(f'{s}.best', f3(m)); put(f'{s}.best.name', SHORT[k]); put(f'{s}.best.pct', p1(inc(o_, m)))
    put(f'{s}.pers', f3(ft(ds, 'B0 last released value [lag 3]')))
    for r in SG[ds]:
        key = {WANG: 'wang', FLEET: 'fleet', KIM: 'kim', YAN: 'yan'}.get(r['baseline'])
        if key:
            put(f'sig.{s}.{key}.wins', f"{r['wins_turbine_month']}/{r['n_pairs']}"); put(f'sig.{s}.{key}.p', ptex(r['p_holm']))
            put(f'sig.{s}.{key}.lo', f3(r['ci95'][0])); put(f'sig.{s}.{key}.hi', f3(r['ci95'][1])); put(f'sig.{s}.{key}.diff', f3(r['mae_diff']))
    put(f'sig.{s}.n', str(SG[ds][0]['n_pairs'])); put(f'sig.{s}.pmax', ptex(max(r['p_holm'] for r in SG[ds])))
    put(f'sig.{s}.nbase', str(len(SG[ds]))); assert all(r['ours_better'] for r in SG[ds]), ds
    put(f'sig.{s}.minwins', f"{min(r['wins_turbine_month'] for r in SG[ds])}/{SG[ds][0]['n_pairs']}")
put('k18.sync.kim', f3(ft('kelmarsh2018', 'B6 Kim 2013 [lag 1]'))); put('k18.sync.kim.pct', p1(inc(ft('kelmarsh2018', OURS), ft('kelmarsh2018', 'B6 Kim 2013 [lag 1]'))))
lag1 = [(ft('kelmarsh2018', k), k) for k in [x['method'] for x in FT['kelmarsh2018']] if '[lag 1]' in k]; m, k = min(lag1)
assert k == 'B6 Kim 2013 [lag 1]', k
dv1 = {k: (ft('kelmarsh2018', k.replace('v2', 'v1')), ft('kelmarsh2018', k)) for k in ['B7 Yan 2023 [v2]', 'B4 Fu 2019 [v2]', 'B8 Dai 2025 [v2]', 'B9 Grataloup 2025 [v2]']}
gain = [a - b for a, b in dv1.values()]; put('deep.v2gain.min', f'{min(gain):.2f}'); put('deep.v2gain.max', f'{max(gain):.2f}')
put('k18.wang.online', f3(ft('kelmarsh2018', 'B10 Wang 2022 (online, as published)')))
put('sig.pmax.held', ptex(max(r['p_holm'] for _, ds in DS for r in SG[ds])))   # the three held-out farm-years of Section VI-C
assert V['k18.best.name'] == V['p19.best.name'] == 'Yan et al.' and V['h19.best.name'] == 'Kim et al.', 'text names the best published methods'
NBM = ['B2f Schlechtingen 2011 FSRC [lag 3]', 'B5 Yang 2021 [lag 3]', 'B8 Dai 2025 [v2]', 'B9 Grataloup 2025 [v2]']
nbm = [ft(ds, k) for _, ds in DS for k in NBM]; put('nbm.min', f'{min(nbm):.2f}'); put('nbm.max', f'{max(nbm):.2f}')
for _, ds in DS:   # on each farm these four are the least accurate published models (the diverged online Kalman filter aside)
    others = [ft(ds, k) for _, k in PUB if k not in NBM and ft(ds, k) is not None and 'online' not in k]
    assert max(others) < min(ft(ds, k) for k in NBM), ds

# ---------------------------------------------------------------- release delay (S3) and claim C3
D1, D2 = load(PB/'DELAY_SENSITIVITY_part1.json'), load(PB/'DELAY_SENSITIVITY_part2.json'); DL = {**D1, **D2}
NM = {'ours': 'ours', 'our fleet model (no late truth)': 'fleet', 'Wang 2022 (frozen NN)': 'wang', 'Kim 2013': 'kim', 'Bangalore 2015': 'bang'}
for k, s in NM.items():
    for d in ['1', '3', '6', '12', '24']: put(f'dl.{s}.{d}', f3(DL[k][d]['all']))
    put(f'dl.{s}.inc', p1(inc(DL[k]['1']['all'], DL[k]['12']['all']))); put(f'dl.{s}.abs', f3(DL[k]['12']['all'] - DL[k]['1']['all']))
loc_d = [red(DL[FLEET][d]['all'], DL['ours'][d]['all']) for d in ['1', '3', '6', '12', '24']]
kal_d = [inc(DL[FLEET][d]['all'], DL['Wang 2022 (frozen NN)'][d]['all']) for d in ['1', '3', '6', '12', '24']]
put('dl.loc.min', p1(min(loc_d))); put('dl.loc.max', p1(max(loc_d))); put('dl.kal.min', signed(min(kal_d), 2)); put('dl.kal.max', signed(max(kal_d), 2))
assert [k < 0 for k in kal_d] == [False, True, False, False, False], 'text: the filter improved the fleet model only at d = 3'
put('dl.kal.bestgain', f'{max(-min(kal_d), 0):.2f}'); put('dl.kal.worst', f'{max(kal_d):.2f}')
yan = ft('kelmarsh2018', YAN, 'mae_own_rows'); put('dl.yan', f3(yan))
for d in ['1', '3', '6', '12', '24']:   # qualitative statements of the text
    assert DL['ours'][d]['all'] == min(DL[k][d]['all'] for k in NM) and DL['ours'][d]['all'] < yan, d
for k in ['Kim 2013', 'Bangalore 2015']: assert DL[k]['1']['all'] < yan < DL[k]['3']['all'], k
d3 = {'ours': OURS, 'our fleet model (no late truth)': FLEET, 'Wang 2022 (frozen NN)': WANG, 'Kim 2013': KIM, 'Bangalore 2015': 'B1 Bangalore 2015 [lag 3]'}
assert [k for k, v in d3.items() if abs(DL[k]['3']['all'] - ft('kelmarsh2018', v, 'mae_own_rows')) > 1e-9] == ['Bangalore 2015'], 'text: only Bangalore re-run differs at d = 3'
put('dl.rerun.maxdiff', f3(math.ceil(1000 * max(abs(DL[k]['3']['all'] - ft('kelmarsh2018', v, 'mae_own_rows')) for k, v in d3.items())) / 1000))
put('dl.ours.10to30', p1(inc(DL['ours']['1']['all'], DL['ours']['3']['all']))); put('dl.vswang.24', p1(red(DL['Wang 2022 (frozen NN)']['24']['all'], DL['ours']['24']['all'])))
sync = load(PB/'ours_sync_kelmarsh2018.json')['ours_sync']['all']; o3 = ft('kelmarsh2018', OURS, 'mae_own_rows')
assert abs(sync - DL['ours']['1']['all']) < 1e-12
put('c3.ours.rel', p1(inc(sync, o3))); put('c3.ours.abs', f3(o3 - sync))
c3 = [(inc(ft('kelmarsh2018', f'{b} [lag 1]', 'mae_own_rows'), ft('kelmarsh2018', f'{b} [lag 3]', 'mae_own_rows')),
       ft('kelmarsh2018', f'{b} [lag 3]', 'mae_own_rows') - ft('kelmarsh2018', f'{b} [lag 1]', 'mae_own_rows'))
      for b in ['B6 Kim 2013', 'B1 Bangalore 2015', 'B3 Liu 2020', 'B2 Schlechtingen 2011 AR']]
put('c3.rel.min', f'{min(r for r, _ in c3):.0f}'); put('c3.rel.max', f'{max(r for r, _ in c3):.0f}')
put('c3.abs.min', f3(min(a for _, a in c3))); put('c3.abs.max', f3(max(a for _, a in c3)))

# ---------------------------------------------------------------- local-correction grid (S4)
G = load(PB/'LOCAL_GRID_kelmarsh2018.json'); best = min(G, key=lambda r: r['mae']); pre = [r for r in G if r['k'] == 20 and r['memory'] == 8640][0]
put('grid.min', f4(min(r['mae'] for r in G))); put('grid.max', f4(max(r['mae'] for r in G))); put('grid.pre', f4(pre['mae']))
put('grid.best.k', str(best['k'])); put('grid.best.M', str(best['memory'])); put('grid.best.days', str(best['memory'] // 144)); put('grid.gap', f4(pre['mae'] - best['mae']))
for k in [5, 80]:
    g = [r['mae'] for r in G if r['k'] == k]; put(f'grid.k{k}.lo', f4(min(g))); put(f'grid.k{k}.hi', f4(max(g)))
for k in [20]:
    g = [r['mae'] for r in G if r['k'] == k]; put(f'grid.k{k}.lo', f4(min(g))); put(f'grid.k{k}.hi', f4(max(g)))
put('grid.n', str(len(G))); put('grid.wang.own', f4(ft('kelmarsh2018', WANG, 'mae_own_rows'))); put('grid.fleet.own', f4(ft('kelmarsh2018', FLEET, 'mae_own_rows')))
assert max(r['mae'] for r in G) < ft('kelmarsh2018', WANG, 'mae_own_rows')

# ---------------------------------------------------------------- pre-registered rules R1-R8 on the held-out farm-years
A18, B18, C18, D18 = (res(K18/f'results_{x}.json') for x in 'ABCD'); R19 = res(P19/'results.json')
AH, BH, CH, HH = (res(H19/f'results_{x}.json') for x in ['A', 'B', 'C', 'h36'])
S18, SH = {**B18, **C18}, {**BH, **CH}
RU = {}   # rule -> farm -> (cell text, verdict)


def months_better(a, b): return sum(y < x for x, y in zip(a['months'], b['months']))


for s, R in [('k18', A18), ('p19', R19)]:
    RU.setdefault('R1', {})[s] = red(R['h30+local']['all'], R['h30_x+local']['all'])
    RU.setdefault('R2', {})[s] = (red(R['h30_x']['all'], R['h30_x+local']['all']), months_better(R['h30_x'], R['h30_x+local']))
    RU.setdefault('R3', {})[s] = (red(R['base']['transient'], R['h29']['transient']), inc(R['base']['calm'], R['h29']['calm']))
RU['R1']['h19'] = red(AH['base+local']['all'], AH['base_x+local']['all'])
RU['R2']['h19'] = (red(AH['base_x']['all'], AH['base_x+local']['all']), months_better(AH['base_x'], AH['base_x+local']))
for s in ['k18', 'p19', 'h19']:
    put(f'r1.{s}', p1(RU['R1'][s])); put(f'r2.{s}', p1(RU['R2'][s][0])); put(f'r2.{s}.m', str(RU['R2'][s][1]))
for s in ['k18', 'p19']: put(f'r3.{s}.tr', p1(RU['R3'][s][0])); put(f'r3.{s}.calm', p1(-RU['R3'][s][1]))
put('r1.min', p1(min(RU['R1'].values()))); put('r1.max', p1(max(RU['R1'].values())))
fresh = {}   # farm -> (all-fresh MAE, {channel: % increase}, joint % increase)
SEN = {'Power': 'power', 'Wind speed': 'wind speed', 'Rotor speed': 'rotor speed', 'Generator RPM': 'generator speed',
       'Nacelle ambient': 'ambient temperature', 'Nacelle temperature': 'nacelle temperature', 'Grid current': 'grid current',
       'Generator bearing rear': 'generator bearing', 'Gear oil inlet': 'oil inlet', 'Front bearing': 'front axle bearing',
       'Rear bearing': 'rear axle bearing', 'Rotor bearing': 'rotor bearing', 'Gearbox speed': 'gearbox speed'}
SIE = {'wtc_ActPower': 'power', 'wtc_AcWindSp': 'wind speed', 'wtc_MainSRpm': 'rotor speed', 'wtc_GenRpm': 'generator speed',
       'wtc_AmbieTmp': 'ambient temperature', 'wtc_NacelTmp': 'nacelle temperature', 'wtc_AmpPhR': 'grid current',
       'wtc_GenBeGTm': 'generator bearing', 'wtc_HSGenTmp': 'HS bearing, generator side', 'wtc_HSRotTmp': 'HS bearing, rotor side',
       'wtc_IMSGenTm': 'IMS bearing, generator side', 'wtc_IMSRotTm': 'IMS bearing, rotor side', 'wtc_MainBTmp': 'main bearing'}
FV = res(E/'freshness_value_2026-10-14_v001/results.json'); b0 = FV['all_fresh']['all']
txt = (E/'freshness_value_2026-10-14_v001/both_stale_output.txt').read_text(encoding='utf-8-sig'); SRC.append('06_Experiments/freshness_value_2026-10-14_v001/both_stale_output.txt')
j17 = float(re.search(r'stale: all ([0-9.]+)', txt).group(1))
fresh['k17'] = (b0, {n: inc(b0, pick(FV, p)['all']) for p, n in SEN.items()}, inc(b0, j17))
fresh['k18'] = (A18['h30_x']['all'], {n: inc(A18['h30_x']['all'], pick(S18, 'stale:' + p)['all']) for p, n in SEN.items()},
                inc(A18['h30_x']['all'], D18['stale:inlet+bearings']['all']))
fresh['p19'] = (R19['h30_x']['all'], {n: inc(R19['h30_x']['all'], pick(R19, 'stale:' + p)['all']) for p, n in SEN.items()},
                inc(R19['h30_x']['all'], R19['stale:inlet+bearings']['all']))
fresh['h19'] = (AH['base_x']['all'], {n: inc(AH['base_x']['all'], pick(SH, 'stale:' + p)['all']) for p, n in SIE.items()},
                inc(AH['base_x']['all'], AH['stale:all_gearbox_bearings']['all']))
CARRIERS = {'k17': ['oil inlet', 'rear axle bearing', 'front axle bearing'], 'k18': ['oil inlet', 'rear axle bearing', 'front axle bearing'],
            'p19': ['oil inlet', 'rear axle bearing', 'front axle bearing'],
            'h19': ['HS bearing, generator side', 'HS bearing, rotor side', 'IMS bearing, generator side', 'IMS bearing, rotor side']}
for s, (b_, ch, jt) in fresh.items():
    for n, v in ch.items(): put(f'fr.{s}.{n.replace(" ", "").replace(",", "")}', p1(v))
    oth = [v for n, v in ch.items() if n not in CARRIERS[s]]
    put(f'fr.{s}.oth.min', signed(min(oth))); put(f'fr.{s}.oth.max', signed(max(oth)))
    put(f'fr.{s}.joint', f'{jt:.0f}'); put(f'fr.{s}.sum', p1(sum(ch[n] for n in CARRIERS[s]))); put(f'fr.{s}.ratio', f'{jt / sum(ch[n] for n in CARRIERS[s]):.1f}')
    put(f'fr.{s}.base', f3(b_))
NAMED = {'k17': ['oil inlet', 'rear axle bearing'], 'k18': ['oil inlet', 'rear axle bearing'],   # channels named in the text
         'p19': ['oil inlet', 'rear axle bearing', 'front axle bearing'], 'h19': CARRIERS['h19']}
rest = [v for s in fresh for n, v in fresh[s][1].items() if n not in NAMED[s]]
put('fr.oth.min.all', signed(min(rest))); put('fr.oth.max.all', signed(max(rest))); assert min(rest) < 0 < max(rest), 'text: both directions'
sen = [fresh[s][2] / sum(fresh[s][1][n] for n in CARRIERS[s]) for s in ['k17', 'k18', 'p19']]
put('fr.sen.ratio.min', f'{min(sen):.1f}'); put('fr.sen.ratio.max', f'{max(sen):.1f}')
put('fr.h19.nontemp.max', p1(max(fresh['h19'][1][n] for n in ['power', 'wind speed', 'rotor speed', 'generator speed', 'grid current'])))
put('fr.allstale.h19', f3(AH['stale:all_gearbox_bearings']['all'])); put('fr.nobear.h19', f3(AH['base']['all']))
put('fr.noise', p1(max(-min(v for s in fresh for v in fresh[s][1].values()), 0)))
r4 = {'k18': fresh['k18'][1]['oil inlet'] > 10 and fresh['k18'][1]['rear axle bearing'] > 10 and max(v for n, v in fresh['k18'][1].items() if n not in ['oil inlet', 'rear axle bearing']) < 3,
      'p19': False}
assert r4['k18'] and fresh['k18'][2] > fresh['k18'][1]['oil inlet'] + fresh['k18'][1]['rear axle bearing'] + fresh['k18'][1]['front axle bearing']
SP = res(E/'send_schedule_2026-10-14_v001/results.json')
for k in ['always_bearing', 'always_inlet', 'alternate', 'event']: put(f'sp.k17.{k}', f3(SP[k]['all']))
put('sp.k17.event.share', f"{100 * SP['event']['bearing_fresh_share']:.0f}")
for s, R in [('k18', D18), ('p19', R19)]:
    put(f'sp.{s}.always_bearing', f3(R['policy:always_bearing']['all'])); put(f'sp.{s}.always_inlet', f3(R['policy:always_inlet']['all']))
put('oil.k18.stale', f3(D18['stale:inlet+bearings']['all'])); put('oil.k18.press', f3(D18['stale:inlet+bearings+pressure']['all']))
put('oil.k18.pct', p1(red(D18['stale:inlet+bearings']['all'], D18['stale:inlet+bearings+pressure']['all'])))
st, pr, r7, r8 = AH['stale:all_gearbox_bearings']['all'], AH['stale:all_gearbox_bearings+pressure']['all'], HH['stale:all_gearbox_bearings+pressure+r7']['all'], HH['stale:all_gearbox_bearings+pressure+r8']['all']
put('oil.h19.stale', f3(st)); put('oil.h19.press', f3(pr)); put('oil.h19.pct', p1(red(st, pr))); put('oil.h19.r7', f3(r7)); put('oil.h19.r7pct', p1(red(pr, r7)))
put('oil.h19.r8', f3(r8)); put('oil.h19.r8pct', p1(red(r7, r8))); put('oil.h19.basex', f3(AH['base_x']['all']))
for n, x in [('r6', pr), ('r7', r7)]: put(f'oil.h19.recover.{n}', f'{100 * (st - x) / (st - AH["base_x"]["all"]):.0f}')
put('oil.k18.recover', f"{100 * (D18['stale:inlet+bearings']['all'] - D18['stale:inlet+bearings+pressure']['all']) / (D18['stale:inlet+bearings']['all'] - A18['h30_x']['all']):.0f}")
for n, R in [('press', AH['stale:all_gearbox_bearings+pressure']), ('r7', HH['stale:all_gearbox_bearings+pressure+r7'])]:
    put(f'oil.h19.{n}.calm', f3(R['calm'])); put(f'oil.h19.{n}.tr', f3(R['transient']))
PH = load(E/'physics_oil_circuit_2026-10-15_v001/results.json'); bins = PH['hs_stage_bins']
put('phy.a', f"{PH['per_turbine_resistance']['a_flow']:.2f}"); put('phy.b', f"{PH['per_turbine_resistance']['b_viscosity']:.2f}")
put('phy.r2', f"{PH['per_turbine_resistance']['r2']:.2f}"); put('phy.r2.pooled', f"{PH['pooled']['r2']:.2f}"); put('phy.n', f"{PH['n_rows']:,}".replace(',', '\\,'))
put('phy.dp.cold', f"{bins['30-35']['median_dP_bar']:.2f}"); put('phy.dp.warm', f"{bins['45-50']['median_dP_bar']:.2f}")
put('phy.dp.ratio', f"{bins['30-35']['median_dP_bar'] / bins['45-50']['median_dP_bar']:.2f}")
put('phy.nu.ratio', f"{bins['30-35']['walther_nu_cSt'] / bins['45-50']['walther_nu_cSt']:.2f}")

# ---------------------------------------------------------------- Discussion diagnostics (Amendment 13, exploratory)
RS, RW = load(PB/'DISCUSSION_RECENCY_VS_SIMILARITY.json'), load(PB/'DISCUSSION_RECENCY_VS_SIMILARITY_WINS.json')
for s, ds in [('k17', 'kelmarsh2017')] + DS:
    r = RS[ds]; put(f'rs.{s}.last1', signed(inc(r['fleet'], r['last1']))); put(f'rs.{s}.last20', p1(red(r['fleet'], r['last20'])))
    put(f'rs.{s}.sim', p1(red(r['fleet'], r['similar20']))); put(f'rs.{s}.extra', p1(red(r['last20'], r['similar20'])))
    put(f'rs.{s}.share', f"{100 * (r['fleet'] - r['last20']) / (r['fleet'] - r['similar20']):.0f}")
    put(f'rs.{s}.wins', f"{RW[ds]['wins_similar_over_last20']}/{RW[ds]['turbine_months']}")
sen = [RS[d] for d in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019']]
put('rs.sen.last1.min', p1(min(inc(r['fleet'], r['last1']) for r in sen))); put('rs.sen.last1.max', p1(max(inc(r['fleet'], r['last1']) for r in sen)))
put('rs.sen.extra.min', p1(min(red(r['last20'], r['similar20']) for r in sen))); put('rs.sen.extra.max', p1(max(red(r['last20'], r['similar20']) for r in sen)))
put('rs.share.min', f"{min(100 * (r['fleet'] - r['last20']) / (r['fleet'] - r['similar20']) for r in RS.values() if isinstance(r, dict)):.0f}")
put('rs.share.max', f"{max(100 * (r['fleet'] - r['last20']) / (r['fleet'] - r['similar20']) for r in RS.values() if isinstance(r, dict)):.0f}")
put('rs.h19.last1.abs', p1(red(RS['hot2019']['fleet'], RS['hot2019']['last1'])))
put('rs.pmax', ptex(max(RW[d]['sign_test_p'] for d in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019'])))
assert all(inc(r['fleet'], r['last1']) > 0 for r in sen) and RS['hot2019']['last1'] < RS['hot2019']['fleet'], 'text: newest residual hurts on Senvion, helps on Siemens'
for _, ds in DS:   # text: the fleet model alone beats every published method that does not build on it
    assert ft(ds, FLEET) < min(ft(ds, k) for _, k in PUB if ft(ds, k) is not None and k not in (WANG, 'B10 Wang 2022 (online, as published)')), ds
FI = load(E/'filter_index_2026-10-15_v001/results.json')['summary']
put('fi.window', str(FI['max_turbines_within_any_6_weeks'])); put('fi.n', str(FI['n_turbines'])); put('fi.drop', f"{-FI['drops_median']:.2f}")
put('fi.before.neg', str(FI['n_turbines'] - FI['index_before_drop_positive']))

# ---------------------------------------------------------------- later held-out years (LATER_YEARS_v001)
DSL = [('k24', 'kelmarsh2024'), ('p22', 'penmanshiel2022'), ('h25', 'hot2025')]
for s, ds in DSL:
    o_, f_, w_ = ft(ds, OURS), ft(ds, FLEET), ft(ds, WANG)
    put(f'{s}.ours', f3(o_)); put(f'{s}.fleet', f3(f_)); put(f'{s}.wang', f3(w_)); put(f'{s}.vswang', p1(red(w_, o_))); put(f'{s}.loc', p1(red(f_, o_)))
    put(f'{s}.ours.own', f3(ft(ds, OURS, 'mae_own_rows')))
    pub = [(ft(ds, k), k) for _, k in PUB if ft(ds, k) is not None and k != WANG and 'online' not in k]
    m, k = min(pub); put(f'{s}.best', f3(m)); put(f'{s}.best.name', SHORT[k]); put(f'{s}.best.pct', p1(inc(o_, m)))
    put(f'{s}.pers', f3(ft(ds, 'B0 last released value [lag 3]'))); put(f'sig.{s}.n', str(SG[ds][0]['n_pairs'])); put(f'sig.{s}.nbase', str(len(SG[ds])))
    put(f'sig.{s}.nbetter', str(sum(r['ours_better'] for r in SG[ds]))); put(f'sig.{s}.pmax', ptex(max(r['p_holm'] for r in SG[ds])))
    for r in SG[ds]:
        key = {WANG: 'wang', FLEET: 'fleet', KIM: 'kim', YAN: 'yan'}.get(r['baseline'])
        if key:
            put(f'sig.{s}.{key}.wins', f"{r['wins_turbine_month']}/{r['n_pairs']}"); put(f'sig.{s}.{key}.p', ptex(r['p_holm']))
            put(f'sig.{s}.{key}.lo', f3(r['ci95'][0])); put(f'sig.{s}.{key}.hi', f3(r['ci95'][1])); put(f'sig.{s}.{key}.diff', f3(r['mae_diff']))
assert [r['baseline'] for r in SG['kelmarsh2024'] if not r['ours_better']] == [KIM], 'text: only Kim et al. not shown on Kelmarsh 2024'
assert all(r['ours_better'] for ds in ['penmanshiel2022', 'hot2025'] for r in SG[ds])
XTRA = ('proposed (similarity + recency)', 'proposed + standstill gate', 'reference: mean of 20 newest residuals', 'reference: EWMA of residuals')
assert all(ft(ds, OURS) == min(x['mae_common_rows'] for x in FT[ds] if x['method'] not in XTRA) for _, ds in DSL), 'similarity-only lowest among the original methods on every later year'
put('k24.harder', p1(inc(ft('kelmarsh2018', OURS), ft('kelmarsh2024', OURS)))); put('k24.fleet.harder', p1(inc(ft('kelmarsh2018', FLEET), ft('kelmarsh2024', FLEET))))
TL = load(PB/'TURBINE_LEVEL.json')   # turbine as the unit (dependence-aware companion of S1)
for s, ds in DS + DSL:
    w = min(TL[ds].values(), key=lambda v: v['wins']); put(f'tl.{s}.minwins', f"{w['wins']}/{w['n']}"); put(f'tl.{s}.p', ptex(max(v['p_sign'] for v in TL[ds].values())))
assert [ds for _, ds in DS + DSL if min(v['wins'] for v in TL[ds].values()) == TL[ds][KIM]['n']] == ['kelmarsh2018', 'penmanshiel2019', 'penmanshiel2022', 'hot2025']
assert min(TL['hot2019'].values(), key=lambda v: v['wins']) is TL['hot2019'][KIM] and min(TL['kelmarsh2024'].values(), key=lambda v: v['wins']) is TL['kelmarsh2024'][KIM]
DST = load(PB/'DATASET_STATS.json'); VF = load(PB/'VERIFY_FINAL_TABLE.json'); assert VF['pass']; put('verify.n', str(VF['entries']))
cm_old = min(DST[d]['common_min_turbine_pct'] for d in ['kelmarsh2017', 'kelmarsh2018', 'penmanshiel2019', 'hot2019'])
cm_new = min(DST[d]['common_min_turbine_pct'] for _, d in DSL)
assert cm_old >= 99.5; put('common.old.min', f'{math.floor(10 * cm_old) / 10:.1f}'); put('common.new.min', f'{math.floor(10 * cm_new) / 10:.1f}')
# standstill gate (Amendment 14) and diagnostics
GT = load(PB/'GATE.json'); SEL = GT['selection']; SD = load(PB/'STANDSTILL_DIAG.json')
put('gate.h', str(SEL['chosen']['h'] // 6)); put('gate.form', SEL['chosen']['form']); assert SEL['chosen']['form'] == 'hold' and SEL['gate_kept']
put('gate.k17.before', f4(SEL['ours_kelmarsh2017'])); put('gate.k17.after', f4(SEL['chosen']['mae']))
ALL7 = [('k17', 'kelmarsh2017')] + DS + DSL
for s, ds in ALL7:
    g = GT[ds]['mae_gate_common']; o_ = ft(ds, OURS); put(f'gate.{s}', f3(g)); put(f'gate.{s}.pct', p1(red(o_, g)))
    put(f'gate.{s}.share', f"{100 * SD[ds]['gated_share']:.1f}")
    assert g < o_, ds
for s, ds in DS + DSL: assert all(r['gate_better'] for r in GT[ds]['vs'] if r['baseline'] != OURS), ds   # gated estimator better than every baseline
kg = [r for r in GT['kelmarsh2024']['vs'] if r['baseline'] == KIM][0]
put('gate.k24.kim.lo', f3(kg['ci95'][0])); put('gate.k24.kim.hi', f3(kg['ci95'][1])); put('gate.k24.kim.wins', f"{kg['wins']}/{kg['n_pairs']}")
st_o = [SD[d]['stop']['ours'] for _, d in ALL7]; st_g = [SD[d]['stop']['gate'] for _, d in ALL7]; st_k = [SD[d]['stop']['kim'] for _, d in ALL7]
put('ss.ours.min', f'{min(st_o):.2f}'); put('ss.ours.max', f'{max(st_o):.2f}'); put('ss.gate.min', f'{min(st_g):.2f}'); put('ss.gate.max', f'{max(st_g):.2f}')
put('ss.kim.min', f'{min(st_k):.2f}'); put('ss.kim.max', f'{max(st_k):.2f}')
put('ss.share.old.min', f"{100 * min(SD[d]['gated_share'] for _, d in [('k17', 'kelmarsh2017')] + DS):.1f}"); put('ss.share.old.max', f"{100 * max(SD[d]['gated_share'] for _, d in [('k17', 'kelmarsh2017')] + DS):.1f}")
put('ss.share.new.min', f"{100 * min(SD[d]['gated_share'] for _, d in DSL):.1f}"); put('ss.share.new.max', f"{100 * max(SD[d]['gated_share'] for _, d in DSL):.1f}")
assert all(SD[d]['rest']['ours'] < SD[d]['rest']['kim'] for _, d in ALL7), 'text: outside the standstills the proposed beats Kim everywhere'
assert all(abs(SD[d]['rest']['gate'] - SD[d]['rest']['ours']) < 1e-12 for _, d in ALL7)
assert ft('kelmarsh2024', OURS) - GT['kelmarsh2024']['mae_gate_common'] > 0.5 * (ft('kelmarsh2024', OURS) - ft('kelmarsh2018', OURS)), 'text: most of the K24 increase is standstills'
CS = SD['case_k24_t4']; put('case.days', p1(CS['stop_days'])); put('case.oilmin', p1(CS['oil_min']))
for m in ['fleet', 'ours', 'gate', 'kim']: put(f'case.{m}', f'{CS["feb_mae"][m]:.2f}')
# Cao et al. (Amendment 12), on the turbines where it was run
CT = load(PB/'CAO_TABLE.json')
for s, ds in [('k18', 'kelmarsh2018'), ('p19', 'penmanshiel2019')]:
    put(f'cao.{s}.ours', f3(CT[ds]['ours_mae_common']))
    for v in ['v1', 'v2']:
        r = [x for x in CT[ds]['vs'] if x['method'] == f'B12 Cao 2026 [{v}]'][0]; assert r['ours_better']
        put(f'cao.{s}.{v}', f3(r['mae_common'])); put(f'cao.{s}.{v}.wins', f"{r['wins']}/{r['n_pairs']}"); put(f'cao.{s}.{v}.p', ptex(r['p_holm']))
    put(f'cao.{s}.turbines', ', '.join(str(t) for t in CT[ds]['turbines']))
HIST = ['B2 Schlechtingen 2011 AR [lag 3]', 'B6 Kim 2013 [lag 3]', 'B1 Bangalore 2015 [lag 3]', 'B3 Liu 2020 [lag 3]', WANG]   # methods with target history
for ds in CT:   # text: Cao is the least accurate history method in both configurations; p of the text is the largest of the four Cao tests
    mc = {x['method']: x['mae_common'] for x in CT[ds]['vs']}; assert min(mc['B12 Cao 2026 [v1]'], mc['B12 Cao 2026 [v2]']) > max(mc[k] for k in HIST), ds
assert max(x['p_holm'] for ds in CT for x in CT[ds]['vs'] if 'Cao' in x['method']) == [x for x in CT['kelmarsh2018']['vs'] if x['method'] == 'B12 Cao 2026 [v2]'][0]['p_holm']
c2 = [x for x in CT['kelmarsh2018']['vs'] if x['method'] == 'B12 Cao 2026 [v2]'][0]['cao_months']; put('cao.k18.v2.first', f'{c2[0]:.2f}'); put('cao.k18.v2.last', f'{c2[-1]:.2f}')
# computing cost (S5)
TM = load(PB/'S5_TIMING.json'); sec = lambda k: TM[k]['seconds']
put('s5.ours.month', p1(sec('ours_fleet') + sec('ours_local'))); put('s5.fleet', p1(sec('ours_fleet'))); put('s5.local.month', p1(sec('ours_local')))
put('s5.local.year', p1(sec('ours_local_year'))); put('s5.wang.year', p1(sec('B10f'))); put('s5.kim', p1(sec('B6'))); put('s5.bang', p1(sec('B1')))
put('s5.cao.full', f"{sec('B12_v2_full100') / 60:.0f}"); put('s5.cao.extrap.err', f"{100 * (sec('B12_v2') / sec('B12_v2_full100') - 1):.0f}")
for k, n in [('B4', 'fu'), ('B7', 'yan'), ('B8', 'dai'), ('B9', 'grat')]: put(f's5.{n}', p1(sec(k)))
put('s5.classic.max', p1(max(sec(k) for k in ['B0', 'B1', 'B2', 'B2f', 'B3', 'B5', 'B6', 'B11'])))
put('s5.cao.ratio', f"{sec('B12_v2_full100') / (sec('ours_fleet') + sec('ours_local')):.0f}")
put('s5.rec.year', f"{sec('rec_year'):.2f}")

# ---------------------------------------------------------------- revision of 2026-10-17: proposed = similarity + recency
# PUBLISHED_BASELINES_v001 Amendments 17-19 and 05_Protocols/CONFIRM_TWO_TIMESCALE_v001.md
PROP, SIMK, RECK, RECG = 'proposed (similarity + recency)', OURS, 'ours+rec', 'ours+rec+gate'
EWMAK, M20K = 'reference: EWMA of residuals', 'reference: mean of 20 newest residuals'
DSC = [('k20', 'kelmarsh2020'), ('p20', 'penmanshiel2020'), ('h22', 'hot2022')]; DS9 = DS + DSL + DSC
S9 = load(PB/'REC_VS_BASELINES_all9.json'); RRF = load(PB/'RESIDUAL_REFERENCES.json'); ROS = load(PB/'RECENCY_ON_SIMILARITY.json')
CF = load(PB/'CONFIRM_TWO_TIMESCALE.json'); BB = load(PB/'BLOCK_BOOTSTRAP.json')
put('rr.h', str(RRF['ewma_half_life_rows'] // 6)); put('rr.c', f"{RRF['ewma_c']:.2f}")
assert ROS['h'] == 18 and ROS['c'] == 0.5 and RRF['ewma_half_life_rows'] == 18
k17p = load(PB/'DESIGN_LADDER_L6_kelmarsh2017.json')['L6 = L5 + recency term (two time scales)']
put('lad.L6', f3(k17p['all'])); put('lad.rec', p1(red(lad['L5']['all'], k17p['all']))); put('lad.total6', p1(red(lad['L1']['all'], k17p['all'])))
put('lad.rec.tr', p1(red(lad['L5']['transient'], k17p['transient']))); put('lad.rec.calm', p1(red(lad['L5']['calm'], k17p['calm'])))
put('lad.L6.tr', f3(k17p['transient'])); put('lad.total6.tr', p1(red(lad['L1']['transient'], k17p['transient'])))
put('lad.rec.nmonths', str(sum(b < a for a, b in zip(lad['L5']['months'], k17p['months']))))
nwin, prw, recs, recabs, vsw, tm_tot, CLOSE = 0, [], [], [], [], 0, {}
for s, ds in DS9:
    p_, so, f_, w_ = ft(ds, PROP), ft(ds, SIMK), ft(ds, FLEET), ft(ds, WANG)
    put(f'pr.{s}', f3(p_)); put(f'pr.{s}.own', f3(ft(ds, PROP, 'mae_own_rows'))); put(f'pr.{s}.sim', f3(so)); put(f'pr.{s}.fleet', f3(f_))
    put(f'pr.{s}.vswang', p1(red(w_, p_))); put(f'pr.{s}.loc', p1(red(f_, p_))); put(f'pr.{s}.rec', p1(red(so, p_))); put(f'pr.{s}.recabs', f3(so - p_))
    put(f'pr.{s}.ewma', f3(ft(ds, EWMAK))); put(f'pr.{s}.m20', f3(ft(ds, M20K))); put(f'pr.{s}.gate', f3(ft(ds, 'proposed + standstill gate')))
    put(f'pr.{s}.kal', signed(-red(f_, w_)))
    assert p_ == min(x['mae_common_rows'] for x in FT[ds] if x['method'] != 'proposed + standstill gate'), ds   # proposed lowest of all but its gated form
    rows = S9[ds][RECK]['vs']; base = [r for r in rows if r['baseline'] != SIMK]; so_row = [r for r in rows if r['baseline'] == SIMK][0]
    assert all(r['better'] and r['better_block7'] and r['block28d_ci95'][0] > 0 for r in rows), ds
    put(f'pr.{s}.nbase', str(len(base))); put(f'pr.{s}.pmax', ptex(max(r['p_holm'] for r in base))); put(f'pr.{s}.vssim.p', ptex(so_row['p_holm'])); put(f'pr.{s}.n', str(so_row['n_pairs']))
    put(f'pr.{s}.vssim.wins', f"{so_row['wins']}/{so_row['n_pairs']}"); put(f'pr.{s}.vssim.lo', f4(so_row['ci95'][0])); put(f'pr.{s}.vssim.hi', f4(so_row['ci95'][1]))
    wr = [r for r in base if r['baseline'] == WANG][0]; put(f'pr.{s}.wang.wins', f"{wr['wins']}/{wr['n_pairs']}"); put(f'pr.{s}.wang.diff', f3(wr['diff']))
    put(f'pr.{s}.tlmin', f"{min(r['turbines_lower'] for r in base)}/{base[0]['n_turbines']}")
    nwin += 1; prw.append(max(r['p_holm'] for r in base)); recs.append(red(so, p_)); recabs.append(so - p_); vsw.append(red(w_, p_)); tm_tot += so_row['n_pairs']
    closest = min(base, key=lambda r: r['diff']); CLOSE[s] = closest['baseline']
    put(f'pr.{s}.close.diff', f3(closest['diff'])); put(f'pr.{s}.close.lo', f3(closest['ci95'][0])); put(f'pr.{s}.close.hi', f3(closest['ci95'][1]))
assert [s for s in CLOSE if CLOSE[s] != WANG] == ['k24'] and CLOSE['k24'] == KIM, CLOSE   # text: Wang closest everywhere except Kim on Kelmarsh 2024
kal9 = [-red(ft(ds, FLEET), ft(ds, WANG)) for _, ds in DS9]; loc9 = [red(ft(ds, FLEET), ft(ds, PROP)) for _, ds in DS9]
put('pr.kal.min', signed(min(kal9))); put('pr.kal.max', signed(max(kal9))); put('pr.loc.min', p1(min(loc9))); put('pr.loc.max', p1(max(loc9)))
for s, ds in DS9:
    pubx = [(ft(ds, k), k) for _, k in PUB if ft(ds, k) is not None and k != WANG and 'online' not in k]; m, k = min(pubx)
    put(f'pr.{s}.best.pct', p1(inc(ft(ds, PROP), m))); put(f'pr.{s}.best.name', SHORT[k])
assert [V[f'pr.{s}.best.name'] for s, _ in DS] == ['Yan et al.', 'Yan et al.', 'Kim et al.']
exc = sorted((s, r['baseline'], r['turbines_lower'], r['n_turbines']) for s, ds in DS9 for r in S9[ds][RECK]['vs'] if r['baseline'] != SIMK and r['turbines_lower'] < r['n_turbines'])
assert exc == [('h22', KIM, 20, 21), ('k24', KIM, 5, 6)], exc   # text: every turbine against every published method except Kim on K24 and H22
for _, ds in DSC:   # text (Discussion): fleet alone beats every independent published method on the confirmation years too
    assert all(ft(ds, FLEET) < ft(ds, k) for _, k in PUB if ft(ds, k) is not None and k not in (WANG, 'B10 Wang 2022 (online, as published)')), ds
PTC = load(PB/'PER_TURBINE_CONTROL.json')   # Amendment 21: matched-input per-turbine control
pc = {s: PTC[ds][0] for s, ds in DS9}; pf = {s: PTC[ds][1] for s, ds in DS9}
assert [s for s in pc if not (pc[s]['pooled_better'] and pc[s]['block7d_ci95'][0] > 0)] == ['p19'], 'text: pooling shown on all but Penmanshiel 2019'
assert [s for s in pf if not pf[s]['pooled_better']] == ['p19']
okp = [pc[s]['pct'] for s in pc if s != 'p19']; put('pt.min', p1(min(okp))); put('pt.max', p1(max(okp)))
put('pt.p19', p1(pc['p19']['pct'])); put('pt.p19.lo', f4(pc['p19']['ci95'][0])); put('pt.p19.hi', f4(pc['p19']['ci95'][1])); put('pt.p19.wins', f"{pc['p19']['wins']}/{pc['p19']['n_pairs']}")
put('ptf.min', p1(min(pf[s]['pct'] for s in pf))); put('ptf.max', p1(max(pf[s]['pct'] for s in pf)))
put('pt.k17', p1(PTC['kelmarsh2017'][0]['pct']))
ptw = [s for s, ds in DS9 if not ft(ds, 'per-turbine model + late-truth correction') < min(ft(ds, k) for _, k in PUB if ft(ds, k) is not None and k not in (WANG, 'B10 Wang 2022 (online, as published)'))]
assert ptw == ['h22'], ptw   # text: per-turbine control beats every independent published method except on Hill of Towie 2022
put('pr.nyears', str(nwin)); put('pr.pmax', ptex(max(prw))); put('pr.rec.min', p1(min(recs))); put('pr.rec.max', p1(max(recs)))
put('pr.recabs.min', f3(min(recabs))); put('pr.recabs.max', f3(max(recabs))); put('pr.vswang.min', p1(min(vsw))); put('pr.vswang.max', p1(max(vsw)))
put('pr.tm.total', f'{tm_tot:,}'.replace(',', '\\,'))
tl_all = [s for s, ds in DS9 if all(r['turbines_lower'] == r['n_turbines'] for r in S9[ds][RECK]['vs'])]
put('pr.tl.alln', str(len(tl_all)))
tl_ex = {s: [(SHORT.get(r['baseline'], r['baseline']), r['turbines_lower'], r['n_turbines']) for r in S9[ds][RECK]['vs'] if r['turbines_lower'] < r['n_turbines']] for s, ds in DS9}
print('turbine-level exceptions (proposed):', {s: v for s, v in tl_ex.items() if v})
# similarity-only estimator: the registered primary of the first six farm-years (kept for the record)
put('sim.notbetter', '; '.join(f"{s}: {r['baseline']}" for s, ds in DS9 for r in S9[ds][SIMK]['vs'] if not r['better']))
assert [(ds, r['baseline']) for s, ds in DS9 for r in S9[ds][SIMK]['vs'] if not r['better']] == [('kelmarsh2024', KIM)]
assert [(ds, r['baseline']) for s, ds in DS9 for r in S9[ds][SIMK]['vs'] if not r['better_block7']] == [('hot2019', KIM), ('kelmarsh2024', KIM)]
# references (Amendment 17) against the similarity-only estimator, Table-7 years
rr_lower = [s for s, ds in DS + DSL if [r for r in RRF[ds]['refs'] if r['method'] == 'ref ewma'][0]['mae_diff'] < 0]
assert rr_lower == ['h19', 'k24', 'h25'], rr_lower
r25 = [r for r in RRF['hot2025']['refs'] if r['method'] == 'ref ewma'][0]; put('rr.h25.lo', f4(-r25['ci95'][1])); put('rr.h25.hi', f4(-r25['ci95'][0]))
put('rr.h25.diff', f4(-r25['mae_diff'])); put('rr.k24.diff', f3(-[r for r in RRF['kelmarsh2024']['refs'] if r['method'] == 'ref ewma'][0]['mae_diff']))
share = [RRF[ds]['share_of_gain']['ref mean20'] for _, ds in DS + DSL]; put('rr.m20share.min', f'{100 * min(share):.0f}'); put('rr.m20share.max', f'{100 * max(share):.0f}')
# proposed against the two references, nine farm-years (Amendment 18 on six, confirmation on three)
pe = []
for s, ds in DS + DSL:
    c = ROS[ds]['comparisons']['rec_vs_ewma']; pe.append((c['diff'], c['ci95'][0], int(c['wins'].split('/')[0]), int(c['wins'].split('/')[1])))
    c2 = ROS[ds]['comparisons']['rec_vs_ours']; assert c2['ci95'][0] > 0 and c['ci95'][0] > 0, ds
for s, ds in DSC:
    c = CF[ds]['rec_vs']['ewma']; pe.append((c['diff'], c['ci95'][0], c['wins'], c['n_pairs']))
put('pe.diff.min', f4(min(x[0] for x in pe))); put('pe.diff.max', f4(max(x[0] for x in pe))); assert all(x[1] > 0 for x in pe)
put('pe.wins', f"{sum(x[2] for x in pe)}/{sum(x[3] for x in pe)}")
# confirmation test (CONFIRM_TWO_TIMESCALE_v001): primary hypothesis on three untouched farm-years
assert CF['n_confirmed'] == 3 and CF['datasets'] == [ds for _, ds in DSC]
for s, ds in DSC:
    for b in ['ours', 'ewma']:
        c = CF[ds]['rec_vs'][b]; put(f'cf.{s}.{b}.diff', f4(c['diff'])); put(f'cf.{s}.{b}.lo', f4(c['ci95'][0])); put(f'cf.{s}.{b}.hi', f4(c['ci95'][1]))
        put(f'cf.{s}.{b}.wins', f"{c['wins']}/{c['n_pairs']}"); put(f'cf.{s}.{b}.p', ptex(c['p_holm'])); assert c['block7d_ci95'][0] > 0 and c['block28d_ci95'][0] > 0
put('cf.pmax', ptex(max(CF[ds]['rec_vs'][b]['p_holm'] for _, ds in DSC for b in ['ours', 'ewma'])))
put('cf.gate.min', p1(min(red(CF[ds]['mae']['rec'], CF[ds]['mae']['rec+gate']) for _, ds in DSC)))
put('cf.gate.max', p1(max(red(CF[ds]['mae']['rec'], CF[ds]['mae']['rec+gate']) for _, ds in DSC)))
# release delay with the recency term (same rows as the delay study)
DR = load(PB/'DELAY_SENSITIVITY_rec.json')['proposed']
for d in ['1', '3', '6', '12', '24']: put(f'dl.prop.{d}', f3(DR[d]['all'])); put(f'dl.rec.{d}', signed(-red(DL['ours'][d]['all'], DR[d]['all'])))
assert [DR[d]['all'] < DL['ours'][d]['all'] for d in ['1', '3', '6', '12', '24']] == [True, True, True, True, False], 'text: the recency term helps up to 2 h, not at 4 h'
for d in ['1', '3', '6', '12', '24']: assert DR[d]['all'] < min(DL[k][d]['all'] for k in NM if k != 'ours') and DR[d]['all'] < yan, d
assert abs(DR['3']['all'] - ft('kelmarsh2018', PROP, 'mae_own_rows')) < 5e-4
put('dl.prop.vswang.24', p1(red(DL['Wang 2022 (frozen NN)']['24']['all'], DR['24']['all'])))
put('c3.prop.rel', p1(inc(DR['1']['all'], ft('kelmarsh2018', PROP, 'mae_own_rows')))); put('c3.prop.abs', f3(ft('kelmarsh2018', PROP, 'mae_own_rows') - DR['1']['all']))
assert inc(DR['1']['all'], ft('kelmarsh2018', PROP, 'mae_own_rows')) > min(r for r, _ in c3), 'text: C3 still fails in relative terms'
put('k18.sync.kim.pct.prop', p1(inc(ft('kelmarsh2018', PROP), ft('kelmarsh2018', 'B6 Kim 2013 [lag 1]'))))
# standstills (STANDSTILL_DIAG with the proposed estimator)
ALL10 = [('k17', 'kelmarsh2017')] + DS9
st_r = [SD[d]['stop']['rec'] for _, d in ALL10]; put('ss.rec.min', f'{min(st_r):.2f}'); put('ss.rec.max', f'{max(st_r):.2f}')
put('ss.gate9.min', f"{min(SD[d]['stop']['recg'] for _, d in ALL10):.2f}"); put('ss.gate9.max', f"{max(SD[d]['stop']['recg'] for _, d in ALL10):.2f}")
put('ss.kim9.min', f"{min(SD[d]['stop']['kim'] for _, d in ALL10):.2f}"); put('ss.kim9.max', f"{max(SD[d]['stop']['kim'] for _, d in ALL10):.2f}")
put('ss.share.cf.min', f"{100 * min(SD[d]['gated_share'] for _, d in DSC):.1f}"); put('ss.share.cf.max', f"{100 * max(SD[d]['gated_share'] for _, d in DSC):.1f}")
assert all(SD[d]['stop']['rec'] < SD[d]['stop']['ours'] for _, d in ALL10) and all(SD[d]['rest']['rec'] <= SD[d]['rest']['ours'] + 5e-4 for _, d in ALL10)
put('case.rec', f"{CS['feb_mae']['rec']:.2f}"); put('case.recg', f"{CS['feb_mae']['recg']:.2f}")
for s, ds in DS9: put(f'pr.{s}.gatepct', p1(red(ft(ds, PROP), ft(ds, 'proposed + standstill gate'))))
gp = [red(ft(ds, PROP), ft(ds, 'proposed + standstill gate')) for _, ds in DS9]; put('pr.gate.min', p1(min(gp))); put('pr.gate.max', p1(max(gp))); assert min(gp) > 0
# Cao et al. against the proposed estimator on the same turbines and rows
CTP = load(PB/'CAO_TABLE_PROPOSED.json')
for s, ds in [('k18', 'kelmarsh2018'), ('p19', 'penmanshiel2019')]:
    put(f'cao.{s}.prop', f3(CTP[ds]['ours_mae_common']))
    for v in ['v1', 'v2']:
        r = [x for x in CTP[ds]['vs'] if x['method'] == f'B12 Cao 2026 [{v}]'][0]; assert r['ours_better']; put(f'cao.{s}.{v}.wins.prop', f"{r['wins']}/{r['n_pairs']}")
put('cao.pmax.prop', ptex(max(x['p_holm'] for ds in CTP for x in CTP[ds]['vs'] if 'Cao' in x['method'])))
# block bootstrap sensitivity for the similarity-only estimator (Amendment 17 b)
put('bb.nblocks7', str(BB['kelmarsh2018']['7d']['n_blocks'])); put('bb.nblocks28', str(BB['kelmarsh2018']['28d']['n_blocks']))
VF2 = load(PB/'VERIFY_FINAL_TABLE.json'); assert VF2['pass'] and VF2['recency_term_max_abs_diff'] == 0.0
cm9 = min(DST[d]['common_min_turbine_pct'] for _, d in DSC); put('common.cf.min', f'{math.floor(10 * cm9) / 10:.1f}')

# ---------------------------------------------------------------- Table: pre-registered rules
def cell(txt, ok): return f'{txt} ({ok})' if txt else ok


k18f, p19f, h19f = fresh['k18'][1], fresh['p19'][1], fresh['h19'][1]
rows = [
    ('R1', 'Bearing channels', 'adding them lowers the MAE by at least 3\\,\\%',
     [cell(f"{RU['R1'][s]:.1f}\\,\\%", 'held') for s in ['k18', 'p19', 'h19']]),
    ('R2', 'Similarity term', 'it lowers the MAE overall and in at least 8 of the 11 windows',
     [cell(f"{RU['R2'][s][0]:.1f}\\,\\%, {RU['R2'][s][1]}/11", 'held' if RU['R2'][s][0] > 0 and RU['R2'][s][1] >= 8 else 'failed') for s in ['k18', 'p19', 'h19']]),
    ('R3', 'Cooler term (\\ref{eq:cooler})', 'adding it lowers the transient MAE by at least 3\\,\\% without raising the calm MAE by more than 1\\,\\%',
     [cell(f"{RU['R3'][s][0]:.1f}\\,\\%", 'held' if RU['R3'][s][0] >= 3 and RU['R3'][s][1] <= 1 else 'failed') for s in ['k18', 'p19']] + ['n.a.']),
    ('R4', 'Fresh carriers$^{a}$', 'each named carrier raises the MAE by more than 10\\,\\% when late, every other channel by less than 3\\,\\%, '
     'and all carriers late together by more than the sum of their single increases',
     [cell(f"{k18f['oil inlet']:.1f} / {k18f['rear axle bearing']:.1f}\\,\\%", 'held'),
      cell(f"{p19f['oil inlet']:.1f} / {p19f['rear axle bearing']:.1f}\\,\\%", 'partly$^{b}$'),
      cell(f"{h19f['IMS bearing, rotor side']:.1f}\\,\\%", 'held')]),
    ('R5', 'Send policy', 'sending the bearing message without delay gives a lower MAE than sending the inlet message',
     [cell(f"{D18['policy:always_bearing']['all']:.3f} vs {D18['policy:always_inlet']['all']:.3f}", 'held'),
      cell(f"{R19['policy:always_bearing']['all']:.3f} vs {R19['policy:always_inlet']['all']:.3f}", 'held'), 'n.a.']),
    ('R6', 'Oil pressures', 'with all carriers late, adding the oil pressures lowers the MAE by at least 2\\,\\%',
     [cell(f"{red(D18['stale:inlet+bearings']['all'], D18['stale:inlet+bearings+pressure']['all']):.1f}\\,\\%", 'held'), 'not registered',
      cell(f"{red(st, pr):.1f}\\,\\%", 'held')]),
    ('R7', 'Pump stage, filter', 'adding the pump stage and the filter pressures to the R6 inputs lowers the MAE by at least 2\\,\\%',
     ['n.a.', 'n.a.', cell(f"{red(pr, r7):.1f}\\,\\%", 'held')]),
]
assert RU['R3']['k18'][0] < 3 and RU['R3']['p19'][0] >= 3 and RU['R3']['p19'][1] <= 1
assert p19f['rear axle bearing'] < 10 and p19f['front axle bearing'] > 3 and fresh['p19'][2] > sum(p19f[n] for n in CARRIERS['p19'])
assert max(fresh['h19'][1], key=fresh['h19'][1].get) == 'IMS bearing, rotor side' and fresh['h19'][2] > sum(h19f[n] for n in CARRIERS['h19'])
assert max(v for n, v in h19f.items() if n in ['power', 'wind speed', 'rotor speed', 'generator speed', 'grid current']) < 3
held = sum(c.endswith('(held)') for _, _, _, cs in rows for c in cs); failed = sum(c.endswith('(failed)') for _, _, _, cs in rows for c in cs)
part = sum('partly' in c for _, _, _, cs in rows for c in cs); appl = held + failed + part
put('rules.appl', str(appl)); put('rules.held', str(held)); put('rules.failed', str(failed)); put('rules.part', str(part))
T = ['\\begin{table*}[t]', '\\caption{Pre-Registered Rules on the Three Held-Out Farm-Years}', '\\label{tab:rules}', '\\centering', '\\footnotesize',
     '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{@{}llp{6.4cm}lll@{}}', '\\toprule',
     'Rule & Component & Success rule & Kelmarsh 2018 & Penmanshiel 2019 & Hill of Towie 2019 \\\\', '\\midrule']
T += [f'{a} & {b} & {c} & ' + ' & '.join(cs) + ' \\\\' for a, b, c, cs in rows]
T += ['\\bottomrule',
      '\\multicolumn{6}{@{}p{\\textwidth}@{}}{\\footnotesize Percentages are reductions of the MAE, except under R4, where they are the increases '
      'caused by delaying a carrier by $d$ steps. On Hill of Towie, which records no inlet temperature, R1 and R2 use the drivers and the delayed state '
      'as the base feature set, and the evaluation starts on 1 February, as registered. $^{a}$Carriers: oil inlet and rear axle bearing (Senvion); a gear-oil-adjacent temperature, tested with the four gearbox '
      'bearings (Siemens). $^{b}$The inlet clause and the joint clause held; the rear bearing lost ' + f"{p19f['rear axle bearing']:.1f}" +
      '\\,\\% and the front bearing ' + f"{p19f['front axle bearing']:.1f}" + '\\,\\%. n.a.: channel not recorded on that fleet; '
      'not registered: rule not part of that protocol.}', '\\end{tabular}', '\\end{table*}']
(PAPER/'tables').mkdir(exist_ok=True); (PAPER/'tables/tab_rules.tex').write_text('\n'.join(T) + '\n', encoding='utf-8')

# ---------------------------------------------------------------- Table: comparison (common rows, d = 3), nine held-out farm-years
NAME = {'kelmarsh2018': 'Kelmarsh 2018', 'penmanshiel2019': 'Penmanshiel 2019', 'hot2019': 'Hill of Towie 2019', 'kelmarsh2024': 'Kelmarsh 2024',
        'penmanshiel2022': 'Penmanshiel 2022', 'hot2025': 'Hill of Towie 2025', 'kelmarsh2020': 'Kelmarsh 2020', 'penmanshiel2020': 'Penmanshiel 2020',
        'hot2022': 'Hill of Towie 2022'}
AB = {'kelmarsh': 'K', 'penmanshiel': 'P', 'hot': 'H'}


def fmt(ds, key, bold=False):
    v = ft(ds, key)
    if v is None: return 'n.r.'
    return f'\\textbf{{{v:.3f}}}' if bold else f'{v:.3f}'


pub = sorted([(lab, k) for lab, k in PUB if ft('kelmarsh2018', k) is not None], key=lambda r: ft('kelmarsh2018', r[1]))
cao = {ds: [x for x in CT[ds]['vs'] if x['method'] == 'B12 Cao 2026 [v2]'][0]['mae_common'] for ds in CT}
hd = ' & '.join(AB[''.join(c for c in ds if c.isalpha())] + ds[-2:] for _, ds in DS9)
T = ['\\begin{table*}[t]', '\\caption{MAE ($^{\\circ}$C) on the Nine Held-Out Farm-Years With a 30-min Release Delay}', '\\label{tab:comparison}',
     '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc@{}}', '\\toprule',
     ' & \\multicolumn{3}{c}{Held-out year and farms} & \\multicolumn{3}{c}{Later years} & \\multicolumn{3}{c}{Confirmation years} \\\\',
     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}', f'Method & {hd} \\\\', '\\midrule',
     'Proposed & ' + ' & '.join(fmt(ds, PROP, True) for _, ds in DS9) + ' \\\\',
     'Proposed + standstill gate$^{a}$ & ' + ' & '.join(fmt(ds, 'proposed + standstill gate') for _, ds in DS9) + ' \\\\',
     'Similarity correction only$^{b}$ & ' + ' & '.join(fmt(ds, SIMK) for _, ds in DS9) + ' \\\\',
     'Fleet model alone & ' + ' & '.join(fmt(ds, FLEET) for _, ds in DS9) + ' \\\\',
     'Per-turbine models$^{f}$ & ' + ' & '.join(fmt(ds, 'per-turbine model + late-truth correction') for _, ds in DS9) + ' \\\\',
     'Reference: EWMA of residuals$^{c}$ & ' + ' & '.join(fmt(ds, EWMAK) for _, ds in DS9) + ' \\\\',
     'Reference: mean of 20 residuals$^{c}$ & ' + ' & '.join(fmt(ds, M20K) for _, ds in DS9) + ' \\\\', '\\midrule']
T += [f'{lab} & ' + ' & '.join(fmt(ds, k) for _, ds in DS9) + ' \\\\' for lab, k in pub]
T += [f"Cao et al.~\\cite{{Cao2026}}$^{{d}}$ & {cao['kelmarsh2018']:.3f} & {cao['penmanshiel2019']:.3f}$^{{e}}$ & " + ' & '.join(['n.r.'] * 7) + ' \\\\']
T += ['\\midrule', 'Persistence, $y_i(t-d)$ & ' + ' & '.join(fmt(ds, 'B0 last released value [lag 3]') for _, ds in DS9) + ' \\\\', '\\bottomrule',
      '\\end{tabular*}', '\\par\\smallskip\\parbox{\\textwidth}{\\footnotesize Lower is better. K: Kelmarsh, P: Penmanshiel, H: Hill of Towie, with the year. '
      'Bold: the proposed estimator. Common rows of all receipt-causal methods of each farm-year except the online Kalman filter and Cao et al., which were '
      'scored separately; published methods are sorted by their Kelmarsh 2018 error, with Cao et al. listed last. $^{a}$Exploratory extension '
      '(Section~\\ref{sec:later}). $^{b}$The estimator registered as primary for the first six farm-years, before the recency term was added '
      '(Section~\\ref{sec:prereg}). $^{c}$Corrections of the fleet model by recent released residuals alone (Section~\\ref{sec:disc-mechanism}). '
      '$^{d}$Second configuration (learning rate $10^{-3}$); the published one is reported in Section~\\ref{sec:compare}. '
      '$^{e}$Turbines ' + V['cao.p19.turbines'] + ' only, with the proposed estimator at ' + V['cao.p19.prop'] + '~$^{\\circ}$C on the same rows. '
      '$^{f}$The proposed estimator with one model per turbine instead of the fleet model, same inputs and late-truth terms '
      '(Section~\\ref{sec:disc-drivers}). n.r.: not run, see Sections~\\ref{sec:baselines} and~\\ref{sec:compare}.}', '\\end{table*}']
(PAPER/'tables/tab_comparison.tex').write_text('\n'.join(T) + '\n', encoding='utf-8')

# ---------------------------------------------------------------- Table: paired tests, proposed vs every baseline, nine farm-years
T = ['\\begin{table*}[t]', '\\caption{Paired Tests of the Proposed Estimator Against the Published Methods Run on Each Farm-Year}', '\\label{tab:significance}', '\\centering',
     '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccc@{}}', '\\toprule',
     ' & & \\multicolumn{3}{c}{Closest published method} & \\multicolumn{2}{c}{All other methods} \\\\', '\\cmidrule(lr){3-5}\\cmidrule(l){6-7}',
     'Farm-year & Closest & $\\Delta$MAE [turbine CI; 7-day CI], $^{\\circ}$C & Wins & $p_{\\mathrm{Holm}}$ & min $\\Delta$MAE & max $p_{\\mathrm{Holm}}$ \\\\', '\\midrule']
for j, (s, ds) in enumerate(DS9):
    rows = [r for r in S9[ds][RECK]['vs'] if r['baseline'] != SIMK and '[v1]' not in r['baseline']]
    c = min(rows, key=lambda r: r['diff']); rest = [r for r in rows if r is not c]
    T += [f"{NAME[ds]} & {SHORT.get(c['baseline'], c['baseline'])} & {c['diff']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}; "
          f"{c['block7d_ci95'][0]:.3f}, {c['block7d_ci95'][1]:.3f}] & {c['wins']}/{c['n_pairs']} & {ptex(c['p_holm'])} & "
          f"{min(r['diff'] for r in rest):.3f} & {ptex(max(r['p_holm'] for r in rest))} \\\\"]
    if j in (2, 5): T += ['\\midrule']
T += ['\\bottomrule', '\\end{tabular*}', '\\par\\smallskip\\parbox{\\textwidth}{\\footnotesize $\\Delta$MAE: method minus proposed, on the common rows, '
      'with the 95\\,\\% intervals of a bootstrap over whole turbines and of a bootstrap over 7-day blocks shared by all turbines. Wins: turbine-months in '
      'which the proposed estimator has the lower MAE. $p_{\\mathrm{Holm}}$: Wilcoxon signed-rank test on turbine-months, Holm-corrected over the '
      'methods of a farm-year (the similarity-only estimator included). Every interval of every method, including 28-day blocks, lies above zero.}', '\\end{table*}']
(PAPER/'tables/tab_significance.tex').write_text('\n'.join(T) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- Table: pre-registered rules on the later years
LY = {'kelmarsh2024': E/'later_years_kelmarsh2024_2026-10-15_v001', 'penmanshiel2022': E/'later_years_penmanshiel2022_2026-10-15_v001/rules',
      'hot2025': E/'later_years_hot2025_2026-10-15_v001'}
K24 = {**res(LY['kelmarsh2024']/'results_A.json'), **res(LY['kelmarsh2024']/'results_B.json'), **res(LY['kelmarsh2024']/'results_C.json'), **res(LY['kelmarsh2024']/'results_D.json')}
P22 = res(LY['penmanshiel2022']/'results.json')
AH25 = res(LY['hot2025']/'results_A.json'); SH25 = {**res(LY['hot2025']/'results_B.json'), **res(LY['hot2025']/'results_C.json')}; HH25 = res(LY['hot2025']/'results_h36.json')
SENP = ['Power', 'Wind speed', 'Rotor speed', 'Generator RPM', 'Nacelle ambient', 'Nacelle temperature', 'Grid current', 'Generator bearing rear',
        'Gear oil inlet', 'Front bearing', 'Rear bearing', 'Rotor bearing', 'Gearbox speed']


def sen_rules(R, three):
    o = {'R1': red(R['h30+local']['all'], R['h30_x+local']['all']), 'R2': (red(R['h30_x']['all'], R['h30_x+local']['all']), months_better(R['h30_x'], R['h30_x+local'])),
         'R3': (red(R['base']['transient'], R['h29']['transient']), inc(R['base']['calm'], R['h29']['calm']))}
    b_ = R['h30_x']['all']; ch = {p: inc(b_, pick(R, 'stale:' + p)['all']) for p in SENP}; jt = inc(b_, R['stale:inlet+bearings']['all'])
    oth = max(v for p, v in ch.items() if p not in ['Gear oil inlet', 'Rear bearing'])   # 'every other single' includes the front bearing
    sm = ch['Gear oil inlet'] + ch['Rear bearing'] + (ch['Front bearing'] if three else 0)
    inl, rear = ch['Gear oil inlet'] > 10, ch['Rear bearing'] > 10
    others_ok = oth < 3
    o['R4'] = (ch, jt, 'held' if inl and rear and others_ok and jt > sm else ('partly' if inl and jt > sm else 'failed'))
    o['R5'] = (R['policy:always_bearing']['all'], R['policy:always_inlet']['all'])
    if 'stale:inlet+bearings+pressure' in R: o['R6'] = red(R['stale:inlet+bearings']['all'], R['stale:inlet+bearings+pressure']['all'])
    return o


rk, rp = sen_rules(K24, True), sen_rules(P22, False)
b25 = AH25['base_x']['all']; ch25 = {n: inc(b25, pick(SH25, 'stale:' + p)['all']) for p, n in SIE.items() if any(k.startswith('stale:' + p) for k in SH25)}
j25 = inc(b25, AH25['stale:all_gearbox_bearings']['all']); top25 = max(ch25, key=ch25.get)
bear4 = ['HS bearing, generator side', 'HS bearing, rotor side', 'IMS bearing, generator side', 'IMS bearing, rotor side']
r4h = top25 in bear4 + ['main bearing'] and max(ch25[n] for n in ['power', 'wind speed', 'rotor speed', 'generator speed', 'grid current']) < 3 and j25 > sum(ch25[n] for n in bear4)
st25, pr25, r725 = AH25['stale:all_gearbox_bearings']['all'], AH25['stale:all_gearbox_bearings+pressure']['all'], HH25['stale:all_gearbox_bearings+pressure+r7']['all']
rh = {'R1': red(AH25['base+local']['all'], AH25['base_x+local']['all']), 'R2': (red(AH25['base_x']['all'], AH25['base_x+local']['all']), months_better(AH25['base_x'], AH25['base_x+local'])),
      'R6': red(st25, pr25), 'R7': red(pr25, r725)}
ok = lambda c: 'held' if c else 'failed'
rowsL = [('R1', [cell(f"{rk['R1']:.1f}\\,\\%", ok(rk['R1'] >= 3)), cell(f"{rp['R1']:.1f}\\,\\%", ok(rp['R1'] >= 3)), cell(f"{rh['R1']:.1f}\\,\\%", ok(rh['R1'] >= 3))]),
         ('R2', [cell(f"{r['R2'][0]:.1f}\\,\\%, {r['R2'][1]}/11", ok(r['R2'][0] > 0 and r['R2'][1] >= 8)) for r in (rk, rp, rh)]),
         ('R3', [cell(f"{r['R3'][0]:.1f}\\,\\%", ok(r['R3'][0] >= 3 and r['R3'][1] <= 1)) for r in (rk, rp)] + ['n.a.']),
         ('R4', [cell(f"{rk['R4'][0]['Gear oil inlet']:.1f} / {rk['R4'][0]['Rear bearing']:.1f}\\,\\%", rk['R4'][2] + ('$^{a}$' if rk['R4'][2] == 'partly' else '')),
                 cell(f"{rp['R4'][0]['Gear oil inlet']:.1f} / {rp['R4'][0]['Rear bearing']:.1f}\\,\\%", rp['R4'][2] + ('$^{a}$' if rp['R4'][2] == 'partly' else '')),
                 cell(f"{ch25['IMS bearing, rotor side']:.1f}\\,\\%", ok(r4h))]),
         ('R5', [cell(f"{r['R5'][0]:.3f} vs {r['R5'][1]:.3f}", ok(r['R5'][0] < r['R5'][1])) for r in (rk, rp)] + ['n.a.']),
         ('R6', [cell(f"{rk['R6']:.1f}\\,\\%", ok(rk['R6'] >= 2)), 'not registered', cell(f"{rh['R6']:.1f}\\,\\%", ok(rh['R6'] >= 2))]),
         ('R7', ['n.a.', 'n.a.', cell(f"{rh['R7']:.1f}\\,\\%", ok(rh['R7'] >= 2))])]
assert top25 == 'IMS bearing, rotor side'
cl = [c for _, cs in rowsL for c in cs]
hL, fL, pL = sum(c.endswith('(held)') for c in cl), sum(c.endswith('(failed)') for c in cl), sum('partly' in c for c in cl)
put('rulesL.appl', str(hL + fL + pL)); put('rulesL.held', str(hL)); put('rulesL.failed', str(fL)); put('rulesL.part', str(pL))
for s_, r in [('k24', rk), ('p22', rp), ('h25', rh)]:
    put(f'rL.{s_}.r1', p1(r['R1'])); put(f'rL.{s_}.r2', p1(r['R2'][0])); put(f'rL.{s_}.r2.m', str(r['R2'][1]))
for s_, r in [('k24', rk), ('p22', rp)]:
    put(f'rL.{s_}.r3', p1(r['R3'][0])); put(f'rL.{s_}.inlet', p1(r['R4'][0]['Gear oil inlet'])); put(f'rL.{s_}.rear', p1(r['R4'][0]['Rear bearing']))
    put(f'rL.{s_}.front', p1(r['R4'][0]['Front bearing'])); put(f'rL.{s_}.joint', f"{r['R4'][1]:.0f}")
r1all = list(RU['R1'].values()) + [rk['R1'], rp['R1'], rh['R1']]; put('r1.all.min', p1(min(r1all))); put('r1.all.max', p1(max(r1all)))
r2all = [v[0] for v in RU['R2'].values()] + [rk['R2'][0], rp['R2'][0], rh['R2'][0]]; put('r2.all.min', f'{min(r2all):.0f}'); put('r2.all.max', f'{max(r2all):.0f}')
for _, ds in DSL:   # text (Discussion): fleet alone beats every independent published method except Kim on Kelmarsh 2024
    ind = {k: ft(ds, k) for _, k in PUB if ft(ds, k) is not None and k not in (WANG, 'B10 Wang 2022 (online, as published)')}
    worse = [k for k, v in ind.items() if v < ft(ds, FLEET)]; assert worse == ([KIM] if ds == 'kelmarsh2024' else []), (ds, worse)
assert sum(1 for r in [RU['R3']['k18'], RU['R3']['p19'], rk['R3'], rp['R3']] if r[0] < 3) == 3, 'text: cooler rule failed on three of four Senvion years'
assert all(r['R2'][1] == 11 for r in (rk, rp, rh)), 'text: local correction helped in all 11 windows of every later year'
put('rL.k24.r6', p1(rk['R6'])); put('rL.h25.r6', p1(rh['R6'])); put('rL.h25.r7', p1(rh['R7'])); put('rL.h25.ims', p1(ch25['IMS bearing, rotor side'])); put('rL.h25.joint', f'{j25:.0f}')
assert rk['R3'][0] < 3 and rp['R3'][0] < 3 and rk['R4'][2] == rp['R4'][2] == 'partly'
T = ['\\begin{table}[t]', '\\caption{Pre-Registered Rules on the Later Years}', '\\label{tab:ruleslater}', '\\centering', '\\footnotesize',
     '\\setlength{\\tabcolsep}{2.4pt}', '\\begin{tabular}{@{}llll@{}}', '\\toprule', 'Rule & Kelmarsh 2024 & Penmanshiel 2022 & Hill of Towie 2025 \\\\', '\\midrule']
T += [f'{a} & ' + ' & '.join(cs) + ' \\\\' for a, cs in rowsL]
T += ['\\bottomrule', '\\multicolumn{4}{@{}p{\\columnwidth}@{}}{\\footnotesize Rules and notation as in Table~\\ref{tab:rules}. $^{a}$The inlet clause and the '
      f"joint clause held; the rear bearing fell short of 10\\,\\%.}}", '\\end{tabular}', '\\end{table}']
(PAPER/'tables/tab_rules_later.tex').write_text('\n'.join(T) + '\n', encoding='utf-8')

# ---------------------------------------------------------------- Table: farm-years used (from DATASET_STATS.json)
COLS = [('kelmarsh2017', 'Kelmarsh 2017', 'development', 'Senvion MM92'), ('kelmarsh2018', 'Kelmarsh 2018', 'held-out year', 'Senvion MM92'),
        ('penmanshiel2019', 'Penmanshiel 2019', 'second farm', 'Senvion MM82'), ('hot2019', 'Hill of Towie 2019', 'third farm', 'Siemens SWT-2.3'),
        ('kelmarsh2024', 'Kelmarsh 2024', 'later year', 'Senvion MM92'), ('penmanshiel2022', 'Penmanshiel 2022', 'later year', 'Senvion MM82'),
        ('hot2025', 'Hill of Towie 2025', 'later year', 'Siemens SWT-2.3'), ('kelmarsh2020', 'Kelmarsh 2020', 'confirmation', 'Senvion MM92'),
        ('penmanshiel2020', 'Penmanshiel 2020', 'confirmation', 'Senvion MM82'), ('hot2022', 'Hill of Towie 2022', 'confirmation', 'Siemens SWT-2.3')]
T = ['\\begin{table*}[t]', '\\caption{Farm-Years Used in This Study}', '\\label{tab:datasets}', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{5pt}',
     '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllcccc@{}}', '\\toprule',
     'Farm-year & Role & Turbine & Turbines & Usable rows (\\%) & Oil mean (sd), $^{\\circ}$C & 30-min change, $^{\\circ}$C \\\\', '\\midrule']
for i, c in enumerate(COLS):
    d = DST[c[0]]
    T += [f"{c[1]} & {c[2]} & {c[3]} & {d['turbines']} & {d['usable_pct']:.1f} & {d['oil_mean']:.1f} ({d['oil_sd']:.1f}) & {d['d30_median']:.2f} / {d['d30_p90']:.2f} \\\\"]
    if i in (0, 3, 6): T += ['\\midrule']
T += ['\\bottomrule', '\\end{tabular*}', '\\par\\smallskip\\parbox{\\textwidth}{\\footnotesize Usable rows: evaluation rows with valid current and lagged target. '
      '30-min change: median / 90th percentile of $\\Delta_i(t)$. 2020 and 2024 are leap years (52\\,704 rows per turbine).}', '\\end{table*}']
assert abs(DST['kelmarsh2017']['usable_pct'] - 99.3) < 0.05 and abs(DST['hot2019']['oil_mean'] - 37.6) < 0.05
(PAPER/'tables/tab_datasets.tex').write_text('\n'.join(T) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- numbers.tex
H = ['% Written by latetruth/make_paper_numbers.py on 2026-10-18 00:21. Do not edit by hand.',
     '% Sources: ' + '; '.join(sorted(set(SRC))), '\\makeatletter',
     '\\newcommand{\\val}[1]{\\ifcsname val@#1\\endcsname\\csname val@#1\\endcsname\\else\\@latex@warning{Undefined value #1}\\textbf{??#1??}\\fi}',
     '\\makeatother']
H += [f'\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}' for k, v in V.items()]
(PAPER/'generated').mkdir(exist_ok=True); (PAPER/'generated/numbers.tex').write_text('\n'.join(H) + '\n', encoding='utf-8', newline='\n')
print(len(V), 'values;', len(set(SRC)), 'source files')
for k, v in V.items(): print(f'{k:28s} {v}')
