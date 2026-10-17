"""Paper figures 3-8 (vector PDF) drawn from the saved result files only; one style for all figures (soft panel backgrounds, rounded
boxes, colour-blind-checked categorical palette, direct labels). usage: python make_paper_figures.py   (needs matplotlib)"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = REPO_DIR; E = REPO_DIR/'results'; PB = E/'published_baselines_2026-10-14_v001'; OUT = PAPER_DIR/'figures'
OUT.mkdir(exist_ok=True)
# categorical slots (validated reference palette, light mode), text and surfaces
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8986', '#e4e3df'
PANEL = ['#eef4fb', '#fdf1ea', '#eaf7f1', '#f3f0fa']        # pastel panel backgrounds
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # fallback fonts are expected off Windows
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
COL, TXT = 3.45, 7.1      # IEEE Access column and text widths (in)


def load(p): return json.loads(p.read_text(encoding='utf-8', errors='replace'))


def panel(fig, ax, color, label=None, pad=(0.055, 0.075, 0.012, 0.03)):
    """Rounded pastel card behind an axes (Aryan's figure style) and an (a)/(b) label in its corner."""
    fig.canvas.draw(); bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    l, b, r, t = pad
    card = FancyBboxPatch((bb.x0 - l * 0.25, bb.y0 - b * 0.15), bb.width + l * 0.25 + r, bb.height + b * 0.15 + t,
                          boxstyle='round,pad=0.004,rounding_size=0.018', transform=fig.transFigure, fc=color, ec='none', zorder=-10)
    fig.patches.append(card)
    if label: fig.text(bb.x0 - l * 0.2 + 0.006, bb.y1 + t - 0.012, label, fontsize=8.5, fontweight='bold', va='top', ha='left')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig(OUT/f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig); print('saved', name)


# ------------------------------------------------------------------ Fig. 3: design ladder (Kelmarsh 2017)
def fig_ladder():
    L = {k.split()[0]: v for k, v in load(PB/'DESIGN_LADDER_kelmarsh2017.json')['results'].items()}
    L['L6'] = load(PB/'DESIGN_LADDER_L6_kelmarsh2017.json')['L6 = L5 + recency term (two time scales)']
    steps = [('L1', 'Per-turbine\nmodels', MUTED), ('L2', '+ Fleet\npooling', VIOLET), ('L3', '+ Cooler,\nload terms', AQUA),
             ('L4', '+ Bearing\nchannels', ORANGE), ('L5', '+ Similarity\nterm', BLUE), ('L6', '+ Recency\nterm', '#174f96')]
    fig, ax = plt.subplots(figsize=(COL, 2.25)); x = np.arange(len(steps))
    for i, (k, lab, c) in enumerate(steps):
        ax.bar(i, L[k]['all'], width=0.62, color=c, zorder=3, linewidth=0)
        ax.text(i, L[k]['all'] + 0.006, f"{L[k]['all']:.3f}", ha='center', va='bottom', fontsize=6.4, color=INK)
    for i in range(1, len(steps)):
        a, b = L[steps[i - 1][0]]['all'], L[steps[i][0]]['all']
        ax.text(i - 0.5, max(a, b) + 0.042, f"$-${100 * (1 - b / a):.1f}%", ha='center', va='bottom', fontsize=6.0, color=INK2,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=GRID, lw=0.5))
    ax.axhline(L['L0']['all'], xmin=0.55, xmax=1.0, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(5.4, L['L0']['all'] + 0.006, f"all inputs, fitted once: {L['L0']['all']:.3f}", ha='right', va='bottom', fontsize=6.0, color=INK2)
    ax.set_xticks(x, [s[1] for s in steps], fontsize=6.0); ax.set_ylabel('MAE (°C)'); ax.set_ylim(0, 0.37); ax.set_xlim(-0.5, 5.5); grid(ax)
    ax.tick_params(axis='x', length=0)
    fig.tight_layout(); panel(fig, ax, PANEL[0]); save(fig, 'fig3_ladder')


# ------------------------------------------------------------------ Fig. 4: per turbine-month scatter
def fig_scatter():
    D = load(PB/'FIGDATA_turbine_month.json')   # 'ours' = proposed estimator (similarity + recency) since the revision of 2026-10-17
    S9 = load(PB/'REC_VS_BASELINES_all9.json'); SG = {ds: [dict(r, wins_turbine_month=r['wins']) for r in S9[ds]['ours+rec']['vs']] for ds in S9 if ds != 'code_sha256'}
    farms = [('kelmarsh2018', 'Kelmarsh 2018', BLUE, 'yan'), ('penmanshiel2019', 'Penmanshiel 2019', ORANGE, 'yan'), ('hot2019', 'Hill of Towie 2019', AQUA, 'kim')]
    key = {'wang': 'B10f Wang 2022 (frozen NN)', 'yan': 'B7 Yan 2023 [v2]', 'kim': 'B6 Kim 2013 [lag 3]'}
    fig, axs = plt.subplots(1, 2, figsize=(TXT, 2.75))
    for j, (ax, which) in enumerate(zip(axs, ['wang', 'best'])):
        lim = [0.08, 0.50] if j == 0 else [0.05, 1.05]
        ax.plot(lim, lim, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=2)
        for ds, name, c, best in farms:
            b = 'wang' if which == 'wang' else best
            xs, ys = np.array(D[ds][b]).ravel(), np.array(D[ds]['ours']).ravel()
            r = [s for s in SG[ds] if s['baseline'] == key[b]][0]
            nm = name if which == 'wang' else f"{name}, vs {'Yan et al.' if b == 'yan' else 'Kim et al.'}"
            ax.scatter(xs, ys, s=11, color=c, edgecolor='white', linewidth=0.4, zorder=3, alpha=0.9,
                       label=f"{nm}: {r['wins_turbine_month']}/{r['n_pairs']}")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel('Kalman filter, Wang et al.: monthly MAE (°C)' if which == 'wang' else 'Most accurate other method: monthly MAE (°C)')
        ax.set_ylabel('Proposed: monthly MAE (°C)')
        ax.text(lim[1] * 0.97, lim[0] + 0.03 * (lim[1] - lim[0]), 'below the line:\nproposed more accurate', ha='right', va='bottom', fontsize=6.4, color=INK2)
        ax.legend(loc='upper left', title='turbine-months won', title_fontsize=6.6, handletextpad=0.2, borderaxespad=0.2)
        grid(ax, 'both')
    fig.tight_layout(w_pad=2.5)
    for ax, c, lab in zip(axs, PANEL[:2], ['(a)', '(b)']): panel(fig, ax, c, lab)
    save(fig, 'fig4_scatter')


# ------------------------------------------------------------------ Fig. 5: delay curves + k x M grid
def fig_delay():
    D = {**load(PB/'DELAY_SENSITIVITY_part1.json'), **load(PB/'DELAY_SENSITIVITY_part2.json'), **load(PB/'DELAY_SENSITIVITY_rec.json')}
    yan = [r for r in load(PB/'FINAL_TABLE_all.json')['kelmarsh2018'] if r['method'] == 'B7 Yan 2023 [v2]'][0]['mae_own_rows']
    G = load(PB/'LOCAL_GRID_kelmarsh2018.json')
    fig, (a, b) = plt.subplots(1, 2, figsize=(TXT, 2.55), gridspec_kw=dict(width_ratios=[1.5, 1]))
    mins = ['10', '30', '60', '120', '240']; xs = np.arange(5)
    ys = [D['ours'][str(d)]['all'] for d in (1, 3, 6, 12, 24)]
    a.plot(xs, ys, color=BLUE, ls=(0, (2, 1.5)), lw=0.9, marker='o', ms=2.6, mfc='white', mec=BLUE, mew=0.7, zorder=3)
    series = [('proposed', 'Proposed', BLUE, '-', 'o'), ('our fleet model (no late truth)', 'Fleet model alone', VIOLET, '-', 's'),
              ('Wang 2022 (frozen NN)', 'Kalman filter, Wang et al.', ORANGE, '--', '^'), ('Kim 2013', 'Kim et al.', AQUA, '-', 'v'),
              ('Bangalore 2015', 'Bangalore and Tjernberg', MAGENTA, '-', 'D')]
    a.axhline(yan, color=MUTED, lw=1.0, ls=(0, (1, 1.5)), zorder=2)
    a.text(1.15, yan - 0.004, 'Yan et al. (no target history)', fontsize=6.2, color=INK2, va='top', ha='left')
    a.text(0.0, 0.113, 'thin dashed blue: proposed without the recency term', fontsize=6.0, color=BLUE, va='bottom', ha='left')
    for k, lab, c, ls, m in series:
        y = [D[k][str(d)]['all'] for d in (1, 3, 6, 12, 24)]
        a.plot(xs, y, color=c, ls=ls, lw=1.6, marker=m, ms=4.2, mfc=c, mec='white', mew=0.6, zorder=3)
        a.text(4.08, y[-1], lab, color=INK, fontsize=6.2, va='center', ha='left')
    # de-overlap the three close end labels (proposed / fleet / Kalman) by nudging the text objects
    t = [x for x in a.texts if x.get_text() in ('Proposed', 'Fleet model alone', 'Kalman filter, Wang et al.')]
    for x, dy in zip(t, (-0.013, 0.0, 0.013)): x.set_y(x.get_position()[1] + dy)
    a.set_xticks(xs, mins); a.set_xlabel('Release delay $d$ (min)'); a.set_ylabel('MAE (°C)')
    a.set_xlim(-0.2, 5.6); a.set_ylim(0.10, 0.47); grid(a)
    ks, Ms = [5, 10, 20, 40, 80], [1440, 4320, 8640, 17280]
    Z = np.array([[[r['mae'] for r in G if r['k'] == k and r['memory'] == M][0] for M in Ms] for k in ks])
    im = b.imshow(Z, cmap='Blues_r', aspect='auto', vmin=Z.min() - 0.0008, vmax=Z.max() + 0.0012)
    for i in range(len(ks)):
        for j in range(len(Ms)):
            b.text(j, i, f'{Z[i, j]:.4f}', ha='center', va='center', fontsize=6.3, color='white' if Z[i, j] < Z.min() + 0.0012 else INK)
    i0, j0 = ks.index(20), Ms.index(8640)
    b.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.6))
    b.set_xticks(range(4), ['10', '30', '60', '120']); b.set_yticks(range(5), [str(k) for k in ks])
    b.set_xlabel('Memory $M$ (days)'); b.set_ylabel('Neighbours $k$'); b.spines[:].set_visible(False); b.tick_params(length=0)
    fig.tight_layout(w_pad=2.2)
    panel(fig, a, PANEL[0], '(a)'); panel(fig, b, PANEL[3], '(b)'); save(fig, 'fig5_delay')


# ------------------------------------------------------------------ Fig. 6: value of freshness per channel
def fig_freshness():
    K18 = E/'kelmarsh_2018_unseen_2026-10-13_v001'; P19 = E/'penmanshiel_replication_2019_2026-10-14_v001'; H19 = E/'hill_of_towie_replication_2019_2026-10-13_v001'
    r = lambda p: load(p).get('results', load(p))
    def pick(R, pre): return [v for k, v in R.items() if k.startswith(pre)][0]['all']
    SEN = [('Gear oil inlet', 'Oil inlet'), ('Rear bearing', 'Rear axle bearing'), ('Front bearing', 'Front axle bearing'), ('Rotor bearing', 'Rotor bearing'),
           ('Gearbox speed', 'Gearbox speed'), ('Generator bearing rear', 'Generator bearing'), ('Nacelle temperature', 'Nacelle temp.'),
           ('Nacelle ambient', 'Ambient temp.'), ('Power', 'Power'), ('Wind speed', 'Wind speed'), ('Rotor speed', 'Rotor speed'),
           ('Generator RPM', 'Generator speed'), ('Grid current', 'Grid current')]
    SIE = [('wtc_IMSRotTm', 'IMS bearing, rotor side'), ('wtc_IMSGenTm', 'IMS bearing, gen. side'), ('wtc_HSGenTmp', 'HS bearing, gen. side'),
           ('wtc_HSRotTmp', 'HS bearing, rotor side'), ('wtc_MainBTmp', 'Main bearing'), ('wtc_GenBeGTm', 'Generator bearing'),
           ('wtc_NacelTmp', 'Nacelle temp.'), ('wtc_AmbieTmp', 'Ambient temp.'), ('wtc_ActPower', 'Power'), ('wtc_AcWindSp', 'Wind speed'),
           ('wtc_MainSRpm', 'Rotor speed'), ('wtc_GenRpm', 'Generator speed'), ('wtc_AmpPhR', 'Grid current')]
    FV = r(E/'freshness_value_2026-10-14_v001/results.json'); A18 = r(K18/'results_A.json'); S18 = {**r(K18/'results_B.json'), **r(K18/'results_C.json')}
    D18 = r(K18/'results_D.json'); R19 = r(P19/'results.json'); AH = r(H19/'results_A.json'); SH = {**r(H19/'results_B.json'), **r(H19/'results_C.json')}
    j17 = float((E/'freshness_value_2026-10-14_v001/both_stale_output.txt').read_text(encoding='utf-8-sig').split('all ')[1].split()[0])
    sets = [('Kelmarsh 2017 (development)', FV['all_fresh']['all'], {lab: pick(FV, p) for p, lab in SEN}, j17, 'oil inlet + both axle bearings'),
            ('Kelmarsh 2018', A18['h30_x']['all'], {lab: pick(S18, 'stale:' + p) for p, lab in SEN}, D18['stale:inlet+bearings']['all'], 'oil inlet + both axle bearings'),
            ('Penmanshiel 2019', R19['h30_x']['all'], {lab: pick(R19, 'stale:' + p) for p, lab in SEN}, R19['stale:inlet+bearings']['all'], 'oil inlet + both axle bearings'),
            ('Hill of Towie 2019', AH['base_x']['all'], {lab: pick(SH, 'stale:' + p) for p, lab in SIE}, AH['stale:all_gearbox_bearings']['all'], 'all four gearbox bearings')]
    carriers = {'Oil inlet', 'Rear axle bearing', 'Front axle bearing', 'IMS bearing, rotor side', 'IMS bearing, gen. side', 'HS bearing, gen. side', 'HS bearing, rotor side'}
    fig = plt.figure(figsize=(TXT, 2.9)); gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.95, 1.05], wspace=0.12)
    axs = [fig.add_subplot(gs[i]) for i in (0, 1, 2, 4)]
    for j, (ax, (name, base, ch, joint, jname)) in enumerate(zip(axs, sets)):
        labs = list(ch); v = np.array([100 * (ch[k] / base - 1) for k in labs]); y = np.arange(len(labs))[::-1]
        cols = [(ORANGE if 'bearing' in k.lower() else BLUE) if k in carriers else '#c9c8c3' for k in labs]
        ax.barh(y, v, height=0.68, color=cols, zorder=3, lw=0)
        for yy, vv in zip(y, v):
            if abs(vv) >= 3: ax.text(vv + 0.6, yy, f'+{vv:.1f}', va='center', fontsize=6.2)
        ax.axvline(0, color=INK2, lw=0.6); ax.set_xlim(-3, 31); grid(ax, 'x')
        ax.set_yticks(y, labs if j in (0, 3) else [''] * len(labs)); ax.tick_params(axis='y', length=0, labelsize=6.3)
        ax.set_title(name, fontsize=7.2, pad=4)
        ax.text(30, y[-1] - 0.3, f'all carriers late\ntogether: +{100 * (joint / base - 1):.0f}%', ha='right', va='bottom', fontsize=6.1,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRID, lw=0.6))
    fig.text(0.5, 0.0, 'Increase of the MAE when the channel arrives $d$ steps late (%)', ha='center', va='bottom', fontsize=7.5)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.12, top=0.9)
    fig.text(0.40, 0.975, 'Senvion turbines', ha='center', fontsize=7.6, fontweight='bold')
    fig.text(0.86, 0.975, 'Siemens turbines', ha='center', fontsize=7.6, fontweight='bold')
    save(fig, 'fig6_freshness')


# ------------------------------------------------------------------ Fig. 7: oil circuit (schematic, viscometer, error ladder)
def box(ax, x, y, w, h, text, fc, ec=INK2, fs=6.6, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.01,rounding_size={r}', fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=4)


def arrow(ax, p, q, c=INK2, cs='arc3'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=1.1, color=c, connectionstyle=cs, zorder=2))


def fig_oil():
    F = load(E/'physics_oil_circuit_2026-10-15_v001/figdata_dp_vs_T.json')
    H19 = E/'hill_of_towie_replication_2019_2026-10-13_v001'; AH = load(H19/'results_A.json')['results']; HH = load(H19/'results_h36.json')['results']
    fig = plt.figure(figsize=(TXT, 2.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.0], wspace=0.42)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[2])
    # (a) schematic of the Siemens circuit (after the service manual's oil diagram, redrawn and simplified)
    a.set_xlim(0, 10); a.set_ylim(0, 7.2); a.axis('off'); OIL = '#f6d58a'
    box(a, 0.3, 0.4, 3.6, 2.6, '', '#dfe7f1'); a.text(2.1, 2.62, 'Gearbox', ha='center', fontsize=7, fontweight='bold')
    for cx, cy, rr in [(1.25, 1.55, 0.62), (2.55, 1.35, 0.42), (3.3, 1.75, 0.28)]:
        a.add_patch(Circle((cx, cy), rr, fc='white', ec=INK2, lw=0.8, zorder=4)); a.add_patch(Circle((cx, cy), rr * 0.35, fc='#c9d6e6', ec=INK2, lw=0.6, zorder=5))
    a.add_patch(Rectangle((0.45, 0.5), 3.3, 0.38, fc=OIL, ec='none', zorder=4)); a.text(2.1, 0.69, 'oil sump  $T$', ha='center', va='center', fontsize=6.2, zorder=5)
    a.text(3.75, 2.2, 'bearings $T_b$', ha='right', fontsize=5.8, color=ORANGE, zorder=6)
    box(a, 4.3, 0.4, 2.15, 1.25, 'Pump\n70 or 142\nL/min', '#fbe3d6', fs=5.8)
    box(a, 7.05, 0.5, 2.4, 1.05, 'Inline filter', '#e3f3eb', fs=6.4)
    for gx, lab in [(7.4, '$p_1$'), (9.1, '$p_2$')]:
        a.add_patch(Circle((gx, 2.05), 0.27, fc='white', ec=AQUA, lw=1.0, zorder=4)); a.text(gx, 2.05, lab, ha='center', va='center', fontsize=6.0, zorder=5)
        a.plot([gx, gx], [1.56, 1.78], color=AQUA, lw=0.9)
    a.text(7.0, 3.25, r'$\Delta p_f = p_1-p_2 = R_i\,\mu(T)\,Q$', ha='center', fontsize=6.6, color=INK,
           bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=AQUA, lw=0.7))
    box(a, 6.4, 4.45, 3.05, 1.1, 'Cooler\n2-stage fan', '#e8f1fb', fs=6.4); box(a, 0.6, 4.45, 3.0, 1.1, 'Nozzles,\nspray to gears', '#f1edf9', fs=6.4)
    arrow(a, (3.9, 1.02), (4.35, 1.02)); arrow(a, (6.4, 1.02), (7.05, 1.02))
    a.plot([9.45, 9.85, 9.85], [1.02, 1.02, 5.0], color=INK2, lw=1.1, zorder=2); arrow(a, (9.85, 5.0), (9.45, 5.0))
    arrow(a, (6.4, 5.0), (3.6, 5.0)); arrow(a, (2.1, 4.45), (2.1, 3.0))
    a.text(5.0, 5.2, '$T_{in}$', fontsize=6.4, ha='center'); a.text(9.9, 6.45, 'controller sets pump and fan stages', ha='right', fontsize=5.8, color=INK2)
    # (b) filter pressure drop against oil temperature, both pump stages, with a viscosity curve fitted per stage
    for st, col, lab in [('142', BLUE, '142 L/min'), ('70', ORANGE, '70 L/min')]:
        R = F['stages'][st]; T = np.array([x['T'] for x in R]); m = np.array([x['median'] for x in R]); q1 = np.array([x['q25'] for x in R]); q3 = np.array([x['q75'] for x in R])
        nu = np.array([x['nu'] for x in R]); w = np.array([x['n'] for x in R], float)
        b.fill_between(T, q1, q3, color=col, alpha=0.18, lw=0, zorder=2); b.plot(T, m, 'o', ms=3.2, color=col, mec='white', mew=0.4, zorder=4, label=lab)
        k = np.sum(w * m * nu) / np.sum(w * nu * nu); b.plot(T, k * nu, color=col, lw=1.1, ls=(0, (3, 1.5)), zorder=3)
    b.plot([], [], color=INK2, lw=1.1, ls=(0, (3, 1.5)), label=r'$\propto\nu(T)$ (Walther)')
    b.set_xlabel('Oil temperature $T$ (°C)'); b.set_ylabel(r'Filter pressure drop $\Delta p_f$ (bar)'); b.legend(loc='upper right', fontsize=6.3); grid(b, 'both')
    # (c) error with the four gearbox bearings late
    bars = [('Bearings\nlate', AH['stale:all_gearbox_bearings']['all'], '#c9c8c3', None), ('+ oil\npressure', AH['stale:all_gearbox_bearings+pressure']['all'], AQUA, None),
            ('+ pump,\nfilter', HH['stale:all_gearbox_bearings+pressure+r7']['all'], BLUE, None),
            ('+ fan,\nheater*', HH['stale:all_gearbox_bearings+pressure+r8']['all'], 'white', BLUE)]
    for i, (lab, v, col, hatch_c) in enumerate(bars):
        c.bar(i, v, width=0.64, color=col, ec=hatch_c or col, hatch='////' if hatch_c else None, lw=0.8 if hatch_c else 0, zorder=3)
        c.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=6.4)
    fresh = AH['base_x']['all']; c.axhline(fresh, color=ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=4, label=f'bearings on time ({fresh:.3f})')
    c.legend(loc='upper right', fontsize=6.2); c.text(3.3, -0.2, '*descriptive only', fontsize=5.8, color=INK2, ha='right', transform=c.transData)
    c.set_xticks(range(4), [x[0] for x in bars], fontsize=6.2); c.tick_params(axis='x', length=0); c.set_ylabel('MAE (°C)'); c.set_ylim(0, 0.75); grid(c)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.2, top=0.93)
    panel(fig, a, PANEL[1], '(a)', pad=(0.01, 0.02, 0.006, 0.02)); panel(fig, b, PANEL[0], '(b)'); panel(fig, c, PANEL[2], '(c)')
    save(fig, 'fig7_oil')


# ------------------------------------------------------------------ Fig. 8: standstill gate (Amendment 14)
def fig_standstill():
    S = load(PB/'STANDSTILL_DIAG.json'); c = S['case_k24_t4']; n = len(c['truth']); h = np.arange(n) / 24.0   # days from 25 Jan
    fig = plt.figure(figsize=(TXT, 3.0)); gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.34)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    g = np.array(c['gated']); st = np.array(c['stopped'])
    for arr, col, lab in [(st, '#e6e5e1', 'rotor stopped'), (g, '#cfe9dd', r'gate active (stopped $\geq$ 12 h)')]:
        edges = np.flatnonzero(np.diff(np.r_[0, arr.astype(int), 0])); first = True
        for x0, x1 in zip(edges[::2], edges[1::2]):
            a.axvspan(h[x0], h[min(x1, n - 1)], color=col, lw=0, zorder=0, label=lab if first else None); first = False
    f = lambda k: np.array([np.nan if v is None else v for v in c[k]])
    gate_only = np.where(g, f('recg'), np.nan)
    a.plot(h, f('truth'), color=INK, lw=1.5, zorder=5)
    a.plot(h, f('fleet'), color=VIOLET, lw=1.0, ls=(0, (4, 2)), zorder=3)
    a.plot(h, f('rec'), color=BLUE, lw=1.1, zorder=4)
    a.plot(h, gate_only, color=AQUA, lw=1.4, zorder=6)
    a.set_xlim(0, h[-1]); a.set_ylim(0, 64); a.set_xlabel('Days from 25 January 2024'); a.set_ylabel('Oil temperature (°C)'); grid(a)
    for txt, yy, col in [('fleet model alone', 31.5, VIOLET), ('proposed', 23.5, BLUE), ('proposed + gate', 15.3, AQUA), ('measured', 7.4, INK)]:
        a.text(8.6, yy, txt, color=col, fontsize=6.4, fontweight='bold' if col in (AQUA, INK) else None, zorder=9,
               bbox=dict(boxstyle='round,pad=0.12', fc=(1, 1, 1, 0.75), ec='none'))
    a.legend(loc='lower left', fontsize=6.0, handlelength=1.4, borderaxespad=0.3)
    a.set_title('Kelmarsh 2024, turbine 4: a 7.9-day standstill', fontsize=7.4, pad=3)
    fm = c['feb_mae']; a.text(0.985, 0.97, f"February MAE (°C)\nfleet {fm['fleet']:.2f} · proposed {fm['rec']:.2f}\n+ gate {fm['recg']:.2f} · Kim et al. {fm['kim']:.2f}",
                              transform=a.transAxes, ha='right', va='top', fontsize=6.1, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRID, lw=0.6), zorder=8)
    # (b) MAE on the gated rows of every turbine-year: frozen vs gated, Kim as reference
    rows = [('kelmarsh2017', 'Kelmarsh 2017'), ('kelmarsh2018', 'Kelmarsh 2018'), ('penmanshiel2019', 'Penmanshiel 2019'), ('hot2019', 'Hill of Towie 2019'),
            ('kelmarsh2024', 'Kelmarsh 2024'), ('penmanshiel2022', 'Penmanshiel 2022'), ('hot2025', 'Hill of Towie 2025'),
            ('kelmarsh2020', 'Kelmarsh 2020'), ('penmanshiel2020', 'Penmanshiel 2020'), ('hot2022', 'Hill of Towie 2022')]
    y = np.arange(len(rows))[::-1]
    for yy, (ds, nm) in zip(y, rows):
        d = S[ds]['stop']; b.plot([d['recg'], d['rec']], [yy, yy], color=GRID, lw=2.2, zorder=2, solid_capstyle='round')
        b.scatter(d['rec'], yy, s=22, color=BLUE, zorder=4, edgecolor='white', lw=0.5); b.scatter(d['recg'], yy, s=22, color=AQUA, zorder=5, edgecolor='white', lw=0.5)
        b.scatter(d['kim'], yy, s=16, marker='D', color='white', edgecolor=ORANGE, lw=0.9, zorder=3)
        b.text(9.5, yy, f"{100 * S[ds]['gated_share']:.1f}%", va='center', ha='right', fontsize=6.0, color=INK2)
    b.set_title('Error inside the standstills', fontsize=7.4, pad=14)
    b.set_xscale('log'); b.set_xlim(0.12, 10); b.set_yticks(y, [r[1] for r in rows]); b.tick_params(axis='y', length=0, labelsize=6.4)
    b.set_xticks([0.2, 0.5, 1, 2, 5], ['0.2', '0.5', '1', '2', '5']); b.set_xlabel('MAE on gated rows (°C, log)'); grid(b, 'x')
    for yl in (5.5, 2.5): b.axhline(yl, color=INK2, lw=0.5, ls=(0, (2, 2)))
    b.text(9.5, len(rows) - 0.35, 'rows', ha='right', fontsize=6.0, color=INK2)
    b.scatter([], [], s=22, color=BLUE, label='proposed'); b.scatter([], [], s=22, color=AQUA, label='+ gate'); b.scatter([], [], s=16, marker='D', color='white', edgecolor=ORANGE, lw=0.9, label='Kim et al.')
    b.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.09), ncol=3, fontsize=6.0, handletextpad=0.1, columnspacing=0.7)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.9)
    panel(fig, a, PANEL[0], '(a)'); panel(fig, b, PANEL[2], '(b)'); save(fig, 'fig8_standstill')


if __name__ == '__main__':
    for f in (fig_ladder, fig_scatter, fig_delay, fig_freshness, fig_oil, fig_standstill): f()
