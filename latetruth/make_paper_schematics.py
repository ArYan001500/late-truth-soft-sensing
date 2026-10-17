"""Schematic figures 1 (information available at decision time) and 2 (pipeline of the estimator), same style as make_paper_figures.py.
usage: python make_paper_schematics.py"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from latetruth.make_paper_figures import BLUE, ORANGE, AQUA, VIOLET, INK, INK2, MUTED, PANEL, COL, TXT, save


def rbox(ax, x, y, w, h, fc, ec=INK2, lw=0.8, r=0.08, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.02,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))


def arr(ax, p, q, c=INK2, lw=1.1, cs='arc3', ms=8, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c, connectionstyle=cs, zorder=z))


# ------------------------------------------------------------------ Fig. 1: what is known at decision time t (d = 3)
def fig_timeline():
    x0, x1, y0, y1 = -11.2, 4.0, -1.0, 3.3
    fig, ax = plt.subplots(figsize=(COL, COL * (y1 - y0) / (x1 - x0))); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    rbox(ax, x0 + 0.1, y0 + 0.1, x1 - x0 - 0.2, y1 - y0 - 0.2, PANEL[0], ec='none', r=0.3, z=0)
    steps = np.arange(-7, 4); d = 3; Y_IN, Y_T, Y_R = 2.25, 1.35, 0.5
    ax.plot([-7.45, 3.5], [-0.05, -0.05], color=INK2, lw=0.8)
    for s in steps:
        ax.plot([s, s], [-0.12, 0.02], color=INK2, lw=0.8)
        ax.text(s, -0.38, 't' if s == 0 else ('t+%d' % s if s > 0 else 't−%d' % -s), ha='center', va='top', fontsize=5.6, color=INK2)
    ax.plot([0, 0], [0.1, 2.75], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1); ax.text(0, 2.8, 'decision time', ha='center', va='bottom', fontsize=6.2, fontweight='bold')
    for y, lab in [(Y_IN, 'inputs $\\mathbf{x}_i(s)$'), (Y_T, 'target $y_i(s)$'), (Y_R, 'released record')]:
        ax.text(-10.9, y, lab, fontsize=6.0, va='center')
    for s in steps:
        ax.add_patch(Circle((s, Y_IN), 0.2, fc=BLUE if s <= 0 else 'white', ec=BLUE if s <= 0 else MUTED, lw=0.9, zorder=3))
        if s <= -d: ax.add_patch(Circle((s, Y_T), 0.2, fc=ORANGE, ec=ORANGE, lw=0.9, zorder=3))
        elif s <= 0: ax.add_patch(Rectangle((s - 0.24, Y_T - 0.24), 0.48, 0.48, fc='white', ec=ORANGE, lw=0.9, hatch='/////', zorder=3))
        else: ax.add_patch(Circle((s, Y_T), 0.2, fc='white', ec=MUTED, lw=0.8, zorder=3))
    rbox(ax, -7.3, Y_R - 0.16, 7.3 - d + 0.3 - 0.0 - 0.0, 0.32, '#cfcec9', ec='none', r=0.12)
    arr(ax, (-1.85, Y_T + 0.3), (0.85, Y_T + 0.3), c=ORANGE, lw=0.9, cs='arc3,rad=-0.45', ms=6)
    ax.text(1.3, Y_R, '$y_i(t{-}2)$ is\nreleased at $t{+}1$', fontsize=5.5, va='center', ha='left', color=ORANGE)
    ax.annotate('', xy=(0.3, Y_T - 0.5), xytext=(-2.3, Y_T - 0.5), arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=0.8))
    ax.text(-1.0, Y_T - 0.62, 'in transit', fontsize=5.5, ha='center', va='top', color=ORANGE)
    save(fig, 'fig1_timeline')


# ------------------------------------------------------------------ Fig. 2: pipeline with a gearbox schematic (v2, 2026-10-17)
def gear(ax, cx, cy, r, n, fc='white', ec=INK2, z=4, rot=0.0):
    """Gear outline with n trapezoidal teeth."""
    pts = []
    for k in range(n):
        a0 = rot + 2 * np.pi * k / n; dt = 2 * np.pi / n
        for f, rr in [(0.0, r), (0.18, r + 0.16 * r ** 0.5), (0.42, r + 0.16 * r ** 0.5), (0.6, r)]:
            a = a0 + f * dt; pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    ax.add_patch(plt.Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.8, zorder=z))
    ax.add_patch(Circle((cx, cy), r * 0.42, fc='#eef2f7', ec=ec, lw=0.6, zorder=z + 1))


def chip(ax, x, y, text, c, fs=5.4, z=6):
    ax.text(x, y, text, fontsize=fs, ha='left', va='center', color='white', zorder=z,
            bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc=c, ec='none'))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(TXT, 2.85)); ax.set_xlim(0, 28); ax.set_ylim(0, 11.2); ax.axis('off'); fig.subplots_adjust(0, 0, 1, 1)
    LATE, FRESH, OIL = ORANGE, BLUE, '#f3c969'
    # ---------------- (a) gearbox, oil circuit, and which channels arrive on time
    rbox(ax, 0.1, 0.15, 8.75, 10.9, PANEL[1], ec='none', r=0.45, z=0)
    ax.text(0.45, 10.55, '(a) Gearbox and its channels', fontsize=7.0, fontweight='bold', va='top')
    rbox(ax, 0.7, 2.55, 5.15, 6.55, '#e3eaf3', ec=INK2, lw=0.9, r=0.35, z=2)                     # casing
    ax.add_patch(Rectangle((0.82, 2.68), 4.92, 1.15, fc=OIL, ec='none', zorder=3))               # oil sump
    xs = np.linspace(0.82, 5.74, 60); ax.plot(xs, 3.83 + 0.07 * np.sin(xs * 6), color='#d9a63a', lw=0.8, zorder=3)
    gear(ax, 2.2, 6.15, 1.2, 18); gear(ax, 4.3, 4.75, 0.78, 12, rot=0.12); gear(ax, 4.85, 7.55, 0.5, 9, rot=0.3)
    for (bx, by) in [(2.2, 6.15), (4.3, 4.75), (4.85, 7.55)]:                                     # bearings at the hubs
        ax.add_patch(Circle((bx, by), 0.2, fc=FRESH, ec='white', lw=0.6, zorder=7))
    ax.text(3.28, 3.2, 'oil temperature $y_i$', fontsize=5.8, ha='center', va='center', zorder=6)
    chip(ax, 0.95, 9.55, 'bearing temperatures: on time', FRESH)
    chip(ax, 1.3, 2.05, 'oil temperature: released $d$ steps late', LATE)
    # oil circuit loop on the right of the casing
    rbox(ax, 6.45, 2.7, 1.9, 1.15, '#fbe3d6', ec=INK2, lw=0.7, r=0.15); ax.text(7.4, 3.27, 'pump +\nfilter', fontsize=5.4, ha='center', va='center', zorder=5)
    rbox(ax, 6.45, 7.2, 1.9, 1.15, '#e1eefa', ec=INK2, lw=0.7, r=0.15); ax.text(7.4, 7.77, 'cooler', fontsize=5.6, ha='center', va='center', zorder=5)
    arr(ax, (5.88, 3.27), (6.42, 3.27), c='#c48a1f', lw=1.1, ms=6); arr(ax, (7.4, 3.9), (7.4, 7.15), c='#c48a1f', lw=1.1, ms=6)
    arr(ax, (6.42, 7.77), (5.88, 7.77), c='#c48a1f', lw=1.1, ms=6)
    ax.add_patch(Circle((7.95, 5.5), 0.38, fc='white', ec=AQUA, lw=1.0, zorder=5))                # pressure gauge
    ax.plot([7.95, 8.15], [5.5, 5.72], color=AQUA, lw=0.9, zorder=6); ax.text(7.95, 4.85, '$\\Delta p_f$', fontsize=5.6, color=AQUA, ha='center', va='top', zorder=6)
    ax.text(6.95, 5.5, 'oil\ncircuit', fontsize=5.0, color=INK2, ha='center', va='center', zorder=6)
    ax.text(4.45, 1.0, 'power, speeds, nacelle and ambient\ntemperatures, oil pressures: on time', fontsize=5.4, ha='center', va='center', color=FRESH)
    # ---------------- (b) estimator
    rbox(ax, 9.05, 0.15, 18.85, 10.9, PANEL[0], ec='none', r=0.45, z=0)
    ax.text(9.4, 10.55, '(b) Receipt-causal estimator at decision time $t$', fontsize=7.0, fontweight='bold', va='top')
    W, G, X0, YB, H = 5.6, 0.55, 9.5, 4.25, 5.35
    heads = ['Fresh physical channels', 'Fleet model', 'Late-truth correction']
    for j, title in enumerate(heads):
        x = X0 + j * (W + G)
        rbox(ax, x, YB, W, H, 'white', ec='#c7d6ea', lw=0.9, r=0.3)
        ax.add_patch(Circle((x + 0.55, YB + H - 0.55), 0.32, fc=FRESH, ec='none', zorder=5))
        ax.text(x + 0.55, YB + H - 0.55, str(j + 1), fontsize=6.4, color='white', fontweight='bold', ha='center', va='center', zorder=6)
        ax.text(x + 1.05, YB + H - 0.55, title, fontsize=6.5, fontweight='bold', va='center', zorder=6)
        if j: arr(ax, (x - G + 0.04, YB + H / 2), (x - 0.04, YB + H / 2), c=INK2, lw=1.2, ms=8)
    arr(ax, (8.9, YB + H / 2), (X0 - 0.04, YB + H / 2), c=FRESH, lw=1.4, ms=9)
    # box 1: feature groups as chips
    x = X0; items = [('drivers', FRESH), ('bearings', FRESH), ('cooler, load terms', FRESH), ('delayed state $y_i(t{-}d)$', LATE),
                     ('oil circuit, if bearings late', AQUA)]
    for i, (t_, c) in enumerate(items): chip(ax, x + 0.45, YB + H - 1.45 - i * 0.72, t_, c, fs=5.4)
    # box 2: pooled turbines -> tree ensemble -> estimate
    x = X0 + W + G
    for i in range(4):
        rbox(ax, x + 0.5 + i * 0.18, YB + 2.05 + i * 0.25, 1.35, 0.85, ['#dbe8f7', '#cfe0f3', '#c3d8ef', '#b6cfeb'][i], ec='white', lw=0.5, r=0.08, z=4 + i)
    ax.text(x + 1.72, YB + 3.25, 'turbines\n$1\\ldots N$', fontsize=5.0, ha='center', va='center', zorder=9)
    for k in range(3):                                                                               # three small binary trees
        tx, ty = x + 3.05 + k * 0.82, YB + 3.55
        nodes = [(tx, ty), (tx - 0.22, ty - 0.42), (tx + 0.22, ty - 0.42), (tx - 0.33, ty - 0.84), (tx - 0.11, ty - 0.84), (tx + 0.11, ty - 0.84), (tx + 0.33, ty - 0.84)]
        for a_, b_ in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
            ax.plot([nodes[a_][0], nodes[b_][0]], [nodes[a_][1], nodes[b_][1]], color=INK2, lw=0.6, zorder=5)
        for i_, (px_, py_) in enumerate(nodes): ax.add_patch(Circle((px_, py_), 0.075 if i_ < 3 else 0.065, fc=VIOLET if i_ < 3 else '#9d93d6', ec='none', zorder=6))
    ax.text(x + 3.87, YB + 2.35, '400 trees', fontsize=4.9, color=INK2, ha='center', va='center', zorder=6)
    arr(ax, (x + 2.35, YB + 2.95), (x + 2.62, YB + 2.95), c=INK2, lw=0.8, ms=5)
    ax.text(x + W / 2, YB + 1.3, '$\\hat{y}_i(t) = g(\\boldsymbol{\\phi}_i(t),\\, i)$', fontsize=6.4, ha='center', va='center', zorder=6)
    ax.text(x + W / 2, YB + 0.62, 'one model per fleet, refitted monthly', fontsize=5.2, ha='center', va='center', color=INK2, zorder=6)
    # box 3: neighbour scatter inset + formula
    x = X0 + 2 * (W + G); rng = np.random.default_rng(3); px = rng.uniform(0.35, 2.85, 46); py = rng.uniform(0.25, 1.65, 46)
    cx_, cy_ = 1.75, 0.95; dd = np.hypot(px - cx_, py - cy_); nn = np.argsort(dd)[:7]
    ox, oy = x + 0.55, YB + 1.85
    rbox(ax, ox + 0.1, oy, 3.1, 1.95, '#f4f6fa', ec='none', r=0.12, z=4)
    ax.scatter(ox + px, oy + py, s=4, color='#b9b8b3', zorder=5, lw=0)
    ax.scatter(ox + px[nn], oy + py[nn], s=7, color=LATE, zorder=6, lw=0)
    ax.add_patch(Circle((ox + cx_, oy + cy_), dd[nn].max() + 0.06, fc='none', ec=LATE, lw=0.6, ls=(0, (2, 1.5)), zorder=6))
    ax.scatter([ox + cx_], [oy + cy_], s=26, marker='*', color=FRESH, zorder=7, lw=0)
    ax.text(ox + 3.35, oy + 1.55, 'current\nstate', fontsize=4.9, color=FRESH, va='center')
    ax.text(ox + 3.35, oy + 0.6, '$k$ nearest\nreleased\nstates', fontsize=4.9, color=LATE, va='center')
    ax.text(x + W / 2, YB + 1.32, '$\\hat{y}^{\\star}_i(t) = \\hat{y}_i(t) + \\frac{k}{k+k_0}\\,\\bar{r}_{N_k}(t)$', fontsize=6.0, ha='center', va='center', zorder=6)
    ax.text(x + W / 2, YB + 0.62, '$\\hat{y}^{\\star\\star}_i(t) = \\hat{y}^{\\star}_i(t) + c\\,e_i(t)$', fontsize=6.0, ha='center', va='center', zorder=6)
    ax.text(x + W / 2, YB + 0.22, 'similar states: slow bias; 3-h average: fast drift', fontsize=4.8, ha='center', va='center', color=INK2, zorder=6)
    # ---------------- released record as a time strip (same visual language as Fig. 1)
    sx0, sw, cells, d = X0, 3 * W + 2 * G, 24, 3; cw = (sw - 4.2) / cells; yb = 0.75
    ax.text(sx0, 2.75, 'released record', fontsize=5.8, color=INK2, va='center')
    for c in range(cells):
        x = sx0 + 4.2 + c * cw; late = c >= cells - d
        ax.add_patch(Rectangle((x + 0.03, yb + 0.95), cw - 0.06, 0.55, fc='white' if late else '#f6b593', ec=LATE if late else 'white',
                               lw=0.6, hatch='/////' if late else None, zorder=4))
    ax.text(sx0 + 4.2 + (cells - d) * cw - 0.1, yb + 0.55, '$t{-}d$  ', fontsize=5.4, ha='right', va='center', color=INK2)
    ax.text(sx0 + 4.2 + cells * cw + 0.05, yb + 1.22, '$t$', fontsize=5.4, ha='left', va='center', color=INK2)
    ax.text(sx0 + 4.2 + (cells - d / 2) * cw, yb + 0.55, 'in transit', fontsize=5.0, ha='center', va='center', color=LATE)
    ax.text(sx0 + 0.0, yb + 1.25, 'targets $y_j(s)$ and\nresiduals $r_i(s)$, $s \\leq t-d$', fontsize=5.0, va='center', color=INK2)
    xf = X0 + W + G + W / 2; xr = X0 + 2 * (W + G) + 1.0
    arr(ax, (xf, yb + 1.55), (xf, YB - 0.05), c=MUTED, lw=0.9, ms=6); ax.text(xf + 0.15, 3.35, 'training rows', fontsize=5.1, color=INK2, va='center')
    arr(ax, (xr, yb + 1.55), (xr, YB - 0.05), c=MUTED, lw=0.9, ms=6); ax.text(xr - 0.15, 3.35, 'residuals', fontsize=5.1, color=INK2, va='center', ha='right')
    xo = X0 + 2 * (W + G) + W - 0.55
    arr(ax, (xo, YB - 0.05), (xo, 3.0), c=FRESH, lw=1.4, ms=9)
    ax.text(xo - 0.2, 3.35, 'estimate $\\hat{y}^{\\star\\star}_i(t)$', fontsize=6.0, color=FRESH, ha='right', va='center', fontweight='bold')
    save(fig, 'fig2_pipeline')

if __name__ == '__main__':
    fig_timeline(); fig_pipeline()
