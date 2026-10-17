"""B10 of protocols/PUBLISHED_BASELINES_v001.md — Wang, Alipouri, Huang 2022 (IEEE TIM 71, 2500509):
DNEKF (fast rate) + SNEKF (delayed accurate measurements) synchronization with exclusive-information fusion (their eqs. 2–25, Lemma 1).
Adaptation (documented): scalar quality state x = gear-oil temperature; state model f(x) = x (random walk, as in their industrial SRU case)
+ state-compensation NN; fast-rate sensor = fleet soft-sensor output P_fleet (h30_x HGB, the same base model our method corrects)
+ output-compensation NN; slow-rate accurate measurement = released oil truth y(s), available at ds = s + 3.
NNs: one hidden layer, 3 tanh units, parameters estimated online by state augmentation (as in the paper).
Noise variances chosen on Kelmarsh 2017 Feb–Mar (grid of 120: q, r_fast, r_slow, q_theta), then frozen for 2018.
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import sys, json, hashlib, itertools
import numpy as np
from latetruth.baselines_common import OUT, EV0, DELAY, score

H = 3; NP = 3 * H + 1


class NN:
    """g(x; th) = w2 . tanh(w1 * xn + b1) + b2, xn = (x - mu) / sd; th = [w1(H), b1(H), w2(H), b2]."""
    def __init__(self, mu, sd): self.mu, self.sd = mu, sd

    def __call__(self, x, th):
        w1, b1, w2, b2 = th[:H], th[H:2 * H], th[2 * H:3 * H], th[3 * H]
        xn = (x - self.mu) / self.sd; h = np.tanh(w1 * xn + b1); dh = 1 - h * h
        g = w2 @ h + b2; dgdx = (w2 * dh * w1).sum() / self.sd
        dgdth = np.concatenate([w2 * dh * xn, w2 * dh, h, [1.0]])
        return g, dgdx, dgdth


def run(yf, y, ok_truth, prm, mu, sd, seed=0, th_init=None, frozen=False):
    """Estimate x_t for t = EV0..n-1 using yf[t] (fast, available at t) and y[t-3] (released at t)."""
    q, rf, ri, qth = prm; n = len(yf); rng = np.random.default_rng(seed); nn = NN(mu, sd)
    th0 = rng.normal(0, 0.01, NP) if th_init is None else th_init[0]
    thy0 = th0.copy() if th_init is None else th_init[1]
    if frozen: qth = 0.0
    p_th = 0.0 if frozen else 1e-2
    # fast augmented state [x, thx, thy]; slow augmented state [x, thx]
    i0 = next(i for i in range(EV0 - DELAY, -1, -1) if ok_truth[i])
    a = np.concatenate([[y[i0]], th0, thy0]); Pf = np.diag([1.0] + [p_th] * (2 * NP))
    b = np.concatenate([[y[i0]], th0.copy()]); Ps = np.diag([1.0] + [p_th] * NP)
    Qf = np.diag([q] + [qth] * (2 * NP)); Qs = np.diag([q] + [qth] * NP)
    ix, iy = slice(1, 1 + NP), slice(1 + NP, 1 + 2 * NP)
    out = np.full(n, np.nan); s_track = i0  # slow posterior is at time s_track
    for t in range(EV0 - DELAY, n):
        # ---- DNEKF prediction (eq. 2, 4, 5)
        g, gx, gth = nn(a[0], a[ix]); F = np.eye(len(a)); F[0, 0] = 1 + gx; F[0, ix] = gth
        a[0] = a[0] + g; Pf = F @ Pf @ F.T + Qf
        # ---- DNEKF correction with fast measurement (eq. 3, 6–9); thx is not updated by fast data
        if np.isfinite(yf[t]):
            gy, gyx, gyth = nn(a[0], a[iy]); Hm = np.zeros(len(a)); Hm[0] = 1 + gyx; Hm[iy] = gyth
            S = Hm @ Pf @ Hm + rf; K = Pf @ Hm / S; K[ix] = 0.0
            a = a + K * (yf[t] - (a[0] + gy)); Pf = (np.eye(len(a)) - np.outer(K, Hm)) @ Pf; Pf = (Pf + Pf.T) / 2
        # ---- SNEKF with the delayed accurate measurement y(s), s = t - 3, released now (ds = t)
        s = t - DELAY
        if s > s_track and ok_truth[s]:
            while s_track < s:  # slow prediction s_track -> s (eq. 10, 12, 13)
                g, gx, gth = nn(b[0], b[1:]); Fs = np.eye(len(b)); Fs[0, 0] = 1 + gx; Fs[0, 1:] = gth
                b[0] = b[0] + g; Ps = Fs @ Ps @ Fs.T + Qs; s_track += 1
            b_prior, P_prior = b.copy(), Ps.copy()
            S = Ps[0, 0] + ri; K = Ps[:, 0] / S  # eq. 14–17, H = [1, 0]
            b = b + K * (y[s] - b[0]); Ps = Ps - np.outer(K, Ps[0, :]); Ps = (Ps + Ps.T) / 2
            a[ix] = b[1:]  # SNN parameters shared with the DNEKF
            # ---- propagation over the delay window (eq. 18–20) and mutual estimate (eq. 23–25)
            def prop(x0, p0):
                x, p = x0, p0
                for _ in range(DELAY):
                    g, gx, _ = nn(x, b[1:]); x = x + g; p = (1 + gx) ** 2 * p + q
                return x, p
            xi, pi = prop(b[0], Ps[0, 0]); xm, pm = prop(b_prior[0], P_prior[0, 0])
            # ---- exclusive information (Lemma 1 inverted) and fusion with the fast estimate (eq. 21–22)
            rz_inv = 1 / pi - 1 / pm
            if rz_inv > 1e-9:
                z = (xi / pi - xm / pm) / rz_inv
                pf = Pf[0, 0]; pc = 1 / (1 / pf + rz_inv); xc = pc * (a[0] / pf + rz_inv * z)
                scale = np.sqrt(pc / pf); Pf[0, 1:] *= scale; Pf[1:, 0] *= scale; Pf[0, 0] = pc; a[0] = xc
        out[t] = a[0]
    return out


def offline_nn(yf, y, mu, sd):
    """Frozen variant (Amendment 2): fit SNN on x(t+1)-x(t) ~ g(x(t)) and MNN on yf(t)-x(t) ~ g(x(t)) over released January rows."""
    from sklearn.neural_network import MLPRegressor
    tr = np.arange(1, EV0 - DELAY); xs = (y[tr - 1] - mu) / sd; ok = np.isfinite(y[tr]) & np.isfinite(y[tr - 1])
    def fit(xin, target):
        m = MLPRegressor(hidden_layer_sizes=(H,), activation='tanh', solver='lbfgs', max_iter=2000, alpha=1e-3, random_state=0).fit(xin[:, None], target)
        return np.concatenate([m.coefs_[0][0], m.intercepts_[0], m.coefs_[1][:, 0], m.intercepts_[1]])
    thx = fit(xs[ok], (y[tr] - y[tr - 1])[ok])
    okf = np.isfinite(yf[tr]) & np.isfinite(y[tr])
    thy = fit(((y[tr] - mu) / sd)[okf], (yf[tr] - y[tr])[okf]) if okf.sum() > 50 else np.zeros(NP)
    return thx, thy


def load_npz(ds):
    z = np.load(OUT/f'ours_{ds}.npz', allow_pickle=False); T = sorted({int(k.split('_')[-1]) for k in z.files if k.startswith('Y_')})
    return z['month'], {t: z[f'Y_{t}'] for t in T}, {t: z[f'OK_{t}'] for t in T}, {t: z[f'P_fleet_{t}'] for t in T}


def apply(ds, prm):
    month, Y, OK, PF = load_npz(ds); n = len(month); P = {}
    for t in Y:
        rel = np.isfinite(Y[t]); mu, sd = np.nanmean(Y[t][:EV0 - DELAY]), np.nanstd(Y[t][:EV0 - DELAY])
        P[t] = run(PF[t], Y[t], rel, prm, mu, sd)
    return score(Y, P, OK, month, n), P


def _val(prm):
    month, Y, OK, PF = load_npz('kelmarsh2017'); val_end = int(np.flatnonzero(month == 4)[0]); e = []
    for t in Y:
        rel = np.isfinite(Y[t]); mu, sd = np.nanmean(Y[t][:EV0 - DELAY]), np.nanstd(Y[t][:EV0 - DELAY])
        p = run(PF[t][:val_end], Y[t][:val_end], rel[:val_end], prm, mu, sd); ev = np.arange(EV0, val_end)
        ok = OK[t][ev] & np.isfinite(p[ev]); e.append(np.abs(Y[t][ev] - p[ev])[ok].mean())
    return float(np.mean(e))


GRID = list(itertools.product([0.01, 0.05, 0.2, 1.0, 5.0], [0.001, 0.01, 0.05, 0.2], [1e-4, 1e-3, 1e-2], [1e-6, 1e-5]))


def tune():
    from multiprocessing import Pool
    with Pool(min(8, JOBS)) as pool: v = pool.map(_val, GRID)
    for prm, m in zip(GRID, v): print('tune', prm, round(m, 4), flush=True)
    i = int(np.argmin(v)); return v[i], GRID[i]


if __name__ == '__main__':
    f = OUT/'B10_wang2022.json'
    if f.exists(): raise FileExistsError(f)
    best = tune(); print('chosen', best, flush=True); res = {'tuning_val_mae_2017FebMar': best[0], 'params(q,rf,ri,qth)': best[1]}
    for ds in ['kelmarsh2017', 'kelmarsh2018']:
        sc, P = apply(ds, best[1]); res[ds] = sc; print(ds, 'B10 Wang2022', round(sc['all'], 4), flush=True)
        np.savez_compressed(OUT/f'B10_wang2022_{ds}.npz', **{f'P_{t}': v for t, v in P.items()})
    res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); f.write_text(json.dumps(res, indent=2))
