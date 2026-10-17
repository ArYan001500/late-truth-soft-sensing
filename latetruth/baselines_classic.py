"""Protocol 05_Protocols/PUBLISHED_BASELINES_v001.md — B0, B1 Bangalore 2015, B2/B2f Schlechtingen 2011, B3 Liu 2020, B5 Yang 2021.
Pure numpy/sklearn/xgboost (CPU implementation). Reads inputs_<ds>.npz written by baselines_common.export_inputs.
All models per turbine, retrained at the same monthly starts as ours on released rows only (rows < start - 3).
Target history: receipt-causal lag 3 (main) or synchronous lag 1 (secondary, claim C3).
Hyper-parameters: chosen once on Kelmarsh 2017 (train rows < 1 March - 3, validate March), then frozen.
usage: python baselines_classic.py <B0|B1|B2|B2f|B3|B5> <lag 3|1>
"""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR, JOBS
import sys
import sys, json, hashlib, itertools
from pathlib import Path
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
OUT = OUT_DIR/'published_baselines_2026-10-14_v001'; EV0 = 4032; DELAY = 3
POWER, WIND, ROTOR, GENRPM, AMB, NAC, CUR, GENB, INLET, FRONT, REAR, ROTB, GBS, VANE, PPUMP, PINL = range(16)


def load(ds):
    z = np.load(OUT/f'inputs_{ds}.npz', allow_pickle=False); T = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('X_')})
    return z['month'], {t: z[f'X_{t}'] for t in T}, {t: z[f'Y_{t}'] for t in T}, {t: z[f'OK_{t}'] for t in T}


def windows(month, n):
    s = [EV0] + [int(np.flatnonzero(month == m)[0]) for m in range(3, 13)]
    return [(a, s[j + 1] if j + 1 < len(s) else n) for j, a in enumerate(s)]


def score(Y, P, OK, month):
    n = len(month); a, ca, tr, mo = [], [], [], []
    for t in Y:
        ev = np.arange(EV0, n); ok = OK[t][ev] & np.isfinite(P[t][ev]); err = np.abs(Y[t][ev] - P[t][ev])
        mv = np.abs(Y[t][ev] - Y[t][ev - 3]); q50, q90 = np.quantile(mv[ok], [.5, .9])
        a.append(err[ok].mean()); ca.append(err[ok & (mv < q50)].mean()); tr.append(err[ok & (mv >= q90)].mean())
        mo.append([err[ok & (month[ev] == m_)].mean() for m_ in range(2, 13)])
    return dict(all=float(np.mean(a)), calm=float(np.mean(ca)), transient=float(np.mean(tr)), months=np.nanmean(mo, 0).round(4).tolist())


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return 0.0 if m.sum() < 50 or np.std(a[m]) == 0 else float(np.corrcoef(a[m], b[m])[0, 1])


def shift(v, k): return np.r_[np.full(k, np.nan), v[:len(v) - k]] if k > 0 else v


class Base:
    """Builds the design matrix at rows r from raw channels X and the target history allowed by `lag`."""
    def __init__(self, hp, lag): self.hp, self.lag = hp, lag

    def prepare(self, X, y, tr): pass          # choose channels / lags on training rows (method-specific)

    def design(self, X, y, r): raise NotImplementedError

    def fit(self, X, y, tr): self.prepare(X, y, tr); self.m = self.make(); A = self.design(X, y, tr); ok = np.isfinite(A).all(1); self._fit(A[ok], y[tr][ok])

    def _fit(self, A, b):
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-9; self.m.fit((A - self.mu) / self.sd, b)

    def predict(self, X, y, r):
        A = self.design(X, y, r); out = np.full(len(r), np.nan); ok = np.isfinite(A).all(1)
        if ok.any(): out[ok] = self.m.predict((A[ok] - self.mu) / self.sd)
        return out


class B0(Base):
    def fit(self, X, y, tr): pass

    def predict(self, X, y, r): return y[r - self.lag]


class B1(Base):
    """Bangalore & Tjernberg 2015: NARX q=1 MLP; inputs power, nacelle temp, rotor rpm, gearbox bearing temps, + previous target."""
    CH = [POWER, NAC, ROTOR, FRONT, REAR, ROTB]
    def make(self): return MLPRegressor(hidden_layer_sizes=(self.hp['h'],), activation='tanh', solver='lbfgs', max_iter=2000, random_state=0)
    def design(self, X, y, r): return np.column_stack([X[r][:, self.CH], y[r - self.lag]])


class B2(Base):
    """Schlechtingen & Santos 2011 AR-NN (and FSRC when ar=False): power, nacelle temp, generator speed, most correlated temperature,
    each shifted by its cross-correlation lag (0..6) on the training rows; min-max scaling; sigmoid MLP; last 3 months of training data."""
    ar = True
    def make(self): return MLPRegressor(hidden_layer_sizes=(self.hp['h'],), activation='logistic', solver='lbfgs', max_iter=2000, random_state=0)

    def prepare(self, X, y, tr):
        temps = [GENB, INLET, FRONT, REAR, ROTB]
        self.ch = [POWER, NAC, GENRPM, max(temps, key=lambda c: abs(corr(X[tr, c], y[tr])))]
        self.lags = [max(range(7), key=lambda k: abs(corr(shift(X[:, c], k)[tr], y[tr]))) for c in self.ch]

    def fit(self, X, y, tr): super().fit(X, y, tr[-13104:])

    def design(self, X, y, r):
        cols = [shift(X[:, c], k)[r] for c, k in zip(self.ch, self.lags)]
        if self.ar: cols.append(y[r - self.lag])
        return np.column_stack(cols)

    def _fit(self, A, b):
        self.mu, self.sd = A.min(0), A.max(0) - A.min(0) + 1e-9; self.m.fit((A - self.mu) / self.sd, b)


class B2f(B2): ar = False


class B3(Base):
    """Liu, Wu, Wang 2020: XGBoost; inputs = channels with |Pearson r| > 0.6 to the target + target of the previous period."""
    def make(self):
        import xgboost as xgb
        return xgb.XGBRegressor(max_depth=self.hp['d'], learning_rate=self.hp['lr'], n_estimators=self.hp['n'], n_jobs=min(4, JOBS), random_state=0)

    def prepare(self, X, y, tr): self.ch = [c for c in range(X.shape[1]) if abs(corr(X[tr, c], y[tr])) > 0.6]

    def design(self, X, y, r): return np.column_stack([X[r][:, self.ch], y[r - self.lag]])


class B5(Base):
    """Yang et al. 2021 M-SVR: channels with 0.4 < |r| <= 0.9 to the target, redundant pairs (|r| > 0.95) pruned; RBF SVR on the
    3000 most recent released rows; C, gamma by random search (ant-lion optimiser replaced; documented deviation). No target history."""
    def make(self): return SVR(kernel='rbf', C=self.hp['C'], gamma=self.hp['g'], epsilon=0.05)

    def prepare(self, X, y, tr):
        r = {c: abs(corr(X[tr, c], y[tr])) for c in range(X.shape[1])}; cand = sorted([c for c in r if 0.4 < r[c] <= 0.9], key=lambda c: -r[c]); self.ch = []
        for c in cand:
            if all(abs(corr(X[tr, c], X[tr, d])) <= 0.95 for d in self.ch): self.ch.append(c)

    def fit(self, X, y, tr): super().fit(X, y, tr[np.isfinite(y[tr])][-3000:])

    def design(self, X, y, r): return X[r][:, self.ch]


GRIDS = {'B0': [{}], 'B1': [{'h': h} for h in (5, 10, 15, 20, 30)], 'B2': [{'h': h} for h in (3, 5, 8, 10, 15)], 'B2f': [{'h': h} for h in (3, 5, 8, 10, 15)],
         'B3': [{'d': d, 'lr': lr, 'n': n} for d, lr, n in itertools.product((3, 6, 9), (0.05, 0.1), (300, 600))],
         'B5': [{'C': float(c), 'g': float(g)} for c, g in zip(10 ** np.random.default_rng(0).uniform(-2, 3, 40), 10 ** np.random.default_rng(1).uniform(-2, 3, 40))]}


def released(OK, y, s): r = np.arange(7, s - DELAY); return r[OK[r] & np.isfinite(y[r])]


def run(name, hp, lag, ds, val=False):
    month, X, Y, OK = load(ds); n = len(month); P = {t: np.full(n, np.nan) for t in Y}; cls = globals()[name]
    wins = [(int(np.flatnonzero(month == 3)[0]), int(np.flatnonzero(month == 4)[0]))] if val else windows(month, n)
    for t in Y:
        for s, e in wins:
            m = cls(hp, lag); m.fit(X[t], Y[t], released(OK[t], Y[t], s)); r = np.arange(s, e); P[t][r] = m.predict(X[t], Y[t], r)
    if val:
        a, b = wins[0]; ev = np.arange(a, b)
        return float(np.mean([np.abs(Y[t][ev] - P[t][ev])[OK[t][ev] & np.isfinite(P[t][ev])].mean() for t in Y]))
    return score(Y, P, OK, month), P


if __name__ == '__main__':
    name, lag = sys.argv[1], int(sys.argv[2]); tag = f'{name}_lag{lag}'; f = OUT/f'{tag}.json'
    if f.exists(): raise FileExistsError(f)
    if len(sys.argv) > 3:  # resume after a crash: hyper-parameters already chosen (logged), finished datasets reloaded from npz
        hp = json.loads(sys.argv[3]); res = {'hp': hp, 'val_mae_2017Mar': 'see earlier log (resumed run)'}
    else:
        v = [run(name, hp, lag, 'kelmarsh2017', val=True) for hp in GRIDS[name]]
        for hp, m in zip(GRIDS[name], v): print('tune', hp, round(m, 4), flush=True)
        hp = GRIDS[name][int(np.argmin(v))]; res = {'hp': hp, 'val_mae_2017Mar': float(min(v))}; print('chosen', hp, flush=True)
    for ds in ['kelmarsh2017', 'kelmarsh2018']:
        fz = OUT/f'{tag}_{ds}.npz'
        if fz.exists():
            month, X, Y, OK = load(ds); z = np.load(fz, allow_pickle=False); P = {t: z[f'P_{t}'] for t in Y}; sc = score(Y, P, OK, month)
        else:
            sc, P = run(name, hp, lag, ds); np.savez_compressed(fz, **{f'P_{t}': p for t, p in P.items()})
        res[ds] = sc; print(tag, ds, round(sc['all'], 4), flush=True)
    res['code_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); f.write_text(json.dumps(res, indent=2))
