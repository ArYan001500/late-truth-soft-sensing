# Local late-truth correction over a long period (H26 + H27) — protocol v001

Registered 2026-10-13 before running. Author's hypotheses: H26 (learn from late truth via equations of similar situations), H27 (external factors such as time of day and ambient temperature define similarity). Exploratory screens on the 2019 dev week gave ~3% for the local variant and nothing for H27; that week lacks daily/seasonal variety.

## Data
Kelmarsh WT1–WT6, year 2017 only. Fit: first 28 days (rows 4..4031). Evaluation: rows 4032..end of 2017 (Feb–Dec, never used for fitting any model in this protocol). 2018 and 2019 are not touched. Synchronous clocks, lag 0, target released 3 steps (30 min) after measurement.

## Base model
Fleet HGB-large (max_iter 400, max_leaf_nodes 63, min_samples_leaf 20, other PARAMS as project) with turbine id; features: 9 helpers at t, their 1/2/3-step changes, oil(t-3), oil(t-3)-oil(t-4). Frozen after fitting.

## Corrections (all causal: only snapshots whose truth was released by the decision time)
- bias: EWMA of released residuals, rate 0.05.
- local_base: mean released residual of the k=20 nearest released snapshots (standardized 9 helper values), shrink k/(k+20).
- local_h27: base features + hour-of-day (sin, cos) + ambient temperature weighted x3.
- local_h27_dir: local_h27 + 3-step changes of all helpers.
- memory: last 2000 released snapshots (~14 days) and last 8640 (~60 days); both reported.
Standardization uses the first 200 evaluation snapshots' released statistics only after they are released (implemented as fit-period statistics to stay causal).

## Metrics and decision
MAE per turbine, averaged; also per month. H27 supported if local_h27 beats local_base in both memories and in at least 8 of 11 months (mean over turbines). Local correction supported if local_base beats frozen and bias overall and in at least 8 of 11 months. All results kept.

## Amendment 1 (2026-10-13, after v001 results, before any retraining run)
Conventional competitor required before any claim: periodic retraining of the same fleet HGB on released truths only (a row is usable at retrain time T only if its truth was released, i.e. row <= T-3, and rows 4..T-3 of all six turbines).
- retrain_monthly: retrain at the first row of each month Feb..Dec, expanding window from row 4.
- retrain_weekly60: retrain every 1008 rows (7 days), training window = last 8640 released rows (60 days).
- each retrain variant also with local_base correction (k=20, memory 8640) computed on its own residuals.
Same hyperparameters, no tuning. The local mechanism is claimed only if retrain+local beats the best retrain-only variant overall and in >= 8 of 11 months.
