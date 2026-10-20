# Comparison with published methods — protocol v001

Registered 2026-10-13, before any baseline result. Papers supplied by Aryan in `Ref Papers/` (method sections read; notes below are from the papers, deviations listed explicitly).

## Baselines (target = gear-oil temperature, 10-min data)
| id | paper | model as published | our implementation |
|---|---|---|---|
| B1 | Bangalore & Tjernberg 2015, IEEE TSG, 10.1109/TSG.2014.2386305 | NARX MLP, q=1 (y(t-1) + inputs at t), inputs power, gearbox oil temp, nacelle temp, rotor rpm, other gearbox-bearing temps; Levenberg–Marquardt; per turbine; trained on one healthy year | sklearn MLPRegressor, 1 hidden layer tanh, L-BFGS (LM not available), hidden size chosen from {5,10,15,20,30} on validation; per turbine |
| B2 | Schlechtingen & Santos 2011, MSSP, 10.1016/j.ymssp.2010.12.007 | autoregressive MLP (AR-NN): power, nacelle temp, generator speed, a correlated temperature, each shifted by its cross-correlation lag, + y(t-1); sigmoid, 2-layer, neurons by >= 10 runs; min–max scaling; 3 months training; per turbine | sklearn MLPRegressor logistic, lags by cross-correlation (0..6 steps) on the training window, neurons chosen by 10 runs over {3,5,8,10,15}; last 3 months of released rows; per turbine. Also the non-AR FSRC variant (B2f) |
| B3 | Liu, Wu, Wang 2020, IEEE Access, 10.1109/ACCESS.2020.3029435 | XGBoost NBM; inputs = SCADA channels with Pearson r > 0.6 to target + target of previous period; depth / learning rate / n trees tuned on validation | xgboost if installable, else sklearn GradientBoostingRegressor (exact-split GBDT); same selection rule |
| B4 | Fu, Chu, Guo, Chen 2019, IEEE Access, 10.1109/ACCESS.2019.2912621 | CNN + LSTM on 10 variables x 10 past steps; adaptive elastic-net variable selection | needs a deep-learning library (torch). Published Table 2 is internally inconsistent (11x11 kernel on 10x10 input giving 88x88x32; softmax output for regression) -> exact replication UNREACHABLE; we implement conv(3x3,16)–LSTM(32)–dense on the same 10x10 window and report the deviation |
| B5 | Yang, Liu, Xin, Wang 2021, Wind Energy, 10.1002/we.2604 | multi-input SVR (RBF) on n past steps of correlated channels (incl. gearbox HS shaft temp) for gear-oil temp; C, gamma by improved ant-lion optimisation (Ps 20, iters 500, ranges [0.01,1000]); 3000 training samples | sklearn SVR RBF; C, gamma by random search with the same ranges and budget 20x25 evaluations (ALO replaced — deviation); 3000 most recent released training rows per turbine |
| B6 | Kim, Okajima, Kano, Hasebe 2013, Chemom. Intell. Lab. Syst., 10.1016/j.chemolab.2013.03.008 | just-in-time LW-PLS with adaptive weighted-Euclidean similarity | LW-PLS with the paper's adaptive similarity; database = released rows within the last 8640 steps (same memory as our local correction); latent variables and bandwidth chosen on validation |

## Fairness rules (identical for all methods, including ours)
- Same data, turbines, months, quality mask (OK) and metric (MAE on Feb–Dec, mean over turbines; calm/transient split as before).
- Receipt-causal regime (main result): at decision time t the target is known only up to t-3. Every published "y(t-1)" input becomes y(t-3) (latest released value). All other channels as published (fresh at t).
- Synchronous regime (secondary, shows the cost of assuming fresh truth): y(t-1) given to baselines and to ours (oil(t-1) instead of oil(t-3)).
- All models retrained monthly on released rows only (same schedule as ours); hyper-parameters chosen once on Kelmarsh 2017 (train Jan–Feb, validate Mar; same data on which our pipeline was finalised), then frozen.
- Ours: h30_x+local (Senvion sites), base_x+local (Hill of Towie).

## Datasets (in this order)
1. Kelmarsh 2018 (unseen; primary table). 2. Penmanshiel 2019. 3. Hill of Towie 2019.

## Pre-declared claims
C1 (main): ours has lower MAE than every baseline in the receipt-causal regime on Kelmarsh 2018.
C2: same on at least 2 of the 3 datasets for every baseline.
C3: the published methods lose more accuracy than ours when moving from synchronous to receipt-causal (they rely more on fresh truth).
All results reported, including any baseline that beats ours.

## Amendment 1 (2026-10-14, before any baseline result) — recent methods from deep research
Source: Aryan's deep-research report `RECENT_BASELINES_DEEP_RESEARCH_REPORT_2026-10-13.md` (+ package `RECENT_BASELINES_RESEARCH_PACKAGE_2026-10-13.zip`).
Added baselines:
| id | paper | adaptation |
|---|---|---|
| B0 | persistence control: y_hat(t) = y(t-3) | none |
| B7 | Yan, Liu, Ren, Li 2023, Energies 16(19) 6786, 10.3390/en16196786 — hybrid attention + spatio-temporal BiConvLSTM for gear-oil temp; inputs incl. inlet-oil temp, oil-pump pressure, front/rear bearing temps (no previous target) | implement from the OA paper; same channels where they exist |
| B8 | Dai, Han, Bai, Kang, Liu 2025, Energies 18(5) 1273, 10.3390/en18051273 — GCN–TCN–LSTM, oil truth excluded | implement from the OA paper |
| B9 | Grataloup, Jonas, Meyer 2025, IEEE Access 13, 10.1109/ACCESS.2025.3599360 — fleet NBM, 24 h histories of wind speed, ambient temp, wind direction sin/cos; target-free; public code | port the public model; train-only scaling, causal filling |
| B10 | delayed-measurement Kalman fusion (textbook state augmentation for out-of-sequence measurements; stands in for Wang, Alipouri, Huang 2022, IEEE TIM 71, 10.1109/TIM.2021.3135537, whose full text is not yet available) | KF on the residual of a fleet model with 3-step delayed measurement update; if Aryan supplies the Wang 2022 PDF, implement that method instead and report both |
Pending full texts (requested from Aryan): Wang et al. 2022 (TIM), Bai et al. 2024 (Applied Energy 371, 123687, 10.1016/j.apenergy.2024.123687), Jian et al. 2026 (IEEE TASE 23, 523–537, 10.1109/TASE.2025.3639628).
Claims C1–C3 apply to the extended set unchanged.

## Amendment 2 (2026-10-14, before any baseline result)
Aryan supplied Wang et al. 2022 (Ref Papers/wang2022.pdf) and Bai et al. 2024 (Ref Papers/Wind-turbine-gearbox-oil-temperature-feature-extraction-and-condition-monitoring-based-on.pdf). Jian et al. 2026 promised later.
B10 becomes the Wang 2022 synchronized SNEKF adapted to our case: state = gear-oil temperature; state model = first-order thermal model fitted on released rows + neural compensation; fast-rate measurement = fleet soft-sensor output; slow-rate accurate measurement = released oil truth with 3-step delay, fused by the paper's synchronization (re-prediction over the delay window). Deviations (e.g. offline instead of online NN-parameter augmentation if the online version diverges) will be documented.
B11 = Bai 2024 energy-flow physical features (gearbox power loss + heat-transfer terms) fed to the four models of the paper (LSTM, GRU, TCN, XGBoost); gear geometry not available for these turbines -> loss terms with the paper's functional forms and coefficients fitted on training data (documented deviation).

## Amendment 3 (2026-10-14) — B10 tuning grid widened
First B10 tuning run (54-point grid) was stopped after a debug on Kelmarsh 2017 Feb–Mar (tuning data only, no 2018 result seen): with a random-walk state model the filter lags the fast measurement unless q >> r_fast, so the grid did not contain the method's best region. New grid: q {0.01,0.05,0.2,1,5} x r_fast {0.001,0.01,0.05,0.2} x r_slow {1e-4,1e-3,1e-2} x q_theta {1e-6,1e-5} (120 points), same validation split. This gives the competitor a fairer chance.

## Amendment 4 (2026-10-14, before any deep-baseline result) — deep baselines implementation (latetruth/baselines_deep.py)
- B4 Fu, B7 Yan, B8 Dai, B9 Grataloup are target-free as published -> one regime only (no lag-1/lag-3 split; C3 not applicable to them).
- Channel maps paper -> Kelmarsh (closest available): see CH in the code. Missing paper channels (torque, blade-motor temp, cooling water, hydraulic oil, cabinet temp, second generator bearing) replaced by the nearest available temperature/speed channel.
- Windows: Fu 10x10 (paper), Yan 6 (paper's stated window width), Dai 12 (paper's l not stated), Grataloup 144 (24 h, paper).
- Architectures: Fu conv(3x3,16)+pool -> FC100 -> LSTM32 (paper Table 2 inconsistent); Yan feature+time attention -> conv-gated BiLSTM (hid 32); Dai Table 5 (GCN 64/32 with |r|>0.3 normalised adjacency in place of static+MIC adjacency; TCN 64,64 k2 d1,2 dropout .05; LSTM 64; dense 64->1); Grataloup LSTM16-LSTM64-FC64-FC32 ReLU, Adam 8e-4, batch 256, local (non-federated) variant.
- Training: Adam (paper lr), first window <= 30 epochs + early stopping (patience 3, last 10 % of released rows), later windows warm-start fine-tuning 5 epochs (papers used 100–1000 epochs on a single split; monthly retraining x 11 windows makes that infeasible). Grataloup training samples strided by 3.
- Inputs causally forward-filled; standardised on the first training window.
- Final comparison will be computed on the common set of rows where every method has a prediction (coverage reported).

## Amendment 5 (2026-10-14, after the v1 deep results on Kelmarsh 2018 — disclosed)
Diagnosis: B8 Dai v1 (MAE 1.03) shows no bias but errors falling month by month (1.5 -> 0.7), i.e. under-training with the v1 budget (30 epochs) versus the paper's 100 epochs. To avoid an unfairly weak competitor, all four deep baselines are re-run with budget v2: first window <= 100 epochs with early stopping patience 10, later windows 10 fine-tuning epochs (DEEP_BUDGET=v2, outputs deepv2_*). Both v1 and v2 are reported; the comparison table uses the better of the two for each baseline.

## Amendment 6 (2026-10-14) — compute device
From now on deep baselines run on the RTX 4060 GPU (torch 2.14.1+cu132). Kelmarsh 2018 v1/v2 results were produced on CPU; GPU runs may differ slightly in floating-point order. Device is recorded per run.

## Amendment 7 (2026-10-14, before any B6/B11 full result) — Kim (B6) and Bai (B11) implementations
- B6 Kim 2013 (latetruth/baselines_kim.py): LW-PLS with adaptive weighted-Euclidean similarity (eqs. 3-16, 20-23). Database per query = released rows in [t-3-8640, t-3) of that turbine (JIT that also uses late truth, same memory as ours). Offline theta on 1000 random released rows per monthly window (full N-model offline part infeasible), I=5; online J=3. Inputs: channels >= 95 % present in the training window (pressures are missing before June 2017), autoscaled, + y(t-lag) for lag 3 / lag 1. Grid R {2,4,6} x phi {0.5,1,2,4} x gamma {0,0.5,1} on Kelmarsh 2017 March, every 6th row.
- B11 Bai 2024 (latetruth/baselines_bai.py): their final energy-flow feature set mapped to Kelmarsh (gearbox bearing temps + 1-step rates, nacelle temp, rotor/gearbox/generator speeds, wind speed, pitch A, yaw error = wind direction - nacelle position, tip-speed ratio with R = 46 m); target-free; XGBoost (one of their four models), B3 grid. Gear geometry needed for their analytic loss terms is not available, so the loss terms enter through these measured drivers rather than computed watts.
- B9 implementation check vs the authors' own code/data split: see REPORT_FA.md (our port within ±4 % of their exact recipe on identical samples).

## Amendment 8 (2026-10-14, before any Penmanshiel / Hill of Towie baseline result) — claim C2 runs
- Datasets: Penmanshiel 2019 (14 Senvion MM82, same channel names as Kelmarsh) and Hill of Towie 2019 (21 Siemens SWT-2.3; 16-slot map in latetruth/baselines_hot.py; missing slots = NaN: gear-oil inlet temp, vane, inlet pressure, wind direction).
- Ours: Penmanshiel h30_x + local (as Kelmarsh); Hill of Towie base_x + local (as HILL_OF_TOWIE_REPLICATION_v001) but with EV0 = 4032 like every comparison here.
- Every baseline hyper-parameter frozen at its Kelmarsh-2017 choice (no re-tuning); receipt-causal regime only (lag 3); deep baselines budget v2 on GPU; Wang frozen-NN variant.
- Strictly one job at a time (latetruth/run_new_datasets.py). Same common-row scoring (latetruth/baselines_table.py extended to the new datasets).

## Amendment 9 (2026-10-14, before any Penmanshiel / Hill of Towie baseline result)
Compute only: CPU baselines now run turbine-parallel (6 worker processes) and the deep baselines in a separate GPU lane (latetruth/deep_lane.py). Kim's offline subsample uses a per-turbine seed (default_rng(turbine)) instead of one sequential generator. Methods and hyper-parameters unchanged.

## Amendment 10 (2026-10-14) — B11 on a farm without a channel
Bai's yaw-error feature needs wind direction, which Hill of Towie does not record; the first HoT run produced no prediction (rows with any NaN feature were dropped). B11 now keeps only feature channels >= 95 % present in the training window. On Kelmarsh and Penmanshiel every channel is >= 99 % present, so their B11 results are unchanged.

## Amendment 11 (2026-10-15, registered before any of these results) — robustness package
S1 Significance: for each dataset (Kelmarsh 2018, Penmanshiel 2019, Hill of Towie 2019), unit = turbine x month MAE on the common rows (Feb–Dec). Ours vs each baseline: two-sided Wilcoxon signed-rank test, Holm correction over the baselines within a dataset; plus a turbine-cluster bootstrap (2000 resamples of turbines) 95 % CI of the MAE difference. Rule: "ours better" is claimed for a baseline only if Holm p < 0.05 AND the CI excludes 0.
S2 Independent verification: latetruth/verify_final_table.py recomputes every MAE in FINAL_TABLE_all.json directly from the saved prediction files with its own code (no shared scoring function); pass = |difference| < 1e-9 for all entries.
S3 Delay sensitivity (Kelmarsh 2018): release delay d in {1, 3, 6, 12} steps (10, 30, 60, 120 min). Ours (fleet model trained with oil(t-d) features, released rows < start-d, local correction with delay d); Wang 2022 frozen (fast sensor = our fleet model for that d, slow measurement delay d); Kim 2013 (lag d, database ends at t-d); Bangalore 2015 (lag d); Yan 2023 shown as the delay-independent target-free reference. Hyper-parameters frozen. Reported as curves; no pass/fail rule.
S4 Local-correction sensitivity (Kelmarsh 2018, saved fleet predictions): k in {5, 10, 20, 40, 80} x memory in {1440, 4320, 8640, 17280}; shrink k0 = 20 fixed. Reported as a grid; the pre-chosen setting (k 20, memory 8640) is not re-selected.
S5 Compute cost: wall-clock fit + predict time per turbine for one monthly window (Kelmarsh 2018, window starting 1 June), same machine, each method alone.
S6 Optional extra baseline Cao et al. 2026 (MSSP 260, 115021, RESCINet) if implementable from the paper; otherwise reported as UNREACHABLE.

S6 implementation note (2026-10-15, before results): latetruth/baselines_cao.py — RESCINet lite from the paper text (121k parameters vs 402k reported; nowcast instead of 6-step forecast; target history visible to t-3; Kalman pre-filter, |r|>=0.4 screening, 3 levels, hidden 16, lr 1e-5, batch 32). Kelmarsh 2018 only.
S6 update (2026-10-15, before any Cao result): Aryan asked for maximum fidelity to the paper. First-window training changed to exactly 100 epochs without early stopping (paper). Remaining unavoidable differences, to be stated in the paper: nowcast of y(t) instead of a 6-step forecast (our task), layer internals not fully specified in the paper and no public code (121k vs 402k parameters), monthly fine-tuning (shared protocol for all methods). Runs on all three farms.

## Amendment 12 (2026-10-15 09:30, after the v1 Cao result on Kelmarsh 2018 — disclosed)
Cao v1 (exact paper settings, lr 1e-5) on Kelmarsh 2018: MAE 1.165, monthly error falling 1.95 -> 0.72 = under-training under the monthly protocol (same pattern as Dai v1, Amendment 5). Cost: about 10 h per turbine on the shared GPU, so the full three-farm run would take 2–3 more days.
Decision (Aryan, 2026-10-15): stop the v1 lanes after the first Penmanshiel batch (turbines 1, 2, 4, 5, 6, 7) finishes; run v2 = identical to v1 except Adam lr 1e-3 (Adam default), CAO_BUDGET=v2, outputs caov2_*. v2 on Kelmarsh 2018 (6 turbines). Both v1 and v2 are reported; Cao's main row is Kelmarsh 2018; the partial Penmanshiel v1 (6 of 14 turbines) is reported as partial. lr fixed now, before any v2 result; no tuning.
Amendment 12 addendum (2026-10-15, before any v2 result, Aryan agreed): v2 is also run on the six Penmanshiel turbines of the finished v1 batch (1, 2, 4, 5, 6, 7), each lane running its Kelmarsh 2018 turbine first. Cao is then reported on Kelmarsh 2018 (6 turbines) and Penmanshiel 2019 (6 of 14 turbines), v1 and v2.

## Amendment 13 (2026-10-15, registered before running; exploratory diagnostic for the Discussion, no success rule)
latetruth/discussion_recency_vs_similarity.py: on the saved fleet predictions of kelmarsh2017/2018, penmanshiel2019, hot2019, compare the proposed correction (20 most similar released states, shrink 1/2) with the same correction built from the 20 most recent released residuals, and with 1/2 times the newest released residual. Reported as exploratory in the Discussion, whatever the outcome.
latetruth/filter_index.py: physics-based filter-condition index on Hill of Towie 2019, D_i(t) = ln dP_f - ln Q - ln nu(T) relative to each turbine's January median (laminar Darcy law with Walther viscosity, exponents fixed at one); weekly medians; report for each turbine the week of the largest weekly drop and its size, and the index trend between drops. Exploratory; no maintenance log is available.

## Amendment 14 (2026-10-16, registered before any gated result) — standstill gate (disclosed, motivated by a held-out result)
Trigger (disclosed): on Kelmarsh 2024 the S1 rule did not show ours better than Kim 2013 (CI [-0.069, +0.129]); turbine 4 was stopped ~7.5 days in February (oil down to 9.5 C) and our error there was 4.28 vs Kim 0.25. A tabulation already seen on kelmarsh2017/2018/2024 and penmanshiel2019/2022 (rows after >= 24 h of rotor speed < 1 RPM; 0.14-2.3 % of rows) gave ours 0.68-2.81 vs Kim 0.13-0.47 on those rows, and ours better than Kim on all other rows. Author's hypothesis H41: a stopped turbine must be declared to the model so it cannot corrupt the estimate.
Design (frozen here): standstill duration s(t) = number of consecutive rows up to t with rotor speed < 1 RPM (missing rotor speed = not stopped), a real-time channel. If s(t) >= h, the estimate is replaced by a standstill estimate; otherwise ours is unchanged. Two forms: (H) hold = last released value y(t-3) (= B0 prediction); (N) Newton cooling = T_nac(t) + (y(t-3) - T_nac(t)) * exp(-3/tau), T_nac = nacelle temperature, tau (rows) fitted by least squares on the January 2017 standstill rows of Kelmarsh. Form and h in {3, 6, 12, 18, 36, 72, 144} rows are chosen ONLY on Kelmarsh 2017 (development year, evaluation rows) by overall MAE; then frozen and applied unchanged to every other dataset. If no (form, h) improves Kelmarsh 2017, the gate is dropped and this is reported.
Reporting: the frozen pipeline stays the primary row of every table; "ours + standstill gate" is an added row in every dataset, scored on the same common rows and tested with the S1 rule. Whatever the outcome on held-out data is reported, including any dataset where the gate hurts.
Amendment 14 outcome (2026-10-16): chosen on Kelmarsh 2017 = hold, h = 72 rows (12 h), MAE 0.1572 -> 0.1547; tau fit 46.3 rows not used. Held-out results in GATE.json.

## Amendment 15 (2026-10-16, before any d = 24 result) — longer delay in S3
Motivation (literature, not results): Schlechtingen & Santos (Offshore 2011, publ. 2012) state that SCADA data are often submitted with a delay, "e.g. 4 h" (to be checked against the PDF before citing). S3 is extended with d = 24 steps (4 h) on Kelmarsh 2018 for the same four methods, same code, same frozen hyper-parameters (delay_part1.py, delay_part2.py). Reported as one more point on the curve; no pass/fail rule.

## S5 implementation note (2026-10-16, before any timing result)
latetruth/s5_timing.py, Kelmarsh 2018, window starting 1 June, turbine 1, machine otherwise idle, one method at a time, each method's own code
with its window list replaced by the single June window (so each "fit" is a full refit on all rows released before 1 June, then predict June):
classic B0-B5, Kim, Bai (CPU); Fu, Yan, Dai, Grataloup v2 and Cao v2 (GPU; first-window training budget = full training).
Ours: fleet model fit on the six turbines pooled (its design) + June prediction, then the local correction for the June rows of turbine 1
(timed as local() over [1 June - memory, end of June] minus local() over [1 June - memory, 1 June]). Wang (frozen): fusion over the full year
for turbine 1 (it has no refit). Wall-clock seconds, 1 repeat (2 for methods under 60 s, median). Reported as a table; no pass/fail rule.
S5 note addendum (2026-10-16, before the Cao timing): Cao v2 is timed with 5 epochs and multiplied by 20 (the 100-epoch first-window budget would take ~5 h alone on the GPU, from the earlier runs); stated as extrapolated in the table.
S5 note addendum 2 (2026-10-16): the 5-epoch run took 89 s, so the full 100-epoch Cao v2 timing (~30 min) is affordable; it is run in full and replaces the extrapolated value (both kept in S5_TIMING.json).
S5 note addendum 3 (2026-10-16, before the result): the local correction is also timed over the full evaluation year of turbine 1, the same span as the Wang fusion timing, so that the two method-specific parts are compared on equal terms.

## Amendment 13 addendum (2026-10-16, before looking at the shutdown data) — filter index vs service stops
Hill of Towie 2019, ShutdownDuration.csv (operator file, no cause given). For each turbine, weekly total shutdown duration (weeks aligned
with filter_index.py). Hit = the week of the largest filter-index drop, or the week before it, is among the 5 weeks with the longest
shutdown of that turbine in 2019. Chance of a hit ~ 1 - C(50,5)/C(52,5) ~ 0.18. Positive (report in the Discussion) if hits >= 10 of 21 and
one-sided binomial p < 0.01; otherwise the result is not used in the paper (Aryan's decision) but kept here.
Amendment 13 addendum outcome (2026-10-16): hits 7/21 (chance 0.185), one-sided binomial p = 0.077 -> not positive; not used in the paper.

## Amendment 16 (2026-10-16, after a code review, before any re-run result) — two defects found
(a) latetruth/baselines_common.py ours_sync (used by the S3 delay curves for ours and Wang, and by claim C3): the row mask tested the
finiteness of the target d steps in the FUTURE (Y[r+d]) instead of the released lagged target (Y[r-d]); it selected training and
predicted rows on future missingness (not future values). Fixed to Y[r-d]. All ours/Wang delay runs (d = 1, 3, 6, 12, 24) and
ours_sync_kelmarsh2018 are recomputed; old files kept in _superseded_2026-10-16/. Kim and Bangalore delay runs do not use this code.
(b) B10f Wang frozen: the measurement-compensation network has no fast estimates to fit in the initial window (the fleet model
issues estimates only from tau_1), so it is zero by construction (fast estimate treated as unbiased). No re-run; the text will
state this. The main comparison (d = 3) does not use ours_sync and is unaffected.

## Amendment 17 (2026-10-17, after the review reports, before any of these results) — residual references and dependence-aware intervals
Motivation: five review reports (2026-10-17) ask that the simple recency correction of Amendment 13 appear in the main
comparison on all six held-out farm-years, with an EWMA reference, and that the turbine bootstrap be complemented by a resampling
that keeps the dependence between turbines. No success rule; every number is reported whatever its sign.
(a) Residual references (latetruth/residual_references.py), same saved fleet predictions (ours_<ds>.npz P_fleet), same released-row rule
(residual of row s usable at t only if s < t - d, d = 3), same 8640-row memory where applicable, residuals collected from EV0:
  mean20 : fleet + 20/40 * mean of the 20 newest released residuals (identical to last20 of Amendment 13; nothing tuned).
  ewma   : fleet + c * EWMA of the released residuals; half-life h in {6, 18, 36, 72, 144, 288, 576, 1152, 2304} rows and c in
           {0.5, 0.75, 1.0} chosen by MAE on the development year Kelmarsh 2017 ONLY (rows >= EV0), then frozen; starts after 20
           released residuals (as mean20).
  Scored on the common rows of Table 7 (rows where every compared method is finite). Same S1 machinery as significance.py:
  turbine-month Wilcoxon (Feb-Dec) + Holm within farm-year (over the Table-8 baselines plus the two references), turbine bootstrap
  2000. Reported as extra rows of Table 7 / Table 8 and in the Discussion. The proposed estimator is not changed.
(b) Dependence-aware sensitivity (latetruth/block_bootstrap.py): for ours vs every Table-8 baseline and the two references, paired
  bootstrap of whole calendar blocks of the evaluation period, the same blocks for all turbines (keeps cross-turbine dependence and
  within-block autocorrelation); blocks of 7 days (primary) and 28 days (sensitivity); 2000 resamples; 95 % percentile CI of the
  fleet-mean MAE difference. Conditional on the fitted models (no refitting). Reported in a table / text; the original rule is not
  replaced.
Amendment 17 outcome (2026-10-17): EWMA tuned on Kelmarsh 2017: half-life 18 rows (3 h), c = 0.75. Common-row MAE (ours / mean20 / ewma):
K18 0.1715/0.1734/0.1748, P19 0.1492/0.1507/0.1505, H19 0.1952/0.1957/0.1907, K24 0.2521/0.2515/0.2259, P22 0.1703/0.1717/0.1720,
H25 0.1865/0.1879/0.1853. Under the S1 rule ours is better than mean20 on K18, P19, H25 and than ewma on K18, P19; not shown elsewhere; ewma
lower than ours on H19, K24, H25 (H25 turbine-bootstrap CI excludes zero). Diagnostic (not registered, reported as such): outside the
standstill rows of Amendment 14, ours is lower than ewma on all six (H19 0.1834 vs 0.1839, H25 0.1820 vs 0.1821). Block bootstrap (b):
every Table-8 baseline keeps a CI above zero with 7- and 28-day blocks except Kim on H19 and K24. RESIDUAL_REFERENCES.json, BLOCK_BOOTSTRAP.json.

## Amendment 18 (2026-10-17, after Amendment 17 results, before any held-out result of this variant) — recency on top of similarity
Motivated by Amendment 17 (disclosed; post hoc, exploratory). Variant "ours+rec": P = P_ours + c * EWMA(h) of the released residuals of
P_ours itself (same released rule s < t - d, start after 20 released residuals). Grid h in {6..2304} (as Am. 17), c in {0.25, 0.5, 0.75, 1}
chosen on Kelmarsh 2017 ONLY: h = 18 rows, c = 0.5 (dev MAE 0.1572 -> 0.1554). Frozen and applied unchanged to the six held-out
farm-years; also with the Amendment 14 gate on top (hold rows replaced by the gate). Reported against ours, ref ewma, gate with the S1
statistics whatever the outcome; the frozen primary estimator of the paper is NOT replaced by this result.
Amendment 18 outcome (2026-10-17): common-row MAE ours / ours+rec / ref ewma / gate / ours+rec+gate: K18 .1715/.1698/.1748/.1699/.1688,
P19 .1492/.1474/.1505/.1469/.1460, H19 .1952/.1866/.1907/.1836/.1802, K24 .2521/.2203/.2259/.1912/.1881, P22 .1703/.1668/.1720/.1608/.1599,
H25 .1865/.1809/.1853/.1821/.1779. ours+rec lower than ours and than ref ewma on all six, turbine-bootstrap CI above zero in all 12
comparisons; with the gate, lower than gate on all six (CI above zero). RECENCY_ON_SIMILARITY.json.

## Amendment 19 (2026-10-17, after CONFIRM_TWO_TIMESCALE_v001 outcome, before any of these results) — ours+rec against the published baselines
(a) Six farm-years of Table 7: ours+rec (rec_<ds>.npz, Am. 18) and ours+rec+gate vs every Table-8 baseline,
same S1 rule (turbine-month Wilcoxon + Holm within farm-year, turbine bootstrap 2000) and the 7/28-day block bootstrap; reported whatever
the outcome. (b) The three confirmation years (Kelmarsh 2020, Penmanshiel 2020, Hill of Towie 2022): all CPU baselines (B0, B1, B2, B2f,
B3, B5, B6, B11, B10f) and the GPU deep baselines (B4, B7, B8, B9, budget v2) with hyper-parameters frozen at their Kelmarsh 2017 choice,
same code; Cao not run (cost). Then the same S1 + block comparison of ours, ours+rec and ours+rec+gate vs every baseline.
Amendment 19 (a) outcome (2026-10-17): on all six Table-7 farm-years ours+rec (no gate) is better than every Table-8 baseline and than
the frozen ours under the S1 rule AND with 7-day blocks (incl. Kim on Kelmarsh 2024 and Hill of Towie 2019, the two exceptions of the
frozen estimator); same for ours+rec+gate. MAE ours+rec: K18 .1698, P19 .1474, H19 .1866, K24 .2203, P22 .1668, H25 .1809.
REC_VS_BASELINES_table7.json. Part (b) (three confirmation years) running.
Amendment 19 (b) outcome (2026-10-17): all CPU and GPU baselines ran on Kelmarsh 2020, Penmanshiel 2020, Hill of Towie 2022 (deep code
patched only to skip a prediction window without valid rows: HoT 2022 T6, February). ours+rec better than every baseline (13) and than ours
on all three, S1 rule and 7-day blocks; also ours alone better than all 13. MAE ours+rec: K20 .1803, P20 .1606, H22 .1853; closest
published baseline Wang (frozen) .1978/.1767/.2015. REC_VS_BASELINES_kelmarsh2020_penmanshiel2020.json, REC_VS_BASELINES_hot2022.json.

## Amendment 20 (2026-10-17, reporting after Amendments 17-19 and CONFIRM_TWO_TIMESCALE_v001) — paper revision
The paper's proposed estimator is now ours+rec (similarity term + recency term, Am. 18); the
similarity-only estimator is reported as the registered primary of stages 2-3. Derived computations for the text, no new choices:
FINAL_TABLE_all.json over 10 farm-years with the extra rows (extras do not shape the common rows; old values unchanged);
REC_VS_BASELINES_all9.json (S1 + 7/28-day blocks, 9 farm-years); DELAY_SENSITIVITY_rec.json (recency term on the saved delay
runs: helps at d <= 12, +1.2 % at d = 24); DESIGN_LADDER_L6_kelmarsh2017.json; CAO_TABLE_PROPOSED.json; STANDSTILL_DIAG.json
(+ rec, recg, + confirmation years); DATASET_STATS.json (+ 3 years); VERIFY_FINAL_TABLE.json (210 entries + own re-implementation
of the recency term, max |diff| 0); S5 recency timing 0.06 s per turbine-year. HoT 2022 T6 (no February rows): reader and deep
code skip empty windows (no other effect).

## Amendment 21 (2026-10-17, before any result of it) — matched-input per-turbine control (review item F02)
Question: how much of the advantage comes from pooling the fleet, with inputs and late-truth correction held fixed?
Control "per-turbine": the same HistGradientBoosting (PARAMS), the same features (h30_x Senvion / base_x Siemens), windows, released-row
rule and evaluation, but one model per turbine fitted on that turbine's own released rows (no turbine index); then the identical
similarity term (local(), k 20, k0 20, M 8640) and recency term (h 18, c 1/2). Farm-years: K17 (descriptive) + the nine held-out.
Files ptfleet_<ds>.npz (model alone) and ptrec_<ds>.npz (with both terms). Reported: MAE on the Table-7 common rows; proposed vs
per-turbine+correction and fleet alone vs per-turbine alone, S1 rule + 7/28-day blocks. No success rule; reported whatever the outcome.
Nothing is tuned. latetruth/per_turbine_matched.py, latetruth/compare_per_turbine.py.
Amendment 21 outcome (2026-10-17): pooled vs per-turbine with identical inputs and late-truth terms, common-row MAE (proposed / per-turbine
+ terms): K18 .1698/.1985, P19 .1474/.1496, H19 .1866/.2065, K24 .2203/.2691, P22 .1668/.1989, H25 .1809/.2008, K20 .1803/.2106,
P20 .1606/.1818, H22 .1853/.2292 (K17 .1551/.1709). Pooling better under the S1 rule and 7-day blocks on 8 of 9 held-out farm-years; not
shown on Penmanshiel 2019 (1.4 %, CI [-0.0027, 0.0079], 65/154 turbine-months). Without the terms: fleet vs per-turbine alone 5.8-36.5 %,
better on 8 of 9 (P19 not shown). PER_TURBINE_CONTROL.json.
