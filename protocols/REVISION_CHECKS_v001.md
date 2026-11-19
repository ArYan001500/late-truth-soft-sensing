# REVISION_CHECKS_v001: additional analyses requested in the review of Access-2026-54204

Registered 2026-11-17, before any of the runs below. Each part is added before its own runs and states the data, the
pipeline, the outputs, and the expectation. All parts are descriptive robustness checks; none changes the registered
results of the earlier protocols. Every outcome will be reported, whatever its sign.

Common settings (all parts unless stated): the pipeline of the submitted article, unchanged: fleet model (histogram
gradient boosting, 400 iterations, learning rate 0.1, 63 leaves, 20 rows per leaf, L2 10, early stopping off, seed
20261012), features `h30_x` (Senvion) and `base_x` (Siemens), retraining windows of Section III, similarity term
(k = 20, k0 = 20, M = 8640) and recency term (h = 18, c = 0.5), release delay d = 3. Runtime: Python 3.12.14,
scikit-learn 1.9.1. Score: MAE of eq. (4) on the rows where the estimator returns an estimate (OK rows, row >= EV0),
mean over turbines, for the fleet model alone and for the proposed estimator (similarity + recency).
Reproduction check first: the unchanged pipeline rerun must reproduce the saved estimates of each farm-year
(`ours_<ds>.npz`, `rec_<ds>.npz`) to within 1e-9 degC; if it does not, stop and report.

## Part A (reviewer 1, concern 2): encoding of the turbine index

The submitted pipeline enters the turbine index i as a numeric column (values 1..N).

Data: the nine held-out farm-years (Kelmarsh 2018, 2024, 2020; Penmanshiel 2019, 2022, 2020; Hill of Towie 2019, 2025,
2022).

Variants, each with the full pipeline (fleet model, similarity term, recency term):
1. `orig`: numeric index as submitted (reproduction).
2. `perm1`..`perm5`: the turbine numbers are relabelled by a random permutation before they enter the model
   (numpy default_rng seeds 1..5; one permutation per seed, the same for all windows of a farm-year).
3. `cat`: the index enters as a categorical feature (`categorical_features` of the same estimator).
4. `noid`: the index column is removed.

Outputs: MAE of the fleet model and of the proposed estimator per variant and farm-year; for 2: the range over the
five permutations; differences against `orig`.

Expectation: no variant changes the MAE of the proposed estimator by more than 2 % on any farm-year.

## Part B (reviewer 1, concern 2): a newly added turbine with no history

Data: Kelmarsh 2018, Penmanshiel 2019, Hill of Towie 2019 (the stage-2 farm-years).

Procedure, for each turbine j of the farm-year in turn: turbine j is treated as newly added at tau_1. Its rows before
tau_1 (the initial window) are removed from all training sets, so the first fleet model has never seen it; from the
second window on, its released rows enter the refits like those of every other turbine. The standardization of its
driver channels for the similarity term uses the mean and standard deviation of the initial windows of the other
turbines. Its delayed target y_j(t-d) and its own released residuals are used as soon as they are released. The other
turbines are unchanged. Index encoding: numeric, as submitted.

Outputs: per turbine j, the MAE of the fleet model and of the proposed estimator on turbine j, in the new-turbine run
and in `orig`; the farm-year mean over j; the same for the first window (tau_1 to 1 March) alone.

Expectation: the annual MAE of the proposed estimator on a new turbine is within 5 % of its MAE in `orig`, with the
difference concentrated in the first window.

Code: `latetruth/revision_turbine_id.py`; results in
`results/revision_checks_2026-11-17_v001/` (`turbine_id_<ds>.json`, `new_turbine_<ds>.json`).

## Part C (reviewer 2, concern 1): a recursive parameter update as a further delayed-measurement reference

Registered 2026-11-17, before the run. The comparison of the article already contains corrections that use each released
target as it arrives: a Kalman update of a state (Wang et al.), and a bias update by an exponentially weighted or a plain
mean of the released residuals. Part C adds the third common mechanism, a recursive update of model parameters
(adaptive soft sensor, recursive least squares with a forgetting factor).

Method: on top of the saved fleet-model estimate f_i(t) of each turbine (the same starting estimate as the other
corrections), the residual is modelled as r_i(s) = b + a (f_i(s) - mu_i), with mu_i the mean of f_i over the initial
window (rows 4 .. EV0-1). The parameters (b, a) start at zero with covariance 100 I and are updated by recursive least
squares with forgetting factor lambda = 0.5^(1/h) on every released pair, in time order. Release rule as for the EWMA
reference: the estimate at row t uses only pairs with s < t - d, and the correction is zero until more than 20 pairs are
released. Estimate: f_i(t) + c (b + a (f_i(t) - mu_i)).
Tuning: (h, c) on Kelmarsh 2017 only, on the grid of the EWMA reference (h in 6, 18, 36, 72, 144, 288, 576, 1152, 2304
rows; c in 0.5, 0.75, 1), by the mean MAE over turbines; then frozen.

Data: the nine held-out farm-years. Score: MAE on the common rows of the comparison table (the row set of
`baselines_table.py`, which the new method does not shape), mean over turbines; reported with the fleet model, the
proposed estimator, the EWMA reference, and the Kalman filter with fixed networks on the same rows; turbine-month wins of
the proposed estimator against the new reference.

Expectation: the recursive update lowers the error of the fleet model by less than the proposed estimator on every
farm-year.

Code: `latetruth/revision_rls.py`; results in `results/revision_checks_2026-11-17_v001/rls_reference.json`.

Amendment C1 (2026-11-17, after a first run that returned no finite estimate and therefore no result): the fleet-model
estimate does not exist on the initial window, so mu_i cannot be its mean there. mu_i is instead the mean of the target
y_i over the initial window (rows 4 .. EV0-1), whose values are all released at EV0. Nothing else changes.

## Part D (reviewer 2, concern 5): practical size of the recency gain

Registered 2026-11-17, before the run. Descriptive analyses of saved estimates only (no refit): the fleet model
(`P_fleet`), the similarity-only estimator (`P_ours`), and the proposed estimator (`rec`), on the nine held-out
farm-years, rows >= EV0 where the target is valid and all three estimates are finite, per turbine and then averaged.
D1. Calm and transient rows as defined in Section III (30-min change of the target below the turbine median, and at or
above its 90th percentile): MAE of the three estimators and the change due to the recency term.
D2. Size in practice: the share of rows with an absolute error above 0.5 and above 1 degC, the 95th and 99th
percentiles of the absolute error, and the distribution over turbine-months of the MAE change due to the recency term
(share of turbine-months with a gain, largest gain, largest loss).
D3. Recording resolution of the target on each farm-year: share of values that are whole degrees, and the median
absolute 10-min change.
No expectation is stated; all values are reported.
Code: `latetruth/revision_recency_size.py`; results in `results/revision_checks_2026-11-17_v001/recency_size.json`.

Amendment D-a (2026-11-17, after the Part D run, as a further descriptive cut, no expectation): the turbine-months are
split by the MAE of the similarity-only estimator; for the 5 % of turbine-months with the largest such error and for the
other 95 %, the MAE change due to the recency term (pooled over turbine-months of each farm-year). Also the share of the
total absolute gain of each farm-year that comes from the top 5 %.

## Part E (reviewer 2, concern 5): release delays on two further farms

Registered 2026-11-17, before the run. The delay study of the article (Section VI-F) used Kelmarsh 2018 only. Part E
repeats it for the proposed estimator on Penmanshiel 2019 (Senvion) and Hill of Towie 2019 (Siemens), the other two
stage-2 farm-years, for d in {1, 3, 6, 12, 24} steps (10 min to 4 h).
Procedure, as in the Kelmarsh study: the full pipeline is rerun for each d; training rows end at s - d; the target lags
among the inputs are read at t - d and t - d - 1 (the target column handed to the unchanged feature code is shifted by
3 - d steps); a row is valid if it is valid in the main run (y(t), y(t-3), y(t-4) finite) and y(t-d) is finite, the
rule of the Kelmarsh study (`baselines_common.ours_sync`; corrected before the run); the similarity and recency terms use the
residuals released by t - d, with all settings unchanged (k = 20, k0 = 20, M = 8640, h = 18, c = 0.5).
Check: the run with d = 3 must reproduce the saved estimates of the main run to within 1e-9 degC.
Outputs: MAE (own rows of each run, metric of Section VI-F) of the fleet model, the similarity-only estimator, and the
proposed estimator for each d; the change due to the similarity term and due to the recency term.
No expectation is stated. All results are reported in the revised article, whatever their sign.
Code: `latetruth/revision_delay.py`; results in `results/revision_checks_2026-11-17_v001/delay_<ds>_d<d>.json`.

## Part F (reviewer 2, concern 3): a wind farm that the model has never seen, without the initial adaptation

Registered 2026-11-17, before the run. Transfer between the two Senvion farms, which record the same channels:
Kelmarsh 2017 (development year) -> Penmanshiel 2019, and Penmanshiel 2019 -> Kelmarsh 2018. The Siemens farm records
other channels (no oil inlet temperature) and is not part of a transfer.
Common: features `h30_x`, fleet-model settings unchanged, the turbine index removed (its values do not carry over
between farms). The source model is fitted once on all valid rows of all source turbines. The driver standardization of
the similarity term uses the mean and standard deviation of the initial windows (rows 4 .. EV0-1) of the source
turbines, so that no statistic of the target farm enters. On the target farm, the estimator starts at row 7 (day 1).
Variants on the target farm:
(a) zero-shot: the source model alone; no value of the target farm is used.
(b) source model + similarity and recency terms on the released residuals of the target farm (settings unchanged); no
    training on the target farm.
(c) start from day 1: the source model up to EV0; from EV0 on, the usual refits at the window starts on all source rows
    plus the released target-farm rows (rows < s - d); similarity and recency terms as in (b).
References: the unchanged pipeline without the turbine index (`noid` of Part A, rerun here to give per-row estimates;
initial 28-day window), the submitted pipeline (`orig`), and, for days 1-28, persistence y(t-d).
Scores (MAE, valid rows, mean over turbines): rows 7 .. EV0-1 (days 1-28) and rows >= EV0; turbine-month wins of (b)
and (c) against `noid` and `orig` on rows >= EV0.
No expectation is stated. All results are reported, whatever their sign.
Code: `latetruth/revision_transfer.py`; results in `results/revision_checks_2026-11-17_v001/transfer_<src>_to_<tgt>.json`.

Amendment D-b (2026-11-17, design audit before any text uses D-a): selecting turbine-months by the error of the
similarity-only estimator can inflate its difference from the proposed estimator (selection on one of the two compared
quantities). D-a is therefore repeated with the selection made by the error of the fleet model, which both estimators
correct, and by the error of the proposed estimator itself (conservative). The text uses the conservative selection.

## Part G (reviewer 1, concern 4; reviewer 2, concern 4): controlled comparison of inputs, pooling, and correction

Registered 2026-11-17, before the run. Purpose: to separate the contributions of the input information, the pooling of
the fleet data, and the late-truth correction, for several model architectures under one update protocol.
Data: Kelmarsh 2018, Penmanshiel 2019, Hill of Towie 2019 (stage 2).
Factors (full factorial, 4 x 2 x 2 x 2 = 32 estimators per farm-year):
1. Architecture: (i) histogram gradient boosting with the settings of the article; (ii) XGBoost 3.4.1, hist, with the
   matching settings (400 trees, learning rate 0.1, at most 63 leaves with loss-guided growth, min_child_weight 20,
   reg_lambda 10, 255 bins, seed 20261012); (iii) ridge regression (alpha 1); (iv) a multilayer perceptron (two hidden
   layers of 64 ReLU units, Adam, learning rate 1e-3, batch 1024, 20 epochs, squared error, seed 20261117). Settings of
   (ii)-(iv) are fixed here a priori and are not tuned. For (iii) and (iv), the inputs are standardized with the mean and
   standard deviation of each training set, missing values are set to zero after standardization and flagged by one
   indicator column per input that has missing values in that training set, and the target is standardized for (iv).
2. Inputs: `base` (the operating channels, their 10- to 30-min changes, and the two released target lags; design-ladder
   level L1/L2) or the full inputs of the article (`h30_x` for Senvion, `base_x` for Siemens).
3. Pooling: one fleet model with the turbine index (numeric for (i) and (ii), as in the article; one-hot for (iii) and
   (iv)), or one model per turbine on its own rows.
4. Correction: none, or the similarity and recency terms of the article with their settings unchanged.
Update protocol for all: refit at the window starts of Section III on released rows only (rows < s - d, d = 3).
Checks: (i) with full inputs and fleet pooling must reproduce the saved fleet and proposed estimates (<= 1e-9 degC).
Scores: MAE on the rows where all 32 estimators of the farm-year return an estimate, mean over turbines; turbine-month
wins for the main contrasts. Decomposition per architecture, from the fleet model with full inputs and correction: the
change in MAE when the base inputs replace the full inputs, when per-turbine models replace the fleet model, and when the
correction is removed; and the same contrasts averaged over the other levels.
No expectation is stated. All results are reported, whatever their sign.
Code: `latetruth/revision_controlled.py`; results in `results/revision_checks_2026-11-17_v001/controlled_<ds>_<arch>.npz`
and `controlled_summary.json`.

Amendment F-a (2026-11-17, after the Part F run, diagnostic only, not an estimator): to see which part of the zero-shot
error is a constant offset per turbine, the MAE of (a) is recomputed after subtracting each turbine's mean error over
rows >= EV0 (an oracle that uses all targets; it is reported only as a decomposition of the error).

Amendment G-a (2026-11-17, while Part G was running, after one progress line of one run had been seen (Kelmarsh 2018,
ridge, full inputs, fleet); wording only, the code is unchanged): the `base` inputs are the driver channels of the similarity term with their 10- to 30-min changes and the
two released target lags. For Senvion these channels are power, wind speed, rotor and generator speeds, ambient and
nacelle temperatures, grid current, the rear generator bearing temperature, and the oil inlet temperature; for Siemens
power, wind speed, main-shaft and generator speeds, ambient and nacelle temperatures, phase current, and the generator
bearing temperature. The full inputs add, for Senvion, the heat-balance terms and the gearbox bearing channels with the
gearbox speed (`h30_x`), and for Siemens the gearbox and main bearing channels (`base_x`). The input contrast therefore
measures the heat-balance terms and the gearbox bearing channels together.

Amendment G-b (2026-11-17, while Part G was running, before any of its results was read except the progress line noted
in G-a): a fifth architecture is added so that a recurrent network, which models the time order of the inputs, is part
of the comparison: (v) a gated recurrent unit (one layer of 64 units, then a linear output) on a window of the six most
recent input vectors (rows t-5 .. t, all available at decision time), Adam, learning rate 1e-3, batch 1024, 20 epochs,
squared error, seed 20261117, inputs prepared as for (iii) and (iv), target standardized. All other factors, rows, and
scores as in Part G, including the common-row set, which is formed over all five architectures. The settings are fixed
here a priori and are not tuned. Code: `latetruth/revision_controlled_gru.py` (imports the functions of
`latetruth/revision_controlled.py`, which is not changed). Further architectures (other gradient-boosting libraries,
forecasting networks) are not added: the comparison asks where the gain comes from, not which architecture is newest,
and the article already compares with deep soft sensors published in 2025 and 2026 for this target.

## Part H (reviewer 2, concern 2): irregular delays, missing data, and simultaneous delays of several channels

Registered 2026-11-17, before the run. Data: Kelmarsh 2018, Penmanshiel 2019, Hill of Towie 2019. One random draw per
turbine and scenario (numpy default_rng, seed 20261118 + 1000 x scenario number + turbine number).
Release model: every record of a turbine reaches the estimator in time order (first in, first out). The release time of
the target value of row s is rel(s) >= s + 1; rel is non-decreasing. The input channels of a row are available at the
row itself unless the scenario delays them.
Scenarios:
0. nominal: rel(s) = s + 3 (the article). Check: must reproduce the saved estimates to within 1e-9 degC.
1. jitter: rel(s) = running maximum of s + u(s), u uniform on {1, ..., 6} (10 to 60 min).
2. batch: target values are sent every 4 h: rel(s) = the next multiple of 24 rows after s (1 to 24 rows).
3. target outage: rel(s) = s + 3, except that no value is released inside outage blocks; values due inside a block are
   released at its end. Blocks cover about 10 % of the year, lengths uniform on 36 .. 432 rows (6 h to 3 days), gaps
   exponential with mean 9 times the mean length.
4. input gaps: rel(s) = s + 3; all input channels of the turbine are unavailable at decision time inside gap blocks
   (about 5 % of the year, lengths uniform on 6 .. 36 rows, gaps exponential with mean 19 times the mean length); the
   values arrive at the end of the block and are used in later refits.
5. combined: jitter as in 1, plus link outages in which the whole record (inputs and target) is unavailable, blocks as
   in 3.
Rules (generalizing the article exactly): at decision time t, the target inputs of the fleet model are the newest and
the second-newest released values (ages a(t) and a(t) + 1, with a(t) = t - max{s : rel(s) <= t}); the residuals usable
by the correction terms are those with rel(s) < t (the newest released one excluded, as in the article), with the
memory of the similarity term counted back from the newest usable residual; the training rows of a refit at s0 are those
with rel(r) < s0. A row is valid if y(t), y(t - a(t)), and y(t - a(t) - 1) are finite. Unavailable inputs are passed as
missing values; the method is not changed for missing inputs (its driver standardization replaces a missing driver by
zero before standardization, as in the article).
Estimators: fleet model, proposed estimator, the EWMA reference of the article (half-life 18 rows, weight 0.75), and
persistence (the newest released value).
Scores: MAE on valid rows >= EV0, mean over turbines: all rows; rows inside blocks (scenarios 3-5); rows outside blocks;
for 1 and 2 also by the age a(t). Compared with scenario 0.
No expectation is stated. All results are reported, whatever their sign. The online Kalman filter is not run: its code
assumes a fixed delay.
Code: `latetruth/revision_irregular.py`; results in `results/revision_checks_2026-11-17_v001/irregular_<ds>_s<k>.json`.

Amendment B-a (2026-11-17, after the Part B results were read; diagnostic, no expectation): five of the 41 new-turbine
runs exceeded the 5 % expectation (Kelmarsh 2018 turbine 5; Penmanshiel 2019 turbines 14 and 15; Hill of Towie 2019
turbines 3 and 16). Penmanshiel 14 and 15 carry the two largest index values. To test whether the numeric index of a turbine
that the first model has never seen drives these cases, each of the five is rerun as new turbine (i) with the categorical
index of Part A, and (ii) with a numeric index moved to the middle of the range (its label swapped with that of the
turbine whose label is the median of the farm; that turbine is scored as well). Also, the share of the annual increase of
each new turbine that falls in the first window is computed exactly from the valid-row counts of both periods.

Amendment H-a (2026-11-17, after the Part H results were read; exploratory, designed after the results, reported as
such, like the standstill gate of the article): an input fallback rule. At decision time t, if the input record of row
t has not arrived, the estimator returns the newest released target value (persistence) instead of its estimate;
otherwise its estimate is unchanged. Applied to the fleet model and to the proposed estimator in scenarios 4 and 5 on
the three farm-years (the same random draws as Part H). Scores as in Part H, and the change against the same estimator
without the rule. Code: `latetruth/revision_irregular_fallback.py` (imports `latetruth/revision_irregular.py`, which is not
changed); results in `irregular_fallback_<ds>_s<k>.json`.

Amendment H-b (2026-11-18, after the H-a results were read; exploratory diagnostic): on Hill of Towie 2019 in scenario 4,
outside the gap blocks the proposed estimator had a higher error than the fleet model. To test whether the released
residuals of the rows whose inputs were missing at decision time drive this, the similarity and recency terms are run
with these residuals excluded from their pools (every other rule unchanged), with and without the fallback of H-a, in
scenarios 4 and 5 on the three farm-years (same random draws). Code: `latetruth/revision_irregular_exclude.py` (built from
`latetruth/revision_irregular_fallback.py`, which is not changed); results in `irregular_exclude_<ds>_s<k>.json`.

Amendment G-c (2026-11-18, after the Part G results were read; diagnostic): the GRU of G-b had the lowest error on the
three farm-years. It differs from the other architectures in two ways: it sees the six most recent input vectors (1 h),
and it is recurrent. To separate the two, three further fleet models with the full inputs are run, each without and with
the correction: (a) HGB of the article on the six most recent input vectors concatenated (same window, no recurrence);
(b) the MLP of Part G on the same concatenated window; (c) the GRU of G-b with a window of one vector (recurrence
without history). Scored on the common rows of Part G (rows where also these estimates exist). Reading: if (a) and (b)
come close to the GRU, the gain is the information in the window; if (c) comes close, it is the architecture.
Code: `latetruth/revision_controlled_window.py`; results in `controlled_window_<ds>_<variant>.npz/json`.
