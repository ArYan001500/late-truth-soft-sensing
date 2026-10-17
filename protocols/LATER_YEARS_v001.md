# Later held-out years (temporal transfer) — protocol v001

Registered 2026-10-15, before any result on these years.
Purpose: test the frozen pipeline and the comparison on the latest complete year of each farm, 5 to 7 years after the
development year (aged turbines, different climate year, possible changes of the SCADA export).

## Data (never used by any model in this project)
- Kelmarsh 2024 (Kelmarsh_SCADA_2024_5962.zip, already on disk), turbines 1-6. Leap year (52 704 rows of 10 min).
  The export repeats each timestamp about 41 times with mostly empty rows; the reader keeps, per timestamp and channel,
  the first non-empty value and re-indexes to the full 10-min grid of the year. Nothing else is changed.
- Penmanshiel 2022 (Penmanshiel_SCADA_2022_WT01-10_4462.zip, Penmanshiel_SCADA_2022_WT11-15_4463.zip; newer dataset
  version, to be downloaded by Aryan), turbines as present (WT03 absent in 2019). 2022 is the latest year with all
  turbines for the whole year (2023 lacks January for WT01-10).
- Hill of Towie 2025 (2025.zip, to be downloaded by Aryan), 21 turbines; latest complete year (2026 is partial).
A completeness scan is the only look at the data before the runs; if a key channel is missing in a farm-year, that
farm-year is dropped and the reason reported.

## Pipeline (frozen, no re-tuning)
Exactly the pipeline, features, PARAMS, windows (EV0 = row 4032, then 1 March .. 1 December), delay d = 3, local
correction (k 20, k0 20, memory 8640), metric and statistics of the earlier evaluations. Baseline hyper-parameters frozen at
their Kelmarsh 2017 choice. Baselines: all CPU baselines of PUBLISHED_BASELINES_v001 (B0, B1, B2, B2f, B3, B5, B6 Kim,
B10f Wang fixed, B11 Bai). The GPU deep baselines (B4, B7, B8, B9) run only if GPU time allows after the Cao runs;
Cao et al. is not run on these years (cost); both facts are reported.

## Rules (same as the earlier held-out years)
- Kelmarsh 2024 and Penmanshiel 2022: R1-R5 as in KELMARSH_2018_UNSEEN_v001 / PENMANSHIEL_REPLICATION_v001; R6 for
  Kelmarsh 2024 (pump and inlet pressures).
- Hill of Towie 2025: R1, R2, R4, R6, R7 as in HILL_OF_TOWIE_REPLICATION_v001 (+ Amendment 1); R3, R5 not applicable.
- Comparison claim (C2 form): ours has a lower MAE than every baseline run on each later year, called better only if the
  Holm-corrected Wilcoxon test on turbine-month MAE (p < 0.05) and the turbine-bootstrap 95 % CI both support it.
All results are reported, including failures.
