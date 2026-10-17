# Confirmatory test of the two-time-scale correction (ours+rec) on untouched years — protocol v001

Registered 2026-10-17, before any look at the data of these years (only the file list was read).
Motivation: ours+rec (Amendment 18) was designed after the Amendment 17 results; this test is its confirmation on years never used.

## Data (never used by any model in this project; checked by grep of latetruth/, results/, protocols/)
- Kelmarsh 2020 (Kelmarsh_SCADA_2020_3086.zip, on disk), turbines 1-6.
- Penmanshiel 2020 (Penmanshiel_SCADA_2020_WT01-10_3109.zip + WT11-15_3118.zip, version 0.0.2, on disk).
- Hill of Towie 2022 (2022.zip, doi:10.5281/zenodo.22662930), 21 turbines; middle year between 2019 and 2025.
Kelmarsh 2021 is excluded in advance (AeroUp works from 2021-09). A completeness scan is the only look before the runs; a farm-year
with a key channel missing is dropped and the reason reported.

## Pipeline (frozen, no re-tuning)
Same reader, features, PARAMS, windows (EV0 = row 4032, then 1 March .. 1 December), d = 3, local correction (k 20, k0 20, memory
8640) as the later-years protocol. Variants, all frozen: fleet; ours (frozen primary); ref mean20; ref ewma (h 18, c 0.75);
ours+rec (h 18, c 0.5); gate and ours+rec+gate (Amendment 14 gate, secondary). Published baselines are not run (cost).

## Primary hypothesis and rule (stated before the results)
H: ours+rec has a lower MAE than ours AND than ref ewma on each farm-year (common rows). "Confirmed" on a farm-year only if the
Holm-corrected (over the two comparisons) turbine-month Wilcoxon p < 0.05 and the turbine-bootstrap 95 % CI of the difference is above
zero, for both comparisons. Reported: confirmed on n of 3 farm-years. Block bootstrap (7 and 28 days) as sensitivity.
All results reported, including failure.
Note (2026-10-17, completeness scan): HoT 2022 turbine 2304515 is absent from all February 2022 tables; its February rows are treated as missing (reader patched to skip absent turbine-months; no effect on 2019/2025).

## Outcome (2026-10-17) — confirmed on 3 of 3 farm-years
Common-row MAE fleet / ours / ewma / ours+rec / gate / ours+rec+gate:
K20 .1990/.1847/.1868/.1805/.1698/.1693; P20 .1770/.1665/.1638/.1608/.1557/.1543; H22 .2000/.1896/.1891/.1854/.1838/.1810.
ours+rec lower than ours and than ref ewma on all three; Holm p < 1e-5, turbine-bootstrap and 7/28-day block CIs above zero in all six
comparisons. CONFIRM_TWO_TIMESCALE.json, latetruth/confirm_two_timescale.py. Readers patched only for a turbine absent in Feb 2022 (HoT).
