# Final test on never-seen data: Kelmarsh 2018 (6x Senvion MM92) — protocol v001

Registered 2026-10-13, before any 2018 model result. Q1 plan step 2. Only a completeness scan was run (all key channels >= 0.97 per month except October 0.69 and November 0.91–0.94; 52560 rows per turbine).
Data: Kelmarsh_SCADA_2018_3084.zip (Zenodo, Kelmarsh), turbines 1–6. Never used by any model in this project. Read directly from the zip.
Pipeline frozen exactly as finalised on Kelmarsh 2017 (no change of features, PARAMS, windows, k, memory): code path of tools/run_penmanshiel_replication.py (run(), feats from run_h29_h30_physics / run_extra_heat_paths, local k=20, memory 8640, delay 3, EV0 = 4032, monthly retraining on released rows, first window = rows < 4032).
Rules (same as Kelmarsh 2017 / Penmanshiel):
R1 h30_x+local beats h30+local by >= 3%.
R2 h30_x+local beats h30_x overall and in >= 8 of 11 months.
R3 h29 vs base: transient improves >= 3%, calm not worse by > 1%.
R4 single stale (h30_x, no local): gear-oil inlet and rear bearing each > 10%; every other single < 3%; inlet+both bearings stale > sum of the three singles.
R5 always_bearing better than always_inlet.
R6 (author's hypothesis H33) with inlet+both bearings stale, adding Gear oil pump pressure and Gear oil inlet pressure (value + 1/2/3-step changes) reduces MAE by >= 2%.
Headline numbers also reported: best pipeline h30_x+local on 2018 vs 2017 (0.1562).
All results reported, including failures.
