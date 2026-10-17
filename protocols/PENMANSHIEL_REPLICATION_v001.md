# Replication on a second wind farm (Penmanshiel, 14x Senvion MM82) — protocol v001

Registered 2026-10-14 before reading any Penmanshiel model result. Purpose: replication of the Kelmarsh claims on a second wind farm, including the freshness finding.
Data: Zenodo 10.5281/zenodo.5946808 (Version 0.0.2, downloaded by Aryan), year 2017 only, turbines 01,02,04..15 (WT03 absent). Read directly from zips; nothing extracted or deleted.
Everything identical to the Kelmarsh pipeline, no re-tuning: fleet HGB (PARAMS of LOCAL_LATE_TRUTH_LONG_v001) with turbine id, monthly retraining on released rows only, January (rows < 4032) as first training window, Feb–Dec evaluation, synchronous, lag 0, gear-oil truth released 3 steps late, local correction k=20 memory 8640.
Claims re-tested (pre-declared success rules, transferred from Kelmarsh):
R1 bearing channels (h30_x vs h30): h30_x+local beats h30+local by >= 3%.
R2 local correction (author's hypothesis H26): +local beats no-local for h30_x overall and in >= 8 of 11 months.
R3 author's hypothesis H29 term (h29 vs base): transient improves by >= 3%, calm not worse by > 1%.
R4 freshness concentration (author's hypothesis H34 analogue): stale(single) loss for gear-oil inlet and rear bearing each > 10%; every other single sensor < 3%; inlet+both bearings stale loss larger than the sum of the two singles (substitutes).
R5 send policy: always_bearing better than always_inlet.
All results reported, including failures.

## Amendment 1 (2026-10-14, before any model result)
Data check: in Penmanshiel 2016–2017 the gear-oil temperature, gear-oil inlet temperature, axle-bearing temperatures and gearbox speed are 100% missing (turbine 01, and all inspected turbines for 2017); 2018 63% (starts mid-year); 2019 99%; 2020 94%; 2021 99% (half year). The replication therefore uses year 2019 (January training window, Feb–Dec evaluation), same pipeline and rules R1–R5. Penmanshiel 2019 has never been used by any model in this project. Zip files: Penmanshiel_SCADA_2019_WT01-10_3112.zip and Penmanshiel_SCADA_2019_WT11-15_3117.zip.
