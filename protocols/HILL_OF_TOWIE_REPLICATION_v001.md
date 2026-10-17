# Replication on a third farm and second manufacturer (Hill of Towie, 21x Siemens SWT-2.3-VS-82) — protocol v001

Registered 2026-10-13, before any Hill of Towie model result. Only a missingness scan was run (all key columns 100% present, all 21 turbines, all 12 months of 2019).
Data: Zenodo 10.5281/zenodo.22662930 (RES, CC-BY-4.0), file 2019.zip. Read directly from the zip; nothing extracted or deleted.
Each turbine (StationId 2304510..2304530 = T01..T21) is re-indexed to a full 10-min grid 2019-01-01 00:00 .. 2019-12-31 23:50 (missing rows = NaN; duplicate timestamps: first kept).

Pipeline identical to Kelmarsh/Penmanshiel, no re-tuning: fleet HGB (PARAMS of LOCAL_LATE_TRUTH_LONG_v001) with turbine id, monthly retraining on released rows only (rows < start-3), January = first training window, Feb–Dec evaluation, synchronous, lag 0, gear-oil truth released 3 steps late, local correction k=20, shrink k/(k+20), memory 8640, delay 3.

Channel mapping (fixed now, before results):
- Target: wtc_GeOilTmp_mean (gear oil temperature).
- Helpers (8; Kelmarsh had 9 — Siemens has no gear-oil inlet temperature): ActPower, AcWindSp, MainSRpm (rotor), GenRpm, AmbieTmp, NacelTmp, AmpPhR (grid current), GenBeGTm (generator NDE bearing ≈ "generator bearing rear").
- Extra heat-path channels (_x): HSGenTmp, HSRotTmp, IMSGenTm, IMSRotTm (four gearbox bearings) and MainBTmp (main bearing). Gearbox speed omitted (= GenRpm on this drive train).
- Features: base = helpers at t + 1/2/3-step changes + oil(t-3), oil(t-3)-oil(t-4); base_x = base + extra channels at t + their 1/2/3-step changes.
- Not testable here (no inlet, no vane channel): author's hypotheses H29/H30 terms (R3), inlet freshness, R5 send policy inlet-vs-bearing. Reported as "not applicable", not as failures.

Claims re-tested (success rules):
R1 bearing channels: base_x+local beats base+local by >= 3%.
R2 local correction (author's hypothesis H26): base_x+local beats base_x overall and in >= 8 of 11 months.
R4 freshness concentration (single sensor shifted 3 steps, base_x, no local): the largest loss is on a gear-oil-adjacent temperature (one of the five extra channels); every helper that is not a temperature (power, wind, rotor rpm, generator rpm, current) loses < 3%; all four gearbox bearings stale together loses more than the sum of the four singles (substitutes).
R6 author's hypothesis H33 pressure proxy: with all four gearbox bearings stale, adding wtc_GearPres_mean (value + 1/2/3-step changes) reduces MAE by >= 2%.
All results reported, including failures.

## Amendment 1 (2026-10-13) — H36 pump stage / filter pressure drop (registered after R1–R6 results, before any R7/R8 result)
Source of physics: Siemens service manual ZSM1005840, Fig. 13.36 (oil cooling/filtration diagram 559253): main pump = 2-speed screw pump 70/142 l/min; in-line filter with two analogue pressure transducers (before/after); oil cooler with 2-speed fan; 3x1000 W heaters; idle pump shaft driven.
Training-window check only (January 2019): pump-stage roughly doubles pressure; within high-speed stage the filter pressure drop falls monotonically with oil temperature (0.27 -> 0.07 bar from <30 to 50–60 °C).
Reference = stale:all_gearbox_bearings+pressure (0.2620). New features added to it (value + 1/2/3-step changes): wtc_GearPuHS_timeon, wtc_GearPuLS_timeon, wtc_InlPrBef_mean, wtc_InlPrAft_mean, filter dP = Bef − Aft.
R7 (author's hypothesis H36): + pump stage + filter pressures reduces MAE by >= 2% vs reference.
R8 (controller-decision leakage, reported separately): R7 features + wtc_GearCoHS_timeon, wtc_GearCoLS_timeon, wtc_GearHeat_timeon. No success rule; descriptive only, because these are controller decisions taken from the fresh oil-temperature reading (valid for the delayed-receipt scenario, not for a failed-sensor scenario).
Code: latetruth/run_hill_of_towie_h36.py (imports the R1–R6 script unchanged).
