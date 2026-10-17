# Reproduction repository validation

Paper: Exploiting Late-Arriving Ground Truth for Fleet-Level Soft Sensing of Wind Turbine Gearbox Oil Temperature.

Repository URL: https://github.com/ArYan001500/late-truth-soft-sensing

Remote state: PRIVATE, empty. `git commit` failed because no author identity is configured. The combined create-and-push command then refused to run without commits; the private repository was created separately and its visibility verified. No files have been pushed.

## Validation

- Mandatory equivalence: PASS. All 30 required arrays have max absolute difference 0.0; shapes and NaN/infinity masks match.
- Local-data quick tier: PASS, 384.72 seconds. Resumed run including development-only standstill selection: PASS, 7.86 seconds.
- Paper tier: PASS, 994 values from 48 source records, five tables and eight figures. Initial measured run: 11.12 seconds. The deterministic reference is reproduced byte-for-byte.
- Fresh-archive numerical check: all 30 required arrays again match with max absolute difference 0.0 and matching nonfinite masks. The resumed quick command passed all six MAE checks in 346.53 seconds; combined first-attempt and resume wall time was 813.02 seconds (13.55 minutes).
- Final paper-tier check with a separate OUT_DIR: PASS, 7.19 seconds, byte-identical numbers.tex. Final synthetic suite also checks that an unfinished raw result cannot affect the paper tier.
- Output-path check after isolating computed artifacts under results/raw: PASS, quick-tier resume in 8.45 seconds. Existing validated arrays were copied within the new repository into the raw layout for this path-only check; no predictions were shipped.
- Numeric source audit: all 14 checked core function ASTs match after normalizing import changes and safe-load keywords.
- Original source preservation: all 55 source hashes still match; all 11 dated protocol copies are byte-identical. No prediction arrays or dataset archives are shipped.

### Test output

```text
.......                                                                  [100%]
7 passed in 8.88s
```

### Fresh-folder quick tier

First attempt: real downloads completed and MD5 passed for both archives, then FAIL from a stale in-process import after resource-control code changed while the process was running.
```json
{
  "exit_code": 1,
  "seconds": 466.242709
}
```

Resumed command, reusing only the verified downloads and starting without saved predictions:
```json
{
  "exit_code": 0,
  "seconds": 346.773446
}
```

Verified existing Kelmarsh_SCADA_2017_3083.zip
Verified existing Kelmarsh_SCADA_2018_3084.zip
Starting fleet and similarity estimator kelmarsh2017
kelmarsh2017 fleet h30_x 0.1628 | ours h30_x+local 0.1572
kelmarsh2017 exported 16 + 2 + 2 channels, 6 turbines
kelmarsh2017 fixed Kalman turbine 1
kelmarsh2017 fixed Kalman turbine 2
kelmarsh2017 fixed Kalman turbine 3
kelmarsh2017 fixed Kalman turbine 4
kelmarsh2017 fixed Kalman turbine 5
kelmarsh2017 fixed Kalman turbine 6
Starting fleet and similarity estimator kelmarsh2018
kelmarsh2018 fleet h30_x 0.1801 | ours h30_x+local 0.173
kelmarsh2018 exported 16 + 2 + 2 channels, 6 turbines
kelmarsh2018 fixed Kalman turbine 1
kelmarsh2018 fixed Kalman turbine 2
kelmarsh2018 fixed Kalman turbine 3
kelmarsh2018 fixed Kalman turbine 4
kelmarsh2018 fixed Kalman turbine 5
kelmarsh2018 fixed Kalman turbine 6
Development-only standstill selection: {'form': 'hold', 'h': 72, 'mae': 0.15469960950358305}
{
  "P_fleet": 0.18006003027836917,
  "P_ours": 0.17299620693740644,
  "rec": 0.17120229548901925,
  "ref_ewma": 0.17631575989461054,
  "ref_mean20": 0.17492864616504664,
  "B10f_wang2022_frozen": 0.17908130303718175
}
PASS: quick (346.53 seconds)

Combined first-attempt and resume time: 813.02 seconds.

A separate preliminary streaming download fetched and verified 436,762,185 bytes in 484.54 seconds. It is separate from the fresh-folder CLI measurement above.

### Expected quick MAE (own evaluation rows, per-turbine mean)

| Output | MAE |
|---|---:|
| P_fleet | 0.18006003027836917 |
| P_ours | 0.17299620693740644 |
| rec | 0.17120229548901925 |
| ref_ewma | 0.17631575989461054 |
| ref_mean20 | 0.17492864616504664 |
| B10f_wang2022_frozen | 0.17908130303718175 |

### Equivalence per array

| Saved file | Array | max absolute difference | Nonfinite masks match |
|---|---|---:|---|
| ours_kelmarsh2018.npz | P_fleet_1 | 0 | True |
| ours_kelmarsh2018.npz | P_ours_1 | 0 | True |
| ours_kelmarsh2018.npz | P_fleet_2 | 0 | True |
| ours_kelmarsh2018.npz | P_ours_2 | 0 | True |
| ours_kelmarsh2018.npz | P_fleet_3 | 0 | True |
| ours_kelmarsh2018.npz | P_ours_3 | 0 | True |
| ours_kelmarsh2018.npz | P_fleet_4 | 0 | True |
| ours_kelmarsh2018.npz | P_ours_4 | 0 | True |
| ours_kelmarsh2018.npz | P_fleet_5 | 0 | True |
| ours_kelmarsh2018.npz | P_ours_5 | 0 | True |
| ours_kelmarsh2018.npz | P_fleet_6 | 0 | True |
| ours_kelmarsh2018.npz | P_ours_6 | 0 | True |
| rec_kelmarsh2018.npz | P_1 | 0 | True |
| rec_kelmarsh2018.npz | P_2 | 0 | True |
| rec_kelmarsh2018.npz | P_3 | 0 | True |
| rec_kelmarsh2018.npz | P_4 | 0 | True |
| rec_kelmarsh2018.npz | P_5 | 0 | True |
| rec_kelmarsh2018.npz | P_6 | 0 | True |
| ref_ewma_kelmarsh2018.npz | P_1 | 0 | True |
| ref_ewma_kelmarsh2018.npz | P_2 | 0 | True |
| ref_ewma_kelmarsh2018.npz | P_3 | 0 | True |
| ref_ewma_kelmarsh2018.npz | P_4 | 0 | True |
| ref_ewma_kelmarsh2018.npz | P_5 | 0 | True |
| ref_ewma_kelmarsh2018.npz | P_6 | 0 | True |
| B10f_wang2022_frozen_kelmarsh2018.npz | P_1 | 0 | True |
| B10f_wang2022_frozen_kelmarsh2018.npz | P_2 | 0 | True |
| B10f_wang2022_frozen_kelmarsh2018.npz | P_3 | 0 | True |
| B10f_wang2022_frozen_kelmarsh2018.npz | P_4 | 0 | True |
| B10f_wang2022_frozen_kelmarsh2018.npz | P_5 | 0 | True |
| B10f_wang2022_frozen_kelmarsh2018.npz | P_6 | 0 | True |

## Zenodo archive map

Live REST API listings were fetched for records 16807551, 16807304, and 22662930. Each lists license `cc-by-4.0`. All 13 required local archives are present in those exact records and match their MD5 values. Archive readers stream the needed CSV members rather than extracting entire archives.

| Farm-year | Record | Destination under DATA_DIR | Bytes | Local and API MD5 | Match |
|---|---:|---|---:|---|---|
| kelmarsh2017 | 16807551 | Kelmarsh/Kelmarsh_SCADA_2017_3083.zip | 174604722 | c78263ee52ee0e48e2cb4bbaa1ba211a | True |
| kelmarsh2018 | 16807551 | Kelmarsh/Kelmarsh_SCADA_2018_3084.zip | 262157463 | 972e5280172216a5fe76dcab19c741b0 | True |
| kelmarsh2020 | 16807551 | Kelmarsh/Kelmarsh_SCADA_2020_3086.zip | 473677799 | e189e02da98f32e74acc72e8559ef0db | True |
| kelmarsh2024 | 16807551 | Kelmarsh/Kelmarsh_SCADA_2024_5962.zip | 704598245 | e9e3318f5b393d8596ff1801f8da0414 | True |
| penmanshiel2019 | 16807304 | Penmanshiel/Version 0.0.2/Penmanshiel_SCADA_2019_WT01-10_3112.zip | 473518587 | 78389603cb4ae20a6a9b878f5a84bc7b | True |
| penmanshiel2019 | 16807304 | Penmanshiel/Version 0.0.2/Penmanshiel_SCADA_2019_WT11-15_3117.zip | 265354191 | 92747a976716c5f35d9ee6462958fd7a | True |
| penmanshiel2020 | 16807304 | Penmanshiel/Version 0.0.2/Penmanshiel_SCADA_2020_WT01-10_3109.zip | 694483651 | 46f679f2e4ab3d69c04b31c8dc625106 | True |
| penmanshiel2020 | 16807304 | Penmanshiel/Version 0.0.2/Penmanshiel_SCADA_2020_WT11-15_3118.zip | 389300869 | c73323f21451c08247dbdb5d1a1434a9 | True |
| penmanshiel2022 | 16807304 | Penmanshiel/version 3/Penmanshiel_SCADA_2022_WT01-10_4462.zip | 710776645 | 91351469cdeb3ca95be1a0414dcfec8f | True |
| penmanshiel2022 | 16807304 | Penmanshiel/version 3/Penmanshiel_SCADA_2022_WT11-15_4463.zip | 396724464 | 5f4743133a9cf65ad33de139b43f96b4 | True |
| hot2019 | 22662930 | HillOfTowie_04/2019.zip | 1430010222 | ca240b179200f5db25a273a1291eb597 | True |
| hot2022 | 22662930 | HillOfTowie_04/2022.zip | 1412250190 | 3861de12eb3ba588f48259c918ade242 | True |
| hot2025 | 22662930 | HillOfTowie_04/2025.zip | 1596220683 | d3cd7edb535a45ffca1c7028b397f035 | True |

Total archive bytes for all ten farm-years: 8,983,677,731. Quick archive bytes: 436,762,185. No cited-version substitution was necessary.

## Included artifacts

- Shipped result records: 92 JSON files, 1,113,865 bytes in total; one required historical text input, 64 bytes. Prediction NPZ files shipped: 0 bytes.
- Dated protocols: 11, copied verbatim, with a separate English README.
- Copied numerical modules: 55. Portable package also contains configuration, downloader, CLI, and tier orchestration.
- Non-scientific changes: package imports; DATA_DIR/OUT_DIR locations and separate results/raw outputs; sys.executable in job runners; safe array and tensor loads; eight-worker cap; output-directory resume guards; stable provenance header and LF line endings for numbers.tex. Numerical values, masks, seeds and estimator settings were retained.

### Module provenance

| Repository module | Original relative path | Original SHA-256 |
|---|---|---|
| latetruth/baseline_wang2022.py | tools/baseline_wang2022.py | 267e9f19303f48911794e33cf3fd3baf5321cde19bd408654bbbf1396a5a03a4 |
| latetruth/baseline_wang2022_frozen.py | tools/baseline_wang2022_frozen.py | d555103b1ab052a3672b8746c19e6a9e9cf04eff559a67fc54a5e33b8477a5bc |
| latetruth/baselines_bai.py | tools/baselines_bai.py | 8167c61a7d10f2a2843e654ccb51fe007431a6ef52e5eba103cb9e6075b29e73 |
| latetruth/baselines_cao.py | tools/baselines_cao.py | d4751f67ab63e0760da975e7ecef3cb85e6ebc0b1552ffc285340be45f4d2425 |
| latetruth/baselines_classic.py | tools/baselines_classic.py | 12fb047e20973aa643bb2c005a7fe0718ffa1b93686795e77a4d4938a413a09d |
| latetruth/baselines_common.py | tools/baselines_common.py | 63f93857d9298e16d4d5b055d53065c99459872e15dd65b6e4b4f4724bf02ffc |
| latetruth/baselines_deep.py | tools/baselines_deep.py | d9a29466a53b4d1b48603cfa673b18b90a92a54ea293891dc653abb9c399b349 |
| latetruth/baselines_hot.py | tools/baselines_hot.py | e92818452d93755712d94f8e59ef7eaf1441ae7d115e32367236966cb78485e7 |
| latetruth/baselines_kim.py | tools/baselines_kim.py | 91262922d37c9546662f3016c204e4d941d4f8b374c233b46c48a36e872f0bda |
| latetruth/baselines_table.py | tools/baselines_table.py | fb4b8a307b1782c1c823fe688b04b703e9d51e8cbadcbb1f4c0b1b3ee11ee247 |
| latetruth/block_bootstrap.py | tools/block_bootstrap.py | 1149e024e516a16dac20032bb88a0262cf27dbbc4f107b559747aa8399c784bc |
| latetruth/cao_table.py | tools/cao_table.py | 0e819fe9ee5e7e28835dbdb2bf826e72f6e28a4916c5cdc56f9a8427a46e2677 |
| latetruth/compare_per_turbine.py | tools/compare_per_turbine.py | 3afa238ebbcfef51f7c30bc146e385e6d38072b323cda317965f553e88ea17dd |
| latetruth/confirm_two_timescale.py | tools/confirm_two_timescale.py | ff5b58d0061c04734607cc7427db05bc38544369e6c63cfe878e31f8b99a0cd8 |
| latetruth/cpu_baseline_one.py | tools/cpu_baseline_one.py | da6553e467d3f401da5162d7b077473b25e65ff04e53459fff07f86fda07a953 |
| latetruth/dataset_stats.py | tools/dataset_stats.py | ef09f1241ec83ef326f8e99c1af4bdabf7b18a15777d528804e7cbf5ed7bffd1 |
| latetruth/deep_lane.py | tools/deep_lane.py | f820c9efec4dadf709c1e010268cc4fc7c1a5b687238188fded27c97e808e7db |
| latetruth/delay_part1.py | tools/delay_part1.py | a63cc5002c82d37a04d704a7926446d306822bcbcdea011cee3198831bc86f89 |
| latetruth/delay_part2.py | tools/delay_part2.py | 1b0d4d77dcd5c6dbeee004d0cd621549109a047a466e092023bfde251f55de58 |
| latetruth/delay_rec.py | tools/delay_rec.py | 9832bfbd213b542b7217ffa323e4a50133b0294b2661c93db2fb9612d905e4af |
| latetruth/design_ladder.py | tools/design_ladder.py | 0ef2d22a198d50b8af4385ac351565105358a6bc1fe4efa74c1796d7d03d746c |
| latetruth/discussion_recency_vs_similarity.py | tools/discussion_recency_vs_similarity.py | d0d5eef5043687d061464b8f90bafda272c22ef786e8feb95a3bdeef9e1ad478 |
| latetruth/discussion_recency_vs_similarity_wins.py | tools/discussion_recency_vs_similarity_wins.py | f9ea7fb7db3107ba4c4aa9a7ad445690d3fc00e0f9b8f0650f33fa8fac8c15a4 |
| latetruth/figure_data.py | tools/figure_data.py | e76d48e4dddb0a84fe99e3750627cd0fd91c6456b04e1f027e0d7e0d96334334 |
| latetruth/ladder_l6.py | tools/ladder_l6.py | 2276322396b2a251d00157c8b542ea1464715d36bba7bc3e3a553d5ca803f756 |
| latetruth/later_years.py | tools/later_years.py | 54a733d65f7f75f2e93c8a76fc503e4cb224b045e6ef6f6cdc9faea4731d51cd |
| latetruth/local_grid.py | tools/local_grid.py | dfed64089a3af206b5ed40977623cb389c72ef2954bb7095672136b894436f2e |
| latetruth/make_paper_figures.py | tools/make_paper_figures.py | b46c57bb74075aefb195f510f2ad5575da549a81b407cbe51eb47fb0a94131c9 |
| latetruth/make_paper_numbers.py | tools/make_paper_numbers.py | d20333cacc481a6849fdb7a70d70985a96d17d2b82edd2ad4f18180fb8d3896c |
| latetruth/make_paper_schematics.py | tools/make_paper_schematics.py | cb07647bb22e1438c9b6112121b374bc3f6c50dd59b3072d2888c6cb51b60ce7 |
| latetruth/make_rec_files.py | tools/make_rec_files.py | eda9d216cbd35107e786a81c6197c6bd278da86451adc081c20efa6744a21589 |
| latetruth/per_turbine_matched.py | tools/per_turbine_matched.py | 2ce84cc6742698f9fe065b0ec047889af1fd79dff5fd88ed17baab575edec3f1 |
| latetruth/rec_vs_baselines.py | tools/rec_vs_baselines.py | 99c86eb344e01cde981a304b247056aee0b6bca6eec0a162bde0290687289dcf |
| latetruth/recency_on_similarity.py | tools/recency_on_similarity.py | b2e609923fe2092ce85bf7d5fa444bf0ab21cfe0be8640da51ee004dc8795aa1 |
| latetruth/residual_references.py | tools/residual_references.py | 3524ef7a06c303e17a8f51429c505bca00cc7d8b135e792fce8e14b0fb63e425 |
| latetruth/run_extra_heat_paths.py | tools/run_extra_heat_paths.py | aa4b3082e89249cc30eb284676f7c08a0306ca690aa667b9ce9cde9141062f6f |
| latetruth/run_h29_h30_physics.py | tools/run_h29_h30_physics.py | 738bec58852ff6e785fa2e3ee1e20d9b60ebcb02a7c5c4491258c80e10781f6a |
| latetruth/run_hill_of_towie_h36.py | tools/run_hill_of_towie_h36.py | abb430747999a978db8c76db2784bd4aa1889d8548a0e2421cb8e4fc8e762208 |
| latetruth/run_hill_of_towie_replication.py | tools/run_hill_of_towie_replication.py | ff912ad34496e6f5a62882c963a25a6943d3a84949859bd118862c70643d0fdb |
| latetruth/run_kelmarsh_2018_unseen.py | tools/run_kelmarsh_2018_unseen.py | 6814d0b941cc7524b778d9a035ef957ccc0e5fb43d1325814717c1276ac7ce9c |
| latetruth/run_local_late_truth_long.py | tools/run_local_late_truth_long.py | ef790b9e13f118f293fa8f881561dcba5bc00832e6d2b72a6b691f92a540a946 |
| latetruth/run_new_datasets.py | tools/run_new_datasets.py | 9b98e461e0cbb8af4b9b7cec55504f9b8d13549e280dbdbf7d735a274d20916a |
| latetruth/run_penmanshiel_replication.py | tools/run_penmanshiel_replication.py | 4f260b05e9763f5a8999dfb2c4afc92cc57c68628ad3808ae63ecd70c436e1e2 |
| latetruth/run_send_schedule.py | tools/run_send_schedule.py | 2ed3208540a6174f94bbbfa0f1bfbc4a4ac0bfb0a9e2429fcaf6c70c9c8c8e17 |
| latetruth/s5_timing.py | tools/s5_timing.py | 9ca278e61ffba4ac64e62659216e1e9d766cf442988033beb8e35a2edc94ae44 |
| latetruth/significance.py | tools/significance.py | 2a0a4d4473725c3a1dcbcb1005bad174bea3fafc694eb485bbbdea4862457997 |
| latetruth/standstill_diag.py | tools/standstill_diag.py | 8e0137125d368dbc0b855bcf8cb27f29179a7a2f579f886591613e61f98ff4ab |
| latetruth/standstill_gate.py | tools/standstill_gate.py | 233debc1e10d9a86e2d81ad1b66a894f7203553efaf90c2e158bb71520e27c48 |
| latetruth/turbine_level.py | tools/turbine_level.py | 00537d043a796019c12690884c35ae639644c64ca4c8ff01a88b197348f8e04b |
| latetruth/verify_final_table.py | tools/verify_final_table.py | f04da3892e74ea20b7ab56a54ffe6e3fbad1d01d544e3d9175b4f6b21ae848cd |
| latetruth/figure_data_oil.py | tools/figure_data_oil.py | 9f81c9b8941befdb7e250b8a59dd0b4c3c6068cbce3b48fe615d764f97d04ce3 |
| latetruth/filter_index.py | tools/filter_index.py | 6e1fb42afc8cdbb44181c50895adb18f78dfd1444c1a48f4bf2e0293515968dc |
| latetruth/physics_oil_circuit_check.py | tools/physics_oil_circuit_check.py | cecc9279649e0259ac007595f8735865f891141a19115de6f8a789d0cb6ce9d8 |
| latetruth/run_freshness_value.py | tools/run_freshness_value.py | 27db438847c4913bd39bdf1081c1546ae7daae1705b0312cdea624d5b1a422fc |
| latetruth/run_later_years.py | tools/run_later_years.py | d550e3779ba00a6eeb48e0738200edc788fb3cb0cf528df509d7e90c9d059dfe |

### File tree

```text
.gitattributes
.github/workflows/tests.yml
.gitignore
CITATION.cff
LICENSE
README.md
REPORT.md
expected/quick.json
generated/numbers.tex
latetruth/__init__.py
latetruth/__main__.py
latetruth/baseline_wang2022.py
latetruth/baseline_wang2022_frozen.py
latetruth/baselines_bai.py
latetruth/baselines_cao.py
latetruth/baselines_classic.py
latetruth/baselines_common.py
latetruth/baselines_deep.py
latetruth/baselines_hot.py
latetruth/baselines_kim.py
latetruth/baselines_table.py
latetruth/block_bootstrap.py
latetruth/cao_table.py
latetruth/compare_per_turbine.py
latetruth/config.py
latetruth/confirm_two_timescale.py
latetruth/cpu_baseline_one.py
latetruth/data.py
latetruth/data_manifest.json
latetruth/dataset_stats.py
latetruth/deep_lane.py
latetruth/delay_part1.py
latetruth/delay_part2.py
latetruth/delay_rec.py
latetruth/design_ladder.py
latetruth/discussion_recency_vs_similarity.py
latetruth/discussion_recency_vs_similarity_wins.py
latetruth/figure_data.py
latetruth/figure_data_oil.py
latetruth/filter_index.py
latetruth/ladder_l6.py
latetruth/later_years.py
latetruth/local_grid.py
latetruth/make_paper_figures.py
latetruth/make_paper_numbers.py
latetruth/make_paper_schematics.py
latetruth/make_rec_files.py
latetruth/per_turbine_matched.py
latetruth/physics_oil_circuit_check.py
latetruth/rec_vs_baselines.py
latetruth/recency_on_similarity.py
latetruth/reproduce.py
latetruth/residual_references.py
latetruth/run_extra_heat_paths.py
latetruth/run_freshness_value.py
latetruth/run_h29_h30_physics.py
latetruth/run_hill_of_towie_h36.py
latetruth/run_hill_of_towie_replication.py
latetruth/run_kelmarsh_2018_unseen.py
latetruth/run_later_years.py
latetruth/run_local_late_truth_long.py
latetruth/run_new_datasets.py
latetruth/run_penmanshiel_replication.py
latetruth/run_send_schedule.py
latetruth/s5_timing.py
latetruth/significance.py
latetruth/standstill_diag.py
latetruth/standstill_gate.py
latetruth/turbine_level.py
latetruth/verify_final_table.py
module_sources.json
protocols/CONFIRM_TWO_TIMESCALE_v001.md
protocols/EXTRA_HEAT_PATHS_v001.md
protocols/FRESHNESS_VALUE_v001.md
protocols/H29_H30_PHYSICS_TERMS_v001.md
protocols/HILL_OF_TOWIE_REPLICATION_v001.md
protocols/KELMARSH_2018_UNSEEN_v001.md
protocols/LATER_YEARS_v001.md
protocols/LOCAL_LATE_TRUTH_LONG_v001.md
protocols/PENMANSHIEL_REPLICATION_v001.md
protocols/PUBLISHED_BASELINES_v001.md
protocols/README.md
protocols/SEND_SCHEDULE_v001.md
pyproject.toml
requirements-gpu.txt
requirements.txt
results/filter_index_2026-10-15_v001/results.json
results/freshness_value_2026-10-14_v001/both_stale_output.txt
results/freshness_value_2026-10-14_v001/results.json
results/hill_of_towie_replication_2019_2026-10-13_v001/results_A.json
results/hill_of_towie_replication_2019_2026-10-13_v001/results_B.json
results/hill_of_towie_replication_2019_2026-10-13_v001/results_C.json
results/hill_of_towie_replication_2019_2026-10-13_v001/results_h36.json
results/kelmarsh_2018_unseen_2026-10-13_v001/results_A.json
results/kelmarsh_2018_unseen_2026-10-13_v001/results_B.json
results/kelmarsh_2018_unseen_2026-10-13_v001/results_C.json
results/kelmarsh_2018_unseen_2026-10-13_v001/results_D.json
results/later_years_hot2025_2026-10-15_v001/results_A.json
results/later_years_hot2025_2026-10-15_v001/results_B.json
results/later_years_hot2025_2026-10-15_v001/results_C.json
results/later_years_hot2025_2026-10-15_v001/results_h36.json
results/later_years_kelmarsh2024_2026-10-15_v001/results_A.json
results/later_years_kelmarsh2024_2026-10-15_v001/results_B.json
results/later_years_kelmarsh2024_2026-10-15_v001/results_C.json
results/later_years_kelmarsh2024_2026-10-15_v001/results_D.json
results/later_years_penmanshiel2022_2026-10-15_v001/rules/results.json
results/penmanshiel_replication_2019_2026-10-14_v001/results.json
results/physics_oil_circuit_2026-10-15_v001/figdata_dp_vs_T.json
results/physics_oil_circuit_2026-10-15_v001/results.json
results/published_baselines_2026-10-14_v001/B0_lag1.json
results/published_baselines_2026-10-14_v001/B0_lag3.json
results/published_baselines_2026-10-14_v001/B0_lag3_newds.json
results/published_baselines_2026-10-14_v001/B10_wang2022.json
results/published_baselines_2026-10-14_v001/B10f_wang2022_frozen.json
results/published_baselines_2026-10-14_v001/B11_bai_xgb.json
results/published_baselines_2026-10-14_v001/B11_bai_xgb_newds.json
results/published_baselines_2026-10-14_v001/B1_lag1.json
results/published_baselines_2026-10-14_v001/B1_lag3.json
results/published_baselines_2026-10-14_v001/B1_lag3_newds.json
results/published_baselines_2026-10-14_v001/B2_lag1.json
results/published_baselines_2026-10-14_v001/B2_lag3.json
results/published_baselines_2026-10-14_v001/B2_lag3_newds.json
results/published_baselines_2026-10-14_v001/B2f_lag1.json
results/published_baselines_2026-10-14_v001/B2f_lag3.json
results/published_baselines_2026-10-14_v001/B2f_lag3_newds.json
results/published_baselines_2026-10-14_v001/B3_lag1.json
results/published_baselines_2026-10-14_v001/B3_lag3.json
results/published_baselines_2026-10-14_v001/B3_lag3_newds.json
results/published_baselines_2026-10-14_v001/B5_lag1.json
results/published_baselines_2026-10-14_v001/B5_lag3.json
results/published_baselines_2026-10-14_v001/B5_lag3_newds.json
results/published_baselines_2026-10-14_v001/B6_lag1.json
results/published_baselines_2026-10-14_v001/B6_lag3.json
results/published_baselines_2026-10-14_v001/B6_lag3_newds.json
results/published_baselines_2026-10-14_v001/B9_grataloup_repro.json
results/published_baselines_2026-10-14_v001/BLOCK_BOOTSTRAP.json
results/published_baselines_2026-10-14_v001/CAO_TABLE.json
results/published_baselines_2026-10-14_v001/CAO_TABLE_PROPOSED.json
results/published_baselines_2026-10-14_v001/CONFIRM_TWO_TIMESCALE.json
results/published_baselines_2026-10-14_v001/DATASET_STATS.json
results/published_baselines_2026-10-14_v001/DELAY_SENSITIVITY_part1.json
results/published_baselines_2026-10-14_v001/DELAY_SENSITIVITY_part2.json
results/published_baselines_2026-10-14_v001/DELAY_SENSITIVITY_rec.json
results/published_baselines_2026-10-14_v001/DESIGN_LADDER_L6_kelmarsh2017.json
results/published_baselines_2026-10-14_v001/DESIGN_LADDER_kelmarsh2017.json
results/published_baselines_2026-10-14_v001/DISCUSSION_RECENCY_VS_SIMILARITY.json
results/published_baselines_2026-10-14_v001/DISCUSSION_RECENCY_VS_SIMILARITY_WINS.json
results/published_baselines_2026-10-14_v001/FIGDATA_turbine_month.json
results/published_baselines_2026-10-14_v001/FINAL_TABLE_all.json
results/published_baselines_2026-10-14_v001/FINAL_TABLE_kelmarsh.json
results/published_baselines_2026-10-14_v001/GATE.json
results/published_baselines_2026-10-14_v001/LOCAL_GRID_kelmarsh2018.json
results/published_baselines_2026-10-14_v001/PER_TURBINE_CONTROL.json
results/published_baselines_2026-10-14_v001/RECENCY_ON_SIMILARITY.json
results/published_baselines_2026-10-14_v001/REC_VS_BASELINES_all9.json
results/published_baselines_2026-10-14_v001/REC_VS_BASELINES_hot2022.json
results/published_baselines_2026-10-14_v001/REC_VS_BASELINES_kelmarsh2020_penmanshiel2020.json
results/published_baselines_2026-10-14_v001/REC_VS_BASELINES_table7.json
results/published_baselines_2026-10-14_v001/RESIDUAL_REFERENCES.json
results/published_baselines_2026-10-14_v001/S5_TIMING.json
results/published_baselines_2026-10-14_v001/SIGNIFICANCE.json
results/published_baselines_2026-10-14_v001/STANDSTILL_DIAG.json
results/published_baselines_2026-10-14_v001/TURBINE_LEVEL.json
results/published_baselines_2026-10-14_v001/VERIFY_FINAL_TABLE.json
results/published_baselines_2026-10-14_v001/deep_kelmarsh2018.json
results/published_baselines_2026-10-14_v001/deepv2_kelmarsh2017.json
results/published_baselines_2026-10-14_v001/deepv2_kelmarsh2018.json
results/published_baselines_2026-10-14_v001/ours_hot2019.json
results/published_baselines_2026-10-14_v001/ours_hot2022.json
results/published_baselines_2026-10-14_v001/ours_hot2025.json
results/published_baselines_2026-10-14_v001/ours_kelmarsh2017.json
results/published_baselines_2026-10-14_v001/ours_kelmarsh2018.json
results/published_baselines_2026-10-14_v001/ours_kelmarsh2020.json
results/published_baselines_2026-10-14_v001/ours_kelmarsh2024.json
results/published_baselines_2026-10-14_v001/ours_penmanshiel2019.json
results/published_baselines_2026-10-14_v001/ours_penmanshiel2020.json
results/published_baselines_2026-10-14_v001/ours_penmanshiel2022.json
results/published_baselines_2026-10-14_v001/ours_sync_kelmarsh2018.json
results/send_schedule_2026-10-14_v001/results.json
tests/test_causality.py
tests/test_download.py
```

## ISSUES

1. Git has no existing `user.name` or `user.email`, and `git var GIT_AUTHOR_IDENT` fails with `Author identity unknown` / `unable to auto-detect email address`. No identity was invented or configured.
2. Dependency discrepancy: the required pins are retained exactly (NumPy 2.4.6, pandas 3.0.3, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1; matplotlib 3.11.2). The available Python 3.12 validation environment actually has NumPy 2.5.4 and pandas 3.0.6, with the other listed numerical packages at those versions, and torch 2.14.1+cu132. Its predictions nevertheless match every mandatory reference array exactly. The clean pinned environment is not claimed as tested.
3. The historical numbers.tex header contains a wall-clock timestamp and source-tool path. The shipped deterministic reference uses the fixed historical timestamp with a package path; all 994 values and the 48-source provenance list retain their historical content. LF output is explicit for cross-platform byte reproducibility.
4. The first fresh-download invocation hit an in-process import mismatch because resource-control code was changed while that process was downloading. The downloaded archives passed MD5; execution was restarted with the completed package and no copied predictions. This packaging error did not alter scientific settings.
5. Final review found that recomputed statistics could overwrite the shipped paper inputs. Computed predictions and statistics now use results/raw, while the paper tier always reads the shipped JSON records. The path-only resume and the paper-tier test with an unfinished raw result both pass. No numerical routine was changed for this correction.
6. Main and full have not been run end-to-end in this session; their day-scale runtimes and result-space ranges are estimates. Main statistics use the CPU methods present in that tier; full adds the deep methods. No absent GPU result was presented as a newly reproduced result.

## Git operations

- `git add .`: succeeded; 181 files staged, no dataset archives or prediction arrays.
- `git commit -m "Initial release of the reproduction code"`: failed because the existing author identity is unset.
- `gh repo create ArYan001500/late-truth-soft-sensing --private --source . --push`: refused because no commit exists.
- `gh repo create ArYan001500/late-truth-soft-sensing --private`: succeeded. API verification returned visibility PRIVATE and isEmpty true.
- Local origin points to the verified private repository. No author configuration was changed. No commit or push is claimed.

## UNREACHABLE

- UNREACHABLE: a Git commit and push require a pre-existing author identity, which is absent. The specified existing-identity rule prevents supplying a replacement identity.
- UNREACHABLE: Linux CI and a clean Python 3.12 installation of the exact pinned dependency set have not been verified in this Windows session. CI configuration is included; no successful remote CI run is claimed.

## Uncertainty and limits

Only the mandatory quick numerical paths, synthetic tests, dataset API/checksum mapping, and paper rendering scripts were executed. The core AST audit is not a proof of equivalence for every deep baseline. Historical JSON outcomes are copied evidence, not newly rerun multi-day experiments. Full-tier RESCINet uses the documented subsets (Kelmarsh 2018 turbines 1–6, Penmanshiel 2019 turbines 1, 2, 4, 5, 6, 7) and original v1/v2 learning rates. No scientific inconsistency was silently repaired. No dataset ZIP was deleted.

No public visibility change, release, tag, or archival-service connection was made.


## Release preparation update (2026-10-18)

The earlier empty-remote and missing-identity statements above describe the initial attempt. The existing local author identity is now configured. Protocol copies were edited for release: software-tool attributions and local paths were removed, and hypothesis labels were updated. Every numeric token remains unchanged, excluding removed directory prefixes. Originals remain untouched. The protocol README contains the release-preparation note. The inlet-age local variable was renamed without changing the calculation.

Repository content scan: the only remaining match is LICENSE:19, the standard MIT license wording. This permitted license wording is retained. No other matches remain in shipped files. Local scratch work is excluded from the versioned tree. Prior numerical validation and dependency limitations above still apply.
