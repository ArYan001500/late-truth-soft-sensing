# Exploiting Late-Arriving Ground Truth for Fleet-Level Soft Sensing of Wind Turbine Gearbox Oil Temperature

This repository implements fleet models and corrections based on released prediction residuals for gearbox oil temperature soft sensing. Monthly model fitting uses only rows released before the retraining boundary. A local similarity correction and a short-timescale residual EWMA exploit late-arriving ground truth. Experiments cover ten farm-years from Kelmarsh, Penmanshiel, and Hill of Towie, with matched per-turbine controls and published-method adaptations documented in the dated protocols.

Aryan Ashoori · [ORCID](https://orcid.org/0009-0001-2226-3894) · aryanashoori0@gmail.com

Citation: **DOI will be added at release**. Software citation metadata are in `CITATION.cff`.

Use Python 3.12 on Windows or Linux. CPU execution needs no GPU. Install the pinned dependencies, then run one command:

```console
python -m pip install -r requirements.txt
python -m latetruth reproduce --tier paper
```

| Tier | Work | Runtime | Dataset archives |
|---|---|---|---|
| `paper` | Rebuild numbers, five tables, and eight figures from shipped results; no download | About 11 seconds measured locally | None |
| `quick` | Kelmarsh 2017 development selection and Kelmarsh 2018 fleet model, similarity and recency corrections, residual references, standstill rule, fixed-network Kalman filter; MAE validation | 13.6 minutes measured across the interrupted first attempt and successful resume; see `REPORT.md` | 436,762,185 bytes |
| `main` | All ten farm-years, proposed method, references, matched per-turbine controls, CPU baselines, statistics | Estimated 1–3 days, hardware dependent | 8,983,677,731 bytes |
| `full` | Main tier plus deep baselines, RESCINet on the documented turbine subsets, delay sensitivity, freshness and oil-circuit analyses | Estimated several days or longer; GPU optional | Same archives; additional prediction files |

Archive sizes are exact. Estimated additional result space: quick 0.2 GB; main 2–5 GB; full 5–20 GB. CSV readers stream the required members directly from the ZIP archives. Archives are kept intact. The main tier compares the methods it computes; the full tier adds the deep methods needed for the paper's full comparison set. Saved paper outcomes remain available immediately through the paper tier. Main and full end-to-end runtime and Linux execution have not been validated locally; consult `REPORT.md` for current limits.

```console
python -m latetruth reproduce --tier quick --jobs 8
python -m latetruth reproduce --tier main --data-dir ./data --jobs 8
python -m pip install -r requirements-gpu.txt
python -m latetruth reproduce --tier full --jobs 8
pytest -q
```

`DATA_DIR` defaults to `./data`; `OUT_DIR` defaults to `./results`. Both environment variables accept alternate directories. `--data-dir` overrides `DATA_DIR`. `--jobs` accepts 1–8; jobs run sequentially to bound memory. Existing prediction outputs are skipped. For intentionally offline use of existing archives, add `--offline`; normal execution checks the Zenodo listing and MD5 before using downloads. The quick check uses per-turbine MAE on each method's valid evaluation rows, as recorded in `expected/quick.json`, with absolute tolerance `1e-6`. The paper's common-row comparisons are provided separately in the saved statistics.

Data are downloaded from these cited records under **CC BY 4.0**; please credit the dataset creators according to each record:

- Kelmarsh: [10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551)
- Penmanshiel: [10.5281/zenodo.16807304](https://doi.org/10.5281/zenodo.16807304)
- Hill of Towie: [10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930)

`latetruth/data_manifest.json` maps every farm-year to its verified filename, destination, size, and MD5. Downloads stream to disk with progress and retries; verified archives are reused. A mismatch or missing file fails explicitly. The data are separate from the MIT-licensed software.

`latetruth/` contains the portable numerical modules and tier runner. `module_sources.json` records the original module paths and source hashes. `results/` contains small published result records; prediction arrays and raw archives are excluded from version control. `protocols/` contains the dated research protocols, with local paths and software-tool names removed for release; dates, rules, thresholds and results are unchanged. `generated/numbers.tex` is the deterministic reference; paper outputs appear under `results/paper/`; recomputed predictions and statistics stay under `results/raw/`. Only its nonnumeric provenance header differs from the historical manuscript output, to make repeated builds byte stable. Numerical content and source order are preserved.

Tests use synthetic arrays and require no dataset download. They exercise training cutoffs, delayed residual release, memory boundaries, EWMA warmup, future invariance, initial-window scaling, safe array loads, and byte-identical paper-number reproduction. The CI matrix runs these tests and the paper tier on Windows and Linux with Python 3.12.
