# Exploiting Late-Arriving Ground Truth for Fleet-Level Soft Sensing of Wind Turbine Gearbox Oil Temperature

Code for the paper of the same title. The estimator shares one gradient-boosted model among the turbines of a farm and
corrects it with released prediction residuals, selected by similarity of operating state and by recency. Training and
correction use only target values that have been released at decision time.

Aryan Ashoori · [ORCID](https://orcid.org/0009-0001-2226-3894) · aryanashoori0@gmail.com

Citation: details will be added when the paper is published (see `CITATION.cff`).

## Quick start

Python 3.12, Windows or Linux, no GPU needed.

```console
python -m pip install -r requirements.txt
python -m latetruth reproduce --tier paper
```

| Tier | What it does | Run time | Download |
|---|---|---|---|
| `paper` | Rebuilds the numbers, tables and figures of the paper from the shipped results | about 10 s | none |
| `quick` | Kelmarsh 2017 and 2018: fleet model, both corrections, references, standstill gate, Kalman filter; checks the MAE against `expected/quick.json` | about 15 min | 0.44 GB |
| `main` | All ten farm-years, references, per-turbine control, CPU baselines, statistics | 1 to 3 days (estimate) | 9.0 GB |
| `full` | `main` plus the deep baselines, RESCINet, delay study, freshness and oil-circuit analyses | several days (estimate), GPU optional | 9.0 GB |

```console
python -m latetruth reproduce --tier quick --jobs 8
python -m latetruth reproduce --tier main --data-dir ./data --jobs 8
python -m pip install -r requirements-gpu.txt
python -m latetruth reproduce --tier full --jobs 8
pytest -q
```

Each tier ends with a PASS or FAIL summary. Baselines with neural networks can differ across platforms in the sixth
decimal; their check uses a tolerance of 1e-4 °C. Finished steps are skipped when a run is restarted. `--data-dir` (or `DATA_DIR`)
sets the data folder, `OUT_DIR` the output folder.

## Data

The data are downloaded automatically from Zenodo and checked against the MD5 listed there (CC BY 4.0; please credit the
dataset creators):

- Kelmarsh: [10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551)
- Penmanshiel: [10.5281/zenodo.16807304](https://doi.org/10.5281/zenodo.16807304)
- Hill of Towie: [10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930)

`latetruth/data_manifest.json` lists the file, size and MD5 of every farm-year.

## Contents

- `latetruth/`: estimator, baselines, statistics and the tier runner.
- `results/`: the result files behind the paper's numbers (prediction arrays are not included).
- `protocols/`: the dated protocols of the evaluation (see `protocols/README.md`).
- `tests/`: tests of the receipt-causal rules on synthetic data: training cutoffs, release of residuals, memory limits,
  recency warm-up, invariance to future targets, and exact reproduction of the paper's numbers.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the records above.
