# Results

Outputs of the registered analysis (`SPEC.md` §9). Every result file records the commit that wrote it: a `meta` key in JSON files, a first comment line in CSV and Markdown files (read CSV files with `skiprows=1`). Each run writes a `MANIFEST.json` (`sdev/`, `frozen/`, `sob/frozen_scoring/`, `sob/after_frozen_scoring/`) with the SHA-256 of every file it wrote, including the files that are not distributed. Result files hold no timestamps, so a rerun at the same commit on the same inputs gives identical bytes; timestamps are only in logs.

## No VitalDB blood-pressure values are published

The per-forecast files below hold, for every forecast, the cuff value it targets (`y_map`) and the last cuff value before it (`last_map`), with the case ID and the reading times. Together they reproduce part of the VitalDB cuff track, so they are not committed (`.gitignore`). Only summary statistics derived from them, per patient and per cohort, are published.

| Withheld file | Written by | SHA-256 listed in |
|---|---|---|
| `sdev/confirmatory/predictions_primary.csv` | `scripts/run_confirmatory.py` | `sdev/MANIFEST.json` |
| `sdev/confirmatory/yardstick_predictions.csv` | `scripts/run_confirmatory.py` | `sdev/MANIFEST.json` |
| `sob/frozen_scoring/predictions_primary.csv` | `scripts/run_frozen_scoring.py score` | `sob/frozen_scoring/MANIFEST.json` |
| `sob/frozen_scoring/yardstick_predictions.csv` | `scripts/run_frozen_scoring.py score` | `sob/frozen_scoring/MANIFEST.json` |
| `sob/frozen_scoring/sensitivity_including_nmb/predictions_primary.csv` | `scripts/run_frozen_scoring.py score` | `sob/frozen_scoring/MANIFEST.json` |

In their place, `scripts/export_public_tables.py` writes per-patient tables from them. These tables are enough to recompute every reported patient-level RMSE, Δ and relative reduction with its bootstrap CI, G and g, the fast-cycling contrast and the yardstick N, D and F:

- `per_patient_primary_wide.csv`: one row per case with primary forecasts. Columns: `caseid`, `subjectid`, `fast_cycling`, `n_forecasts`, `n_forecasts_waveform`, `entered` (at least 3 forecasts; `SPEC.md` §6.1), and `rmse_<model>` for every model of the run (empty below 3 forecasts).
- `per_patient_yardstick.csv`: one row per case with withheld-cuff triples (`SPEC.md` §3). Columns: `caseid`, `subjectid`, `fast_cycling`, `n_triples`, `entered` (at least 3 triples), and `rmse_<model>` for L, LH, T, LH-P1, T-P1 and U (the frozen scoring adds LH-ia, T-ia and the 2-reading-ahead M2, scored on the same rows).
- `public_tables.json`: the SHA-256 of these tables, and of the withheld files they were read from, with the `MANIFEST.json` that lists each one.

Before writing, the script checks the tables against the run's own `per_patient_primary.csv` and `summary.json`. It fits no model and changes no file of the run.

The bootstrap is over subjects (`subjectid`): 2,000 resamples, seed 20261118, percentile intervals; a drawn subject brings all of its cases (`btc/models/bootstrap.py`). A contrast uses the patients with `entered` true. The fast-cycling contrast restricts the patients to `fast_cycling` true and draws its own resamples. Δ is the mean RMSE of the model minus that of the comparator. The relative reduction is 1 minus the ratio of the two means.

The case ID and `subjectid` are VitalDB case-table fields. They are kept in `folds.csv` and in the per-patient tables because the subject bootstrap and the folds cannot be reproduced without them.

## File map

**`sdev/`**: the confirmatory S-dev run (`SPEC.md` §9 steps 4-5), at tag `feature-freeze`.
- `MANIFEST.json`: SHA-256 of every file of the run; `meta` gives the commit, the tag, the git state and the SHA-256 of the derived input tables.
- `data_checksums.csv`: the data snapshot (`SPEC.md` §10). The SHA-256 of every stored track array of the 252 downloaded S-dev cases (columns `caseid`, `array`, `dtype`, `n`, `sha256`; checksums only, no data), downloaded on 2026-11-19 (UTC). It is a copy of `data/raw/sdev_checksums.csv`, written by `scripts/download.py`. A fresh download writes the same list there; a difference shows a change in the public data.
- `confirmatory/summary.json`: the question, the answer line, Sentences 1 and 3, the selected row of the decision table with G and g, every contrast, the fast-cycling contrast, the yardstick, and the case counts (analysable, with a recorded neuromuscular blocker, analysed).
- `confirmatory/headline.md`: the answer line and Sentences 1 and 3.
- `confirmatory/contrasts.csv`, `confirmatory/contrasts_fast_cycling.csv`: Δ and relative reduction with 95% CIs.
- `confirmatory/model_summary.csv`: per model, patient-level mean and SD of RMSE, MAE, mean signed error, R² and FIT, the pooled RMSE, and the counts.
- `confirmatory/per_patient_primary.csv`: per case and model, the per-patient metrics and their sums (`n`, `sse`, `sae`, `se`, `sst`).
- `confirmatory/per_patient_primary_wide.csv`, `confirmatory/per_patient_yardstick.csv`, `confirmatory/public_tables.json`: see above.
- `confirmatory/m2_structure.csv`: the four M2 structures and the one chosen (`SPEC.md` §4.1).
- `confirmatory/penalties.csv`: the penalty chosen for every chain, outer fold, node and model (`SPEC.md` §4.5).
- `confirmatory/folds.csv`: the outer and inner fold of every analysed case.
- `secondary/`, when present: the S-dev secondary analyses and the gradient-boosting check (`SPEC.md` §6.6-§6.8), written by `scripts/run_confirmatory.py --with-secondary`.

**`frozen/`**: every confirmatory model refitted on all analysed S-dev cases (`SPEC.md` §4.5, §9 step 6), written by `scripts/run_frozen_scoring.py freeze`.
- `FROZEN.json`: the SHA-256 of `models.pkl`, the M2 structure, the models of each chain, the S-dev case counts and the Python, NumPy, pandas, scikit-learn and SciPy versions.
- `models.pkl`: the fitted models (Python pickle; `scripts/run_frozen_scoring.py` checks its SHA-256 before it loads it).
- `coefficients.csv`: every fitted model in text. One row per design column: the block (cuff, hr or waveform), that block's penalty, the standardised coefficient, the scaler mean and SD, and the imputation median. The docstring of `coefficient_table` in `btc/models/secondary.py` gives the formula. The models can be applied from this file without `models.pkl`.
- `penalty_path.csv`: for every chain, node and stage of the refit, the inner cross-validation criterion of every candidate penalty and of dropping the block, and the one chosen. This is a by-product of the registered selection, not a pre-specified output.
- `MANIFEST.json`: SHA-256 of the files above; `meta` gives the SHA-256 of the S-dev input tables.

**`sob/`**: the cesarean cases, written by `scripts/run_frozen_scoring.py`.
- `frozen_scoring/`: the one frozen scoring (`SPEC.md` §9 step 7): `summary.json` with Sentence 2 and the cesarean line, `headline.md`, `contrasts.csv`, `model_summary.csv`, `per_patient_primary.csv`, `STATE.json`, and `sensitivity_including_nmb/`, the frozen models scored on all analysable cases (`SPEC.md` §2.2).
- `frozen_scoring_log.jsonl`: every attempt at the frozen scoring, with its time and commit.
- `after_frozen_scoring/`: the analyses computed after the frozen scoring (`SPEC.md` §9 step 8). They are labelled as such and cannot change Sentences 1-3.

## Regenerating the withheld files

1. Download the S-dev cases and build the derived tables (`data/`, not committed): `scripts/download.py --cohort sdev`, then `scripts/build_instances.py`, `scripts/build_windows.py` and `scripts/build_features_wf.py`, each with `--cohort sdev`.
2. Check out tag `feature-freeze` and run `python scripts/run_confirmatory.py --feature-tag feature-freeze --out-dir <folder>`. The confirmatory result is the run made at that tag (`SPEC.md` §9 step 5). With the same derived tables (`meta.inputs_sha256` in `sdev/MANIFEST.json`) and the locked environment (`uv.lock`), every file listed in `sdev/MANIFEST.json`, the two withheld files included, has the same SHA-256 in `<folder>/sdev/`.
3. Back at the latest commit, `python scripts/export_public_tables.py --results <folder>` writes the per-patient tables next to them.

Regenerating `sob/` also needs the cesarean cases, which the code downloads only with `BTC_FROZEN_SCORING=1` (`SPEC.md` §9 step 7), and `scripts/run_frozen_scoring.py score --frozen-tag <tag>` at the frozen-model tag (`after` and `series` need the same `--frozen-tag` and checks).

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5), accessed through the public API at run time. Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/, and the VitalDB Data Use Agreement.
