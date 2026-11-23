# Results

Outputs of the registered analysis (`SPEC.md` §9). Every result file records the commit that wrote it: a `meta` key in JSON files, a first comment line in CSV and Markdown files (read CSV files with `skiprows=1`). The exceptions are the two data-checksum lists (`data_checksums.csv`, `sdev/data_checksums.csv`), plain copies of the downloads' checksum lists, and the state and log of the frozen scoring (`sob/frozen_scoring/STATE.json`, `sob/frozen_scoring_log.jsonl`), whose entries name the commit. Each run writes a `MANIFEST.json` (`sdev/`, `frozen/`, `sob/frozen_scoring/`, `sob/after_frozen_scoring/`, and the two `exploratory_encoder/` folders) with the SHA-256 of the result files it wrote, including the files that are not committed (next section). The per-patient tables are listed in `public_tables.json` instead. The checksum lists, `STATE.json`, the log, the manifests themselves and this file are in no manifest. Result files hold no timestamps, so a rerun at the same commit on the same inputs gives identical bytes on the same platform; timestamps are only in logs. On another CPU or numerical library the last digits of floating-point results can differ, and the hashes with them; compare the numbers then (the per-patient tables below allow that).

## Files listed in a manifest but not committed

### No VitalDB blood-pressure values are published

The per-forecast files below hold, for every forecast, the cuff value it targets (`y_map`) and the last cuff value before it (`last_map`), with the case ID and the reading times. Together they reproduce part of the VitalDB cuff track, so they are not committed (`.gitignore`). Only summary statistics derived from them, per patient and per cohort, are published.

| Withheld per-forecast file | Written by | SHA-256 listed in |
|---|---|---|
| `sdev/confirmatory/predictions_primary.csv` | `scripts/run_confirmatory.py` | `sdev/MANIFEST.json` |
| `sdev/confirmatory/yardstick_predictions.csv` | `scripts/run_confirmatory.py` | `sdev/MANIFEST.json` |
| `sob/frozen_scoring/predictions_primary.csv` | `scripts/run_frozen_scoring.py score` | `sob/frozen_scoring/MANIFEST.json` |
| `sob/frozen_scoring/yardstick_predictions.csv` | `scripts/run_frozen_scoring.py score` | `sob/frozen_scoring/MANIFEST.json` |
| `sob/frozen_scoring/sensitivity_including_nmb/predictions_primary.csv` | `scripts/run_frozen_scoring.py score` | `sob/frozen_scoring/MANIFEST.json` |
| `sdev/exploratory_encoder/predictions_primary.csv` | `scripts/run_encoder_check.py sdev` | `sdev/exploratory_encoder/MANIFEST.json` |
| `sob/after_frozen_scoring/exploratory_encoder/predictions_primary.csv` | `scripts/run_encoder_sob.py` | `sob/after_frozen_scoring/exploratory_encoder/MANIFEST.json` |

### One coefficient table, omitted for its labelling

`sdev/secondary/coefficients_all_sdev.csv` (SHA-256 `7102e1475770fead5b11fa12dd032b06f562f1079b51cf12d075a37442e90da6`, listed in `sdev/MANIFEST.json`) holds no cuff value; it is omitted for another reason. The S-dev secondary run wrote it with the code of tag `feature-freeze`, whose coefficient table has the columns `chain`, `node`, `model`, `column`, `block`, `coef` and `penalty`. Its `penalty` column repeats each stage's own penalty on every row of that stage (the cuff rows of M3 show 31.6 instead of 1; the cuff and heart-rate rows of M4 show 10 instead of 1 and 31.6), and `block` is an index. Its 66 rows have the same coefficients as chain `main` of `frozen/coefficients.csv` (largest difference 0.0). That file labels each block with its own penalty, adds the scaler and imputation values, and is the coefficient table of `SPEC.md` §6.7. To regenerate the omitted file, run the S-dev secondary analyses at tag `feature-freeze` ("Regenerating the withheld files", step 2, with `--with-secondary`); it is written to `<folder>/sdev/secondary/`.

## Per-patient tables

In place of the per-forecast files, `scripts/export_public_tables.py` writes per-patient tables from them, in the folder of each run (`sdev/confirmatory/`, `sob/frozen_scoring/`, `sob/frozen_scoring/sensitivity_including_nmb/`):

- `per_patient_primary_wide.csv`: one row per case with primary forecasts. Columns: `caseid`, `subjectid`, `fast_cycling`, `n_forecasts`, `n_forecasts_waveform`, `entered` (at least 3 forecasts; `SPEC.md` §6.1), and `rmse_<model>` for every model of the run (empty below 3 forecasts). The table of the sensitivity analysis also has `nmb_recorded`.
- `per_patient_yardstick.csv` (not for the sensitivity analysis, which scores no yardstick model): one row per case with withheld-cuff triples (`SPEC.md` §3). Columns: `caseid`, `subjectid`, `fast_cycling`, `n_triples`, `entered` (at least 3 triples), and `rmse_<model>` for L, LH, T, LH-P1, T-P1 and U (the frozen scoring adds LH-ia, T-ia and the 2-reading-ahead M2, scored on the same rows).
- `public_tables.json`: the SHA-256 of these tables, and of the withheld files they were read from, with the `MANIFEST.json` that lists each one.

Before writing, the script checks the tables against the run's own `per_patient_primary.csv` and `summary.json`. It fits no model and changes no file of the run. The S-ob tables were exported at commit `44cfb63`, with the code of tag `models-frozen`; their first line names that commit and the scoring commit `ea852f0`.

The bootstrap is over subjects (`subjectid`): 2,000 resamples, seed 20261118, percentile intervals; a drawn subject brings all of its cases (`btc/models/bootstrap.py`). A contrast uses the patients with `entered` true. The fast-cycling contrast restricts the patients to `fast_cycling` true and draws its own resamples. Δ is the mean RMSE of the model minus that of the comparator. The relative reduction is 1 minus the ratio of the two means.

The case ID and `subjectid` are VitalDB case-table fields. They are kept in `folds.csv` and in the per-patient tables because the subject bootstrap and the folds cannot be reproduced without them.

### What the committed tables allow you to recompute

- **From the per-patient tables:** every patient-level mean RMSE with its CI, Δ and the relative reduction with their CIs, G and g, N, K and the fast-cycling contrast, and the yardstick N, D, F and T − T-P1 (S-dev, all and fast-cycling); Sentence 2, the cesarean line, all S-ob contrasts, the S-ob yardstick with its inflation-artefact variant (`yardstick` in `sob/frozen_scoring/summary.json`, repeated as `yardstick_with_ia` in `sob/after_frozen_scoring/summary.json`) and the S-ob sensitivity analysis. Subsets defined per case can be scored from the same tables once each case is labelled: sex and cerclage from the VitalDB case table, and the vasopressor stratum from the case table and the pump track (`REPORT.md`, "Vasopressor strata, S-ob").
- **Only with a rerun:** the other contents of `sdev/secondary/` (the refitted variants and ablation, the fold seeds, the gradient-boosting check, the S-dev inflation-artefact variant of the yardstick, subsets defined per forecast such as h-bins, waveform availability and quality strata, hypotension, large changes, intervals, the training base per bin, penalties and data quality) need `scripts/run_confirmatory.py --with-secondary` at tag `feature-freeze` on the derived S-dev tables. The other contents of `sob/after_frozen_scoring/`, the gradient-boosting appendix included, also need the cesarean data and `scripts/run_frozen_scoring.py after` at tag `models-frozen` (last section).

## File map

**`data_checksums.csv`**: the data snapshot of `SPEC.md` §10, the published list for both cohorts. The SHA-256 of every stored array of the downloaded cases (columns `cohort`, `caseid`, `array`, `dtype`, `n`, `sha256`; checksums only, no data): 3,773 rows for the 252 S-dev cases, downloaded on 2026-11-19 (UTC), and 325 rows for the 21 S-ob cases, downloaded at the frozen scoring on 2026-11-22 (UTC). The dates come from the local download logs, which are not committed. `scripts/download.py` writes each cohort's list (`data/raw/<cohort>_checksums.csv`); `scripts/fill_site_config.py --collect-checksums` joins them. A fresh download writes the same list, except the `meta_json` rows (one per case): they hash each case's download metadata, which includes the download time. A difference in any other row shows a change in the public data.

**`sdev/`**: the confirmatory S-dev run (`SPEC.md` §9 steps 4–5) and the S-dev secondary analyses, both at tag `feature-freeze`.
- `MANIFEST.json`: SHA-256 of every file of the run; `meta` gives the commit, the tag, the git state and the SHA-256 of the derived input tables. The secondary run (2026-11-23) wrote this file anew: `meta` and the 11 entries of the confirmatory run are unchanged, and 16 entries for `secondary/` were added.
- `data_checksums.csv`: the S-dev part of `data_checksums.csv`, published earlier with the confirmatory result (same rows, without the `cohort` column). It is a copy of `data/raw/sdev_checksums.csv`, written by `scripts/download.py`.
- `confirmatory/summary.json`: the question, the answer line, Sentences 1 and 3, the selected row of the decision table with G and g, every contrast, the fast-cycling contrast, the yardstick, and the case counts (analysable, with a recorded neuromuscular blocker, analysed).
- `confirmatory/headline.md`: the answer line and Sentences 1 and 3.
- `confirmatory/contrasts.csv`, `confirmatory/contrasts_fast_cycling.csv`: Δ and relative reduction with 95% CIs.
- `confirmatory/model_summary.csv`: per model, patient-level mean and SD of RMSE, MAE, mean signed error, R² and FIT, the pooled RMSE, and the counts.
- `confirmatory/per_patient_primary.csv`: per case and model, the per-patient metrics and their sums (`n`, `sse`, `sae`, `se`, `sst`).
- `confirmatory/per_patient_primary_wide.csv`, `confirmatory/per_patient_yardstick.csv`, `confirmatory/public_tables.json`: see "Per-patient tables".
- `confirmatory/m2_structure.csv`: the four M2 structures and the one chosen (`SPEC.md` §4.1).
- `confirmatory/penalties.csv`: the penalty chosen for every chain, outer fold, node and model (`SPEC.md` §4.5).
- `confirmatory/folds.csv`: the outer and inner fold of every analysed case.
- `secondary/`: the S-dev secondary analyses and the gradient-boosting check (`SPEC.md` §6.6–§6.8), written by `scripts/run_confirmatory.py --with-secondary` at tag `feature-freeze` on 2026-11-23, after the S-ob scoring (`REPORT.md`, "Secondary analyses (S-dev)"):
  - `summary.json`: the yardstick with the inflation-artefact variant T-ia and the plain 2-reading-ahead M2 (`yardstick_all`, `yardstick_fast_cycling`), the fold-seed range, the gradient-boosting contrasts with `meets_rule_with_placebo` and `readme_line`, the S-dev half of the README hypotension line (`hypotension_line_sdev`), data quality, and the counts of the sensitivity analysis that includes the cases with a recorded neuromuscular blocker.
  - `restricted_summary.csv`, `restricted_contrasts.csv`: the confirmatory outer-fold predictions scored on h-bins, the waveform-available subset, the fast-cycling cases, each sex, the quality strata and the cerclage cases; nothing refitted.
  - `variants_summary.csv`, `variants_contrasts.csv`: the refitted analyses: horizon structure per-bin, P2, template-correlation quality, h replaced by the median previous interval, 2-reading-ahead targets, secondary window, SBP models, the feature-group ablation, and the cases with a recorded neuromuscular blocker included.
  - `fold_seed_sensitivity.csv`: M4 vs M2 under fold seeds 20261119–20261128 (point estimates).
  - `sensitivity_including_nmb_folds.csv`: the folds of the analysis that includes the cases with a recorded neuromuscular blocker (214 cases, seed 20261118).
  - `penalties.csv`: the penalty chosen for every outer fold, node and model of each refitted analysis. The `chain` column does not name the analysis. The rows follow the order of the analyses in `secondary_sdev` (`btc/models/pipeline.py`), and each analysis starts again at outer fold 0: horizon structure per-bin, P2, template-correlation quality, h replaced, 2-reading-ahead targets, secondary window, SBP models, SBP models with P1, the yardstick inflation-artefact variant, the cases with a recorded neuromuscular blocker included, the ablation groups (heart rate, pulse arrival time, shape, beat-amplitude variability, availability indicators), and the fold seeds 20261119–20261128.
  - `dropped_blocks_outer_folds.csv`: how often each added block was dropped in the outer folds of the confirmatory run, with real and with donor signals.
  - `gb_summary.csv`, `gb_settings.csv`: the gradient-boosting arms (with M2 for comparison): per-model metrics, and the setting chosen per arm and outer fold (learning rate, maximum leaf nodes, iterations).
  - `hypotension.csv`, `large_changes.csv`, `conformal.csv`, `training_base_per_bin.csv`: new-onset hypotension under the three definitions, the large-change table, the cross-conformal intervals per h-bin, and the S-dev training targets and patients per h-bin.
  - Not committed: `coefficients_all_sdev.csv` (see above).

**`frozen/`**: every confirmatory model refitted on all analysed S-dev cases (`SPEC.md` §4.5, §9 step 6), written by `scripts/run_frozen_scoring.py freeze`.
- `FROZEN.json`: the SHA-256 of `models.pkl`, the M2 structure, the models of each chain, the S-dev case counts and the Python, NumPy, pandas, scikit-learn and SciPy versions.
- `models.pkl`: the fitted models (Python pickle; `scripts/run_frozen_scoring.py` checks its SHA-256 before it loads it).
- `coefficients.csv`: every fitted model in text. One row per design column: the block (cuff, hr or waveform), that block's penalty, the standardised coefficient, the scaler mean and SD, and the imputation median. The docstring of `coefficient_table` in `btc/models/secondary.py` gives the formula. The models can be applied from this file without `models.pkl`.
- `penalty_path.csv`: for every chain, node and stage of the refit, the inner cross-validation criterion of every candidate penalty and of dropping the block, and the one chosen. This is a by-product of the registered selection, not a pre-specified output.
- `MANIFEST.json`: SHA-256 of the files above; `meta` gives the SHA-256 of the S-dev input tables.

**`sob/`**: the cesarean cases, written by `scripts/run_frozen_scoring.py` at tag `models-frozen` (commit `ea852f0`).
- `frozen_scoring/`: the one frozen scoring (`SPEC.md` §9 step 7), written by `score`.
  - `summary.json`: Sentence 2 and the cesarean line, the case counts (analysable, with a recorded neuromuscular blocker, analysed), waveform coverage and the coverage rule, every contrast and the yardstick (with T-ia and the plain 2-reading-ahead M2).
  - `headline.md`: Sentence 2 and the cesarean line.
  - `contrasts.csv`, `model_summary.csv`, `per_patient_primary.csv`: as in `sdev/confirmatory/`. The `meets_rule` column is not interpreted for S-ob: the rule is defined on S-dev (`SPEC.md` §6.3).
  - `per_patient_primary_wide.csv`, `per_patient_yardstick.csv`, `public_tables.json`: see "Per-patient tables" (exported at commit `44cfb63`).
  - `STATE.json`: the state of the one scoring (`status`, `commit`, `rerun_after_crash`). `score` refuses to run again once it records a completed run.
  - `MANIFEST.json`: SHA-256 of the result files written by `score` (not `STATE.json`), the withheld files and the sensitivity analysis included; `meta` gives the commit, the tag, the git state, the SHA-256 of the five S-ob input tables and of `frozen/models.pkl`.
  - `sensitivity_including_nmb/`: the frozen models scored, in the same run, on all analysable cases, the one with a recorded neuromuscular blocker included (`SPEC.md` §2.2, §6.7): `summary.json` (counts, the case IDs with a blocker, every contrast), `contrasts.csv`, `model_summary.csv`, `per_patient_primary.csv`, and the per-patient table with its `public_tables.json`. It covers the ladder and its placebo versions, not the yardstick models.
- `frozen_scoring_log.jsonl`: every attempt at the frozen scoring and every later S-ob step, with its time (UTC), event and commit. It holds four events: `started`, `completed`, `after_frozen_scoring_written`, `site_cases_written`.
- `after_frozen_scoring/`: the analyses computed after the frozen scoring (`SPEC.md` §9 step 8), written by `after`. They are labelled as such and cannot change Sentences 1–3. Their `meta` records `state` "dirty" with `status` ["?? results/sob/"]: the scoring's outputs, not yet committed when `after` ran; the guard allows untracked files inside `results/` only.
  - `summary.json`: the yardstick with the inflation-artefact variant (`yardstick_with_ia`), data quality, the gradient-boosting appendix with its settings (`SPEC.md` §6.8; appendix only), and the S-ob half of the README hypotension line (`hypotension_line_sob`).
  - `restricted_summary.csv`, `restricted_contrasts.csv`: the frozen forecasts scored on h-bins, the waveform-available subset, the quality strata and the vasopressor strata.
  - `variants_summary.csv`, `variants_contrasts.csv`: the refitted analyses (models fitted on all analysed S-dev cases), scored on S-ob: P2, template-correlation quality, horizon structure per-bin, h replaced, 2-reading-ahead targets, secondary window, SBP models and the feature-group ablation.
  - `hypotension.csv`, `large_changes.csv`, `conformal.csv`: as in `sdev/secondary/`; the interval half-widths come from all S-dev outer-fold residuals.
  - `readme_block5.md`: the README table (S-dev and S-ob columns) and the hypotension line. The line holds the S-ob half only, because `sdev/secondary/hypotension.csv` did not exist when `after` ran; `REPORT.md` ("Hypotension line, both cohorts") gives the line for both cohorts.
  - `MANIFEST.json`: SHA-256 of the files above.
  - `exploratory_encoder/`: the cesarean step of the exploratory encoder check (`ENCODER_PLAN.md` §7), run once at commit `dda1d6d` after the frozen scoring; appendix only. `summary.json`, `summary.md` (with the departures from the plan), `contrasts.csv`, `model_summary.csv`, `refit_penalties.csv`, the per-patient tables (`per_patient_primary.csv`, `per_patient_primary_wide.csv`, `public_tables.json`) and `MANIFEST.json`. The per-forecast `predictions_primary.csv` is listed in the manifest and not committed.

### `sdev/exploratory_encoder/` (`ENCODER_PLAN.md`)

The exploratory encoder check on S-dev: PaPaGei-S embeddings in place of the hand-crafted waveform features, in the same ridge model and outer folds. `build_check.json` (the signal-level build check, run at `a07e0d2` before any blood-pressure model), `summary.json`, `summary.md` (the one-line summary and the plan's disclosures), `contrasts.csv`, `model_summary.csv`, `penalties.csv`, `dropped_blocks_outer_folds.csv`, the per-patient tables and `MANIFEST.json`, written at `d66ad63`. Every file's `meta` records the plan's SHA-256 and its public commit. The embeddings themselves are derived from VitalDB signals and are not published; `scripts/build_embeddings.py` rebuilds them (`btc/SCHEMA.md` §4d). The S-dev contrasts can be recomputed from `per_patient_primary_wide.csv`. The per-forecast `predictions_primary.csv` is listed in the manifest and not committed (first section).

## Regenerating the withheld files

1. At tag `feature-freeze`, from a clean tree (the derived tables record the commit that built them), download the S-dev cases and build the derived tables in the default `data/` folder (not committed): `scripts/download.py --cohort sdev`, then `scripts/build_instances.py`, `scripts/build_windows.py` and `scripts/build_features_wf.py`, each with `--cohort sdev`.
2. At the same tag, run `python scripts/run_confirmatory.py --feature-tag feature-freeze --out-dir <folder>`. The confirmatory result is the run made at that tag (`SPEC.md` §9 step 5). With the same derived tables (`meta.inputs_sha256` in `sdev/MANIFEST.json`) and the locked environment (`uv.lock`), every file of the confirmatory run listed in `sdev/MANIFEST.json`, the two withheld files included, has the same SHA-256 in `<folder>/sdev/` (on the same platform; see above). With `--with-secondary`, the run also writes `<folder>/sdev/secondary/`, the omitted coefficient table included, and its files can be compared with the `secondary/` entries.
3. Back at the latest commit, `python scripts/export_public_tables.py --results <folder>` writes the per-patient tables next to them.

Regenerating `sob/` also needs the cesarean cases, which the code downloads only with `BTC_FROZEN_SCORING=1` (`SPEC.md` §9 step 7), and `scripts/run_frozen_scoring.py score --frozen-tag <tag>` at the frozen-model tag (`after` and `series` need the same `--frozen-tag` and checks).

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5), accessed through the public API at run time. Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/, and the VitalDB Data Use Agreement.
