# between-the-cuffs: report

This file gives the confirmatory result on the non-cesarean cases, how to read it, the frozen models, the implementation choices, and every deviation from the registration (`PREREG.md` and `SPEC.md`, version 1.5).

Two cohorts are used. **S-dev** is the development cohort: the non-cesarean spinal cases, on which every model is fitted and cross-validated. The answer to the question comes from S-dev. **S-ob** is the cesarean cohort: an external case series, scored once with the frozen models.

**Status of this version** (committed with tag `models-frozen`):

| Step (`SPEC.md` §9) | Status |
|---|---|
| 1. Registration, version 1.5 | done; commit 2026-11-21, public release 2026-11-22 (tag `prereg-v1.5`) |
| 2. Pipeline built and checked on S-dev | done; see "Implementation choices" |
| 3. Feature code frozen | done; commit 2026-11-21, public release 2026-11-22 (tag `feature-freeze`) |
| 4–5. Confirmatory S-dev run | done, 2026-11-22; result below |
| 6. Models frozen | done, 2026-11-23 (tag `models-frozen`) |
| 7. S-ob scored once | **not done yet** |
| 8. Secondary analyses, gradient-boosting check, replays | **not run yet** |

## Registration

| Version | Commit | Date | `PREREG.md` SHA-256 | `SPEC.md` SHA-256 |
|---|---|---|---|---|
| 1.4 | `fca645e` | 2026-11-19 | `7fc6684469a8275eae37f2f3e5178ee267a00d13e040962030e1322c79a00c60` | `91ec51922d62402076bb1d219405b2bb5820918ee3a41fb0f805e18b6309bb29` |
| 1.5 | `59939b7` | 2026-11-21 | `1a866d6a991bbacc4827da3b3bdaa47b4c483e76a41a523530525aedc5eee7d8` | `ccfbb764f836f55a68ac6cc0f352722619d66b9ff28154b1012526d688a78867` |

Version 1.5 replaced version 1.4 before any model was fitted; `PREREG.md` §11 lists the changes. This is a self-registration by git commit. The dates in the table are commit dates, which the author sets. The `published_at` time of each GitHub release is set by GitHub's server; it can be read on the repository's [Releases page](https://github.com/starriet9/between-the-cuffs/releases) or from the GitHub releases API, in UTC. Other dates in this repository are Korea Standard Time (KST, UTC+9) unless marked. The version 1.5 text was first committed locally on 2026-11-20 and amended on 2026-11-21, before its first push, to add the neuromuscular-blocker rule and the reviewer-guide clause; only the pushed commit is the registration. No archive snapshot was made.

| Tag | Commit | Commit date (set by the author, KST) | Release published (set by GitHub, UTC) |
|---|---|---|---|
| `prereg-v1.5` | `59939b7` | 2026-11-21 02:58 | 2026-11-22 11:50:39 |
| `feature-freeze` | `0669d0c` | 2026-11-21 02:58 | 2026-11-22 11:53:08 |
| `models-frozen` | the commit that adds this version of this file | 2026-11-23 04:29 | recorded by GitHub on publication |

**What these records can and cannot show.** GitHub's records show that the registration and the feature tag were public before any result file was: the result files are first published with tag `models-frozen`. They cannot show when a model was first fitted. The author's record is this: the confirmatory run (`SPEC.md` §9 step 5) was made on 2026-11-22 at 21:55 KST (12:55 UTC), about an hour after both releases; and no model was fitted to real data before that run, because earlier runs of the model code used synthetic data only (`btc/models/synthetic.py`, `tests/`). The result files hold no timestamps, so the run time comes from local file times and cannot be checked from the repository.

`PREREG.md` and `SPEC.md` are unchanged since the registration commit (`git diff prereg-v1.5 models-frozen -- PREREG.md SPEC.md` is empty); their SHA-256 values are those of the 1.5 row. Statements in them such as "No model has been fitted to any case so far" (`SPEC.md` §9) describe the time of registration.

## Confirmatory result, S-dev

**Run.** `scripts/run_confirmatory.py` at tag `feature-freeze` (commit `0669d0c`). `meta` in `summary.json` and in `MANIFEST.json` records the commit, the tag at HEAD and that no tracked file was modified; every CSV and Markdown file carries the commit in its first line. Inputs: the derived S-dev tables whose SHA-256 values are in `results/sdev/MANIFEST.json` (`meta.inputs_sha256`). Outputs: `results/sdev/confirmatory/` (file map in `results/README.md`). The run chose the M2 structure and then fitted M2, M3, M4, the placebo models M3-P1 and M4-P1, and the yardstick models L, LH, T, LH-P1, T-P1 and U, each in 5 outer folds with penalties chosen by inner cross-validation (`SPEC.md` §4.1, §4.5). It did not run the secondary analyses (see "Secondary analyses").

**Headline row: 6** (M4 vs M2 does not meet the rule; `summary.json`, `decision`). Placebo validity was not evaluated, because it is checked only for a contrast that meets the rule (`PREREG.md` §6). Unrounded, G = 0.163% and g = 0.0142 mmHg (`summary.json`, `decision`).

The registered sentences, as written by the run (`results/sdev/confirmatory/headline.md`):

> **Answer line.** No. In 200 non-cesarean spinal cases (cross-validated), heart rate and the pleth waveform did not meet the pre-specified rule (−0.1%, from 8.67 to 8.68 mmHg; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 0.2% (0.01 mmHg) are not compatible with these data (approximate 95% CI), for this feature set and a linear model.
>
> **Sentence 1.** In 200 non-cesarean spinal cases (cross-validated), heart rate and pleth waveform features did not meet the pre-specified rule for lowering next-reading MAP forecast error beyond a cuff-only third-order model (−0.1%, Δ 0.01 mmHg, 95% CI −0.01 to 0.03; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 0.2% (0.01 mmHg) are not compatible with these data (approximate 95% CI, see §6.2 of SPEC), for this feature set and a linear model.
>
> **Sentence 2.** Pending; written by the frozen scoring.
>
> **Sentence 3.** In the 30 fast-cycling development cases (median cuff interval < 90 s): 0.3% (Δ −0.03 mmHg, 95% CI −0.07 to 0.01); one extra cuff reading's value recovered by the added signals: −0.9% (95% CI −2.2% to −0.1%) (not distinguishable from placebo inputs).

Terms used below. MAP is mean arterial pressure. The pleth is the pulse-oximeter waveform. RMSE is the root-mean-square forecast error of one patient, and "mean RMSE" is its mean over patients. "Cross-validated" means that each patient's forecasts come from models fitted without that patient. A penalty is the strength of the ridge shrinkage on one block of coefficients; the three blocks are cuff, heart rate and waveform. "Dropped" means that inner cross-validation left a block out.

### Reading the result

*Post hoc and explanatory, not pre-specified (`PREREG.md` §10). Written after the S-dev result was known and before the cesarean cases were scored.*

This section changes no registered sentence. Every number in it is in `results/sdev/confirmatory/` or `results/frozen/`, or follows from those files by the arithmetic stated.

1. **Signs.** Δ is the model's patient-level mean RMSE minus the comparator's, so a positive Δ means the added signals did worse. The relative reduction is 1 minus the ratio of the two means, so a negative percentage means the error rose. For M4 vs M2, Δ = +0.009 mmHg and the relative reduction is −0.11%.
2. **What was compared.** The primary contrast compares two fitted procedures: M2 (cuff only) and M4 (cuff, heart rate and waveform), each with its penalties chosen by inner cross-validation as registered. Δ and its interval describe the difference in their cross-validated error. They do not measure how much information about the next reading the signals contain.
3. **Heavy shrinkage, so a narrow interval.** Inner cross-validation kept both added blocks in all 5 outer folds but chose strong penalties: 3.16 to 31.6 for the heart-rate block and 3.16 to 1000 for the waveform block, against 1 for the cuff block in every fold (`penalties.csv`). Under the objective of `SPEC.md` §4.5, a penalty λ multiplies the coefficient of an isolated standardised column by 1/(1 + λ): 1 halves it, 10 leaves 9% of it and 1000 leaves 0.1%. M4's forecasts therefore stay close to M2's. Per patient, M4's RMSE was lower than M2's in 102 of the 200 patients and higher in 98; the difference has a median absolute value of 0.08 mmHg, a largest absolute value of 0.94 mmHg and an SD of 0.18 mmHg, while both models' mean RMSE is about 8.7 mmHg (`per_patient_primary.csv`). Two nearly identical sets of forecasts have nearly identical errors, so their paired difference has a narrow interval. A normal approximation from these 200 differences gives almost the same interval as the bootstrap (−0.015 to 0.033 mmHg). Had every fold dropped both blocks, M4 would equal M2, Δ would be exactly zero, and the bound sentence would be omitted (G ≤ 0; `SPEC.md` §6.3).
4. **What the bound sentence covers.** "Gains larger than 0.2% (0.01 mmHg) are not compatible with these data" refers to this procedure in this cohort: ridge regression on these 11 features (3 heart-rate features, and 4 waveform features in 2 forms each, every one also multiplied by the horizon; `SPEC.md` §4.2–§4.5), with penalties chosen as registered. It is not a bound for other features, other models, other monitors or other patients. The two numbers in the sentence come from two intervals: G = 0.163% is the upper 95% limit of the relative reduction, and g = 0.0142 mmHg is minus the lower 95% limit of Δ. The registered formats give percentages one decimal and mmHg two, which is why 0.2% and 0.01 mmHg do not match each other exactly (0.163% of 8.67 mmHg is 0.014 mmHg). The interval also treats the five fold models as fixed (`SPEC.md` §6.2). The fold-seed sensitivity analysis, not run yet, will show how far Δ moves with the folds.
5. **Less shrinkage was available, and it did worse.** The penalty grid starts at 0.001, which is almost no penalty, and ends at 1000 (13 values; `SPEC.md` §4.1). Each added block could also be dropped. For the refit on all analysed S-dev cases, `results/frozen/penalty_path.csv` gives the cross-validation criterion of every candidate ("Frozen models" below). With a penalty of 0.001 the added blocks raised the criterion by 0.12 mmHg (heart rate) and 0.17 mmHg (waveform) against leaving the block out. At the chosen penalties they lowered it by 0.0007 and 0.003 mmHg. A waveform block built from another patient's signals lowered it by about as much (0.003 mmHg), and in the yardstick models the donor blocks lowered it more than the patients' own. At this size, whether a block is "kept" or "dropped" says little about whether it carries signal.
6. **Positive controls.** The same scoring detects an informative input. One extra cuff reading lowers the yardstick error by D = 1.94 mmHg (95% CI 1.46 to 2.45), from 10.45 mmHg without it (L) to 8.51 mmHg with it (U), in 122 patients. M2 lowers error by 31.0% (95% CI 26.0% to 35.7%) against the running mean (M00) and by 4.7% (95% CI 3.0% to 6.3%) against repeating the last reading (M0). On synthetic data with a strong planted waveform or heart-rate signal, the pipeline finds it, attributes it to its block and does not find it in the placebo version (`tests/test_models_end_to_end.py`). These controls check the code path. They do not show how weak a waveform signal the pipeline could detect.
7. **Between the cuffs.** The primary forecast is issued when a new cuff reading arrives. That is when the cuff value is newest and a continuous signal has the least to add. The withheld-cuff yardstick (`SPEC.md` §3) asks the harder question: with the latest reading's values hidden, how much of that reading's worth do the signals recover? N = −0.01 mmHg (95% CI −0.04 to 0.02) against D = 1.94 mmHg, in 122 patients and 909 triples, so F = −0.5% (95% CI −2.0% to 1.0%). In the 30 fast-cycling cases N = −0.025 mmHg (95% CI −0.048 to −0.002): the forecast with the signals (T) was slightly worse than the one without them (L), which is why F is negative in Sentence 3. A negative F does not mean that the signals took value away from the cuff.
8. **The cuff-only model is modest.** M2's mean RMSE is 8.67 mmHg, 4.7% below that of repeating the last reading (9.10 mmHg). Its mean per-patient R² is −0.64 and its mean FIT −18.8%, in the 138 patients with at least 5 forecasts and a non-zero SST. The mean is pulled down by a few patients: the median R² is −0.28, and R² is below zero in 90 of the 138 (`per_patient_primary.csv`). A negative R² means that the model did worse than that patient's own in-window mean of the observed readings, a reference that is not available in real time (`SPEC.md` §6.1). Patients have few forecasts (a median of 6 among the 200 who enter), so these per-patient values are noisy.
9. **Weighting.** With every forecast weighted equally (pooled RMSE over the 1,416 forecasts of the 200 patients), M4 scores 10.104 mmHg and M2 10.117 mmHg: M4 is 0.12% lower. This is the opposite sign to the primary metric, in which M4 is 0.11% higher. No interval was computed for the pooled difference, and both differences are far below the 5% margin. The primary metric is per patient (`SPEC.md` §6.1, following the per-patient metrics of the earlier work cited in `SPEC.md` §1), so every patient counts once. A pooled RMSE gives more weight to patients with more forecasts: the 30 fast-cycling patients are 15% of the patients but hold 414 of the 1,416 forecasts (29%).
10. **Placebo.** M4 on another patient's signals (M4-P1) had a mean RMSE of 8.675 mmHg, and M4 on the patient's own signals 8.681 mmHg (M4 − M4-P1 = 0.007 mmHg, 95% CI −0.018 to 0.032). Inner cross-validation kept the heart-rate block in all 5 folds with real signals and dropped it in 4 of 5 with donor signals, yet M3 was not better than M2 in the outer folds (Δ 0.008 mmHg, 95% CI −0.006 to 0.024).

### Who enters

- **Primary outcome.** The 203 analysed cases ("Cohort flow, S-dev" below) have 1,421 primary next-reading forecasts; 1,386 of them (97.5%) have waveform features. Three cases have fewer than 3 forecasts: 3540 (2), 4896 (2) and 5707 (1). The registered minimum for RMSE is 3 scored forecasts per patient (`SPEC.md` §6.1, `PREREG.md` §6), so they do not enter. They stay in the folds (`folds.csv`) and their forecasts are used in fitting; only the scoring leaves them out. N = 200 patients with 1,416 forecasts, a median of 6 per patient (interquartile range 4 to 9, range 3 to 24). Each analysed case belongs to a different subject (`folds.csv`), so the subject bootstrap resamples 200 subjects.
- **Fast-cycling cases** (median cuff interval < 90 s, `SPEC.md` §6.7): 30 of the analysed cases. All 30 have at least 3 forecasts (K = 30; 414 forecasts) and at least 3 scored triples (365 triples), so F in Sentence 3 uses all 30.
- **Withheld-cuff yardstick, all S-dev.** 152 cases have at least one triple (952 triples); 122 of them have at least 3 and enter, with 909 triples. A triple needs three consecutive valid readings within 400 s (`SPEC.md` §3), and S-dev cuffs cycle slowly (median interval 154 s in the analysable cases; `SPEC.md` §3), so fewer cases enter here than in the primary outcome.

### Error by model

Patient-level mean ± SD over the 200 entering patients, in mmHg, with the 95% CI of the mean from the subject bootstrap (2,000 resamples, seed 20261118). R² and FIT use the 138 patients with at least 5 forecasts and a non-zero SST (65 of the 203 do not qualify). Mean signed error is forecast minus observed. Pooled RMSE weights every forecast equally (1,416 forecasts of the 200 entering patients). Source: `model_summary.csv`. Tables in this file give more decimals than the registered sentences; the files give full precision.

| Model | RMSE, mean ± SD (95% CI) | MAE, mean ± SD | Mean signed error, mean ± SD | R², mean ± SD | FIT %, mean ± SD | Pooled RMSE (95% CI) |
|---|---|---|---|---|---|---|
| M00 | 12.57 ± 8.13 (11.52 to 13.66) | 11.15 ± 7.64 | 7.89 ± 8.65 | −2.69 ± 4.59 | −71.3 ± 87.2 | 14.729 (13.21 to 16.37) |
| M0 | 9.10 ± 5.97 (8.34 to 9.92) | 7.22 ± 4.63 | 1.87 ± 3.09 | −0.73 ± 1.33 | −24.6 ± 42.7 | 10.654 (9.69 to 11.65) |
| M1 | 12.56 ± 7.58 (11.59 to 13.63) | 10.03 ± 5.91 | −0.59 ± 3.55 | −2.90 ± 4.81 | −79.4 ± 82.8 | 14.321 (13.18 to 15.52) |
| M2 | 8.67 ± 5.57 (7.95 to 9.44) | 6.94 ± 4.43 | 0.18 ± 3.90 | −0.64 ± 1.92 | −18.8 ± 47.6 | 10.117 (9.20 to 11.06) |
| M3 | 8.68 ± 5.56 (7.96 to 9.44) | 6.95 ± 4.42 | 0.18 ± 3.92 | −0.64 ± 1.87 | −18.9 ± 47.2 | 10.119 (9.20 to 11.06) |
| M4 | 8.68 ± 5.55 (7.97 to 9.44) | 6.95 ± 4.42 | 0.15 ± 3.93 | −0.64 ± 1.91 | −19.2 ± 47.5 | 10.104 (9.20 to 11.05) |
| M3-P1 | 8.67 ± 5.57 (7.95 to 9.44) | 6.95 ± 4.43 | 0.18 ± 3.90 | −0.64 ± 1.92 | −18.9 ± 47.6 | 10.117 (9.20 to 11.06) |
| M4-P1 | 8.67 ± 5.57 (7.95 to 9.44) | 6.95 ± 4.43 | 0.18 ± 3.90 | −0.64 ± 1.92 | −18.9 ± 47.6 | 10.118 (9.20 to 11.06) |

M00 is the running mean of the readings so far; M0 repeats the last reading; M1 extends the trend of the last three readings; M2 is the cuff-only third-order ridge model; M3 adds heart rate; M4 adds the waveform features; P1 marks a model refitted on another patient's signals (`SPEC.md` §4.1, §6.6). Over all 1,421 forecasts of the 203 analysed cases, the pooled RMSE is 10.103 mmHg for M2, 10.105 for M3 and 10.091 for M4 (`pooled_rmse` in the same file).

### Contrasts

200 patients in every row. Δ = mean RMSE of the model minus that of the comparator; relative reduction = 1 − (model / comparator). Source: `contrasts.csv`. No contrast with M3 or M4 as the model meets the rule.

| Model | Comparator | Mean RMSE, model / comparator (mmHg) | Δ (95% CI), mmHg | Relative reduction (95% CI) |
|---|---|---|---|---|
| M4 | M2 | 8.681 / 8.672 | 0.009 (−0.014 to 0.033) | −0.11% (−0.38% to 0.16%) |
| M3 | M2 | 8.680 / 8.672 | 0.008 (−0.006 to 0.024) | −0.09% (−0.28% to 0.07%) |
| M4 | M3 | 8.681 / 8.680 | 0.001 (−0.017 to 0.019) | −0.01% (−0.22% to 0.20%) |
| M4-P1 | M2 | 8.675 / 8.672 | 0.002 (−0.003 to 0.008) | −0.03% (−0.09% to 0.03%) |
| M3-P1 | M2 | 8.675 / 8.672 | 0.002 (−0.002 to 0.007) | −0.03% (−0.08% to 0.03%) |
| M4 | M4-P1 | 8.681 / 8.675 | 0.007 (−0.018 to 0.032) | −0.08% (−0.36% to 0.20%) |
| M3 | M3-P1 | 8.680 / 8.675 | 0.006 (−0.010 to 0.022) | −0.07% (−0.26% to 0.12%) |
| M2 | M0 | 8.672 / 9.100 | −0.428 (−0.585 to −0.269) | 4.70% (3.04% to 6.26%) |
| M2 | M00 | 8.672 / 12.567 | −3.895 (−4.710 to −3.114) | 30.99% (26.04% to 35.73%) |
| M2 | M1 | 8.672 / 12.564 | −3.892 (−4.440 to −3.391) | 30.98% (28.05% to 34.13%) |

### Fast-cycling cases

The 30 fast-cycling cases (414 forecasts), scored from the same outer-fold predictions; nothing is refitted on the subset (`SPEC.md` §6.7). Source: `contrasts_fast_cycling.csv`.

| Model | Comparator | Mean RMSE, model / comparator (mmHg) | Δ (95% CI), mmHg | Relative reduction (95% CI) |
|---|---|---|---|---|
| M4 | M2 | 8.939 / 8.964 | −0.026 (−0.074 to 0.014) | 0.29% (−0.17% to 0.80%) |
| M3 | M2 | 8.959 / 8.964 | −0.005 (−0.022 to 0.014) | 0.06% (−0.17% to 0.25%) |
| M4 | M3 | 8.939 / 8.959 | −0.021 (−0.067 to 0.017) | 0.23% (−0.19% to 0.76%) |
| M4-P1 | M2 | 8.967 / 8.964 | 0.002 (−0.003 to 0.008) | −0.03% (−0.10% to 0.03%) |
| M3-P1 | M2 | 8.965 / 8.964 | 0.001 (−0.003 to 0.004) | −0.01% (−0.04% to 0.03%) |
| M4 | M4-P1 | 8.939 / 8.967 | −0.028 (−0.076 to 0.013) | 0.31% (−0.15% to 0.83%) |
| M3 | M3-P1 | 8.959 / 8.965 | −0.006 (−0.022 to 0.014) | 0.06% (−0.16% to 0.25%) |
| M2 | M0 | 8.964 / 9.246 | −0.281 (−0.485 to −0.069) | 3.04% (0.79% to 5.31%) |
| M2 | M00 | 8.964 / 12.436 | −3.471 (−5.380 to −1.761) | 27.91% (18.34% to 34.98%) |
| M2 | M1 | 8.964 / 12.379 | −3.415 (−4.214 to −2.614) | 27.59% (21.51% to 33.41%) |

### Withheld-cuff yardstick

The forecasts of reading r+2 (`SPEC.md` §3): L, issued at reading r+1 with that reading's values withheld and no added signals; U, M2 issued at reading r+1; T, L plus the heart-rate and waveform blocks; LH, the intermediate model, L plus the heart-rate block. T-P1 and LH-P1 use donor signals. RMSE values are patient-level means in mmHg. Source: `summary.json` (`yardstick_all`, `yardstick_fast_cycling`).

| | All S-dev | Fast-cycling |
|---|---|---|
| Cases entering (at least 3 triples) / cases with a triple | 122 / 152 | 30 / 30 |
| Triples in entering cases / all triples | 909 / 952 | 365 / 365 |
| RMSE of L / U / T / T-P1 | 10.450 / 8.513 / 10.460 / 10.492 | 11.468 / 8.854 / 11.493 / 11.478 |
| RMSE of LH / LH-P1 | 10.447 / 10.467 | 11.467 / 11.454 |
| D = RMSE L − RMSE U (95% CI) | 1.937 (1.456 to 2.452) | 2.614 (1.484 to 3.929) |
| N = RMSE L − RMSE T (95% CI) | −0.010 (−0.037 to 0.019) | −0.025 (−0.048 to −0.002) |
| F = N / D (95% CI) | −0.5% (−2.0% to 1.0%) | −0.9% (−2.2% to −0.1%) |
| RMSE T − RMSE T-P1 (95% CI) | −0.032 (−0.103 to 0.041) | 0.014 (−0.049 to 0.086) |
| Placebo guard passed | no | no |

F is reported in both, because the 95% CI of D lies above zero (no bootstrap resample had D ≤ 0). A negative F means that T, with the added signals, had a slightly larger error than L, without them. The placebo guard did not pass, because the CI of RMSE T − RMSE T-P1 does not lie below zero; Sentence 3 therefore ends "(not distinguishable from placebo inputs)".

### M2 structure

Chosen on the outer-fold predictions of the patients with at least 3 forecasts (`SPEC.md` §4.1, §4.5). Source: `m2_structure.csv`.

| M2 form | Horizon structure | Patient-level mean RMSE, mmHg | Chosen |
|---|---|---|---|
| (a) level, slope, curvature | one model per h-bin | 8.759 | |
| (a) level, slope, curvature | pooled, with horizon interactions | 8.672 | yes |
| (b) raw-lag AR(3) | one model per h-bin | 8.864 | |
| (b) raw-lag AR(3) | pooled, with horizon interactions | 8.732 | |

The chosen structure is used for M2, M3, M4 and their placebo versions; the yardstick models use the same form, pooled (`SPEC.md` §3).

### Penalties per outer fold

The penalty chosen by inner cross-validation in each outer fold, from the grid 0.001 … 1000 (13 values), or "dropped": the block is left out and the model returns the forecast of the model below it (`SPEC.md` §4.5). Each row is one stage of a chain and gives the penalty of the block that stage adds: M2 and L the cuff block, M3 and LH the heart-rate block, M4 and T the waveform block; U is M2 issued one reading later. Lower blocks keep their penalties. All models use the pooled structure. Source: `penalties.csv` (chain names as in the file).

| Chain | Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| `ladder:real/real` | M2 | 1 | 1 | 1 | 1 | 1 |
| `ladder:real/real` | M3 | 10 | 10 | 10 | 31.6 | 3.16 |
| `ladder:real/real` | M4 | 10 | 1000 | 10 | 3.16 | 10 |
| `ladder:p1/p1` | M2 | 1 | 1 | 1 | 1 | 1 |
| `ladder:p1/p1` | M3-P1 | 10 | dropped | dropped | dropped | dropped |
| `ladder:p1/p1` | M4-P1 | dropped | 31.6 | dropped | dropped | 1000 |
| `yard:real` | L | 1 | 0.316 | 1 | 1 | 0.316 |
| `yard:real` | LH | 10 | 10 | 10 | 31.6 | 100 |
| `yard:real` | T | 1000 | 1000 | dropped | 1000 | 10 |
| `yard:p1` | L | 1 | 0.316 | 1 | 1 | 0.316 |
| `yard:p1` | LH-P1 | 10 | dropped | 100 | 1 | dropped |
| `yard:p1` | T-P1 | dropped | 1 | 1 | dropped | 1 |
| `yard:U` | U | 1 | 1 | 3.16 | 1 | 1 |

How often each added block was dropped across the outer folds (`SPEC.md` §6.7): with the patients' own signals, heart rate 0 of 5 and waveform 0 of 5; with donor signals, heart rate 4 of 5 and waveform 3 of 5.

## Frozen models (`SPEC.md` §9 step 6)

Every confirmatory model was refitted on all 203 analysed S-dev cases. Penalties were chosen by cross-validation over the outer folds of the confirmatory run (`SPEC.md` §4.5). The M2 structure was chosen again and came out the same (form (a), pooled). The refit also fits two secondary models that the cesarean scoring uses: T-ia, the inflation-artefact variant of T (with its intermediate model LH-ia), and the 2-reading-ahead M2.

**Tag** `models-frozen`, on the commit that adds this version of this file, 2026-11-23. Its GitHub release records when it was published. The refit ran on a clean tree at commit `dcd0ddd`, which `results/frozen/FROZEN.json` records in `meta` (`commit`, `state`). The tagged commit comes after that one and adds the refit's outputs and this version of the documents, so the two hashes differ. No code differs between them: `git diff dcd0ddd models-frozen -- btc scripts tests pyproject.toml uv.lock` is empty.

**`results/frozen/FROZEN.json`** (SHA-256 `6ecadc9d7ed2a569857c124beac92243d4ee74cd134bbc012751db6d81c8d47a`) gives the SHA-256 of `models.pkl` (`ad4c8b4702d152dff0f172a24502298be9579b6f61f7ea8caac858c28fd988c1`), the M2 structure, the models of each chain, the S-dev case counts and the Python, NumPy, pandas, scikit-learn and SciPy versions. Its `meta.inputs_sha256` equals that of the confirmatory run (`results/sdev/MANIFEST.json`).

**Coefficients** (`SPEC.md` §6.7): `results/frozen/coefficients.csv` is the coefficient table. It has one row per design column of every frozen model: the block (cuff, hr or waveform), that block's penalty, the standardised coefficient, the scaler mean and SD, and the imputation median. The models can be applied from this file without unpickling `models.pkl` (`tests/test_frozen_guards.py` checks this on synthetic data). In the frozen M4, the standardised coefficients (mmHg of forecast change per SD of a column) are 0.04 to 0.88 in absolute value in the cuff block, at most 0.02 in the heart-rate block and at most 0.07 in the waveform block.

**Penalties of the refit**, beside those of the outer folds ("Penalties per outer fold" above):

| Chain in `FROZEN.json` | Model (block added) | Refit on all 203 cases | Outer folds 0–4 |
|---|---|---|---|
| `main` | M2 (cuff) | 1 | 1, 1, 1, 1, 1 |
| `main` | M3 (heart rate) | 31.6 | 10, 10, 10, 31.6, 3.16 |
| `main` | M4 (waveform) | 10 | 10, 1000, 10, 3.16, 10 |
| `p1` | M3-P1 (heart rate, donor) | dropped | 10, then dropped in 4 |
| `p1` | M4-P1 (waveform, donor) | 10 | dropped, 31.6, dropped, dropped, 1000 |
| `yard_T`, `yard_TP1`, `yard_Tia` | L (cuff) | 0.001 | 1, 0.316, 1, 1, 0.316 |
| `yard_T` | LH (heart rate) | 31.6 | 10, 10, 10, 31.6, 100 |
| `yard_T` | T (waveform) | 1000 | 1000, 1000, dropped, 1000, 10 |
| `yard_TP1` | LH-P1 (heart rate, donor) | 0.1 | 10, dropped, 100, 1, dropped |
| `yard_TP1` | T-P1 (waveform, donor) | 1 | dropped, 1, 1, dropped, 1 |
| `yard_U` | U (cuff) | 1 | 1, 1, 3.16, 1, 1 |
| `yard_Tia` | LH-ia (heart rate) | 0.00316 | not in the confirmatory run |
| `yard_Tia` | T-ia (waveform) | dropped | not in the confirmatory run |
| `m2_two_ahead` | M2, 2 readings ahead (cuff) | 0.001 | not in the confirmatory run |

**Penalty path** (post hoc; a by-product of the registered selection, not a pre-specified output): `results/frozen/penalty_path.csv` gives, for every chain and stage of the refit, the cross-validation criterion of every candidate penalty and of dropping the block, and the one chosen. The criterion is the patient-level mean RMSE on the validation folds, in mmHg. It counts every case with a validation forecast, with no minimum count (`SPEC.md` §4.5), so it is not the same number as the 8.67 mmHg of the primary outcome. For the main chain, where the cuff-only criterion is 8.6040:

| Block | Chosen penalty | Criterion at the chosen penalty | Criterion at 0.001 | Criterion with the block dropped |
|---|---|---|---|---|
| heart rate (M3) | 31.6 | 8.6033 | 8.7266 | 8.6040 |
| waveform (M4) | 10 | 8.6002 | 8.7694 | 8.6033 |

Three things to note in this file:

- **The gains at the chosen penalties are very small**: 0.0007 mmHg for the heart-rate block and 0.0031 mmHg for the waveform block.
- **Donor signals gave gains of the same size or larger.** In the placebo chain the donor heart-rate block was dropped, and the donor waveform block lowered the criterion by 0.0028 mmHg. In the yardstick chains the donor blocks lowered it by 0.033 mmHg (heart rate) and 0.013 mmHg (waveform), against 0.0004 and 0.0071 mmHg for the patients' own signals. Criterion differences of this size do not separate real from placebo inputs, which agrees with the outer-fold result.
- **Two cuff-only models sit at the lower edge of the grid.** The yardstick model L and the 2-reading-ahead M2 took the smallest penalty, 0.001, where the outer folds chose 0.316 to 1 for L. LH, T, their donor versions and the inflation-artefact versions keep L's cuff penalty. The criterion is nearly flat there (L: 9.748 at 0.316 and 9.730 at 0.001), and 0.001 is close to no penalty, so a wider grid would not give a materially different model. Still, the frozen L is less shrunk than the L models behind the S-dev yardstick numbers above. These models are used only for the cesarean yardstick, which is descriptive.

**Dropped blocks.** The refit dropped the donor heart-rate block (M3-P1) and the waveform block of T-ia. The frozen M3-P1 therefore equals the frozen M2, and the frozen T-ia equals LH-ia. In the cesarean cases the contrast M3-P1 vs M2 will be exactly zero by construction.

**Unchanged code.** The feature code and the model core are unchanged from tag `feature-freeze` to tag `models-frozen`: `git diff feature-freeze models-frozen` is empty for `btc/beats.py`, `waveform.py`, `hr.py`, `cuff.py`, `pqi.py`, `rpeaks.py`, `instances.py`, `io.py`, `constants.py`, `paths.py`, `provenance.py`, `__init__.py`, `scripts/build_instances.py`, `build_windows.py`, `build_features_wf.py`, `scripts/download.py`, `scripts/select_rpeak_detector.py`, `scripts/pleth_coverage.py`, and for `btc/models/ridge.py`, `design.py`, `data.py`, `folds.py`, `analyses.py`, `bootstrap.py`, `metrics.py`, `headline.py`, `constants.py` and `gbm.py`. The same command with `--stat` lists what did change: the freeze and scoring scripts with their checks, the code that writes the result tables, the site code and the tests.

## Cesarean cases (S-ob): not scored yet

The frozen models will be scored once on the cesarean cases (`SPEC.md` §9 step 7). That step downloads the cases, recounts them under the final cuff rule, builds their features and scores every frozen model, including the placebo and yardstick models, all at tag `models-frozen`. It writes Sentence 2 and the cesarean line. In the same run it also scores the frozen models on all analysable cases, including any with a neuromuscular blocker recorded (`SPEC.md` §6.7). Until then no cesarean signal is opened (`PREREG.md` §9). The result is a case series and cannot change the answer or Sentences 1 and 3.

## Secondary analyses (S-dev): not run yet

The confirmatory run wrote the primary outcome, the decomposition (M3 vs M2, M4 vs M3), the placebo contrasts, the fast-cycling contrasts, the yardstick, the per-model metrics and the M2-structure choice (above). The other pre-specified analyses have not been run yet. They are (`SPEC.md` §6.6–§6.7): placebo P2, the feature-group ablation, the quality strata, the h-bins, the 2-reading-ahead targets, the inflation-artefact variant of the yardstick, the waveform-available subset, S-dev by sex, the secondary window, the sensitivity analyses (horizon structure, replaced h, template-correlation quality, fold seed, the cases with a neuromuscular blocker), new-onset hypotension, large changes, intervals and data quality. The gradient-boosting check (`SPEC.md` §6.8) has not been run either. All of these run after the cesarean scoring, in the registered order (`SPEC.md` §9 step 8), and will be reported here with the commit that ran them. None of them can change the answer or Sentences 1 and 3 (`SPEC.md` §6.7, §6.8, §9).

Until they are reported, two questions about the result above stay open in this file: how far Δ moves with the fold seed, and whether a non-linear learner uses the same features differently.

## Data and result files

- **No VitalDB data are redistributed** (`SPEC.md` §2.1). The per-forecast files `predictions_primary.csv` and `yardstick_predictions.csv` hold, for every forecast, the cuff value it targets and the last cuff value before it, with the case ID and the reading times. They are therefore not published. Their SHA-256 values are in `results/sdev/MANIFEST.json`.
- **The per-patient tables are published.** `per_patient_primary.csv` gives, per case and model, the number of forecasts, the sums of squared, absolute and signed errors, RMSE, MAE, R² and FIT. `per_patient_primary_wide.csv` and `per_patient_yardstick.csv` give, per case, the subject ID, the fast-cycling flag, the counts and the RMSE of every model; `scripts/export_public_tables.py` wrote them from the two withheld files, and their first line names both the export commit and the commit of the run. With them anyone can recompute every headline number: N, each mean RMSE, Δ and the relative reduction with their subject-bootstrap CIs, G and g, K and the fast-cycling contrast, and the yardstick N, D and F (`results/README.md` gives the bootstrap recipe).
- **Identifiers.** Case IDs and subject IDs are VitalDB case-table fields. They are kept in `folds.csv` and the per-patient tables because the folds and the subject bootstrap cannot be reproduced without them.
- **Repeating the run.** `scripts/run_confirmatory.py` runs on real data only at tag `feature-freeze`: it refuses unless HEAD carries that tag and no tracked file is modified. The result files hold no timestamps and every seed is fixed, so a rerun with the same derived tables and the locked environment gives the SHA-256 values of `results/sdev/MANIFEST.json` on the same platform, the two withheld files included (`results/README.md` gives the steps; on another CPU or numerical library the last digits, and so the hashes, can differ). `tests/test_models_end_to_end.py` checks on synthetic data that two runs give identical files. On the real data, a rerun at tag `feature-freeze` into a separate folder (2026-11-23) gave the same SHA-256 for all 11 files listed in `results/sdev/MANIFEST.json`, the two withheld files included.
- **Data snapshot** (`SPEC.md` §10). The S-dev tracks were downloaded on 2026-11-19 (UTC). `results/sdev/data_checksums.csv` lists the SHA-256 of every stored array of the 252 S-dev cases (columns `caseid`, `array`, `dtype`, `n`, `sha256`; checksums only, no data), so that a later change in the public data can be detected. The date comes from the local download log, which the download script writes and which is not committed; `results/README.md` repeats it. The list for the cesarean cases follows with their scoring.

## Limits

- The answer comes from the non-cesarean cases. The cesarean cases are an external case series (21 cases, 20 analysable at the data check before registration) and cannot confirm or refute it.
- The analysis is retrospective, with 21 cesarean cases (20 analysable at G0, the data check made before registration), 19 flagged emergency.
- The spinal injection time is unknown, and the baseline is a record-start value, not pre-spinal.
- Bolus times are absent and the models have no drug input, so a heart-rate gain may reflect unrecorded drugs.
- Drug totals have no times. Cases with rocuronium or vecuronium recorded are excluded; general anesthesia without them (for example with succinylcholine, which the case table does not record) and sedation are not (`SPEC.md` §2.2).
- The primary window extends past incision in 15 of the 21 cesarean cases.
- The development cohort is mostly older men (of the 252 S-dev cases, 179 male and 73 female; median age 60) with slower cuff cycling (median interval 168 s over all S-dev cases).
- Pump rates are recorded in only 5 cesarean cases.
- The bound in Sentence 1 covers this feature set under a linear model fitted as registered. The added coefficients were heavily shrunk, and the interval treats the fold models as fixed (`SPEC.md` §6.2, §7; "Reading the result", items 3 to 5).
- The features are hand-crafted timing and shape features. No pleth amplitude is used, because the stored pleth looks display-scaled, and the probe site is not documented (`SPEC.md` §4.3).
- Each patient has few forecasts (median 6), so per-patient values, R² and FIT above all, are noisy.
- This is a research prototype, not medical advice.

## Cohort flow, S-dev

252 non-cesarean spinal cases → 214 analysable (`SPEC.md` §2.2; the reasons are in `data/derived/sdev_cohort_flow.md` when the code is run) → 11 excluded because the case table records rocuronium (30–60 mg; none records vecuronium; cases 1023, 1877, 2191, 3017, 3071, 3217, 3495, 3772, 4833, 5276 and 5930) → **203 analysed cases**. The analysed cases have 1,421 primary next-reading forecasts, and 200 of the 203 have at least 3, so they can enter the primary outcome. 30 of the 203 are fast-cycling, and so are 5 of the 11 excluded cases. The case table's airway field agrees with the rule: it records oral intubation in 9 of the 11, and of the analysed cases only 1953 has an airway entry (nasal, with no tube size and no blocker or induction drug), which is kept. The exclusion was decided from these case-table fields before any model was fitted. Features are built for all 214 analysable cases, because the sensitivity analysis that includes the 11 needs them, and the counts in the next section refer to these 214 cases unless stated otherwise.

## Implementation choices made before the feature freeze

These choices were made while the feature code was built and checked on S-dev (`SPEC.md` §9, step 2), on signal-level grounds only. No model had been fitted, no forecast error had been computed, no feature had been compared with blood pressure, and no S-ob signal had been opened. They settle points that the registration leaves open; none changes a registered definition. Every number below is recomputed from the S-dev tables that the code builds (`scripts/build_instances.py`, `build_windows.py`, `build_features_wf.py`, `pleth_coverage.py`, `select_rpeak_detector.py`); the tables themselves are not distributed.

### Cuff readings (`SPEC.md` §2.4)

- **Incomplete readings.** A merged reading is dropped as "incomplete" when one of its SBP, DBP or MAP values is missing at the reading time: the track has no sample after the reading, or its next sample is already a later change. Such a reading cannot satisfy SBP > MAP > DBP, so the registered order rule would drop it anyway; it is counted on its own, before the MAP-range and order rules. S-dev has 17 incomplete readings over the whole records, 5 of them in the primary windows of analysable cases. The MAP-range rule drops 14 and 10.
- **The issue time is identified retrospectively.** The reading time is the last change of its merged cluster, and whether a later change joins the cluster is known only up to 15 s afterwards. In practice this rarely matters: in 4,975 of the 4,979 S-dev readings, all changes of SBP, DBP and MBP fall within 0.1 s; the other 4 spread over up to 12.0 s.
- **Gap flags** use the case's primary-window median interval, the same quantity that defines the cycling class (`SPEC.md` §6.7). This flags 603 readings over all S-dev records and 171 in the primary windows of analysable cases.
- **Repeated timestamps.** Of several samples of one track with the same timestamp, the last in file order is kept (cuff and heart rate). No S-dev heart-rate track has two different values at one timestamp.

### Heart rate (`SPEC.md` §4.2)

- **A monitor value of 0 is missing**, like the file's missing-value code. In the first 30 min of the 214 analysable cases this affects 1,487 of 181,367 samples (0.8%), in 17 cases.
- **The change can be missing on its own.** If only the 60 s before the previous reading hold fewer than 10 valid samples, the change is missing and the level and slope are kept. Of the 1,552 primary issue points, the whole group is missing at 51 and the change alone at 38.
- **Inflation-artefact variant.** The 60-s windows stay anchored at the reading times; samples from 30 s before the issue time on are dropped from both windows.

### Waveform windows and per-forecast features (`SPEC.md` §4.3, §5)

- **Window grid.** The 10-s windows, and the 125 Hz decimation of the quality index, start at the first stored pleth sample, not at time 0 (median 1.46 s after the record start in the analysable cases). Waveforms are stored for 0–3660 s of each record, which covers every window used.
- **Which windows count.** A window counts for a forecast only if it lies wholly between the previous reading (at most 120 s back) and the issue time, or the variant's cutoff. A window that straddles the issue time is not used.
- **Minimum per feature.** The registered minimum of 2 usable windows is applied to each feature: a feature is missing unless at least 2 windows in the interval have a value for it.
- **Reference.** For each feature, the median over the first 6 usable windows of the record, with the same 2-value minimum. The level form is missing until the 6th of these windows has ended by the cutoff.
- **Change form.** The same aggregation at the previous reading, over the windows between max(reading before it, previous reading − 120 s) and the previous reading.

### Beat delineation (`SPEC.md` §4.3, §5)

- **Systolic peaks** by the method of Elgendi et al. 2013 (PLoS One 8:e76585), applied to the band-passed window. The band-pass is a Butterworth filter of order 2 per band edge (0.5 and 8 Hz; the standard design, a 4th-order band-pass), run forward and backward, so it has zero phase and twice the attenuation of one pass.
- **A beat** is a systolic peak with a valley on both sides, a positive maximum upslope, and a tangent foot that does not fall before its foot valley. The ≥ 4-beats condition of `SPEC.md` §5 counts these beats. The tangent-foot bound changes no S-dev window.
- **Beat-amplitude variability** is the sample standard deviation (n − 1) of the beat amplitudes over their mean.

### Dicrotic-notch features dropped (`SPEC.md` §4.3)

The dicrotic-notch time and notch height ratio of version 1.4 are not model inputs. They are still computed per window for the data-quality report. The decision was made on pleth shape and pulse rate only. In the rebuilt window table:
- The window-median notch time is bimodal, with an early cluster at 0.08–0.12 s after the systolic peak and a later one near 0.19–0.24 s. Of the 75,557 usable windows of the 250 S-dev cases with a pleth, 37.1% lie below 0.14 s, the trough between the two modes, and 33.2% lie in 0.08–0.12 s.
- The early point does not shorten with pulse rate, as a dicrotic notch would: in the analysable cases its median is 0.093 s at 40–60 beats/min and 0.106 s at 100–180 beats/min, while the later point shortens from 0.225 s to 0.174 s. The early share rises with pulse rate, from 19.2% to 64.7%.
- 192 of the 214 analysable cases have windows of both kinds, and 94 have between 10% and 90% early windows.
- On synthetic beats that have no notch at all, the detector still returns a candidate in at least 90% of beats after the band-pass (`tests/test_waveform.py`).

### Pulse arrival time (`SPEC.md` §4.3)

- **Pairing.** A beat's R peak is the unique R peak 0.40–0.90 s before its pleth foot; with none or more than one, the beat has no pulse arrival time. A window needs ≥ 4 paired beats.
- **Why this range.** The stored pleth lags the ECG by a device delay of about 0.65 s: the median window value is 0.646 s, and the per-case medians range from 0.577 to 0.722 s (214 analysable cases). This exceeds the RR interval above about 90 beats/min, so the latest R peak before a foot can belong to the next beat. The range is centred on the observed delay and is 0.5 s wide, so it holds at most one R peak while RR > 0.5 s. When RR < 0.5 s (above 120 beats/min), a beat can lose its pulse arrival time by construction; 324 usable windows of analysable cases are above that rate, and 319 of them still have a window value.
- **When the range was fixed.** After a signal-level survey of the foot-to-R lags (every 12th S-dev case); `data/derived/rpeak_selection.md` reproduces the lag histogram on the detector-selection cases. No blood-pressure value was used.
- 98.2% of the usable windows of analysable cases have a pulse arrival time. It enters the models only as a within-case difference.

### R-peak detector (`SPEC.md` §4.3, §9)

The candidates (four neurokit2 methods), the cases, the windows and the selection rule were written in `scripts/select_rpeak_detector.py` before the comparison was first run. That order is not independently time-stamped. The criterion is the share of 10-s windows whose detector rate lies within 5 beats/min of the median monitor heart rate, on every 5th S-dev case with pleth and ECG present (50 cases, 15,485 windows):

| method | share within 5 beats/min | median absolute difference (beats/min) |
|---|---|---|
| **neurokit (chosen)** | 78.8% | 1.53 |
| elgendi2010 | 76.9% | 1.87 |
| pantompkins1985 | 65.0% | 3.10 |
| hamilton2002 | 45.1% | 5.57 |

Only the chosen detector used a powerline filter (60 Hz, the mains frequency where VitalDB was recorded); the other three methods have no such step in neurokit2. No detector raised an error on any window, in the selection or in the rebuilt window table; such windows would be counted in the logs and have no R peaks. Software: neurokit2 (Makowski D, et al. Behav Res Methods 2021;53:1689–1696, doi:10.3758/s13428-020-01516-y).

### Signal integrity (`SPEC.md` §2.3)

A waveform whose header rate is not 500 Hz is treated as absent, whether or not the block check could be made. No S-dev waveform is affected; S-dev cases 2030 and 2328 are not analysable for the reasons stated in `SPEC.md` §2.3.

### Placebo donors (`SPEC.md` §6.6, §4.5)

The analysable cases of a cohort, in ascending case-ID order, are permuted with `numpy.random.default_rng(20261118).permutation`. A recipient's donor is the next case after it in the permutation, wrapping around, that belongs to another subject and whose heart-rate, pleth and ECG tracks cover the recipient's primary-window issue times (first to last issue point, with or without a target). The same donor serves the secondary window. All 214 analysable S-dev cases have a donor. Donors are drawn from all analysable cases, including the 11 excluded for a neuromuscular blocker; recipients are the analysed cases, and all analysable cases in the sensitivity analysis that includes the 11. The donor assignment was built before the rule was set and was not redrawn; a donor only has to be another patient. 11 of the 203 analysed recipients have a donor with a recorded neuromuscular blocker.

### Vasopressor strata, S-ob (`SPEC.md` §2.6)

Fixed in the code before the S-ob scoring: "pump track present" means that `Orchestra/PHEN_RATE` has a record in the case file; "bolus total > 0" compares the phenylephrine (µg) and ephedrine (mg) chart totals with 0 each on its own; a missing total counts as not recorded. Both versions are built: phenylephrine or ephedrine, and phenylephrine alone. The two units are never added.

### Signal coverage on S-dev after the rebuild (descriptive)

Of the 1,552 primary next-reading forecasts in the 214 analysable cases, 97.7% have ≥ 2 usable pleth windows (P1 98.7%, P2 73.8%, inflation-artefact variant 93.7%, stricter quality rule 92.4%). Quality strata: 35 forecasts with fewer than 2 usable windows, 124 with 2 or more but under 80% usable, and 1,393 with 80% or more. With real signals, the level and change forms are missing in 2.4% and 3.5% of forecasts for rise time, width and beat-amplitude variability, and in 2.9% and 4.1% for pulse arrival time.

### Reproducibility (`SPEC.md` §10)

Python 3.13.15 (`.python-version`); every imported package is pinned exactly in `pyproject.toml`, and `uv.lock` pins the full environment. Every derived table and log records the git commit and state of the code that wrote it; a builder refuses to run from uncommitted code unless told to, and then records that. Every generated Markdown report carries the VitalDB credit. S-ob signals can be read or downloaded only with `BTC_FROZEN_SCORING=1`, which is set only from the frozen scoring on (the scoring run and the S-ob exports that follow it).

## Deviations

None so far. This list is checked again when the cesarean scoring and the secondary analyses are added.

The following are not deviations from `PREREG.md`. They are listed so that nothing has to be inferred:

- **Interim README.** `SPEC.md` §8.1 describes the README of the finished analysis. Until the cesarean scoring, `README.md` is written by hand in that block order. The parts that need the cesarean result are marked there as pending: the cesarean line, Sentence 2, the summary figure, the replay, the cesarean column of the table, the per-patient figure and the hypotension line. It also has two short paragraphs outside the block list: "Why it matters" (from `SPEC.md` §0) and "How to read these numbers" (from "Reading the result" above).
- **Order of the secondary analyses.** They run after the cesarean scoring (`SPEC.md` §9 step 8). Block 2 of the README therefore refers to a gradient-boosting check that is not yet in this file, and says so.
- **Post hoc material.** "Reading the result" and `results/frozen/penalty_path.csv` were not pre-specified and are labelled as such (`PREREG.md` §10).
- **Withheld files.** The two per-forecast result files are not published ("Data and result files"); per-patient tables are published in their place.

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
