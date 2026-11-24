# Exploratory lead–lag and detection-limit analyses: plan

**Status.** Both analyses are exploratory, computed after the registered results were known. This file was written on 2026-11-24, after the S-dev confirmatory result (no gain), the frozen S-ob scoring, the secondary analyses and the encoder check, and before either analysis was run. (1) is an item that `SPEC.md` §6.8 lists ("Lead–lag between feature changes and the cuff MAP change"); its details are fixed here. (2) is new, so it is post hoc exploratory (`SPEC.md` §9). Both use S-dev only, read no S-ob file, and change no registered outcome, rule or sentence. The choices below rest on the design and on one signal-level check of the feature tables (described in 1.1); no blood-pressure value or forecast error was looked at to make them.

## 1. Lead–lag between feature changes and cuff MAP changes

**Question.** Within a patient, do changes in the heart-rate and pleth features go with the cuff MAP change over the same interval, and with the next one?

1.1 **Feature change.** For reading r, Δfeature = the registered per-forecast value at r minus that at r−1, read from the existing tables (`btc/SCHEMA.md` §4): heart-rate level; pulse arrival time, rise time, pulse width (the shape group) and beat-amplitude variability. This difference is the registered change form, `hr_change` and `wf_<f>_chg` (`SPEC.md` §4.2, §4.3), so that column is used. A signal-level check made for this plan confirmed it: wherever the levels at r and r−1 both exist, the change column equals their difference (largest gap 1.1e-16), and it also exists where the level at r−1 does not (reference not yet set, or reading r−1 not an issue point). The heart-rate slope and the change forms themselves are not differenced again: that would be a second difference, not a feature change.

1.2 **MAP changes**, on consecutive valid readings r−1, r, r+1 (dropped readings ignored, as everywhere): (a) concurrent, MAP_r − MAP_{r−1}; (b) next, MAP_{r+1} − MAP_r, the change that the primary forecast targets; (c) added here: the next-reading residual of the cross-validated cuff-only model M2 (observed minus M2 forecast). Reason for (c): a feature that only tracks the current MAP can correlate with (b) through the MAP series' own trends and its measurement noise, which the cuff history already carries; (c) keeps only the part of the next change that M2 missed. M2 is re-derived with the registered folds and must reproduce the confirmatory M2 error.

1.3 **Rows.** The primary-window next-reading forecasts of the analysed S-dev cases (the rows of the primary outcome), so that (a), (b) and (c) use the same pairs. (b) therefore keeps the 400-s cap of the forecast targets; (a) has no cap, because the feature change compares the same two reading times.

1.4 **Missing values.** A pair is used only if Δfeature exists; there is no imputation, because a constant fill would only dilute a correlation. MAP changes always exist. **Minimum:** a case enters a feature's correlation with at least 3 usable pairs, the project's minimum per patient for RMSE (`SPEC.md` §6.1); cases below it are counted.

1.5 **Statistic.** The within-case correlation: each case's Δfeature and MAP change are centred at that case's own means over its usable pairs, and one Pearson correlation is computed from the pooled centred values. Centring removes differences between patients, such as one whose pressure falls throughout.

1.6 **Interval.** 95% percentile interval from the project's subject bootstrap (`SPEC.md` §6.2: 2,000 resamples, seed 20261118, `btc.models.bootstrap.SubjectBootstrap`), built on the patients of the primary outcome, so the resamples are those of the primary outcome. A case below the minimum contributes zero to the sums. One set of resamples serves every feature, MAP change and variant, so the intervals are paired.

1.7 **Placebo.** The same statistic on the P1 donor features (`SPEC.md` §6.6: another patient's signals at this patient's reading times), each variant with its own usable pairs, and real minus placebo with its interval from the same resamples. The placebo shows what timing and shared trends alone give.

1.8 **Expectation, not a claim.** Pulse arrival time shortens when blood pressure rises, so (a) is expected to be negative for pulse arrival time. No direction is stated for the other features: heart rate can move either way with vasopressors (`SPEC.md` §7).

1.9 **Limits.** The current interval ends at t_r and includes the inflation for reading r; if the probe is on the cuff arm, this may affect (a) (`SPEC.md` §3, inflation length unverified). A correlation is not an increment over the cuff; the forecast contrasts test that. Thirty intervals (5 features × 3 MAP changes × 2 variants) and the differences are unadjusted and descriptive.

## 2. Detection limit of the registered pipeline

**Question.** How large a waveform signal would the registered fitting procedure have found? This tests the procedure on the real S-dev design, not the physiology.

2.1 **Planted column.** One column s is added to the waveform block of the real S-dev design; nothing else changes. s = y + κ·σ_c·z, where y is the forecast's target change (next MAP minus last MAP), σ_c the root mean square of y over that patient's primary forecasts, and z standard normal noise from `numpy.random.default_rng(20261118)`, one draw per forecast in (case ID, issue reading) order, the same draw at every strength. Zero strength is s = σ_c·z. σ_c makes the signal-to-noise ratio the same in every patient, so that the strength holds in the per-patient metric.

2.2 **Strengths, fixed now.** Nominal reductions q = 0, 1, 2, 3, 5, 10 and 20% of RMSE, with κ² = (1−q)² / (1 − (1−q)²). If the cuff explained none of y, the forecast (1−β)·(M2 forecast change) + β·s with β = 1 − (1−q)² would multiply each patient's expected mean squared error by (1−q)², that is, lower its RMSE by about q. **Oracle:** that forecast on the rows with waveform features and M2 elsewhere, scored like every model. Its realised relative reduction against M2 is the known amount reported beside each estimate; it differs from q because M2 explains part of y and the noise is one draw.

2.3 **Why it cannot leak.** A forecast's planted value depends only on its own target, its own patient's σ_c, and noise drawn independently of the data. Folds are grouped by subject and every analysed case is a different subject, so all of a patient's forecasts lie on one side of every outer and inner split: no training row's value depends on any validation or test row. Nothing fitted enters s; an out-of-fold M2 residual would not satisfy this, because a training row's residual comes from an M2 fit that includes the test fold. σ_c multiplies zero-mean noise, so it moves no forecast in expectation. The M2 and M3 stages must be bit-identical to those of the plain ladder. Planted values are never written to a file.

2.4 **Fit and score, unchanged.** The chain M2 → M3 → M4+planted: the registered M2 structure, blocks, penalty grid with "dropped", inner and outer folds, seed, tie rule, fallback to M3 without waveform features, and the paired subject bootstrap. The plain ladder is rerun alongside, and its M4 vs M2 contrast must equal the confirmatory summary. **Reported per strength:** the oracle's relative reduction with its 95% CI; M4+planted vs M2 as Δ and relative reduction with 95% CIs, whether the 95% CI of Δ lies below zero, and whether it meets the rule (`SPEC.md` §6.3: at least 5% and the upper limit of Δ below zero); the waveform-block penalty in each outer fold. Placebo validity is not evaluated: a donor version of a synthetic column has no meaning.

2.5 **Replicates.** The noise is one draw, so the same run is repeated with noise seeds 20261119–20261128, and for each strength the number of the ten draws that meet the rule and the range of the estimate are reported. The draw with seed 20261118 is the main result.

## 3. Record and standing
- Code: `btc/models/exploratory.py`, `scripts/run_exploratory.py` (commands `lead-lag`, `detection-limit`). The script refuses to run unless the work tree is clean, this file is committed at HEAD, `BTC_FROZEN_SCORING` is unset, the S-dev tables equal the confirmatory inputs, the folds equal the confirmatory `folds.csv`, and the output folder does not exist yet. Every attempt is logged (git-ignored); each result file records the commit and this file's SHA-256.
- Outputs: `results/sdev/exploratory_lead_lag/` and `results/sdev/exploratory_detection_limit/`, each with `MANIFEST.json`. Aggregate tables only; no per-forecast or per-case value is written.
- The first complete run of each command after this file is committed is the result. A rerun after a crash with unchanged code is allowed and logged; any other rerun is a departure, and both results are shown.
- Stop rule: if the results are not committed by 2026-11-24 15:00 KST, applied by the clock, what exists is reported as incomplete.
