# Pre-registration — between-the-cuffs

**Status: registered on 2026-11-21 (this commit, version 1.5). It supersedes version 1.4 (commit `fca645e`, 2026-11-19), which was replaced before any model was fitted.**

**What is registered.** This file and `SPEC.md` v1.5 are committed together, before any model is fitted to any case, and together they form the registration. Version 1.4 is the first commit of the public history (`fca645e`). This version replaces it in place, so the git history shows every change, and §11 lists them. Both versions are made public before any model is fitted. The commit hashes of both versions and the SHA-256 hashes of both files in each version are quoted in `REPORT.md`. This file restates the confirmatory parts and wins any conflict with `SPEC.md`.

**G0** = a data-availability check run before registration. It looked at track presence, cuff timing and reading counts, plus a small pleth-quality sample. No model was fitted.

## 1. Hypothesis
After spinal anesthesia, adding heart rate and pulse-oximeter (pleth) waveform features to a cuff-only third-order autoregressive model lowers the error of the next-cuff-reading MAP forecast.

**Hypothesis risk, stated in advance.** A null or small result is plausible:
- **Large changes are hard.** In the closest published method (Lee et al., Diagnostics 2026;16:3713), a cuff-calibrated PPG delta model reproduced only 37–48% of SBP changes larger than 40 mmHg, and its accuracy decayed with time since the last cuff.
- **Little error is left.** Fast cuff cycling in the cesarean cases (60–90 s) leaves little error for added signals to remove.
- **Small training base.** The h-bin that holds most cesarean targets is trained mainly on the fast-cycling development cases (34 at G0; 30 analysed under the final rules).

**What each outcome would mean** [judgment]:
- **Positive.** Heart rate and the pleth may partly stand in for the missing drug input, so a positive result cannot separate physiology from drug effect. It is probably an upper estimate for a system that already has timed doses.
- **Null.** A null is harder to attribute to the drug proxy, but opposite heart-rate responses to the two vasopressors could also cancel in a linear model. It bounds only this feature set under a linear model. The gradient-boosting check (§7) asks the same question of the same features without the linear form.
- **Either way,** only data with time-stamped vasopressor doses can settle the question.

## 2. Data and cohorts
- **Source.** VitalDB public API (CC BY 4.0). Data are downloaded at run time and not redistributed. Tracks are read from the versioned case files (1.0.1). A waveform with a broken sampling clock is treated as absent: its recorded blocks, checked in the original case file, imply a rate more than 1% away from 500 Hz, or more than 1% of its samples are overwritten (`SPEC.md` §2.3 defines both measures). If the original file cannot be read, the waveform is kept and flagged.
- **S-dev.** Spinal non-cesarean cases, used for fitting and cross-validation. Six cervical cerclage cases are kept and flagged.
- **S-ob.** Spinal cesarean cases, scored once after the freeze.
- **Subject overlap.** If a subject has cases in both cohorts, the S-dev case is excluded from fitting. At registration, no subject is in both cohorts.
- **Analysable.** Pleth and ECG present in the case file and not treated as absent, and ≥ 6 valid cuff readings in the primary window.
  - Counted at G0, with the provisional cuff rule and with track presence taken from the VitalDB track list: 218 S-dev and 20 S-ob cases. These counts predate the source rules above, under which S-dev cases 2030 and 2328 are not analysable.
  - Final counts are recomputed under the rule below. For S-ob this happens only at the frozen scoring.
- **Neuromuscular blocker.** An analysable case whose case table records rocuronium or vecuronium (`intraop_rocu` or `intraop_vecu` > 0; missing counts as none) very likely also had general anesthesia, at an unrecorded time. It is excluded. Only the remaining **analysed** cases are fitted or scored, except in one sensitivity analysis that includes them (§7). Placebo donors are still drawn from all analysable cases. The case table records no other blocker (for example succinylcholine), so the rule can miss general anesthesia (`SPEC.md` §2.2).
  - Set before any fit, from S-dev case-table drug totals only (the airway fields agree): 11 of the 214 analysable S-dev cases have rocuronium recorded, leaving 203 analysed. The same rule applies to S-ob, whose other drug totals have not been looked at.
- **Cuff reading.**
  - A reading is any change in SBP, DBP or MBP; a change within 15 s of the previous change is merged with it. The first value of a track counts as a reading at its first sample.
  - Its time t_r is the last change in the cluster, and its values are those held at the first sample after t_r.
  - It is dropped, and counted, if MAP is outside 30–160 mmHg or SBP > MAP > DBP fails.
- **Time zero and windows.**
  - Time zero is the recording start (`casestart` = 0).
  - Primary window: 0–30 min.
  - Secondary window: 0 to incision (`opstart`), capped at 60 min.
- **Record-start baseline.** The mean of the first three valid readings. It is never called pre-spinal.

## 3. Forecasts
- **Issue points.** Every valid reading in the window from the third valid reading of the record on. The issue time is t_r.
  - Cuff features use readings up to and including that reading.
  - HR and waveform features use only samples earlier than t_r.
- **Primary target.** The next reading, if it falls within 400 s and inside the window.
- **Horizon.** Each model outputs a forecast curve ŷ(h) and is scored at the realised time of the target reading.
  - h is the only input not known at the issue time. Every model receives the same h; in the pooled structure, the added features of M3 and M4, and of their P1 versions, also interact with it (§4).
  - Sensitivity analysis: h is replaced by the median of the patient's previous intervals.
  - Results are reported by h-bin: 15–90, 90–150, 150–210 and 210–400 s.
- **Leakage test.** A unit test replaces with random values every HR, pleth and ECG sample at or after the issue time, every later cuff reading, and the withheld reading of the yardstick (§7). All features except h must be unchanged.
- **Per-window processing.** Filtering, beat detection and normalisation run per 10-s window, never on the whole record.

## 4. Models
| Model | Inputs |
|---|---|
| M00 | Causal running mean of the readings so far |
| M0 | Persistence |
| M1 | Linear trend through the last 3 readings |
| **M2** | **Cuff only, third order:** last MAP minus the baseline; slope and curvature from the last 3 readings at their real timestamps; last pulse pressure; time since record start; h |
| **M3** | M2 + heart rate: 60-s median level, change since the previous reading, 60-s slope |
| **M4** | M3 + pleth and ECG waveform features (pulse arrival time, rise time, half-height width, beat-amplitude coefficient of variation; the two notch features of version 1.4 are not used, `SPEC.md` §4.3). Each feature enters as its level relative to the median of the record's first 6 usable windows, and as its change since the previous reading. **No absolute amplitude.** Pulse arrival time enters in the same two forms and never as an absolute value. |

**Fitting**
- **Method.** Ridge regression. The scaler and the imputation medians are fitted on training folds only.
- **Outer folds.** 5 folds grouped by `subjectid` and stratified by cycling class (case median interval < 90 s), seed 20261118.
- **Penalties.** M2's penalty is chosen by inner grouped CV from 13 values, 10⁻³ … 10³. There are three blocks: cuff, heart rate and waveform. In M3, the cuff block keeps M2's penalty, and the heart-rate block gets its own from the same grid plus a "dropped" option. In M4, the cuff and heart-rate blocks keep their penalties from M2 and M3, and the waveform block gets its own in the same way. "Dropped" means that the model returns the forecast of the model below it.
- **Training targets.** The primary models are fitted on next-reading targets in the primary window only, with every forecast weighted equally.
- **Horizon and added features.** In the pooled structure, each heart-rate and waveform feature enters as itself and as feature × h, with the feature centred at its training-fold median before the product.
- **Details.** `SPEC.md` §4.5 fixes the remaining fitting details: inner folds and criterion, tie rules, fold construction, bin edges and the refitting of variants. They are part of this registration.
- **Before S-ob scoring,** every model is refitted on all of S-dev.

**M2 structure, fixed on S-dev before M3/M4 are fitted.** All four combinations of two choices are compared, and the one with the lowest patient-level mean RMSE wins:
- **M2 form:** level/slope/curvature, or a raw-lag AR(3) with the gaps as inputs (pulse pressure, time since record start and h stay in both);
- **Horizon structure:** per-bin models (any bin with fewer than 300 S-dev training targets uses the pooled model), or one pooled model with the interactions slope × h, curvature × h² and level × h. With the raw-lag form, the interactions are each lagged value × h and × h².

The numbers of training targets and patients per h-bin are reported.

**Missing data and fallbacks**
- A missing HR group (fewer than 10 valid samples in 60 s) or a missing waveform feature group is set to the training-fold median and gets a missing-indicator.
- With fewer than 2 usable pleth windows between the previous reading (at most 120 s back) and t_r, M4 returns the M3 forecast and the forecast is flagged. M4 is fitted on the training forecasts that have waveform features.

**Feature-code freeze.** The feature-extraction code and parameters are frozen and tagged before the first M3/M4 fit. They are chosen only on signal-level criteria (for example, R-peak agreement with the monitor HR), never on BP or forecast error. **The first complete S-dev cross-validation run at that tag is the confirmatory result.** It includes M2–M4, M3-P1, M4-P1 and the yardstick models, and it fixes Sentences 1 and 3 before S-ob is scored. A rerun with changed code is a deviation, and both results are shown. Before that run, the fitting code is tested only on M2 and on synthetic added features, and no model with real heart-rate or waveform inputs is fitted. The feature tag is pushed to the public repository before that run. The frozen-model tag, with the S-dev confirmatory outputs and Sentences 1 and 3, is pushed before S-ob is scored.

## 5. Pleth quality index
**Primary index: the published BP-blind rule of Lee et al. 2026, with its criteria and thresholds unchanged and one added condition.**

**Preparation**
- The 500 Hz record is reduced to 125 Hz by taking every 4th sample, with no interpolation.
- It is cut into 10-s windows of 1,250 samples.
- A window with missing samples, or a flat window (SD 0), is dropped.
- Each remaining window is standardised.

**A window is unusable if any of these holds** (all computed on the unfiltered 125 Hz window):
- the dominant frequency in 0.5–8 Hz lies outside 0.67–3 Hz;
- the spectral peak holds less than 15% of in-band power;
- max |z| > 6;
- constant runs of at least 0.5 s cover at least 10% of the window.

**Operational details**
- **Spectrum:** Hann-windowed periodogram of the full window, constant detrend, 0.1 Hz bins.
- **Dominant frequency:** the largest bin in 0.5–8 Hz.
- **Peak power:** that bin ± 1 bin, divided by the 0.5–8 Hz sum.
- **Constant runs:** runs of at least 63 equal consecutive samples; the window is unusable if they total at least 125 samples.
- **Added condition:** at least 4 delineated beats.

**Disclosure.** A beat-template correlation threshold of 0.85 was considered during design. It was chosen after seeing the usable-window shares of cesarean cases. It is used only as a labelled, stricter sensitivity analysis.

## 6. Primary outcome and decision rule
**Primary outcome: Δ = patient-level mean RMSE(M4) − patient-level mean RMSE(M2).**
- Issue time: at a reading. Target: the next reading. Window: the primary window.
- **All forecasts are scored**, fallbacks included.
- A patient enters with at least 3 scored forecasts.
- Reported in mmHg for S-dev (outer-CV predictions) and for S-ob (frozen models).

**Relative reduction = 1 − mean patient RMSE(model) / mean patient RMSE(comparator).**
- The comparator is M2, or M3 for M4 vs M3.
- 95% CIs come from a paired patient-level percentile bootstrap: 2,000 resamples, seed 20261118, S-dev resampled by `subjectid`.

**A contrast meets the rule** if, in S-dev:
- the relative reduction is ≥ 5%; **and**
- the upper 95% CI limit of Δ is below zero.

The 5% is a detection margin, not a clinical threshold.

**Multiplicity and sample size.** There is one primary contrast: M4 vs M2 in S-dev. M3 vs M2 and M4 vs M3 are secondary. They use the same rule with no adjustment, and a statement that rests on one of them alone (the sentence appended in rows 1 and 6) is a secondary finding. Every other interval is an unadjusted 95% interval and is descriptive. No p-values are reported. Every analysed case (§2) is used, and no power calculation was made; a null is reported with the largest gain the data are compatible with (row 6).

**Placebo validity.** A positive claim about M4 requires both of the following, and so does a claim about heart rate alone (M3):
- its placebo version P1 does not meet the rule against M2;
- the 95% CI of RMSE(model) − RMSE(model-P1) lies below zero.

For P1, each recipient gets one donor: the next analysable case in a seeded permutation within the same cohort that belongs to another subject and has HR, pleth and ECG tracks that start by the recipient's first issue time and last at least until its last issue time. The donor's signals are processed at the recipient's reading and issue times, and the model is refitted.

**S-ob** is an external case series. Its result is reported but does not decide the headline.

**Coverage rule.** If fewer than 50% of S-ob primary forecasts have waveform features, the S-ob result is reported as "insufficient signal coverage", with its numbers still shown.

**Headline (chosen mechanically).**
- **Order of checks:** M4 vs M2 first, then M4 placebo validity, then M3 vs M2 and M4 vs M3.
- **Placebo checks:** placebo validity is checked only for a contrast that meets the rule, so a plain null is never reported as a placebo failure.
- **Row selection:** the first row that applies, in table order, is used, so rows 2–5 presuppose that M4 passed placebo validity. Apart from row 1, row selection depends only on whether each contrast meets the rule. M3's placebo result changes the wording, never the row.
- **Symbols:** X, Z, Y and P are the relative reductions of M4 vs M2, M3 vs M2, M4 vs M3 and M4-P1 vs M2.
- **If M3 fails placebo validity:**
  - in row 2, "heart rate alone gave Z% (95% CI …)" becomes "heart rate's share did not pass its placebo check (Z%)";
  - in row 3, "and the gain came mainly from heart rate" becomes "but heart rate's share did not pass its placebo check (Z%)".
- **Waveform share:** it has no separate placebo; the feature-group ablation is its check.

Sentence 1 always begins "In N non-cesarean spinal cases (cross-validated), …"; N, and n in Sentence 2, count the cases that enter the primary outcome:

| Result in S-dev | Sentence 1 continues |
|---|---|
| 1. M4 vs M2 meets the rule, but M4 placebo validity fails | "… the added signals met the rule (X%), but the placebo control failed (placebo inputs: P% against the cuff-only model; real minus placebo inputs: Δ … mmHg, 95% CI …), so no claim is made about the added signals." If M3 vs M2 meets the rule and passes placebo validity, append "Heart rate alone met the rule (Z%, 95% CI …)." |
| 2. M4 vs M2, M3 vs M2 and M4 vs M3 all meet the rule | "… heart rate and pleth waveform features together lowered next-reading MAP forecast error by X% (Δ … mmHg, 95% CI …) beyond a cuff-only third-order model; heart rate alone gave Z% (95% CI …), and the waveform added Y% (95% CI …) beyond heart rate." |
| 3. M4 vs M2 and M3 vs M2 meet the rule; M4 vs M3 does not | "… the added signals lowered next-reading MAP forecast error by X% (Δ … mmHg, 95% CI …), and the gain came mainly from heart rate; pleth waveform features did not meet the rule beyond it (Y%, 95% CI …)." |
| 4. M4 vs M2 meets the rule; M3 vs M2 does not; M4 vs M3 does | "… the added signals lowered next-reading MAP forecast error by X% (Δ … mmHg, 95% CI …), and the gain came from the waveform features; heart rate alone did not meet the rule (Z%, 95% CI …)." |
| 5. M4 vs M2 meets the rule; neither increment does | "… heart rate and pleth waveform features together lowered next-reading MAP forecast error by X% (Δ … mmHg, 95% CI …); neither heart rate alone (Z%, 95% CI …) nor the waveform beyond it (Y%, 95% CI …) met the rule." |
| 6. M4 vs M2 does not meet the rule | "… heart rate and pleth waveform features did not meet the pre-specified rule for lowering next-reading MAP forecast error beyond a cuff-only third-order model (X%, Δ … mmHg, 95% CI …; rule: at least 5%, with the 95% CI excluding zero). Gains larger than G% (g mmHg) are not compatible with these data (approximate 95% CI, see §6.2 of SPEC), for this feature set and a linear model." If the CI of Δ lies above zero, "; they raised it" is inserted before the parenthesis. If M3 vs M2 meets the rule and passes placebo validity, append "Heart rate alone met the rule (Z%, 95% CI …)." |

In row 6:
- G is the upper 95% CI limit of the relative reduction.
- g is minus the lower 95% CI limit of Δ, that is, the largest reduction in mmHg compatible with the data.
- If G ≤ 0, the bound sentence is omitted.

Sentence 2 is always: "In the n cesarean cases (external, frozen models): X% (Δ … mmHg, 95% CI …), waveform coverage C%; this is a case series and neither confirms nor refutes the first sentence." If the coverage rule applies, the variant is: "… insufficient signal coverage (C%); X% (95% CI …) is shown for completeness."

Sentence 3 (key secondary, the cesarean-like cuff regime) is always: "In the K fast-cycling development cases (median cuff interval < 90 s): X% (Δ … mmHg, 95% CI …); one extra cuff reading's value recovered by the added signals: F% (95% CI …)."
- X% here is the M4-vs-M2 contrast scored on these cases. F uses those of them with ≥ 3 scored triples; `REPORT.md` gives that number.
- If F is not reported, the clause after the semicolon becomes "one extra cuff reading had no detectable value at this interval".
- If T fails its placebo guard, "(not distinguishable from placebo inputs)" follows F.
- **Numbers.** In Sentences 1–3 and the plain lines, the CI after "Δ … mmHg" is that of Δ, and the CI after a percentage alone is that of the percentage. Percentages have one decimal and mmHg two.

**Plain lines.** The README shows these first, above Sentences 1–3 (§8.1 of SPEC). They are chosen with the same row, use the same numbers, and may say no more than Sentences 1 and 2.
- **Question, always:** "After spinal anesthesia, do heart rate and the pulse-oximeter waveform improve the forecast of the next cuff reading (MAP) beyond the cuff readings alone?"
- **Answer line:** the label of the selected row, then "In N non-cesarean spinal cases (cross-validated), …" and the row's continuation below. A and B are the patient-level mean RMSE of M2 and of M4, in mmHg.
- **Cesarean line, always:** "In the n cesarean cases (external, frozen models): X% (95% CI …); this is a case series and neither confirms nor refutes the answer above." If the coverage rule applies, the variant of Sentence 2 is used instead.

| Row | Label | Answer line continues |
|---|---|---|
| 1 | "No claim." | "… forecast error fell by X%, but the placebo control failed, so no claim is made about the added signals." |
| 2 | "Yes." | "… heart rate and the pleth waveform together lowered next-reading forecast error by X% (from A to B mmHg); heart rate alone gave Z%, and the waveform added Y% beyond heart rate." |
| 3 | "Yes, mainly from heart rate." | "… the added signals lowered next-reading forecast error by X% (from A to B mmHg), and the gain came mainly from heart rate; pleth waveform features did not meet the rule beyond it (Y%)." |
| 4 | "Yes, from the pleth waveform." | "… the added signals lowered next-reading forecast error by X% (from A to B mmHg), and the gain came from the waveform features; heart rate alone did not meet the rule (Z%)." |
| 5 | "Yes, together." | "… heart rate and the pleth waveform together lowered next-reading forecast error by X% (from A to B mmHg); neither heart rate alone (Z%) nor the waveform beyond it (Y%) met the rule." |
| 6 | "No." | "… heart rate and the pleth waveform did not meet the pre-specified rule (X%, from A to B mmHg; rule: at least 5%, with the 95% CI excluding zero). Gains larger than G% (g mmHg) are not compatible with these data (approximate 95% CI), for this feature set and a linear model." |

- **Same variants as Sentence 1.** In row 6, "; they raised it" is inserted under the same condition, and the bound sentence is omitted if G ≤ 0. The sentence appended in rows 1 and 6 is added without its CI: "Heart rate alone met the rule (Z%)." If M3 fails placebo validity, the replacements of Sentence 1 apply to the same words in rows 2 and 3, and the label of row 3 becomes "Yes."
- **Row 6 labels.** If G ≥ 5, the label is "Not shown." If the CI of Δ lies above zero, the label is "No: the added signals raised the error."

## 7. Secondary analyses (pre-specified)
- **Decomposition:** M3 vs M2 and M4 vs M3.
- **Metrics.** Each metric is computed per patient and summarised as mean ± SD.
  - MAE, mean signed error (forecast minus observed), R² and FIT for every model; the pooled RMSE over all forecasts for M2–M4.
  - R² and FIT need at least 5 forecasts and a non-zero SST.
  - FIT = 100 × (1 − ‖y − ŷ‖ / ‖y − ȳ‖), with ȳ the patient's in-window mean. This is the MATLAB System Identification Toolbox definition. It has not been confirmed against the full text of Davoud et al. 2025 (doi:10.1007/s10877-025-01288-w) [unverified].
- **Subsets and variants:**
  - h-bins;
  - 2-reading-ahead targets;
  - the waveform-available subset;
  - the fast-cycling S-dev cases: median interval between consecutive valid readings in the primary window < 90 s under the final cuff rule (34 at G0; 30 analysed under the final rules). Only the scoring is restricted: predictions are the outer-CV predictions of the models fitted on all of S-dev, and nothing is refitted on the subset. K counts such cases with ≥ 3 scored forecasts;
  - S-dev by sex (scoring only);
  - the secondary window.
- **Sensitivity analyses:**
  - the alternative horizon structure;
  - h replaced by the median of previous intervals;
  - the template-correlation quality variant;
  - the fold seed: M2–M4 are refitted under ten further seeds (20261119–20261128), and the range of the M4-vs-M2 result is reported. The run with seed 20261118 stays the confirmatory result;
  - the cases with a neuromuscular blocker: in S-dev, M2–M4 are refitted on all analysable cases with their own folds; in S-ob, the frozen scoring run also scores all analysable cases. This result cannot change the headline row or Sentences 1–3.
- **Withheld-cuff yardstick (key secondary).**
  - **Triples.** Consecutive valid readings r, r+1, r+2, with r at least the third reading and t_{r+2} − t_r ≤ 400 s.
  - **Three forecasts of reading r+2**, fitted on S-dev on the same triples and folds:
    - **L:** issued at t_{r+1} with reading r+1's values withheld and no added signals. Inputs: M2's cuff features at r with h = t_{r+2} − t_r, plus the elapsed time t_{r+1} − t_r.
    - **U:** M2 issued at t_{r+1}.
    - **T:** L plus M4's HR and waveform blocks at issue time t_{r+1}, with M4's missing-data and fallback rules. Its fallback is L plus the HR block.
    - **T-P1:** T with donor signals.
  - **Fitting.** L, U and T use the chosen M2 form and the pooled horizon structure. A patient enters with ≥ 3 scored triples.
  - **F = N / D**, with N = RMSE_L − RMSE_T and D = RMSE_L − RMSE_U (patient-level means).
    - N and D are reported in mmHg with CIs.
    - F's CI is the percentile CI of N*/D*. A resample with D* ≤ 0 counts as +∞ if N* > 0, else −∞.
  - **When F is reported:** only if the 95% CI of D lies above zero.
  - **Placebo guard:** if the CI of RMSE_T − RMSE_T-P1 does not lie below zero, F is labelled "not distinguishable from placebo inputs".
  - **Inflation-artefact variant:** T is also reported with the 30 s before t_{r+1} excluded.
  - **Scope:** reported per cohort and in the fast-cycling S-dev cases, with the counts of triples and patients. S-ob is descriptive.
- **Controls:**
  - Placebo P2 uses the same patient's features from 10 min earlier. It is scored on forecasts after 12 min. A gain that survives P2 is reported as slow patient-state information.
  - Ablation by feature group: each group is added to M2 on its own. One group is "availability indicators only".
  - Results by quality stratum: fewer than 2 usable windows; 2 or more but under 80% usable; 2 or more and 80% or more usable.
  - Usable-window share from −60 to +60 s around each reading's display time.
- **New-onset hypotension.**
  - Definitions: below 80% of the record-start baseline; MAP < 65 mmHg; SBP < 100 mmHg.
  - An alarm is raised when the forecast falls below the threshold; no margin is tuned. For the SBP definition, the models are refitted with SBP in place of MAP.
  - Reported: counts, sensitivity, PPV, false alarms per 30 min and the median h of detected events. There is no AUROC for S-ob.
- **Large changes (descriptive).** Next-reading forecasts are split by the realised change in MAP from the issue reading: a fall of 10 mmHg or more, a rise of 10 mmHg or more, or neither. Per group: counts, pooled MAE and pooled mean signed error for M0, M2, M3, M4 and M4-P1. The split uses the outcome, so this is not a test. The 10 mmHg is fixed here and not tuned.
- **Intervals.**
  - Cross-conformal 90% for M2 and M4: the out-of-fold absolute-residual quantile per h-bin, with constant width. Coverage and half-width are reported per h-bin.
  - S-ob coverage is an empirical check only.
- **S-ob by vasopressor stratum** (phenylephrine or ephedrine). Expected at G0: pump 5, bolus without pump 11, neither recorded 4.
  - Phenylephrine (µg) and ephedrine (mg) are never summed.
  - The pump strip is in mL/h, with a secondary µg/min axis labelled "at the documented 100 µg/mL".
- **Also reported.** The standardised coefficients and selected penalties of M2–M4. A data-quality table: exclusions, gaps, the missing share of each feature group, and the agreement of detected beats with the monitor HR.

**Exploratory** (may be omitted without counting as a deviation, except the last item):
- the whole record;
- 3-reading-ahead targets;
- lead–lag;
- within-case relative amplitude;
- S-dev AUROC;
- leave-one-patient-out inside S-ob, run only after the frozen scoring and shown in the appendix only;
- a non-linear check: embeddings from a frozen, publicly available pretrained PPG encoder in place of the hand-crafted waveform features, in the same ridge model, with no fine-tuning. It runs on S-dev cross-validation first; on S-ob only after the frozen scoring, and in the appendix only. Any overlap between the encoder's pretraining data and VitalDB is disclosed. It never changes the headline;
- a gradient-boosting check, which is always run. scikit-learn `HistGradientBoostingRegressor` is fitted on the same targets, features, imputation and outer folds, in four arms: cuff only; plus heart rate; plus waveform; and the full arm again on P1 donor signals. The grid is fixed: learning rate 0.03 / 0.1; maximum leaf nodes 3 / 7 / 15; iterations 30 / 100 / 300, chosen by inner grouped CV. It is reported for S-dev cross-validation in `REPORT.md`. On S-ob it runs only after the frozen scoring and is shown in the appendix only. It never changes the primary outcome, the rule or Sentences 1–3. If its answer (rule met or not met) differs from Sentence 1, the README says so in one fixed line (`SPEC.md` §6.8).

## 8. Featured replays (rules fixed now)
Only cases with ≥ 8 scored primary forecasts are eligible for rules 1–2.
1. Among cases with ≥ 1 new-onset reading below 80% of the record-start baseline in the primary window, the S-ob case with the median M4 RMSE. With an even count, take the higher-RMSE case of the two middle cases. This case is the README GIF.
2. The S-ob case with the highest M4 RMSE.
3. The S-ob pump case with the most `PHEN_RATE` changes in the primary window. This rule was written after the five pump tracks had been seen; it uses no forecast.

Ties go to the lowest case ID. If one case meets two rules, the later rule takes its next case.

## 9. Information seen before this registration
**No model had been fitted to any case.**

**Seen during design, S-ob:**
- Case-table metadata for all 21 cases (the fields listed in `SPEC.md` §9; no drug total other than phenylephrine and ephedrine).
- Cuff timing, counts and hypotension-onset counts for all 21 cases, under two window anchors:
  - MAP < 65: 10 cases with 20 onsets, 12 of them in cases 5306, 3970 and 921;
  - below 80% of the baseline: 17 cases with 35 onsets.
- Cuff tracks of 265, 1780, 4147, 5306 and 6187.
- Pleth quality, template heuristic 1 at ≥ 0.90: 6187 43%, 1780 47%, 4147 92%, 307 48%.
- Pleth, HR and cuff tracks of 1780, 6187 and 307, checked with heuristic 2:
  - usable windows at ≥ 0.85 / 0.90 / 0.95: 1780 88 / 86 / 83%; 6187 96 / 96 / 96%; 307 78 / 73 / 68%;
  - forecast-level coverage: 22/22, 21/21 and 15/15;
  - coverage of a +30 s update (an analysis considered during design and later dropped): 19/22, 21/21 and 15/15;
  - no dip in the usable share around reading times.
- Phenylephrine pump tracks of 123, 265, 858, 4343 and 5530.

**Seen during design, S-dev:**
- Cuff timing and counts for all cases.
- Cuff and HR tracks of case 15.
- The pleth range of cases 15 and 39.
- Pleth quality of 249, 5409, 131 and 1711.
- The pump track of 3574.

**Between version 1.4 and this version:** the S-dev tracks were downloaded and work on the pipeline began. No model was fitted, no forecast error was computed, and no S-ob signal was accessed. The download checks (track presence, sample counts and times, missing-sample share, sampling clock, and the CSV time offset measured on heart rate) found two signal problems (S-dev cases 2030 and 2328, `SPEC.md` §2.3). The pipeline was then built and run on S-dev up to the features: cuff readings under the final rule, with counts of readings, exclusions and gaps (214 analysable cases); the case-table drug totals and airway fields of the analysable cases, which led to the neuromuscular-blocker rule (§2; 203 analysed cases); heart-rate features; pleth quality and coverage; the R-peak detector choice; and work on the waveform features. No other BP value was summarised, no feature was compared with BP, no model was fitted and no forecast error was computed.

From the commit of version 1.4 until the frozen scoring, no S-ob signal is accessed. The scoring itself is one run of the frozen tag. A rerun after a crash with unchanged code is allowed and logged. If the run cannot finish without a code change, the change is limited to the fault, is made without viewing any S-ob forecast or error, leaves the fitted models unchanged, and is reported with its diff. Anything computed on S-ob afterwards is labelled as computed after the frozen scoring and cannot change Sentences 1–3.

## 10. Deviations
Any departure from this file is listed in `REPORT.md` with its date and reason. A change made after S-ob scoring is labelled post hoc. An analysis added after this version is labelled post hoc exploratory, whenever it is added.

## 11. Changes from version 1.4
Version 1.4 (commit `fca645e`, 2026-11-19) was replaced by this version before any model was fitted. The hypothesis, windows, primary outcome, decision rule and pre-written sentences are unchanged. The cohorts gain one exclusion, and the waveform features lose the two notch features (both below).

**Changed in the confirmatory path**
- **Penalties.** Three blocks (cuff, heart rate, waveform), chosen in ladder order. Version 1.4 gave M4 one added block, so a shared penalty could affect the M4-vs-M3 contrast.
- **Horizon.** In the pooled structure, each heart-rate and waveform feature also enters as feature × h. Version 1.4 gave horizon interactions to the cuff features only.
- **Cohort.** Analysable cases with rocuronium or vecuronium recorded are excluded from fitting and scoring (§2; 11 of 214 S-dev cases). Set before any fit, from S-dev case-table drug totals; a sensitivity analysis includes them.
- **Notch features removed** (`SPEC.md` §4.3): a signal-level check before any fit showed that the notch detector is not reliable on this pleth.

**Fixed where version 1.4 left a choice open**
- **Cohort and cuff rule:** the cesarean rule, as used at G0; how changes are merged; the first value of a track; a value that returns after missing samples.
- **Fitting** (`SPEC.md` §4.1 and §4.5): training targets, weights, the rows used for M4, the meaning of "dropped", inner cross-validation, fold construction, the raw-lag form, bin edges and the refitting of variants.
- **Data source:** the versioned case files and the signal-integrity rule (`SPEC.md` §2.3). The rule was written after the S-dev download, on signal grounds only, and it makes S-dev cases 2030 and 2328 non-analysable.
- **Definitions:** the fallback and penalties of the yardstick model T; the P1 donor rule; the ablation, P2 and the quality strata; the SBP model behind the SBP hypotension definition.
- **Workflow:** what may run before the confirmatory run; public pushes of the two tags; a technical fault during S-ob scoring; the status of later S-ob outputs; the label for analyses added after this version.

**Added**
- **Statements** on multiplicity and sample size.
- **Secondary, descriptive:** mean signed error; pooled RMSE; the large-change table; S-dev by sex; fold-seed sensitivity; coefficients; a data-quality table; interval details.
- **Exploratory:** the gradient-boosting check.
- **Reproducibility:** lock file; data checksums.
- **README first lines:** a plain question, a plain answer line for each headline row and a cesarean line, shown above Sentences 1–3 and chosen with the same row (§6); the order of the README blocks and a summary figure (`SPEC.md` §8.1); one descriptive hypotension line under the README table; an author's note of one sentence; the top of the web page repeats the README's first block (`SPEC.md` §8.3). Sentences 1–3 are unchanged.
- **Details fixed before any fit** (`SPEC.md` §4.5): M2-structure ties, per-bin counts, the yardstick's elapsed-time input, P2 training rows, hypotension scoring, gradient-boosting ties, and which CI follows each number in Sentences 1–3 (§6). The placebo-guard wording of `SPEC.md` §3 now matches §6.3.
- **Reviewer guide and record check** (`SPEC.md` §8.3): `AGENTS.md` and `scripts/verify.py` may be added at any time; they change no analysis.
- **Signal integrity** (`SPEC.md` §2.3): a waveform with a header rate other than 500 Hz is treated as absent; waveforms are stored for 0–3660 s.
