# Pre-registration — between-the-cuffs

**Status: registered on 2026-11-19 (this commit).**

**What is registered.** This file and `SPEC.md` v1.4 are committed together, before any model is fitted to any case, and together they form the registration. The registration commit is the first commit of the public history and is made public before any model is fitted. The commit hash and the SHA-256 hashes of both files are quoted in `REPORT.md`. This file restates the confirmatory parts and wins any conflict with `SPEC.md`.

**G0** = a data-availability check run before registration. It looked at track presence, cuff timing and reading counts, plus a small pleth-quality sample. No model was fitted.

## 1. Hypothesis
After spinal anesthesia, adding heart rate and pulse-oximeter (pleth) waveform features to a cuff-only third-order autoregressive model lowers the error of the next-cuff-reading MAP forecast.

**Hypothesis risk, stated in advance.** A null or small result is plausible:
- **Large changes are hard.** In the closest published method (Lee et al., Diagnostics 2026;16:3713), a cuff-calibrated PPG delta model reproduced only 37–48% of SBP changes larger than 40 mmHg, and its accuracy decayed with time since the last cuff.
- **Little error is left.** Fast cuff cycling in the cesarean cases (60–90 s) leaves little error for added signals to remove.
- **Small training base.** The h-bin that holds most cesarean targets is trained on 34 development patients.

**What each outcome would mean** [judgment]:
- **Positive.** Heart rate and the pleth may partly stand in for the missing drug input, so a positive result cannot separate physiology from drug effect. It is probably an upper estimate for a system that already has timed doses.
- **Null.** A null is harder to attribute to the drug proxy, but opposite heart-rate responses to the two vasopressors could also cancel in a linear model. It bounds only this feature set under a linear model.
- **Either way,** only data with time-stamped vasopressor doses can settle the question.

## 2. Data and cohorts
- **Source.** VitalDB public API (CC BY 4.0). Data are downloaded at run time and not redistributed.
- **S-dev.** Spinal non-cesarean cases, used for fitting and cross-validation. Six cervical cerclage cases are kept and flagged.
- **S-ob.** Spinal cesarean cases, scored once after the freeze.
- **Subject overlap.** If a subject has cases in both cohorts, the S-dev case is excluded from fitting. At registration, no subject is in both cohorts.
- **Analysable.** Pleth and ECG present, and ≥ 6 valid cuff readings in the primary window.
  - Expected under the provisional G0 cuff rule: 218 S-dev and 20 S-ob cases.
  - Final counts are recomputed under the rule below. For S-ob this happens only at the frozen scoring.
- **Cuff reading.**
  - A reading is any change in SBP, DBP or MBP; changes within 15 s are merged.
  - Its time t_r is the last change in the cluster, and its values are those held at the first sample after t_r.
  - It is dropped, and counted, if MAP is outside 30–160 mmHg or SBP > MAP > DBP fails.
- **Time zero and windows.**
  - Time zero is the recording start (`casestart` = 0).
  - Primary window: 0–30 min.
  - Secondary window: 0 to incision (`opstart`), capped at 60 min.
- **Record-start baseline.** The mean of the first three valid readings. It is never called pre-spinal.

## 3. Forecasts
- **Issue points.** Every valid reading in the window from the third reading on. The issue time is t_r.
  - Cuff features use readings up to and including that reading.
  - HR and waveform features use only samples earlier than t_r.
- **Primary target.** The next reading, if it falls within 400 s and inside the window.
- **Horizon.** Each model outputs a forecast curve ŷ(h) and is scored at the realised time of the target reading.
  - h is the only input not known at the issue time, and it enters M1–M4 identically.
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
| **M4** | M3 + pleth and ECG waveform features (pulse arrival time, rise time, half-height width, notch timing, notch height ratio, beat-amplitude coefficient of variation). Each feature enters as its level relative to the median of the record's first 6 usable windows, and as its change since the previous reading. **No absolute amplitude.** Pulse arrival time is used only as a within-case change. |

**Fitting**
- **Method.** Ridge regression. The scaler and the imputation medians are fitted on training folds only.
- **Outer folds.** 5 folds grouped by `subjectid` and stratified by cycling class (case median interval < 90 s), seed 20261118.
- **Penalties.** M2's penalty is chosen by inner grouped CV from 13 values, 10⁻³ … 10³. In M3 and M4, the cuff block keeps M2's penalty; the added block gets its own penalty from the same grid plus a "dropped" option.
- **Before S-ob scoring,** every model is refitted on all of S-dev.

**M2 structure, fixed on S-dev before M3/M4 are fitted.** All four combinations of two choices are compared, and the one with the lowest patient-level mean RMSE wins:
- **M2 form:** level/slope/curvature, or a raw-lag AR(3) with the gaps as inputs;
- **Horizon structure:** per-bin models (any bin with fewer than 300 S-dev training targets uses the pooled model), or one pooled model with the interactions slope × h, curvature × h² and level × h.

The numbers of training targets and patients per h-bin are reported.

**Missing data and fallbacks**
- A missing HR group (fewer than 10 valid samples in 60 s) or a missing waveform feature group is set to the training-fold median and gets a missing-indicator.
- With fewer than 2 usable pleth windows between the previous reading (at most 120 s back) and t_r, M4 returns the M3 forecast and the forecast is flagged.

**Feature-code freeze.** The feature-extraction code and parameters are frozen and tagged before the first M3/M4 fit. They are chosen only on signal-level criteria (for example, R-peak agreement with the monitor HR), never on BP or forecast error. **The first complete S-dev cross-validation run at that tag is the confirmatory result.** It includes M2–M4, M3-P1, M4-P1 and the yardstick models, and it fixes Sentences 1 and 3 before S-ob is scored. A rerun with changed code is a deviation, and both results are shown.

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

**Placebo validity.** A positive claim about M4 requires both of the following, and so does a claim about heart rate alone (M3):
- its placebo version P1 does not meet the rule against M2;
- the 95% CI of RMSE(model) − RMSE(model-P1) lies below zero.

For P1, each recipient gets one donor: the next case in a seeded permutation within the same cohort whose HR and pleth cover the window. The donor's signals are processed at the recipient's reading and issue times, and the model is refitted.

**S-ob** is an external case series. Its result is reported but does not decide the headline.

**Coverage rule.** If fewer than 50% of S-ob primary forecasts have waveform features, the S-ob result is reported as "insufficient signal coverage", with its numbers still shown.

**Headline (chosen mechanically).**
- **Order of checks:** M4 vs M2 first, then M4 placebo validity, then M3 vs M2 and M4 vs M3.
- **Placebo checks:** placebo validity is checked only for a contrast that meets the rule, so a plain null is never reported as a placebo failure.
- **Row selection:** the first row that applies, in table order, is used, so rows 2–5 presuppose that M4 passed placebo validity. Row selection depends only on whether each contrast meets the rule. M3's placebo result changes the wording, never the row.
- **Symbols:** X, Z, Y and P are the relative reductions of M4 vs M2, M3 vs M2, M4 vs M3 and M4-P1 vs M2.
- **If M3 fails placebo validity:**
  - in row 2, "heart rate alone gave Z% (95% CI …)" becomes "heart rate's share did not pass its placebo check (Z%)";
  - in row 3, "and the gain came mainly from heart rate" becomes "but heart rate's share did not pass its placebo check (Z%)".
- **Waveform share:** it has no separate placebo; the feature-group ablation is its check.

Sentence 1 always begins "In N non-cesarean spinal cases (cross-validated), …":

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
- X% here is the M4-vs-M2 contrast scored on these cases.
- If F is not reported, the clause after the semicolon becomes "one extra cuff reading had no detectable value at this interval".
- If T fails its placebo guard, "(not distinguishable from placebo inputs)" follows F.

## 7. Secondary analyses (pre-specified)
- **Decomposition:** M3 vs M2 and M4 vs M3.
- **Metrics.** Each metric is computed per patient and summarised as mean ± SD.
  - MAE, R² and FIT for every model.
  - R² and FIT need at least 5 forecasts and a non-zero SST.
  - FIT = 100 × (1 − ‖y − ŷ‖ / ‖y − ȳ‖), with ȳ the patient's in-window mean. This is the MATLAB System Identification Toolbox definition. It has not been confirmed against the full text of Davoud et al. 2025 (doi:10.1007/s10877-025-01288-w) [unverified].
- **Subsets and variants:**
  - h-bins;
  - 2-reading-ahead targets;
  - the waveform-available subset;
  - the fast-cycling S-dev cases: median interval between consecutive valid readings in the primary window < 90 s under the final cuff rule (34 at G0). Only the scoring is restricted: predictions are the outer-CV predictions of the models fitted on all of S-dev, and nothing is refitted on the subset. K counts such cases with ≥ 3 scored forecasts;
  - the secondary window.
- **Sensitivity analyses:**
  - the alternative horizon structure;
  - h replaced by the median of previous intervals;
  - the template-correlation quality variant.
- **Withheld-cuff yardstick (key secondary).**
  - **Triples.** Consecutive valid readings r, r+1, r+2, with r at least the third reading and t_{r+2} − t_r ≤ 400 s.
  - **Three forecasts of reading r+2**, fitted on S-dev on the same triples and folds:
    - **L:** issued at t_{r+1} with reading r+1's values withheld and no added signals. Inputs: M2's cuff features at r with h = t_{r+2} − t_r, plus the elapsed time t_{r+1} − t_r.
    - **U:** M2 issued at t_{r+1}.
    - **T:** L plus M4's HR and waveform block at issue time t_{r+1}, with M4's missing-data and fallback rules.
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
  - Ablation by feature group, including an "availability indicators only" group.
  - Results by quality stratum.
  - Usable-window share from −60 to +60 s around each reading's display time.
- **New-onset hypotension.**
  - Definitions: below 80% of the record-start baseline; MAP < 65 mmHg; SBP < 100 mmHg.
  - An alarm is raised when the forecast falls below the threshold; no margin is tuned.
  - Reported: counts, sensitivity, PPV, false alarms per 30 min and the median h of detected events. There is no AUROC for S-ob.
- **Intervals.**
  - Cross-conformal 90%: the out-of-fold absolute-residual quantile per h-bin, with constant width.
  - S-ob coverage is an empirical check only.
- **S-ob by vasopressor stratum** (phenylephrine or ephedrine). Expected at G0: pump 5, bolus without pump 11, neither recorded 4.
  - Phenylephrine (µg) and ephedrine (mg) are never summed.
  - The pump strip is in mL/h, with a secondary µg/min axis labelled "at the documented 100 µg/mL".

**Exploratory** (may be omitted without counting as a deviation):
- the whole record;
- 3-reading-ahead targets;
- lead–lag;
- within-case relative amplitude;
- S-dev AUROC;
- leave-one-patient-out inside S-ob, run only after the frozen scoring and shown in the appendix only;
- a non-linear check: embeddings from a frozen, publicly available pretrained PPG encoder in place of the hand-crafted waveform features, in the same ridge model, with no fine-tuning. It runs on S-dev cross-validation first; on S-ob only after the frozen scoring, and in the appendix only. Any overlap between the encoder's pretraining data and VitalDB is disclosed. It never changes the headline.

## 8. Featured replays (rules fixed now)
Only cases with ≥ 8 scored primary forecasts are eligible for rules 1–2.
1. Among cases with ≥ 1 new-onset reading below 80% of the record-start baseline in the primary window, the S-ob case with the median M4 RMSE. With an even count, take the higher-RMSE case of the two middle cases. This case is the README GIF.
2. The S-ob case with the highest M4 RMSE.
3. The S-ob pump case with the most `PHEN_RATE` changes in the primary window. This rule was written after the five pump tracks had been seen; it uses no forecast.

Ties go to the lowest case ID. If one case meets two rules, the later rule takes its next case.

## 9. Information seen before this registration
**No model had been fitted to any case.**

**Seen during design, S-ob:**
- Case-table metadata for all 21 cases.
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

From this registration until the frozen scoring, no S-ob signal is accessed. The scoring itself is one run of the frozen tag. A rerun after a crash with unchanged code is allowed and logged.

## 10. Deviations
Any departure from this file is listed in `REPORT.md` with its date and reason. A change made after S-ob scoring is labelled post hoc.
