# between-the-cuffs — design specification

Version 1.4 (2026-11-19). This document fixes the design. It is committed together with `PREREG.md`, which restates the confirmatory parts. Both are part of the registration, and where they disagree, `PREREG.md` wins.

Marks: **[judgment]** = a design decision or assessment; **[unverified]** = a fact that could not be confirmed.
**G0** = a data-availability check run before this specification. It looked only at track presence, cuff timing and reading counts, plus a small pleth-quality sample; no model was fitted.

## 0. Question

After spinal anesthesia, blood pressure is measured by an oscillometric cuff every 1–5 minutes, while the pulse oximeter and ECG record continuously. **Do heart rate and pulse-oximeter (pleth) waveform features lower the error of a next-cuff-reading MAP forecast beyond a cuff-only third-order autoregressive model?**

The models are fitted on non-cesarean spinal cases in VitalDB, then scored once, frozen, on the spinal cesarean cases in VitalDB. The result is reported whichever way it falls.

The contribution is narrow. This analysis measures the *increment* of heart rate and pleth timing and shape features over a cuff-only autoregressive forecast of the next reading after spinal anesthesia. Every forecast is scored, and the result is checked with a placebo-input control. The question is **not** whether the pleth tracks blood pressure; earlier work has addressed that (§1).

**Why it matters.** A forecasting system that runs on intermittent cuff readings could add the pulse-oximeter waveform without a new sensor. The cost is integrating and quality-checking a second signal. Whether that is worth doing depends on how much next-reading error the waveform removes at the cuff interval in use. This analysis estimates that amount and states the largest gain the data are compatible with. A secondary yardstick (§3) expresses the result against the cuff itself: the share of one extra cuff reading's value that the added signals recover.

## 1. Prior work this design builds on

| Work | What it shows | Relation to this analysis |
|---|---|---|
| Kovacheva VP, Mahesh N, Davoud SC, et al. medRxiv 2026, doi:10.64898/2026.02.03.26345255 | Third-order ARX forecasts of MAP at 1, 2 and 3 min during cesarean delivery under spinal anesthesia, from 1-min cuff readings and time-stamped phenylephrine. Metrics are computed per patient and summarised as mean ± SD (RMSE, MAE, R², FIT). It notes that "waveform-based approaches require continuous pressure signals" and that sensitivity "may improve through hypotension-threshold recalibration, additional physiologic inputs (e.g., heart rate, trend features), or hybrid approaches …". Its stated next step is a randomized evaluation. | The cuff-only third-order model (M2), the per-patient metrics and the FIT definition follow this work. Heart rate is among the additional inputs the paper lists as one possible way to improve sensitivity; the pleth features are our own addition. This analysis does not reproduce, re-implement or compare against that model: the patients, protocol and inputs differ, and there is no drug term here. |
| Davoud SC, …, Doyle FJ, Kovacheva VP. J Clin Monit Comput 2025;39(4), doi:10.1007/s10877-025-01288-w | Empirical pharmacodynamic ARX model built with the MATLAB System Identification Toolbox. It uses the past three data points and is compared with a "mean constant model". | Source of the order-3 choice and of the likely FIT definition (§6.1). Full text not read [unverified]. |
| **Lee S, Al-antari MA, Lee G. Diagnostics 2026;16:3713, doi:10.3390/diagnostics16223713** | **Closest method neighbour.** 65 hand-crafted PPG features. A "delta" model predicts the change from the last cuff reading (MIMIC-BP, plus 17 external VitalDB cases with arterial lines). With cuff calibration, SBP MAE is 11.8 mmHg with 1 cuff and 10.8 mmHg with 5 cuffs. With one cuff, the model reproduced only 37–48% of SBP changes larger than 40 mmHg. Accuracy fell with time since the last cuff, and the recalibration interval mattered more than model complexity (VitalDB: 10.3 mmHg at 15 min, 8.0 mmHg at 3 min). The paper publishes a BP-blind PPG quality check on VitalDB `SNUADC/PLETH`. | We adopt its quality-check criteria and thresholds unchanged as our primary pleth quality index (§5). Differences: it tracks current BP between cuffs, while we forecast the next cuff reading. It does not address spinal anesthesia or cesarean delivery, and it has no cuff-only autoregressive baseline. Its large-change finding is a pre-stated reason to expect a small or null increment (§7). |
| Sharwood-Smith G, Bruce J, Drummond G. Br J Anaesth 2006, doi:10.1093/bja/aei266 | In cesarean delivery under spinal anesthesia, pulse transit time (ECG R wave to the upstroke of a toe plethysmograph, inside the blocked territory) tracked the change in MAP (r² = 0.55). A 20% rise in PTT was 74% sensitive and 70% specific for a MAP fall of more than 10%. | Precedent for the pulse-arrival-time feature, measured at a site inside the block. It tracks; it does not forecast, and it has no autoregressive baseline. The VitalDB probe site is not documented [unverified]; at a finger the result may not carry over (§4.3). |
| Bolea J, et al. Ann Biomed Eng 2017, doi:10.1007/s10439-017-1864-y; Higashi M, et al. Eur J Anaesthesiol Intensive Care 2025, doi:10.1097/EA9.0000000000000086 | ECG-plus-pleth timing in cesarean delivery under spinal anesthesia. Bolea: pulse-transit-time and pulse-rate variability measured before the block classified later hypotension (sensitivity 76%, specificity 70%; n = 51). Higashi: a randomized trial in which vasopressor dosing guided by a cardiac-output estimate based on pulse-wave transit time lowered the incidence of hypotension (19% vs 49%). Abstracts only. | Evidence that ECG-to-pleth timing carries usable information in this setting. Neither measures the increment over a cuff-only autoregressive forecast of the next reading. |
| Frassanito L, et al. Anesth Analg 2022, doi:10.1213/ane.0000000000005754; Shih PY, et al. Anaesthesia 2026, doi:10.1111/anae.70161 | Hypotension Prediction Index on continuous finger-cuff pressure waveforms in elective cesarean (retrospective n = 50; randomized n = 171). | Continuous-*pressure* waveform prediction in the same clinical setting. This analysis uses the pulse oximeter, which needs no pressure waveform. |
| Lee S, Lee HC, et al. Br J Anaesth 2021, doi:10.1016/j.bja.2020.12.035 | Models using PPG and other non-invasive signals to predict hypotension 5–15 min ahead in 3,301 noncardiac surgical cases. | PPG-based prediction, not specific to spinal anesthesia, and without a cuff-autoregressive baseline. |
| Kondo Y, et al. PLoS One 2025, doi:10.1371/journal.pone.0316715 | Meta-analysis: the perfusion index and pleth variability index measured before spinal anesthesia predict post-spinal hypotension in cesarean delivery (summary AUC about 0.75). | Baseline risk scores, not forecasts. A non-peer-reviewed preprint (ref. 10) reports that the public VitalDB release has no monitor perfusion-index channel. |

How prior work was searched: targeted searches of Europe PMC and the web, then reading the papers above. This is not a systematic review.

## 2. Data

### 2.1 Source, licence, credit
- **Source.** VitalDB (Lee HC, et al. Sci Data 2022, doi:10.1038/s41597-022-01411-5), accessed through the public API at run time. It is released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) and the VitalDB Data Use Agreement. **No VitalDB data are redistributed in this repository**; the code downloads them when it runs.
- **Credit.** Every public output carries the credit, the licence link and this statement of changes: "Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person."

### 2.2 Cohorts

| Cohort | Definition | Cases | Analysable at G0 | Role |
|---|---|---|---|---|
| S-dev | `ane_type` = spinal, operation not cesarean. Six cervical cerclage cases are kept and flagged. | 252 (251 subjects) | 218 | Fitting and model selection by cross-validation grouped by `subjectid` |
| S-ob | `ane_type` = spinal, cesarean | 21 (21 subjects) | 20 (case 4147 has 3 readings in the window) | External case series, scored once after the freeze |

- **Analysable** = `SNUADC/PLETH` and `SNUADC/ECG_II` present, and at least 6 valid cuff readings in the primary window.
- **Subjects.** The only subject with two spinal cases has both of them in S-dev. No subject appears in both cohorts. A rule covers any later overlap: if a subject has cases in both cohorts, the S-dev case is excluded from fitting.
- **Provisional counts.** The G0 counts were made with a provisional cuff rule: SBP and MBP changes only, MAP 20–200 mmHg. Under the final rule (§2.4), the S-dev counts are recomputed before any model is fitted. The S-ob counts are recomputed only at the frozen scoring, and one S-ob case has exactly 6 readings at G0. Both cohorts are reported with a cohort-flow diagram.
- **Who is in each cohort.**
  - S-ob: age 28–44; 19 of 21 flagged emergency; includes twin and one triplet deliveries.
  - S-dev: 179 male / 73 female, median age 60; mostly hernia, vein and anal surgery.

### 2.3 Signals
- **Cuff:** `Solar8000/NIBP_SBP`, `NIBP_DBP`, `NIBP_MBP` (held values).
- **Heart rate:** `Solar8000/HR`.
- **Waveforms:** `SNUADC/PLETH` and `SNUADC/ECG_II`, both at 500 Hz.
- **Pump:** `Orchestra/PHEN_RATE`, the phenylephrine pump rate in **mL/h**. VitalDB documents the syringe as 100 µg/mL.
- **Case-table fields:** `opstart`, `anestart`, `intraop_phe` (**µg**), `intraop_eph` (**mg**), `subjectid`, `emop`.

### 2.4 Cuff readings (final rule)
- **What counts as a reading.** The cuff tracks hold the last value. A new reading is any time at which SBP, DBP or MBP changes; changes within 15 s are merged into one reading.
- **Reading time.** The reading time t_r is the time of the *last* change in the merged cluster. The reading's values are those held at the first sample after t_r.
- **Exclusions.** A reading is dropped, and counted, if MAP lies outside 30–160 mmHg or if SBP > MAP > DBP does not hold. This is the only outcome-dependent exclusion, and it is reported.
- **Gaps.** A repeated identical reading is invisible to this rule. Gaps longer than twice the case's median interval are flagged and counted.
- **No imputation.** Every target is a real reading.
- **Display time.** t_r is the time the reading is displayed. The pressure was measured during the inflation just before it; the inflation length is [unverified].

### 2.5 Time zero, windows, baselines
- **Time zero:** start of the recording (`casestart` = 0).
- **Primary window: 0–30 min.**
  - Incision (`opstart`) falls at a median of 28 min in S-ob (quartiles 20 / 28 / 36; range 2–46) and 27 min in S-dev, so this window roughly covers the period up to incision.
  - In about half of the cesarean cases, the primary window runs past incision and may include delivery and oxytocin.
- **Secondary window:** 0 to incision, capped at 60 min. It ends at incision so that delivery and oxytocin are excluded.
- **Injection time is unknown.** `anestart` is a charted time recorded on a 5-min grid, not the injection time. The recording starts after `anestart` in 254 of 273 spinal cases, and the first cuff reading comes after `anestart` in 261. A pre-spinal baseline therefore does not exist for most cases.
- **Record-start baseline:** the mean of the first three valid readings of the record. It is never called "pre-spinal".

### 2.6 Vasopressor information (descriptive only)
- **Bolus times** are not in VitalDB. Every forecast is therefore made "under usual care", and the models have no drug input.
- **Chart totals miss pump infusions.** In three cesarean cases, the pump delivered 13–61 mL while `intraop_phe` was 0.
- **S-ob strata**, counting phenylephrine *or* ephedrine on the 20 cases analysable at G0:
  - pump track present: 5;
  - bolus total > 0 without a pump track: 11;
  - neither recorded: 4. "Neither recorded" does not prove that no drug was given.

  Phenylephrine alone gives 5 / 10 / 5. Final strata follow the final analysable set.
- **Phenylephrine (µg) and ephedrine (mg) are never added together**, in any table or chart.
- **Pump strip.** The primary axis is in **mL/h**. A secondary axis in µg/min is labelled "at the VitalDB-documented concentration of 100 µg/mL" (1 mL/h = 1.67 µg/min). Whether every syringe held that concentration cannot be checked [unverified].

## 3. Forecast task

- **Issue point and issue time.** An issue point is any valid reading in the window from the third reading of the record onwards. The issue time is t_r.
  - Cuff features use readings up to and including that reading.
  - Heart-rate and waveform features use only samples earlier than t_r.
- **Targets.** The next 1, 2 and 3 readings, if they fall within 400 s of t_r and inside the window. **Primary target: the next reading.** At most 1,593 primary targets in S-dev and 297 in S-ob under the G0 rule (readings − 3, before the 400-s cap).
- **The horizon h.**
  - **What h is.** Each model outputs a forecast curve ŷ(h) for h in 15–400 s and is scored at the realised time of the target reading. h is therefore the **only input not known at the issue time**. It enters M1–M4 identically.
  - **Sensitivity analysis.** Because extra readings may be taken when staff suspect a change, h can carry outcome information. A sensitivity analysis replaces h with the median of the patient's previous intervals.
- **h-bins for reporting:** 15–90 s, 90–150 s, 150–210 s and 210–400 s. These line up with the 1-, 2- and 3-minute horizons.
  - Most S-ob next-reading targets fall in the first bin: 16 of 21 cases cycle every 60–90 s.
  - Most S-dev targets fall in the third or fourth bin. The median interval is 168 s over all S-dev cases and 154 s over the analysable ones.
- **Training base per bin.** The first bin, which holds most S-ob targets, is trained mainly on the **34 analysable fast-cycling S-dev cases** (median interval < 90 s). The numbers of training targets and patients per bin are reported.
- **Withheld-cuff yardstick (key secondary).** This measures how much of one cuff reading's information the added signals recover.
  - **Triples.** Three consecutive valid readings r, r+1, r+2 in the window, where r is at least the third reading of the record and t_{r+2} − t_r ≤ 400 s.
  - **Three forecasts of reading r+2**, fitted on S-dev on the same set of triples with the same folds:
    - **L (one reading less):** issued at t_{r+1}, with the values of reading r+1 withheld and no added signals.
      - Inputs: M2's cuff features at reading r with h = t_{r+2} − t_r, plus the elapsed time t_{r+1} − t_r.
      - The time of an inflation is known in real time even when its values are hidden, so L and T share the same timing inputs and differ only in the added signals.
      - The plain 2-reading-ahead M2 forecast issued at t_r is reported beside it.
    - **U (one reading more):** M2 issued at t_{r+1}, in the same form as the primary M2 forecast.
    - **T (withheld cuff):** L plus M4's heart-rate and waveform block at issue time t_{r+1}. These are the same features M4 uses at that issue point; their "change" terms are taken since t_r. T follows M4's missing-data and fallback rules, and every triple is scored.
    - **T-P1:** T with donor signals, as in P1 (§6.6).
  - **Fitting.**
    - L, U and T use the M2 form chosen in §4.1 and the pooled horizon structure.
    - Penalties are chosen as in §4.1. T keeps L's cuff-block penalty; its added block has its own penalty, with the "dropped" option.
    - A patient enters with ≥ 3 scored triples.
    - These models are part of the confirmatory S-dev run and are frozen with the other models (§9).
  - **Yardstick.**
    - N = RMSE_L − RMSE_T; D = RMSE_L − RMSE_U; F = N / D. All RMSEs are patient-level means.
    - N and D are always reported in mmHg with 95% CIs.
    - The CI of F is the percentile CI of N*/D* over the same paired patient-level bootstrap. A resample with D* ≤ 0 counts as +∞ if N* > 0, and as −∞ otherwise.
    - F is not truncated to 0–100%.
  - **When F is reported.** Only if the 95% CI of D lies above zero, that is, if one extra cuff reading is measurably worth something. Otherwise the result reads "one extra cuff reading had no detectable value at this interval".
  - **Placebo guard.** If the 95% CI of RMSE_T − RMSE_T-P1 does not lie below zero, F is reported as "not distinguishable from placebo inputs".
  - **Inflation-artefact variant.** T is also reported with signal samples from the 30 s before t_{r+1} excluded. The 30 s is [judgment]; the inflation length is [unverified].
  - **Reporting.**
    - Results are given per cohort and in the fast-cycling S-dev cases (§6.7; 34 at G0). S-ob is descriptive.
    - The numbers of triples and patients are reported, because the 400-s cap thins S-dev, where the median interval is about 160 s.
  - **Leakage.** The leakage test (§4.4) also perturbs the withheld values of reading r+1, and T's features must not change.
  - **Interpretation.**
    - F is the share of one reading's value recovered by this feature set under a linear model.
    - It bears on whether a 1-minute cycle could be relaxed in an awake patient, but it does not show that this would be safe.
    - The inflation for reading r+1 did take place. If the oximeter probe is on the cuff arm, the pleth during that inflation may carry pressure information that would not exist had the reading been skipped. The probe site is [unverified].

## 4. Models

### 4.1 Ladder
All models except M00 predict the change in MAP from the last reading to the target.

| Model | Inputs |
|---|---|
| M00 | Causal running mean of the readings so far in the record |
| M0 | Persistence: the last reading |
| M1 | Linear trend through the last three readings, using their real timestamps, extrapolated by h |
| **M2** | **Cuff only, third order:** last MAP minus the record-start baseline; slope (mmHg/min) and curvature (mmHg/min²) of the last three readings, using their real timestamps; last pulse pressure; time since record start; h |
| **M3** | M2 + heart rate (§4.2) |
| **M4** | M3 + pleth and ECG waveform features (§4.3) |

**Structure of M2, fixed on S-dev before M3/M4 are fitted.** All four combinations of two choices are compared, and the combination with the lowest patient-level mean RMSE in S-dev cross-validation wins. The winning structure is then used for M2, M3 and M4.
- **The M2 form:**
  - (a) level / slope / curvature, as in the table;
  - (b) a raw-lag AR(3) on the last three MAP values (minus the baseline), with the two gaps and h as inputs.
- **The horizon structure:**
  - (a) one model per h-bin. A bin with fewer than 300 training targets in all of S-dev uses structure (b) instead.
  - (b) one pooled model with horizon interactions: slope × h, curvature × h², level × h.

The structure that loses on the horizon choice is reported as a sensitivity analysis. The stronger cuff-only baseline is chosen on purpose, so that any gain from the added signals is not inflated.

**Fitting.**
- **Method.** Ridge regression on standardised features. The scaler and the imputation medians are fitted on training folds only.
- **Outer loop.** 5 folds grouped by `subjectid` and stratified by cycling class (case median interval < 90 s or not), seed 20261118.
- **Penalties.**
  - The penalty for M2 is chosen from a fixed grid (10⁻³ … 10³, 13 values) by inner cross-validation grouped by subject.
  - In M3 and M4, the cuff block keeps M2's penalty. The added block has its own penalty from the same grid plus a "dropped" option, chosen by inner cross-validation. This way, a nested model cannot do worse than M2 only because one penalty is shared [judgment].
- **For S-ob scoring,** each model is refitted on all S-dev, with penalties selected by cross-validation on all S-dev.
- **Why linear.** About 220 patients cannot support a deep network, and each coefficient can be read and defended.

### 4.2 Heart-rate features (M3)
From `Solar8000/HR`:
- **Level:** the median over the 60 s before t_r.
- **Change:** that level minus the same median before the previous reading.
- **Slope:** the least-squares slope over the 60 s before t_r.

HR is **missing** if there are fewer than 10 valid samples in those 60 s. A missing group is set to the training-fold median and gets a missing-indicator. The monitor's HR averaging lag is undocumented [unverified]; it delays information but does not leak it.

### 4.3 Waveform features (M4)
- **Processing.** Each 10-s window is processed on its own; nothing is filtered or normalised over the whole record.
  - **Fiducial points:** found at 500 Hz. The pleth is band-passed 0.5–8 Hz within the window (2nd-order Butterworth, zero-phase).
  - **ECG R peaks:** found with a fixed detector chosen on S-dev by its agreement with the monitor HR (§9).
- **Per beat:**
  - **Pulse arrival time:** R peak to the pleth foot, with the foot found by the intersecting-tangent method.
  - **Rise time:** foot to systolic peak.
  - **Pulse width** at half height.
  - **Dicrotic-notch time** relative to the systolic peak.
  - **Notch height ratio:** notch height ÷ peak height, both measured from the foot.
- **Per window:**
  - **Beat-amplitude variability:** the coefficient of variation of beat amplitude within the window. As a ratio, it does not depend on scale.
  - The median over beats of each per-beat feature. A beat without a detectable notch leaves the notch features missing for that beat.
- **Per forecast.** Each feature is the median over usable windows (§5) between the previous reading (at most 120 s back) and t_r. At least **2 usable windows** are required. Each feature enters in two forms, both of which are differences within the case:
  - (i) its level relative to a **reference** = the median over the first 6 usable windows of the record. The level is missing until those 6 windows end before t_r.
  - (ii) its change since the previous reading.
- **Pulse arrival time is used only as a within-case change.** Lee et al. 2026 report that ECG and PPG channels in public databases are not reliably time-aligned. A constant offset between the channels cancels out; an offset that drifts would not [unverified].
- **What pulse arrival time mixes.** Pulse arrival time = pre-ejection period + pulse transit time. The pre-ejection period is cardiac: it changes with filling, contractility and diastolic pressure. The transit time is vascular: it changes with pressure and with the tone of the arteries on the path. One ECG lead and one pleth site cannot separate the two, so a change in this feature is not read as a change in pressure or arterial stiffness alone. This limits interpretation, not the forecast test. Sharwood-Smith et al. 2006 measured to a toe, where the block itself lengthens the transit time (Babchenko et al. 2000, ref. 12; abstract only). If the VitalDB probe is on a finger, that component is absent, and the association with MAP may be weaker here [judgment].
- **Fallback.** With fewer than 2 usable windows, M4 returns the M3 forecast and is flagged "no waveform". A single missing feature group is handled like missing HR: training-fold median plus a missing-indicator. Examples are no detectable notch, or no R peaks for pulse arrival time.
- **No absolute pleth amplitude.**
  - The stored pleth looks display-scaled: two S-dev cases share the same bounded range and centre level [judgment].
  - A non-peer-reviewed preprint reports that the VitalDB pleth scale is auto-ranged across cases (ref. 10).
  - Whether the gain changes within a case is [unverified]. Within-case relative amplitude is therefore exploratory.

### 4.4 Leakage rule
- **Perturbation test.** A unit test replaces with random values:
  - every HR, pleth and ECG sample at or after the issue time;
  - every cuff reading after the issue reading.

  All features except h must be unchanged. The test covers every feature group, and also the withheld values of reading r+1 in the yardstick (§3).
- **Causal baseline.** The record-start baseline uses readings 1–3, all of which come at or before the first issue point.

## 5. Pleth quality index (pre-specified)

**Primary index: the published BP-blind rule of Lee et al. 2026, criteria and thresholds unchanged.**

**Steps.**
1. Resample the pleth to 125 Hz by taking every 4th sample of the 500 Hz record, with no interpolation.
2. Cut it into consecutive non-overlapping 10-s windows of 1,250 samples.
3. Discard a window that has missing samples, or that is flat (standard deviation 0).
4. Standardise each remaining window to zero mean and unit variance.

**Criteria.** All four are computed on the *unfiltered* 125 Hz window. A window is **unusable** if any of them holds:
1. the dominant frequency lies outside 0.67–3 Hz (40–180 beats/min);
2. the spectral peak holds less than 15% of the in-band power;
3. the standardised amplitude exceeds 6;
4. constant runs of at least 0.5 s cover 10% or more of the window.

**Operational definitions.** The paper does not state these details, so they are fixed here [judgment]:
- **Spectrum:** Hann-windowed periodogram of the full 1,250-sample window, constant detrend, 0.1 Hz bins.
- **Dominant frequency:** the largest bin in 0.5–8 Hz.
- **Peak power:** that bin ± 1 bin, divided by the 0.5–8 Hz sum.
- **Standardised amplitude:** max |z|.
- **Constant run:** at least 63 equal consecutive samples. The window is unusable if such runs total at least 125 samples.

**Added condition.** A window contributes waveform features only if beat delineation finds at least 4 beats. This is our addition, not part of the published rule.

**Caveat.** The rule was published on anesthetised patients, and 97% of segments passed it. Awake cesarean patients move more, and the rule may be lenient for them [judgment].

**Disclosure.** An earlier draft used a beat-template correlation threshold of ≥ 0.85. That value was chosen *after* seeing the usable-window shares of cesarean cases at three thresholds, so it is **not** the primary index. It is kept as the stricter sensitivity analysis (primary rule plus template correlation ≥ 0.85), labelled as chosen after a look at test-set quality. On the same cesarean cases, two quick heuristics used during design disagreed by 25–53 percentage points at 0.90 (§9). Design-time window shares are therefore not treated as stable.

**Coverage.** Reported per cohort as the share of forecasts with ≥ 2 usable windows. The usable-window share is also reported by time relative to each reading's display time (−60 s to +60 s), to check whether the pleth is lost during cuff inflation.

## 6. Evaluation

### 6.1 Metrics
Each metric is computed **per patient** over that patient's scored forecasts, then summarised as mean ± SD across patients.
- **Minimum forecasts.** A patient enters RMSE and MAE with ≥ 3 scored forecasts in the analysis at hand. The patient enters R² and FIT with ≥ 5 forecasts and a non-zero SST. Patients dropped by these rules are counted [judgment on the thresholds].
- **RMSE and MAE** in mmHg.
- **R² = 1 − SSR/SST.** SST uses the patient's in-window mean of the observed target readings.
- **FIT = 100 × (1 − ‖y − ŷ‖ / ‖y − ȳ‖),** with ȳ the patient's in-window mean.
  - This is the MATLAB System Identification Toolbox definition (`compare`, ref. 11).
  - It fits the R²/FIT pairs reported by Kovacheva et al. 2026: a mean FIT of 21.1% against an R² of 0.343 rules out FIT = 100 × R².
  - Not confirmed against the full text of Davoud et al. 2025 [unverified].
- **M00** is a **causal running mean**. It is not the "mean constant model" of the earlier work, which is probably the in-window mean already built into FIT [judgment].

### 6.2 Primary outcome
**Δ = (patient-level mean RMSE of M4) − (patient-level mean RMSE of M2)**, computed with:
- the issue time at a reading;
- the next reading as target;
- the primary window;
- **all forecasts scored**, including those where M4 fell back.

Δ is reported in mmHg for S-dev, using outer cross-validation predictions, and for S-ob, using the frozen models.

**Relative reduction** = 1 − (mean patient RMSE of the model) / (mean patient RMSE of the comparator). Its 95% CI is a percentile CI from the same paired patient-level bootstrap as Δ (2,000 resamples, seed 20261118; S-dev resamples by `subjectid`).

**Limits of the CIs.** The S-dev CI treats the fold models as fixed, so it is somewhat narrow. The S-ob CI rests on about 20 patients and is descriptive. Per-patient differences are plotted.

### 6.3 Decision rule and pre-written headlines
- **A contrast "meets the rule"** if, in S-dev, the relative reduction is **≥ 5%** and the **upper 95% CI limit of Δ is below zero**.
  - The comparator is the lower model of the pair: M2 for M3 vs M2 and M4 vs M2; M3 for M4 vs M3.
  - The 5% is a detection margin, not a clinical threshold. At typical errors it is a few tenths of a mmHg. A rule of the same form (a minimum gain plus a CI that excludes zero) was used by Lee et al. 2026.
- **Placebo validity.** A positive claim about a model requires that its placebo P1 version (§6.6) does not meet the rule against M2. It also requires that the 95% CI of RMSE(model) − RMSE(model-P1) lies below zero. This applies to M4, and to M3 when heart rate alone is claimed.
- **S-ob is an external case series.** Its result is reported beside S-dev and does not decide the headline.

**The headline is chosen mechanically.**
- **Order of checks:** M4 vs M2 first, then M4 placebo validity, then M3 vs M2 and M4 vs M3.
- **Placebo checks:** placebo validity is checked only for a contrast that meets the rule, so a plain null is never reported as a placebo failure.
- **Row selection:** the first row that applies, in table order, is used, so rows 2–5 presuppose that M4 passed placebo validity. Row selection depends only on whether each contrast meets the rule. M3's placebo result changes the wording, never the row.
- **Symbols:** X, Z, Y and P are the relative reductions of M4 vs M2, M3 vs M2, M4 vs M3 and M4-P1 vs M2.
- **If M3 fails placebo validity:**
  - in row 2, "heart rate alone gave Z% (95% CI …)" becomes "heart rate's share did not pass its placebo check (Z%)";
  - in row 3, "and the gain came mainly from heart rate" becomes "but heart rate's share did not pass its placebo check (Z%)".
- **Waveform share:** it has no separate placebo; the feature-group ablation is its check.

**Sentence 1** (S-dev) always begins "In N non-cesarean spinal cases (cross-validated), …":

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

**Sentence 2** (S-ob) is always: "In the n cesarean cases (external, frozen models): X% (Δ … mmHg, 95% CI …), waveform coverage C%; this is a case series and neither confirms nor refutes the first sentence." If the coverage rule (§6.5) applies, the variant is: "… insufficient signal coverage (C%); X% (95% CI …) is shown for completeness."

**Sentence 3** (key secondary, the cesarean-like cuff regime) is always: "In the K fast-cycling development cases (median cuff interval < 90 s): X% (Δ … mmHg, 95% CI …); one extra cuff reading's value recovered by the added signals: F% (95% CI …)."
- X% here is the M4-vs-M2 contrast scored on these cases.
- If F is not reported, the clause after the semicolon becomes "one extra cuff reading had no detectable value at this interval".
- If T fails its placebo guard, "(not distinguishable from placebo inputs)" follows F.

### 6.4 Decomposition (pre-specified, secondary)
M3 vs M2 tests heart rate alone; M4 vs M3 tests the waveform beyond heart rate. Both use the same rule.

### 6.5 Coverage rule
If fewer than 50% of S-ob primary forecasts have waveform features, the S-ob result is reported as "insufficient signal coverage", whatever the RMSE.

### 6.6 Echo controls — forecast, or an echo of the last cuff?
1. **New onset only.** A hypotension event counts only when the previous reading was not hypotensive.
2. **No selection.** Every issue point is scored; nothing is selected on the outcome.
3. **Placebo inputs**, each refitted with the full pipeline:
   - **P1 (primary placebo).** Each recipient gets one donor: the next case in a seeded random permutation within the same cohort (S-dev donors for S-dev, S-ob donors for S-ob) whose HR and pleth cover the window. Features are computed by running the recipient's reading and issue times on the donor's signals, under the donor's own quality mask. Both M3-P1 and M4-P1 are fitted. P1 should remove any gain.
   - **P2.** The same patient's features from 10 min earlier. It is scored only on forecasts issued after 12 min, against M4 on the same forecasts. P2 shows how much of any gain needs recent signal. A gain that survives P2 is slow patient-state information and is reported as such.
4. **Feature-group ablation.** The groups are HR; pulse arrival time; the shape features; beat-amplitude variability; and *availability indicators only*. The last group checks that a gain from signal loss is not reported as waveform morphology.
5. **Quality strata.** Results by quality stratum, and the usable-window share around reading times (§5).

### 6.7 Secondary analyses (pre-specified; reported in `REPORT.md`)
- **Bins and samples.**
  - Results by h-bin.
  - 2-reading-ahead targets.
  - The subset of forecasts with waveform features. This subset is never primary, because clean-signal windows may select calmer patients.
  - Results restricted to the fast-cycling S-dev cases: median interval between consecutive valid readings in the primary window < 90 s under the final cuff rule (34 at G0). Only the scoring is restricted: predictions are the outer-CV predictions of the models fitted on all of S-dev, and nothing is refitted on the subset. K counts such cases with ≥ 3 scored forecasts.
- **Variants.**
  - The withheld-cuff yardstick (§3), in both cohorts and in the fast-cycling S-dev cases.
  - The secondary window.
  - The sensitivity analyses: horizon structure; h replaced by the median of previous intervals; template-correlation quality.
- **Metrics.** MAE, R² and FIT for every model.
- **New-onset hypotension.**
  - Three definitions side by side: below 80% of the record-start baseline; MAP < 65 mmHg; SBP < 100 mmHg.
  - An alarm fires when the forecast is below the threshold; there is no tuned margin.
  - Reported: event counts, sensitivity, PPV, false alarms per 30 min, and the median h of detected events. A forecast's lead time is simply h.
  - S-ob has only 20 MAP < 65 onsets, in 10 cases, and 12 of them come from 3 cases. There is therefore no AUROC for S-ob.
- **Intervals.**
  - Cross-conformal 90% intervals: the 90% quantile of out-of-fold absolute residuals per h-bin, with constant width.
  - S-dev coverage is computed with the test fold left out of calibration.
  - No coverage guarantee holds for S-ob, because the population differs and bin 1 rests on 34 patients. S-ob coverage is an empirical check.
- **Vasopressors.** S-ob by vasopressor stratum (§2.6).

### 6.8 Exploratory (may be omitted without counting as a deviation)
- Whole-record window.
- 3-reading-ahead targets.
- Lead–lag between feature changes and the cuff MAP change.
- Within-case relative amplitude.
- S-dev AUROC for hypotension.
- Leave-one-patient-out refitting inside S-ob (M2 vs M4). This runs only after the frozen scoring, appears in the appendix only, and never in the README table.
- A non-linear check: embeddings from a frozen, publicly available pretrained PPG encoder, fed into the same ridge model in place of the hand-crafted waveform features. There is no fine-tuning. It runs on S-dev cross-validation first; on S-ob only after the frozen scoring, and in the appendix only. Any overlap between the encoder's pretraining data and VitalDB is disclosed. It never changes the headline. This is built only if everything else is finished.

## 7. Hypothesis risk and interpretation limits (stated in advance)
**A null or small result is plausible:**
- In Lee et al. 2026, a cuff-calibrated PPG delta model reproduced only **37–48% of SBP changes larger than 40 mmHg**, and its accuracy decayed with time since the last cuff. Rapid post-spinal falls are exactly the large changes that matter.
- With 60–90 s cycling in S-ob, persistence and a cuff-only third-order model leave little error for added signals to remove.
- The first h-bin is trained on 34 patients.

**Heart rate may act as a drug proxy** [judgment]. Unrecorded boluses move heart rate in opposite directions: it falls after phenylephrine and rises after ephedrine. A gain from heart rate may therefore reflect undocumented vasopressor doses rather than physiology, and its sign may not transfer between cohorts if the drug mix differs.

**What each outcome would mean** [judgment]:
- **Waveform share.** It has no separate placebo; the feature-group ablation is its check.
- **Positive.** Heart rate and the pleth may partly stand in for the missing drug input. Such a result cannot separate physiology from drug effect, and it is probably an upper estimate of what the same signals would add to a system that already has timed doses.
- **Null.** A null is harder to attribute to the drug proxy, but opposite heart-rate responses to the two vasopressors could also cancel in a linear model. It bounds only this feature set under a linear model.
- **Either way,** only data with time-stamped vasopressor doses can settle the question. §8.1 states the next experiment.

A clean null is a valid outcome and is reported with its controls.

## 8. Outputs

### 8.1 README
The README holds only these blocks, in this order. Anything else goes to `REPORT.md`.
1. **Headline.** Sentences 1–3 (§6.3), whichever way they fell.
2. **What this means.** Two or three lines, taken from the pre-written outcome reading in §7 that matches the result.
3. **Replay GIF.** A 20-s replay of the rule-1 case (§8.2), with its case ID shown. It contains:
   - cuff dots;
   - reference lines at 80% of the record-start baseline and at 65 mmHg;
   - M2 and M4 forecast curves with their interval bands;
   - the pleth strip, with unusable windows greyed out;
   - the phenylephrine pump strip in mL/h, if the case has one.
4. **One table and one figure.**
   - Table rows: M00, M0, M2, M3, M4 and M4-P1. Columns: patient-level mean RMSE with 95% CI for S-dev and S-ob, and waveform coverage.
   - Figure: per-patient paired dots of RMSE for M0, M2, M4 and M4-P1. It shows a null as clearly as a gain.
   - MAE, R², FIT, M1, the bins and the sensitivity analyses go to `REPORT.md`.
5. **Next experiment.** One paragraph: on a platform with time-stamped vasopressor doses and 1-minute cuffs, test the increment of the waveform over an ARX model *with* the drug term — retrospectively first, then scored silently and prospectively — with the same placebo and echo controls, and with the drug-proxy confound named. The paragraph may also propose measurements that this dataset lacks (additional sensors or monitor numerics) and say what each would separate. Such a proposal is not a result, and the same proposal is made for either outcome. Drafts for both outcomes are written before S-ob is scored.
6. **What this does not show** (top four; the full list is in `REPORT.md`):
   - the analysis is retrospective, with 21 cesarean cases (20 analysable at G0), 19 flagged emergency;
   - the spinal injection time is unknown, and the baseline is a record-start value, not pre-spinal;
   - bolus times are absent and the models have no drug input, so a heart-rate gain may reflect unrecorded drugs;
   - this is a research prototype, not medical advice.
7. **Author's note.** One sentence: the echo and placebo controls ask of a time-series model the question the author studies in imaging — does the model use the evidence in front of it (here, the pleth), or infer from context (the cuff history)? A second sentence: "The analysis was designed, coded and reviewed with AI assistance; the author made the design decisions and is responsible for them."
8. **Credit.** VitalDB credit, the licence link and the changes made.

The full limits list in `REPORT.md` also covers:
- the window extending past incision in about half of the cesarean cases;
- a development cohort of mostly older men with slower cuff cycling;
- pump rates in only 5 cesarean cases.

### 8.2 Featured replays
The selection rules are fixed in advance and do not depend on which model wins. Rules 1 and 2 consider only cases with ≥ 8 scored primary forecasts.
1. **Median case.** Among cases with at least one new-onset reading below 80% of the record-start baseline in the primary window, the S-ob case with the median patient-level M4 RMSE. With an even count, take the case with the higher RMSE of the two middle cases. This case is the README GIF.
2. **Worst case.** The S-ob case with the highest patient-level M4 RMSE.
3. **Pump case.** The S-ob pump case with the most `PHEN_RATE` changes in the primary window. This rule was written after the five pump tracks had been seen; it uses no forecast.

Ties go to the lowest case ID. If one case meets two rules, the later rule takes its next case.

### 8.3 Other deliverables
- `docs/index.html`: a static browser for all scored S-ob cases.
- `REPORT.md`: all secondary analyses and every deviation from `PREREG.md`.
- `WHY.md`: one page on why each choice was made.
- A 60–90-s narrated replay of the rule-1 case, in the author's own voice.
- A one-page PDF: the question, the cohort flow, the main table, one figure, the limits and the next experiment.
- One command reproduces everything from the public API.

## 9. Pre-registration and test-set discipline

### Order of work
1. `SPEC.md` and `PREREG.md` are committed together as the first commit of the public history. That commit is made public before any model is fitted.
2. The pipeline is built on S-dev only, and its outputs are checked there. Cuff counts are recomputed under the final rule, and pleth coverage is measured.
3. The feature-extraction code and its parameters are frozen and tagged. They are chosen on signal-level criteria only (for example, R-peak agreement with the monitor HR), never on BP or forecast error.
4. The M2 structure is chosen on S-dev.
5. M3, M4, their placebo versions (M3-P1, M4-P1) and the yardstick models (L, U, T, T-P1) are fitted. **The first complete S-dev cross-validation run of all of these at the frozen feature tag is the confirmatory result. It fixes Sentences 1 and 3 before S-ob is scored.** Any rerun with changed code is a deviation, and both results are shown.
6. All models are frozen and tagged.
7. S-ob is scored **once**. "Once" means one run of the frozen tag. A rerun after a crash, with unchanged code, is allowed and logged. The run scores every frozen model, including the P1 versions and the yardstick models.
8. The remaining controls (P2, ablation, strata), the report and the replays follow.

**No model has been fitted to any case so far.**

### Information already seen during design (S-ob)
- **Case-table metadata** for all 21 cases: age, emergency flag, operation name, subject ID, `opstart`, `anestart`, `intraop_phe`, `intraop_eph`.
- **Cuff timing, reading counts and hypotension-onset counts** for all 21 cases, computed twice: once with windows anchored at the first cuff reading, and once with windows anchored at the recording start.
  - Second run, MAP < 65: 10 cases with an onset (20 onsets). Cases 5306, 3970 and 921 hold 12 of the 20.
  - Second run, below 80% of the record-start baseline: 17 cases (35 onsets).
- **Cuff tracks** inspected for cases 265, 1780, 4147, 5306 and 6187.
- **Pleth window quality, heuristic 1** (beat-template correlation ≥ 0.90): 43%, 47%, 92% and 48% for cases 6187, 1780, 4147 and 307.
- **Pleth, HR and cuff tracks** opened for cases 1780, 6187 and 307 and checked with heuristic 2:
  - usable windows at ≥ 0.85 / 0.90 / 0.95: 1780 88 / 86 / 83%; 6187 96 / 96 / 96%; 307 78 / 73 / 68%;
  - forecast-level waveform coverage: 22/22, 21/21 and 15/15;
  - coverage of a +30 s update (an analysis considered during design and later dropped): 19/22, 21/21 and 15/15;
  - the usable share showed no dip around reading times.
- **Phenylephrine pump tracks** for cases 123, 265, 858, 4343 and 5530: 2–5 rate changes in the first 30 min. Cases 123, 858 and 5530 delivered 13–61 mL with `intraop_phe` = 0.

### Information already seen during design (S-dev)
- Cuff timing and counts for all cases.
- Cuff and HR tracks for case 15.
- The pleth value range for cases 15 and 39.
- Pleth quality (heuristic 1) for cases 249, 5409, 131 and 1711.
- The pump track for case 3574.

### Rules from here on
- **No further access to S-ob signals until the frozen scoring.** That scoring includes the recount under the final cuff rule.
- **Deviations** from `PREREG.md` are listed in `REPORT.md` with the reason and the date. A change made after S-ob scoring is labelled post hoc.

## 10. Reproducibility and compute
- **Software and cost.** Python, CPU only, no paid services; the cost of this analysis is $0.
- **Dependencies** are pinned.
- **Raw data** are cached locally in a git-ignored folder and never committed.
- **Seeds.** 20261118 for folds, bootstrap and placebo donors.
- **Logs.** Every step that sets analysable counts, coverage or exclusions writes a machine-readable log.
- **File hashes.** SHA-256 hashes of `SPEC.md` and `PREREG.md` at registration are listed in `REPORT.md`.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Null result | Pre-written headlines; reported with its controls (§6.3, §7) |
| Low pleth quality in awake cesarean patients | Published BP-blind index plus a stricter sensitivity variant; forecast-level fallback; all forecasts scored; 50% coverage rule |
| Clean-signal forecasts select calm patients | The primary outcome scores every forecast; the waveform subset is secondary |
| Gain from signal loss, not morphology | Ablation with availability indicators only |
| h carries outcome information | Same h in every model; sensitivity analysis with h replaced by the median of previous intervals |
| Hidden non-causal processing | Per-window processing; perturbation leakage test |
| Analyst freedom in feature code | Feature code frozen and tagged before the first M3/M4 fit; first full run is confirmatory |
| Cuff cycling: S-dev 154–168 s vs S-ob 60–90 s | h as an input; slopes from real timestamps; horizon structure chosen by CV; fast-cycling subset; per-bin training base reported |
| S-dev differs from S-ob (older, mostly male, other surgery) | S-ob reported as a case series; no obstetric claim rests on S-dev alone |
| No pre-spinal baseline; injection time unknown | RMSE needs no baseline; three event definitions; record-start label |
| No bolus timing; chart totals miss pump infusions; HR as a drug proxy | Three-way strata; pump strip in mL/h; stated limits |
| Display-scaled pleth; ECG–PPG offset | No absolute amplitude; within-case differences only |
| Repeated identical cuff values are missed | Joint SBP/DBP/MBP change detection; long gaps flagged |
| Test-set information already seen | Listed in §9 and `PREREG.md`; no model fitted before the commit |
| Scope creep | README limited to §8.1; exploratory list (§6.8) may be omitted |

## 12. References
1. Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Sci Data 2022;9:279. doi:10.1038/s41597-022-01411-5
2. Kovacheva VP, Mahesh N, Davoud SC, et al. Autoregressive with exogenous input (ARX) decision support for blood pressure maintenance during cesarean delivery under spinal anesthesia: a prospective pilot study with matched nonconcurrent controls. medRxiv 2026. doi:10.64898/2026.02.03.26345255
3. Davoud SC, Ozaslan B, Aiello EM, Kleinlein R, Eberhard B, Hassan H, Doyle FJ, Kovacheva VP. Empirical pharmacodynamic model of phenylephrine and intrathecal bupivacaine for mean arterial pressure prediction. J Clin Monit Comput 2025;39(4). doi:10.1007/s10877-025-01288-w
4. Lee S, Al-antari MA, Lee G. Lightweight calibration-based blood pressure tracking from photoplethysmography features for smart devices: accuracy, model size and recalibration interval. Diagnostics 2026;16:3713. doi:10.3390/diagnostics16223713
5. Sharwood-Smith G, Bruce J, Drummond G. Br J Anaesth 2006. doi:10.1093/bja/aei266
6. Frassanito L, et al. Anesth Analg 2022. doi:10.1213/ane.0000000000005754
7. Shih PY, et al. Anaesthesia 2026. doi:10.1111/anae.70161
8. Lee S, Lee HC, et al. Br J Anaesth 2021. doi:10.1016/j.bja.2020.12.035
9. Kondo Y, et al. PLoS One 2025. doi:10.1371/journal.pone.0316715
10. Research Square preprint, posted 2026-11-09, not peer-reviewed. doi:10.21203/rs.3.rs-11404280/v1
11. MathWorks. `compare` — System Identification Toolbox documentation. https://www.mathworks.com/help/ident/ref/compare.html
12. Babchenko A, Davidson E, Adler D, Ginosar Y, Kurz V, Nitzan M. Increase in pulse transit time to the foot after epidural anaesthesia treatment. Med Biol Eng Comput 2000;38(6). doi:10.1007/BF02344874 (abstract only)
13. Bolea J, Lázaro J, Gil E, et al. Pulse rate and transit time analysis to predict hypotension events after spinal anesthesia during programmed cesarean labor. Ann Biomed Eng 2017;45(9):2253–2263. doi:10.1007/s10439-017-1864-y (abstract only)
14. Higashi M, Shirozu K, Sumie M, et al. Prevention of spinal anaesthesia-induced hypotension during caesarean section using noninvasive cardiac output monitoring: a randomised clinical trial. Eur J Anaesthesiol Intensive Care 2025. doi:10.1097/EA9.0000000000000086 (abstract only)
