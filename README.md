# between-the-cuffs

**After spinal anesthesia, do heart rate and the pulse-oximeter waveform improve the forecast of the next cuff reading (MAP) beyond the cuff readings alone?**

**No.** In 200 non-cesarean spinal cases (cross-validated), heart rate and the pleth waveform did not meet the pre-specified rule (−0.1%, from 8.67 to 8.68 mmHg; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 0.2% (0.01 mmHg) are not compatible with these data (approximate 95% CI), for this feature set and a linear model.

**Cesarean cases: not scored yet.** The frozen models (tag `models-frozen`) will be scored once on the analysable cesarean cases (20 at the data check before registration), and the result will be added here. It is a case series and cannot change the answer above. No cesarean signal is opened before that run; what was seen during design is listed in [`PREREG.md`](PREREG.md#9-information-seen-before-this-registration) §9.

[Pre-registered](PREREG.md) on 2026-11-21, before any model was fitted (commit date; public release on 2026-11-22, [times in `REPORT.md`](REPORT.md#registration)) · confirmatory run 2026-11-22 · models frozen 2026-11-23 (KST) · cesarean cases: not scored yet · placebo control · open data ([VitalDB](https://vitaldb.net/))

[Full report](REPORT.md) · [Result files and how to recompute the numbers](results/README.md) · [Design](SPEC.md) · To be added with the cesarean scoring: the summary figure, a 20-s replay of one case and a browser of all scored cases.

**Why it matters.** After a spinal, the cuff reads blood pressure every 1–5 minutes, and pressure can fall between two readings. The pulse oximeter and the ECG record continuously and are already on every patient, so a gain from them would need no new sensor. This test asks how much they improve a forecast of the next cuff reading that already uses the past cuff readings.

**Result in full.**

In 200 non-cesarean spinal cases (cross-validated), heart rate and pleth waveform features did not meet the pre-specified rule for lowering next-reading MAP forecast error beyond a cuff-only third-order model (−0.1%, Δ 0.01 mmHg, 95% CI −0.01 to 0.03; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 0.2% (0.01 mmHg) are not compatible with these data (approximate 95% CI, see §6.2 of SPEC), for this feature set and a linear model.

*Sentence 2 (cesarean cases): pending; it is written by the frozen scoring.*

In the 30 fast-cycling development cases (median cuff interval < 90 s): 0.3% (Δ −0.03 mmHg, 95% CI −0.07 to 0.01); one extra cuff reading's value recovered by the added signals: −0.9% (95% CI −2.2% to −0.1%) (not distinguishable from placebo inputs).

**How to read these numbers** (explanatory, not pre-specified).
- MAP is mean arterial pressure, and the pleth is the pulse-oximeter waveform. Cross-validated means that each patient's forecasts come from models fitted without that patient.
- The error is each patient's root-mean-square forecast error (RMSE), averaged over patients. Δ is that error with heart rate and waveform features minus the error without them. The positive Δ and the negative percentage say the same thing: with the added signals the error was slightly higher.
- 203 cases were analysed. 200 of them have the 3 forecasts needed for an RMSE and are scored.
- The interval is narrow because cross-validation shrank the added coefficients heavily, so the two models gave nearly the same forecasts. The bound describes this fitted procedure and this feature set. It does not measure how much information the signals hold.
- The last number of the third sentence comes from a harder test, in which the latest cuff reading is hidden from the model. The signals recovered none of that reading's value there.

More in [`REPORT.md`, "Reading the result"](REPORT.md#reading-the-result).

## What this means

*The next two paragraphs are taken from text written before the result (`SPEC.md` §7). "The drug proxy": heart rate may stand in for vasopressor doses, whose times these records do not give.*

A null is harder to attribute to the drug proxy, but opposite heart-rate responses to the two vasopressors could also cancel in a linear model. It bounds only this feature set under a linear model; the gradient-boosting check in `REPORT.md` asks the same question without the linear form.

Either way, only data with time-stamped vasopressor doses can settle the question.

The gradient-boosting check has not been run yet. It runs with the other secondary analyses, after the cesarean scoring (`SPEC.md` §9, step 8), and its result will be added to `REPORT.md`.

## Next experiment

On a platform with time-stamped vasopressor doses and 1-minute cuffs, test the increment of the pleth waveform over an ARX model that includes the drug term: retrospectively first, then scored silently and prospectively, with the same placebo and echo controls. The drug-proxy confound is named in advance: heart rate falls after phenylephrine and rises after ephedrine, so only with the doses in the model can a gain from heart rate or the pleth be told apart from the drug. Two measurements this dataset lacks would separate more: the monitor's perfusion-index numeric (does pleth amplitude follow vascular tone, or only the display scaling?) and a recorded probe site (does pulse arrival time cross the blocked territory, and does inflation of a cuff on the same arm shape the pleth?). These are proposals, not results.

*ARX: an autoregressive model with an external input, here the drug. Echo controls: the checks of `SPEC.md` §6.6.*

## Error by model

| Model | Inputs | Non-cesarean, N = 200: mean RMSE (95% CI), mmHg |
|---|---|---:|
| M00 | running mean of the readings so far | 12.57 (11.52 to 13.66) |
| M0 | last reading (persistence) | 9.10 (8.34 to 9.92) |
| M2 | cuff only, third order | 8.67 (7.95 to 9.44) |
| M3 | M2 + heart rate | 8.68 (7.96 to 9.44) |
| M4 | M3 + pleth waveform | 8.68 (7.97 to 9.44) |
| M4-P1 | M4 on another patient's signals (placebo) | 8.67 (7.95 to 9.44) |
| Waveform coverage | share of forecasts with waveform features | 97.5% |

Cross-validated; lower is better. M2, M3, M4 and M4-P1 differ from each other by less than 0.01 mmHg; `REPORT.md` gives each paired difference with its interval. To be added with the cesarean scoring: the cesarean column, the per-patient figure and the line on new-onset hypotension.

## What this does not show

- The analysis is retrospective, with 21 cesarean cases (20 analysable at G0, the data check made before registration), 19 flagged emergency.
- The spinal injection time is unknown, and the baseline is a record-start value, not pre-spinal.
- Bolus times are absent and the models have no drug input, so a heart-rate gain may reflect unrecorded drugs.
- This is a research prototype, not medical advice.

The full list is in [`REPORT.md`](REPORT.md#limits).

## Author's note

The echo and placebo controls ask of a time-series model the question I study in imaging: does the model use the evidence in front of it (here, the pleth), or infer from context (the cuff history)?

## Credit

Derived from VitalDB ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.

Data: VitalDB, Lee HC, et al. Sci Data 2022;9:279, [doi:10.1038/s41597-022-01411-5](https://doi.org/10.1038/s41597-022-01411-5). No VitalDB data are redistributed here; the code downloads them from the public API at run time. Result files that hold cuff values for each forecast are not published; the per-patient tables are ([`results/README.md`](results/README.md)).

Research prototype, not medical advice.

---

Jun-Young Cha · [github.com/starriet9](https://github.com/starriet9)
