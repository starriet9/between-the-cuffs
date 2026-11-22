<!-- commit 0669d0c742c102ccf741bd5782517848ae7f377e (feature-freeze); data: VitalDB derived tables (derived/sdev_*.parquet) -->

# Confirmatory S-dev result (SPEC §9 step 5)

**Question.** After spinal anesthesia, do heart rate and the pulse-oximeter waveform improve the forecast of the next cuff reading (MAP) beyond the cuff readings alone?

**Answer line.** No. In 200 non-cesarean spinal cases (cross-validated), heart rate and the pleth waveform did not meet the pre-specified rule (−0.1%, from 8.67 to 8.68 mmHg; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 0.2% (0.01 mmHg) are not compatible with these data (approximate 95% CI), for this feature set and a linear model.

**Sentence 1.** In 200 non-cesarean spinal cases (cross-validated), heart rate and pleth waveform features did not meet the pre-specified rule for lowering next-reading MAP forecast error beyond a cuff-only third-order model (−0.1%, Δ 0.01 mmHg, 95% CI −0.01 to 0.03; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 0.2% (0.01 mmHg) are not compatible with these data (approximate 95% CI, see §6.2 of SPEC), for this feature set and a linear model.

**Sentence 3.** In the 30 fast-cycling development cases (median cuff interval < 90 s): 0.3% (Δ −0.03 mmHg, 95% CI −0.07 to 0.01); one extra cuff reading's value recovered by the added signals: −0.9% (95% CI −2.2% to −0.1%) (not distinguishable from placebo inputs).

Sentence 2 and the cesarean line are written by the frozen S-ob scoring.

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
