<!-- commit 3232c9af96b67d98972dfdc2eec4fff728221e97 (models-frozen-22-g3232c9a); data: VitalDB derived tables (derived/sdev_*.parquet) -->

# Exploratory lead–lag, S-dev (EXPLORATORY_PLAN.md §1)

**Label.** exploratory, computed after the registered results were known (EXPLORATORY_PLAN.md). SPEC.md §6.8 lists this item; its details were fixed in the plan.

Within-case (case-centred, pooled) Pearson correlation of the feature change at reading r with the cuff MAP change, on the primary-window next-reading forecasts of the analysed cases; a case enters with at least 3 usable pairs. 95% percentile intervals from the subject bootstrap of the primary outcome (2000 resamples, seed 20261118, 200 patients). Concurrent: MAP_r − MAP_(r−1). Next: MAP_(r+1) − MAP_r. Next residual: MAP_(r+1) minus the cross-validated M2 forecast. P1: another patient's signals at this patient's reading times.

Expectation (not a claim): pulse arrival time shortens when blood pressure rises, so the concurrent correlation of pulse arrival time is expected to be negative (a stated expectation, not a claim; plan §1.8).

| Feature | MAP change | Real r (95% CI) | Cases / pairs | P1 r (95% CI) | Cases / pairs | Real − P1 (95% CI) |
|---|---|---|---|---|---|---|
| hr_level | concurrent | 0.102 (0.019 to 0.185) | 191 / 1326 | 0.027 (−0.042 to 0.098) | 191 / 1293 | 0.075 (−0.036 to 0.190) |
| hr_level | next | −0.074 (−0.128 to −0.017) | 191 / 1326 | −0.009 (−0.082 to 0.074) | 191 / 1293 | −0.065 (−0.165 to 0.026) |
| hr_level | next_residual_m2 | −0.049 (−0.105 to 0.010) | 191 / 1326 | −0.008 (−0.081 to 0.074) | 191 / 1293 | −0.040 (−0.142 to 0.054) |
| pat | concurrent | −0.071 (−0.144 to 0.003) | 197 / 1350 | 0.006 (−0.054 to 0.065) | 194 / 1347 | −0.077 (−0.161 to 0.006) |
| pat | next | 0.028 (−0.028 to 0.086) | 197 / 1350 | −0.048 (−0.123 to 0.023) | 194 / 1347 | 0.076 (−0.016 to 0.176) |
| pat | next_residual_m2 | 0.010 (−0.045 to 0.066) | 197 / 1350 | −0.052 (−0.128 to 0.018) | 194 / 1347 | 0.062 (−0.029 to 0.166) |
| rise | concurrent | −0.027 (−0.110 to 0.063) | 198 / 1360 | 0.063 (−0.007 to 0.129) | 197 / 1369 | −0.090 (−0.199 to 0.031) |
| rise | next | 0.088 (0.009 to 0.161) | 198 / 1360 | −0.041 (−0.119 to 0.045) | 197 / 1369 | 0.129 (−0.004 to 0.246) |
| rise | next_residual_m2 | 0.083 (0.007 to 0.151) | 198 / 1360 | −0.044 (−0.121 to 0.041) | 197 / 1369 | 0.127 (0.001 to 0.240) |
| width | concurrent | 0.018 (−0.059 to 0.095) | 198 / 1360 | −0.022 (−0.085 to 0.043) | 197 / 1369 | 0.041 (−0.076 to 0.148) |
| width | next | 0.051 (−0.018 to 0.121) | 198 / 1360 | −0.030 (−0.086 to 0.025) | 197 / 1369 | 0.081 (−0.012 to 0.176) |
| width | next_residual_m2 | 0.042 (−0.023 to 0.110) | 198 / 1360 | −0.043 (−0.093 to 0.011) | 197 / 1369 | 0.084 (−0.005 to 0.173) |
| bav | concurrent | 0.075 (0.023 to 0.129) | 198 / 1360 | 0.026 (−0.029 to 0.082) | 197 / 1369 | 0.050 (−0.019 to 0.117) |
| bav | next | −0.014 (−0.064 to 0.033) | 198 / 1360 | 0.022 (−0.041 to 0.083) | 197 / 1369 | −0.036 (−0.116 to 0.045) |
| bav | next_residual_m2 | −0.008 (−0.055 to 0.040) | 198 / 1360 | 0.025 (−0.037 to 0.086) | 197 / 1369 | −0.032 (−0.111 to 0.050) |

Thirty correlations and fifteen differences; unadjusted 95% intervals, descriptive. A correlation is not an increment over the cuff.

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
