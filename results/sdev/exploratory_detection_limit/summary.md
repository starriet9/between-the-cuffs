<!-- commit 236df0ccdbf010c60bef5ac92d573af2dd696663 (models-frozen-23-g236df0c); data: VitalDB derived tables (derived/sdev_*.parquet) -->

# Exploratory detection limit, S-dev (EXPLORATORY_PLAN.md §2)

**Label.** exploratory, computed after the registered results were known (EXPLORATORY_PLAN.md). New analysis (post hoc exploratory). It tests the fitting procedure on the real S-dev design, not the physiology.

Planted column: s = y + kappa * sigma_c * z (zero strength: sigma_c * z); y = next MAP - last MAP; sigma_c = root mean square of y over the patient's primary forecasts; z standard normal, one draw per forecast in (case ID, issue reading) order. Oracle: (1 - beta) * M2 change + beta * s on forecasts with waveform features, M2 elsewhere; beta = 1 - (1 - q)^2. Rule: relative reduction >= 5% and upper 95% CI limit of Delta < 0 (SPEC.md §6.3); placebo validity is not evaluated (plan §2.4). 200 patients, 1416 forecasts; noise seed 20261118.

| Nominal q | Oracle reduction (95% CI) | M4+planted vs M2: relative reduction (95% CI) | Δ, mmHg (95% CI) | CI of Δ below 0 | Meets rule | Waveform penalty, folds 0–4 |
|---|---|---|---|---|---|---|
| 0% | 0.00% (0.00% to 0.00%) | −0.03% (−0.43% to 0.35%) | 0.003 (−0.031 to 0.036) | no | no | 3.16, 3.16, 3.16, 3.16, 10 |
| 1% | 1.78% (0.83% to 2.68%) | 1.49% (0.40% to 2.53%) | −0.130 (−0.227 to −0.033) | yes | no | 1, 1, 1, 1, 1 |
| 2% | 3.05% (1.73% to 4.33%) | 2.46% (1.02% to 3.81%) | −0.213 (−0.340 to −0.086) | yes | no | 1, 1, 1, 0.316, 1 |
| 3% | 4.23% (2.65% to 5.80%) | 3.49% (1.77% to 5.12%) | −0.302 (−0.453 to −0.148) | yes | no | 1, 0.316, 0.316, 0.316, 1 |
| 5% | 6.46% (4.50% to 8.38%) | 5.75% (3.64% to 7.81%) | −0.499 (−0.691 to −0.303) | yes | yes | 0.316, 0.316, 0.316, 0.316, 0.316 |
| 10% | 11.61% (9.09% to 14.18%) | 10.59% (8.08% to 13.06%) | −0.918 (−1.162 to −0.678) | yes | yes | 0.316, 0.316, 0.316, 0.316, 0.316 |
| 20% | 21.26% (18.12% to 24.28%) | 19.82% (16.71% to 22.90%) | −1.719 (−2.042 to −1.404) | yes | yes | 0.1, 0.1, 0.1, 0.1, 0.316 |

Ten further noise draws (seeds 20261119–20261128):

| Nominal q | Draws meeting the rule | Draws with CI of Δ below 0 | Relative reduction, min / median / max |
|---|---|---|---|
| 0% | 0 of 10 | 0 of 10 | −0.38% / −0.16% / 0.01% |
| 1% | 0 of 10 | 0 of 10 | −0.14% / 0.14% / 0.74% |
| 2% | 0 of 10 | 1 of 10 | −0.08% / 0.94% / 1.73% |
| 3% | 0 of 10 | 7 of 10 | 0.30% / 1.70% / 3.28% |
| 5% | 1 of 10 | 10 of 10 | 1.30% / 3.69% / 5.68% |
| 10% | 10 of 10 | 10 of 10 | 5.25% / 8.76% / 11.19% |
| 20% | 10 of 10 | 10 of 10 | 14.15% / 18.41% / 21.55% |

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
