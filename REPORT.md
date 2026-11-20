# between-the-cuffs: report

This file lists the implementation choices and any deviations from the registration (`PREREG.md` and `SPEC.md`, version 1.5). The results follow in later sections, after the confirmatory S-dev run and the frozen S-ob scoring.

## Registration

| Version | Commit | Date | `PREREG.md` SHA-256 | `SPEC.md` SHA-256 |
|---|---|---|---|---|
| 1.4 | `fca645e` | 2026-11-19 | `7fc6684469a8275eae37f2f3e5178ee267a00d13e040962030e1322c79a00c60` | `91ec51922d62402076bb1d219405b2bb5820918ee3a41fb0f805e18b6309bb29` |
| 1.5 | `59939b7` | 2026-11-21 | `1a866d6a991bbacc4827da3b3bdaa47b4c483e76a41a523530525aedc5eee7d8` | `ccfbb764f836f55a68ac6cc0f352722619d66b9ff28154b1012526d688a78867` |

Version 1.5 replaced version 1.4 before any model was fitted; `PREREG.md` §11 lists the changes. This is a self-registration by git commit, and commit dates are set by the author. The `published_at` time of each GitHub release (`prereg-v1.5`, `feature-freeze`, and later the frozen models) is set by GitHub's server; it can be read on the repository's Releases page or from the GitHub releases API, in UTC. Dates in this repository are Korea Standard Time (UTC+9). The version 1.5 text was first committed locally on 2026-11-20 and amended on 2026-11-21, before its first push, to add the neuromuscular-blocker rule and the reviewer-guide clause; only the pushed commit is the registration. No archive snapshot was made.

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

None so far.

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
