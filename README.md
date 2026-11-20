# between-the-cuffs

**After spinal anesthesia, do heart rate and the pulse-oximeter waveform improve the forecast of the next cuff reading (MAP) beyond the cuff readings alone?**

**Answer: not known yet.** The test was registered on 2026-11-21, before any model was fitted. The result is due by 5 December 2026 and will be posted here whichever way it falls.

[Pre-registered](PREREG.md) · fitted on 203 non-cesarean spinal cases · scored once, frozen, on up to 20 cesarean cases · placebo control · open data ([VitalDB](https://vitaldb.net/))

## Why it matters

After a spinal, a cuff reads blood pressure every 1–5 minutes, and pressure can fall between two readings. The pulse oximeter and ECG record continuously and are already on every patient. If they improve a cuff-based forecast, the gain needs no new sensor. If they do not, the analysis reports the largest gain the data still allow.

## What is locked

- **One comparison.** Cuff + heart rate + pleth waveform features, against a cuff-only third-order autoregressive model, on the error of the next-reading MAP forecast.
- **One rule.** At least 5% lower error, with a 95% CI that excludes zero.
- **One look at the test set.** The models are fitted on the non-cesarean cases, frozen, and scored once on the cesarean cases. No cesarean signal is opened before that.
- **A placebo.** For a positive claim, the same models refitted on another patient's signals must not meet the rule.
- **The wording.** The headline for each of six possible outcomes is already written ([`PREREG.md`](PREREG.md) §6). The data pick one.

## Registration

**No model has been fitted yet.** The registration is version 1.5 of two files:
- [`PREREG.md`](PREREG.md): the hypothesis, primary outcome, decision rule, pre-written headlines and everything seen before registration;
- [`SPEC.md`](SPEC.md): the full design.

Version 1.5 supersedes version 1.4, the first commit of this repository (`fca645e`). Version 1.4 was replaced before any model was fitted. The git history shows every change, and `PREREG.md` §11 lists them.

Commits after the registration add the analysis code, then the results, a replay of one case and a browser of all scored cesarean cases. Any deviation from the registration will be listed in `REPORT.md`.

This is a self-registration by git commit. Commit dates are set by the author; the time of each GitHub release (here `prereg-v1.5`) is recorded by GitHub. There is no archive snapshot. The SHA-256 hashes of both files, for both versions, will be listed in `REPORT.md`.

## Data

Derived from VitalDB (Lee HC, et al. *Sci Data* 2022;9:279, doi:10.1038/s41597-022-01411-5), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **No VitalDB data are redistributed here.** The code downloads them from the public API at run time.
- **Changes:** cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived.
- No attempt was made to identify any person.

*Research prototype, not medical advice.*
