# between-the-cuffs

After spinal anesthesia, blood pressure is measured by a cuff every 1–5 minutes, while the pulse oximeter and ECG record continuously. **Do heart rate and pulse-oximeter waveform features lower the error of a next-cuff-reading MAP forecast beyond a cuff-only third-order autoregressive model?**

This repository answers that question on open data ([VitalDB](https://vitaldb.net/)):
- the models are fitted on non-cesarean spinal cases;
- they are scored once, frozen, on the spinal cesarean cases;
- the result is reported whichever way it falls.

## Status

**Pre-registered; no model has been fitted yet.** This first commit holds the registration:
- [`PREREG.md`](PREREG.md): the hypothesis, primary outcome, decision rule, pre-written headlines and everything seen before registration;
- [`SPEC.md`](SPEC.md): the full design.

The analysis code, results, replay and case browser will follow in later commits. Any deviation from the registration will be listed in `REPORT.md`.

This is a self-registration by git commit. It has no independent third-party timestamp. The SHA-256 hashes of both files at registration will be listed in `REPORT.md`.

## Data

Derived from VitalDB (Lee HC, et al. *Sci Data* 2022;9:279, doi:10.1038/s41597-022-01411-5), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **No VitalDB data are redistributed here.** The code downloads them from the public API at run time.
- **Changes:** cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived.
- No attempt was made to identify any person.

*Research prototype, not medical advice.*
