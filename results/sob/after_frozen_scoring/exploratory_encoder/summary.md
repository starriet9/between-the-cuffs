<!-- commit dda1d6dafffe9d0a286bd69640f3a9c277b6b784 (models-frozen-10-gdda1d6d); data: VitalDB derived tables (derived/sob_*, derived/sdev_*) and encoder tables (derived/sob_enc_*, derived/sdev_enc_*) -->

# Exploratory encoder check, S-ob (ENCODER_PLAN.md §7)

**Label.** computed after the frozen scoring (ENCODER_PLAN.md §7); appendix only; exploratory; cannot change Sentences 1-3

**One-line summary.** Exploratory, computed after the frozen scoring: in the 19 cesarean cases (a case series), the relative reduction of next-reading MAP forecast error by M4-enc against the cuff-only model was 0.0% (95% CI −0.1% to 0.0%).

**Departures from ENCODER_PLAN.md (S-dev and S-ob steps).**

- S-dev embeddings were built twice: the first build's log recorded the torch settings of the parent process instead of the workers'; the record was fixed (commit a07e0d2) and the embeddings rebuilt before any blood-pressure model. Both builds gave byte-identical arrays (SHA-256 7e1c56b2...d812).
- The S-ob code (this step) was written after the S-dev encoder result was known (ENCODER_PLAN.md §7 fixes what it does and says it runs whatever the S-dev result shows).
- The fast-cycling row of ENCODER_PLAN.md §6 is omitted for S-ob (an S-dev restriction).

**Plan.** ENCODER_PLAN.md, SHA-256 5e1bc2c18362d9ad872321b8174c5aced5017fbe56dc4fcca973ad06501dc95f, unchanged since commit 29db44bb87fb766f190184003920b86486fa7d19.

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
