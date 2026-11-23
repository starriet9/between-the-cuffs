<!-- commit d66ad631230a618a66f70b26e467b73c635840b4 (models-frozen-8-gd66ad63); data: VitalDB derived tables (derived/sdev_*.parquet) and encoder tables (derived/sdev_enc_*) -->

# Exploratory encoder check, S-dev (ENCODER_PLAN.md)

**One-line summary (fixed wording, ENCODER_PLAN.md §9).** An exploratory check replaced the hand-crafted pleth features with embeddings from a frozen pretrained PPG encoder (PaPaGei-S) in the same model. It did not meet the rule either: −0.1% against the cuff-only model (95% CI −0.3% to 0.0%).

**Build check (§5).** Out-of-fold R² 0.6871 (threshold 0.5) over 61341 usable windows of 203 analysed cases.

**Plan.** ENCODER_PLAN.md, SHA-256 5e1bc2c18362d9ad872321b8174c5aced5017fbe56dc4fcca973ad06501dc95f, last changed in commit 29db44bb87fb766f190184003920b86486fa7d19; public on origin/main at 2026-11-23T03:54:36Z (commit 29db44bb87fb766f190184003920b86486fa7d19).

The disclosures of ENCODER_PLAN.md §8, printed with the result:

## 8. Disclosures (always printed with the result)
- **Pretraining overlap.** PaPaGei was pretrained on unlabelled `SNUADC/PLETH` of 5,866 VitalDB participants. The case list is not published, so any case here, in S-dev or S-ob, may have been seen. No blood-pressure or clinical label was used in pretraining. The overlap can only help the encoder on these recordings: it could inflate a gain, and it does not explain a null [judgment].
- **Licence.** The code is BSD-3-Clause. The Zenodo record states no licence for the weights, so they are downloaded by the user and not re-hosted.
- **Differences from the authors' examples.** The filter runs on each 10-s window, not on the whole recording, so that no later sample can enter; this adds edge effects. Segments are selected by this project's quality index, not by the authors' flat-line and morphology filters. Embeddings enter as within-case differences, not as per-subject averages. The authors' own examples differ in the order of steps; the pretraining order is followed.
- **Timing.** This plan was fixed after the registered results were known.

---

Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; signals resampled, filtered and segmented; features computed; plots and summary statistics derived. No attempt was made to identify any person.
Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
