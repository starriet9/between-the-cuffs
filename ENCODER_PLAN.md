# Exploratory encoder check: analysis plan

**Status.** Exploratory (`SPEC.md` §6.8, `PREREG.md` §7). This file fixes the details that those two files leave open. It was written on 2026-11-23, after the S-dev confirmatory result (no gain), the gradient-boosting check and the frozen S-ob scoring were known, and before any encoder output was computed. Every choice below rests on the design, on the encoder's public documentation, or on signal-level counts in S-dev. No blood-pressure value and no forecast error was used to make it. The check changes no registered outcome, rule or sentence. `SPEC.md` §6.8 says it is built only if everything else is finished; the web page and the report are being completed alongside it, which is why §10 sets a cut-off.

**Question.** Sentence 1 ends "for this feature set and a linear model". The gradient-boosting check addressed the linear model. This check addresses the feature set: were the hand-crafted pleth features too poor to show a gain that the pleth holds?

## 1. Encoder
- **One encoder: PaPaGei-S** (Pillai et al., ICLR 2025; arXiv:2410.20542). It was pretrained on finger pleth, including intra-operative VitalDB pleth, so it is the public encoder closest to these signals [judgment]. That makes a null informative. It is the variant its authors recommend.
- **No second encoder.** Pulse-PPG (no VitalDB exposure) was considered. It was pretrained on wrist PPG at 50 Hz with 4-minute inputs, and its use on 10-s windows is not documented [unverified]. One encoder fixed in advance gives one answer [judgment]. If M4-enc meets the rule, the report names an encoder without VitalDB exposure as the next check.
- **Weights.** `papagei_s.pt` from Zenodo record 13983110 (doi:10.5281/zenodo.13983110), 23,339,216 bytes. A script downloads it at run time into a git-ignored folder. The published MD5 (`a4cdb32392e2a7b25999128af92813b5`) is checked before loading, and the SHA-256 is recorded. The file is not redistributed here.
- **Model.** The authors' class `ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18, n_classes=512, n_experts=3)` (BSD-3-Clause; repository commit `0c537da`), copied with its licence notice, with unused imports removed. The state dict is loaded strictly on CPU with `weights_only=True`. The model runs in `eval()` mode, without gradients, in float32. No weight is changed and nothing is fine-tuned.
- **Output.** Element 0 of the forward output: the 512-dimensional projected embedding, the one used in the authors' linear probes.
- **Software.** torch 2.15.0, CPU build (`2.15.0+cpu`, Python 3.13), pinned. One thread per process and a fixed batch order.

## 2. Input per forecast
- **Segments.** The same 10-s windows that M4 uses: the usable windows (`SPEC.md` §5) between the previous reading (at most 120 s back) and t_r. There are 2 to 12 per forecast (S-dev median 10). Only the feature extractor changes.
- **Per-segment processing.** Each window is processed from its own 5,000 samples only, in the order of the authors' pretraining code:
  1. at 500 Hz, a Chebyshev type II band-pass (order 4, 20 dB stop-band attenuation, 0.5–12 Hz), run forward and backward, then a 50-ms moving average, run forward and backward. This is the authors' filter, written with SciPy in second-order sections; pyPPG is not used;
  2. z-score: (x − mean) / (SD + 10⁻⁷), population SD;
  3. resampling to 125 Hz with `scipy.signal.resample_poly(x, 1, 4)`, which gives 1,250 samples.
- **No future sample.** Every window used ends at or before t_r, and nothing is filtered or normalised across windows. The perturbation test of `SPEC.md` §4.4 is extended to these features.
- **Aggregation.** The mean of the segment embeddings (the authors also average segments). It enters in the two within-case forms of `SPEC.md` §4.3: (i) level = the current mean minus a reference, the mean over the first 6 usable windows of the record, missing until the 6th ends by t_r; (ii) change = the current mean minus the same mean at the previous reading. This gives 1,024 columns. The absolute embedding is not used, as M4 uses no absolute waveform value [judgment].

## 3. Model
- **M4-enc = M3 + an encoder block.** The block takes the place of the waveform block in the same ridge model (`SPEC.md` §4.5): the 1,024 columns, their × h columns (the chosen M2 structure is pooled), and the missing-indicators. Medians, centring and standardisation are fitted on training rows only.
- **No dimension reduction.** The authors' linear probes use ridge on all 512 standardised dimensions. Ridge already shrinks low-variance directions most, and a fixed number of principal components would add a free choice [judgment].
- **Penalty.** The block has its own penalty, from the same 13-value grid plus "dropped", chosen by the same inner cross-validation, folds, seed and tie rule as the waveform block. In this project's units the authors' probe grid is about 10⁻³ to 1 (10 in their notebook); the grid here reaches 10³, so it covers theirs and much stronger shrinkage.
- **Check.** The M2 and M3 stages are re-derived in the same chain. Their forecasts must equal those of the confirmatory run.

## 4. Missing data and fallback
Identical to M4. With fewer than 2 usable windows, M4-enc returns the M3 forecast and the forecast is flagged; these are exactly M4's fallback forecasts. A missing level or change is set to the training-fold median and gets a missing-indicator. All forecasts are scored.

## 5. Build check (signal-level, before any BP model)
The embeddings must predict the pulse rate of their own window (`pleth_rate` in the window table). Ridge with penalty 1 on the 512 standardised dimensions, over the usable windows of the analysed S-dev cases, with the registered outer folds, must reach an out-of-fold R² of at least 0.5 [judgment]. Below that, the build is checked (class, weights, input) by signal-level means only. The value is reported either way. If it stays below 0.5, the result is reported and labelled "not interpretable".

## 6. Arms and contrasts (S-dev, outer cross-validation)
- **Arms.** M4-enc, and its placebo M4-enc-P1: the same chain on the P1 donors of `SPEC.md` §6.6 (donor heart rate and donor embeddings, the recipient's times, the donor's quality mask). M2, M3, M4 and M4-P1 are the confirmatory forecasts.
- **Contrasts.** M4-enc vs M2 (the form of Sentence 1); M4-enc vs M3 (the waveform increment); M4-enc vs M4 (embeddings against hand-crafted features); M4-enc-P1 vs M2; M4-enc vs M4-enc-P1. Each is given as Δ and relative reduction with 95% CIs from the paired subject bootstrap of `SPEC.md` §6.2 (2,000 resamples, seed 20261118), on the same patients as the primary outcome. M4-enc vs M2 is also scored on the fast-cycling cases (scoring restricted only).
- **"Meets the rule"** means: M4-enc vs M2 has a relative reduction of at least 5% and an upper 95% CI limit of Δ below zero (`SPEC.md` §6.3), and placebo validity holds (M4-enc-P1 vs M2 does not meet the rule, and the 95% CI of RMSE(M4-enc) − RMSE(M4-enc-P1) lies below zero).
- **M4-enc vs M4** is descriptive. The embeddings are said to do better or worse than the hand-crafted features only if the 95% CI of that Δ excludes zero.
- **Standing.** The first complete S-dev run after this file is committed is the result. A rerun with changed code is a departure, and both results are shown. No interval is adjusted for multiplicity and no p-value is reported. The primary outcome, the rule and Sentences 1–3 do not change.

## 7. S-ob
After the S-dev result is committed, and whatever it shows, M4-enc and M4-enc-P1 are refitted on all analysed S-dev cases, with penalties chosen on the outer folds (`SPEC.md` §4.5). The analysed S-ob cases are scored once, with S-ob donors for the placebo, against the frozen M2, M3 and M4 forecasts. S-ob pleth is opened for this step only. The result goes to the appendix only and is labelled "computed after the frozen scoring".

## 8. Disclosures (always printed with the result)
- **Pretraining overlap.** PaPaGei was pretrained on unlabelled `SNUADC/PLETH` of 5,866 VitalDB participants. The case list is not published, so any case here, in S-dev or S-ob, may have been seen. No blood-pressure or clinical label was used in pretraining. The overlap can only help the encoder on these recordings: it could inflate a gain, and it does not explain a null [judgment].
- **Licence.** The code is BSD-3-Clause. The Zenodo record states no licence for the weights, so they are downloaded by the user and not re-hosted.
- **Differences from the authors' examples.** The filter runs on each 10-s window, not on the whole recording, so that no later sample can enter; this adds edge effects. Segments are selected by this project's quality index, not by the authors' flat-line and morphology filters. Embeddings enter as within-case differences, not as per-subject averages. The authors' own examples differ in the order of steps; the pretraining order is followed.
- **Timing.** This plan was fixed after the registered results were known.

## 9. Reporting
- **`REPORT.md`, appendix, whatever the result:** this file's commit and SHA-256; the build check; the numbers of windows and forecasts; the patient-level mean RMSE of M2, M3, M4, M4-enc and M4-enc-P1 with 95% CIs; every contrast of §6; the penalty chosen in each outer fold and how often the encoder block was dropped; the S-ob table; the disclosures of §8; every departure from this plan with its reason.
- **README.** No change if the check agrees with Sentence 1. If it differs (rule met against not met), block 2 gets one line: "An exploratory pretrained-encoder check on the pleth gave a different answer (X%, 95% CI …); see `REPORT.md`." `SPEC.md` §8.1 has no such line, so it is listed as a deviation.
- **One-line summary, fixed wording.** X is the relative reduction of M4-enc vs M2 with its 95% CI, and P that of M4-enc-P1 vs M2.
  - Rule not met: "An exploratory check replaced the hand-crafted pleth features with embeddings from a frozen pretrained PPG encoder (PaPaGei-S) in the same model. It did not meet the rule either: X% against the cuff-only model (95% CI …)." If the block was dropped in all five outer folds, add: "The cross-validation dropped the embedding block in every fold."
  - Rule met: "An exploratory check replaced the hand-crafted pleth features with embeddings from a frozen pretrained PPG encoder (PaPaGei-S) in the same model. It met the rule: X% against the cuff-only model (95% CI …; placebo inputs P%). The encoder was pretrained on unlabelled VitalDB pleth, the check is exploratory, and the registered answer is unchanged."
  - Rule met on M4-enc vs M2 but placebo validity failed: "… met the rule (X%), but its placebo control failed, so no claim is made from it."

## 10. Stop rule
The cut-off is 2026-11-24 12:00 KST. It is applied by the clock and never by the numbers. If no S-dev contrast has been computed by then, the analysis is dropped and `REPORT.md` says "optional, not run". If the S-dev contrasts exist but their review is not finished, they are listed in the appendix as "computed, not reviewed" and nothing is concluded from them. If S-ob has not been scored by then, the appendix says so.

## 11. Record
- This file is `ENCODER_PLAN.md` in the repository root, beside `PREREG.md`. Its commit precedes every encoder output. The run scripts refuse to start unless it is committed, and they record its SHA-256.
- Results: `results/sdev/exploratory_encoder/` and `results/sob/after_frozen_scoring/exploratory_encoder/`, each with a `MANIFEST.json`. Per-patient tables are published. Embeddings and per-forecast files are derived from VitalDB signals and are not published.
