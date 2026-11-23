"""Exploratory encoder check on S-dev (ENCODER_PLAN.md §3-§6, §9; SPEC.md §6.8). Exploratory: it changes
no registered outcome, rule or sentence.

Model (§3). M4-enc = M3 + an encoder block in the same ridge chain (`ridge.ChainSpec`): the block takes
the place of the waveform block and holds the 1,024 encoder columns (`btc.encoder.ENC_COLS`), their
x h columns (the chosen structure is pooled) and the missing-indicators, all built by `design.Design`
on the training rows. The block has its own penalty from the 13-value grid or "dropped", chosen by
`ridge.select_node` with the folds, seed and tie rule of the waveform block. It requires
`wf_available@<variant>`, so M4-enc is fitted on M4's rows and every other forecast gets the M3
forecast (§4): M4's fallback rows. The M2 and M3 stages are re-derived in the same chain and must equal
the confirmatory forecasts (§3 "Check"). M4-enc-P1 is the same chain on the P1 donor features (§6).

Build check (§5). Ridge, penalty 1 in this project's units (`ridge.ridge_fit`: mean squared error plus
penalty times the sum of squared coefficients), on the 512 window embeddings standardised on the
training windows, predicting `pleth_rate` of the same window, over the usable windows of the analysed
S-dev cases, with the registered outer folds; the out-of-fold R^2 is pooled over all windows, with
SST about the mean of all those windows [judgment]. Threshold 0.5.

Contrasts (§6), from the paired subject bootstrap of SPEC §6.2 on the patients of the primary outcome
(`analyses.Scored`): M4-enc vs M2, vs M3, vs M4; M4-enc-P1 vs M2; M4-enc vs M4-enc-P1; and M4-enc vs M2
on the fast-cycling cases (scoring restricted). M2, M3, M4 and M4-P1 are the stored confirmatory
forecasts. "Meets the rule" = M4-enc vs M2 meets SPEC §6.3 and placebo validity holds
(`headline.placebo_valid`). M4-enc vs M4 is descriptive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import encoder as enc
from . import secondary as sec
from .analyses import Scored, run_cv
from .constants import N_BOOT, SEED
from .data import build_frame
from .design import Block, Cols
from .folds import FoldPlan
from .headline import ci_pct, pct, placebo_valid
from .ridge import ChainSpec, CVResult, hr_block, ridge_fit

ENC_COLS = enc.ENC_COLS
VARIANTS = enc.VARIANTS
REAL_NAMES = ("M2", "M3", "M4-enc")
P1_NAMES = ("M2", "M3-P1", "M4-enc-P1")
CONTRASTS = (("M4-enc", "M2"), ("M4-enc", "M3"), ("M4-enc", "M4"), ("M4-enc-P1", "M2"), ("M4-enc", "M4-enc-P1"))
# The confirmatory primary contrast, recomputed from the stored forecasts: it must equal the confirmatory
# summary, which shows that the patients and bootstrap resamples are those of the primary outcome.
REFERENCE = ("M4", "M2")
STORED_MODELS = ("M2", "M3", "M4", "M4-P1", "M3-P1")
REPORTED_MODELS = ("M2", "M3", "M4", "M4-P1", "M4-enc", "M4-enc-P1")
BUILD_CHECK_PENALTY = 1.0
BUILD_CHECK_MIN_R2 = 0.5
REPRO_ATOL = 1e-8          # re-derived M2 / M3 / M3-P1 against the stored forecasts (§3 "Check") [judgment]
LEAD = ("An exploratory check replaced the hand-crafted pleth features with embeddings from a frozen "
        "pretrained PPG encoder (PaPaGei-S) in the same model.")


# ----------------------------------------------------------------------------
# Chains (§3, §6)
# ----------------------------------------------------------------------------

def enc_block(variant: str = "real", hcol: str = "h_min", cols=ENC_COLS) -> Block:
    """Encoder block of a variant; fitted on M4's rows (`wf_available@<variant>`, §4)."""
    return Block(f"enc@{variant}", tuple(f"{c}@{variant}" for c in cols), f"wf_available@{variant}", hcol)


def enc_ladder(form: str, structure: str, variant: str = "real", names=None, cols=ENC_COLS) -> ChainSpec:
    """M2 -> M3 (heart-rate block) -> M4-enc (encoder block) (§3); P1 donor features with variant p1 (§6)."""
    names = names or (REAL_NAMES if variant == "real" else P1_NAMES)
    return ChainSpec(f"enc:{variant}", tuple(names), form=form, structure=structure,
                     blocks=(hr_block(variant), enc_block(variant, cols=cols)))


def attach_encoder(frame: pd.DataFrame, table: pd.DataFrame, cols=ENC_COLS, window: str = "primary") -> pd.DataFrame:
    """The frame with `<col>@<variant>`, `enc_available@<variant>` and `enc_n_usable@<variant>` joined on
    (caseid, issue_n), in one concat. A forecast without an encoder row has every column missing and
    `enc_available` False. Requires enc_available == wf_available on every forecast (§2, §4)."""
    t = table.loc[table["window"] == window]
    key = pd.MultiIndex.from_arrays([frame["caseid"].to_numpy(), frame["issue_n"].to_numpy()])
    parts = []
    for v in VARIANTS:
        d = t.loc[t["variant"] == v, ["caseid", "issue_n", "enc_available", "enc_n_usable", *cols]]
        d = d.set_index(["caseid", "issue_n"])
        if d.index.has_duplicates:
            raise ValueError("duplicate (caseid, window, issue_n, variant) encoder rows")
        d = d.reindex(key).add_suffix(f"@{v}")
        d.index = frame.index
        parts.append(d)
    out = pd.concat([frame, *parts], axis=1)
    for v in VARIANTS:
        a = out[f"enc_available@{v}"].fillna(False).astype(bool)
        out[f"enc_available@{v}"] = a
        if not np.array_equal(a.to_numpy(), out[f"wf_available@{v}"].to_numpy(bool)):
            raise ValueError(f"enc_available@{v} differs from wf_available@{v} on some forecasts")
    return out


# ----------------------------------------------------------------------------
# Build check (§5)
# ----------------------------------------------------------------------------

def build_check(E: np.ndarray, index: pd.DataFrame, win: pd.DataFrame, plan: FoldPlan) -> dict:
    """Out-of-fold R^2 of ridge (penalty 1) from the 512 window embeddings to the window's pleth rate."""
    m = index[["caseid", "w", "row"]].merge(win[["caseid", "w", "usable", "pleth_rate"]], on=["caseid", "w"],
                                            how="left", validate="1:1")
    if not m["usable"].eq(True).all():
        raise ValueError("an embedded window is not a usable window of the window table")
    m = m.loc[m["caseid"].isin(plan.caseids)].sort_values(["caseid", "w"]).reset_index(drop=True)
    y = m["pleth_rate"].to_numpy(float)
    if not np.all(np.isfinite(y)):
        raise ValueError("a usable window has no pleth rate")
    X = E[m["row"].to_numpy(np.int64)].astype(np.float64)
    fold = plan.outer_of(m["caseid"].to_numpy(int))
    oof = np.full(len(y), np.nan)
    per_fold = []
    for j in range(int(fold.max()) + 1):
        tr, te = fold != j, fold == j
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        b0, beta = ridge_fit((X[tr] - mu) / sd, y[tr], np.full(X.shape[1], BUILD_CHECK_PENALTY))
        oof[te] = b0 + ((X[te] - mu) / sd) @ beta
        yt = y[te]
        per_fold.append({"outer_fold": j, "n_windows": int(te.sum()), "n_cases": int(m.loc[te, "caseid"].nunique()),
                         "r2_within_fold": float(1 - np.sum((yt - oof[te]) ** 2) / np.sum((yt - yt.mean()) ** 2))})
    sse, sst = float(np.sum((y - oof) ** 2)), float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst
    return {"target": "pleth_rate (beats/min, window table)", "penalty": BUILD_CHECK_PENALTY,
            "n_windows": int(len(y)), "n_cases": int(m["caseid"].nunique()), "embedding_dim": int(X.shape[1]),
            "r2_out_of_fold": r2, "threshold": BUILD_CHECK_MIN_R2, "passed": bool(r2 >= BUILD_CHECK_MIN_R2),
            "rmse_out_of_fold_bpm": float(np.sqrt(sse / len(y))), "target_sd_bpm": float(y.std()),
            "per_fold": per_fold}


# ----------------------------------------------------------------------------
# The S-dev run (§6)
# ----------------------------------------------------------------------------

def align_stored(frame: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Stored confirmatory forecasts in frame order, aligned on (caseid, issue_n); targets must agree."""
    s = stored.set_index(["caseid", "issue_n"])
    if s.index.has_duplicates:
        raise ValueError("duplicate forecasts in the stored predictions")
    key = pd.MultiIndex.from_arrays([frame["caseid"].to_numpy(), frame["issue_n"].to_numpy()])
    if len(s) != len(frame) or not key.isin(s.index).all():
        raise ValueError("the stored predictions do not hold exactly the forecasts of the frame")
    s = s.reindex(key)
    for c in ("y_map", "last_map", "t_issue", "t_target"):
        if c in s.columns and not np.array_equal(s[c].to_numpy(float), frame[c].to_numpy(float)):
            raise ValueError(f"stored predictions: {c} differs from the frame")
    return s.reset_index(drop=True)


def reproduction(cv: dict, stored: pd.DataFrame) -> dict:
    """§3 "Check": the re-derived M2 and M3 stages (and M3-P1) against the stored confirmatory forecasts."""
    pairs = {"M2 (enc chain)": (cv["enc"].stage("M2"), "pred_M2"), "M3": (cv["enc"].stage("M3"), "pred_M3"),
             "M2 (enc-P1 chain)": (cv["enc_p1"].stage("M2"), "pred_M2"),
             "M3-P1": (cv["enc_p1"].stage("M3-P1"), "pred_M3-P1")}
    out = {}
    for k, (a, c) in pairs.items():
        b = stored[c].to_numpy(float)
        d = float(np.max(np.abs(a - b)))
        out[k] = {"max_abs_diff_mmHg": d, "bit_identical": bool(np.array_equal(a, b)),
                  "close": bool(np.allclose(a, b, rtol=0.0, atol=REPRO_ATOL))}
    bad = [k for k, v in out.items() if not v["close"]]
    if bad:
        raise RuntimeError(f"re-derived stages differ from the confirmatory forecasts: {bad} ({out}); the frame, "
                           "folds or environment differ from the confirmatory run")
    return out


def dropped_counts(cv: CVResult, model: str) -> dict:
    if any(list(fold) != ["pooled"] for fold in cv.lam):
        raise ValueError("the encoder check uses the pooled structure only")
    s = cv.spec.stages.index(model)
    lam = [fold["pooled"][s] for fold in cv.lam]
    return {"penalty_per_outer_fold": ["dropped" if l is None else l for l in lam],
            "n_folds_dropped": int(sum(l is None for l in lam)), "n_folds": len(lam)}


def one_line(enc_m2, p1_m2, enc_p1, dropped_all: bool) -> str:
    """The one-line summary in the fixed wording of ENCODER_PLAN.md §9 (number formats of SPEC §6.3)."""
    x = f"{pct(enc_m2.rr)}%"
    if not enc_m2.meets_rule:
        s = f"{LEAD} It did not meet the rule either: {x} against the cuff-only model ({ci_pct(enc_m2.rr_lo, enc_m2.rr_hi)})."
        return s + (" The cross-validation dropped the embedding block in every fold." if dropped_all else "")
    if placebo_valid(p1_m2, enc_p1):
        return (f"{LEAD} It met the rule: {x} against the cuff-only model ({ci_pct(enc_m2.rr_lo, enc_m2.rr_hi)}; "
                f"placebo inputs {pct(p1_m2.rr)}%). The encoder was pretrained on unlabelled VitalDB pleth, the "
                "check is exploratory, and the registered answer is unchanged.")
    return f"{LEAD} It met the rule ({x}), but its placebo control failed, so no claim is made from it."


def encoder_sdev(T, table: pd.DataFrame, stored: pd.DataFrame, conf: dict, w, n_jobs: int = 2,
                 n_boot: int = N_BOOT, cols=ENC_COLS, conf_folds: pd.DataFrame | None = None,
                 label: str | None = None, extra: dict | None = None) -> dict:
    """The S-dev run of ENCODER_PLAN.md §6. `table`: the per-forecast encoder table; `stored`: the
    confirmatory `predictions_primary.csv`; `conf`: the confirmatory summary.json; `w`: a Writer for
    `results/sdev/exploratory_encoder/`; `extra`: further entries for summary.json. Returns the summary."""
    from .folds import make_fold_plan
    from .pipeline import _analysed_only, _check_entering, _pred_table
    _analysed_only(T)
    m2 = conf["m2_structure"]
    form, st = m2["form"], m2["structure"]
    if st != "pooled":
        raise RuntimeError("ENCODER_PLAN.md §3 rests on the pooled structure chosen by the confirmatory run")
    frame = attach_encoder(build_frame(T, "primary", 1), table, cols)
    F = Cols(frame)
    plan = make_fold_plan(T.analysed, SEED)
    if conf_folds is not None:
        pt = plan.table()
        if len(conf_folds) != len(pt) or any(c not in conf_folds.columns or not np.array_equal(
                conf_folds[c].to_numpy(), pt[c].to_numpy()) for c in pt.columns):
            raise RuntimeError("the fold plan differs from the confirmatory folds.csv")
    st_al = align_stored(frame, stored)
    cv = run_cv({"enc": (F, enc_ladder(form, st, "real", cols=cols), plan),
                 "enc_p1": (F, enc_ladder(form, st, "p1", cols=cols), plan)}, n_jobs)
    repro = reproduction(cv, st_al)
    avail = frame["wf_available@real"].to_numpy(bool)
    e4, m3 = cv["enc"].stage("M4-enc"), cv["enc"].stage("M3")
    if not np.array_equal(e4[~avail], m3[~avail]):
        raise RuntimeError("an M4-enc fallback forecast differs from the M3 forecast")
    preds = {m: st_al[f"pred_{m}"].to_numpy(float) for m in ("M2", "M3", "M4", "M4-P1")}
    preds["M4-enc"], preds["M4-enc-P1"] = e4, cv["enc_p1"].stage("M4-enc-P1")
    fb = np.abs(preds["M4-enc"][~avail] - preds["M4"][~avail])
    fallback = {"n_forecasts": int((~avail).sum()), "max_abs_diff_from_M4_mmHg": float(fb.max()) if fb.size else 0.0,
                "bit_identical_to_M4": bool(np.array_equal(preds["M4-enc"][~avail], preds["M4"][~avail]))}
    sc = Scored(frame, preds, n_boot=n_boot)
    _check_entering(sc, T)
    if sc.n_patients != conf["N"]:
        raise RuntimeError(f"{sc.n_patients} patients enter, the primary outcome has {conf['N']}")
    con = {(a, b): sc.contrast(a, b) for a, b in (*CONTRASTS, REFERENCE)}
    ref, cref = con[REFERENCE].as_dict(), conf["contrasts"]["M4 vs M2"]
    if any(not np.isclose(ref[k], cref[k], rtol=0, atol=1e-12) for k in ("delta", "delta_lo", "delta_hi", "rr_lo", "rr_hi")):
        raise RuntimeError("the recomputed M4 vs M2 contrast differs from the confirmatory summary")
    fast = frame["fast_cycling"].astype(bool).to_numpy()
    scf = Scored(frame, preds, mask=fast, n_boot=n_boot)
    xf = scf.contrast("M4-enc", "M2")
    e_m2, p_m2, e_p1, e_m4 = (con[("M4-enc", "M2")], con[("M4-enc-P1", "M2")], con[("M4-enc", "M4-enc-P1")],
                              con[("M4-enc", "M4")])
    valid = bool(placebo_valid(p_m2, e_p1))
    meets = bool(e_m2.meets_rule and valid)
    drop_real, drop_p1 = dropped_counts(cv["enc"], "M4-enc"), dropped_counts(cv["enc_p1"], "M4-enc-P1")
    dropped_all = drop_real["n_folds_dropped"] == drop_real["n_folds"]
    line = one_line(e_m2, p_m2, e_p1, dropped_all)
    summ = sc.summary()
    rmse = {r["model"]: {"mean": r["rmse_mean"], "ci_lo": r["rmse_mean_ci_lo"], "ci_hi": r["rmse_mean_ci_hi"]}
            for r in summ.to_dict("records") if r["model"] in REPORTED_MODELS}
    w.csv("predictions_primary.csv", _pred_table(frame, preds, {"outer_fold": cv["enc"].fold}))
    w.csv("per_patient_primary.csv", sc.per_patient_long())
    w.csv("model_summary.csv", summ)
    w.csv("contrasts.csv", pd.DataFrame([{"subset": "all", **c.as_dict()} for c in con.values()] +
                                        [{"subset": "fast_cycling", **xf.as_dict()}]))
    w.csv("penalties.csv", pd.concat([sec.penalty_table(cv["enc"]), sec.penalty_table(cv["enc_p1"])], ignore_index=True))
    w.csv("dropped_blocks_outer_folds.csv",
          pd.concat([sec.dropped_frequency(cv["enc"]), sec.dropped_frequency(cv["enc_p1"])], ignore_index=True))
    summary = {
        "stage": "exploratory encoder check, S-dev outer cross-validation (ENCODER_PLAN.md §6)",
        "label": label or "exploratory (ENCODER_PLAN.md); changes no registered outcome, rule or sentence",
        "m2_structure": {"form": form, "structure": st},
        "N": sc.n_patients, "n_forecasts": sc.n_forecasts, "n_forecasts_all": int(len(frame)),
        "n_forecasts_encoder_available": int(avail.sum()),
        "n_forecasts_encoder_available_p1": int(frame["wf_available@p1"].sum()),
        "encoder_coverage": float(avail.mean()), **T.counts(),
        "rmse_patient_mean": rmse,
        "contrasts": {f"{a} vs {b}": c for (a, b), c in con.items()},
        "reference_contrast_note": "M4 vs M2 is the confirmatory primary contrast, recomputed from the stored "
                                   "forecasts; it equals the confirmatory summary (checked)",
        "fast_cycling": {"K": scf.n_patients, "M4-enc vs M2": xf},
        # placebo validity is checked only for a contrast that meets the rule (SPEC §6.3), so a null is never
        # reported as a placebo failure
        "m4_enc_vs_m2_meets_rule": bool(e_m2.meets_rule),
        "placebo_validity_if_rule_met": valid if e_m2.meets_rule else None, "meets_rule_with_placebo": meets,
        "m4_enc_vs_m4": {"ci_excludes_zero": bool(e_m4.delta_lo > 0 or e_m4.delta_hi < 0),
                         "direction_if_excluded": ("lower error with embeddings" if e_m4.delta_hi < 0 else
                                                   "higher error with embeddings" if e_m4.delta_lo > 0 else None)},
        "encoder_block_outer_folds": {"M4-enc": drop_real, "M4-enc-P1": drop_p1,
                                      "dropped_in_every_fold": bool(dropped_all)},
        "reproduction_of_confirmatory_stages": repro, "fallback": fallback,
        "n_bootstrap_resamples": n_boot, "bootstrap_seed": SEED,
        "one_line_summary": line, **(extra or {}),
    }
    w.json("summary.json", summary)
    return summary
