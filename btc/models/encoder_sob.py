"""Exploratory encoder check on S-ob, after the frozen scoring (ENCODER_PLAN.md §7; SPEC.md §6.8). Appendix
only; it changes no registered outcome, rule or sentence.

This module was written after the S-dev encoder result was committed (ENCODER_PLAN.md §7 runs "whatever it
shows"). It adds code only: the S-dev code of `btc/models/encoder.py` and `btc/encoder.py` is reused
unchanged.

Refit (§7, SPEC §4.5). M4-enc and M4-enc-P1 are refitted on all analysed S-dev cases with
`analyses.run_refit`, the function the freeze step uses (`pipeline.freeze`): every stage's penalty is chosen
with the outer folds as the CV folds, by `ridge.select_node`. Their M2 and M3 stages must equal those of the
frozen models (results/frozen/models.pkl) parameter for parameter (`frozen_equality`).

Scoring. The analysed S-ob cases are scored once against the frozen M2, M3, M4 and M4-P1 forecasts of the
frozen scoring (results/sob/frozen_scoring/predictions_primary.csv), with S-ob P1 donors (the donor mapping
of sob_cases.parquet, used by the frozen scoring), with the paired subject bootstrap of SPEC §6.2 on the
patients of that scoring. Contrasts: those of ENCODER_PLAN.md §6, plus M4 vs M2 recomputed from the stored
forecasts, which must equal the frozen-scoring summary. The fast-cycling row of §6 is omitted [judgment]:
it is an S-dev restriction (Sentence 3), and in S-ob most cases are fast-cycling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analyses import Scored, run_refit
from .constants import N_BOOT, SEED
from .data import build_frame
from .design import Cols
from .encoder import (CONTRASTS, ENC_COLS, REFERENCE, REPORTED_MODELS, REPRO_ATOL, align_stored, attach_encoder,
                      enc_ladder)
from .headline import ci_pct, pct

SOB_LABEL = ("computed after the frozen scoring (ENCODER_PLAN.md §7); appendix only; exploratory; cannot change "
             "Sentences 1-3")
STAGES = {"enc": ("main", ("M2", "M3")), "enc_p1": ("p1", ("M2", "M3-P1"))}   # refit chain -> frozen chain


def refit_encoder(T_dev, table_dev: pd.DataFrame, form: str, structure: str, n_jobs: int = 2, cols=ENC_COLS) -> dict:
    """M4-enc and M4-enc-P1 refitted on all analysed S-dev cases, penalties chosen with the outer folds."""
    from .folds import make_fold_plan
    from .pipeline import _analysed_only
    _analysed_only(T_dev)
    if structure != "pooled":
        raise RuntimeError("ENCODER_PLAN.md §3 rests on the pooled structure")
    F = Cols(attach_encoder(build_frame(T_dev, "primary", 1), table_dev, cols))
    plan = make_fold_plan(T_dev.analysed, SEED)
    return run_refit({"enc": (F, enc_ladder(form, structure, "real", cols=cols), plan),
                      "enc_p1": (F, enc_ladder(form, structure, "p1", cols=cols), plan)}, n_jobs)


def _arrays_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.shape(a) == np.shape(b) and bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _stage_equal(a, b) -> dict:
    if a.dropped or b.dropped:
        return {"equal": bool(a.dropped and b.dropped), "dropped": [bool(a.dropped), bool(b.dropped)]}
    da, db = a.design, b.design
    parts = {"b0": a.b0 == b.b0, "beta": _arrays_equal(a.beta, b.beta), "scaler_mean": _arrays_equal(da.mean, db.mean),
             "scaler_sd": _arrays_equal(da.sd, db.sd), "columns": list(da.names) == list(db.names),
             "imputation_medians": len(da.med) == len(db.med) and all(_arrays_equal(x, y) for x, y in zip(da.med, db.med)),
             "indicator_columns": list(da.keep) == list(db.keep), "requires": a.requires == b.requires}
    d = float(np.max(np.abs(a.beta - b.beta))) if np.shape(a.beta) == np.shape(b.beta) else None
    return {"equal": bool(all(parts.values())), "parts": {k: bool(v) for k, v in parts.items()},
            "max_abs_coefficient_diff": d, "intercept_diff": float(a.b0 - b.b0)}


def frozen_equality(fits: dict, frozen: dict) -> tuple[dict, list]:
    """The refitted M2 and M3 (M3-P1) stages against the frozen models, parameter for parameter: penalty,
    intercept, coefficients, scaler, imputation medians, columns. Returns (record, unequal stages)."""
    out = {}
    for ours, (theirs, names) in STAGES.items():
        a, b = fits[ours], frozen["chains"][theirs]
        if set(a.nodes) != {"pooled"} or set(b.nodes) != {"pooled"}:
            raise ValueError("pooled structure expected")
        na, nb = a.nodes["pooled"], b.nodes["pooled"]
        for s, name in enumerate(names):
            pa, pb = na.lam[s], nb.lam[s]
            out[f"{name} (refit chain {ours} vs frozen chain {theirs})"] = {
                "penalty": ["dropped" if pa is None else pa, "dropped" if pb is None else pb],
                "penalty_equal": pa == pb, **_stage_equal(na.stages[s], nb.stages[s])}
    bad = [k for k, v in out.items() if not (v["equal"] and v["penalty_equal"])]
    return out, bad


def refit_penalties(fits: dict) -> pd.DataFrame:
    rows = []
    for k, f in fits.items():
        for node, nf in f.nodes.items():
            for s, l in enumerate(nf.lam):
                rows.append({"chain": f.spec.name, "node": node, "model": f.spec.stages[s],
                             "penalty": "dropped" if l is None else l})
    return pd.DataFrame(rows)


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    return {"max_abs_diff_mmHg": float(np.max(np.abs(a - b))) if len(a) else 0.0,
            "bit_identical": bool(np.array_equal(a, b)), "close": bool(np.allclose(a, b, rtol=0.0, atol=REPRO_ATOL))}


def one_line_sob(n: int, x) -> str:
    """One factual sentence (ENCODER_PLAN.md has no fixed S-ob wording) [judgment]."""
    return (f"Exploratory, computed after the frozen scoring: in the {n} cesarean cases (a case series), the "
            f"relative reduction of next-reading MAP forecast error by M4-enc against the cuff-only model was "
            f"{pct(x.rr)}% ({ci_pct(x.rr_lo, x.rr_hi)}).")


def encoder_sob(T_ob, table_ob: pd.DataFrame, stored: pd.DataFrame, fsum: dict, fits: dict, frozen: dict, w,
                n_boot: int = N_BOOT, cols=ENC_COLS, extra: dict | None = None) -> dict:
    """Score the analysed S-ob cases once (see module docstring). `stored`: the frozen scoring's
    predictions_primary.csv; `fsum`: its summary.json; `w`: a Writer for the output folder."""
    from .pipeline import _analysed_only, _check_entering, _pred_table
    _analysed_only(T_ob)
    frame = attach_encoder(build_frame(T_ob, "primary", 1), table_ob, cols)
    F = Cols(frame)
    st_al = align_stored(frame, stored)
    eq, bad = frozen_equality(fits, frozen)
    P, Pp = fits["enc"].predict(F), fits["enc_p1"].predict(F)
    Fz, Fzp = frozen["chains"]["main"].predict(F), frozen["chains"]["p1"].predict(F)
    repro = {"M2 vs stored frozen forecasts": _compare(P[:, 0], st_al["pred_M2"].to_numpy(float)),
             "M3 vs stored frozen forecasts": _compare(P[:, 1], st_al["pred_M3"].to_numpy(float)),
             "M2 (P1 chain) vs stored frozen forecasts": _compare(Pp[:, 0], st_al["pred_M2"].to_numpy(float)),
             "M3-P1 vs stored frozen forecasts": _compare(Pp[:, 1], st_al["pred_M3-P1"].to_numpy(float)),
             "M2 vs frozen model on this frame": _compare(P[:, 0], Fz[:, 0]),
             "M3 vs frozen model on this frame": _compare(P[:, 1], Fz[:, 1]),
             "M3-P1 vs frozen model on this frame": _compare(Pp[:, 1], Fzp[:, 1])}
    if not all(v["close"] for v in repro.values()):
        raise RuntimeError(f"refitted stages differ from the frozen forecasts: {repro}")
    avail = frame["wf_available@real"].to_numpy(bool)
    if not np.array_equal(P[~avail, 2], P[~avail, 1]):
        raise RuntimeError("an M4-enc fallback forecast differs from the M3 forecast")
    preds = {m: st_al[f"pred_{m}"].to_numpy(float) for m in ("M2", "M3", "M4", "M4-P1")}
    preds["M4-enc"], preds["M4-enc-P1"] = P[:, 2], Pp[:, 2]
    fallback = {"n_forecasts": int((~avail).sum()), **_compare(preds["M4-enc"][~avail], preds["M4"][~avail])}
    sc = Scored(frame, preds, n_boot=n_boot)
    _check_entering(sc, T_ob)
    if sc.n_patients != fsum["n"]:
        raise RuntimeError(f"{sc.n_patients} patients enter, the frozen scoring has {fsum['n']}")
    con = {(a, b): sc.contrast(a, b) for a, b in (*CONTRASTS, REFERENCE)}
    ref, cref = con[REFERENCE].as_dict(), fsum["contrasts"]["M4 vs M2"]
    if any(not np.isclose(ref[k], cref[k], rtol=0, atol=1e-12) for k in ("delta", "delta_lo", "delta_hi", "rr_lo", "rr_hi")):
        raise RuntimeError("the recomputed M4 vs M2 contrast differs from the frozen-scoring summary")
    summ = sc.summary()
    rmse = {r["model"]: {"mean": r["rmse_mean"], "ci_lo": r["rmse_mean_ci_lo"], "ci_hi": r["rmse_mean_ci_hi"]}
            for r in summ.to_dict("records") if r["model"] in REPORTED_MODELS}
    pen = refit_penalties(fits)
    e_m2 = con[("M4-enc", "M2")]
    w.csv("predictions_primary.csv", _pred_table(frame, preds))
    w.csv("per_patient_primary.csv", sc.per_patient_long())
    w.csv("model_summary.csv", summ)
    w.csv("contrasts.csv", pd.DataFrame([c.as_dict() for c in con.values()]))
    w.csv("refit_penalties.csv", pen)
    summary = {
        "stage": "exploratory encoder check, S-ob after the frozen scoring (ENCODER_PLAN.md §7)",
        "label": SOB_LABEL,
        "n": sc.n_patients, "n_forecasts": sc.n_forecasts, "n_forecasts_all": int(len(frame)),
        "n_forecasts_encoder_available": int(avail.sum()),
        "n_forecasts_encoder_available_p1": int(frame["wf_available@p1"].sum()),
        "encoder_coverage": float(avail.mean()), **T_ob.counts(),
        "rmse_patient_mean": rmse,
        "contrasts": {f"{a} vs {b}": c for (a, b), c in con.items()},
        "reference_contrast_note": "M4 vs M2 is the frozen-scoring primary contrast, recomputed from the stored "
                                   "forecasts; it equals the frozen-scoring summary (checked)",
        "fast_cycling_row": "omitted (an S-dev restriction; see btc/models/encoder_sob.py)",
        "m4_enc_vs_m4": {"ci_excludes_zero": bool(con[("M4-enc", "M4")].delta_lo > 0 or con[("M4-enc", "M4")].delta_hi < 0)},
        "refit": {"penalties": {f"{r['chain']}/{r['model']}": r["penalty"] for r in pen.to_dict("records")},
                  "encoder_block_dropped": {"M4-enc": fits["enc"].nodes["pooled"].lam[2] is None,
                                            "M4-enc-P1": fits["enc_p1"].nodes["pooled"].lam[2] is None},
                  "frozen_equality": eq, "stages_not_equal_to_frozen": bad},
        "reproduction_of_frozen_forecasts": repro, "fallback": fallback,
        "n_bootstrap_resamples": n_boot, "bootstrap_seed": SEED,
        "one_line_summary": one_line_sob(sc.n_patients, e_m2), **(extra or {}),
    }
    w.json("summary.json", summary)
    return summary
