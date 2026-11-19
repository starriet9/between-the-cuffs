"""Confirmatory S-dev run, S-dev secondary analyses, freeze, frozen S-ob scoring and the analyses
after it (SPEC §9; PREREG §4, §6, §7, §9).

Stages and output folders (under the output root):
- `sdev/confirmatory/`  §9 steps 4-5: M2 structure, M2-M4, M3-P1, M4-P1, yardstick L/U/T/T-P1;
                        Sentences 1 and 3 and the plain answer line.
- `sdev/secondary/`     §6.6-§6.8 on S-dev (refitted variants, restricted scoring, gradient boosting).
- `frozen/`             §9 step 6: every confirmatory model refitted on all S-dev and pickled.
- `sob/frozen_scoring/` §9 step 7: one scoring run of the frozen models on S-ob; Sentence 2 and the
                        cesarean line.
- `sob/after_frozen_scoring/` §9 step 8: everything else on S-ob, labelled as computed after the
                        frozen scoring; models fitted on S-dev alone.
- `<cases-dir>`         §9 step 8: the per-case files of the case browser, the summary figure and the
                        replays (btc/site/series.py; `site_cases`), from the frozen models without a
                        refit. They hold VitalDB-derived series, so they go to a git-ignored folder
                        (default `data/derived/site/`), never to `results/`.

Every analysis fits and scores the analysed cases only (`Tables.analysed`; SPEC §2.2: analysable
cases without a recorded neuromuscular blocker). The one exception is the sensitivity analysis
"including cases with a recorded neuromuscular blocker" (S-dev secondary, refitted with its own folds;
S-ob, the frozen models scored on all analysable cases in the same frozen scoring run).
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import secondary as sec
from .analyses import (M2_COMBOS, Scored, ablation_chain, baseline_preds, choose_m2, ladder, m2_chain,
                       run_cv, run_refit, yard_T_chain, yard_U_chain, yardstick_scores)
from .constants import (ABLATION_GROUPS, FOLD_SEEDS_SENSITIVITY, N_BOOT, P2_MIN_ISSUE_S, SEED)
from .data import Tables, build_frame, build_triples
from .design import Cols
from .folds import make_fold_plan
from .gbm import GB_ARMS, cv_gb, predict_gb, refit_gb
from .headline import (HeadlineInputs, answer_line, cesarean_line, coverage_rule_applies, gb_line,
                       placebo_valid, question, select_row, sentence1, sentence2, sentence3)
from .outputs import Writer

MAIN_CONTRASTS = (("M4", "M2"), ("M3", "M2"), ("M4", "M3"), ("M4-P1", "M2"), ("M3-P1", "M2"),
                  ("M4", "M4-P1"), ("M3", "M3-P1"), ("M2", "M0"), ("M2", "M00"), ("M2", "M1"))
README_MODELS = ("M00", "M0", "M2", "M3", "M4", "M4-P1")      # SPEC §8.1 block 5
NMB_ANALYSIS = "including cases with a recorded neuromuscular blocker"   # SPEC §2.2 sensitivity analysis
NMB_DIR = "sensitivity_including_nmb"
LADDER_PAIRS = (("M4", "M2"), ("M3", "M2"), ("M4", "M3"))


def _analysed_only(*tables: Tables) -> None:
    """Confirmatory, secondary and frozen analyses fit and score the analysed cases (SPEC §2.2); only
    the sensitivity analyses set `include_nmb`, on a copy of the tables."""
    if any(t.include_nmb for t in tables):
        raise ValueError("tables with include_nmb=True are for the SPEC §2.2 sensitivity analysis only")


def _check_entering(sc: Scored, T: Tables) -> None:
    """Every patient entering the scoring is an analysed case of `T` (SPEC §2.2)."""
    extra = set(sc.enter.tolist()) - set(T.analysed["caseid"].astype(int))
    if extra:
        raise RuntimeError(f"cases outside the analysed set entered the scoring: {sorted(extra)[:5]}")


def other(structure: str) -> str:
    return "pooled" if structure == "perbin" else "perbin"


@dataclass
class Context:
    T: Tables
    frame: pd.DataFrame
    F: Cols
    plan: object
    form: str
    structure: str
    cv: dict = field(default_factory=dict)
    preds: dict = field(default_factory=dict)
    tri: pd.DataFrame | None = None
    u: pd.DataFrame | None = None
    ypreds: dict = field(default_factory=dict)
    row: int | None = None


def _pred_table(frame, preds, extra=None) -> pd.DataFrame:
    keys = ["caseid", "subjectid", "issue_n", "t_issue", "t_target", "h_s", "h_bin", "y_map", "last_map",
            "wf_available@real", "fast_cycling"]
    d = frame[[k for k in keys if k in frame.columns]].copy()
    for k, v in (extra or {}).items():
        d[k] = v
    for k, v in preds.items():
        d[f"pred_{k}"] = v
    return d


def _headline_inputs(sc: Scored) -> tuple:
    con = {(a, b): sc.contrast(a, b) for a, b in MAIN_CONTRASTS}
    hi = HeadlineInputs(N=sc.n_patients, m4_m2=con[("M4", "M2")], m3_m2=con[("M3", "M2")],
                        m4_m3=con[("M4", "M3")], m4p1_m2=con[("M4-P1", "M2")], m4_m4p1=con[("M4", "M4-P1")],
                        m3p1_m2=con[("M3-P1", "M2")], m3_m3p1=con[("M3", "M3-P1")])
    return hi, con


def _confirmatory_jobs(F, Ft, Fu, form, structure, plan) -> dict:
    return {"main": (F, ladder(form, structure), plan),
            "p1": (F, ladder(form, structure, "p1", names=("M2", "M3-P1", "M4-P1")), plan),
            "yard_T": (Ft, yard_T_chain(form), plan),
            "yard_TP1": (Ft, yard_T_chain(form, "p1", names=("L", "LH-P1", "T-P1")), plan),
            "yard_U": (Fu, yard_U_chain(form), plan)}


def _ypreds(cv) -> dict:
    return {"L": cv["yard_T"].stage("L"), "LH": cv["yard_T"].stage("LH"), "T": cv["yard_T"].stage("T"),
            "LH-P1": cv["yard_TP1"].stage("LH-P1"), "T-P1": cv["yard_TP1"].stage("T-P1"),
            "U": cv["yard_U"].stage("U")}


# ----------------------------------------------------------------------------
# §9 steps 4-5: the confirmatory S-dev run
# ----------------------------------------------------------------------------

def confirmatory(T: Tables, w: Writer, n_jobs: int = 1, n_boot: int = N_BOOT) -> Context:
    _analysed_only(T)
    frame = build_frame(T, "primary", 1)
    F = Cols(frame)
    plan = make_fold_plan(T.analysed, SEED)
    w.csv("confirmatory/folds.csv", plan.table())
    # step 4: M2 structure (§4.1)
    m2cv = run_cv({c: (F, m2_chain(*c), plan) for c in M2_COMBOS}, n_jobs)
    (form, structure), m2tab = choose_m2(frame, m2cv)
    w.csv("confirmatory/m2_structure.csv", m2tab)
    # step 5: M3, M4, M3-P1, M4-P1, L, U, T, T-P1
    tri, u = build_triples(T)
    Ft, Fu = Cols(tri), Cols(u)
    cv = run_cv(_confirmatory_jobs(F, Ft, Fu, form, structure, plan), n_jobs)
    if not np.allclose(cv["main"].stage("M2"), m2cv[(form, structure)].stage("M2")):
        raise RuntimeError("M2 differs between the structure choice and the ladder")
    preds = {**baseline_preds(frame), "M2": cv["main"].stage("M2"), "M3": cv["main"].stage("M3"),
             "M4": cv["main"].stage("M4"), "M3-P1": cv["p1"].stage("M3-P1"), "M4-P1": cv["p1"].stage("M4-P1")}
    sc = Scored(frame, preds, n_boot=n_boot)
    _check_entering(sc, T)                       # N = analysed patients entering the primary outcome
    hi, con = _headline_inputs(sc)
    sel = select_row(hi)
    s1, ans = sentence1(hi, sel), answer_line(hi, sel)
    # Sentence 3: fast-cycling cases, scoring restricted (§6.3, §6.7)
    fast = frame["fast_cycling"].astype(bool).to_numpy()
    scf = Scored(frame, preds, mask=fast, n_boot=n_boot)
    xf = scf.contrast("M4", "M2")
    yp = _ypreds(cv)
    y_all = yardstick_scores(tri, yp, n_boot=n_boot)
    y_fast = yardstick_scores(tri, yp, mask=tri["fast_cycling"].astype(bool).to_numpy(), n_boot=n_boot)
    s3 = sentence3(scf.n_patients, xf, y_fast if y_fast.get("n_patients") else None,
                   y_fast.get("placebo_guard_passed"))
    coverage = float(frame["wf_available@real"].mean())
    # outputs
    w.csv("confirmatory/predictions_primary.csv", _pred_table(frame, preds, {"outer_fold": cv["main"].fold}))
    w.csv("confirmatory/per_patient_primary.csv", sc.per_patient_long())
    w.csv("confirmatory/model_summary.csv", sc.summary())
    w.csv("confirmatory/contrasts.csv", pd.DataFrame([c.as_dict() for c in con.values()]))
    w.csv("confirmatory/contrasts_fast_cycling.csv",
          pd.DataFrame(sec.contrast_rows(scf, [p for p in MAIN_CONTRASTS], subset="fast_cycling")))
    w.csv("confirmatory/penalties.csv", pd.concat([sec.penalty_table(v) for v in cv.values()], ignore_index=True))
    w.csv("confirmatory/yardstick_predictions.csv",
          _pred_table(tri, yp, {"outer_fold": cv["yard_T"].fold, "t_r1": tri["t_r1"]}))
    summary = {
        "stage": "confirmatory S-dev cross-validation (SPEC §9 steps 4-5)",
        "m2_structure": {"form": form, "structure": structure, "table": m2tab,
                         "bins_fitted_on_their_own": list(cv["main"].bins)},
        "N": sc.n_patients, "n_forecasts": sc.n_forecasts, **T.counts(),
        "waveform_coverage": coverage,
        "contrasts": {f"{a} vs {b}": c for (a, b), c in con.items()},
        "decision": sel, "headline_row": sel["row"],
        "fast_cycling": {"K": scf.n_patients, "m4_vs_m2": xf},
        "yardstick_all": y_all, "yardstick_fast_cycling": y_fast,
        "n_bootstrap_resamples": n_boot, "bootstrap_seed": SEED,
        "question": question(), "answer_line": ans, "sentence_1": s1, "sentence_3": s3,
    }
    w.json("confirmatory/summary.json", summary)
    w.text("confirmatory/headline.md",
           "# Confirmatory S-dev result (SPEC §9 step 5)\n\n"
           f"**Question.** {question()}\n\n**Answer line.** {ans}\n\n**Sentence 1.** {s1}\n\n"
           f"**Sentence 3.** {s3}\n\n"
           "Sentence 2 and the cesarean line are written by the frozen S-ob scoring.\n")
    return Context(T=T, frame=frame, F=F, plan=plan, form=form, structure=structure, cv=cv, preds=preds,
                   tri=tri, u=u, ypreds=yp, row=sel["row"])


# ----------------------------------------------------------------------------
# S-dev secondary analyses (§6.6-§6.8)
# ----------------------------------------------------------------------------

def secondary_sdev(ctx: Context, w: Writer, n_jobs: int = 1, n_boot: int = N_BOOT, with_gb: bool = True) -> dict:
    T, frame, F, plan, form, st = ctx.T, ctx.frame, ctx.F, ctx.plan, ctx.form, ctx.structure
    _analysed_only(T)
    preds = dict(ctx.preds)
    f2 = build_frame(T, "primary", 2)
    fsec = build_frame(T, "secondary", 1)
    F2, Fsec, Fsbp = Cols(f2), Cols(fsec), Cols(frame, "sbp")
    Ft = Cols(ctx.tri)
    plans = {s: make_fold_plan(T.analysed, s) for s in FOLD_SEEDS_SENSITIVITY}
    # SPEC §2.2 sensitivity: the same ladder on all analysable cases, with its own folds (seed 20261118)
    Tn = replace(T, include_nmb=True)
    fnmb = build_frame(Tn, "primary", 1)
    plan_nmb = make_fold_plan(Tn.analysed, SEED)
    jobs = {
        "other_structure": (F, ladder(form, other(st)), plan),
        "p2": (F, ladder(form, st, "p2", names=("M2", "M3-P2", "M4-P2")), plan),
        "strict": (F, ladder(form, st, "real", wf_variant="strict", names=("M2", "M3", "M4-strict")), plan),
        "hmed": (F, ladder(form, st, hcol="hmed_min", bincol="hmed_bin"), plan),
        "two_ahead": (F2, ladder(form, st), plan),
        "secondary_window": (Fsec, ladder(form, st), plan),
        "sbp": (Fsbp, ladder(form, st, target="sbp"), plan),
        "sbp_p1": (Fsbp, ladder(form, st, "p1", names=("M2", "M3-P1", "M4-P1"), target="sbp"), plan),
        "yard_Tia": (Ft, yard_T_chain(form, "ia", names=("L", "LH-ia", "T-ia")), plan),
        "include_nmb": (Cols(fnmb), ladder(form, st), plan_nmb),
        **{f"ablation_{g}": (F, ablation_chain(form, st, g), plan) for g in ABLATION_GROUPS},
        **{f"seed_{s}": (F, ladder(form, st), p) for s, p in plans.items()},
    }
    cv = run_cv(jobs, n_jobs)
    out = {}
    # h-bins, waveform subset, fast cycling, sex, quality strata (scoring restricted)
    masks = {"all": np.ones(len(frame), dtype=bool), **sec.bin_masks(frame),
             "waveform available": frame["wf_available@real"].to_numpy(),
             "fast_cycling": frame["fast_cycling"].astype(bool).to_numpy(), **sec.sex_masks(frame),
             **{f"quality {k}": v for k, v in sec.quality_masks(frame).items()},
             **({"cerclage": frame["cerclage"].astype(bool).to_numpy()} if "cerclage" in frame else {})}
    s, c = sec.restricted(frame, preds, masks)
    w.csv("secondary/restricted_summary.csv", s)
    w.csv("secondary/restricted_contrasts.csv", c)
    w.csv("secondary/training_base_per_bin.csv", sec.training_base_per_bin(frame))
    # refitted variants
    var_rows, var_sum = [], []

    def add(name, fr, pr, pairs, mask=None, target="map"):
        sc = Scored(fr, pr, mask=mask, target=target, n_boot=n_boot)
        var_sum.append(sec.summary_rows(sc, analysis=name))
        var_rows.extend(sec.contrast_rows(sc, pairs, analysis=name))

    o = cv["other_structure"]
    add(f"horizon structure {other(st)}", frame,
        {"M2": o.stage("M2"), "M3": o.stage("M3"), "M4": o.stage("M4")}, [("M4", "M2"), ("M3", "M2"), ("M4", "M3")])
    p2 = cv["p2"]
    late = frame["t_issue"].to_numpy() > P2_MIN_ISSUE_S
    add("P2 (issued after 12 min)", frame, {"M2": preds["M2"], "M4": preds["M4"], "M3-P2": p2.stage("M3-P2"),
                                            "M4-P2": p2.stage("M4-P2")},
        [("M4", "M2"), ("M4-P2", "M2"), ("M4", "M4-P2")], mask=late)
    sx = cv["strict"]
    add("template-correlation quality (strict)", frame, {"M2": preds["M2"], "M4": preds["M4"],
                                                         "M4-strict": sx.stage("M4-strict")},
        [("M4-strict", "M2"), ("M4-strict", "M4")])
    hm = cv["hmed"]
    add("h replaced by median previous interval", frame,
        {"M2": hm.stage("M2"), "M3": hm.stage("M3"), "M4": hm.stage("M4")}, [("M4", "M2"), ("M3", "M2"), ("M4", "M3")])
    ta = cv["two_ahead"]
    add("2-reading-ahead targets", f2, {"M2": ta.stage("M2"), "M3": ta.stage("M3"), "M4": ta.stage("M4")},
        [("M4", "M2"), ("M3", "M2"), ("M4", "M3")])
    sw = cv["secondary_window"]
    add("secondary window", fsec, {"M2": sw.stage("M2"), "M3": sw.stage("M3"), "M4": sw.stage("M4")},
        [("M4", "M2"), ("M3", "M2"), ("M4", "M3")])
    sb, sbp1 = cv["sbp"], cv["sbp_p1"]
    sbp_preds = {"M2": sb.stage("M2"), "M3": sb.stage("M3"), "M4": sb.stage("M4"), "M4-P1": sbp1.stage("M4-P1")}
    add("SBP models", frame, sbp_preds, [("M4", "M2"), ("M3", "M2"), ("M4", "M3"), ("M4-P1", "M2")], target="sbp")
    abl = {"M2": preds["M2"]}
    for g in ABLATION_GROUPS:
        abl[f"M2+{g}"] = cv[f"ablation_{g}"].stage(f"M2+{g}")
    add("feature-group ablation", frame, abl, [(f"M2+{g}", "M2") for g in ABLATION_GROUPS])
    nm = cv["include_nmb"]
    add(NMB_ANALYSIS, fnmb, {"M2": nm.stage("M2"), "M3": nm.stage("M3"), "M4": nm.stage("M4")}, LADDER_PAIRS)
    w.csv(f"secondary/{NMB_DIR}_folds.csv", plan_nmb.table())
    out["including_nmb"] = {"analysis": NMB_ANALYSIS, **Tn.counts(), "n_forecasts": int(len(fnmb)),
                            "fold_seed": SEED, "form": form, "structure": st}
    w.csv("secondary/variants_summary.csv", pd.concat(var_sum, ignore_index=True))
    w.csv("secondary/variants_contrasts.csv", pd.DataFrame(var_rows))
    # fold-seed sensitivity (§6.7): point estimates; seed 20261118 stays confirmatory
    seed_rows = []
    for s in FOLD_SEEDS_SENSITIVITY:
        r = cv[f"seed_{s}"]
        sc = Scored(frame, {"M2": r.stage("M2"), "M4": r.stage("M4")}, n_boot=1)
        a, b = np.mean(sc.rmse("M4")), np.mean(sc.rmse("M2"))
        seed_rows.append({"seed": s, "rr_m4_vs_m2": 1 - a / b, "delta_m4_minus_m2": a - b})
    seeds = pd.DataFrame(seed_rows)
    w.csv("secondary/fold_seed_sensitivity.csv", seeds)
    out["fold_seed_range"] = {"rr_min": seeds["rr_m4_vs_m2"].min(), "rr_max": seeds["rr_m4_vs_m2"].max(),
                              "delta_min": seeds["delta_m4_minus_m2"].min(), "delta_max": seeds["delta_m4_minus_m2"].max()}
    # yardstick: inflation-artefact variant; plain 2-reading-ahead M2 beside L
    yp = dict(ctx.ypreds)
    yp["T-ia"] = cv["yard_Tia"].stage("T-ia")
    yp["M2 2-ahead"] = ta.stage("M2")         # k = 2 instances are exactly the triples, same order
    if not np.array_equal(f2["caseid"].to_numpy(), ctx.tri["caseid"].to_numpy()):
        raise RuntimeError("2-ahead rows and triples are not aligned")
    out["yardstick_all"] = yardstick_scores(ctx.tri, yp, n_boot=n_boot)
    out["yardstick_fast_cycling"] = yardstick_scores(ctx.tri, yp, mask=ctx.tri["fast_cycling"].astype(bool).to_numpy(),
                                                     n_boot=n_boot)
    # hypotension, large changes, intervals (§6.7)
    hyp = sec.hypotension(frame, {k: preds[k] for k in ("M2", "M3", "M4", "M4-P1")}, sbp_preds)
    w.csv("secondary/hypotension.csv", hyp)
    out["hypotension_line_sdev"] = sec.hypotension_line({"S-dev": hyp})
    w.csv("secondary/large_changes.csv", sec.large_changes(frame, preds))
    w.csv("secondary/conformal.csv", sec.conformal_sdev(frame, preds, ctx.cv["main"].fold))
    # coefficients, penalties, dropped blocks (§6.7)
    fits = run_refit({"main": (F, ladder(form, st), plan)}, n_jobs)
    w.csv("secondary/coefficients_all_sdev.csv", sec.coefficient_table(fits["main"]))
    w.csv("secondary/dropped_blocks_outer_folds.csv",
          pd.concat([sec.dropped_frequency(ctx.cv["main"]), sec.dropped_frequency(ctx.cv["p1"])], ignore_index=True))
    w.csv("secondary/penalties.csv", pd.concat([sec.penalty_table(v) for v in cv.values()], ignore_index=True))
    out["data_quality"] = sec.data_quality(T, frame)
    # gradient-boosting check (§6.8)
    if with_gb:
        out["gradient_boosting"] = gb_sdev(ctx, w, n_jobs, n_boot)
    w.json("secondary/summary.json", {"stage": "S-dev secondary analyses (SPEC §6.6-§6.8)", **out})
    return out


def gb_sdev(ctx: Context, w: Writer, n_jobs: int, n_boot: int) -> dict:
    g = cv_gb(ctx.F, ctx.form, ctx.plan, n_jobs=n_jobs)
    pr = {**g["pred"], "M2": ctx.preds["M2"]}
    sc = Scored(ctx.frame, pr, n_boot=n_boot)
    pairs = [("GB-full", "GB-cuff"), ("GB-hr", "GB-cuff"), ("GB-full", "GB-full-P1"), ("GB-full-P1", "GB-cuff"),
             ("GB-cuff", "M2")]
    con = {f"{a} vs {b}": sc.contrast(a, b) for a, b in pairs}
    full = con["GB-full vs GB-cuff"]
    meets = bool(full.meets_rule and placebo_valid(con["GB-full-P1 vs GB-cuff"], con["GB-full vs GB-full-P1"]))
    line = gb_line(meets, ctx.row, full)
    w.csv("secondary/gb_summary.csv", sc.summary())
    w.csv("secondary/gb_settings.csv", pd.DataFrame([{"arm_fold": k, "setting": v} for k, v in g["chosen"].items()]))
    return {"contrasts": con, "meets_rule_with_placebo": meets, "readme_line": line}


# ----------------------------------------------------------------------------
# §9 step 6: freeze
# ----------------------------------------------------------------------------

FROZEN_FILE = "models.pkl"


def freeze(T: Tables, w: Writer, n_jobs: int = 1, expected_structure: tuple | None = None) -> dict:
    """Refit every confirmatory model on all S-dev (penalties chosen with the outer folds, §4.5): all
    analysed S-dev cases (SPEC §2.2)."""
    _analysed_only(T)
    frame = build_frame(T, "primary", 1)
    F = Cols(frame)
    plan = make_fold_plan(T.analysed, SEED)
    m2cv = run_cv({c: (F, m2_chain(*c), plan) for c in M2_COMBOS}, n_jobs)
    (form, structure), _ = choose_m2(frame, m2cv)
    if expected_structure is not None and tuple(expected_structure) != (form, structure):
        raise RuntimeError(f"M2 structure {form, structure} differs from the confirmatory {expected_structure}")
    tri, u = build_triples(T)
    Ft, Fu, F2 = Cols(tri), Cols(u), Cols(build_frame(T, "primary", 2))
    jobs = _confirmatory_jobs(F, Ft, Fu, form, structure, plan)
    jobs["yard_Tia"] = (Ft, yard_T_chain(form, "ia", names=("L", "LH-ia", "T-ia")), plan)
    jobs["m2_two_ahead"] = (F2, m2_chain(form, structure), plan)
    fits = run_refit(jobs, n_jobs)
    frozen = {"form": form, "structure": structure, "chains": fits, "sdev_cases": T.analysed["caseid"].tolist(),
              "sdev_subjects": sorted(set(T.analysed["subjectid"].tolist()))}
    p = w.out / "frozen" / FROZEN_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(frozen, protocol=5)
    p.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    w.files[f"frozen/{FROZEN_FILE}"] = sha
    w.csv("frozen/coefficients.csv", pd.concat([sec.coefficient_table(f) for f in fits.values()], ignore_index=True))
    w.json("frozen/FROZEN.json", {"stage": "freeze (SPEC §9 step 6)", "models_file": FROZEN_FILE,
                                  "models_sha256": sha, "m2_structure": {"form": form, "structure": structure},
                                  "chains": {k: list(v.spec.stages) for k, v in fits.items()},
                                  "n_sdev_cases": len(frozen["sdev_cases"]),
                                  "sdev_case_counts": T.counts()})
    return frozen


def load_frozen(root: Path) -> dict:
    meta = __import__("json").loads((root / "frozen" / "FROZEN.json").read_text())
    blob = (root / "frozen" / meta["models_file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["models_sha256"]:
        raise RuntimeError("frozen models do not match their recorded SHA-256")
    return pickle.loads(blob)


def frozen_predictions(frozen: dict, T: Tables) -> dict:
    """Every frozen model on a cohort's tables."""
    ch = frozen["chains"]
    frame = build_frame(T, "primary", 1)
    F = Cols(frame)
    pm, pp1 = ch["main"].predict(F), ch["p1"].predict(F)
    preds = {**baseline_preds(frame), "M2": pm[:, 0], "M3": pm[:, 1], "M4": pm[:, 2],
             "M3-P1": pp1[:, 1], "M4-P1": pp1[:, 2]}
    tri, u = build_triples(T)
    Ft, Fu = Cols(tri), Cols(u)
    yt, yp1, yia = ch["yard_T"].predict(Ft), ch["yard_TP1"].predict(Ft), ch["yard_Tia"].predict(Ft)
    ypreds = {"L": yt[:, 0], "LH": yt[:, 1], "T": yt[:, 2], "LH-P1": yp1[:, 1], "T-P1": yp1[:, 2],
              "LH-ia": yia[:, 1], "T-ia": yia[:, 2], "U": ch["yard_U"].predict(Fu)[:, 0],
              "M2 2-ahead": ch["m2_two_ahead"].predict(Cols(build_frame(T, "primary", 2)))[:, 0]}
    return {"frame": frame, "preds": preds, "tri": tri, "ypreds": ypreds}


# ----------------------------------------------------------------------------
# §9 step 7: the one frozen S-ob scoring
# ----------------------------------------------------------------------------

def score_sob(frozen: dict, T: Tables, w: Writer, n_boot: int = N_BOOT) -> dict:
    """The one frozen S-ob scoring: the primary outputs on the analysed cases, and in the same run the
    SPEC §2.2 sensitivity analysis on all analysable cases (`frozen_scoring/sensitivity_including_nmb/`,
    written also when no case has a recorded neuromuscular blocker)."""
    overlap = set(T.cases["subjectid"]) & set(frozen["sdev_subjects"])
    if overlap:
        raise RuntimeError("a subject has cases in both cohorts; its S-dev case must be excluded (§2.2)")
    _analysed_only(T)
    fp = frozen_predictions(frozen, T)
    frame, preds = fp["frame"], fp["preds"]
    sc = Scored(frame, preds, n_boot=n_boot)
    _check_entering(sc, T)                       # n = analysed patients entering the primary outcome
    con = {(a, b): sc.contrast(a, b) for a, b in MAIN_CONTRASTS}
    coverage = float(frame["wf_available@real"].mean())
    x = con[("M4", "M2")]
    s2 = sentence2(sc.n_patients, x, coverage)
    ces = cesarean_line(sc.n_patients, x, coverage)
    y = yardstick_scores(fp["tri"], fp["ypreds"], n_boot=n_boot)
    # the sensitivity analysis runs before any primary output is written, so a fault in it cannot
    # call for a code change after Sentence 2 is on disk (§9 step 7)
    sens = score_sob_including_nmb(frozen, T, w, n_boot)
    w.csv("frozen_scoring/predictions_primary.csv", _pred_table(frame, preds))
    w.csv("frozen_scoring/per_patient_primary.csv", sc.per_patient_long())
    w.csv("frozen_scoring/model_summary.csv", sc.summary())
    w.csv("frozen_scoring/contrasts.csv", pd.DataFrame([c.as_dict() for c in con.values()]))
    w.csv("frozen_scoring/yardstick_predictions.csv", _pred_table(fp["tri"], fp["ypreds"]))
    counts = T.counts()
    summary = {"stage": "frozen S-ob scoring (SPEC §9 step 7)", "n": sc.n_patients, "n_forecasts": sc.n_forecasts,
               "n_analysable_cases": counts["n_analysable_cases"],
               "n_nmb_recorded_cases": counts["n_nmb_recorded_cases"],
               "n_analysed_cases": counts["n_analysed_cases"], "waveform_coverage": coverage,
               "coverage_rule_applies": coverage_rule_applies(coverage),
               "contrasts": {f"{a} vs {b}": c for (a, b), c in con.items()}, "yardstick": y,
               "sentence_2": s2, "cesarean_line": ces, "n_bootstrap_resamples": n_boot}
    w.json("frozen_scoring/summary.json", summary)
    w.text("frozen_scoring/headline.md", f"# Frozen S-ob scoring (SPEC §9 step 7)\n\n**Sentence 2.** {s2}\n\n"
                                         f"**Cesarean line.** {ces}\n")
    summary["sensitivity_including_nmb"] = sens
    return summary


def score_sob_including_nmb(frozen: dict, T: Tables, w: Writer, n_boot: int = N_BOOT) -> dict:
    """SPEC §2.2 sensitivity analysis of the frozen scoring: the frozen models on all analysable S-ob
    cases, including those with a recorded neuromuscular blocker. Part of the one frozen scoring run;
    it cannot change Sentence 2 or the cesarean line."""
    Tn = replace(T, include_nmb=True)
    fp = frozen_predictions(frozen, Tn)
    frame, preds = fp["frame"], fp["preds"]
    sc = Scored(frame, preds, n_boot=n_boot)
    _check_entering(sc, Tn)
    con = {(a, b): sc.contrast(a, b) for a, b in MAIN_CONTRASTS}
    counts = Tn.counts()
    d = f"frozen_scoring/{NMB_DIR}"
    w.csv(f"{d}/predictions_primary.csv", _pred_table(frame, preds, {"nmb_recorded": frame["nmb_recorded"]}))
    w.csv(f"{d}/per_patient_primary.csv", sc.per_patient_long())
    w.csv(f"{d}/model_summary.csv", sc.summary())
    w.csv(f"{d}/contrasts.csv", pd.DataFrame([c.as_dict() for c in con.values()]))
    n_nmb = counts["n_nmb_recorded_cases"]
    note = (f"{n_nmb} analysable case(s) with a recorded neuromuscular blocker are included here and excluded "
            "from the primary frozen scoring" if n_nmb else
            "No analysable case has a recorded neuromuscular blocker (count 0); these results equal the "
            "primary frozen scoring")
    summary = {"stage": "frozen S-ob scoring (SPEC §9 step 7), sensitivity analysis (SPEC §2.2)",
               "analysis": NMB_ANALYSIS, **counts, "note": note,
               "n": sc.n_patients, "n_forecasts": sc.n_forecasts,
               "waveform_coverage": float(frame["wf_available@real"].mean()) if len(frame) else float("nan"),
               "contrasts": {f"{a} vs {b}": c for (a, b), c in con.items()}, "n_bootstrap_resamples": n_boot}
    w.json(f"{d}/summary.json", summary)
    return {k: summary[k] for k in ("analysis", "n_nmb_recorded_cases", "nmb_recorded_caseids", "n")}


# ----------------------------------------------------------------------------
# §9 step 8: S-ob analyses after the frozen scoring (labelled; models fitted on S-dev alone)
# ----------------------------------------------------------------------------

AFTER_LABEL = "computed after the frozen scoring (SPEC §9 step 8); cannot change Sentences 1-3"


def after_frozen(T_dev: Tables, T_ob: Tables, frozen: dict, w: Writer, n_jobs: int = 1, n_boot: int = N_BOOT,
                 with_gb: bool = True) -> dict:
    form, st = frozen["form"], frozen["structure"]
    _analysed_only(T_dev, T_ob)
    fp = frozen_predictions(frozen, T_ob)
    frame, preds = fp["frame"], fp["preds"]
    out = {"label": AFTER_LABEL}
    masks = {"all": np.ones(len(frame), dtype=bool), **sec.bin_masks(frame),
             "waveform available": frame["wf_available@real"].to_numpy(),
             **{f"quality {k}": v for k, v in sec.quality_masks(frame).items()}}
    if "vaso_stratum" in frame.columns:
        masks.update({f"vasopressor stratum {s}": (frame["vaso_stratum"] == s).to_numpy()
                      for s in sorted(frame["vaso_stratum"].dropna().unique())})
    s, c = sec.restricted(frame, preds, masks)
    w.csv("after_frozen_scoring/restricted_summary.csv", s)
    w.csv("after_frozen_scoring/restricted_contrasts.csv", c)
    # refits on S-dev alone, scored on S-ob
    dframe = build_frame(T_dev, "primary", 1)
    Fd, Fdsbp = Cols(dframe), Cols(dframe, "sbp")
    plan = make_fold_plan(T_dev.analysed, SEED)
    jobs = {"p2": (Fd, ladder(form, st, "p2", names=("M2", "M3-P2", "M4-P2")), plan),
            "strict": (Fd, ladder(form, st, "real", wf_variant="strict", names=("M2", "M3", "M4-strict")), plan),
            "sbp": (Fdsbp, ladder(form, st, target="sbp"), plan),
            "sbp_p1": (Fdsbp, ladder(form, st, "p1", names=("M2", "M3-P1", "M4-P1"), target="sbp"), plan),
            "other_structure": (Fd, ladder(form, other(st)), plan),
            "hmed": (Fd, ladder(form, st, hcol="hmed_min", bincol="hmed_bin"), plan),
            "two_ahead": (Cols(build_frame(T_dev, "primary", 2)), ladder(form, st), plan),
            "secondary_window": (Cols(build_frame(T_dev, "secondary", 1)), ladder(form, st), plan),
            **{f"ablation_{g}": (Fd, ablation_chain(form, st, g), plan) for g in ABLATION_GROUPS}}
    fits = run_refit(jobs, n_jobs)
    F = Cols(frame)
    Fsbp = Cols(frame, "sbp")
    rows, summ = [], []

    def add(name, fr, pr, pairs, mask=None, target="map"):
        sc = Scored(fr, pr, mask=mask, target=target, n_boot=n_boot)
        summ.append(sec.summary_rows(sc, analysis=name))
        rows.extend(sec.contrast_rows(sc, pairs, analysis=name))

    p2 = fits["p2"].predict(F)
    add("P2 (issued after 12 min)", frame, {"M2": preds["M2"], "M4": preds["M4"], "M4-P2": p2[:, 2]},
        [("M4", "M2"), ("M4-P2", "M2"), ("M4", "M4-P2")], mask=frame["t_issue"].to_numpy() > P2_MIN_ISSUE_S)
    sx = fits["strict"].predict(F)
    add("template-correlation quality (strict)", frame, {"M2": preds["M2"], "M4": preds["M4"], "M4-strict": sx[:, 2]},
        [("M4-strict", "M2"), ("M4-strict", "M4")])
    for key, name in (("other_structure", f"horizon structure {other(st)}"),
                      ("hmed", "h replaced by median previous interval")):
        pr = fits[key].predict(F)
        add(name, frame, {"M2": pr[:, 0], "M3": pr[:, 1], "M4": pr[:, 2]}, [("M4", "M2"), ("M3", "M2"), ("M4", "M3")])
    f2, fs = build_frame(T_ob, "primary", 2), build_frame(T_ob, "secondary", 1)
    for key, fr, name in (("two_ahead", f2, "2-reading-ahead targets"), ("secondary_window", fs, "secondary window")):
        pr = fits[key].predict(Cols(fr))
        add(name, fr, {"M2": pr[:, 0], "M3": pr[:, 1], "M4": pr[:, 2]}, [("M4", "M2"), ("M3", "M2"), ("M4", "M3")])
    sb, sbp1 = fits["sbp"].predict(Fsbp), fits["sbp_p1"].predict(Fsbp)
    sbp_preds = {"M2": sb[:, 0], "M3": sb[:, 1], "M4": sb[:, 2], "M4-P1": sbp1[:, 2]}
    add("SBP models", frame, sbp_preds, [("M4", "M2"), ("M3", "M2"), ("M4", "M3"), ("M4-P1", "M2")], target="sbp")
    abl = {"M2": preds["M2"]}
    for g in ABLATION_GROUPS:
        abl[f"M2+{g}"] = fits[f"ablation_{g}"].predict(F)[:, 1]
    add("feature-group ablation", frame, abl, [(f"M2+{g}", "M2") for g in ABLATION_GROUPS])
    w.csv("after_frozen_scoring/variants_summary.csv", pd.concat(summ, ignore_index=True))
    w.csv("after_frozen_scoring/variants_contrasts.csv", pd.DataFrame(rows))
    hyp = sec.hypotension(frame, {k: preds[k] for k in ("M2", "M3", "M4", "M4-P1")}, sbp_preds)
    w.csv("after_frozen_scoring/hypotension.csv", hyp)
    w.csv("after_frozen_scoring/large_changes.csv", sec.large_changes(frame, preds))
    # intervals: widths from all S-dev outer-fold residuals (§6.7)
    dcv = run_cv({"main": (Fd, ladder(form, st), plan)}, n_jobs)["main"]
    widths = sec.conformal_widths(dframe, {"M2": dcv.stage("M2"), "M4": dcv.stage("M4")})
    w.csv("after_frozen_scoring/conformal.csv", sec.conformal_apply(frame, preds, widths))
    out["yardstick_with_ia"] = yardstick_scores(fp["tri"], fp["ypreds"], n_boot=n_boot)
    out["data_quality"] = sec.data_quality(T_ob, frame)
    if with_gb:   # appendix only (§6.8)
        gfit = refit_gb(Fd, form, plan, n_jobs=n_jobs)
        gp = {**predict_gb(gfit, F), "M2": preds["M2"]}
        sc = Scored(frame, gp, n_boot=n_boot)
        out["gradient_boosting_appendix"] = {
            f"{a} vs {b}": sc.contrast(a, b) for a, b in (("GB-full", "GB-cuff"), ("GB-hr", "GB-cuff"),
                                                          ("GB-full", "GB-full-P1"), ("GB-cuff", "M2"))}
        out["gradient_boosting_settings"] = {a: v["setting"] for a, v in gfit.items()}
    out["hypotension_line_sob"] = sec.hypotension_line({"S-ob": hyp})
    w.json("after_frozen_scoring/summary.json", out)
    return {**out, "frame": frame, "preds": preds, "hyp": hyp}


def site_cases(frozen: dict, T_ob: Tables, source, root: Path, cases_dir: Path, meta: dict,
               synthetic: bool) -> dict:
    """Per-case site files (§8.1-§8.3) after the frozen scoring (§9 step 8); see `site_cases.py`.
    Needs `root/sob/frozen_scoring/summary.json` (n) and `root/sob/after_frozen_scoring/conformal.csv`
    (the §6.7 half-widths). No model is fitted. Returns the manifest."""
    import json
    from .site_cases import build_site_cases, read_widths, write_site_cases
    root = Path(root)
    n = json.loads((root / "sob" / "frozen_scoring" / "summary.json").read_text())["n"]
    widths = read_widths(root / "sob" / "after_frozen_scoring" / "conformal.csv")
    cases, series = build_site_cases(frozen, T_ob, source, widths, expected_n=int(n))
    return write_site_cases(cases_dir, cases, series, synthetic, {**meta, "stage": AFTER_LABEL})


def readme_blocks(sdev_dir: Path, sob_dir: Path) -> str:
    """README block 5 table and hypotension line (§8.1) from the S-dev and S-ob outputs."""
    rd = lambda p: pd.read_csv(p, skiprows=1)
    d, o = rd(sdev_dir / "confirmatory" / "model_summary.csv"), rd(sob_dir / "frozen_scoring" / "model_summary.csv")
    import json
    cd = json.loads((sdev_dir / "confirmatory" / "summary.json").read_text())["waveform_coverage"]
    co = json.loads((sob_dir / "frozen_scoring" / "summary.json").read_text())["waveform_coverage"]
    lines = ["| Model | S-dev mean RMSE (95% CI), mmHg | S-ob mean RMSE (95% CI), mmHg |", "|---|---|---|"]
    for m in README_MODELS:
        a, b = d.set_index("model").loc[m], o.set_index("model").loc[m]
        lines.append(f"| {m} | {a.rmse_mean:.2f} ({a.rmse_mean_ci_lo:.2f} to {a.rmse_mean_ci_hi:.2f}) | "
                     f"{b.rmse_mean:.2f} ({b.rmse_mean_ci_lo:.2f} to {b.rmse_mean_ci_hi:.2f}) |")
    lines.append(f"| Waveform coverage | {100 * cd:.1f}% | {100 * co:.1f}% |")
    hd = rd(sdev_dir / "secondary" / "hypotension.csv") if (sdev_dir / "secondary" / "hypotension.csv").exists() else None
    ho = rd(sob_dir / "after_frozen_scoring" / "hypotension.csv")
    tabs = ({"S-dev": hd} if hd is not None else {}) | {"S-ob": ho}
    return "\n".join(lines) + "\n\n" + sec.hypotension_line(tabs) + "\n"
