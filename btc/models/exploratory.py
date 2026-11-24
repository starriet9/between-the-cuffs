"""Exploratory lead-lag and detection-limit analyses on S-dev (EXPLORATORY_PLAN.md). Both are exploratory,
computed after the registered results were known, and change no registered outcome, rule or sentence. The
lead-lag analysis is an item of SPEC.md §6.8; the detection limit is new (post hoc exploratory, SPEC.md §9).

Lead-lag (plan §1). On the primary-window next-reading forecasts of the analysed cases, the within-case
correlation between a feature change and a cuff MAP change. The feature change at reading r is the
registered change form (`hr_change`, `wf_<f>_chg`), which equals the per-forecast value at r minus that at
r-1 (`change_identity` checks this on the feature tables). MAP changes: concurrent MAP_r - MAP_{r-1} (valid
readings table), next MAP_{r+1} - MAP_r (the forecast target), and the next-reading residual of the
cross-validated M2. Each case's pairs are centred at the case's own means; one Pearson correlation is
computed from the pooled centred values; a case enters with >= 3 usable pairs. The interval comes from the
subject bootstrap of the primary outcome (`analyses.Scored`), applied to per-case sums, so one set of
resamples serves every feature, MAP change and variant (real, and the P1 donor features as a placebo).

Detection limit (plan §2). A planted column s = y + kappa * sigma_c * z (y the target change, sigma_c the
root mean square of y over the patient's forecasts, z seeded standard normal noise; s = sigma_c * z at zero
strength) is added to the waveform block of the real design, and the registered chain M2 -> M3 -> M4 is
fitted with it by the unchanged nested cross-validation. Nominal strengths q set kappa^2 = (1-q)^2 /
(1 - (1-q)^2); the oracle forecast (1 - beta) * (M2 change) + beta * s, beta = 1 - (1-q)^2, on the rows with
waveform features (M2 elsewhere) gives the realised known amount. Planted values are never written out.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from .. import paths, provenance
from ..io import FROZEN_SCORING_ENV
from . import secondary as sec
from .analyses import Scored, ladder, m2_chain, run_cv
from .bootstrap import percentile_ci
from .constants import MIN_FORECASTS_RMSE, N_BOOT, SEED, WF_COLS
from .data import Tables, build_frame
from .design import Cols
from .folds import make_fold_plan
from .ridge import ChainSpec, hr_block, wf_block

PLAN_FILE = "EXPLORATORY_PLAN.md"
LABEL = "exploratory, computed after the registered results were known (EXPLORATORY_PLAN.md)"

# ----------------------------------------------------------------------------
# Lead-lag (plan §1)
# ----------------------------------------------------------------------------

# (feature, group as in SPEC §6.6 item 4, registered change column, registered level column)
LL_FEATURES = (("hr_level", "hr", "hr_change", "hr_level"),
               ("pat", "pat", "wf_pat_chg", "wf_pat_lvl"),
               ("rise", "shape", "wf_rise_chg", "wf_rise_lvl"),
               ("width", "shape", "wf_width_chg", "wf_width_lvl"),
               ("bav", "bav", "wf_bav_chg", "wf_bav_lvl"))
LL_VARIANTS = ("real", "p1")
MAP_CHANGES = ("concurrent", "next", "next_residual_m2")
MIN_PAIRS = MIN_FORECASTS_RMSE          # plan §1.4: the project's minimum per patient
IDENTITY_ATOL = 1e-9                    # plan §1.1 [judgment]: floating-point rounding only
REPRO_ATOL = 1e-12                      # recomputed confirmatory values


def _analysed_only(T: Tables) -> None:
    if T.include_nmb:
        raise ValueError("the exploratory analyses use the analysed cases only (SPEC §2.2)")


def _structure(conf: dict) -> tuple:
    m2 = conf["m2_structure"]
    return m2["form"], m2["structure"]


def change_identity(T: Tables, require: bool = True) -> list:
    """Plan §1.1: on the primary-window feature tables of the analysed cases, the registered change column at
    reading r against the level column at r minus that at r-1. Returns one row per (feature, variant); with
    `require`, raises if they differ beyond rounding where both exist, or if the difference exists where the
    change column does not."""
    ok = set(T.analysed["caseid"].astype(int))
    out = []
    for feat, _, chg, lvl in LL_FEATURES:
        tab = T.features_hr if chg.startswith("hr_") else T.features_wf
        for v in LL_VARIANTS:
            d = tab.loc[(tab["window"] == "primary") & (tab["variant"] == v) & tab["caseid"].isin(ok),
                        ["caseid", "issue_n", lvl, chg]]
            prev = d[["caseid", "issue_n", lvl]].assign(issue_n=d["issue_n"] + 1)
            m = d.merge(prev, on=["caseid", "issue_n"], how="left", suffixes=("", "_prev"), validate="1:1")
            diff = (m[lvl] - m[f"{lvl}_prev"]).to_numpy(float)
            c = m[chg].to_numpy(float)
            both = np.isfinite(diff) & np.isfinite(c)
            gap = float(np.max(np.abs(c[both] - diff[both]))) if both.any() else 0.0
            row = {"feature": feat, "variant": v, "change_column": chg, "level_column": lvl, "n_rows": int(len(m)),
                   "n_change_finite": int(np.isfinite(c).sum()), "n_level_difference_finite": int(np.isfinite(diff).sum()),
                   "n_both": int(both.sum()), "max_abs_difference": gap,
                   "n_level_difference_without_change": int((np.isfinite(diff) & ~np.isfinite(c)).sum())}
            out.append(row)
            if require and (gap > IDENTITY_ATOL or row["n_level_difference_without_change"]):
                raise RuntimeError(f"the change column {chg}@{v} is not the level difference: {row}")
    return out


def map_changes(T: Tables, frame: pd.DataFrame, pred_m2: np.ndarray) -> dict:
    """Plan §1.2: concurrent, next, and next-reading M2 residual, one value per forecast row of `frame`."""
    r = T.readings.loc[T.readings["valid"].astype(bool), ["caseid", "n", "map"]].copy()
    r["n"] = r["n"].astype(int)
    s = r.set_index(["caseid", "n"])["map"]
    if s.index.has_duplicates:
        raise ValueError("duplicate valid reading index")
    cid, n = frame["caseid"].to_numpy(int), frame["issue_n"].to_numpy(int)
    cur = s.reindex(pd.MultiIndex.from_arrays([cid, n])).to_numpy(float)
    prev = s.reindex(pd.MultiIndex.from_arrays([cid, n - 1])).to_numpy(float)
    last, y = frame["last_map"].to_numpy(float), frame["y_map"].to_numpy(float)
    if not (np.all(np.isfinite(prev)) and np.allclose(cur, last, rtol=0, atol=1e-9)):
        raise RuntimeError("the valid readings table does not hold readings r-1 and r of every forecast")
    conc = cur - prev
    lag = (frame["cuff_lag0"] - frame["cuff_lag1"]).to_numpy(float)
    if not np.allclose(conc, lag, rtol=0, atol=1e-9):
        raise RuntimeError("MAP_r - MAP_{r-1} differs from cuff_lag0 - cuff_lag1")
    return {"concurrent": conc, "next": y - last, "next_residual_m2": y - np.asarray(pred_m2, float)}


def case_sums(caseid: np.ndarray, x: np.ndarray, y: np.ndarray, cases: np.ndarray, min_pairs: int = MIN_PAIRS) -> dict:
    """Within-case centred sums (Sxx, Syy, Sxy) over the pairs with finite x and y, one row per case of
    `cases` (ascending case ID, the bootstrap's patients). A case with fewer than `min_pairs` pairs contributes
    zero. Also the pair counts."""
    caseid, x, y = np.asarray(caseid), np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = pd.DataFrame({"c": caseid[ok], "x": x[ok], "y": y[ok]})
    g = d.groupby("c")
    xc, yc = d["x"] - g["x"].transform("mean"), d["y"] - g["y"].transform("mean")
    s = pd.DataFrame({"c": d["c"], "xx": xc * xc, "yy": yc * yc, "xy": xc * yc}).groupby("c").sum()
    npair = g.size()
    outside = npair.loc[~npair.index.isin(cases)]
    if (outside >= min_pairs).any():
        raise RuntimeError(f"cases outside the bootstrap's patients have >= {min_pairs} pairs: {list(outside.index[:5])}")
    n = npair.reindex(cases, fill_value=0).to_numpy(int)
    enter = n >= min_pairs
    S = s.reindex(cases, fill_value=0.0)[["xx", "yy", "xy"]].to_numpy(float) * enter[:, None]
    below = (npair > 0) & (npair < min_pairs)
    return {"S": S, "n": n, "enter": enter, "n_cases": int(enter.sum()), "n_pairs": int(n[enter].sum()),
            "n_cases_below_min": int(below.sum()), "n_pairs_below_min": int(npair[below].sum()),
            "n_pairs_usable": int(ok.sum())}


def within_r(S: np.ndarray) -> float:
    xx, yy, xy = S.sum(axis=0)
    return float(xy / np.sqrt(xx * yy))


def within_r_boot(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    A = W @ S
    return A[:, 2] / np.sqrt(A[:, 0] * A[:, 1])


def lead_lag(T: Tables, conf: dict, n_jobs: int = 1, n_boot: int = N_BOOT, require_identity: bool = True) -> dict:
    """Plan §1. Returns {"table", "identity", "summary"}; writes nothing."""
    _analysed_only(T)
    form, st = _structure(conf)
    frame = build_frame(T, "primary", 1)
    F = Cols(frame)
    plan = make_fold_plan(T.analysed, SEED)
    pred_m2 = run_cv({"m2": (F, m2_chain(form, st), plan)}, n_jobs)["m2"].stage("M2")
    sc = Scored(frame, {"M2": pred_m2}, n_boot=n_boot)
    if sc.n_patients != conf["N"]:
        raise RuntimeError(f"{sc.n_patients} patients enter, the primary outcome has {conf['N']}")
    m2_mean = float(np.mean(sc.rmse("M2")))
    if not np.isclose(m2_mean, conf["contrasts"]["M4 vs M2"]["mean_comp"], rtol=0, atol=REPRO_ATOL):
        raise RuntimeError("the re-derived M2 does not reproduce the confirmatory M2 error")
    identity = change_identity(T, require=require_identity)
    dmap = map_changes(T, frame, pred_m2)
    cid, W, cases = frame["caseid"].to_numpy(int), sc.bs.W, sc.enter
    rows, boot = [], {}
    for feat, group, chg, _ in LL_FEATURES:
        for v in LL_VARIANTS:
            x = frame[f"{chg}@{v}"].to_numpy(float)
            for mc in MAP_CHANGES:
                cs = case_sums(cid, x, dmap[mc], cases)
                r, rb = within_r(cs["S"]), within_r_boot(W, cs["S"])
                lo, hi = percentile_ci(rb)
                boot[(feat, v, mc)] = rb
                rows.append({"feature": feat, "group": group, "change_column": chg, "variant": v, "map_change": mc,
                             "n_cases": cs["n_cases"], "n_pairs": cs["n_pairs"],
                             "n_cases_below_min": cs["n_cases_below_min"], "n_pairs_below_min": cs["n_pairs_below_min"],
                             "r": r, "r_ci_lo": lo, "r_ci_hi": hi})
        for mc in MAP_CHANGES:
            a = next(x for x in rows if (x["feature"], x["variant"], x["map_change"]) == (feat, "real", mc))
            b = next(x for x in rows if (x["feature"], x["variant"], x["map_change"]) == (feat, "p1", mc))
            lo, hi = percentile_ci(boot[(feat, "real", mc)] - boot[(feat, "p1", mc)])
            rows.append({"feature": feat, "group": group, "change_column": chg, "variant": "real minus p1",
                         "map_change": mc, "r": a["r"] - b["r"], "r_ci_lo": lo, "r_ci_hi": hi})
    table = pd.DataFrame(rows)
    for c in ("n_cases", "n_pairs", "n_cases_below_min", "n_pairs_below_min"):
        table[c] = table[c].astype("Int64")             # blank in the real-minus-placebo rows
    summary = {
        "stage": "exploratory lead-lag, S-dev (EXPLORATORY_PLAN.md §1; SPEC.md §6.8)", "label": LABEL,
        "m2_structure": {"form": form, "structure": st}, "n_forecasts": int(len(frame)),
        "n_cases_with_forecasts": int(frame["caseid"].nunique()), "n_bootstrap_patients": sc.n_patients,
        "n_bootstrap_subjects": int(sc.bs.n_subjects), "n_bootstrap_resamples": n_boot, "bootstrap_seed": SEED,
        "min_pairs_per_case": MIN_PAIRS, **T.counts(),
        "m2_reproduction": {"patient_mean_rmse": m2_mean,
                            "confirmatory": conf["contrasts"]["M4 vs M2"]["mean_comp"]},
        "change_identity": identity,
        "expectation": "pulse arrival time shortens when blood pressure rises, so the concurrent correlation of "
                       "pulse arrival time is expected to be negative (a stated expectation, not a claim; plan §1.8)",
        "map_changes": {"concurrent": "MAP_r - MAP_{r-1}", "next": "MAP_{r+1} - MAP_r (the forecast target)",
                        "next_residual_m2": "MAP_{r+1} minus the cross-validated M2 forecast"},
        "lead_lag": rows,
    }
    return {"table": table, "identity": pd.DataFrame(identity), "summary": summary}


# ----------------------------------------------------------------------------
# Detection limit (plan §2)
# ----------------------------------------------------------------------------

STRENGTHS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20)      # nominal RMSE reductions (plan §2.2)
NOISE_SEED = SEED                                          # main draw (plan §2.5)
REPLICATE_SEEDS = tuple(range(20261119, 20261129))         # ten further draws (plan §2.5)
PLANTED = "M4+planted"


def kappa(q: float) -> float:
    return float(np.sqrt((1.0 - q) ** 2 / (1.0 - (1.0 - q) ** 2)))


def beta(q: float) -> float:
    return float(1.0 - (1.0 - q) ** 2)


def patient_rms(caseid: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sigma_c per row: the root mean square of y over the rows of the same case."""
    d = pd.DataFrame({"c": np.asarray(caseid), "y2": np.asarray(y, float) ** 2})
    return np.sqrt(d.groupby("c")["y2"].transform("mean").to_numpy(float))


def noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def planted_column(y: np.ndarray, sigma: np.ndarray, z: np.ndarray, q: float) -> np.ndarray:
    """Plan §2.1: s = y + kappa(q) sigma_c z; zero strength: s = sigma_c z."""
    return sigma * z if q == 0 else y + kappa(q) * sigma * z


def plant_name(seed: int, q: float) -> str:
    return f"plant_s{seed}_q{int(round(q * 1000)):03d}"


def planted_chain(form: str, structure: str, col: str) -> ChainSpec:
    """M2 -> M3 -> M4+planted: the registered ladder with the planted column added to the waveform block."""
    return ChainSpec(f"planted:{col}", ("M2", "M3", PLANTED), form=form, structure=structure,
                     blocks=(hr_block("real"), wf_block("real", cols=WF_COLS + (col,), name=f"wf+{col}@real")))


def oracle_forecast(frame: pd.DataFrame, pred_m2: np.ndarray, s: np.ndarray, q: float) -> np.ndarray:
    """Plan §2.2: last + (1 - beta)(M2 change) + beta s on the rows with waveform features; M2 elsewhere."""
    last = frame["last_map"].to_numpy(float)
    avail = frame["wf_available@real"].to_numpy(bool)
    b = beta(q)
    if b == 0:
        return np.asarray(pred_m2, float).copy()             # zero strength: the oracle is M2
    o = last + (1.0 - b) * (pred_m2 - last) + b * s
    return np.where(avail, o, pred_m2)


def _key(kind: str, seed: int, q: float) -> str:
    return f"{kind} s{seed} q{100 * q:g}%"


def _wf_penalties(cv) -> list:
    if any(list(fold) != ["pooled"] for fold in cv.lam):
        raise ValueError("the detection limit is reported for the pooled structure")
    return ["dropped" if f["pooled"][2] is None else f["pooled"][2] for f in cv.lam]


def detection_limit(T: Tables, conf: dict, n_jobs: int = 1, n_boot: int = N_BOOT, strengths=STRENGTHS,
                    seeds=(NOISE_SEED, *REPLICATE_SEEDS)) -> dict:
    """Plan §2. Returns {"main", "replicates", "replicate_summary", "penalties", "summary"}; writes nothing."""
    _analysed_only(T)
    form, st = _structure(conf)
    if st != "pooled":
        raise RuntimeError("the detection limit is reported for the pooled structure chosen by the confirmatory run")
    frame = build_frame(T, "primary", 1)
    key = frame[["caseid", "issue_n"]].to_numpy()
    if len(key) > 1 and not np.all((np.diff(key[:, 0]) > 0) | ((np.diff(key[:, 0]) == 0) & (np.diff(key[:, 1]) > 0))):
        raise RuntimeError("the forecast frame is not in (case ID, issue reading) order")
    y = (frame["y_map"] - frame["last_map"]).to_numpy(float)
    sigma = patient_rms(frame["caseid"].to_numpy(), y)
    cols, add = {}, {}
    for seed in seeds:
        z = noise(len(frame), seed)
        for q in strengths:
            cols[(seed, q)] = plant_name(seed, q)
            add[f"{cols[(seed, q)]}@real"] = planted_column(y, sigma, z, q)
    frame = pd.concat([frame, pd.DataFrame(add, index=frame.index)], axis=1)
    F = Cols(frame)
    plan = make_fold_plan(T.analysed, SEED)
    jobs = {"ladder": (F, ladder(form, st), plan)}
    jobs.update({k: (F, planted_chain(form, st, c), plan) for k, c in cols.items()})
    cv = run_cv(jobs, n_jobs)
    base = cv["ladder"]
    avail = frame["wf_available@real"].to_numpy(bool)
    for k in cols:
        if not (np.array_equal(cv[k].stage("M2"), base.stage("M2")) and np.array_equal(cv[k].stage("M3"), base.stage("M3"))):
            raise RuntimeError(f"the M2 or M3 stage of the planted chain {k} differs from the plain ladder")
        if not np.array_equal(cv[k].stage(PLANTED)[~avail], base.stage("M3")[~avail]):
            raise RuntimeError(f"a fallback forecast of the planted chain {k} differs from M3")
    pm2 = base.stage("M2")
    preds = {"M2": pm2, "M3": base.stage("M3"), "M4": base.stage("M4")}
    for (seed, q), c in cols.items():
        preds[_key(PLANTED, seed, q)] = cv[(seed, q)].stage(PLANTED)
        preds[_key("oracle", seed, q)] = oracle_forecast(frame, pm2, frame[f"{c}@real"].to_numpy(float), q)
    sc = Scored(frame, preds, n_boot=n_boot)
    if sc.n_patients != conf["N"]:
        raise RuntimeError(f"{sc.n_patients} patients enter, the primary outcome has {conf['N']}")
    ref, cref = sc.contrast("M4", "M2").as_dict(), conf["contrasts"]["M4 vs M2"]
    if any(not np.isclose(ref[k], cref[k], rtol=0, atol=REPRO_ATOL) for k in ("delta", "delta_lo", "delta_hi", "rr_lo", "rr_hi")):
        raise RuntimeError("the recomputed M4 vs M2 contrast differs from the confirmatory summary")
    rows, pens = [], []
    for (seed, q), c in cols.items():
        o, p = sc.contrast(_key("oracle", seed, q), "M2"), sc.contrast(_key(PLANTED, seed, q), "M2")
        wp = _wf_penalties(cv[(seed, q)])
        rows.append({"noise_seed": seed, "draw": "main" if seed == NOISE_SEED else "replicate", "strength_nominal": q,
                     "kappa": kappa(q) if q else None, "beta": beta(q),
                     "oracle_rr": o.rr, "oracle_rr_ci_lo": o.rr_lo, "oracle_rr_ci_hi": o.rr_hi,
                     "oracle_delta": o.delta, "mean_rmse_m2": p.mean_comp, "mean_rmse_planted": p.mean_model,
                     "delta": p.delta, "delta_ci_lo": p.delta_lo, "delta_ci_hi": p.delta_hi,
                     "rr": p.rr, "rr_ci_lo": p.rr_lo, "rr_ci_hi": p.rr_hi,
                     "delta_ci_below_zero": bool(p.delta_hi < 0), "meets_rule": p.meets_rule,
                     **{f"wf_penalty_fold{j}": v for j, v in enumerate(wp)},
                     "n_folds_wf_dropped": sum(v == "dropped" for v in wp)})
        t = sec.penalty_table(cv[(seed, q)])
        t.insert(1, "noise_seed", seed)
        t.insert(2, "strength_nominal", q)
        pens.append(t)
    allrows = pd.DataFrame(rows)
    main = allrows.loc[allrows["draw"] == "main"].drop(columns=["draw"]).reset_index(drop=True)
    rep = allrows.loc[allrows["draw"] == "replicate"].drop(columns=["draw"]).reset_index(drop=True)
    rsum = pd.DataFrame([{"strength_nominal": q, "n_draws": int(len(g)), "n_meets_rule": int(g["meets_rule"].sum()),
                          "n_delta_ci_below_zero": int(g["delta_ci_below_zero"].sum()),
                          "rr_min": g["rr"].min(), "rr_median": g["rr"].median(), "rr_max": g["rr"].max(),
                          "oracle_rr_min": g["oracle_rr"].min(), "oracle_rr_max": g["oracle_rr"].max(),
                          "n_draws_wf_dropped_in_every_fold": int((g["n_folds_wf_dropped"] == 5).sum())}
                         for q, g in rep.groupby("strength_nominal", sort=True)]) if len(rep) else pd.DataFrame()
    base_pen = sec.penalty_table(base)
    base_pen.insert(1, "noise_seed", None)
    base_pen.insert(2, "strength_nominal", None)
    penalties = pd.concat([base_pen, *pens], ignore_index=True)
    summary = {
        "stage": "exploratory detection limit, S-dev (EXPLORATORY_PLAN.md §2)", "label": LABEL,
        "tests": "the fitting procedure on the real S-dev design, not the physiology",
        "m2_structure": {"form": form, "structure": st}, "N": sc.n_patients, "n_forecasts": sc.n_forecasts,
        "n_forecasts_all": int(len(frame)), "n_forecasts_with_waveform": int(avail.sum()), **T.counts(),
        "strengths_nominal": list(strengths), "noise_seed_main": NOISE_SEED,
        "noise_seeds_replicates": [s for s in seeds if s != NOISE_SEED],
        "planted_column": "s = y + kappa * sigma_c * z (zero strength: sigma_c * z); y = next MAP - last MAP; sigma_c "
                          "= root mean square of y over the patient's primary forecasts; z standard normal, one draw "
                          "per forecast in (case ID, issue reading) order",
        "oracle": "(1 - beta) * M2 change + beta * s on forecasts with waveform features, M2 elsewhere; beta = "
                  "1 - (1 - q)^2",
        "rule": "relative reduction >= 5% and upper 95% CI limit of Delta < 0 (SPEC.md §6.3); placebo validity is "
                "not evaluated (plan §2.4)",
        "checks": {"m2_m3_stages_bit_identical_to_plain_ladder": True, "fallback_rows_equal_m3": True,
                   "plain_m4_vs_m2_equals_confirmatory": True, "plain_m4_vs_m2": ref},
        "n_bootstrap_resamples": n_boot, "bootstrap_seed": SEED,
        "main": main, "replicate_summary": rsum,
    }
    return {"main": main, "replicates": rep, "replicate_summary": rsum, "penalties": penalties, "summary": summary}


# ----------------------------------------------------------------------------
# Plan record and start guards (plan §3)
# ----------------------------------------------------------------------------

def plan_record() -> dict:
    """SHA-256 of EXPLORATORY_PLAN.md, whether it equals the file committed at HEAD, and its last commit."""
    p = paths.ROOT / PLAN_FILE
    sha = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    head = provenance._git("rev-parse", "--verify", "--quiet", f"HEAD:{PLAN_FILE}")
    here = provenance._git("hash-object", "--", PLAN_FILE) if p.is_file() else None
    last = provenance._git("log", "-1", "--format=%H", "--", PLAN_FILE)
    return {"file": PLAN_FILE, "sha256": sha, "committed_at_head": bool(head and here and head == here),
            "last_commit": last or None}


def start_problems(g: dict, plan: dict, conf: dict | None, tables_sha: dict | None, folds_ok: bool, out) -> list:
    """Why a run on real data must not start (plan §3)."""
    probs = []
    if g.get("state") != "clean":
        probs.append(f"the work tree is not clean (git state {g.get('state')}; untracked files count)")
    if not plan.get("committed_at_head"):
        probs.append(f"{PLAN_FILE} is not committed at HEAD, or differs from the committed file")
    if os.environ.get(FROZEN_SCORING_ENV) is not None:
        probs.append(f"{FROZEN_SCORING_ENV} is set; these S-dev analyses never read S-ob data")
    if conf is None:
        probs.append("no confirmatory summary.json")
    elif (conf.get("meta") or {}).get("inputs_sha256") != tables_sha:
        probs.append("the S-dev derived tables differ from the inputs of the confirmatory run (inputs_sha256)")
    if not folds_ok:
        probs.append("the fold plan differs from the confirmatory folds.csv")
    if out is not None and out.exists():
        probs.append(f"the output folder {out.name} exists; the first complete run is the result and a rerun is a "
                     "departure (EXPLORATORY_PLAN.md §3)")
    return probs
