"""Secondary and control analyses that only restrict or summarise scoring (SPEC §6.6, §6.7).

Analyses that change targets, window or inputs are refitted in `pipeline.py`
through the same chains; everything here reuses predictions (§4.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analyses import Scored
from .bootstrap import SubjectBootstrap, percentile_ci
from .constants import (CONFORMAL_LEVEL, DROPPED, HR_COLS, HYPO_MAP_MMHG, HYPO_REL_BASELINE, HYPO_SBP_MMHG,
                        LARGE_CHANGE_MMHG, MIN_USABLE_WINDOWS, MINUTES_PER_ALARM_RATE, N_BINS,
                        QUALITY_SHARE_CUT, WF_COLS)
from .ridge import ChainFit, CVResult

BIN_LABELS = ("15-90 s", "90-150 s", "150-210 s", "210-400 s")


def contrast_rows(sc: Scored, pairs, **tags) -> list:
    out = []
    for m, c in pairs:
        d = sc.contrast(m, c).as_dict()
        out.append({**tags, "n_patients": sc.n_patients, "n_forecasts": sc.n_forecasts, **d})
    return out


def summary_rows(sc: Scored, **tags) -> pd.DataFrame:
    s = sc.summary()
    for k, v in reversed(list(tags.items())):
        s.insert(0, k, v)
    s["n_patients_entering"] = sc.n_patients
    return s


# ----------------------------------------------------------------------------
# Restricted scoring: h-bins, subsets, strata, subgroups
# ----------------------------------------------------------------------------

STANDARD_PAIRS = (("M4", "M2"), ("M3", "M2"), ("M4", "M3"), ("M4-P1", "M2"), ("M4", "M4-P1"))


def restricted(frame, preds, masks: dict, pairs=STANDARD_PAIRS):
    """Summaries and contrasts for each named mask (scoring restricted, nothing refitted)."""
    summ, con = [], []
    for name, m in masks.items():
        sc = Scored(frame, preds, mask=m)
        summ.append(summary_rows(sc, subset=name))
        con += contrast_rows(sc, [p for p in pairs if p[0] in preds and p[1] in preds], subset=name)
    return pd.concat(summ, ignore_index=True), pd.DataFrame(con)


def bin_masks(frame) -> dict:
    return {f"bin{i + 1} {BIN_LABELS[i]}": (frame["h_bin"] == i).to_numpy() for i in range(N_BINS)}


def training_base_per_bin(frame) -> pd.DataFrame:
    """Training targets and patients per h-bin (§3, PREREG §4), all forecasts and with waveform."""
    rows = []
    for i in range(N_BINS):
        m = frame["h_bin"] == i
        w = m & frame["wf_available@real"]
        rows.append({"bin": BIN_LABELS[i], "n_targets": int(m.sum()), "n_patients": int(frame.loc[m, "caseid"].nunique()),
                     "n_targets_with_waveform": int(w.sum()),
                     "n_patients_with_waveform": int(frame.loc[w, "caseid"].nunique()),
                     "n_targets_fast_cycling": int((m & frame["fast_cycling"].astype(bool)).sum()),
                     "n_patients_fast_cycling": int(frame.loc[m & frame["fast_cycling"].astype(bool), "caseid"].nunique())})
    return pd.DataFrame(rows)


def quality_masks(frame, variant="real") -> dict:
    """Quality strata (§6.6 item 5) by the usable share of the 10-s windows between the previous
    reading (at most 120 s back) and t_r."""
    nu = frame[f"wf_n_usable@{variant}"].fillna(0).to_numpy()
    sh = frame[f"wf_usable_share@{variant}"].to_numpy()
    lo = nu < MIN_USABLE_WINDOWS
    return {"<2 usable windows": lo,
            ">=2 and <80% usable": (~lo) & (sh < QUALITY_SHARE_CUT),
            ">=2 and >=80% usable": (~lo) & (sh >= QUALITY_SHARE_CUT)}


def sex_masks(frame) -> dict:
    if "sex" not in frame.columns:
        return {}
    return {f"sex={s}": (frame["sex"] == s).to_numpy() for s in sorted(frame["sex"].dropna().unique())}


# ----------------------------------------------------------------------------
# New-onset hypotension (§6.6 item 1, §6.7)
# ----------------------------------------------------------------------------

def hypotension(frame: pd.DataFrame, preds_map: dict, preds_sbp: dict | None = None) -> pd.DataFrame:
    """Alarms (forecast below the threshold, no tuned margin) against new-onset events (target below
    the threshold while the issue reading was not) [judgment: scored over the issue points whose
    reading is not hypotensive (the at-risk forecasts); false alarms per 30 min use the summed
    horizons of the at-risk forecasts as time]."""
    defs = [("below 80% of record-start baseline", "map", HYPO_REL_BASELINE * frame["base_map"].to_numpy()),
            ("MAP < 65 mmHg", "map", np.full(len(frame), HYPO_MAP_MMHG)),
            ("SBP < 100 mmHg", "sbp", np.full(len(frame), HYPO_SBP_MMHG))]
    rows = []
    h = frame["h_s"].to_numpy()
    for name, var, thr in defs:
        preds = preds_map if var == "map" else (preds_sbp or {})
        last = frame[f"last_{var}"].to_numpy()
        y = frame[f"y_{var}"].to_numpy()
        risk = last >= thr
        event = risk & (y < thr)
        for m, p in preds.items():
            alarm = risk & (np.asarray(p) < thr)
            tp, fa = int((alarm & event).sum()), int((alarm & ~event).sum())
            t30 = h[risk].sum() / (MINUTES_PER_ALARM_RATE * 60.0)
            rows.append({"definition": name, "model": m, "n_at_risk_forecasts": int(risk.sum()),
                         "n_events": int(event.sum()), "n_event_patients": int(frame.loc[event, "caseid"].nunique()),
                         "n_alarms": int(alarm.sum()), "flagged": tp,
                         "sensitivity": tp / event.sum() if event.sum() else np.nan,
                         "ppv": tp / alarm.sum() if alarm.sum() else np.nan,
                         "false_alarms": fa, "false_alarms_per_30min": fa / t30 if t30 > 0 else np.nan,
                         "median_h_detected_s": float(np.median(h[alarm & event])) if tp else np.nan})
    return pd.DataFrame(rows)


def hypotension_line(tabs: dict) -> str:
    """README line under the table (§8.1 block 5): new-onset readings below 80% of the record-start
    baseline flagged one reading ahead, flagged / total, with false alarms per 30 min, for M2, M3, M4
    and M4-P1 in each cohort. `tabs`: cohort label -> hypotension table."""
    parts = []
    for coh, t in tabs.items():
        d = t.loc[t["definition"] == "below 80% of record-start baseline"].set_index("model")
        items = []
        for m in ("M2", "M3", "M4", "M4-P1"):
            if m in d.index:
                r = d.loc[m]
                items.append(f"{m} {int(r['flagged'])}/{int(r['n_events'])} "
                             f"({r['false_alarms_per_30min']:.2f} false alarms per 30 min)")
        parts.append(f"{coh}: " + ", ".join(items))
    return ("New-onset readings below 80% of the record-start baseline flagged one reading ahead "
            "(secondary, descriptive): " + "; ".join(parts) + ".")


# ----------------------------------------------------------------------------
# Large changes (§6.7)
# ----------------------------------------------------------------------------

def large_changes(frame: pd.DataFrame, preds: dict, models=("M0", "M2", "M3", "M4", "M4-P1")) -> pd.DataFrame:
    """Forecasts split by the realised change from the issue reading (fall >= 10, rise >= 10,
    neither). Pooled MAE and pooled mean signed error per group; the M4 - M2 MAE difference with a
    patient-level (subject) bootstrap CI [judgment: patients with any forecast in the group]."""
    d = frame["y_map"].to_numpy() - frame["last_map"].to_numpy()
    groups = {"fall >= 10 mmHg": d <= -LARGE_CHANGE_MMHG, "rise >= 10 mmHg": d >= LARGE_CHANGE_MMHG}
    groups["neither"] = ~(groups["fall >= 10 mmHg"] | groups["rise >= 10 mmHg"])
    rows = []
    y = frame["y_map"].to_numpy()
    for g, m in groups.items():
        sub = frame.loc[m]
        cases = np.sort(sub["caseid"].unique())
        subj = sub.drop_duplicates("caseid").set_index("caseid")["subjectid"].reindex(cases).to_numpy()
        code = np.searchsorted(cases, sub["caseid"].to_numpy())
        n_c = np.bincount(code, minlength=len(cases)).astype(float)
        bs = SubjectBootstrap(cases, subj) if len(cases) else None
        sae = {}
        for k in models:
            if k not in preds:
                continue
            e = np.asarray(preds[k])[m] - y[m]
            sae[k] = np.bincount(code, np.abs(e), minlength=len(cases))
            rows.append({"group": g, "model": k, "n_forecasts": int(m.sum()), "n_patients": len(cases),
                         "pooled_mae": float(np.abs(e).mean()) if m.any() else np.nan,
                         "pooled_mean_signed_error": float(e.mean()) if m.any() else np.nan})
        if bs is not None and "M4" in sae and "M2" in sae:
            diff = bs.ratio(sae["M4"], n_c) - bs.ratio(sae["M2"], n_c)
            lo, hi = percentile_ci(diff)
            rows.append({"group": g, "model": "M4 - M2 (MAE difference)", "n_forecasts": int(m.sum()),
                         "n_patients": len(cases), "pooled_mae": float((sae["M4"].sum() - sae["M2"].sum()) / m.sum()),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Cross-conformal intervals (§6.7)
# ----------------------------------------------------------------------------

def conformal_sdev(frame, preds: dict, fold: np.ndarray, models=("M2", "M4")) -> pd.DataFrame:
    """90% half-width per h-bin = 90% quantile of out-of-fold absolute residuals, constant width;
    S-dev coverage with the test fold left out of calibration [judgment: numpy's default quantile]."""
    rows = []
    y = frame["y_map"].to_numpy()
    b = frame["h_bin"].to_numpy()
    for m in models:
        r = np.abs(np.asarray(preds[m]) - y)
        for i in range(N_BINS):
            inb = b == i
            cov, widths = [], []
            for j in np.unique(fold):
                te, cal = inb & (fold == j), inb & (fold != j)
                if not te.any() or not cal.any():
                    continue
                w = float(np.quantile(r[cal], CONFORMAL_LEVEL))
                widths.append(w)
                cov.append(r[te] <= w)
            cov = np.concatenate(cov) if cov else np.array([])
            rows.append({"model": m, "bin": BIN_LABELS[i], "n_forecasts": int(inb.sum()),
                         "coverage": float(cov.mean()) if len(cov) else np.nan,
                         "half_width_mean_over_folds": float(np.mean(widths)) if widths else np.nan,
                         "half_width_all_sdev": float(np.quantile(r[inb], CONFORMAL_LEVEL)) if inb.any() else np.nan})
    return pd.DataFrame(rows)


def conformal_widths(sdev_frame, sdev_preds, models=("M2", "M4")) -> dict:
    """Half-widths for S-ob: from all S-dev outer-fold residuals per h-bin (§6.7)."""
    y = sdev_frame["y_map"].to_numpy()
    b = sdev_frame["h_bin"].to_numpy()
    return {m: {i: float(np.quantile(np.abs(np.asarray(sdev_preds[m]) - y)[b == i], CONFORMAL_LEVEL))
                for i in range(N_BINS) if (b == i).any()} for m in models}


def conformal_apply(frame, preds, widths: dict) -> pd.DataFrame:
    rows = []
    y = frame["y_map"].to_numpy()
    b = frame["h_bin"].to_numpy()
    for m, wb in widths.items():
        r = np.abs(np.asarray(preds[m]) - y)
        for i in range(N_BINS):
            inb = b == i
            w = wb.get(i, np.nan)
            rows.append({"model": m, "bin": BIN_LABELS[i], "n_forecasts": int(inb.sum()), "half_width": w,
                         "coverage": float((r[inb] <= w).mean()) if inb.any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Coefficients, penalties, dropped blocks (§6.7)
# ----------------------------------------------------------------------------

def block_kind(block) -> str:
    """"hr" or "waveform" for a block of heart-rate or waveform features (any variant), else the
    block's own name (e.g. the availability-indicator ablation block)."""
    base = {c.split("@")[0] for c in block.cols}
    if base <= set(HR_COLS):
        return "hr"
    if base <= set(WF_COLS):
        return "waveform"
    return block.name


def _design_columns(d) -> list:
    """(block index, Block or None, kind of column, raw-column index) per design column, in the
    order of `Design._raw` (cuff block first). Checked against `d.names`."""
    out = [(0, None, "cuff", None) for _ in d.cuff_cols] + [(0, None, "h", None)]
    out += [(0, None, "cuff*h", None) for _ in d.cuff_inter] + [(0, None, "extra", None) for _ in d.extra_cols]
    names = list(d.cuff_cols) + ["h"] + [f"{c}*h^{p}" for c, p in d.cuff_inter] + list(d.extra_cols)
    for bi, b in enumerate(d.blocks, start=1):
        k = d.keep[bi - 1]
        if b.kind == "indicator":
            out += [(bi, b, "indicator", j) for j in k]
            names += [b.cols[j] for j in k]
            continue
        out += [(bi, b, "feature", j) for j in range(len(b.cols))]
        names += list(b.cols)
        if d.pooled:
            out += [(bi, b, "feature*h", j) for j in range(len(b.cols))]
            names += [f"{c}*h" for c in b.cols]
        out += [(bi, b, "missing", j) for j in k]
        names += [f"miss:{b.cols[j]}" for j in k]
    if names != list(d.names):
        raise RuntimeError("design columns do not follow Design._raw")
    return out


def coefficient_table(fit: ChainFit) -> pd.DataFrame:
    """Every stage of a chain fitted on all S-dev, one row per design column, enough to apply the
    model without unpickling it (SPEC §4.5; `btc/models/design.py`, `ridge.py`):

    forecast change = coef of "(intercept)" + sum over columns of coef * (raw - scaler_mean) / scaler_sd,

    forecast = last cuff value + change. `block` is cuff, hr or waveform; `penalty` is that block's
    own penalty (the cuff block keeps M2's, §4.1); the intercept is not penalised. A raw feature
    that is missing takes `impute_median`; a `*h` column is (imputed feature - impute_median) times
    the horizon column `h_input` (minutes); a `miss:` column is 1 where the feature is missing (merged
    with identical indicators). `requires`: the stage is applied only where that column is true,
    else the stage below. A "(dropped)" row: the stage returns the stage below (§4.5). `node` is
    `pooled`, or `bin<i>` for an h-bin fitted on its own (§4.1); a forecast whose h-bin has no node
    of its own uses `pooled`."""
    rows = []
    for node, nf in fit.nodes.items():
        for s, st in enumerate(nf.stages):
            base = {"chain": fit.spec.name, "node": node, "model": fit.spec.stages[s]}
            spec_block = fit.spec.blocks[s - 1] if s else None
            if st.dropped:
                rows.append({**base, "column": "(dropped)",
                             "block": block_kind(spec_block) if spec_block else "cuff",
                             "block_name": spec_block.name if spec_block else "cuff",
                             "penalty": "dropped" if nf.lam[s] is DROPPED else nf.lam[s]})
                continue
            d = st.design
            lamvec = [nf.lam[0]] + [nf.lam[i] for i in range(1, s + 1) if nf.lam[i] is not DROPPED]
            if len(lamvec) != len(d.blocks) + 1:
                raise RuntimeError("penalties and active blocks do not line up")
            rows.append({**base, "column": "(intercept)", "coef": float(st.b0), "requires": st.requires})
            for c, (bi, blk, kind, j), b, mu, sd in zip(d.names, _design_columns(d), st.beta, d.mean, d.sd):
                med = d.med[bi - 1] if bi else None
                h_input = (d.hcol if kind in ("h", "cuff*h") else blk.hcol if kind == "feature*h" else None)
                rows.append({**base, "column": c, "block": block_kind(blk) if blk else "cuff",
                             "block_name": blk.name if blk else "cuff", "block_index": int(bi),
                             "coef": float(b), "penalty": float(lamvec[bi]),
                             "scaler_mean": float(mu), "scaler_sd": float(sd),
                             "impute_median": float(med[j]) if med is not None and kind in ("feature", "feature*h")
                             else None,
                             "h_input": h_input, "requires": st.requires})
    cols = ["chain", "node", "model", "column", "block", "block_name", "block_index", "coef", "penalty",
            "scaler_mean", "scaler_sd", "impute_median", "h_input", "requires"]
    return pd.DataFrame(rows).reindex(columns=cols)


def penalty_path_table(fit: ChainFit) -> pd.DataFrame:
    """The registered penalty selection of a chain fitted on all S-dev, as kept by `ridge.refit_chain`
    (`keep_crit`): for every node and stage, the inner-CV criterion (patient-level mean RMSE, mmHg,
    §4.5) of every candidate ("dropped" and each grid penalty), and the one chosen. A by-product of
    the registered selection, not a pre-specified output."""
    rows = []
    for node, nf in fit.nodes.items():
        if len(nf.crit) != len(fit.spec.stages):
            raise RuntimeError(f"chain {fit.spec.name}, node {node}: no selection path kept (keep_crit)")
        for s, crit in enumerate(nf.crit):
            blk = fit.spec.blocks[s - 1] if s else None
            chosen = "dropped" if nf.lam[s] is DROPPED else nf.lam[s]
            if chosen not in crit:
                raise RuntimeError(f"chain {fit.spec.name}, node {node}: chosen penalty not on the path")
            for cand, v in crit.items():
                rows.append({"chain": fit.spec.name, "node": node, "model": fit.spec.stages[s],
                             "block": block_kind(blk) if blk else "cuff", "block_name": blk.name if blk else "cuff",
                             "candidate": cand, "inner_cv_patient_mean_rmse": float(v),
                             "chosen": cand == chosen})
    return pd.DataFrame(rows)


def penalty_table(cv: CVResult) -> pd.DataFrame:
    rows = []
    for j, nodes in enumerate(cv.lam):
        for node, lam in nodes.items():
            for s, l in enumerate(lam):
                rows.append({"chain": cv.spec.name, "seed": cv.seed, "outer_fold": j, "node": node,
                             "model": cv.spec.stages[s], "penalty": "dropped" if l is None else l})
    return pd.DataFrame(rows)


def dropped_frequency(cv: CVResult) -> pd.DataFrame:
    t = penalty_table(cv)
    t = t.loc[t["model"] != cv.spec.stages[0]]
    t["dropped"] = t["penalty"] == "dropped"
    return t.groupby(["chain", "node", "model"], as_index=False)["dropped"].agg(["sum", "count"]).rename(
        columns={"sum": "n_folds_dropped", "count": "n_folds"})


# ----------------------------------------------------------------------------
# Data quality (§6.7)
# ----------------------------------------------------------------------------

def data_quality(tables, frame) -> dict:
    """Exclusions (SPEC §2.2: not analysable; analysable with a recorded neuromuscular blocker), flagged
    gaps, the missing share of each feature group, and coverage. Readings are those of the analysed cases."""
    a = tables.analysed
    cnt = tables.counts()
    r = tables.readings.loc[tables.readings["caseid"].isin(a["caseid"])]
    out = {"n_cases": int(len(tables.cases)), "n_analysable": cnt["n_analysable_cases"],
           "n_nmb_recorded": cnt["n_nmb_recorded_cases"], "n_analysed": cnt["n_analysed_cases"],
           "nmb_recorded_caseids": cnt["nmb_recorded_caseids"],
           "not_analysable_reasons": tables.cases.loc[~tables.cases["analysable"].astype(bool),
                                                      "not_analysable_reason"].value_counts().to_dict()
           if "not_analysable_reason" in tables.cases else {},
           "readings_analysed_cases": int(len(r)),
           "readings_dropped_by_reason": r["drop_reason"].value_counts().to_dict() if "drop_reason" in r else {},
           "gaps_flagged": int(r["gap_flag"].sum()) if "gap_flag" in r else None,
           "n_primary_forecasts": int(len(frame))}
    hr = frame[[f"{c}@real" for c in HR_COLS]]
    out["hr_missing_share_any"] = float(hr.isna().any(axis=1).mean())
    out["hr_missing_share_by_column"] = {c: float(hr[f"{c}@real"].isna().mean()) for c in HR_COLS}
    wa = frame["wf_available@real"]
    out["waveform_coverage"] = float(wa.mean())
    out["waveform_missing_share_by_column_given_available"] = {
        c: float(frame.loc[wa, f"{c}@real"].isna().mean()) if wa.any() else np.nan for c in WF_COLS}
    out["p1_waveform_coverage"] = float(frame["wf_available@p1"].mean())
    return out
