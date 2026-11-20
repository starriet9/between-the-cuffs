"""The registered model chains and their cross-validated runs (SPEC §3, §4, §6.6, §6.7).

Every ridge model is fitted through `ridge.fit_chain` (penalties as in §4.5).
Chains that share stage 0 (M2) re-derive it with the same data and folds, so
its penalty is identical across chains ("the cuff block keeps M2's penalty").
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .bootstrap import Contrast, SubjectBootstrap, contrast, mean_ci, percentile_ci
from .constants import ABLATION_GROUPS, MIN_FORECASTS_RMSE, MIN_TRIPLES, N_BOOT, SEED
from .data import AVAIL_COLS
from .design import Block, Cols
from .metrics import per_patient, summarise
from .ridge import ChainSpec, CVResult, cv_chain, hr_block, refit_chain, wf_block

M2_COMBOS = (("a", "perbin"), ("a", "pooled"), ("b", "perbin"), ("b", "pooled"))  # SPEC §4.1


# ----------------------------------------------------------------------------
# Chains
# ----------------------------------------------------------------------------

def m2_chain(form, structure, **kw) -> ChainSpec:
    return ChainSpec("M2", ("M2",), form=form, structure=structure, **kw)


def ladder(form, structure, variant="real", wf_variant=None, names=("M2", "M3", "M4"), **kw) -> ChainSpec:
    """M2 -> M3 (HR block) -> M4 (waveform block, fitted on rows with waveform) (§4.1, §4.5).
    `variant` selects real or placebo signals (§6.6); `wf_variant` the waveform variant if it differs
    (template-correlation quality variant: HR real, waveform strict)."""
    hcol = kw.get("hcol", "h_min")
    wv = wf_variant or variant
    return ChainSpec(f"ladder:{variant}/{wv}", names, form=form, structure=structure,
                     blocks=(hr_block(variant, hcol), wf_block(wv, hcol)), **kw)


def ablation_chain(form, structure, group, **kw) -> ChainSpec:
    """M2 + one feature group with its own penalty (§6.6 item 4) [judgment: waveform groups are fitted
    on rows with waveform, other rows get M2, as for M4; the availability group on every row]."""
    hcol = kw.get("hcol", "h_min")
    if group == "hr":
        blk = hr_block("real", hcol, name="abl:hr")
    elif group == "avail":
        blk = Block("abl:avail", AVAIL_COLS, None, hcol, kind="indicator")
    else:
        blk = wf_block("real", hcol, cols=ABLATION_GROUPS[group], name=f"abl:{group}")
    return ChainSpec(f"ablation:{group}", ("M2", f"M2+{group}"), form=form, structure=structure,
                     blocks=(blk,), **kw)


def yard_T_chain(form, variant="real", names=("L", "LH", "T")) -> ChainSpec:
    """Yardstick L -> L + HR -> T (§3): pooled, M2's cuff features at r with h = t_{r+2} - t_r plus the
    elapsed time; HR and waveform features at issue r+1 interact with the remaining horizon."""
    return ChainSpec(f"yard:{variant}", names, form=form, structure="pooled", hcol="h_min",
                     extra_cuff=("elapsed_min",),
                     blocks=(hr_block(variant, "h2_min"), wf_block(variant, "h2_min")))


def yard_U_chain(form) -> ChainSpec:
    """Yardstick U: M2 issued at t_{r+1} (§3), pooled structure."""
    return ChainSpec("yard:U", ("U",), form=form, structure="pooled")


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

def _cv_job(F, spec, plan):
    with threadpool_limits(1):
        return cv_chain(F, spec, plan)


def _refit_job(F, spec, plan):
    with threadpool_limits(1):
        return refit_chain(F, spec, plan)


def run_cv(jobs: dict, n_jobs: int = 1) -> dict:
    """jobs: key -> (Cols, ChainSpec, FoldPlan). Returns key -> CVResult (deterministic)."""
    keys = list(jobs)
    if n_jobs == 1:
        res = [_cv_job(*jobs[k]) for k in keys]
    else:
        res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_cv_job)(*jobs[k]) for k in keys)
    return dict(zip(keys, res))


def run_refit(jobs: dict, n_jobs: int = 1) -> dict:
    keys = list(jobs)
    if n_jobs == 1:
        res = [_refit_job(*jobs[k]) for k in keys]
    else:
        res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_refit_job)(*jobs[k]) for k in keys)
    return dict(zip(keys, res))


def baseline_preds(frame: pd.DataFrame) -> dict:
    """M00 (causal running mean), M0 (persistence), M1 (last + h x slope through the last three
    readings) (§4.1). Not fitted."""
    return {"M00": frame["run_mean_map"].to_numpy(dtype=float),
            "M0": frame["last_map"].to_numpy(dtype=float),
            "M1": (frame["last_map"] + frame["h_min"] * frame["cuff_slope"]).to_numpy(dtype=float)}


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

class Scored:
    """Per-patient metrics of several models on the same forecasts, and their paired bootstrap.

    `mask` restricts scoring (subsets, h-bins, strata, subgroups reuse the predictions, §4.5).
    A patient enters with >= `min_n` scored forecasts in the analysis at hand (§6.1)."""

    def __init__(self, frame: pd.DataFrame, preds: dict, mask=None, target: str = "map",
                 min_n: int = MIN_FORECASTS_RMSE, n_boot: int = N_BOOT, seed: int = SEED):
        m = np.ones(len(frame), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.frame = frame.loc[m]
        self.y = self.frame[f"y_{target}"].to_numpy(dtype=float)
        self.caseid = self.frame["caseid"].to_numpy().astype(int)
        self.pp = {k: per_patient(self.caseid, self.y, np.asarray(v, dtype=float)[m]) for k, v in preds.items()}
        any_pp = next(iter(self.pp.values())) if self.pp else pd.DataFrame({"n": []})
        enter = any_pp.index[any_pp["n"] >= min_n].to_numpy()
        self.enter = np.sort(enter)
        subj = self.frame.drop_duplicates("caseid").set_index("caseid")["subjectid"]
        self.subject = subj.reindex(self.enter).to_numpy()
        self.n_patients = len(self.enter)
        self.n_forecasts = int(np.isin(self.caseid, self.enter).sum())
        self.bs = SubjectBootstrap(self.enter, self.subject, n_boot, seed) if self.n_patients else None

    def rmse(self, model: str) -> np.ndarray:
        return self.pp[model].loc[self.enter, "rmse"].to_numpy()

    def contrast(self, model: str, comp: str) -> Contrast:
        if not self.n_patients:
            nan = float("nan")
            return Contrast(model, comp, 0, nan, nan, nan, nan, nan, nan, nan, nan)
        return contrast(self.bs, self.rmse(model), self.rmse(comp), model, comp)

    def summary(self) -> pd.DataFrame:
        rows = []
        for k, pp in self.pp.items():
            s = summarise(pp)
            if self.n_patients:
                mu, lo, hi = mean_ci(self.bs, self.rmse(k))
                sse = pp.loc[self.enter, "sse"].to_numpy()
                n = pp.loc[self.enter, "n"].to_numpy()
                pr = np.sqrt(self.bs.ratio(sse, n))
                plo, phi = percentile_ci(pr)
                s.update(rmse_mean_ci_lo=lo, rmse_mean_ci_hi=hi, pooled_rmse_entering=float(
                    np.sqrt(sse.sum() / n.sum())), pooled_rmse_ci_lo=plo, pooled_rmse_ci_hi=phi)
            rows.append({"model": k, **s})
        return pd.DataFrame(rows)

    def per_patient_long(self) -> pd.DataFrame:
        out = []
        for k, pp in self.pp.items():
            d = pp.reset_index()
            d.insert(1, "model", k)
            out.append(d)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def choose_m2(frame: pd.DataFrame, cvs: dict) -> tuple:
    """M2 structure (§4.1): lowest patient-level mean RMSE of the outer-CV M2 forecasts among the four
    combinations [judgment: patients with >= 3 forecasts, as in §6.1; exact ties go to the first
    combination in the order (a, per-bin), (a, pooled), (b, per-bin), (b, pooled)]."""
    sc = Scored(frame, {f"{f}/{s}": cvs[(f, s)].stage("M2") for f, s in M2_COMBOS}, n_boot=1)
    vals = [float(np.mean(sc.rmse(f"{f}/{s}"))) for f, s in M2_COMBOS]
    i = int(np.argmin(vals))
    table = pd.DataFrame({"form": [c[0] for c in M2_COMBOS], "structure": [c[1] for c in M2_COMBOS],
                          "patient_mean_rmse": vals, "chosen": [j == i for j in range(4)]})
    return M2_COMBOS[i], table


def yardstick_scores(tri: pd.DataFrame, preds: dict, mask=None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """N, D, F and the placebo guard (§3) on the triples of `mask`; >= 3 scored triples per patient."""
    from .bootstrap import fraction_F
    sc = Scored(tri, preds, mask=mask, min_n=MIN_TRIPLES, n_boot=n_boot, seed=seed)
    out = {"n_patients": sc.n_patients, "n_triples": sc.n_forecasts}
    if not sc.n_patients:
        return out
    out.update(fraction_F(sc.bs, sc.rmse("L"), sc.rmse("T"), sc.rmse("U")))
    for m in preds:
        out[f"rmse_{m}"] = float(np.mean(sc.rmse(m)))
    if "T-P1" in preds:
        g = sc.contrast("T", "T-P1")
        out["T_minus_TP1"] = g.as_dict()
        out["placebo_guard_passed"] = bool(g.delta_hi < 0)       # CI of RMSE_T - RMSE_T-P1 below zero
    if "T-ia" in preds:
        from .bootstrap import fraction_F as ff
        out["ia"] = ff(sc.bs, sc.rmse("L"), sc.rmse("T-ia"), sc.rmse("U"))
    return out
