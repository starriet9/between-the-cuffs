"""Derived tables (btc/SCHEMA.md) and the analysis frames built from them.

A frame has one row per forecast. Signal features of variant v are joined from
the HR and waveform tables as `<column>@<v>`; a missing feature row means every
feature of that variant is missing and `wf_available@<v>` is False [judgment].
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (H_BIN_EDGES_S, HR_COLS, VARIANTS_HR, VARIANTS_WF, WF_COLS)

TABLES = ("cases", "readings", "instances", "features_hr", "features_wf")

REQUIRED = {
    "cases": ["caseid", "subjectid", "analysable", "nmb_recorded", "fast_cycling"],
    "readings": ["caseid", "t", "valid"],
    "instances": ["caseid", "window", "issue_n", "k", "subjectid", "t_issue", "t_prev", "t_target",
                  "h_s", "h_min", "h_med_prev_s", "y_map", "y_sbp", "last_map", "last_sbp", "base_map",
                  "base_sbp", "run_mean_map", "cuff_level", "cuff_slope", "cuff_curv", "cuff_pp",
                  "cuff_tsr_min", "cuff_lag0", "cuff_lag1", "cuff_lag2", "cuff_gap1_min", "cuff_gap2_min",
                  "cuff_sbp_level", "cuff_sbp_slope", "cuff_sbp_curv", "cuff_sbp_lag0", "cuff_sbp_lag1",
                  "cuff_sbp_lag2"],
    "features_hr": ["caseid", "window", "issue_n", "variant", *HR_COLS],
    "features_wf": ["caseid", "window", "issue_n", "variant", *WF_COLS, "wf_n_windows", "wf_n_usable",
                    "wf_usable_share", "wf_available"],
}

WF_QUALITY = ("wf_n_windows", "wf_n_usable", "wf_usable_share")
INSTANCE_KEEP = ["caseid", "subjectid", "window", "issue_n", "k", "t_issue", "t_prev", "t_target", "h_s",
                 "h_min", "h_med_prev_s", "y_map", "y_sbp", "y_dbp", "last_map", "last_sbp", "last_dbp",
                 "base_map", "base_sbp", "run_mean_map", "cuff_level", "cuff_slope", "cuff_curv", "cuff_pp",
                 "cuff_tsr_min", "cuff_lag0", "cuff_lag1", "cuff_lag2", "cuff_gap1_min", "cuff_gap2_min",
                 "cuff_sbp_level", "cuff_sbp_slope", "cuff_sbp_curv", "cuff_sbp_lag0", "cuff_sbp_lag1",
                 "cuff_sbp_lag2"]
# case columns carried into the frames; the optional ones may be absent
CASE_KEEP = ["fast_cycling", "nmb_recorded", "sex", "cerclage", "opstart", "vaso_stratum"]


def _nmb(cases: pd.DataFrame) -> pd.Series:
    """`nmb_recorded` as bool; a missing value counts as not recorded (SPEC §2.2)."""
    return cases["nmb_recorded"].eq(True).fillna(False).astype(bool)


@dataclass
class Tables:
    cohort: str                      # "sdev" or "sob"
    cases: pd.DataFrame
    readings: pd.DataFrame
    instances: pd.DataFrame
    features_hr: pd.DataFrame
    features_wf: pd.DataFrame
    sha256: dict
    include_nmb: bool = False        # SPEC §2.2 sensitivity analysis: also fit and score nmb_recorded cases

    @property
    def analysable(self) -> pd.DataFrame:
        """Analysable cases (§2.2 signal and cuff rule), ascending case ID, with or without a recorded
        neuromuscular blocker. P1 donors come from these; fitting and scoring use `analysed`."""
        c = self.cases.loc[self.cases["analysable"].astype(bool)].sort_values("caseid")
        return c.reset_index(drop=True)

    @property
    def analysed(self) -> pd.DataFrame:
        """Cases that are fitted and scored (SPEC §2.2), ascending case ID: analysable cases without a
        recorded neuromuscular blocker, or all analysable cases when `include_nmb` is set."""
        keep = self.cases["analysable"].astype(bool)
        if not self.include_nmb:
            keep &= ~_nmb(self.cases)
        return self.cases.loc[keep].sort_values("caseid").reset_index(drop=True)

    def counts(self) -> dict:
        """Case counts for reports (SPEC §2.2): analysable, nmb_recorded among them, analysed (by this
        table's rule), and the nmb_recorded analysable case IDs."""
        a = self.analysable
        nmb = a.loc[_nmb(a), "caseid"].astype(int).tolist()
        return {"n_analysable_cases": int(len(a)), "n_nmb_recorded_cases": len(nmb),
                "n_analysed_cases": int(len(self.analysed)), "nmb_recorded_caseids": nmb,
                "include_nmb": bool(self.include_nmb)}


def validate(tables: dict) -> None:
    missing = {name: [c for c in cols if c not in tables[name].columns] for name, cols in REQUIRED.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise ValueError(f"derived tables lack required columns (btc/SCHEMA.md): {missing}")
    c = tables["cases"]
    if "analysed" in c.columns:          # the builder's column must follow the §2.2 rule
        rule = c["analysable"].astype(bool) & ~_nmb(c)
        if not np.array_equal(c["analysed"].eq(True).to_numpy(), rule.to_numpy()):
            raise ValueError("cases.analysed differs from analysable and not nmb_recorded (SPEC §2.2)")


def load_tables(data_dir: Path, cohort: str) -> Tables:
    """Read `<cohort>_<table>.parquet` from `data_dir` (btc/SCHEMA.md)."""
    data_dir = Path(data_dir)
    out, sha = {}, {}
    for name in TABLES:
        p = data_dir / f"{cohort}_{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing derived table {p.name}")
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        out[name] = pd.read_parquet(p)
    validate(out)
    return Tables(cohort=cohort, sha256=sha, **out)


def tables_from_frames(cohort: str, frames: dict) -> Tables:
    validate(frames)
    sha = {f"{cohort}_{k}": hashlib.sha256(pd.util.hash_pandas_object(v, index=True).values.tobytes())
           .hexdigest() for k, v in frames.items()}
    return Tables(cohort=cohort, sha256=sha, **{k: frames[k] for k in TABLES})


def h_bin_of(h_s) -> np.ndarray:
    """Reporting bin 0..3 (§3, §4.5): lower edge in, upper out; last bin includes 400 s; -1 outside."""
    h = np.asarray(h_s, dtype=float)
    e = H_BIN_EDGES_S
    b = np.full(h.shape, -1, dtype=int)
    for i in range(4):
        hi_ok = (h <= e[i + 1]) if i == 3 else (h < e[i + 1])
        b[(h >= e[i]) & hi_ok] = i
    return b


def _signal_wide(t: Tables, window: str) -> pd.DataFrame:
    """HR and waveform features of every variant, one row per (caseid, issue_n)."""
    parts = []
    hr = t.features_hr.loc[t.features_hr["window"] == window]
    for v in VARIANTS_HR:
        d = hr.loc[hr["variant"] == v, ["caseid", "issue_n", *HR_COLS]]
        parts.append(d.set_index(["caseid", "issue_n"]).add_suffix(f"@{v}"))
    wf = t.features_wf.loc[t.features_wf["window"] == window]
    for v in VARIANTS_WF:
        d = wf.loc[wf["variant"] == v, ["caseid", "issue_n", *WF_COLS, *WF_QUALITY, "wf_available"]]
        parts.append(d.set_index(["caseid", "issue_n"]).add_suffix(f"@{v}"))
    for p in parts:
        if p.index.has_duplicates:
            raise ValueError("duplicate (caseid, window, issue_n, variant) feature rows")
    return pd.concat(parts, axis=1, join="outer")


def _attach_signals(frame: pd.DataFrame, wide: pd.DataFrame, issue_col: str) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays([frame["caseid"].to_numpy(), frame[issue_col].to_numpy()])
    sig = wide.reindex(key)
    sig.index = frame.index
    out = pd.concat([frame, sig], axis=1)
    for v in VARIANTS_WF:
        c = f"wf_available@{v}"
        if c not in out.columns:
            out[c] = False
        out[c] = out[c].fillna(False).astype(bool)
    need = [f"{c}@{v}" for v in VARIANTS_HR for c in HR_COLS]
    need += [f"{c}@{v}" for v in VARIANTS_WF for c in (*WF_COLS, *WF_QUALITY)]
    for c in need:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _add_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Availability-indicator group of the ablation (§6.6 item 4): the missing-indicators of the
    real HR and waveform columns and a "no waveform" flag, with no feature values."""
    av = {f"av_miss_{c}": frame[f"{c}@real"].isna().astype(float) for c in (*HR_COLS, *WF_COLS)}
    av["av_nowf"] = (~frame["wf_available@real"]).astype(float)
    return pd.concat([frame, pd.DataFrame(av, index=frame.index)], axis=1)


AVAIL_COLS = tuple([f"av_miss_{c}" for c in (*HR_COLS, *WF_COLS)] + ["av_nowf"])


def _case_cols(t: Tables, frame: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in CASE_KEEP if c in t.cases.columns]
    cc = t.cases.set_index("caseid")[keep]
    add = cc.reindex(frame["caseid"].to_numpy())
    add.index = frame.index
    return pd.concat([frame, add], axis=1)


def build_frame(t: Tables, window: str = "primary", k: int = 1) -> pd.DataFrame:
    """Forecast frame (one row per instance) of the analysed cases (§3; SPEC §2.2: analysable without
    a recorded neuromuscular blocker, unless `t.include_nmb`)."""
    ok = set(t.analysed["caseid"].astype(int))
    inst = t.instances
    inst = inst.loc[(inst["window"] == window) & (inst["k"] == k) & inst["caseid"].isin(ok)]
    inst = inst[[c for c in INSTANCE_KEEP if c in inst.columns]]
    inst = inst.sort_values(["caseid", "issue_n"]).reset_index(drop=True)
    f = _case_cols(t, inst)
    f = _attach_signals(f, _signal_wide(t, window), "issue_n")
    f = _add_availability(f)
    f["h_bin"] = h_bin_of(f["h_s"])
    f["hmed_min"] = f["h_med_prev_s"] / 60.0
    f["hmed_bin"] = h_bin_of(f["h_med_prev_s"])
    return f.copy()


def build_triples(t: Tables, window: str = "primary") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Withheld-cuff yardstick triples (§3): readings r, r+1, r+2 in the window, r >= 3rd reading,
    t_{r+2} - t_r <= 400 s. These are exactly the k = 2 instances.

    Returns (tri, u):
    - `tri` (models L, T, T-P1, T-ia): cuff features at r (from the k = 2 instance at r, which uses
      readings 1..r only), h = t_{r+2} - t_r, elapsed = t_{r+1} - t_r, remaining horizon
      t_{r+2} - t_{r+1}, and the HR/waveform features at issue r+1 (time-based only). No value of
      reading r+1 enters `tri` (§3, §4.4 leakage rule).
    - `u` (model U): the k = 1 instance at issue r+1, row-aligned with `tri`.
    """
    base = build_frame(t, window, k=2)
    base = base[[c for c in base.columns if "@" not in c and not c.startswith("av_")]]
    k1 = t.instances.loc[(t.instances["window"] == window) & (t.instances["k"] == 1)]
    k1 = k1.set_index(["caseid", "issue_n"])
    key_r = pd.MultiIndex.from_arrays([base["caseid"], base["issue_n"]])
    t_r1 = k1["t_target"].reindex(key_r).to_numpy()
    if np.any(~np.isfinite(t_r1)):
        raise ValueError("a triple lacks its k = 1 instance at issue r")
    tri = base.copy()
    tri["t_r1"] = t_r1
    tri["elapsed_min"] = (tri["t_r1"] - tri["t_issue"]) / 60.0
    tri["h2_min"] = (tri["t_target"] - tri["t_r1"]) / 60.0
    tri["issue_r1"] = tri["issue_n"] + 1
    tri = _attach_signals(tri, _signal_wide(t, window), "issue_r1")
    # U: M2 at r+1 (k = 1 instance at issue r+1)
    u = build_frame(t, window, k=1)
    u = u.set_index(["caseid", "issue_n"]).reindex(
        pd.MultiIndex.from_arrays([tri["caseid"], tri["issue_r1"]], names=["caseid", "issue_n"]))
    if u["y_map"].isna().any():
        raise ValueError("a triple lacks its k = 1 instance at issue r+1")
    u = u.reset_index()
    if not np.allclose(u["y_map"].to_numpy(), tri["y_map"].to_numpy()) or \
            not np.allclose(u["t_target"].to_numpy(), tri["t_target"].to_numpy()):
        raise ValueError("U and L targets differ")
    u.index = tri.index
    return tri, u
