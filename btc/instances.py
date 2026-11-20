"""Windows, baselines, cohort rules, issue points, targets and cuff features
(SPEC §2.2, §2.5, §3, §4.1, §6.6, §6.7).

Cohort rules (§2.2): `analysable` is the signal and cuff rule; `analysed` additionally excludes a
case with a recorded neuromuscular blocker (`nmb_recorded`). Instances, features and P1 donors are
built for every analysable case, so that a sensitivity analysis can include the excluded cases.

Every feature of an issue point is computed from the valid readings 1..issue_n
only (§3: cuff features use readings up to and including the issue reading);
`cuff_features` slices its inputs before any computation, and the leakage test
(`tests/test_leakage_cuff_hr.py`, §4.4) checks it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import FAST_CYCLING_S, MAX_H_S, PRIMARY_END_S, SECONDARY_CAP_S
from .constants import H_BIN_EDGES_S as H_BIN_EDGES
from .constants import SEED as DONOR_SEED     # §10: seed for placebo donors
from .cuff import valid_readings

MIN_VALID_PRIMARY = 6       # §2.2: >= 6 valid readings in the primary window
FIRST_ISSUE_N = 3           # §3: issue points from the third valid reading of the record
BASELINE_N = 3              # §2.5: record-start baseline = mean of the first 3 valid readings
K_MAX = 3                   # §3: next 1, 2 and 3 readings

WINDOWS = ("primary", "secondary")

CUFF_FEATURES = [
    "t_issue", "t_prev", "h_med_prev_s",
    "last_map", "last_sbp", "last_dbp", "base_map", "base_sbp", "run_mean_map",
    "cuff_level", "cuff_slope", "cuff_curv", "cuff_pp", "cuff_tsr_min",
    "cuff_lag0", "cuff_lag1", "cuff_lag2", "cuff_gap1_min", "cuff_gap2_min",
    "cuff_sbp_level", "cuff_sbp_slope", "cuff_sbp_curv",
    "cuff_sbp_lag0", "cuff_sbp_lag1", "cuff_sbp_lag2",
]
TARGET_COLUMNS = ["k", "target_n", "t_target", "t_next", "h_s", "h_min", "h_bin", "y_map", "y_sbp", "y_dbp"]
INSTANCE_COLUMNS = (["caseid", "window", "issue_n", "subjectid", "fast_cycling"]
                    + TARGET_COLUMNS + [c for c in CUFF_FEATURES])


def window_bounds(window: str, opstart: float) -> tuple[float, float] | None:
    """(start, end) of a window in seconds, both ends included (§2.5)."""
    if window == "primary":
        return 0.0, PRIMARY_END_S
    if window == "secondary":
        if opstart is None or not np.isfinite(opstart):
            return None                         # [judgment] no incision time -> no secondary window
        return 0.0, float(min(max(opstart, 0.0), SECONDARY_CAP_S))
    raise ValueError(window)


def h_bin(h_s):
    """1-4 for the reporting bins of §3 (lower edge in, upper out; last bin includes 400 s)."""
    h = np.asarray(h_s, dtype=float)
    e = H_BIN_EDGES
    out = np.full(h.shape, 0, dtype=int)
    out[(h >= e[0]) & (h < e[1])] = 1
    out[(h >= e[1]) & (h < e[2])] = 2
    out[(h >= e[2]) & (h < e[3])] = 3
    out[(h >= e[3]) & (h <= e[4])] = 4
    return out


# ---------------------------------------------------------------------------
# Cuff features (M00, M0, M1, M2 forms (a) and (b), SBP analogs)
# ---------------------------------------------------------------------------

def _ls_slope(t_min, y):
    """Least-squares slope through the points (§4.1, M1/M2), per minute."""
    tm = t_min - t_min.mean()
    return float(np.sum(tm * (y - y.mean())) / np.sum(tm * tm))


def _parabola_curv(t_min, y):
    """Second derivative of the parabola through three points (per minute^2)."""
    (t1, t2, t3), (y1, y2, y3) = t_min, y
    return float(2.0 * ((y3 - y2) / (t3 - t2) - (y2 - y1) / (t2 - t1)) / (t3 - t1))


def cuff_features(t, mp, sbp, dbp, n: int) -> dict:
    """Cuff features at issue reading n (1-based index among valid readings).

    Inputs are the arrays of *valid* readings of the record in order. Only
    entries 0..n-1 (readings 1..n) are read.
    """
    if n < FIRST_ISSUE_N:
        raise ValueError("issue points start at the third valid reading (§3)")
    t = np.asarray(t, dtype=float)[:n]
    mp = np.asarray(mp, dtype=float)[:n]
    sbp = np.asarray(sbp, dtype=float)[:n]
    dbp = np.asarray(dbp, dtype=float)[:n]
    base_map = float(mp[:BASELINE_N].mean())      # §2.5 record-start baseline
    base_sbp = float(sbp[:BASELINE_N].mean())
    t3 = t[-3:] / 60.0                            # last three valid readings, minutes
    m3, s3 = mp[-3:], sbp[-3:]
    gaps = np.diff(t)
    f = {
        "t_issue": float(t[-1]),
        "t_prev": float(t[-2]),
        "h_med_prev_s": float(np.median(gaps)),   # §3 sensitivity: median of the patient's previous intervals
        "last_map": float(mp[-1]), "last_sbp": float(sbp[-1]), "last_dbp": float(dbp[-1]),
        "base_map": base_map, "base_sbp": base_sbp,
        "run_mean_map": float(mp.mean()),         # M00: causal running mean of readings 1..n (§4.1)
        # M2 form (a) (§4.1)
        "cuff_level": float(mp[-1] - base_map),
        "cuff_slope": _ls_slope(t3, m3),
        "cuff_curv": _parabola_curv(t3, m3),
        "cuff_pp": float(sbp[-1] - dbp[-1]),
        "cuff_tsr_min": float(t[-1] / 60.0),      # time since record start (t = 0)
        # M2 form (b): raw-lag AR(3) with the two gaps (§4.1)
        "cuff_lag0": float(mp[-1] - base_map),
        "cuff_lag1": float(mp[-2] - base_map),
        "cuff_lag2": float(mp[-3] - base_map),
        "cuff_gap1_min": float((t[-1] - t[-2]) / 60.0),
        "cuff_gap2_min": float((t[-2] - t[-3]) / 60.0),
        # SBP analogs for the SBP models (§6.7)
        "cuff_sbp_level": float(sbp[-1] - base_sbp),
        "cuff_sbp_slope": _ls_slope(t3, s3),
        "cuff_sbp_curv": _parabola_curv(t3, s3),
        "cuff_sbp_lag0": float(sbp[-1] - base_sbp),
        "cuff_sbp_lag1": float(sbp[-2] - base_sbp),
        "cuff_sbp_lag2": float(sbp[-3] - base_sbp),
    }
    return f


# ---------------------------------------------------------------------------
# Issue points and targets (§3)
# ---------------------------------------------------------------------------

def issue_points(valid: pd.DataFrame, bounds: tuple[float, float]) -> pd.DataFrame:
    """Issue points (§3): valid readings with t_r in the window, n >= 3."""
    lo, hi = bounds
    t = valid["t"]
    return valid.loc[(valid["n"] >= FIRST_ISSUE_N) & (t >= lo) & (t <= hi)]


def case_instances(caseid: int, subjectid: int, fast_cycling: bool, readings: pd.DataFrame,
                   opstart: float) -> pd.DataFrame:
    """Instance rows (caseid, window, issue_n, k) of one case (§3)."""
    v = valid_readings(readings)
    t = v["t"].to_numpy()
    mp, sbp, dbp = (v[c].to_numpy() for c in ("map", "sbp", "dbp"))
    rows = []
    for window in WINDOWS:
        b = window_bounds(window, opstart)
        if b is None:
            continue
        lo, hi = b
        for n in issue_points(v, b)["n"].to_numpy():
            n = int(n)
            feats = None
            for k in range(1, K_MAX + 1):
                j = n + k                      # target = valid reading n + k
                if j > len(t):
                    break
                tt = t[j - 1]
                h = tt - t[n - 1]
                if not (lo <= tt <= hi and h <= MAX_H_S):
                    break                      # later targets are later still
                if feats is None:
                    feats = cuff_features(t, mp, sbp, dbp, n)
                rows.append({"caseid": caseid, "window": window, "issue_n": n,
                             "subjectid": subjectid, "fast_cycling": fast_cycling,
                             "k": k, "target_n": j, "t_target": tt,
                             "t_next": t[n] if n < len(t) else np.nan,
                             "h_s": h, "h_min": h / 60.0, "h_bin": int(h_bin(h)),
                             "y_map": mp[j - 1], "y_sbp": sbp[j - 1], "y_dbp": dbp[j - 1],
                             **feats})
    return pd.DataFrame(rows, columns=INSTANCE_COLUMNS)


# ---------------------------------------------------------------------------
# Cohort rules (§2.2) and the P1 donor map (§6.6)
# ---------------------------------------------------------------------------

def primary_issue_span(readings: pd.DataFrame) -> tuple[float, float]:
    """(first, last) issue time in the primary window by the §3 definition
    (valid readings from the third on, with or without a target).
    [judgment] Used for P1 donor coverage (§6.6)."""
    ip = issue_points(valid_readings(readings), window_bounds("primary", None))
    if len(ip) == 0:
        return float("nan"), float("nan")
    return float(ip["t"].iloc[0]), float(ip["t"].iloc[-1])


def analysable_reasons(pleth_present: bool, ecg_present: bool, n_valid_primary: int,
                       subject_in_sob: bool) -> list[str]:
    """Reasons a case is not analysable (§2.2), in a fixed order; empty if analysable."""
    r = []
    if subject_in_sob:
        r.append("subject_also_in_sob")      # §2.2: S-dev case excluded from fitting
    if not pleth_present:
        r.append("pleth_absent")
    if not ecg_present:
        r.append("ecg_absent")
    if n_valid_primary < MIN_VALID_PRIMARY:
        r.append("lt6_valid_readings_primary")
    return r


# SPEC §2.2: VitalDB case-table totals of the two neuromuscular blockers (mg)
NMB_COLUMNS = ("intraop_rocu", "intraop_vecu")


def nmb_recorded(*totals) -> bool:
    """SPEC §2.2: a neuromuscular blocker is recorded if the case-table rocuronium or vecuronium
    total is > 0. Each total is compared with 0 on its own; a missing total counts as not recorded."""
    return any(x is not None and pd.notna(x) and float(x) > 0 for x in totals)


def nmb_columns(case_table_row, analysable: bool) -> dict:
    """`nmb_recorded` and `analysed` (SPEC §2.2) of one case. `analysable` keeps its meaning (signal
    and cuff rule); `analysed` = analysable and no neuromuscular blocker recorded. Only analysed
    cases are fitted or scored; instances, features and P1 donors stay those of all analysable cases."""
    nmb = nmb_recorded(*(case_table_row[c] for c in NMB_COLUMNS))
    return {"nmb_recorded": bool(nmb), "analysed": bool(analysable and not nmb)}


def assign_donors(cases: pd.DataFrame, seed: int = DONOR_SEED) -> pd.Series:
    """P1 donor for every analysable case (§6.6; PREREG §6). Donors are drawn from all analysable
    cases, with or without a recorded neuromuscular blocker (SPEC §2.2).

    The analysable cases, in ascending case-ID order, are permuted with
    `numpy.random.default_rng(seed).permutation` [judgment: generator and input
    order]. A recipient's donor is the next case after it in the permutation,
    wrapping around, that belongs to another subject and whose HR, pleth and
    ECG tracks start no later than the recipient's first issue time and last at
    least until its last issue time. NaN if no case qualifies.

    `cases` needs: caseid, subjectid, analysable, first_issue_t, last_issue_t,
    hr_t_first, hr_t_last, pleth_t_first, pleth_t_last, ecg_t_first, ecg_t_last.
    """
    a = cases.loc[cases["analysable"]].sort_values("caseid")
    ids = a["caseid"].to_numpy()
    perm = np.random.default_rng(seed).permutation(ids)
    rec = a.set_index("caseid")
    donor = pd.Series(np.nan, index=cases["caseid"].to_numpy(), dtype=float, name="donor_caseid")
    m = len(perm)
    for i, cid in enumerate(perm):
        r = rec.loc[cid]
        t0, t1 = r["first_issue_t"], r["last_issue_t"]
        for step in range(1, m):
            d = perm[(i + step) % m]
            dr = rec.loc[d]
            if dr["subjectid"] == r["subjectid"]:
                continue
            ok = all(dr[f"{s}_t_first"] <= t0 and dr[f"{s}_t_last"] >= t1 for s in ("hr", "pleth", "ecg"))
            if ok:
                donor[cid] = float(d)
                break
    return donor


# ---------------------------------------------------------------------------
# One case: readings and the case-table row
# ---------------------------------------------------------------------------

def case_readings(case) -> tuple[pd.DataFrame, float]:
    """Readings of a loaded case with gap flags (§2.4), and the case median
    interval in the primary window. [judgment] "The case's median interval" for
    gap flags is the primary-window median, the same quantity that defines the
    cycling class (§4.1, §6.7)."""
    from .cuff import flag_gaps, median_interval, readings_from_case
    r = readings_from_case(case)
    med = median_interval(r, 0.0, PRIMARY_END_S)
    return flag_gaps(r, med), med


def parse_age(x) -> tuple[float, bool]:
    """(age in years, censored). [judgment] A censored entry such as ">89" gives its bound and
    True; an unreadable entry gives (nan, False)."""
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in "><":
            try:
                return float(s[1:].strip()), True
            except ValueError:
                return float("nan"), False
        x = s
    try:
        return float(x), False
    except (TypeError, ValueError):
        return float("nan"), False


# Vasopressor strata (SPEC §2.6; descriptive, S-ob)
VASO_PUMP, VASO_BOLUS, VASO_NEITHER = "pump", "bolus_no_pump", "neither_recorded"


def vaso_stratum(pump_present: bool, *bolus_totals) -> str:
    """SPEC §2.6 stratum: pump track present; else a recorded bolus total > 0; else neither.
    Each total is compared with 0 on its own: phenylephrine (µg) and ephedrine (mg) are never
    added. A missing total counts as not recorded."""
    if pump_present:
        return VASO_PUMP
    if any(np.isfinite(float(x)) and float(x) > 0 for x in bolus_totals if x is not None):
        return VASO_BOLUS
    return VASO_NEITHER


def vaso_columns(case, cohort_row) -> dict:
    """S-ob vasopressor columns (SPEC §2.6): `pump_present` (`Orchestra/PHEN_RATE` has a record in
    the case file), the chart totals with their units, and the strata with phenylephrine or
    ephedrine (`vaso_stratum`) and with phenylephrine alone (`vaso_stratum_phe`)."""
    pump = len(case.phen) > 0
    phe = float(cohort_row["intraop_phe"]) if pd.notna(cohort_row["intraop_phe"]) else np.nan
    eph = float(cohort_row["intraop_eph"]) if pd.notna(cohort_row["intraop_eph"]) else np.nan
    return {"pump_present": bool(pump), "intraop_phe_ug": phe, "intraop_eph_mg": eph,
            "vaso_stratum": vaso_stratum(pump, phe, eph), "vaso_stratum_phe": vaso_stratum(pump, phe)}


def case_row(case, cohort_row, readings: pd.DataFrame, median_primary_s: float,
             subject_in_sob: bool) -> dict:
    """One `<cohort>_cases` row (btc/SCHEMA.md §1) for a loaded case."""
    v = valid_readings(readings)
    in_p = (readings["t"] >= 0.0) & (readings["t"] <= PRIMARY_END_S)
    n_valid_primary = int((readings["valid"] & in_p).sum())
    pleth_present, ecg_present = case.pleth is not None, case.ecg is not None
    reasons = analysable_reasons(pleth_present, ecg_present, n_valid_primary, subject_in_sob)
    t_first, t_last = primary_issue_span(readings)
    ext = {"hr": case.hr.extent(),
           "pleth": case.pleth.extent() if pleth_present else (np.nan, np.nan),
           "ecg": case.ecg.extent() if ecg_present else (np.nan, np.nan)}
    row = {
        "caseid": int(case.caseid),
        "subjectid": int(cohort_row["subjectid"]),
        "cohort": str(cohort_row["cohort"]),
        "cerclage": bool(cohort_row["cerclage"]),
        "opstart": float(cohort_row["opstart"]),
        "analysable": len(reasons) == 0,
        "not_analysable_reason": ";".join(reasons),
        "n_valid_primary": n_valid_primary,
        "median_interval_primary_s": median_primary_s,
        # §6.7: fast cycling = primary-window median interval < 90 s (False if undefined)
        "fast_cycling": bool(np.isfinite(median_primary_s) and median_primary_s < FAST_CYCLING_S),
        "pleth_present": pleth_present, "ecg_present": ecg_present,
        "pleth_status": case.pleth_status, "ecg_status": case.ecg_status,
        "waveform_flag": case.waveform_flag,
        "hr_t_first": ext["hr"][0], "hr_t_last": ext["hr"][1],
        "pleth_t_first": ext["pleth"][0], "pleth_t_last": ext["pleth"][1],
        "ecg_t_first": ext["ecg"][0], "ecg_t_last": ext["ecg"][1],
        "first_issue_t": t_first, "last_issue_t": t_last,
        "n_issue_primary": int(len(issue_points(v, window_bounds("primary", None)))),
        # exclusion and gap counts (§2.4); drop counts by first failing rule
        "n_readings_all": int(len(readings)),
        "n_valid_all": int(readings["valid"].sum()),
        "n_readings_primary": int(in_p.sum()),
    }
    for reason in ("incomplete", "map_range", "order"):
        row[f"n_drop_{reason}_all"] = int((readings["drop_reason"] == reason).sum())
        row[f"n_drop_{reason}_primary"] = int(((readings["drop_reason"] == reason) & in_p).sum())
    row["n_gaps_all"] = int(readings["gap_flag"].sum())
    row["n_gaps_primary"] = int((readings["gap_flag"] & in_p).sum())
    return row
