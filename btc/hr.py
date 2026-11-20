"""Heart-rate features for M3 (SPEC §4.2) and their placebo/variant versions
(§3 inflation-artefact variant, §6.6 P1 and P2).

From `Solar8000/HR`, for an issue reading at t_r with previous valid reading t_prev:
- level  = median of the valid samples in [t_r - 60, t_r);
- change = level minus the same median in [t_prev - 60, t_prev);
- slope  = least-squares slope of the valid samples in [t_r - 60, t_r), bpm/min.
A 60-s window with fewer than 10 valid samples is missing. Only samples strictly
earlier than t_r (or than `t_end`, see variants) are ever read (§3, §4.4).

Variants (btc/SCHEMA.md §4):
- real: own signal, t_end = t_r.
- p1:   the recipient's t_r and t_prev on the donor's HR (§6.6), t_end = t_r.
- p2:   own signal, t_r and t_prev shifted back by 600 s (§6.6), t_end = t_r - 600.
- ia:   own signal, samples from the 30 s before t_r excluded (§3), t_end = t_r - 30.
        [judgment] The 60-s windows stay anchored at t_r and t_prev; samples at or
        after t_end are dropped from both windows ("signal samples from the 30 s
        before t_{r+1} excluded", §3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import IA_EXCLUDE_S, P2_SHIFT_S  # noqa: F401  (§6.6 P2 shift; §3 ia exclusion)

HR_WINDOW_S = 60.0    # §4.2: 60 s before t_r
HR_MIN_SAMPLES = 10   # §4.2: missing if fewer than 10 valid samples in those 60 s

VARIANTS = ("real", "p1", "p2", "ia")
HR_COLUMNS = ["hr_level", "hr_change", "hr_slope", "hr_n", "hr_n_prev"]
FEATURE_COLUMNS = (["caseid", "window", "issue_n", "variant", "signal_caseid",
                    "t_issue", "t_prev", "time_shift_s", "t_end"] + HR_COLUMNS)


def clean_hr(t, v):
    """Valid HR samples: finite and > 0 bpm, sorted by time, one sample per timestamp.
    [judgment] A value of 0 bpm is not a heart rate (the monitor writes 0 when it
    has none) and is treated as missing, like the format's missing-value code.
    [judgment] Of several valid samples with the same timestamp, the last in file order
    is kept (as for the cuff tracks, `btc.cuff.clean_track`); no S-dev HR track has two
    different values at one timestamp."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(v) & (v > 0)
    t, v = t[ok], v[ok]
    o = np.argsort(t, kind="stable")      # file order kept for equal times
    t, v = t[o], v[o]
    if len(t) > 1:
        keep = np.r_[t[1:] != t[:-1], True]
        t, v = t[keep], v[keep]
    return t, v


def window_stats(t, v, anchor: float, t_end: float):
    """(median, slope bpm/min, n) of the valid samples with
    anchor - 60 <= t < min(anchor, t_end). `t, v` must come from `clean_hr`."""
    hi = min(anchor, t_end)
    i0 = int(np.searchsorted(t, anchor - HR_WINDOW_S, side="left"))
    i1 = int(np.searchsorted(t, hi, side="left"))     # strictly earlier than hi
    n = max(i1 - i0, 0)
    if n < HR_MIN_SAMPLES:
        return float("nan"), float("nan"), n
    tt, vv = t[i0:i1], v[i0:i1]
    tm = (tt - tt.mean()) / 60.0
    den = float(np.sum(tm * tm))
    slope = float(np.sum(tm * (vv - vv.mean())) / den) if den > 0 else float("nan")
    return float(np.median(vv)), slope, n


def hr_features(t, v, t_issue: float, t_prev: float, t_end: float | None = None) -> dict:
    """HR features at one issue point; `t, v` from `clean_hr`. With t_end None,
    samples stop at t_issue. Only samples strictly before min(t_issue, t_end) are read."""
    t_end = t_issue if t_end is None else min(t_end, t_issue)
    lvl, slope, n = window_stats(t, v, t_issue, t_end)
    lvl_prev, _, n_prev = window_stats(t, v, t_prev, t_end)
    if not np.isfinite(lvl):
        # §4.2: the HR group is missing when the issue window has < 10 valid samples
        return {"hr_level": np.nan, "hr_change": np.nan, "hr_slope": np.nan, "hr_n": n, "hr_n_prev": n_prev}
    # [judgment] if only the previous-reading window is short, only `hr_change` is missing
    return {"hr_level": lvl, "hr_change": lvl - lvl_prev, "hr_slope": slope, "hr_n": n, "hr_n_prev": n_prev}


def variant_times(variant: str, t_issue: float, t_prev: float):
    """(anchor issue time, anchor previous time, t_end, time shift) for a variant."""
    if variant in ("real", "p1"):
        return t_issue, t_prev, t_issue, 0.0
    if variant == "p2":
        return t_issue - P2_SHIFT_S, t_prev - P2_SHIFT_S, t_issue - P2_SHIFT_S, -P2_SHIFT_S
    if variant == "ia":
        return t_issue, t_prev, t_issue - IA_EXCLUDE_S, 0.0
    raise ValueError(variant)


def case_hr_features(issues: pd.DataFrame, own_hr, donor_hr=None, donor_caseid=None) -> pd.DataFrame:
    """HR feature rows for one case, all variants.

    `issues`: one row per (caseid, window, issue_n) with t_issue and t_prev.
    `own_hr`, `donor_hr`: (t, v) raw HR samples (donor may be None -> p1 rows missing).
    """
    own = clean_hr(*own_hr)
    don = clean_hr(*donor_hr) if donor_hr is not None else None
    rows = []
    for r in issues.itertuples(index=False):
        for variant in VARIANTS:
            a_issue, a_prev, t_end, shift = variant_times(variant, r.t_issue, r.t_prev)
            if variant == "p1":
                src = don
                sig = float(donor_caseid) if (don is not None and donor_caseid is not None) else np.nan
            else:
                src, sig = own, float(r.caseid)
            if src is None:
                f = {c: np.nan for c in HR_COLUMNS}
            else:
                f = hr_features(src[0], src[1], a_issue, a_prev, t_end)
            rows.append({"caseid": r.caseid, "window": r.window, "issue_n": r.issue_n,
                         "variant": variant, "signal_caseid": sig,
                         "t_issue": r.t_issue, "t_prev": r.t_prev,
                         "time_shift_s": shift, "t_end": t_end, **f})
    out = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    for c in ("hr_n", "hr_n_prev"):
        out[c] = out[c].astype("float64")
    return out
