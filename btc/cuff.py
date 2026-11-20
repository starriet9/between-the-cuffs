"""Cuff readings from the held-value NIBP tracks (SPEC §2.4, final rule).

- A new reading is any time at which SBP, DBP or MBP changes. Each track is
  compared with its own previous non-missing sample, so a value that returns
  unchanged after missing samples is not a change. The first value of a track
  counts as a change at its first sample.
- Changes are merged: a change joins the current cluster if it comes within
  15 s of the previous change, so two readings are always more than 15 s apart.
- The reading time t_r is the last change in the cluster; the reading's values
  are those held at the first sample after t_r.
- A reading is dropped (and counted) if MAP lies outside 30-160 mmHg or if
  SBP > MAP > DBP does not hold.
- Gaps between consecutive valid readings longer than twice the case's median
  interval are flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MERGE_S = 15.0              # §2.4: changes within 15 s of the previous change are merged
MAP_MIN, MAP_MAX = 30.0, 160.0  # §2.4: MAP outside 30-160 mmHg is dropped (ends valid)
GAP_FACTOR = 2.0            # §2.4: gaps > 2x the case median interval are flagged

DROP_INCOMPLETE = "incomplete"   # a value is missing, so SBP > MAP > DBP cannot hold
DROP_MAP_RANGE = "map_range"
DROP_ORDER = "order"

CUFF_KEYS = ("sbp", "dbp", "mbp")
READING_COLUMNS = ["n_all", "t", "sbp", "dbp", "map", "valid", "drop_reason",
                   "fail_incomplete", "fail_map_range", "fail_order", "n", "gap_flag", "gap_s"]


def clean_track(t, v):
    """Drop missing samples (NaN), sort by time (stable) and collapse samples that
    share a timestamp, keeping the last in file order ([judgment]; in S-dev every
    such pair holds identical values)."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    if len(t) > 1:
        keep = np.r_[t[1:] != t[:-1], True]
        t, v = t[keep], v[keep]
    return t, v


def change_times(t, v) -> np.ndarray:
    """Times at which a held-value track changes (§2.4). `t, v` must be cleaned.
    The first sample counts as a change; later samples count if their value
    differs from the previous non-missing sample."""
    if len(t) == 0:
        return np.zeros(0)
    ch = np.r_[True, v[1:] != v[:-1]]
    return t[ch]


def merge_changes(times: np.ndarray, merge_s: float = MERGE_S) -> list[np.ndarray]:
    """Clusters of change times: a change joins the cluster if it is within
    `merge_s` (inclusive) of the previous change (§2.4)."""
    times = np.sort(np.asarray(times, dtype=np.float64))
    if len(times) == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) > merge_s) + 1
    return np.split(times, breaks)


def value_after(t, v, t_r: float) -> float:
    """Value held at the first sample strictly after t_r (§2.4); `t, v` cleaned.

    [judgment] If that sample is itself a change (it can only lie more than 15 s
    after t_r, past a run of missing samples), it holds a later reading's value,
    so this reading's value of the track is missing. The same holds if the track
    has no sample after t_r. A value that continues unchanged after missing
    samples is still this reading's value (§2.4).
    """
    i = int(np.searchsorted(t, t_r, side="right"))
    if 0 < i < len(t) and v[i] == v[i - 1]:
        return float(v[i])
    return float("nan")


def extract_readings(tracks: dict) -> pd.DataFrame:
    """Readings of one case from raw cuff tracks.

    `tracks` maps 'sbp', 'dbp', 'mbp' to (t, v) arrays (missing = NaN).
    Returns one row per merged reading (valid or dropped) with READING_COLUMNS and
    `spread_s` (time from the first to the last change of the merged cluster);
    `n` is the 1-based index among valid readings (NaN if dropped). Gap flags
    are added by `flag_gaps`.
    """
    clean = {k: clean_track(*tracks[k]) for k in CUFF_KEYS}
    all_changes = np.concatenate([change_times(*clean[k]) for k in CUFF_KEYS])
    rows = []
    for i, cl in enumerate(merge_changes(all_changes)):
        t_r = float(cl[-1])   # §2.4: reading time = last change in the cluster
        vals = {k: value_after(*clean[k], t_r) for k in CUFF_KEYS}
        rows.append({"n_all": i + 1, "t": t_r, "sbp": vals["sbp"], "dbp": vals["dbp"], "map": vals["mbp"],
                     "spread_s": float(cl[-1] - cl[0])})
    df = pd.DataFrame(rows, columns=["n_all", "t", "sbp", "dbp", "map", "spread_s"])
    out = apply_exclusions(df)
    out["spread_s"] = df["spread_s"].to_numpy()   # first to last change of the merged cluster
    return out


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Validity, drop reason and valid index `n` of readings (§2.4).

    Drop reasons, first failing rule in this order: `incomplete` (a value is
    missing, so SBP > MAP > DBP cannot hold; [judgment] reported separately),
    `map_range` (MAP outside 30-160 mmHg), `order` (SBP > MAP > DBP fails).
    """
    df = df[["n_all", "t", "sbp", "dbp", "map"]].copy()
    sbp, dbp, mp = (df[c].to_numpy(dtype=float) for c in ("sbp", "dbp", "map"))
    incomplete = ~(np.isfinite(sbp) & np.isfinite(dbp) & np.isfinite(mp))
    with np.errstate(invalid="ignore"):
        fail_range = np.isfinite(mp) & ~((mp >= MAP_MIN) & (mp <= MAP_MAX))
        fail_order = ~((sbp > mp) & (mp > dbp))      # a missing value fails
    reason = np.where(incomplete, DROP_INCOMPLETE,
                      np.where(fail_range, DROP_MAP_RANGE, np.where(fail_order, DROP_ORDER, "")))
    valid = reason == ""
    df["valid"] = valid
    df["drop_reason"] = reason.astype(object) if len(df) else np.array([], dtype=object)
    df["fail_incomplete"] = incomplete
    df["fail_map_range"] = fail_range
    df["fail_order"] = fail_order
    n = np.full(len(df), np.nan)
    n[valid] = np.arange(1, int(valid.sum()) + 1)
    df["n"] = n
    df["gap_flag"] = False
    df["gap_s"] = np.nan
    return df[READING_COLUMNS]


def valid_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Valid readings only, ordered by `n` (§2.4: dropped readings are ignored everywhere)."""
    v = readings.loc[readings["valid"]].sort_values("n").reset_index(drop=True)
    v["n"] = v["n"].astype(int)
    return v


def median_interval(readings: pd.DataFrame, t_lo: float, t_hi: float) -> float:
    """Median interval between consecutive valid readings whose times both lie in
    [t_lo, t_hi] (both ends included, §2.5). NaN if fewer than 2 such readings."""
    t = valid_readings(readings)["t"].to_numpy()
    t = t[(t >= t_lo) & (t <= t_hi)]
    if len(t) < 2:
        return float("nan")
    return float(np.median(np.diff(t)))


def flag_gaps(readings: pd.DataFrame, case_median_s: float) -> pd.DataFrame:
    """Flag valid readings that end a gap longer than GAP_FACTOR x the case's
    median interval (§2.4). `gap_s` = interval to the previous valid reading."""
    df = readings.copy()
    df["gap_flag"] = False
    df["gap_s"] = np.nan
    vi = np.flatnonzero(df["valid"].to_numpy())
    if len(vi) >= 2:
        t = df["t"].to_numpy()[vi]
        gaps = np.diff(t)
        df.loc[df.index[vi[1:]], "gap_s"] = gaps
        if np.isfinite(case_median_s):
            df.loc[df.index[vi[1:]], "gap_flag"] = gaps > GAP_FACTOR * case_median_s
    df["gap_flag"] = df["gap_flag"].astype(bool)
    return df


def readings_from_case(case) -> pd.DataFrame:
    """Readings of a loaded `btc.io.Case` (without gap flags)."""
    return extract_readings({"sbp": (case.sbp.t, case.sbp.v),
                             "dbp": (case.dbp.t, case.dbp.v),
                             "mbp": (case.mbp.t, case.mbp.v)})
