"""Load cases, the case table and the download manifest (SPEC §2.1-§2.3, §2.6).

Cohort access (SPEC §9). `load_case(caseid, cohort)` reads only a case that `data/cohorts.csv`
lists in that cohort. S-ob signals may be read only at the frozen scoring: every S-ob read
requires the environment variable `BTC_FROZEN_SCORING=1` (`check_cohort_access`).

Signal integrity (SPEC §2.3) is decided from the cohort's download manifest, which holds the
checks made at download time:
- a waveform absent from the case file is absent;
- a waveform whose header rate is not 500 Hz is treated as absent;
- a waveform whose block rate (`<key>_srate_est`, from the original case file) departs from
  500 Hz by more than 1%, or whose overwritten share (`<key>_overlap_frac`) exceeds 1%, is
  treated as absent;
- if the block check could not be made (original file unreadable), the waveform is kept and
  flagged.
Times are seconds from the start of the recording (`casestart` = 0). Waveform samples keep
their own start time `t0`; sample i is at `t0 + i / srate`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

EXPECTED_SRATE = 500.0   # §2.3: SNUADC/PLETH and SNUADC/ECG_II at 500 Hz
SRATE_TOL = 0.01         # §2.3: block rate may depart from 500 Hz by at most 1%
OVERWRITE_TOL = 0.01     # §2.3: overwritten share may be at most 1%

FROZEN_SCORING_ENV = "BTC_FROZEN_SCORING"   # SPEC §9: S-ob signals only at the frozen scoring

NUMERIC_KEYS = ("sbp", "dbp", "mbp", "hr", "phen")   # phen = Orchestra/PHEN_RATE, mL/h (§2.3, §2.6)
WAVE_KEYS = ("pleth", "ecg")

# Vital-file missing-value code by sample format (§2.3: converted to missing).
# Float formats (1, 2) use NaN. The download already converted waveforms.
_FMT_MISSING = {3: -128, 4: 255, 5: -32768, 6: 65535, 7: -2147483648, 8: 4294967295}

# Waveform status values (§2.3)
OK = "ok"
ABSENT_IN_FILE = "absent_in_file"          # not in the case file
ABSENT_INTEGRITY = "absent_integrity"      # header rate, block rate or overwritten share fails §2.3
KEPT_FLAGGED = "kept_flagged"              # block check could not be made; kept and flagged
_PRESENT = (OK, KEPT_FLAGGED)

# Cervical cerclage procedures in S-dev (§2.2: six cases kept and flagged).
CERCLAGE_PATTERNS = ("mcdonald", "shirodkar")  # [judgment] the two cerclage operation names in the case table


@dataclass(frozen=True)
class Track:
    """A numeric track as raw (time, value) samples, sorted by time; NaN = missing."""
    t: np.ndarray
    v: np.ndarray

    def __len__(self):
        return len(self.t)

    def extent(self):
        """(first, last) time of a non-missing sample, or (nan, nan)."""
        ok = np.isfinite(self.v)
        if not ok.any():
            return float("nan"), float("nan")
        tt = self.t[ok]
        return float(tt[0]), float(tt[-1])


@dataclass(frozen=True)
class Wave:
    """A waveform at its own rate; sample i is at t0 + i / srate. NaN = missing."""
    x: np.ndarray
    t0: float
    srate: float

    def __len__(self):
        return len(self.x)

    def times(self, i=None) -> np.ndarray:
        """Times of sample indices `i` (all samples if None)."""
        i = np.arange(len(self.x)) if i is None else np.asarray(i)
        return self.t0 + i / self.srate

    def extent(self):
        """(first, last) time of a non-missing sample, or (nan, nan)."""
        ok = np.flatnonzero(np.isfinite(self.x))
        if len(ok) == 0:
            return float("nan"), float("nan")
        return float(self.t0 + ok[0] / self.srate), float(self.t0 + ok[-1] / self.srate)


@dataclass
class Case:
    caseid: int
    cohort: str
    meta: dict
    sbp: Track
    dbp: Track
    mbp: Track
    hr: Track
    phen: Track                  # phenylephrine pump rate, mL/h; empty if the case file has no pump track
    pleth: Wave | None           # None if absent (in file or by §2.3)
    ecg: Wave | None
    pleth_status: str = OK
    ecg_status: str = OK
    waveform_flag: str = ""
    raw: dict = field(default_factory=dict, repr=False)  # waveforms as stored, before §2.3


# ---------------------------------------------------------------------------
# Cohort access, case table and manifest
# ---------------------------------------------------------------------------

def check_cohort_access(cohort: str) -> str:
    """Raise unless signals of `cohort` may be read (SPEC §9)."""
    paths._key(cohort)
    if cohort == "sob" and os.environ.get(FROZEN_SCORING_ENV) != "1":
        raise PermissionError(f"S-ob signals may be read only at the frozen scoring (SPEC §9 step 7): "
                              f"set {FROZEN_SCORING_ENV}=1 for that run only")
    return cohort


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


@lru_cache(maxsize=4)
def _csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _cohorts_df() -> pd.DataFrame:
    return _csv_cached(str(paths.COHORTS), _mtime(paths.COHORTS))


def load_cohorts() -> pd.DataFrame:
    """Case table (case-table metadata only; no signal). Adds `cerclage` (§2.2)."""
    c = _cohorts_df().copy()
    name = c["opname"].astype(str).str.lower()
    c["cerclage"] = (c["cohort"] == "S-dev") & name.str.contains("|".join(CERCLAGE_PATTERNS))
    return c


def caseids(cohort: str) -> list[int]:
    """Case IDs of a cohort, ascending (case-table metadata only)."""
    c = _cohorts_df()
    return sorted(int(x) for x in c.loc[c["cohort"] == paths.COHORT_LABEL[paths._key(cohort)], "caseid"])


def sdev_caseids() -> list[int]:
    return caseids("sdev")


def _assert_cohort(caseid: int, cohort: str) -> int:
    check_cohort_access(cohort)
    c = _cohorts_df()
    row = c.loc[c["caseid"] == int(caseid)]
    if len(row) != 1 or row["cohort"].iloc[0] != paths.COHORT_LABEL[cohort]:
        raise PermissionError(f"case {caseid} is not an {paths.COHORT_LABEL[cohort]} case; "
                              "only that cohort's signals may be read here (SPEC §9)")
    return int(caseid)


def load_manifest(cohort: str = "sdev") -> pd.DataFrame:
    check_cohort_access(cohort)
    p = paths.manifest(cohort)
    return _csv_cached(str(p), _mtime(p)).copy()


def _isnan(x) -> bool:
    try:
        return x is None or bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return True


def waveform_status(row: pd.Series | dict, key: str) -> tuple[str, str]:
    """§2.3 status of waveform `key` ('pleth' or 'ecg') from one manifest row.

    Returns (status, detail). The rules are applied to every case from the manifest columns;
    no case ID is special-cased.
    """
    present = row.get(f"{key}_present")
    if not (isinstance(present, (bool, np.bool_)) and bool(present)
            or str(present).strip().lower() == "true"):
        return ABSENT_IN_FILE, f"{key}: absent from case file"
    hdr = row.get(f"{key}_srate_hdr")
    est = row.get(f"{key}_srate_est")
    ov = row.get(f"{key}_overlap_frac")
    fails = []
    if not _isnan(hdr) and float(hdr) != EXPECTED_SRATE:
        fails.append(f"header rate {float(hdr):g} Hz")
    if not _isnan(est) and abs(float(est) - EXPECTED_SRATE) > SRATE_TOL * EXPECTED_SRATE:
        fails.append(f"block rate {float(est):.2f} Hz")
    if not _isnan(ov) and float(ov) > OVERWRITE_TOL:
        fails.append(f"overwritten share {float(ov):.3f}")
    if fails:
        return ABSENT_INTEGRITY, f"{key}: " + ", ".join(fails)
    if _isnan(est) or _isnan(ov):
        # [judgment] any block check that could not be computed counts as "not made" (§2.3: kept and flagged)
        return KEPT_FLAGGED, f"{key}: integrity check not made (original file not readable)"
    return OK, ""


def integrity_table(manifest: pd.DataFrame | None = None, cohort: str = "sdev") -> pd.DataFrame:
    """One row per manifest case: §2.3 status of pleth and ECG."""
    m = load_manifest(cohort) if manifest is None else manifest
    rows = []
    for _, r in m.iterrows():
        out = {"caseid": int(r["caseid"])}
        flags = []
        for k in WAVE_KEYS:
            st, detail = waveform_status(r, k)
            out[f"{k}_status"] = st
            out[f"{k}_present"] = st in _PRESENT
            if st != OK:
                flags.append(f"{st}: {detail}")
        out["waveform_flag"] = "; ".join(flags)
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _numeric(z, key: str, meta: dict) -> Track:
    if f"{key}_t" not in z.files:
        return Track(np.zeros(0), np.zeros(0))
    t = np.asarray(z[f"{key}_t"], dtype=np.float64)
    v = np.asarray(z[f"{key}_v"], dtype=np.float64).copy()
    code = _FMT_MISSING.get(int(meta.get(f"{key}_fmt", 1) or 1))
    if code is not None:
        v[v == code] = np.nan          # §2.3: missing-value code -> missing
    v[~np.isfinite(v)] = np.nan
    o = np.argsort(t, kind="stable")   # file order kept for equal times
    return Track(t[o], v[o])


def _wave(z, key: str) -> Wave | None:
    if key not in z.files:
        return None
    x = np.asarray(z[key], dtype=np.float32)
    return Wave(x=x, t0=float(z[f"{key}_t0"]), srate=float(z[f"{key}_srate"]))


def load_case(caseid: int, cohort: str = "sdev", apply_integrity: bool = True) -> Case:
    """Load one case of `cohort` from `data/raw/<cohort>/<caseid>.npz`.

    With `apply_integrity` (default), a waveform that §2.3 treats as absent is returned as None;
    the stored array stays available in `case.raw`.
    """
    cid = _assert_cohort(caseid, cohort)
    path = paths.raw_dir(cohort) / f"{cid}.npz"
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        tracks = {k: _numeric(z, k, meta) for k in NUMERIC_KEYS}
        waves = {k: _wave(z, k) for k in WAVE_KEYS}
    m = load_manifest(cohort)
    mrow = m.loc[m["caseid"] == cid]
    if len(mrow) != 1:
        raise KeyError(f"case {cid} missing from the {cohort} manifest")
    mrow = mrow.iloc[0]
    status, flags = {}, []
    for k in WAVE_KEYS:
        st, detail = waveform_status(mrow, k)
        if (waves[k] is None) != (st == ABSENT_IN_FILE):
            raise ValueError(f"case {cid}: {k} presence in archive disagrees with manifest")
        if st in _PRESENT and waves[k].srate != EXPECTED_SRATE:
            # §2.3: the archive's own header rate decides as well, so no other rate reaches the features
            st, detail = ABSENT_INTEGRITY, f"{k}: header rate {waves[k].srate:g} Hz"
        status[k] = st
        if st != OK:
            flags.append(f"{st}: {detail}")
    use = {k: (waves[k] if (not apply_integrity or status[k] in _PRESENT) else None) for k in WAVE_KEYS}
    return Case(caseid=cid, cohort=cohort, meta=meta, sbp=tracks["sbp"], dbp=tracks["dbp"],
                mbp=tracks["mbp"], hr=tracks["hr"], phen=tracks["phen"],
                pleth=use["pleth"], ecg=use["ecg"],
                pleth_status=status["pleth"], ecg_status=status["ecg"],
                waveform_flag="; ".join(flags), raw=waves)
