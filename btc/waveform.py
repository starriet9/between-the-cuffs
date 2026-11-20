"""Waveform signals, the per-case window table, and per-forecast waveform features.

SPEC.md §2.3 (signals, via `btc.io`), §4.3 (features), §5 (quality index), §6.6 (placebo
variants).

Window table (one row per 10-s window, `data/derived/<cohort>_windows.parquet`)
------------------------------------------------------------------------------
Windows are cut from the first sample of the stored pleth record (SPEC §5): window w spans
[t_start, t_end) with t_start = pleth_t0 + 10 w and t_end = t_start + 10 s. The ECG window
holds the ECG samples whose times lie in the same interval; it is used only if it is
complete (5,000 samples, none missing) and not flat. [judgment] Each window is processed on
its own (SPEC §4.3). Columns: the quality measures and failure flags of `btc.pqi`;
`pqi_pass`; beat counts and per-window features of `btc.beats` (computed for windows that
pass the primary index); `usable` = pqi_pass and >= 4 beats (SPEC §5 added condition);
`usable_strict` = usable and template correlation >= 0.85 (SPEC §5 disclosure); R-peak
count and rate, and `r_failed` (the detector raised on the window, which then has no R
peaks); the median monitor HR (`Solar8000/HR`, values > 0) over the window, used only for
the signal-level agreement summaries. The notch quantities `notch_t` and `notch_ratio` are
kept here for the data-quality report only; they are not model inputs (SPEC §4.3).

Per-forecast features (SPEC §4.3) [judgment, as stated here]
-------------------------------------------------------------
Inputs are times only (never cuff values): the issue time t_r, the previous valid reading
t_p, the valid reading before it t_pp, and the cutoff t_end (signal samples at or after
t_end are never used; t_end = t_r for the primary features).
- A window lies **between** times a and b if t_start >= a and t_end_window <= b. Because the
  window end is exclusive, every sample of such a window is strictly earlier than b.
- Current interval: a = max(t_p, t_r - 120 s), b = min(t_r, t_end) ("between the previous
  reading, at most 120 s back, and t_r").
- `wf_n_windows` = windows between a and b; `wf_n_usable` = the usable ones;
  `wf_usable_share` = their ratio (NaN with no window); `wf_available` = wf_n_usable >= 2
  (otherwise M4 falls back to M3).
- Each feature's per-forecast value F(t_r) = median of its per-window values over the usable
  windows between a and b that have a value; it is missing unless at least 2 windows have
  a value (the 2-window minimum of SPEC §4.3 applied to each feature).
- Record reference = per feature, the median over the first 6 usable windows of the record
  (in time order), with the same 2-value minimum; missing unless 6 usable windows exist and
  the 6th ends at or before min(t_r, t_end).
- Form (i) `wf_<f>_lvl` = F(t_r) - reference.
- Form (ii) `wf_<f>_chg` = F(t_r) - F(t_p), where F(t_p) is the same aggregation at the
  previous reading: windows between max(t_pp, t_p - 120 s) and min(t_p, t_end). This mirrors
  the heart-rate change (SPEC §4.2: "the same median before the previous reading").
- Features: pulse arrival time, rise time, width and beat-amplitude variability
  (`btc.beats.FEATURES`). Pulse arrival time enters only in these two within-case forms
  (SPEC §4.3). No absolute pleth amplitude enters (SPEC §4.3); `bav` is a within-window
  ratio. The notch quantities never enter this table (SPEC §4.3).

Variants (btc/SCHEMA.md §4, SPEC §3, §5, §6.6):
- real: own signals, t_end = t_r.
- p1: the donor's window table (its own quality mask) with the recipient's times, t_end = t_r.
- p2: own signals with t_r, t_p and t_pp shifted back by 600 s; t_end = t_r - 600 s.
- ia: own signals, t_end = t_r - 30 s; the interval starts are unchanged, so only the 30 s
  before t_r are excluded (from every window used, including the reference).
- strict: own signals, `usable_strict` in place of `usable` everywhere (also the reference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc import beats, io, pqi, rpeaks
from btc.constants import IA_EXCLUDE_S, LOOKBACK_S, P2_SHIFT_S
from btc.constants import MIN_USABLE_WINDOWS as MIN_WINDOWS
from btc.io import Wave  # noqa: F401  (the one waveform type; sample i at t0 + i / srate)

N_REF = 6                 # SPEC §4.3: reference = first 6 usable windows of the record

FEATURES = beats.FEATURES             # model features only (SPEC §4.3); no notch quantities
VARIANTS = ("real", "p1", "p2", "ia", "strict")
FEATURE_COLS = [f"wf_{f}_{form}" for f in FEATURES for form in ("lvl", "chg")]


# --------------------------------------------------------------------------- signals
def load_case(caseid: int, cohort: str = "sdev") -> dict:
    """Pleth, ECG and monitor HR of one case, via `btc.io.load_case`.

    The SPEC §2.3 integrity rule has one implementation, in `btc.io`; a waveform it treats as
    absent (including a header rate other than 500 Hz) is None here. `status` holds its
    pleth/ECG status strings.
    """
    c = io.load_case(caseid, cohort)
    out = {"caseid": c.caseid, "status": {"pleth": c.pleth_status, "ecg": c.ecg_status}}
    for key in ("pleth", "ecg"):
        w = getattr(c, key)
        out[key] = None if w is None else Wave(np.asarray(w.x, dtype=np.float64), float(w.t0), float(w.srate))
    out["hr"] = (c.hr.t, c.hr.v) if len(c.hr) else None
    return out


def present(status: str) -> bool:
    """True if `btc.io` keeps the waveform (status ok or kept-and-flagged, SPEC §2.3)."""
    return status in (io.OK, io.KEPT_FLAGGED)


# --------------------------------------------------------------------------- window table
WINDOW_COLS = [
    "caseid", "w", "t_start", "t_end",
    "n_missing", "dom_freq", "peak_share", "max_abs_z", "const_run_n",
    "fail_missing", "fail_flat", "fail_freq", "fail_peak", "fail_amp", "fail_const", "pqi_pass",
    "n_peaks", "n_beats", "fail_beats", "usable", "tcorr", "usable_strict",
    "n_pat", "n_notch", *beats.WINDOW_FEATURES, "pleth_rate",
    "ecg_ok", "r_failed", "n_r", "r_rate", "hr_monitor", "n_hr",
]


def _ecg_window(ecg: Wave | None, t_start: float, t_end: float):
    """(first index, samples) of the ECG in [t_start, t_end); None if absent or incomplete."""
    if ecg is None:
        return None
    i0 = int(np.ceil((t_start - ecg.t0) * ecg.srate - 1e-6))
    i1 = int(np.ceil((t_end - ecg.t0) * ecg.srate - 1e-6))
    if i0 < 0 or i1 > ecg.x.size or i1 - i0 != pqi.WIN_N_RAW:
        return None
    return i0, ecg.x[i0:i1]


def case_windows(caseid: int, pleth: Wave | None, ecg: Wave | None, hr=None,
                 detector: str = rpeaks.DETECTOR) -> pd.DataFrame:
    """The window table of one case (see module docstring)."""
    if pleth is None:
        return pd.DataFrame(columns=WINDOW_COLS)
    nw = pqi.n_windows(pleth.x.size)
    starts, ends = pqi.window_times(pleth.t0, nw, pleth.srate)
    hr_t, hr_v = (np.asarray(hr[0]), np.asarray(hr[1])) if hr is not None else (None, None)
    rows = []
    for w in range(nw):
        ts, te = float(starts[w]), float(ends[w])
        r = {"caseid": int(caseid), "w": w, "t_start": ts, "t_end": te}
        r.update(pqi.assess_window(pqi.window_125(pleth.x, w)))
        # ECG: R peaks in the same 10-s interval, detected on that window alone
        ew = _ecg_window(ecg, ts, te)
        r_times = np.zeros(0)
        r["ecg_ok"] = r["r_failed"] = False
        if ew is not None:
            i0, xe = ew
            if np.all(np.isfinite(xe)) and xe.std() > 0:
                r["ecg_ok"] = True
                pk, r["r_failed"] = rpeaks.detect_checked(xe, ecg.srate, detector)
                r_times = ecg.times(i0 + pk)
                r["n_r"] = int(pk.size)
                r["r_rate"] = rpeaks.rate_bpm(pk, ecg.srate)
        # beats and features, for windows that pass the primary index
        if r["pqi_pass"]:
            seg = pleth.x[w * pqi.WIN_N_RAW:(w + 1) * pqi.WIN_N_RAW]
            r.update(beats.window_features(seg, r_times, ts, pleth.srate))
        if hr_t is not None:
            m = (hr_t >= ts) & (hr_t < te) & np.isfinite(hr_v) & (hr_v > 0)
            r["n_hr"] = int(m.sum())
            r["hr_monitor"] = float(np.median(hr_v[m])) if m.any() else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).reindex(columns=WINDOW_COLS)
    for c in ("n_peaks", "n_beats", "n_pat", "n_notch", "n_r", "n_hr"):
        df[c] = df[c].fillna(0).astype(np.int64)
    df["fail_beats"] = df["pqi_pass"].astype(bool) & (df["n_beats"] < beats.MIN_BEATS)
    df["usable"] = df["pqi_pass"].astype(bool) & (df["n_beats"] >= beats.MIN_BEATS)
    df["usable_strict"] = df["usable"] & (df["tcorr"] >= beats.TCORR_MIN)
    for c in ("fail_missing", "fail_flat", "fail_freq", "fail_peak", "fail_amp", "fail_const",
              "pqi_pass", "ecg_ok", "r_failed"):
        df[c] = df[c].astype(bool)
    return df


# --------------------------------------------------------------------------- per forecast
class CaseWindows:
    """Arrays of one case's window table, for fast per-forecast aggregation."""

    def __init__(self, df: pd.DataFrame | None):
        df = df if df is not None else pd.DataFrame(columns=WINDOW_COLS)
        df = df.sort_values("w")
        self.ts = df["t_start"].to_numpy(np.float64)
        self.te = df["t_end"].to_numpy(np.float64)
        self.usable = {"usable": df["usable"].to_numpy(bool),
                       "usable_strict": df["usable_strict"].to_numpy(bool)}
        self.vals = np.column_stack([df[f].to_numpy(np.float64) for f in FEATURES]) \
            if len(df) else np.zeros((0, len(FEATURES)))

    def aggregate(self, a: float, b: float, mask: str):
        """(feature medians, n windows, n usable) over windows between a and b."""
        sel = (self.ts >= a) & (self.te <= b)
        use = sel & self.usable[mask]
        return _medians(self.vals[use]), int(sel.sum()), int(use.sum())

    def reference(self, t_end: float, mask: str) -> np.ndarray:
        idx = np.flatnonzero(self.usable[mask])[:N_REF]
        if idx.size < N_REF or self.te[idx[-1]] > t_end:
            return np.full(len(FEATURES), np.nan)
        return _medians(self.vals[idx])


def _medians(v: np.ndarray) -> np.ndarray:
    out = np.full(v.shape[1], np.nan)
    for j in range(v.shape[1]):
        col = v[:, j]
        col = col[np.isfinite(col)]
        if col.size >= MIN_WINDOWS:
            out[j] = np.median(col)
    return out


def forecast_features(cw: CaseWindows, t_issue: float, t_prev: float, t_prev2: float,
                      t_end: float, mask: str = "usable") -> dict:
    """Waveform features of one forecast (see module docstring). Inputs are times only."""
    b = min(t_issue, t_end)
    a = max(t_prev, t_issue - LOOKBACK_S)
    cur, n_win, n_use = cw.aggregate(a, b, mask)
    pa, pb = max(t_prev2, t_prev - LOOKBACK_S), min(t_prev, t_end)
    prv, _, _ = cw.aggregate(pa, pb, mask)
    ref = cw.reference(b, mask)
    available = n_use >= MIN_WINDOWS
    out = {"t_lo": a, "wf_n_windows": n_win, "wf_n_usable": n_use,
           "wf_usable_share": n_use / n_win if n_win > 0 else np.nan,
           "wf_available": bool(available)}
    for j, f in enumerate(FEATURES):
        out[f"wf_{f}_lvl"] = cur[j] - ref[j] if available else np.nan
        out[f"wf_{f}_chg"] = cur[j] - prv[j] if available else np.nan
    return out


def variant_times(variant: str, t_issue: float, t_prev: float, t_prev2: float):
    """(t_issue, t_prev, t_prev2, t_end, mask) used by a variant (SCHEMA §4)."""
    if variant in ("real", "p1"):
        return t_issue, t_prev, t_prev2, t_issue, "usable"
    if variant == "p2":
        s = P2_SHIFT_S
        return t_issue - s, t_prev - s, t_prev2 - s, t_issue - s, "usable"
    if variant == "ia":
        return t_issue, t_prev, t_prev2, t_issue - IA_EXCLUDE_S, "usable"
    if variant == "strict":
        return t_issue, t_prev, t_prev2, t_issue, "usable_strict"
    raise ValueError(variant)
