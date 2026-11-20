"""Pleth beat delineation and per-beat / per-window waveform features (SPEC.md §4.3, §5).

Everything here works on one 10-s window and uses only that window's samples
(SPEC §4.3, PREREG §3: per-window processing).

Processing of one window (500 Hz, 5,000 samples):
- Band-pass 0.5-8 Hz: Butterworth of order 2 per band edge (`scipy.signal.butter(2, (0.5, 8),
  "bandpass")`, a 4th-order band-pass), run forward and backward (`sosfiltfilt`, zero-phase,
  padded by reflecting the window's own samples) (SPEC §4.3).
- Systolic peaks: Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D. Systolic peak
  detection in acceleration photoplethysmograms measured from emergency responders in tropical
  conditions. PLoS One 2013;8:e76585, doi:10.1371/journal.pone.0076585; on the band-passed
  window: negative values clipped, squared; moving averages of 111 ms and 667 ms; offset
  0.02 x mean; blocks of interest at least 111 ms long; the peak is the maximum of each
  complete block; a peak within 0.3 s of the previous one is skipped, as in the neurokit2
  implementation (Makowski D, et al. Behav Res Methods 2021;53:1689-1696,
  doi:10.3758/s13428-020-01516-y). Moving averages are centred, edges repeat the end sample.
- Valleys: the minimum between consecutive systolic peaks. Before the first peak (after the
  last), the minimum between the window start (end) and the peak, kept only if it is not at
  the window edge. [judgment]
- A **beat** is a systolic peak with a valley before it (its foot valley), a valley after it
  (its end valley), a positive maximum upslope and a tangent foot that does not fall before
  the foot valley. The >= 4-beats condition of SPEC §5 counts these beats. [judgment]

Per beat (SPEC §4.3); heights are measured from the foot level = the foot-valley value:
- foot: intersecting tangents: the horizontal line through the foot valley meets the
  tangent at the point of maximum slope between the foot valley and the peak (sub-sample).
- rise time: systolic peak time minus foot time.
- amplitude: peak value minus foot level (used only within a window, for the CV).
- width: time between the up- and down-crossings of foot level + amplitude / 2 (linear
  interpolation); missing if the beat does not fall below half height before its end valley.
- pulse arrival time (PAT): foot time minus the time of the R peak of the same heartbeat.
  In the stored VitalDB signals the pleth lags the ECG by a processing delay: on S-dev the
  foot-to-R lags cluster at about 0.55-0.75 s in every case surveyed (signal-level survey of
  every 12th S-dev case, before the range was fixed; the same histogram on the detector-
  selection cases is written by `scripts/select_rpeak_detector.py`). This exceeds the RR
  interval at heart rates above about 90/min, so the latest R peak before a foot is then
  the next beat's.
  The R peak of a beat is therefore the unique R peak whose lag to the foot lies in
  [0.40, 0.90) s; with none or more than one, the beat has no PAT. The range is centred on
  the observed cluster and is 0.5 s wide, so it holds at most one R peak while RR > 0.5 s;
  with RR < 0.5 s (heart rate above 120/min) a beat can have no PAT by construction.
  [judgment] A constant ECG-pleth offset cancels in the within-case forms (SPEC §4.3).

Dicrotic notch: data-quality report only, **not a model input** (SPEC §4.3: the detector does
not find the notch reliably in this band-limited pleth). [judgment] The detector is the "e"
wave of the second derivative (acceleration plethysmogram; Takazawa K, et al. Hypertension
1998;32:365-370; Elgendi M. Curr Cardiol Rev 2012;8:14-25): the largest strict local maximum
of the second derivative of the band-passed pleth strictly between the systolic peak and the
end valley, at which the second derivative is positive. notch_t = notch time minus systolic
peak time; notch_ratio = (notch value - foot level) / amplitude. Both are kept in the window
table (`QC_FEATURES`) and never enter the per-forecast table.

Per window (SPEC §4.3): the median over beats of rise, width and PAT, and beat-amplitude
variability = CV of the beat amplitudes (sample SD, ddof = 1, over the mean); also the
medians of the two notch quantities (data quality only; beats with a notch). PAT needs >= 4
beats with a PAT, the same minimum as the >= 4-beats condition, to guard against spurious
R peaks. [judgment]

Beat-template correlation (stricter sensitivity rule, SPEC §5 disclosure) [judgment]: each
beat is the band-passed segment starting at its foot valley, of length = the median
valley-to-valley interval of the window's beats; beats running past the window end are left
out. The template is the sample-wise median of these segments; the window's value is the
median Pearson correlation of each segment with the template. The strict rule requires
>= 0.85 (and >= 2 segments).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

FS = 500.0                     # fiducial points at 500 Hz (SPEC §4.3)
BP_BAND = (0.5, 8.0)           # SPEC §4.3
BP_ORDER = 2                   # SPEC §4.3
MIN_BEATS = 4                  # SPEC §5 added condition
MIN_PAT_BEATS = 4              # [judgment] see module docstring
PAT_LAG = (0.40, 0.90)         # [judgment] R-peak lag range for PAT pairing, s (docstring)
TCORR_MIN = 0.85               # SPEC §5 disclosure (stricter sensitivity rule)

# Elgendi et al. 2013 parameters (and neurokit2's minimum peak spacing).
EG_PEAK_S, EG_BEAT_S, EG_OFFSET, EG_MINDELAY_S = 0.111, 0.667, 0.02, 0.3

_SOS = butter(BP_ORDER, BP_BAND, btype="bandpass", fs=FS, output="sos")

FEATURES = ("pat", "rise", "width", "bav")          # per-window features that enter M4 (SPEC §4.3)
QC_FEATURES = ("notch_t", "notch_ratio")            # data-quality report only (SPEC §4.3), never a model input
WINDOW_FEATURES = ("pat", "rise", "width", "notch_t", "notch_ratio", "bav")   # columns of the window table


def bandpass(x: np.ndarray) -> np.ndarray:
    """0.5-8 Hz, 2nd-order Butterworth, zero-phase, on one window (SPEC §4.3)."""
    return sosfiltfilt(_SOS, np.asarray(x, dtype=np.float64))


def systolic_peaks(xf: np.ndarray, fs: float = FS) -> np.ndarray:
    """Systolic peak indices by Elgendi et al. 2013 on a band-passed window."""
    y = np.where(xf > 0, xf, 0.0) ** 2
    w1 = int(np.rint(EG_PEAK_S * fs))
    w2 = int(np.rint(EG_BEAT_S * fs))
    ma_peak = uniform_filter1d(y, w1, mode="nearest")
    ma_beat = uniform_filter1d(y, w2, mode="nearest")
    blocks = ma_peak > ma_beat + EG_OFFSET * y.mean()
    d = np.diff(blocks.astype(np.int8))
    beg = np.flatnonzero(d == 1) + 1          # first sample of a block
    end = np.flatnonzero(d == -1) + 1         # one past its last sample
    if beg.size == 0:
        return np.zeros(0, dtype=np.int64)
    end = end[end > beg[0]]                   # only complete blocks
    mindelay = int(np.rint(EG_MINDELAY_S * fs))
    peaks: list[int] = []
    for b, e in zip(beg, end):
        if e - b < w1:
            continue
        p = b + int(np.argmax(xf[b:e]))
        if peaks and p - peaks[-1] < mindelay:
            continue
        peaks.append(p)
    return np.asarray(peaks, dtype=np.int64)


def _crossing_up(xf: np.ndarray, lo: int, hi: int, level: float) -> float:
    """Last up-crossing of `level` in xf[lo..hi] (fractional index), NaN if none."""
    seg = xf[lo:hi + 1]
    below = np.flatnonzero(seg < level)
    if below.size == 0:
        return np.nan
    j = lo + int(below[-1])
    if j >= hi:
        return np.nan
    x0, x1 = xf[j], xf[j + 1]
    return j + (level - x0) / (x1 - x0)


def _crossing_down(xf: np.ndarray, lo: int, hi: int, level: float) -> float:
    """First down-crossing of `level` in xf[lo..hi] (fractional index), NaN if none."""
    seg = xf[lo:hi + 1]
    below = np.flatnonzero(seg < level)
    if below.size == 0:
        return np.nan
    j = lo + int(below[0])
    if j <= lo:
        return np.nan
    x0, x1 = xf[j - 1], xf[j]
    return (j - 1) + (x0 - level) / (x0 - x1)


def _notch(d2: np.ndarray, p: int, v_end: int) -> int:
    """Dicrotic-notch index (APG "e" wave) strictly between peak p and end valley v_end, or -1."""
    j = np.arange(p + 1, v_end)
    if j.size == 0:
        return -1
    cand = j[(d2[j] > d2[j - 1]) & (d2[j] >= d2[j + 1]) & (d2[j] > 0)]
    return int(cand[np.argmax(d2[cand])]) if cand.size else -1


def delineate(xf: np.ndarray, fs: float = FS) -> dict:
    """Delineate beats in one band-passed window.

    Returns arrays over beats (fractional sample indices relative to the window start):
    `foot`, `peak`, `v_foot`, `v_end`, `rise_s`, `amp`, `width_s`, `notch_s`, `notch_ratio`,
    plus `all_peaks` (every systolic peak).
    """
    n = xf.size
    pk = systolic_peaks(xf, fs)
    m = pk.size
    empty = {k: np.zeros(0) for k in ("foot", "peak", "v_foot", "v_end", "rise_s", "amp",
                                      "width_s", "notch_s", "notch_ratio")}
    empty["all_peaks"] = pk
    if m == 0:
        return empty
    dx = np.gradient(xf)
    d2 = np.gradient(dx)
    # valleys[i] precedes peak i; valleys[m] follows the last peak; -1 = none
    valleys = np.full(m + 1, -1, dtype=np.int64)
    v0 = int(np.argmin(xf[:pk[0]])) if pk[0] > 0 else 0
    valleys[0] = v0 if 0 < v0 else -1
    for i in range(1, m):
        valleys[i] = pk[i - 1] + int(np.argmin(xf[pk[i - 1]:pk[i]]))
    vl = pk[-1] + int(np.argmin(xf[pk[-1]:]))
    valleys[m] = vl if vl < n - 1 else -1

    # tangent foot for every peak with a foot valley
    foot_all = np.full(m, np.nan)
    for i in range(m):
        v = valleys[i]
        if v < 0 or v >= pk[i]:
            continue
        seg = dx[v:pk[i] + 1]
        k = v + int(np.argmax(seg))
        s = dx[k]
        if not s > 0:
            continue
        f = k - (xf[k] - xf[v]) / s
        if f >= v:                            # a tangent foot before the foot valley is not a foot
            foot_all[i] = f

    rows = []
    for i in range(m):
        v, ve, p = valleys[i], valleys[i + 1], pk[i]
        if v < 0 or ve < 0 or not np.isfinite(foot_all[i]):
            continue
        f = foot_all[i]
        amp = xf[p] - xf[v]
        if not amp > 0:
            continue
        half = xf[v] + 0.5 * amp
        up = _crossing_up(xf, v, p, half)
        dn = _crossing_down(xf, p, ve, half)
        nt = _notch(d2, p, ve)
        rows.append((f, p, v, ve, (p - f) / fs, amp, (dn - up) / fs,
                     (nt - p) / fs if nt >= 0 else np.nan,
                     (xf[nt] - xf[v]) / amp if nt >= 0 else np.nan))
    if not rows:
        return empty
    a = np.asarray(rows, dtype=np.float64)
    out = dict(zip(("foot", "peak", "v_foot", "v_end", "rise_s", "amp", "width_s", "notch_s",
                    "notch_ratio"), a.T))
    out["all_peaks"] = pk
    return out


def pair_pat(beats: dict, r_times: np.ndarray, t_start: float, fs: float = FS) -> np.ndarray:
    """PAT (s) per beat; `r_times` are absolute R-peak times, `t_start` the window start."""
    foot_t = t_start + beats["foot"] / fs
    pat = np.full(foot_t.size, np.nan)
    if r_times.size == 0:
        return pat
    for i in range(foot_t.size):
        lag = foot_t[i] - r_times
        inside = lag[(lag >= PAT_LAG[0]) & (lag < PAT_LAG[1])]
        if inside.size == 1:
            pat[i] = inside[0]
    return pat


def template_corr(xf: np.ndarray, beats: dict) -> float:
    """Median correlation of each beat with the window's median beat (SPEC §5 disclosure)."""
    v = beats["v_foot"].astype(np.int64)
    ve = beats["v_end"].astype(np.int64)
    if v.size < 2:
        return np.nan
    L = int(np.median(ve - v))
    if L < 2:
        return np.nan
    segs = np.asarray([xf[s:s + L] for s in v if s + L <= xf.size])
    if segs.shape[0] < 2:
        return np.nan
    tmpl = np.median(segs, axis=0)
    tc = tmpl - tmpl.mean()
    sc = segs - segs.mean(axis=1, keepdims=True)
    den = np.sqrt((sc ** 2).sum(axis=1) * (tc ** 2).sum())
    r = np.where(den > 0, (sc @ tc) / np.where(den > 0, den, 1.0), np.nan)
    return float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan


def _median_or_nan(a: np.ndarray, min_n: int = 1) -> float:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size >= min_n else np.nan


def window_features(x_raw: np.ndarray, r_times: np.ndarray, t_start: float,
                    fs: float = FS) -> dict:
    """All per-window beat quantities for one 500 Hz pleth window (SPEC §4.3, §5).

    Includes the notch quantities (`QC_FEATURES`), which are for the data-quality report only.

    `x_raw`: the window's raw 500 Hz pleth samples; `r_times`: absolute R-peak times found in
    the same window (empty if no ECG). A window with a missing 500 Hz sample has no beats.
    """
    out = {"n_beats": 0, "n_peaks": 0, "n_pat": 0, "n_notch": 0, "pleth_rate": np.nan,
           "tcorr": np.nan, **{f: np.nan for f in WINDOW_FEATURES}}
    x = np.asarray(x_raw, dtype=np.float64)
    if x.size < int(2 * fs) or not np.all(np.isfinite(x)) or not x.std() > 0:
        return out
    xf = bandpass(x)
    b = delineate(xf, fs)
    pk = b["all_peaks"]
    out["n_peaks"] = int(pk.size)
    if pk.size >= 2 and pk[-1] > pk[0]:
        out["pleth_rate"] = float(60.0 * (pk.size - 1) / ((pk[-1] - pk[0]) / fs))
    nb = b["foot"].size
    out["n_beats"] = int(nb)
    if nb == 0:
        return out
    pat = pair_pat(b, np.asarray(r_times, dtype=np.float64), t_start, fs)
    out["n_pat"] = int(np.isfinite(pat).sum())
    out["n_notch"] = int(np.isfinite(b["notch_s"]).sum())
    out["pat"] = _median_or_nan(pat, MIN_PAT_BEATS)
    out["rise"] = _median_or_nan(b["rise_s"])
    out["width"] = _median_or_nan(b["width_s"])
    out["notch_t"] = _median_or_nan(b["notch_s"])
    out["notch_ratio"] = _median_or_nan(b["notch_ratio"])
    amp = b["amp"]
    out["bav"] = float(amp.std(ddof=1) / amp.mean()) if nb >= 2 else np.nan
    out["tcorr"] = template_corr(xf, b)
    return out
