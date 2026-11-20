"""Pleth quality index (SPEC.md §5, PREREG.md §5).

Primary index: the BP-blind rule of Lee, Al-antari & Lee, Diagnostics 2026;16:3713,
criteria and thresholds unchanged, with the operational details fixed in SPEC §5.

Steps (SPEC §5):
1. The 500 Hz pleth is reduced to 125 Hz by taking every 4th sample, starting at the
   first sample of the stored record, with no interpolation.
2. The 125 Hz series is cut into consecutive non-overlapping 10-s windows of 1,250
   samples, starting at the first sample of the record. A trailing partial window is
   not a window.
3. A window with any missing sample, or a flat window (SD 0), is discarded (unusable).
4. Each remaining window is standardised to zero mean and unit variance (population SD).

A window is unusable if any criterion holds, all on the unfiltered 125 Hz window:
1. dominant frequency outside 0.67-3 Hz;
2. spectral peak share < 15% of the 0.5-8 Hz power;
3. max |z| > 6 (PREREG §5);
4. constant runs of >= 63 equal consecutive samples total >= 125 samples.

Window times: window w starts at t0 + 10 w (its first sample) and ends at t0 + 10 (w + 1),
exclusive; every sample of the window lies strictly before its end time.

The added >= 4-beats condition (SPEC §5) and the stricter template-correlation rule
(SPEC §5 disclosure) need beat delineation and live in `btc.beats`.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

FS_RAW = 500.0          # SNUADC/PLETH header rate (SPEC §2.3)
DECIM = 4               # every 4th sample -> 125 Hz (SPEC §5 step 1)
FS = FS_RAW / DECIM     # 125 Hz
WIN_N = 1250            # samples per window at 125 Hz (SPEC §5 step 2)
WIN_S = WIN_N / FS      # 10 s
WIN_N_RAW = WIN_N * DECIM  # 5,000 samples at 500 Hz

BAND = (0.5, 8.0)       # in-band range for the spectrum (SPEC §5 operational definitions)
HR_BAND = (0.67, 3.0)   # criterion 1: 40-180 beats/min
PEAK_SHARE_MIN = 0.15   # criterion 2
Z_MAX = 6.0             # criterion 3: unusable if max |z| > 6
RUN_MIN = 63            # criterion 4: a constant run is >= 63 equal samples (0.5 s)
RUN_TOTAL = 125         # criterion 4: unusable if runs total >= 125 samples (10%)

# Spectrum: 1,250-point Hann periodogram at 125 Hz gives 0.1 Hz bins; bin k is k * 0.1 Hz.
# Integer bin indices avoid floating-point edge effects. [judgment] band edges inclusive.
_K_LO, _K_HI = 5, 80    # 0.5 Hz and 8.0 Hz


def n_windows(n_raw: int) -> int:
    """Number of complete 10-s windows in a 500 Hz record of n_raw samples (SPEC §5 steps 1-2)."""
    n125 = (n_raw + DECIM - 1) // DECIM
    return n125 // WIN_N


def window_times(t0: float, n_win: int, fs_raw: float = FS_RAW) -> tuple[np.ndarray, np.ndarray]:
    """Start and (exclusive) end time of each window, in seconds from time zero."""
    w = np.arange(n_win)
    start = t0 + w * (WIN_N_RAW / fs_raw)
    return start, start + WIN_N_RAW / fs_raw


def window_125(x_raw: np.ndarray, w: int) -> np.ndarray:
    """The 125 Hz samples of window w: raw samples 5000 w, 5000 w + 4, ... (SPEC §5 steps 1-2)."""
    return x_raw[w * WIN_N_RAW: (w + 1) * WIN_N_RAW: DECIM]


def constant_run_total(x: np.ndarray, run_min: int = RUN_MIN) -> int:
    """Total samples in runs of >= run_min exactly equal consecutive samples."""
    if x.size == 0:
        return 0
    change = np.flatnonzero(np.diff(x) != 0) + 1
    bounds = np.concatenate(([0], change, [x.size]))
    lengths = np.diff(bounds)
    return int(lengths[lengths >= run_min].sum())


def assess_window(x125: np.ndarray) -> dict:
    """Apply the primary index to one 125 Hz window (SPEC §5).

    Returns the measured quantities, one boolean per failure reason and `pqi_pass`.
    A discarded window (missing or flat) has NaN measurements and fails.
    """
    out = {
        "n_missing": int(np.count_nonzero(~np.isfinite(x125))),
        "dom_freq": np.nan, "peak_share": np.nan, "max_abs_z": np.nan, "const_run_n": -1,
        "fail_missing": False, "fail_flat": False, "fail_freq": False,
        "fail_peak": False, "fail_amp": False, "fail_const": False,
    }
    if x125.size != WIN_N or out["n_missing"] > 0:
        out["fail_missing"] = True
        out["pqi_pass"] = False
        return out
    x = x125.astype(np.float64)
    sd = x.std()
    if not sd > 0:
        out["fail_flat"] = True
        out["pqi_pass"] = False
        return out
    z = (x - x.mean()) / sd                                   # step 4
    f, p = periodogram(z, fs=FS, window="hann", nfft=WIN_N,   # 0.1 Hz bins
                       detrend="constant", scaling="density")
    band = p[_K_LO:_K_HI + 1]
    k = _K_LO + int(np.argmax(band))                          # largest bin in 0.5-8 Hz
    dom = k * FS / WIN_N
    # Peak power: the dominant bin +/- 1 bin, divided by the 0.5-8 Hz sum. [judgment] the
    # +/- 1 bins are clipped to the band; this never changes a decision, because a dominant
    # bin at a band edge (0.5 or 8 Hz) already fails criterion 1.
    lo, hi = max(k - 1, _K_LO), min(k + 1, _K_HI)
    tot = band.sum()
    share = p[lo:hi + 1].sum() / tot if tot > 0 else 0.0
    out["dom_freq"] = float(dom)
    out["peak_share"] = float(share)
    out["max_abs_z"] = float(np.abs(z).max())
    out["const_run_n"] = constant_run_total(x)                # on the unstandardised window
    # [judgment] 0.67 <= f <= 3.0 Hz is in range (0.7 Hz is the lowest bin that passes).
    out["fail_freq"] = not (HR_BAND[0] <= dom <= HR_BAND[1] + 1e-9)
    out["fail_peak"] = share < PEAK_SHARE_MIN
    out["fail_amp"] = out["max_abs_z"] > Z_MAX
    out["fail_const"] = out["const_run_n"] >= RUN_TOTAL
    out["pqi_pass"] = not (out["fail_freq"] or out["fail_peak"] or out["fail_amp"] or out["fail_const"])
    return out
