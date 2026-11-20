"""ECG R-peak detection (SPEC.md §4.3, §9).

One fixed detector, chosen on S-dev by its agreement with the monitor heart rate
(`Solar8000/HR`), on signal-level criteria only (SPEC §9 step 3). The candidates and
the selection rule are written in `scripts/select_rpeak_detector.py`; the comparison table
and the choice are written to `data/derived/rpeak_selection.md`. No blood-pressure value is
used.

The detectors are the neurokit2 implementations (`ecg_clean` and `ecg_findpeaks`): Makowski D,
Pham T, Lau ZJ, et al. NeuroKit2: a Python toolbox for neurophysiological signal processing.
Behav Res Methods 2021;53:1689-1696, doi:10.3758/s13428-020-01516-y.

Each 10-s ECG window is processed on its own (SPEC §4.3): it is cleaned and searched using
only that window's samples. A window with any missing sample, or a flat window, has no R
peaks. A detector that raises on a window finds no peaks there; `detect_checked` reports
this, and the builders count such windows in their logs. [judgment]
"""
from __future__ import annotations

import warnings

import numpy as np

FS = 500.0  # SNUADC/ECG_II header rate (SPEC §2.3)

# Candidates (neurokit2 method names; each uses its own cleaning step).
CANDIDATES = ("neurokit", "pantompkins1985", "hamilton2002", "elgendi2010")

# The fixed detector, chosen by scripts/select_rpeak_detector.py on 50 S-dev cases: share of
# windows within 5 beats/min of the monitor HR 0.788 (neurokit), 0.769 (elgendi2010), 0.650
# (pantompkins1985), 0.451 (hamilton2002); see rpeak_selection.md.
DETECTOR = "neurokit"

# Mains frequency for the neurokit2 powerline filter (method "neurokit" only; the other
# candidates have no such step). VitalDB was recorded in Korea, where mains power is 60 Hz;
# the neurokit2 default is 50 Hz. [judgment]
POWERLINE_HZ = 60

_EMPTY = np.zeros(0, dtype=np.int64)


def detect_checked(ecg: np.ndarray, fs: float = FS, method: str = DETECTOR) -> tuple[np.ndarray, bool]:
    """(R-peak sample indices, sorted, unique, 0-based; True if the detector raised)."""
    import neurokit2 as nk

    x = np.asarray(ecg, dtype=np.float64)
    if x.size < int(2 * fs) or not np.all(np.isfinite(x)) or not x.std() > 0:
        return _EMPTY, False
    kw = {"powerline": POWERLINE_HZ} if method == "neurokit" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clean = nk.ecg_clean(x, sampling_rate=int(fs), method=method, **kw)
            peaks = nk.ecg_findpeaks(clean, sampling_rate=int(fs), method=method)["ECG_R_Peaks"]
        except (ValueError, IndexError, ZeroDivisionError):
            return _EMPTY, True
    peaks = np.asarray(peaks, dtype=np.float64)
    peaks = peaks[np.isfinite(peaks)].astype(np.int64)
    peaks = peaks[(peaks >= 0) & (peaks < x.size)]
    return np.unique(peaks), False


def detect(ecg: np.ndarray, fs: float = FS, method: str = DETECTOR) -> np.ndarray:
    """R-peak sample indices (sorted, unique, 0-based) within one ECG window."""
    return detect_checked(ecg, fs, method)[0]


def rate_bpm(peaks: np.ndarray, fs: float = FS) -> float:
    """Mean rate over a window: 60 (n - 1) / (last - first). NaN with fewer than 2 peaks."""
    if len(peaks) < 2:
        return float("nan")
    span = (peaks[-1] - peaks[0]) / fs
    return float(60.0 * (len(peaks) - 1) / span) if span > 0 else float("nan")
