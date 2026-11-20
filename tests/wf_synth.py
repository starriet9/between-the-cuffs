"""Synthetic pleth and ECG with known fiducial points, for the waveform tests.

Each pleth beat starts at its foot f (value 0, zero slope) and is built from raised-cosine
pieces, scaled by the beat amplitude A:
- upstroke 0 -> A over R s; the tangent at its steepest point (midpoint, slope A pi / 2R)
  meets the foot level at f + R (1/2 - 1/pi);
- descent A -> n A over N s, ending in a local minimum (the dicrotic notch, where the
  second derivative is largest after the peak);
- diastolic rise n A -> d A over Dr s; decay d A -> 0 over the rest of the period T.
Known values: rise = R (1/2 + 1/pi); notch time = N after the peak; notch ratio = n;
width = R/2 + (N/pi) arccos(2 (0.5 - n) / (1 - n) - 1) for n < 0.5.
ECG: a narrow Gaussian R wave and a broad T wave per beat; R at tangent foot - PAT.
"""
from __future__ import annotations

import numpy as np

FS = 500.0


class Beat:
    def __init__(self, T=0.9, R=0.14, N=0.20, n=0.35, Dr=0.10, d=0.45, notch=True):
        self.T, self.R, self.N, self.n, self.Dr, self.d, self.notch = T, R, N, n, Dr, d, notch

    def shape(self, s: np.ndarray) -> np.ndarray:
        """Unit-amplitude beat at times s since the foot, 0 <= s < T."""
        T, R, N, n, Dr, d = self.T, self.R, self.N, self.n, self.Dr, self.d
        x = np.zeros_like(s)
        m = s < R
        x[m] = 0.5 * (1 - np.cos(np.pi * s[m] / R))
        if not self.notch:                       # one smooth descent, no notch or shoulder
            L = T - R
            m = s >= R
            x[m] = 0.5 * (1 + np.cos(np.pi * (s[m] - R) / L))
            return x
        m = (s >= R) & (s < R + N)
        x[m] = n + (1 - n) * 0.5 * (1 + np.cos(np.pi * (s[m] - R) / N))
        m = (s >= R + N) & (s < R + N + Dr)
        x[m] = n + (d - n) * 0.5 * (1 - np.cos(np.pi * (s[m] - R - N) / Dr))
        L = T - R - N - Dr
        m = s >= R + N + Dr
        x[m] = d * 0.5 * (1 + np.cos(np.pi * (s[m] - R - N - Dr) / L))
        return x

    @property
    def foot_offset(self):          # tangent foot after the beat start
        return self.R * (0.5 - 1 / np.pi)

    @property
    def rise(self):
        return self.R * (0.5 + 1 / np.pi)

    @property
    def width(self):
        n = self.n
        u = self.N / np.pi * np.arccos(2 * (0.5 - n) / (1 - n) - 1)
        return self.R / 2 + u


def make_signals(duration=600.0, beat: Beat | None = None, pat=0.62, t0=1.0, amp_mod=0.0,
                 seed=0, noise=0.0):
    """Pleth and ECG sampled at 500 Hz from t0; returns (pleth, ecg, starts, r_times)."""
    beat = beat or Beat()
    n = int(round(duration * FS))
    t = t0 + np.arange(n) / FS
    starts = t0 + 0.3 + np.arange(int(duration / beat.T) + 2) * beat.T
    starts = np.round(starts * FS) / FS          # beat starts on the sample grid
    amps = 1.0 + amp_mod * np.sin(2 * np.pi * np.arange(starts.size) / 7.0)
    pleth = np.zeros(n)
    k = np.searchsorted(starts, t, side="right") - 1
    ok = k >= 0
    s = t[ok] - starts[k[ok]]
    pleth[ok] = amps[k[ok]] * beat.shape(s)
    pleth = 1000.0 + 50.0 * pleth                # arbitrary display scale and offset
    r_times = starts + beat.foot_offset - pat
    ecg = np.zeros(n)
    for r in r_times:
        i0, i1 = np.searchsorted(t, [r - 0.1, r + 0.5])
        tt = t[i0:i1]
        ecg[i0:i1] += np.exp(-0.5 * ((tt - r) / 0.008) ** 2) \
            + 0.25 * np.exp(-0.5 * ((tt - r - 0.25) / 0.04) ** 2)
    if noise:
        rng = np.random.default_rng(seed)
        pleth = pleth + noise * rng.standard_normal(n)
        ecg = ecg + 0.02 * rng.standard_normal(n)
    return pleth, ecg, starts, r_times, amps
