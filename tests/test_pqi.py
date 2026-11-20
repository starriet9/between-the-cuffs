"""Tests of the primary pleth quality index (SPEC.md §5) and the window grid."""
import numpy as np
import pytest

from btc import pqi

FS = pqi.FS
t = np.arange(pqi.WIN_N) / FS


def pulse(f=1.2):
    """A clean pulse-like 125 Hz window: fundamental plus two harmonics."""
    return 100 + 10 * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t + 0.5)
                       + 0.15 * np.sin(6 * np.pi * f * t + 1.0))


def test_clean_pulse_passes():
    r = pqi.assess_window(pulse())
    assert r["pqi_pass"]
    assert r["dom_freq"] == pytest.approx(1.2)
    assert r["peak_share"] > 0.15


def test_missing_and_flat_are_discarded():
    x = pulse()
    x[100] = np.nan
    r = pqi.assess_window(x)
    assert r["fail_missing"] and not r["pqi_pass"]
    r = pqi.assess_window(np.full(pqi.WIN_N, 7.0))
    assert r["fail_flat"] and not r["pqi_pass"]
    r = pqi.assess_window(pulse()[:-1])          # not a complete window
    assert not r["pqi_pass"]


@pytest.mark.parametrize("f,ok", [(0.6, False), (0.7, True), (3.0, True), (3.1, False), (4.0, False)])
def test_dominant_frequency_range(f, ok):
    x = 100 + 10 * np.sin(2 * np.pi * f * t)
    r = pqi.assess_window(x)
    assert r["dom_freq"] == pytest.approx(f)
    assert r["fail_freq"] == (not ok)


def test_dominant_frequency_searched_in_band_only():
    # a large 0.2 Hz component is outside 0.5-8 Hz and must not be the dominant bin
    x = pulse() + 200 * np.sin(2 * np.pi * 0.2 * t)
    assert pqi.assess_window(x)["dom_freq"] == pytest.approx(1.2)


def test_low_peak_share_fails():
    rng = np.random.default_rng(1)
    r = pqi.assess_window(rng.standard_normal(pqi.WIN_N))
    assert r["peak_share"] < 0.15 and r["fail_peak"] and not r["pqi_pass"]


def test_amplitude_criterion():
    x = pulse()
    x[600] += 50 * x.std()
    r = pqi.assess_window(x)
    assert r["max_abs_z"] > 6 and r["fail_amp"] and not r["pqi_pass"]
    # max |z| of a sinusoid is sqrt(2): passes
    assert not pqi.assess_window(pulse())["fail_amp"]


def test_constant_runs():
    x = pulse()
    y = x.copy()
    y[100:163] = y[100]          # one run of 63
    y[400:463] = y[400]          # and another: 126 >= 125 -> unusable
    r = pqi.assess_window(y)
    assert r["const_run_n"] == 126 and r["fail_const"]
    y = x.copy()
    y[100:224] = y[100]          # one run of 124 < 125 -> passes this criterion
    assert not pqi.assess_window(y)["fail_const"]
    y = x.copy()
    for s in range(0, 1240, 100):
        y[s:s + 62] = y[s]       # many runs of 62 (< 63) do not count
    assert pqi.assess_window(y)["const_run_n"] == 0


def test_constant_run_total():
    assert pqi.constant_run_total(np.array([1, 1, 1, 2, 2, 3.0]), run_min=2) == 5
    assert pqi.constant_run_total(np.array([1, 2, 3.0]), run_min=2) == 0


def test_window_grid():
    # 500 Hz record of 25 s + 3 samples: two complete 10-s windows from the first sample
    n_raw = 25 * 500 + 3
    assert pqi.n_windows(n_raw) == 2
    assert pqi.n_windows(2 * 5000 - 4) == 1      # the 125 Hz series has only 2,499 samples
    assert pqi.n_windows(2 * 5000 - 3) == 2      # 2,500 samples at 125 Hz
    st, en = pqi.window_times(3.25, 2)
    np.testing.assert_allclose(st, [3.25, 13.25])
    np.testing.assert_allclose(en, [13.25, 23.25])
    x = np.arange(n_raw, dtype=float)
    w1 = pqi.window_125(x, 1)
    assert w1.size == 1250 and w1[0] == 5000 and w1[1] == 5004 and w1[-1] == 5000 + 4 * 1249
