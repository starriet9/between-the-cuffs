"""Leakage test for the waveform features (SPEC.md §4.4, PREREG.md §3).

Every pleth and ECG sample at or after the cutoff t_end of a forecast is replaced with
random values; the window table is recomputed from the perturbed signals, and every
waveform feature of that forecast must be unchanged, for each variant (real, p1, p2, ia,
strict). Run on synthetic signals and on several real S-dev cases.

The yardstick clause of §4.4 (the withheld values of reading r+1) holds by construction:
waveform features take reading *times* only, never cuff values (checked below).
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from btc import waveform as W
from btc.paths import DERIVED, MANIFEST_SDEV, RAW_SDEV
from wf_synth import FS, Beat, make_signals

REAL_CASES = (102, 705, 4324)
TRUNC_S = 1900.0          # real cases: only the first 1,900 s are processed (both runs)


def _perturb(w: W.Wave | None, t_end: float, rng) -> W.Wave | None:
    if w is None:
        return None
    x = w.x.copy()
    k = w.t0 + np.arange(x.size) / w.srate >= t_end
    sd = np.nanstd(w.x) or 1.0
    x[k] = np.nanmean(w.x) + 3 * sd * rng.standard_normal(int(k.sum()))
    x[k & (rng.random(x.size) < 0.001)] = np.nan          # some missing samples too
    return W.Wave(x, w.t0, w.srate)


def _features(src: dict, times, variant):
    ti, tp, tpp, te, mask = W.variant_times(variant, *times)
    cw = W.CaseWindows(W.case_windows(0, src["pleth"], src["ecg"], src.get("hr")))
    return W.forecast_features(cw, ti, tp, tpp, te, mask), te


def _check(own: dict, donor: dict, issues, seed=0):
    rng = np.random.default_rng(seed)
    n_avail = n_lvl = 0
    for times in issues:
        for variant in W.VARIANTS:
            src = donor if variant == "p1" else own
            before, te = _features(src, times, variant)
            pert = {"pleth": _perturb(src["pleth"], te, rng), "ecg": _perturb(src["ecg"], te, rng)}
            after, _ = _features(pert, times, variant)
            assert before.keys() == after.keys()
            for k in before:
                a, b = before[k], after[k]
                assert (a == b) or (np.isnan(a) and np.isnan(b)), (variant, times, k, a, b)
            n_avail += before["wf_available"]
            n_lvl += np.isfinite(before["wf_pat_lvl"]) and np.isfinite(before["wf_rise_chg"])
    return n_avail, n_lvl


def test_feature_inputs_are_times_only():
    params = list(inspect.signature(W.forecast_features).parameters)
    assert params == ["cw", "t_issue", "t_prev", "t_prev2", "t_end", "mask"]


def test_leakage_synthetic():
    p, e, *_ = make_signals(900, Beat(), noise=0.3, seed=1)
    own = {"pleth": W.Wave(p, 1.0, FS), "ecg": W.Wave(e, 1.0, FS)}
    p2, e2, *_ = make_signals(900, Beat(T=0.75, N=0.18), pat=0.6, t0=0.4, noise=0.3, seed=2)
    donor = {"pleth": W.Wave(p2, 0.4, FS), "ecg": W.Wave(e2, 0.4, FS)}
    issues = [(300.4, 211.9, 150.2), (455.1, 380.0, 300.4), (612.7, 455.1, 380.0),
              (800.0, 700.0, 612.7), (845.3, 832.0, 800.0)]
    n_avail, n_lvl = _check(own, donor, issues)
    # 17 of 25 forecasts have waveform features (the 13-s interval and early P2 have none)
    assert n_avail >= 15 and n_lvl >= 12          # the test exercises real feature values


def test_leakage_synthetic_early_issue():
    """Issue points before the reference is established (SPEC §4.3: level missing until the first
    6 usable windows end by the cutoff); perturbing later samples must not supply it."""
    p, e, *_ = make_signals(200, Beat(), noise=0.3, seed=1)
    own = {"pleth": W.Wave(p, 1.0, FS), "ecg": W.Wave(e, 1.0, FS)}
    n_avail, _ = _check(own, own, [(45.3, 12.0, 3.0), (58.0, 30.5, 12.0)])
    assert n_avail >= 4


def _real_case(caseid):
    d = W.load_case(caseid)
    for k in ("pleth", "ecg"):
        w = d[k]
        if w is not None:
            n = int(max(0, min(w.x.size, np.ceil((TRUNC_S - w.t0) * w.srate))))
            d[k] = W.Wave(w.x[:n], w.t0, w.srate)
    return d


def _real_issues(caseid):
    """Issue points per case from the real reading times, if the cuff tables exist: always the
    first issue point (before the reference is established), the first with issue_n >= 15 and
    the last."""
    try:
        inst = pd.read_parquet(DERIVED / "sdev_instances.parquet",
                               columns=["caseid", "window", "issue_n", "t_issue", "t_prev"])
        rd = pd.read_parquet(DERIVED / "sdev_readings.parquet", columns=["caseid", "n", "t", "valid"])
    except (FileNotFoundError, OSError):
        return [(45.2, 20.1, 5.0), (700.3, 560.1, 410.0), (1500.7, 1350.2, 1200.4)]
    i = inst[(inst.caseid == caseid) & (inst.window == "primary")].drop_duplicates("issue_n")
    i = i.sort_values("issue_n")
    r = rd[(rd.caseid == caseid) & rd.valid].set_index("n")["t"]
    first = (float(r.loc[3]), float(r.loc[2]), float(r.loc[1]))   # valid reading 3 (SPEC §3)
    pick = [len(i) - 1]
    late = np.flatnonzero(i.issue_n.to_numpy() >= 15)
    if late.size:
        pick.insert(0, int(late[0]))
    i = i.iloc[sorted(set(pick))]
    rest = [(float(a), float(b), float(r.loc[n - 2])) for a, b, n in zip(i.t_issue, i.t_prev, i.issue_n)]
    return [first] + [x for x in rest if x != first]


@pytest.mark.skipif(not MANIFEST_SDEV.exists(), reason="S-dev data not downloaded")
@pytest.mark.parametrize("caseid", REAL_CASES)
def test_leakage_real(caseid):
    donor_id = REAL_CASES[(REAL_CASES.index(caseid) + 1) % len(REAL_CASES)]
    if not all((RAW_SDEV / f"{c}.npz").exists() for c in (caseid, donor_id)):
        pytest.skip("case files missing")
    own, donor = _real_case(caseid), _real_case(donor_id)
    n_avail, n_lvl = _check(own, donor, _real_issues(caseid), seed=caseid)
    assert n_avail >= 4
