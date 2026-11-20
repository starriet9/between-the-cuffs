"""Tests of beat delineation, per-window features and per-forecast aggregation (SPEC §4.3, §5)."""
import numpy as np
import pandas as pd
import pytest

from btc import beats, pqi, provenance, waveform as W
from btc.paths import DERIVED, MANIFEST_SDEV, RAW_SDEV
from wf_synth import FS, Beat, make_signals


# --------------------------------------------------------------------------- delineation
def test_unfiltered_delineation_is_exact():
    bt = Beat()
    p, _, starts, _, _ = make_signals(30, bt, t0=0.0)
    seg = p[5000:10000] - 1000.0
    b = beats.delineate(seg)
    assert b["foot"].size >= 9
    foot_t = 10.0 + b["foot"] / FS
    k = np.searchsorted(starts, foot_t) - 1
    np.testing.assert_allclose(foot_t, starts[k] + bt.foot_offset, atol=1e-4)
    np.testing.assert_allclose(b["rise_s"], bt.rise, atol=1e-3)
    np.testing.assert_allclose(b["width_s"], bt.width, atol=1e-3)


def test_notch_candidate_found_without_a_notch():
    """Documents why the notch quantities are not model inputs (SPEC §4.3): on band-passed
    synthetic beats that have no dicrotic notch at all, the detector still returns a
    candidate in >= 90% of beats, so a detected point is no evidence of a notch."""
    for T in (0.6, 0.9, 1.1):
        p, e, *_ = make_signals(60, Beat(T=T, notch=False), noise=0.3, seed=5)
        df = W.case_windows(1, W.Wave(p, 1.0, FS), W.Wave(e, 1.0, FS))
        assert df["n_beats"].sum() > 0
        assert df["n_notch"].sum() >= 0.9 * df["n_beats"].sum(), T


def test_notch_is_window_level_only():
    assert set(beats.QC_FEATURES) == {"notch_t", "notch_ratio"}
    assert not set(beats.QC_FEATURES) & set(beats.FEATURES)
    assert set(beats.WINDOW_FEATURES) == set(beats.FEATURES) | set(beats.QC_FEATURES)
    assert set(beats.WINDOW_FEATURES) <= set(W.WINDOW_COLS)
    assert not any("notch" in c for c in W.FEATURE_COLS)
    r = W.forecast_features(W.CaseWindows(_table()), 200.0, 155.0, 100.0, 200.0)
    assert not any("notch" in k for k in r)


def test_tangent_foot_not_before_foot_valley():
    # a jump right after a valley pushes the tangent intersection before the valley: no beat there
    p, *_ = make_signals(10.0, Beat(), t0=0.0)
    xf = beats.bandpass(p[:5000])
    b = beats.delineate(xf)
    v = int(b["v_foot"][2])
    xj = xf.copy()
    xj[v + 1:] += 0.6 * (xf[int(b["peak"][2])] - xf[v])      # step just after that valley
    bj = beats.delineate(xj)
    assert np.all(bj["foot"] >= bj["v_foot"])
    assert v not in set(bj["v_foot"].astype(int))
    rng = np.random.default_rng(11)
    for _ in range(20):                                       # and on noisy windows
        q = beats.bandpass(p[:5000] + 15.0 * rng.standard_normal(5000))
        bq = beats.delineate(q)
        assert np.all(bq["foot"] >= bq["v_foot"])


def test_pair_pat_unique_r_in_lag_range():
    b = {"foot": np.array([500.0, 1000.0, 1500.0])}           # feet at 1, 2, 3 s
    r = np.array([0.38, 1.38, 1.55, 2.70])
    pat = beats.pair_pat(b, r, t_start=0.0)
    # foot 1 s: lags 0.62 -> one; foot 2 s: lags 0.62 and 0.45 -> two -> missing;
    # foot 3 s: lags 0.30 (out), 1.40 (out), 1.62 (out) -> none -> missing
    np.testing.assert_allclose(pat, [0.62, np.nan, np.nan])
    assert np.isnan(beats.pair_pat(b, np.zeros(0), 0.0)).all()


def _windows(bt, pat=0.62, amp_mod=0.1, dur=120.0, noise=0.0):
    p, e, starts, r, amps = make_signals(dur, bt, pat=pat, amp_mod=amp_mod, noise=noise)
    return W.case_windows(1, W.Wave(p, 1.0, FS), W.Wave(e, 1.0, FS)), starts, amps


def test_full_pipeline_recovers_known_values():
    bt = Beat()
    df, starts, amps = _windows(bt)
    assert df.pqi_pass.all() and df.usable.all() and df.usable_strict.all()
    assert (df.n_beats >= 9).all() and (df.n_pat >= 8).all()
    m = df.median(numeric_only=True)
    # band-limited (0.5-8 Hz) versions of the synthetic shape: small, known-direction shifts
    assert m["pat"] == pytest.approx(0.62, abs=0.010)
    assert m["rise"] == pytest.approx(bt.rise, abs=0.020)
    assert m["width"] == pytest.approx(bt.width, abs=0.010)
    assert m["bav"] == pytest.approx(np.std(amps, ddof=1) / np.mean(amps), abs=0.01)
    assert m["pleth_rate"] == pytest.approx(60 / bt.T, abs=0.5)
    assert m["r_rate"] == pytest.approx(60 / bt.T, abs=0.5)


def test_full_pipeline_tracks_changes():
    base, _, _ = _windows(Beat())
    m0 = base.median(numeric_only=True)
    # PAT: the ECG moved 30 ms earlier -> PAT longer by 30 ms
    m1 = _windows(Beat(), pat=0.65)[0].median(numeric_only=True)
    assert m1["pat"] - m0["pat"] == pytest.approx(0.030, abs=0.002)
    # slower upstroke: rise time longer
    b3 = Beat(R=0.18)
    m3 = _windows(b3)[0].median(numeric_only=True)
    assert m3["rise"] - m0["rise"] == pytest.approx(b3.rise - Beat().rise, abs=0.012)
    # longer systolic descent: wider at half height
    b4 = Beat(N=0.25)
    m4 = _windows(b4)[0].median(numeric_only=True)
    assert m4["width"] - m0["width"] == pytest.approx(b4.width - Beat().width, abs=0.015)
    # no amplitude modulation: CV near 0 (window-edge effects of the zero-phase filter remain)
    m5 = _windows(Beat(), amp_mod=0.0)[0].median(numeric_only=True)
    assert m5["bav"] < 0.02 < m0["bav"] - 0.04


def test_amplitude_scale_does_not_change_features():
    bt = Beat()
    p, e, *_ = make_signals(60, bt)
    a = W.case_windows(1, W.Wave(p, 1.0, FS), W.Wave(e, 1.0, FS))
    b = W.case_windows(1, W.Wave(3.0 * p + 17.0, 1.0, FS), W.Wave(e, 1.0, FS))
    cols = list(beats.FEATURES) + ["n_beats", "usable", "tcorr"]
    pd.testing.assert_frame_equal(a[cols], b[cols], rtol=1e-6, atol=1e-9)


def test_few_beats_not_usable():
    # beats only in the first 3 s, then a still baseline: fewer than 4 beats (SPEC §5)
    p, e, *_ = make_signals(10.0, Beat(), t0=0.0)
    p[1500:5000] = 1000.0
    r = beats.window_features(p[:5000], np.zeros(0), 0.0)
    assert 0 < r["n_beats"] < beats.MIN_BEATS


def test_missing_ecg_and_missing_pleth():
    bt = Beat()
    p, e, *_ = make_signals(60, bt)
    e2 = e.copy()
    e2[7000] = np.nan                             # second window's ECG incomplete
    df = W.case_windows(1, W.Wave(p, 1.0, FS), W.Wave(e2, 1.0, FS))
    assert not df.loc[1, "ecg_ok"] and np.isnan(df.loc[1, "pat"]) and df.loc[1, "usable"]
    df = W.case_windows(1, W.Wave(p, 1.0, FS), None)  # ECG absent (SPEC §2.3)
    assert df["pat"].isna().all() and df["usable"].all()
    p2 = p.copy()
    p2[5001] = np.nan                             # a 500 Hz sample not on the 125 Hz grid
    df = W.case_windows(1, W.Wave(p2, 1.0, FS), W.Wave(e, 1.0, FS))
    assert df.loc[1, "pqi_pass"] and df.loc[1, "n_beats"] == 0 and not df.loc[1, "usable"]
    p2[5004] = np.nan                             # on the grid: discarded by the index
    df = W.case_windows(1, W.Wave(p2, 1.0, FS), W.Wave(e, 1.0, FS))
    assert df.loc[1, "fail_missing"] and not df.loc[1, "pqi_pass"]


def test_template_correlation():
    bt = Beat()
    p, e, *_ = make_signals(30, bt, amp_mod=0.0)
    xf = beats.bandpass(p[:5000])
    assert beats.template_corr(xf, beats.delineate(xf)) > 0.99
    rng = np.random.default_rng(3)
    q = p[:5000] + rng.standard_normal(5000) * 30.0
    xq = beats.bandpass(q)
    tc = beats.template_corr(xq, beats.delineate(xq))
    assert np.isnan(tc) or tc < 0.99


# --------------------------------------------------------------------------- aggregation
def _table(n=40, usable=None, vals=None, t0=0.0):
    """Hand-made window table: windows [t0 + 10 w, t0 + 10 w + 10)."""
    w = np.arange(n)
    df = pd.DataFrame({"w": w, "t_start": t0 + 10.0 * w, "t_end": t0 + 10.0 * w + 10.0})
    df["usable"] = True if usable is None else usable
    df["usable_strict"] = df["usable"]
    for j, f in enumerate(beats.FEATURES):
        df[f] = (np.arange(n) * 1.0 + j) if vals is None else vals
    return df


def test_window_membership_and_lookback():
    cw = W.CaseWindows(_table())
    # windows start >= 155 and end <= 200: starts 160, 170, 180, 190 -> 4 windows
    r = W.forecast_features(cw, t_issue=200.0, t_prev=155.0, t_prev2=100.0, t_end=200.0)
    assert r["wf_n_windows"] == 4 and r["wf_n_usable"] == 4 and r["wf_available"]
    # lookback capped at 120 s: previous reading 10 s earlier than 120 s back
    r = W.forecast_features(cw, t_issue=300.0, t_prev=170.0, t_prev2=100.0, t_end=300.0)
    assert r["t_lo"] == 180.0 and r["wf_n_windows"] == 12
    # t_issue inside a window: that window is not used (its samples reach past t_issue)
    r = W.forecast_features(cw, t_issue=205.0, t_prev=155.0, t_prev2=100.0, t_end=205.0)
    assert r["wf_n_windows"] == 4


def test_levels_and_changes():
    cw = W.CaseWindows(_table())
    r = W.forecast_features(cw, t_issue=200.0, t_prev=155.0, t_prev2=100.0, t_end=200.0)
    # current: windows 16..19 -> median 17.5 (+ j); reference: windows 0..5 -> 2.5 (+ j)
    # previous: start >= max(100, 35) and end <= 155: windows 10..14 -> 12 (+ j)
    assert r["wf_pat_lvl"] == pytest.approx(17.5 - 2.5)
    assert r["wf_pat_chg"] == pytest.approx(17.5 - 12.0)
    assert r["wf_bav_lvl"] == pytest.approx(15.0)


def test_reference_not_before_six_windows_end():
    use = np.zeros(40, bool)
    use[[3, 5, 8, 9, 12, 20, 21, 22]] = True
    cw = W.CaseWindows(_table(usable=use))
    # 6th usable window is w = 20, ending at 210
    r = W.forecast_features(cw, 209.0, 150.0, 100.0, 209.0)
    assert np.isnan(r["wf_pat_lvl"]) and not r["wf_available"]
    r = W.forecast_features(cw, 230.0, 190.0, 100.0, 230.0)
    assert r["wf_available"] and r["wf_n_usable"] == 3       # windows 20, 21, 22
    assert r["wf_pat_lvl"] == pytest.approx(21.0 - np.median([3, 5, 8, 9, 12, 20]))
    assert np.isnan(r["wf_pat_chg"])                         # only window 12 in [100, 190)


def test_level_missing_until_reference_windows_end():
    use = np.zeros(40, bool)
    use[16:22] = True                                  # usable windows 16..21; the 6th ends at 220 s
    cw = W.CaseWindows(_table(usable=use))
    for t in (200.0, 219.9):                           # 4 and 5 usable windows have ended
        r = W.forecast_features(cw, t, 155.0, 100.0, t)
        assert r["wf_available"] and r["wf_n_usable"] >= 4
        assert all(np.isnan(r[f"wf_{f}_lvl"]) for f in beats.FEATURES)
    r = W.forecast_features(cw, 220.0, 155.0, 100.0, 220.0)
    assert all(np.isfinite(r[f"wf_{f}_lvl"]) for f in beats.FEATURES)
    # inflation-artefact cutoff: the 6th window must end by t_end, not by t_issue
    r = W.forecast_features(cw, 240.0, 155.0, 100.0, 210.0)
    assert r["wf_available"] and np.isnan(r["wf_rise_lvl"])


def test_ecg_window_bounds():
    ecg = W.Wave(np.arange(60000, dtype=float), 0.3137, 500.0)
    for ts in (1.0687, 11.0687, 0.3137, 21.07):
        i0, x = W._ecg_window(ecg, ts, ts + 10.0)
        t = ecg.times(i0 + np.arange(x.size))
        assert x.size == pqi.WIN_N_RAW and t[0] >= ts - 1e-9 and t[-1] < ts + 10.0
        assert i0 == 0 or ecg.times(i0 - 1) < ts
    assert W._ecg_window(ecg, 0.0, 10.0) is None              # starts before the first sample
    assert W._ecg_window(ecg, 115.0, 125.0) is None           # runs past the last sample
    assert W._ecg_window(None, 1.0, 11.0) is None


def test_fewer_than_two_usable_windows():
    use = np.ones(40, bool)
    use[17:20] = False
    cw = W.CaseWindows(_table(usable=use))
    r = W.forecast_features(cw, 200.0, 155.0, 100.0, 200.0)
    assert r["wf_n_windows"] == 4 and r["wf_n_usable"] == 1 and not r["wf_available"]
    assert r["wf_usable_share"] == pytest.approx(0.25)
    assert all(np.isnan(r[c]) for c in W.FEATURE_COLS)


def test_feature_needs_two_window_values():
    vals = np.arange(40, dtype=float)
    vals[17:20] = np.nan                                     # only window 16 has a value
    cw = W.CaseWindows(_table(vals=vals))
    r = W.forecast_features(cw, 200.0, 155.0, 100.0, 200.0)
    assert r["wf_available"] and np.isnan(r["wf_pat_lvl"]) and np.isnan(r["wf_pat_chg"])


def test_no_windows():
    cw = W.CaseWindows(None)
    r = W.forecast_features(cw, 200.0, 155.0, 100.0, 200.0)
    assert r["wf_n_windows"] == 0 and np.isnan(r["wf_usable_share"]) and not r["wf_available"]


def test_variant_times():
    assert W.variant_times("real", 700, 600, 500) == (700, 600, 500, 700, "usable")
    assert W.variant_times("p1", 700, 600, 500) == (700, 600, 500, 700, "usable")
    assert W.variant_times("p2", 700, 600, 500) == (100, 0, -100, 100, "usable")
    assert W.variant_times("ia", 700, 600, 500) == (700, 600, 500, 670, "usable")
    assert W.variant_times("strict", 700, 600, 500) == (700, 600, 500, 700, "usable_strict")
    cw = W.CaseWindows(_table())
    r = W.forecast_features(cw, *W.variant_times("ia", 200.0, 155.0, 100.0)[:4])
    assert r["wf_n_windows"] == 1                            # only [160, 170) ends by 170


def test_present_follows_io_status():
    from btc import io
    assert W.present(io.OK) and W.present(io.KEPT_FLAGGED)
    assert not W.present(io.ABSENT_IN_FILE) and not W.present(io.ABSENT_INTEGRITY)


@pytest.mark.skipif(not MANIFEST_SDEV.exists(), reason="S-dev data not downloaded")
def test_load_case_applies_integrity_rule():
    # SPEC §2.3: 2328 both waveforms absent (broken clock); 2030 ECG absent from the file
    if not all((RAW_SDEV / f"{c}.npz").exists() for c in (2328, 2030)):
        pytest.skip("case files missing")
    d = W.load_case(2328)
    assert d["pleth"] is None and d["ecg"] is None
    d = W.load_case(2030)
    assert d["pleth"] is not None and d["ecg"] is None
    df = W.case_windows(2030, d["pleth"], d["ecg"], d["hr"])
    assert len(df) > 0 and df["pat"].isna().all() and not df["ecg_ok"].any()


# --------------------------------------------------------------------------- derived table
FWF = DERIVED / "sdev_features_wf.parquet"


@pytest.mark.skipif(not (FWF.exists() and (DERIVED / "sdev_instances.parquet").exists()),
                    reason="run scripts/build_features_wf.py first")
def test_features_wf_table_consistent():
    """btc/SCHEMA.md §4: five variants per issue point, variant cutoffs, P1 donors."""
    f = pd.read_parquet(FWF)
    inst = pd.read_parquet(DERIVED / "sdev_instances.parquet", columns=["caseid", "window", "issue_n"])
    cases = pd.read_parquet(DERIVED / "sdev_cases.parquet", columns=["caseid", "donor_caseid"])
    keys = inst.drop_duplicates()
    assert not any("notch" in c for c in f.columns)
    assert set(W.FEATURE_COLS) <= set(f.columns)
    assert not f.duplicated(["caseid", "window", "issue_n", "variant"]).any()
    assert len(f) == len(W.VARIANTS) * len(keys)
    per = f.groupby(["caseid", "window", "issue_n"])["variant"].apply(frozenset)
    assert (per == frozenset(W.VARIANTS)).all() and len(per) == len(keys)
    cut = {"real": 0.0, "p1": 0.0, "strict": 0.0, "ia": -30.0, "p2": -600.0}
    d = f["t_end"] - f["t_issue"]
    for v, c in cut.items():
        np.testing.assert_allclose(d[f["variant"] == v], c, atol=1e-9)
    shift = f["time_shift_s"].where(f["variant"] == "p2", 0.0)
    np.testing.assert_allclose(f["time_shift_s"], shift, atol=1e-9)
    np.testing.assert_allclose(f.loc[f["variant"] == "p2", "time_shift_s"], -600.0)
    p1 = f[f["variant"] == "p1"].merge(cases, on="caseid", how="left")
    assert np.array_equal(p1["signal_caseid"].to_numpy(), p1["donor_caseid"].to_numpy(), equal_nan=True)
    own = f[f["variant"] != "p1"]
    assert (own["signal_caseid"] == own["caseid"]).all()
    assert (f["wf_available"] == (f["wf_n_usable"] >= 2)).all()
    assert (f["wf_n_usable"] <= f["wf_n_windows"]).all()
    na = ~f["wf_available"]
    assert f.loc[na, W.FEATURE_COLS].isna().all().all()
    assert provenance.read_meta(FWF).get("git", {}).get("commit")
