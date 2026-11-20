"""Cuff-reading rule (SPEC §2.4) on synthetic held-value tracks."""
import numpy as np
import pytest

from btc import cuff

DT = 2.0                          # monitor writes a numeric sample every 2 s
OFFSETS = {"sbp": 0.0, "dbp": 0.003, "mbp": 0.006}   # tracks have their own clocks


def held(changes, t_end=1200.0, t_start=0.0, offset=0.0, drop=(), nan=()):
    """Held-value track sampled every 2 s. `changes` = [(t_change, value), ...]:
    the value from t_change on. Samples in `drop` intervals are absent; samples in
    `nan` intervals carry the missing code (NaN). No sample before the first change."""
    t = np.arange(t_start, t_end, DT) + offset
    ct = np.array([c[0] for c in changes], dtype=float)
    cv = np.array([c[1] for c in changes], dtype=float)
    idx = np.searchsorted(ct, t, side="right") - 1
    keep = idx >= 0
    t, v = t[keep], cv[idx[keep]]
    for a, b in drop:
        m = (t >= a) & (t < b)
        t, v = t[~m], v[~m]
    v = v.copy()
    for a, b in nan:
        v[(t >= a) & (t < b)] = np.nan
    return t, v


def tracks_from_readings(readings, lag=None, **kw):
    """Three tracks from readings [(t, sbp, dbp, map), ...]; `lag` delays a track's
    changes by the given seconds."""
    lag = lag or {}
    out = {}
    for j, k in enumerate(("sbp", "dbp", "mbp")):
        ch = [(r[0] + lag.get(k, 0.0), r[1 + j]) for r in readings]
        out[k] = held(ch, offset=OFFSETS[k], **kw.get(k, {}))
    return out


def first_sample_at_or_after(t, x):
    return t[np.searchsorted(t, x, side="left")]


def test_basic_readings_and_times():
    rd = [(10.0, 120, 70, 90), (100.0, 118, 69, 88), (250.0, 110, 60, 80)]
    tr = tracks_from_readings(rd)
    r = cuff.extract_readings(tr)
    assert len(r) == 3 and r["valid"].all()
    for i, (t0, s, d, m) in enumerate(rd):
        # all three tracks change at the first tick at or after t0; t_r = the latest of them
        exp = max(first_sample_at_or_after(tr[k][0], t0) for k in tr)
        assert r["t"].iloc[i] == pytest.approx(exp)
        assert (r["sbp"].iloc[i], r["dbp"].iloc[i], r["map"].iloc[i]) == (s, d, m)
    assert list(r["n"]) == [1, 2, 3]


def test_merge_takes_last_change_and_held_values():
    # SBP changes at 100, MBP 4 s later, DBP 10 s later: one reading at DBP's change
    rd = [(10.0, 120, 70, 90), (100.0, 140, 80, 100)]
    tr = tracks_from_readings(rd, lag={"mbp": 4.0, "dbp": 10.0})
    r = cuff.extract_readings(tr)
    assert len(r) == 2
    exp = first_sample_at_or_after(tr["dbp"][0], 110.0)
    assert r["t"].iloc[1] == pytest.approx(exp)
    assert (r["sbp"].iloc[1], r["dbp"].iloc[1], r["map"].iloc[1]) == (140, 80, 100)


def test_chain_merge_beyond_15s_span():
    # each change within 15 s of the previous one, total span 24 s -> one reading
    tr = {"sbp": held([(0, 120), (100, 130)], offset=0.0),
          "mbp": held([(0, 90), (112, 95)], offset=0.0),
          "dbp": held([(0, 70), (124, 75)], offset=0.0)}
    r = cuff.extract_readings(tr)
    assert len(r) == 2
    assert r["t"].iloc[1] == pytest.approx(124.0)
    assert (r["sbp"].iloc[1], r["dbp"].iloc[1], r["map"].iloc[1]) == (130, 75, 95)


@pytest.mark.parametrize("gap, n_expected", [(15.0, 2), (16.0, 3)])
def test_merge_boundary_15s(gap, n_expected):
    # a change exactly 15 s after the previous change joins it; 16 s does not
    # DBP samples on the odd-second grid so that a change at 115 s is sampled exactly
    tr = {"sbp": held([(0, 120), (100, 130)]),
          "dbp": held([(0, 70), (100 + gap, 75)], offset=gap % 2.0),
          "mbp": held([(0, 90)])}
    r = cuff.extract_readings(tr)
    assert len(r) == n_expected
    assert np.all(np.diff(r["t"]) > cuff.MERGE_S)


def test_readings_always_more_than_15s_apart():
    rng = np.random.default_rng(0)
    ch = np.sort(rng.uniform(0, 1000, 60))
    tr = {k: held([(0, 100 + j)] + [(c, 100 + j + i + 1) for i, c in enumerate(ch[j::3])])
          for j, k in enumerate(("sbp", "dbp", "mbp"))}
    r = cuff.extract_readings(tr)
    assert np.all(np.diff(r["t"]) > cuff.MERGE_S)


def test_first_value_counts_at_first_sample():
    tr = {k: held([(37.0, v)], t_start=37.0, offset=OFFSETS[k])
          for k, v in (("sbp", 120), ("dbp", 70), ("mbp", 90))}
    r = cuff.extract_readings(tr)
    assert len(r) == 1
    assert r["t"].iloc[0] == pytest.approx(max(tr[k][0][0] for k in tr))
    assert r["valid"].iloc[0]


def test_return_unchanged_after_missing_is_not_a_reading():
    rd = [(10.0, 120, 70, 90), (500.0, 110, 65, 85)]
    gap = {"drop": [(100.0, 300.0)]}
    tr = tracks_from_readings(rd, sbp=gap, dbp=gap, mbp=gap)
    r = cuff.extract_readings(tr)
    assert len(r) == 2
    # the same with the missing-value code (NaN) instead of absent samples
    nan = {"nan": [(100.0, 300.0)]}
    r2 = cuff.extract_readings(tracks_from_readings(rd, sbp=nan, dbp=nan, mbp=nan))
    assert len(r2) == 2 and np.allclose(r["t"], r2["t"])


def test_change_after_missing_is_a_reading_at_first_sample():
    rd = [(10.0, 120, 70, 90), (200.0, 110, 65, 85)]       # changed during the gap
    gap = {"drop": [(150.0, 300.0)]}
    tr = tracks_from_readings(rd, sbp=gap, dbp=gap, mbp=gap)
    r = cuff.extract_readings(tr)
    assert len(r) == 2
    exp = max(first_sample_at_or_after(tr[k][0], 300.0) for k in tr)
    assert r["t"].iloc[1] == pytest.approx(exp)
    assert (r["sbp"].iloc[1], r["dbp"].iloc[1], r["map"].iloc[1]) == (110, 65, 85)


def test_value_continuing_after_gap_is_held():
    # all tracks change at 100, then no samples for 100 s, then the same values
    rd = [(10.0, 120, 70, 90), (100.0, 133, 80, 100)]
    gap = {"drop": [(102.0, 202.0)]}
    tr = tracks_from_readings(rd, sbp=gap, dbp=gap, mbp=gap)
    r = cuff.extract_readings(tr)
    assert len(r) == 2 and r["valid"].all()
    assert (r["sbp"].iloc[1], r["dbp"].iloc[1], r["map"].iloc[1]) == (133, 80, 100)


def test_mbp_only_change_during_sbp_dbp_gap_is_incomplete():
    # SBP/DBP stop at 100 while MBP changes; SBP/DBP return at 250 with new values
    tr = {"sbp": held([(0, 120), (250, 125)], drop=[(100.0, 250.0)]),
          "dbp": held([(0, 70), (250, 72)], drop=[(100.0, 250.0)]),
          "mbp": held([(0, 90), (100, 95), (250, 93)])}
    r = cuff.extract_readings(tr)
    assert len(r) == 3
    assert r["drop_reason"].iloc[1] == cuff.DROP_INCOMPLETE and not r["valid"].iloc[1]
    assert r["valid"].iloc[2] and r["map"].iloc[2] == 93
    assert list(r["n"].fillna(-1)) == [1, -1, 2]


def test_no_sample_after_last_change_is_incomplete():
    t = np.arange(0.0, 42.0, 2.0)          # last sample at 40 s holds a new value
    tr = {k: (t, np.where(t < 40.0, v, v + 5)) for k, v in
          (("sbp", 120.0), ("dbp", 70.0), ("mbp", 90.0))}
    r = cuff.extract_readings(tr)
    assert len(r) == 2 and r["drop_reason"].iloc[1] == cuff.DROP_INCOMPLETE


def test_joint_detection_when_map_unchanged():
    tr = {"sbp": held([(0, 120), (100, 140)]), "dbp": held([(0, 70), (100, 60)]),
          "mbp": held([(0, 90)])}
    r = cuff.extract_readings(tr)
    assert len(r) == 2 and r["map"].iloc[1] == 90 and r["sbp"].iloc[1] == 140


@pytest.mark.parametrize("vals, reason", [
    ((120, 70, 29), cuff.DROP_MAP_RANGE),
    ((120, 20, 30), ""),
    ((180, 100, 160), ""),
    ((200, 100, 161), cuff.DROP_MAP_RANGE),
    ((90, 70, 90), cuff.DROP_ORDER),      # SBP > MAP fails
    ((120, 90, 90), cuff.DROP_ORDER),     # MAP > DBP fails
    ((80, 70, 90), cuff.DROP_ORDER),
])
def test_exclusions(vals, reason):
    s, d, m = vals
    rd = [(10.0, 121, 71, 91), (100.0, s, d, m), (200.0, 122, 72, 92)]
    r = cuff.extract_readings(tracks_from_readings(rd))
    assert len(r) == 3
    assert r["drop_reason"].iloc[1] == reason
    assert r["valid"].iloc[1] == (reason == "")
    # dropped readings are skipped by the valid index n
    assert r["n"].iloc[2] == (3 if reason == "" else 2)


def test_gap_flags_use_valid_intervals_and_case_median():
    times = [10, 70, 130, 190, 390, 450, 510]
    rd = [(t, 120 + i, 70, 90 + i) for i, t in enumerate(times)]
    r = cuff.extract_readings(tracks_from_readings(rd, t_end=1200.0))
    med = cuff.median_interval(r, 0.0, 1800.0)
    assert med == pytest.approx(60.0, abs=0.1)
    g = cuff.flag_gaps(r, med)
    assert list(g["gap_flag"]) == [False, False, False, False, True, False, False]
    assert g["gap_s"].iloc[4] == pytest.approx(200.0, abs=0.1)


def test_gap_flags_ignore_dropped_readings():
    times = [10, 70, 130, 190, 250, 310]
    rd = [(t, 120 + i, 70, 90 + i) for i, t in enumerate(times)]
    rd[3] = (190, 120, 70, 200)            # MAP out of range -> dropped
    r = cuff.extract_readings(tracks_from_readings(rd))
    g = cuff.flag_gaps(r, cuff.median_interval(r, 0.0, 1800.0))
    assert not r["valid"].iloc[3]
    assert g["gap_s"].iloc[4] == pytest.approx(120.0, abs=0.1)   # 130 -> 250, skipping the dropped one


def test_duplicate_timestamps_do_not_create_readings():
    t = np.array([0.0, 2.0, 2.0, 4.0, 6.0])
    v = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    tr = {"sbp": (t, v + 20), "dbp": (t, v - 30), "mbp": (t, v)}
    r = cuff.extract_readings(tr)
    assert len(r) == 1


def test_window_inclusion_counts_both_ends():
    rd = [(0.0, 120, 70, 90), (900.0, 121, 71, 91), (1800.0, 122, 72, 92)]
    tr = {k: held([(r[0], r[1 + j]) for r in rd], t_end=1900.0)
          for j, k in enumerate(("sbp", "dbp", "mbp"))}
    r = cuff.extract_readings(tr)
    assert list(r["t"]) == [0.0, 900.0, 1800.0]
    assert cuff.median_interval(r, 0.0, 1800.0) == pytest.approx(900.0)
