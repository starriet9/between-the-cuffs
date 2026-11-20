"""Windows, issue points, targets, cuff features, cohort rules, donors and
signal integrity (SPEC §2.2, §2.3, §2.5, §3, §4.1, §6.6)."""
import numpy as np
import pandas as pd
import pytest

from btc import cuff, io, paths
from btc import instances as ins

HAVE_RAW = paths.MANIFEST_SDEV.exists() and paths.RAW_SDEV.exists()
DERIVED = {n: paths.DERIVED / f"sdev_{n}.parquet" for n in ("cases", "readings", "instances", "features_hr")}
HAVE_DERIVED = all(p.exists() for p in DERIVED.values())


def readings_df(rows):
    """Readings table from [(t, sbp, dbp, map), ...] with §2.4 exclusions applied."""
    df = pd.DataFrame([{"n_all": i + 1, "t": t, "sbp": s, "dbp": d, "map": m}
                       for i, (t, s, d, m) in enumerate(rows)])
    return cuff.apply_exclusions(df)


def regular(n, dt, t0=10.0, f=lambda t: 90.0):
    return [(t0 + i * dt, f(t0 + i * dt) + 40, f(t0 + i * dt) - 25, f(t0 + i * dt)) for i in range(n)]


# ---------------------------------------------------------------------------
# Issue points and targets (§3)
# ---------------------------------------------------------------------------

def test_issue_points_start_at_third_valid_reading():
    r = readings_df(regular(8, 60.0))
    out = ins.case_instances(1, 1, False, r, opstart=5000.0)
    p = out[out["window"] == "primary"]
    assert p["issue_n"].min() == 3
    assert sorted(p.loc[p["k"] == 1, "issue_n"]) == [3, 4, 5, 6, 7]   # reading 8 has no next reading


def test_dropped_readings_do_not_count():
    rows = regular(8, 60.0)
    rows[1] = (rows[1][0], 120.0, 70.0, 200.0)       # MAP out of range -> dropped
    r = readings_df(rows)
    out = ins.case_instances(1, 1, False, r, opstart=5000.0)
    p = out[(out["window"] == "primary") & (out["k"] == 1)]
    first = p.loc[p["issue_n"] == 3].iloc[0]
    assert first["t_issue"] == rows[3][0]             # third *valid* reading is the 4th reading
    assert first["base_map"] == pytest.approx(np.mean([rows[0][3], rows[2][3], rows[3][3]]))
    assert first["t_prev"] == rows[2][0]


def test_targets_within_400s_and_inside_window():
    times = [0.0, 100.0, 200.0, 300.0, 700.0, 1100.5, 1300.0, 1700.0, 1800.0, 1900.0]
    r = readings_df([(t, 130.0, 70.0, 90.0) for t in times])
    out = ins.case_instances(1, 1, False, r, opstart=5000.0)
    p = out[out["window"] == "primary"].set_index(["issue_n", "k"])
    assert (3, 1) in p.index and (3, 2) not in p.index           # 200 -> 300 ok; 200 -> 700 = 500 s
    assert (4, 1) in p.index and p.loc[(4, 1), "h_s"] == 400.0   # exactly 400 s is included
    assert (5, 1) not in p.index                                  # 700 -> 1100.5 = 400.5 s
    assert (8, 1) in p.index and p.loc[(8, 1), "t_target"] == 1800.0   # window end included
    assert (9, 1) not in p.index                                  # 1900 is outside the window
    assert set(p.index.get_level_values("k")) <= {1, 2, 3}


def test_k_up_to_three_and_t_next():
    r = readings_df(regular(10, 60.0))
    out = ins.case_instances(1, 1, False, r, opstart=5000.0)
    p = out[(out["window"] == "primary") & (out["issue_n"] == 3)].set_index("k")
    assert list(p.index) == [1, 2, 3]
    assert (p["target_n"] == [4, 5, 6]).all()
    assert (p["t_next"] == p.loc[1, "t_target"]).all()


def test_secondary_window_ends_at_incision_capped_at_60_min():
    r = readings_df(regular(80, 60.0))
    out = ins.case_instances(1, 1, False, r, opstart=1000.0)
    s = out[out["window"] == "secondary"]
    assert s["t_target"].max() <= 1000.0
    out = ins.case_instances(1, 1, False, r, opstart=9000.0)
    s = out[out["window"] == "secondary"]
    assert s["t_target"].max() <= 3600.0 and s["t_target"].max() > 3500.0
    p = out[out["window"] == "primary"]
    assert p["t_target"].max() <= 1800.0
    assert ins.window_bounds("secondary", float("nan")) is None


def test_h_bins_edges():
    h = [15.0, 89.999, 90.0, 149.9, 150.0, 210.0, 399.9, 400.0]
    assert list(ins.h_bin(h)) == [1, 1, 2, 2, 3, 4, 4, 4]


# ---------------------------------------------------------------------------
# Cuff features (§2.5, §4.1)
# ---------------------------------------------------------------------------

def test_cuff_features_forms_a_and_b():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(40, 200, 12))
    a, b, c = 0.8, -3.0, 95.0                          # MAP = a tm^2 + b tm + c, tm in minutes
    mp = a * (t / 60) ** 2 + b * (t / 60) + c
    sbp, dbp = mp + 35 + rng.uniform(0, 5, 12), mp - 25 - rng.uniform(0, 5, 12)
    for n in range(3, 13):
        f = ins.cuff_features(t, mp, sbp, dbp, n)
        tm = t[n - 3:n] / 60
        base = mp[:3].mean()
        assert f["cuff_curv"] == pytest.approx(2 * a, rel=1e-6)
        assert f["cuff_slope"] == pytest.approx(np.polyfit(tm, mp[n - 3:n], 1)[0], rel=1e-9)
        assert f["cuff_sbp_slope"] == pytest.approx(np.polyfit(tm, sbp[n - 3:n], 1)[0], rel=1e-9)
        assert f["cuff_sbp_curv"] == pytest.approx(2 * np.polyfit(tm, sbp[n - 3:n], 2)[0], rel=1e-6)
        assert f["cuff_level"] == pytest.approx(mp[n - 1] - base)
        assert f["base_map"] == pytest.approx(base) and f["base_sbp"] == pytest.approx(sbp[:3].mean())
        assert (f["cuff_lag0"], f["cuff_lag1"], f["cuff_lag2"]) == pytest.approx(
            (mp[n - 1] - base, mp[n - 2] - base, mp[n - 3] - base))
        assert f["cuff_gap1_min"] == pytest.approx((t[n - 1] - t[n - 2]) / 60)
        assert f["cuff_gap2_min"] == pytest.approx((t[n - 2] - t[n - 3]) / 60)
        assert f["cuff_pp"] == pytest.approx(sbp[n - 1] - dbp[n - 1])
        assert f["cuff_tsr_min"] == pytest.approx(t[n - 1] / 60)
        assert f["run_mean_map"] == pytest.approx(mp[:n].mean())
        assert f["h_med_prev_s"] == pytest.approx(np.median(np.diff(t[:n])))
        assert f["last_map"] == mp[n - 1] and f["t_issue"] == t[n - 1] and f["t_prev"] == t[n - 2]


def test_cuff_features_require_three_readings():
    with pytest.raises(ValueError):
        ins.cuff_features([1.0, 2.0], [90, 91], [120, 121], [70, 71], 2)


# ---------------------------------------------------------------------------
# Cohort rules (§2.2) and the P1 donor map (§6.6)
# ---------------------------------------------------------------------------

def test_analysable_reasons():
    assert ins.analysable_reasons(True, True, 6, False) == []
    assert ins.analysable_reasons(True, True, 5, False) == ["lt6_valid_readings_primary"]
    assert ins.analysable_reasons(False, True, 9, False) == ["pleth_absent"]
    assert ins.analysable_reasons(True, False, 9, True) == ["subject_also_in_sob", "ecg_absent"]


@pytest.mark.parametrize("rocu, vecu, expected", [
    (50, 0, True),                       # rocuronium recorded
    (0, 4, True),                        # vecuronium recorded
    (0, 0, False),
    (np.nan, np.nan, False),             # missing counts as not recorded
    (np.nan, 2.0, True),
    (None, 0, False),
    (pd.NA, 30, True),
])
def test_nmb_rule(rocu, vecu, expected):
    """SPEC §2.2: rocuronium or vecuronium total > 0, each on its own; missing = not recorded."""
    assert ins.nmb_recorded(rocu, vecu) is expected
    row = pd.Series({"intraop_rocu": rocu, "intraop_vecu": vecu})
    assert ins.nmb_columns(row, analysable=True) == {"nmb_recorded": expected, "analysed": not expected}
    assert ins.nmb_columns(row, analysable=False) == {"nmb_recorded": expected, "analysed": False}


def test_nmb_columns_need_the_case_table_fields():
    with pytest.raises(KeyError):
        ins.nmb_columns(pd.Series({"intraop_rocu": 0}), analysable=True)


def donor_table(n=8, subjects=None, short=()):
    ids = np.arange(100, 100 + n)
    subjects = list(range(n)) if subjects is None else subjects
    rows = []
    for i, c in enumerate(ids):
        last = 900.0 if c in short else 3000.0
        rows.append({"caseid": int(c), "subjectid": subjects[i], "analysable": True,
                     "first_issue_t": 300.0, "last_issue_t": 1700.0,
                     "hr_t_first": 2.0, "hr_t_last": last, "pleth_t_first": 1.0, "pleth_t_last": last,
                     "ecg_t_first": 1.0, "ecg_t_last": last})
    return pd.DataFrame(rows)


def test_donor_is_next_in_seeded_permutation_with_wraparound():
    cases = donor_table()
    d = ins.assign_donors(cases)
    perm = np.random.default_rng(ins.DONOR_SEED).permutation(cases["caseid"].to_numpy())
    for i, c in enumerate(perm):
        assert d[c] == perm[(i + 1) % len(perm)]


def test_donor_skips_same_subject_and_non_covering_cases():
    cases = donor_table()
    perm = np.random.default_rng(ins.DONOR_SEED).permutation(cases["caseid"].to_numpy())
    r, nxt, nxt2 = perm[0], perm[1], perm[2]
    subj = dict(zip(cases["caseid"], cases["subjectid"]))
    subj[nxt] = subj[r]                                   # same subject as the recipient
    cases["subjectid"] = cases["caseid"].map(subj)
    cases.loc[cases["caseid"] == nxt2, ["pleth_t_last"]] = 1000.0   # ends before last issue time
    d = ins.assign_donors(cases)
    assert d[r] == perm[3]
    late = cases.copy()
    late.loc[late["caseid"] == perm[4], "ecg_t_first"] = 400.0     # starts after first issue time
    d2 = ins.assign_donors(late)
    assert d2[perm[3]] == perm[5]


def test_donors_unchanged_by_the_nmb_rule():
    """P1 donors come from all analysable cases, with or without a recorded neuromuscular blocker (SPEC §2.2)."""
    cases = donor_table(n=12)
    ref = ins.assign_donors(cases)
    flagged = cases.assign(nmb_recorded=[i % 3 == 0 for i in range(12)])
    flagged["analysed"] = flagged["analysable"] & ~flagged["nmb_recorded"]
    pd.testing.assert_series_equal(ins.assign_donors(flagged), ref)
    nmb_ids = set(flagged.loc[flagged["nmb_recorded"], "caseid"])
    assert nmb_ids & set(ref.dropna().astype(int)) and ref.loc[sorted(nmb_ids)].notna().all()


def test_donor_missing_when_none_covers_and_non_analysable_excluded():
    cases = donor_table(short=tuple(range(100, 108)))
    assert ins.assign_donors(cases).isna().all()
    cases = donor_table()
    cases.loc[cases["caseid"] == 103, "analysable"] = False
    d = ins.assign_donors(cases)
    assert np.isnan(d[103]) and (d.dropna() != 103).all()


# ---------------------------------------------------------------------------
# Signal integrity (§2.3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("est, ov, status", [
    (500.0, 0.0, io.OK),
    (505.0, 0.0, io.OK),                  # exactly 1% off is not "more than 1%"
    (505.01, 0.0, io.ABSENT_INTEGRITY),
    (494.9, 0.0, io.ABSENT_INTEGRITY),
    (500.0, 0.01, io.OK),
    (500.0, 0.0101, io.ABSENT_INTEGRITY),
    (np.nan, np.nan, io.KEPT_FLAGGED),    # original file not readable: kept and flagged
])
def test_waveform_status_rules(est, ov, status):
    row = {"pleth_present": True, "pleth_srate_hdr": 500.0, "pleth_srate_est": est, "pleth_overlap_frac": ov}
    assert io.waveform_status(row, "pleth")[0] == status
    assert io.waveform_status({"pleth_present": False}, "pleth")[0] == io.ABSENT_IN_FILE


@pytest.mark.skipif(not HAVE_RAW, reason="S-dev raw data not downloaded")
def test_integrity_on_sdev_manifest():
    it = io.integrity_table().set_index("caseid")
    assert it.loc[2328, "pleth_status"] == io.ABSENT_INTEGRITY
    assert it.loc[2328, "ecg_status"] == io.ABSENT_INTEGRITY
    assert it.loc[2030, "ecg_status"] == io.ABSENT_IN_FILE and it.loc[2030, "pleth_status"] == io.OK
    others = it.drop(index=[2328, 2030])
    for k in ("pleth", "ecg"):
        assert set(others[f"{k}_status"]) <= {io.OK, io.ABSENT_IN_FILE}
    c = io.load_case(2328)
    assert c.pleth is None and c.ecg is None and c.raw["pleth"] is not None
    c = io.load_case(2030)
    assert c.ecg is None and c.pleth is not None


def test_load_case_refuses_sob():
    c = io.load_cohorts()
    sob = int(c.loc[c["cohort"] == "S-ob", "caseid"].iloc[0])
    with pytest.raises(PermissionError):
        io.load_case(sob)


def test_cerclage_flag_counts_six():
    c = io.load_cohorts()
    assert int(c["cerclage"].sum()) == 6 and (c.loc[c["cerclage"], "cohort"] == "S-dev").all()


# ---------------------------------------------------------------------------
# Derived tables (btc/SCHEMA.md), if built
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_DERIVED, reason="run scripts/build_instances.py first")
def test_derived_tables_consistent():
    cases = pd.read_parquet(DERIVED["cases"])
    rd = pd.read_parquet(DERIVED["readings"])
    inst = pd.read_parquet(DERIVED["instances"])
    hrf = pd.read_parquet(DERIVED["features_hr"])
    schema_cases = ["caseid", "subjectid", "cohort", "cerclage", "opstart", "analysable",
                    "not_analysable_reason", "n_valid_primary", "median_interval_primary_s", "fast_cycling",
                    "pleth_present", "ecg_present", "waveform_flag", "hr_t_first", "hr_t_last",
                    "pleth_t_first", "pleth_t_last", "ecg_t_first", "ecg_t_last", "first_issue_t",
                    "last_issue_t", "donor_caseid"]
    assert set(schema_cases) <= set(cases.columns)
    assert set(["caseid", "n_all", "t", "sbp", "dbp", "map", "valid", "drop_reason", "gap_flag", "n"]) <= set(rd.columns)
    assert set(ins.INSTANCE_COLUMNS) <= set(inst.columns)
    assert len(cases) == 252
    a = cases[cases["analysable"]]
    assert not cases.set_index("caseid").loc[[2030, 2328], "analysable"].any()
    assert set(inst["caseid"]) == set(a["caseid"])
    assert not inst.duplicated(["caseid", "window", "issue_n", "k"]).any()
    assert (inst["issue_n"] >= 3).all() and (inst["h_s"] <= 400).all() and (inst["h_s"] > 15).all()
    assert (inst.loc[inst["window"] == "primary", "t_target"] <= 1800).all()
    # one HR row per issue point and variant
    keys = inst.drop_duplicates(["caseid", "window", "issue_n"])[["caseid", "window", "issue_n"]]
    assert len(hrf) == 4 * len(keys)
    assert not hrf.duplicated(["caseid", "window", "issue_n", "variant"]).any()
    # donors: another subject, analysable, covering
    cs = cases.set_index("caseid")
    for r in a.itertuples():
        d = r.donor_caseid
        if np.isnan(d):
            continue
        dr = cs.loc[int(d)]
        assert dr["analysable"] and dr["subjectid"] != r.subjectid
        for s in ("hr", "pleth", "ecg"):
            assert dr[f"{s}_t_first"] <= r.first_issue_t and dr[f"{s}_t_last"] >= r.last_issue_t
    p1 = hrf[hrf["variant"] == "p1"].merge(cases[["caseid", "donor_caseid"]], on="caseid")
    assert np.array_equal(p1["signal_caseid"].to_numpy(), p1["donor_caseid"].to_numpy(), equal_nan=True)
    # readings: valid index consistent, readings more than 15 s apart
    for _, g in rd.groupby("caseid"):
        assert np.all(np.diff(g["t"]) > cuff.MERGE_S)
        assert list(g.loc[g["valid"], "n"]) == list(range(1, int(g["valid"].sum()) + 1))


@pytest.mark.skipif(not HAVE_DERIVED, reason="run scripts/build_instances.py first")
def test_derived_cases_nmb_rule():
    """SPEC §2.2 columns of the built S-dev cases table: 11 of the 214 analysable cases have rocuronium
    recorded (vecuronium is 0 in all), so 203 are analysed; instances stay those of all analysable cases."""
    cases = pd.read_parquet(DERIVED["cases"])
    if not {"nmb_recorded", "analysed"} <= set(cases.columns):
        pytest.skip("sdev_cases.parquet predates the SPEC §2.2 nmb rule; rebuild with scripts/build_instances.py")
    a = cases[cases["analysable"]]
    assert len(a) == 214 and int(a["nmb_recorded"].sum()) == 11 and int(cases["analysed"].sum()) == 203
    assert (cases["analysed"] == (cases["analysable"] & ~cases["nmb_recorded"])).all()
    inst = pd.read_parquet(DERIVED["instances"])
    assert set(inst["caseid"]) == set(a["caseid"])
