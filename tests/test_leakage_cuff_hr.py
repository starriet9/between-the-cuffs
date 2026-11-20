"""Leakage test (SPEC §4.4) for the cuff and heart-rate feature groups.

Every HR sample at or after the issue time and every cuff reading after the
issue reading is replaced with random values; every feature except h must be
unchanged. The yardstick case (§3) is covered too: perturbing the withheld
values of reading r+1 must not change the features T uses at issue t_{r+1}
(cuff features at r, and the HR features at r+1). Runs on synthetic data and on
several real S-dev cases.
"""
import numpy as np
import pandas as pd
import pytest

from btc import cuff, hr, io, paths
from btc import instances as ins

RNG_SEED = 4404
HR_COLS = hr.HR_COLUMNS
CUFF_COLS = ins.CUFF_FEATURES
HAVE_RAW = paths.MANIFEST_SDEV.exists() and paths.RAW_SDEV.exists()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def same(a, b, cols):
    for c in cols:
        x, y = a[c], b[c]
        assert (np.isnan(x) and np.isnan(y)) or x == y, f"{c}: {x} != {y}"


def readings_df(t, sbp, dbp, mp):
    df = pd.DataFrame({"n_all": np.arange(1, len(t) + 1), "t": t, "sbp": sbp, "dbp": dbp, "map": mp})
    return cuff.apply_exclusions(df)


def feats_at(readings, n):
    v = cuff.valid_readings(readings)
    return ins.cuff_features(v["t"], v["map"], v["sbp"], v["dbp"], n)


def perturb_later_readings(readings, n, rng):
    """Replace every reading after valid reading n (times and values) with random ones."""
    r = readings[["n_all", "t", "sbp", "dbp", "map"]].copy()
    i = int(np.flatnonzero((readings["n"] == n).to_numpy())[0])
    later = np.arange(i + 1, len(r))
    t0 = r["t"].iloc[i]
    m = len(later) + int(rng.integers(0, 4))
    t_new = t0 + cuff.MERGE_S + np.cumsum(rng.uniform(16, 400, m))
    mp = rng.uniform(20, 200, m)
    s, d = mp + rng.uniform(-10, 80, m), mp - rng.uniform(-10, 60, m)
    mp[rng.random(m) < 0.1] = np.nan
    new = pd.DataFrame({"n_all": np.arange(i + 2, i + 2 + m), "t": t_new, "sbp": s, "dbp": d, "map": mp})
    return cuff.apply_exclusions(pd.concat([r.iloc[:i + 1], new], ignore_index=True))


def perturb_hr(t, v, t0, rng):
    """Replace every HR sample at or after t0 with a random value and add random samples."""
    t, v = np.asarray(t, float).copy(), np.asarray(v, float).copy()
    late = t >= t0
    v[late] = rng.uniform(0, 250, late.sum())
    v[late & (rng.random(len(t)) < 0.1)] = np.nan
    v[late & (rng.random(len(t)) < 0.1)] = 0.0
    extra_t = t0 + rng.uniform(0, 600, 50)
    extra_t[:3] = t0                                # samples exactly at t0 count as "at or after"
    t2 = np.r_[t, extra_t]
    v2 = np.r_[v, rng.uniform(0, 250, 50)]
    o = rng.permutation(len(t2))                    # order must not matter (timestamps before t0 are distinct)
    return t2[o], v2[o]


def hr_rows(issue, own, donor, donor_id=999):
    df = pd.DataFrame([issue])
    out = hr.case_hr_features(df, own, donor, donor_id)
    return {r["variant"]: r for _, r in out.iterrows()}


def synthetic_hr(rng, t_end=3700.0):
    t = np.arange(0.0, t_end, 2.0) + rng.normal(0, 0.01, int(t_end / 2))
    v = 70 + 10 * np.sin(t / 300.0) + rng.normal(0, 2, len(t))
    v[rng.random(len(t)) < 0.02] = 0.0            # monitor zeros
    gap = (t > 1500) & (t < 1560)
    return t[~gap], v[~gap]


def synthetic_readings(rng, m=30):
    t = np.cumsum(rng.uniform(20, 250, m))
    mp = rng.uniform(55, 110, m)
    return readings_df(t, mp + rng.uniform(20, 60, m), mp - rng.uniform(10, 35, m), mp)


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------

def test_cuff_features_ignore_later_readings_synthetic():
    rng = np.random.default_rng(RNG_SEED)
    checked = 0
    for _ in range(20):
        r = synthetic_readings(rng)
        v = cuff.valid_readings(r)
        for n in v.loc[v["n"] >= 3, "n"]:
            rp = perturb_later_readings(r, int(n), rng)
            same(feats_at(r, int(n)), feats_at(rp, int(n)), CUFF_COLS)
            checked += 1
    assert checked > 400


def test_cuff_features_ignore_later_raw_samples_synthetic():
    """Raw-track version: samples after t_r + 15 s (when the issue reading's
    merge cluster has closed, §2.4) are replaced with random held values."""
    rng = np.random.default_rng(RNG_SEED + 1)
    tg = np.arange(0.0, 3600.0, 2.0)
    for _ in range(5):
        r0 = synthetic_readings(rng, m=25)
        tracks = {}
        for k, col in (("sbp", "sbp"), ("dbp", "dbp"), ("mbp", "map")):
            idx = np.searchsorted(r0["t"].to_numpy(), tg, side="right") - 1
            ok = idx >= 0
            tracks[k] = (tg[ok], r0[col].to_numpy()[idx[ok]])
        r = cuff.extract_readings(tracks)
        v = cuff.valid_readings(r)
        for n in v.loc[(v["n"] >= 3) & (v["t"] < 3000), "n"]:
            t_r = float(v.loc[v["n"] == n, "t"].iloc[0])
            pt = {}
            for k, (t, val) in tracks.items():
                val = val.copy()
                late = t > t_r + cuff.MERGE_S
                ch = np.sort(rng.uniform(t_r + cuff.MERGE_S, 3600, 12))
                newv = rng.uniform(20, 220, 13)
                val[late] = newv[np.searchsorted(ch, t[late], side="right")]
                pt[k] = (t, val)
            rp = cuff.extract_readings(pt)
            same(feats_at(r, int(n)), feats_at(rp, int(n)), CUFF_COLS)


@pytest.mark.parametrize("variant", hr.VARIANTS)
def test_hr_features_ignore_samples_at_or_after_issue_synthetic(variant):
    rng = np.random.default_rng(RNG_SEED + 2)
    own, donor = synthetic_hr(rng), synthetic_hr(rng)
    for t_issue in rng.uniform(100, 3500, 200):
        issue = {"caseid": 1, "window": "primary", "issue_n": 5, "t_issue": t_issue,
                 "t_prev": t_issue - rng.uniform(16, 400)}
        a = hr_rows(issue, own, donor)[variant]
        b = hr_rows(issue, perturb_hr(*own, t_issue, rng), perturb_hr(*donor, t_issue, rng))[variant]
        same(a, b, HR_COLS)


@pytest.mark.parametrize("variant, stop", [("real", 0.0), ("p1", 0.0), ("ia", hr.IA_EXCLUDE_S),
                                           ("p2", hr.P2_SHIFT_S)])
def test_hr_variants_respect_their_stop_time(variant, stop):
    """Each variant also ignores samples at or after its own t_end; real/p1 must
    react to samples just before t_issue (the test can detect a change)."""
    rng = np.random.default_rng(RNG_SEED + 3)
    own, donor = synthetic_hr(rng), synthetic_hr(rng)
    changed = 0
    for t_issue in rng.uniform(800, 3500, 100):
        issue = {"caseid": 1, "window": "primary", "issue_n": 5, "t_issue": t_issue, "t_prev": t_issue - 120}
        a = hr_rows(issue, own, donor)[variant]
        t_end = t_issue - stop
        b = hr_rows(issue, perturb_hr(*own, t_end, rng), perturb_hr(*donor, t_end, rng))[variant]
        same(a, b, HR_COLS)
        c = hr_rows(issue, perturb_hr(*own, t_end - 20, rng), perturb_hr(*donor, t_end - 20, rng))[variant]
        changed += not np.isclose(a["hr_level"], c["hr_level"], equal_nan=True) or a["hr_n"] != c["hr_n"]
    assert changed > 50


def test_yardstick_withheld_values_synthetic():
    rng = np.random.default_rng(RNG_SEED + 4)
    own = synthetic_hr(rng)
    checked = 0
    for _ in range(10):
        r = synthetic_readings(rng)
        base = ins.case_instances(1, 1, False, r, opstart=5000.0)
        trip = base[(base["window"] == "primary") & (base["k"] == 2)]
        for row in trip.itertuples():
            rr, rp = row.issue_n, r.copy()
            j = int(np.flatnonzero((r["n"] == rr + 1).to_numpy())[0])
            m = rng.uniform(40, 150)
            rp.loc[j, ["sbp", "dbp", "map"]] = [m + rng.uniform(5, 60), m - rng.uniform(5, 30), m]
            rp = cuff.apply_exclusions(rp)
            assert rp["valid"].iloc[j]
            pert = ins.case_instances(1, 1, False, rp, opstart=5000.0)
            key = (pert["window"] == "primary") & (pert["issue_n"] == rr) & (pert["k"] == 2)
            a = base[(base["window"] == "primary") & (base["issue_n"] == rr) & (base["k"] == 2)].iloc[0]
            same(a, pert.loc[key].iloc[0], CUFF_COLS + ["t_next", "h_s"])   # L/T inputs at r
            ia = base.drop_duplicates(["window", "issue_n"])
            ip = pert.drop_duplicates(["window", "issue_n"])
            fa = hr.case_hr_features(ia.loc[ia["issue_n"] == rr + 1], own, own, 1)
            fb = hr.case_hr_features(ip.loc[ip["issue_n"] == rr + 1], own, own, 1)
            for (_, x), (_, y) in zip(fa.iterrows(), fb.iterrows()):
                same(x, y, HR_COLS + ["t_issue", "t_prev"])
            checked += 1
    assert checked > 50


# ---------------------------------------------------------------------------
# real S-dev cases
# ---------------------------------------------------------------------------

def real_cases():
    """A few analysable S-dev cases: the four lowest case IDs and two fast-cycling ones."""
    out, fast = [], []
    for cid in io.sdev_caseids():
        case = io.load_case(cid)
        r, med = ins.case_readings(case)
        n_p = int((r["valid"] & (r["t"] <= ins.PRIMARY_END_S)).sum())
        if case.pleth is None or case.ecg is None or n_p < ins.MIN_VALID_PRIMARY:
            continue
        if len(out) < 4:
            out.append(case)
        elif med < ins.FAST_CYCLING_S and len(fast) < 2:
            fast.append(case)
        if len(out) == 4 and len(fast) == 2:
            break
    return out + fast


@pytest.fixture(scope="module")
def cases():
    if not HAVE_RAW:
        pytest.skip("S-dev raw data not downloaded")
    return real_cases()


def raw_tracks(case):
    return {"sbp": (case.sbp.t, case.sbp.v), "dbp": (case.dbp.t, case.dbp.v), "mbp": (case.mbp.t, case.mbp.v)}


def test_real_cuff_leakage(cases):
    rng = np.random.default_rng(RNG_SEED + 5)
    for case in cases:
        tracks = raw_tracks(case)
        r = cuff.extract_readings(tracks)
        v = cuff.valid_readings(r)
        ns = v.loc[(v["n"] >= 3) & (v["t"] <= ins.PRIMARY_END_S), "n"].to_numpy()
        for n in ns:
            n = int(n)
            # readings level: every later reading replaced
            same(feats_at(r, n), feats_at(perturb_later_readings(r, n, rng), n), CUFF_COLS)
            # raw level: every sample after the issue reading's cluster closed (t_r + 15 s)
            t_r = float(v.loc[v["n"] == n, "t"].iloc[0])
            pt = {}
            for k, (t, val) in tracks.items():
                val = val.copy()
                late = t > t_r + cuff.MERGE_S
                ch = np.sort(rng.uniform(t_r + cuff.MERGE_S, t.max() + 1, 20))
                newv = rng.integers(20, 220, 21).astype(float)
                val[late] = newv[np.searchsorted(ch, t[late], side="right")]
                pt[k] = (t, val)
            same(feats_at(r, n), feats_at(cuff.extract_readings(pt), n), CUFF_COLS)


def test_real_hr_leakage(cases):
    rng = np.random.default_rng(RNG_SEED + 6)
    donor = cases[-1]
    for case in cases[:-1]:
        r, _ = ins.case_readings(case)
        inst = ins.case_instances(case.caseid, 0, False, r, opstart=3600.0)
        iss = inst.drop_duplicates(["window", "issue_n"])
        own, don = (case.hr.t, case.hr.v), (donor.hr.t, donor.hr.v)
        base = hr.case_hr_features(iss, own, don, donor.caseid)
        assert base.loc[base["variant"] == "real", "hr_level"].notna().mean() > 0.5
        for _, row in iss.iterrows():
            t0 = row["t_issue"]
            a = base[(base["window"] == row["window"]) & (base["issue_n"] == row["issue_n"])]
            b = hr.case_hr_features(pd.DataFrame([row]), perturb_hr(*own, t0, rng),
                                    perturb_hr(*don, t0, rng), donor.caseid)
            for (_, x), (_, y) in zip(a.iterrows(), b.iterrows()):
                assert x["variant"] == y["variant"]
                same(x, y, HR_COLS)


def test_real_yardstick_withheld_values(cases):
    rng = np.random.default_rng(RNG_SEED + 7)
    checked = 0
    for case in cases:
        r, _ = ins.case_readings(case)
        base = ins.case_instances(case.caseid, 0, False, r, opstart=3600.0)
        own = (case.hr.t, case.hr.v)
        trip = base[(base["window"] == "primary") & (base["k"] == 2)]
        for row in trip.itertuples():
            rr = row.issue_n
            j = int(np.flatnonzero((r["n"] == rr + 1).to_numpy())[0])
            rp = r.copy()
            m = float(rng.uniform(40, 150))
            rp.loc[rp.index[j], ["sbp", "dbp", "map"]] = [m + rng.uniform(5, 60), m - rng.uniform(5, 30), m]
            rp = cuff.apply_exclusions(rp)
            pert = ins.case_instances(case.caseid, 0, False, rp, opstart=3600.0)
            sel = lambda d: d[(d["window"] == "primary") & (d["issue_n"] == rr) & (d["k"] == 2)].iloc[0]
            same(sel(base), sel(pert), CUFF_COLS + ["t_next", "h_s"])
            ia = base.drop_duplicates(["window", "issue_n"])
            ip = pert.drop_duplicates(["window", "issue_n"])
            fa = hr.case_hr_features(ia.loc[ia["issue_n"] == rr + 1], own, own, case.caseid)
            fb = hr.case_hr_features(ip.loc[ip["issue_n"] == rr + 1], own, own, case.caseid)
            assert len(fa) == len(fb) > 0
            for (_, x), (_, y) in zip(fa.iterrows(), fb.iterrows()):
                same(x, y, HR_COLS + ["t_issue", "t_prev"])
            checked += 1
    assert checked >= 10
