"""Cohort access (SPEC §9), the cohort-generic builders on a fake S-ob cohort, signal integrity
(SPEC §2.3), shared constants, provenance (SPEC §10) and the data credit (SPEC §2.1).

No real S-ob data are read or requested anywhere here. `BTC_FROZEN_SCORING=1` is set only
inside `test_sob_builders_on_fake_cohort`, with every data path pointed at a temporary folder
that holds fake cases.
"""
import importlib.util
import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from btc import beats, constants as C, hr, instances as ins, io, paths, provenance, waveform as W
from btc.models import constants as MC
from btc.models import data as MD
from btc.models import synthetic as MS
from wf_synth import FS, Beat, make_signals

ENV = io.FROZEN_SCORING_ENV


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", paths.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # worker processes find its functions
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- S-ob refused
@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_sob_access_refused_without_env(no_env):
    with pytest.raises(PermissionError):
        io.check_cohort_access("sob")
    with pytest.raises(PermissionError):
        io.load_manifest("sob")
    with pytest.raises(ValueError):
        io.check_cohort_access("other")
    if not paths.COHORTS.exists():
        pytest.skip("case table not downloaded (scripts/download.py)")
    sob = io.caseids("sob")                              # case-table metadata only
    assert len(sob) == 21
    with pytest.raises(PermissionError):
        io.load_case(sob[0], "sob")
    with pytest.raises(PermissionError):
        io.load_case(sob[0])                             # not an S-dev case either
    with pytest.raises(PermissionError):
        W.load_case(sob[0], "sob")
    with pytest.raises(ValueError):
        io.check_cohort_access("other")


@pytest.mark.parametrize("name", ["build_instances", "build_windows", "build_features_wf", "pleth_coverage"])
def test_builders_refuse_sob_without_env(no_env, name):
    with pytest.raises(PermissionError):
        _script(name).main(["--cohort", "sob", "--allow-dirty"])


def test_download_refuses_sob_without_env(no_env):
    with pytest.raises(SystemExit) as e:
        _script("download").main(["--cohort", "sob"])     # stops before any request
    assert "BTC_FROZEN_SCORING" in str(e.value)


def test_env_must_be_exactly_one(monkeypatch):
    for v in ("", "0", "true", "yes"):
        monkeypatch.setenv(ENV, v)
        with pytest.raises(PermissionError):
            io.check_cohort_access("sob")


# --------------------------------------------------------------------------- fake S-ob cohort
FAKE = {  # caseid: (subjectid, pump track, intraop_phe ug, intraop_eph mg, ECG header rate)
    90001: (1, True, 0.0, 0.0, 500.0),
    90002: (2, False, 100.0, 0.0, 500.0),
    90003: (3, False, 0.0, 10.0, 500.0),
    90004: (4, False, np.nan, np.nan, 500.0),
    90005: (5, False, 0.0, 0.0, 250.0),     # ECG header rate 250 Hz: absent (SPEC §2.3)
}
NMB = {  # caseid: (intraop_rocu mg, intraop_vecu mg) in the fake case table (SPEC §2.2)
    90001: (0, 0), 90002: (0, 0), 90003: (50, 0), 90004: (np.nan, np.nan), 90005: (0, 4),
}


def _held(t_read, vals, t_end, step=2.0):
    """A held-value track sampled every `step` s that changes at the reading times."""
    t = np.arange(0.0, t_end, step)
    k = np.searchsorted(t_read, t, side="right") - 1
    ok = k >= 0
    return t[ok], np.asarray(vals, float)[k[ok]]


def _fake_case(cid, pump, ecg_rate, rng, dur=1900.0):
    t_read = 25.0 + np.cumsum(rng.uniform(55, 95, 26))
    t_read = t_read[t_read < dur - 20]
    mp = np.round(85 + rng.normal(0, 6, t_read.size))
    arrays = {}
    for k, v in (("sbp", mp + 35), ("dbp", mp - 20), ("mbp", mp)):
        arrays[f"{k}_t"], arrays[f"{k}_v"] = _held(t_read, v, dur)
    arrays["hr_t"] = np.arange(1.0, dur, 2.0)
    arrays["hr_v"] = 75 + rng.normal(0, 2, arrays["hr_t"].size)
    if pump:
        arrays["phen_t"], arrays["phen_v"] = np.array([300.0, 900.0]), np.array([30.0, 15.0])
    p, e, *_ = make_signals(dur, Beat(T=rng.uniform(0.7, 1.0)), pat=0.62, t0=0.8, noise=0.3,
                            seed=int(cid) % 1000)
    arrays["pleth"], arrays["pleth_t0"], arrays["pleth_srate"] = p.astype(np.float32), 0.8, FS
    if ecg_rate == FS:
        arrays["ecg"], arrays["ecg_t0"], arrays["ecg_srate"] = e.astype(np.float32), 0.8, FS
    else:
        arrays["ecg"], arrays["ecg_t0"], arrays["ecg_srate"] = e[::2].astype(np.float32), 0.8, ecg_rate
    meta = {"caseid": cid, "fake": True, **{f"{k}_fmt": 1 for k in ("sbp", "dbp", "mbp", "hr", "phen")}}
    arrays["meta_json"] = np.array(json.dumps(meta))
    return arrays


@pytest.fixture
def fake_sob(tmp_path, monkeypatch):
    """Points every data path at `tmp_path` and writes a fake S-ob cohort there."""
    raw = tmp_path / "raw"
    (raw / "sob").mkdir(parents=True)
    (raw / "meta").mkdir()
    rng = np.random.default_rng(7)
    cohort, demo, man = [], [], []
    for cid, (sid, pump, phe, eph, ecg_rate) in FAKE.items():
        cohort.append({"caseid": cid, "subjectid": sid, "cohort": "S-ob", "opname": "Cesarean section (fake)",
                       "casestart": 0, "anestart": -300, "opstart": 1500.0, "intraop_phe": phe,
                       "intraop_eph": eph, "emop": 1})
        demo.append({"caseid": cid, "sex": "F", "age": ">89" if cid == 90004 else 33,
                     "intraop_rocu": NMB[cid][0], "intraop_vecu": NMB[cid][1],
                     "intraop_mdz": 2.0 if cid == 90002 else 0.0, "intraop_ppf": 0.0})
        np.savez_compressed(raw / "sob" / f"{cid}.npz", **_fake_case(cid, pump, ecg_rate, rng))
        man.append({"caseid": cid, "pleth_present": True, "pleth_srate_hdr": FS, "pleth_srate_est": 500.1,
                    "pleth_overlap_frac": 0.0, "ecg_present": True, "ecg_srate_hdr": ecg_rate,
                    "ecg_srate_est": ecg_rate, "ecg_overlap_frac": 0.0})
    pd.DataFrame(cohort).to_csv(tmp_path / "cohorts.csv", index=False)
    pd.DataFrame(demo).to_csv(raw / "meta" / "cases.csv", index=False)
    pd.DataFrame(man).to_csv(raw / "sob_manifest.csv", index=False)
    for k, v in {"COHORTS": tmp_path / "cohorts.csv", "RAW": raw, "RAW_SOB": raw / "sob",
                 "RAW_SDEV": raw / "sdev", "MANIFEST_SOB": raw / "sob_manifest.csv",
                 "MANIFEST_SDEV": raw / "sdev_manifest.csv", "META": raw / "meta",
                 "CASE_TABLE": raw / "meta" / "cases.csv", "DERIVED": tmp_path / "derived"}.items():
        monkeypatch.setattr(paths, k, v)
    return tmp_path


def test_sob_builders_on_fake_cohort(fake_sob, monkeypatch):
    derived = fake_sob / "derived"
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(PermissionError):
        io.load_case(90001, "sob")                       # refused even with fake data
    monkeypatch.setenv(ENV, "1")                         # this test only, fake data only
    assert paths.raw_dir("sob") == fake_sob / "raw" / "sob"
    c = io.load_case(90001, "sob")
    assert c.cohort == "sob" and c.pleth is not None and len(c.phen) == 2
    with pytest.raises(PermissionError):
        io.load_case(90001, "sdev")                      # an S-ob case is never read as S-dev
    c5 = io.load_case(90005, "sob")                      # §2.3: header rate != 500 Hz -> absent
    assert c5.ecg is None and c5.ecg_status == io.ABSENT_INTEGRITY and "header rate" in c5.waveform_flag
    assert W.load_case(90005, "sob")["ecg"] is None      # no error from the waveform side

    _script("build_instances").main(["--cohort", "sob", "--allow-dirty"])
    _script("build_windows").main(["--cohort", "sob", "--jobs", "2", "--allow-dirty"])
    _script("build_features_wf").main(["--cohort", "sob", "--allow-dirty"])
    _script("pleth_coverage").main(["--cohort", "sob", "--allow-dirty"])

    cases = pd.read_parquet(derived / "sob_cases.parquet").set_index("caseid")
    assert (cases["cohort"] == "S-ob").all() and not cases["cerclage"].any()
    assert not cases.loc[90005, "analysable"] and "ecg_absent" in cases.loc[90005, "not_analysable_reason"]
    assert cases.loc[[90001, 90002, 90003, 90004], "analysable"].all()
    assert cases.loc[90001, "vaso_stratum"] == "pump" and bool(cases.loc[90001, "pump_present"])
    assert cases.loc[90002, "vaso_stratum"] == "bolus_no_pump"
    assert cases.loc[90003, "vaso_stratum"] == "bolus_no_pump"     # ephedrine alone counts
    assert cases.loc[90003, "vaso_stratum_phe"] == "neither_recorded"
    assert cases.loc[90004, "vaso_stratum"] == "neither_recorded"  # missing totals: not recorded
    assert cases.loc[90004, "age"] == 89.0 and bool(cases.loc[90004, "age_censored"])
    assert cases.loc[90002, "intraop_phe_ug"] == 100.0 and cases.loc[90003, "intraop_eph_mg"] == 10.0
    # SPEC §2.2: rocuronium or vecuronium > 0 -> nmb_recorded; analysed = analysable and not nmb_recorded
    assert cases["nmb_recorded"].to_dict() == {90001: False, 90002: False, 90003: True, 90004: False, 90005: True}
    assert cases["analysed"].to_dict() == {90001: True, 90002: True, 90003: False, 90004: True, 90005: False}
    assert cases["analysable"].sum() == 4                                # analysable keeps its meaning
    d = cases.loc[cases["analysable"], "donor_caseid"]
    assert d.notna().all() and set(d.astype(int)) <= set(FAKE) and (d.index != d.astype(int)).all()
    assert 90003 in set(pd.read_parquet(derived / "sob_instances.parquet")["caseid"])   # built for all analysable
    for name in ("cases", "readings", "instances", "features_hr", "windows", "features_wf"):
        m = provenance.read_meta(derived / f"sob_{name}.parquet")
        assert m["cohort"] == "S-ob" and m["git"]["allow_dirty"] is True and "commit" in m["git"]
    for log in ("sob_cohort_flow.json", "sob_windows_log.json", "sob_features_wf_log.json",
                "sob_pleth_coverage.json"):
        assert "git" in json.loads((derived / log).read_text())
    flow = json.loads((derived / "sob_cohort_flow.json").read_text())
    assert flow["cohort"] == "S-ob" and flow["vaso_strata_analysable"]["phenylephrine_or_ephedrine"] == {
        "bolus_no_pump": 2, "pump": 1, "neither_recorded": 1}
    assert (flow["analysable"], flow["nmb_recorded_analysable"], flow["analysed"]) == (4, 1, 3)
    assert flow["nmb_recorded_analysable_cases"] == [90003] and flow["nmb_recorded_all_cases"] == 2
    # counts a reviewer can check (SPEC §2.2): sedation among the analysed, fast cycling, donors
    assert flow["sedation_analysed"] == {"intraop_mdz": 1, "intraop_ppf": 0}
    assert flow["fast_cycling_analysed"] + flow["fast_cycling_nmb_recorded"] == flow["fast_cycling_analysable"]
    assert 0 <= flow["analysed_cases_with_ge3_primary_targets"] <= flow["analysed"]
    assert flow["analysed_with_nmb_donor"] is not None
    assert "**Analysed: 3**" in (derived / "sob_cohort_flow.md").read_text()
    for md in ("sob_cohort_flow.md", "sob_pleth_coverage.md"):
        assert provenance.CREDIT in (derived / md).read_text()
    f = pd.read_parquet(derived / "sob_features_wf.parquet")
    assert set(f["variant"]) == set(W.VARIANTS) and not any("notch" in col for col in f.columns)
    assert f.loc[f["variant"] == "real", "wf_available"].mean() > 0.5
    T = MD.load_tables(derived, "sob")                   # the model code accepts the S-ob tables
    frame = MD.build_frame(T)
    assert len(frame) > 0 and "vaso_stratum" in frame.columns
    assert set(frame["caseid"]) == {90001, 90002, 90004}  # analysed cases only (SPEC §2.2)
    assert T.counts()["n_nmb_recorded_cases"] == 1 and T.counts()["nmb_recorded_caseids"] == [90003]
    assert 90003 in set(MD.build_frame(replace(T, include_nmb=True))["caseid"])


def test_build_windows_raises_on_missing_archive(fake_sob, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    (fake_sob / "raw" / "sob" / "90003.npz").unlink()
    with pytest.raises(FileNotFoundError):
        _script("build_windows").main(["--cohort", "sob", "--jobs", "1", "--allow-dirty"])


# --------------------------------------------------------------------------- §2.3 header rate
def test_header_rate_not_500_is_absent():
    row = {"pleth_present": True, "pleth_srate_hdr": 250.0, "pleth_srate_est": 250.0, "pleth_overlap_frac": 0.0}
    st, detail = io.waveform_status(row, "pleth")
    assert st == io.ABSENT_INTEGRITY and "header rate 250 Hz" in detail
    row["pleth_srate_est"] = row["pleth_overlap_frac"] = np.nan     # block check not made
    assert io.waveform_status(row, "pleth")[0] == io.ABSENT_INTEGRITY


# --------------------------------------------------------------------------- small rules
def test_clean_hr_keeps_last_value_of_a_timestamp():
    t, v = hr.clean_hr([1.0, 2.0, 2.0, 3.0, 2.0, 0.5], [70, 71, 72, 0, 73, np.nan])
    np.testing.assert_array_equal(t, [1.0, 2.0])
    np.testing.assert_array_equal(v, [70, 73])              # last valid value at t = 2 in file order


def test_parse_age():
    assert ins.parse_age(61) == (61.0, False)
    assert ins.parse_age("45") == (45.0, False)
    assert ins.parse_age(">89") == (89.0, True)
    a, c = ins.parse_age("unknown")
    assert np.isnan(a) and not c


def test_vaso_stratum_never_adds_units():
    assert ins.vaso_stratum(True, 0.0, 0.0) == "pump"
    assert ins.vaso_stratum(False, 50.0, 0.0) == "bolus_no_pump"
    assert ins.vaso_stratum(False, 0.0, 5.0) == "bolus_no_pump"
    assert ins.vaso_stratum(False, 0.0) == "neither_recorded"
    assert ins.vaso_stratum(False, np.nan, None) == "neither_recorded"
    assert ins.vaso_stratum(False, -1.0, 0.5) == "bolus_no_pump"    # each total on its own, never summed


# --------------------------------------------------------------------------- shared constants (S4)
def test_constants_have_one_source():
    assert MC.PRIMARY_END_S == ins.PRIMARY_END_S == C.PRIMARY_END_S == 1800.0
    assert MC.H_BIN_EDGES_S == ins.H_BIN_EDGES == C.H_BIN_EDGES_S == (15.0, 90.0, 150.0, 210.0, 400.0)
    assert MC.FAST_CYCLING_S == ins.FAST_CYCLING_S == 90.0
    assert MC.SEED == ins.DONOR_SEED == 20261118
    assert hr.P2_SHIFT_S == W.P2_SHIFT_S == MC.P2_SHIFT_S == 600.0
    assert hr.IA_EXCLUDE_S == W.IA_EXCLUDE_S == MC.IA_EXCLUDE_S == 30.0
    assert W.MIN_WINDOWS == MC.MIN_USABLE_WINDOWS == 2
    assert MS.PRIMARY_END == C.PRIMARY_END_S and MS.MAX_H == C.MAX_H_S and MS.SECONDARY_CAP == C.SECONDARY_CAP_S
    assert tuple(MC.WF_FEATURES) == tuple(beats.FEATURES) == tuple(W.FEATURES)
    assert tuple(MC.WF_COLS) == tuple(W.FEATURE_COLS)
    assert set(MC.ABLATION_GROUPS["shape"]) == {"wf_rise_lvl", "wf_rise_chg", "wf_width_lvl", "wf_width_chg"}
    h = np.array([14.9, 15.0, 89.9, 90.0, 149.9, 150.0, 209.9, 210.0, 399.9, 400.0, 400.1])
    np.testing.assert_array_equal(MD.h_bin_of(h) + 1, ins.h_bin(h))


def test_variant_times_agree_between_sides():
    t, tp, tpp = 1000.0, 900.0, 800.0
    for v in hr.VARIANTS:
        ai, ap_, te, shift = hr.variant_times(v, t, tp)
        wi, wp, wpp, wte, _ = W.variant_times(v, t, tp, tpp)
        s, cut = MS.VARIANT_TIMES[v]
        assert (ai, ap_, te) == (wi, wp, wte) and shift == s and wte == wi + cut


# --------------------------------------------------------------------------- provenance (M3, S6)
def test_require_clean():
    clean = {"commit": "a" * 40, "describe": "x", "state": "clean", "dirty": False, "status": []}
    assert provenance.require_clean(False, clean)["allow_dirty"] is False
    for st in ("dirty", "unknown"):
        bad = {**clean, "state": st}
        with pytest.raises(provenance.DirtyStateError):
            provenance.require_clean(False, bad)
        assert provenance.require_clean(True, bad)["allow_dirty"] is True


def test_git_state_shape():
    s = provenance.git_state()
    assert s["state"] in ("clean", "dirty", "unknown")
    assert s["state"] == "unknown" or len(s["commit"]) == 40
    assert not any(str(paths.ROOT) in line for line in s["status"])


@pytest.mark.parametrize("name", ["build_instances", "build_windows", "build_features_wf", "pleth_coverage",
                                  "select_rpeak_detector"])
def test_builders_refuse_dirty_state(monkeypatch, name):
    dirty = {"commit": "b" * 40, "describe": "x-dirty", "state": "dirty", "dirty": True, "status": [" M x"]}
    monkeypatch.setattr(provenance, "git_state", lambda: dict(dirty))
    with pytest.raises(provenance.DirtyStateError):
        _script(name).main([])


def test_parquet_meta_roundtrip(tmp_path):
    p = provenance.write_parquet(pd.DataFrame({"a": [1, 2]}), tmp_path / "x.parquet", {"git": {"commit": "c"}})
    assert provenance.read_meta(p) == {"git": {"commit": "c"}}
    assert pd.read_parquet(p)["a"].tolist() == [1, 2]


def test_credit_statement_matches_spec():
    spec = (paths.ROOT / "SPEC.md").read_text()
    assert provenance.CREDIT in spec
    md = provenance.with_credit("# Report\n")
    assert md.startswith("# Report\n") and provenance.CREDIT in md
    assert "https://creativecommons.org/licenses/by/4.0/" in md and "10.1038/s41597-022-01411-5" in md
