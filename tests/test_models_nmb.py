"""SPEC §2.2 neuromuscular-blocker rule in the model code, on synthetic tables only (SPEC §9 step 5).

Only analysed cases (analysable, no neuromuscular blocker recorded) are fitted or scored; the
analysable set and the P1 donors are unchanged; the sensitivity analyses include the excluded cases.
"""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from btc.models import pipeline as P
from btc.models.data import build_frame, build_triples, tables_from_frames
from btc.models.folds import make_fold_plan
from btc.models.outputs import Writer
from btc.models.synthetic import SDEV, SOB, make_synthetic

META = {"commit": "test", "data": "synthetic"}


@pytest.fixture(scope="module")
def syn():
    return make_synthetic(seed=1, wf_signal=2.0)


def _T(syn, cohort="sdev"):
    return tables_from_frames(cohort, syn[cohort])


def _nmb(T) -> set:
    return set(T.cases.loc[T.cases["nmb_recorded"], "caseid"].astype(int))


# --------------------------------------------------------------------------- the tables
def test_synthetic_cohorts_have_nmb_cases(syn):
    for cohort, k in (("sdev", SDEV.n_nmb), ("sob", SOB.n_nmb)):
        c = syn[cohort]["cases"]
        assert k > 0 and int(c["nmb_recorded"].sum()) == k
        assert c.loc[c["nmb_recorded"], "analysable"].all()
        assert (c["analysed"] == (c["analysable"] & ~c["nmb_recorded"])).all()


def test_tables_analysed_with_and_without_include_nmb(syn):
    T = _T(syn)
    nmb = _nmb(T)
    a = T.analysable
    assert set(a["caseid"]) == set(T.cases.loc[T.cases["analysable"], "caseid"]) and nmb <= set(a["caseid"])
    an = T.analysed
    assert set(an["caseid"]) == set(a["caseid"]) - nmb and np.all(np.diff(an["caseid"].to_numpy()) > 0)
    Tn = replace(T, include_nmb=True)
    pd.testing.assert_frame_equal(Tn.analysed, a)
    pd.testing.assert_frame_equal(Tn.analysable, a)               # analysable keeps its meaning
    assert T.counts() == {"n_analysable_cases": len(a), "n_nmb_recorded_cases": len(nmb),
                          "n_analysed_cases": len(a) - len(nmb), "nmb_recorded_caseids": sorted(nmb),
                          "include_nmb": False}
    assert Tn.counts()["n_analysed_cases"] == len(a) and Tn.counts()["n_nmb_recorded_cases"] == len(nmb)


def test_missing_nmb_value_counts_as_not_recorded(syn):
    S = {k: v.copy() for k, v in syn["sdev"].items()}
    c = S["cases"].drop(columns="analysed")
    c["nmb_recorded"] = c["nmb_recorded"].astype(object)
    i = c.index[c["nmb_recorded"].astype(bool)][0]
    c.loc[i, "nmb_recorded"] = np.nan
    T = tables_from_frames("sdev", {**S, "cases": c})
    assert int(c.loc[i, "caseid"]) in set(T.analysed["caseid"])
    assert len(T.analysed) == len(T.analysable) - (SDEV.n_nmb - 1)


def test_nmb_recorded_is_required_and_analysed_must_follow_the_rule(syn):
    S = syn["sdev"]
    with pytest.raises(ValueError, match="nmb_recorded"):
        tables_from_frames("sdev", {**S, "cases": S["cases"].drop(columns=["nmb_recorded"])})
    bad = S["cases"].assign(analysed=S["cases"]["analysable"])
    with pytest.raises(ValueError, match="SPEC §2.2"):
        tables_from_frames("sdev", {**S, "cases": bad})


def test_frames_folds_and_triples_exclude_nmb_cases(syn):
    T = _T(syn)
    nmb = _nmb(T)
    assert not nmb & set(make_fold_plan(T.analysed).caseids)
    for window, k in (("primary", 1), ("primary", 2), ("secondary", 1)):
        assert not nmb & set(build_frame(T, window, k)["caseid"])
    tri, u = build_triples(T)
    assert not nmb & set(tri["caseid"]) and not nmb & set(u["caseid"])
    Tn = replace(T, include_nmb=True)
    assert nmb <= set(build_frame(Tn)["caseid"]) and nmb <= set(make_fold_plan(Tn.analysed).caseids)
    # their instances and features exist: the sensitivity analysis needs them
    assert nmb <= set(T.instances["caseid"]) and nmb <= set(T.features_hr["caseid"]) \
        and nmb <= set(T.features_wf["caseid"])


def test_donors_unchanged_by_the_rule():
    """The nmb draw is a separate random stream: with or without nmb cases every table is the same
    apart from nmb_recorded and analysed, so the P1 donors are those drawn from all analysable cases."""
    a = make_synthetic(seed=3)
    b = make_synthetic(seed=3, sdev=replace(SDEV, n_nmb=0), sob=replace(SOB, n_nmb=0))
    drop = ["nmb_recorded", "analysed"]
    for cohort in ("sdev", "sob"):
        ca, cb = a[cohort]["cases"], b[cohort]["cases"]
        assert ca["nmb_recorded"].sum() > 0 and cb["nmb_recorded"].sum() == 0
        pd.testing.assert_series_equal(ca["donor_caseid"], cb["donor_caseid"])
        pd.testing.assert_frame_equal(ca.drop(columns=drop), cb.drop(columns=drop))
        for name in ("readings", "instances", "features_hr", "features_wf"):
            pd.testing.assert_frame_equal(a[cohort][name], b[cohort][name])
    c = a["sdev"]["cases"]
    nmb = set(c.loc[c["nmb_recorded"], "caseid"])
    assert c.loc[c["nmb_recorded"], "donor_caseid"].notna().all()          # nmb cases keep their donors
    assert nmb & set(c["donor_caseid"].dropna().astype(int))                # and still serve as donors


def test_primary_stages_refuse_include_nmb_tables(syn, tmp_path):
    Tn = replace(_T(syn), include_nmb=True)
    with pytest.raises(ValueError, match="sensitivity"):
        P.confirmatory(Tn, Writer(tmp_path, META))
    with pytest.raises(ValueError, match="sensitivity"):
        P.freeze(Tn, Writer(tmp_path, META))


# --------------------------------------------------------------------------- confirmatory N
def test_headline_N_is_the_analysed_patients_entering(syn, tmp_path):
    T = _T(syn)
    nmb = _nmb(T)
    ctx = P.confirmatory(T, Writer(tmp_path / "sdev", META), n_jobs=2, n_boot=200)
    s = json.loads((tmp_path / "sdev/confirmatory/summary.json").read_text())
    per_case = ctx.frame.groupby("caseid").size()
    assert s["N"] == int((per_case >= 3).sum()) and not nmb & set(ctx.frame["caseid"])
    assert set(per_case.index) <= set(T.analysed["caseid"])
    assert (s["n_analysable_cases"], s["n_nmb_recorded_cases"], s["n_analysed_cases"]) == (
        len(T.analysable), len(nmb), len(T.analysable) - len(nmb))
    assert s["sentence_1"].startswith(f"In {s['N']} non-cesarean spinal cases")
    assert f" In {s['N']} non-cesarean spinal cases" in s["answer_line"]
    folds = pd.read_csv(tmp_path / "sdev/confirmatory/folds.csv", skiprows=1)
    assert set(folds["caseid"]) == set(T.analysed["caseid"])
    pp = pd.read_csv(tmp_path / "sdev/confirmatory/per_patient_primary.csv", skiprows=1)
    assert not nmb & set(pp["caseid"])


# --------------------------------------------------------------------------- frozen S-ob scoring
@pytest.mark.parametrize("n_nmb", [0, 1])
def test_sob_sensitivity_output(tmp_path, n_nmb):
    syn = make_synthetic(seed=1, wf_signal=2.0, sob=replace(SOB, n_nmb=n_nmb))
    Td, To = _T(syn), _T(syn, "sob")
    nmb = sorted(_nmb(To))
    fr = P.freeze(Td, Writer(tmp_path, META), n_jobs=2)
    assert fr["sdev_cases"] == Td.analysed["caseid"].tolist()          # frozen models: analysed S-dev only
    s = P.score_sob(fr, To, Writer(tmp_path / "sob", META), n_boot=100)
    fz = tmp_path / "sob/frozen_scoring"
    d = fz / P.NMB_DIR
    sens = json.loads((d / "summary.json").read_text())
    prim = json.loads((fz / "summary.json").read_text())
    assert len(nmb) == n_nmb and sens["nmb_recorded_caseids"] == nmb
    assert sens["n_nmb_recorded_cases"] == prim["n_nmb_recorded_cases"] == n_nmb
    assert prim["n_analysed_cases"] == prim["n_analysable_cases"] - n_nmb
    assert sens["analysis"] == P.NMB_ANALYSIS and sens["n"] == prim["n"] + n_nmb   # synthetic nmb cases enter
    assert prim["sentence_2"].startswith(f"In the {prim['n']} cesarean cases") and "sensitivity" not in json.dumps(prim)
    for f in ("model_summary.csv", "contrasts.csv", "per_patient_primary.csv", "predictions_primary.csv"):
        assert (d / f).exists()
    pp = pd.read_csv(fz / "per_patient_primary.csv", skiprows=1)
    ppn = pd.read_csv(d / "per_patient_primary.csv", skiprows=1)
    assert not set(nmb) & set(pp["caseid"]) and set(nmb) <= set(ppn["caseid"])
    assert s["sensitivity_including_nmb"]["n_nmb_recorded_cases"] == n_nmb
    if n_nmb == 0:
        assert "count 0" in sens["note"]
        pd.testing.assert_frame_equal(pd.read_csv(d / "contrasts.csv", skiprows=1),
                                      pd.read_csv(fz / "contrasts.csv", skiprows=1))
