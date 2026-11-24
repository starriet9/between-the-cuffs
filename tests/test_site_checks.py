"""Checks: µg and mg never in one number, placeholders, forbidden strings, requests; featured rules; series files."""
import json

import numpy as np
import pytest

from btc.site import checks, fixtures as FX, text as T
from btc.site.featured import select_featured
from btc.site.series import NMB_REASON, CaseSeries, CaseSummary, case_status, load_cases, write_cases

from site_support import cases_and_series


# --- phenylephrine (µg) and ephedrine (mg) -------------------------------------------------------------

def test_facts_keep_ug_and_mg_apart():
    cases, series = cases_and_series()
    seen_ug = seen_mg = False
    for c in cases:
        if not c.scored:
            continue
        for label, vals in T.facts(c, series[c.caseid]):
            for v in vals:
                assert not ("µg" in v and checks.MG.search(v)), v
            if label.startswith("Phenylephrine"):       # presence only: the amounts are VitalDB data
                assert vals == [T.phenylephrine_text(c)] and vals[0] in ("charted (amount not shown)", "none charted", "not recorded")
                seen_ug = True
            if label.startswith("Ephedrine"):
                assert vals == [T.ephedrine_text(c)] and vals[0] in ("charted (amount not shown)", "none charted", "not recorded")
                seen_mg = True
    assert seen_ug and seen_mg


def test_unit_check_catches_a_mixed_or_summed_line():
    assert checks.unit_problems("<p>Vasopressors: 100 µg and 5 mg</p>", "")
    assert checks.unit_problems("<p>ok</p>", "vasopressor total: 3")
    assert checks.unit_problems("<dd>100 µg</dd><dd>5 mg</dd>", "") == []
    assert checks.unit_problems("<p>µg/min at the VitalDB-documented concentration of 100 µg/mL</p>", "") == []


def test_stratum_never_sums_the_two_drugs():
    c = CaseSummary(caseid=1, status="scored", phenylephrine_ug=0.0, ephedrine_mg=5.0)
    assert c.stratum == "bolus total only"
    assert CaseSummary(caseid=2, status="scored", phenylephrine_ug=0.0, ephedrine_mg=0.0).stratum == "neither recorded"
    assert CaseSummary(caseid=3, status="scored", pump_track=True).stratum == "pump"


# --- placeholders and forbidden strings ----------------------------------------------------------------

@pytest.mark.parametrize("bad", ["{{AUTHOR_NAME}}", "fell by [X]%", "RMSE NaN mmHg", "TODO: fill", "value None",
                                 "case NNNN", "TBD"])
def test_placeholders_are_found(bad):
    assert any("placeholder" in p or "NaN" in p or "None" in p or "TODO" in p or "NNNN" in p
               for p in checks.content_problems({"x": f"<p>{bad}</p>"}))


@pytest.mark.parametrize("bad", ["/" + "home/someone/x", "C:\\Users\\x", "file:///tmp/a", "/Users/x/y"])
def test_forbidden_strings_are_found(bad):
    assert checks.content_problems({"x": bad}, extra_patterns=[])


def test_private_patterns_are_loaded_and_applied(tmp_path):
    p = tmp_path / "patterns.txt"
    p.write_text("# comment\n(?i)zebrafinch\n\n")
    pats = checks.private_patterns(p)
    assert pats == [("(?i)zebrafinch", "private leak pattern")]
    assert checks.content_problems({"x": "a ZebraFinch b"}, extra_patterns=pats)
    assert checks.private_patterns(tmp_path / "absent.txt") == []


@pytest.mark.skipif(not checks.PRIVATE_PATTERNS.exists(), reason="private leak patterns not present (fresh clone)")
def test_private_patterns_present_locally():
    assert len(checks.private_patterns()) >= 3


def test_clean_text_passes():
    assert checks.content_problems({"x": "Derived from VitalDB (CC BY 4.0). Pump track: none. 2.5% (95% CI 1.0% "
                                          "to 4.0%); infusions; information; https://example.org/a/b"}) == []


def test_email_check():
    assert checks.content_problems({"x": "write to someone@example.org"}, email="someone@example.org")


def test_request_check():
    assert checks.request_problems('<img src="https://example.org/a.png">', "")
    assert checks.request_problems('<link rel="stylesheet" href="//cdn.example.org/a.css">', "")
    assert checks.request_problems("<p></p>", "fetch('data.json')")
    assert checks.request_problems('<a href="https://example.org">x</a><img src="assets/a.png">', "") == []


# --- featured cases (SPEC §8.2) ------------------------------------------------------------------------

def _c(cid, m4, n=10, onset=1, pump=False, changes=0):
    return CaseSummary(caseid=cid, status="scored", n_forecasts=n, rmse={"M2": m4 + 1, "M3": m4, "M4": m4,
                       "M4-P1": m4 + 1}, new_onset_80=onset, pump_track=pump, pump_changes=changes)


def test_rule1_median_odd_and_even():
    odd = [_c(1, 3.0), _c(2, 5.0), _c(3, 4.0)]
    assert select_featured(odd)[1] == 3
    even = [_c(1, 3.0), _c(2, 5.0), _c(3, 4.0), _c(4, 6.0)]
    assert select_featured(even)[1] == 2          # the higher-RMSE case of the two middle cases


def test_rule1_needs_onset_and_eight_forecasts():
    cs = [_c(1, 3.0, onset=0), _c(2, 4.0, n=7), _c(3, 5.0), _c(4, 9.0, onset=0)]
    f = select_featured(cs)
    assert f[1] == 3 and f[2] == 4


def test_rule2_takes_next_case_and_ties_go_to_lowest_id():
    cs = [_c(5, 6.0), _c(2, 6.0), _c(3, 1.0)]     # rule 1: median of (1.0, 6.0, 6.0) = 6.0 -> lowest id 2
    f = select_featured(cs)
    assert f[1] == 2 and f[2] == 5


def test_rule3_pump_with_most_changes_skips_taken_cases():
    cs = [_c(1, 4.0, pump=True, changes=5), _c(2, 5.0), _c(3, 3.0, pump=True, changes=5), _c(4, 2.0, pump=True,
          changes=2)]
    f = select_featured(cs)
    assert f[1] == 1                                # median (2, 3, 4, 5) -> 4.0 -> case 1
    assert f[2] == 2 and f[3] == 3


def test_fixture_featured_cases_exist():
    cases, series = cases_and_series()
    f = select_featured(cases)
    assert all(f[r] in series for r in (1, 2, 3)) and len(set(f.values())) == 3


# --- series files ---------------------------------------------------------------------------------------

def test_series_roundtrip_and_validation(tmp_path):
    cases, series = cases_and_series()
    write_cases(tmp_path, cases, series, synthetic=True)
    syn, c2, s2 = load_cases(tmp_path)
    assert syn and [c.caseid for c in c2] == [c.caseid for c in cases]
    a, b = series[9004], s2[9004]
    assert np.allclose(a.cuff_map, b.cuff_map) and len(a.issues) == len(b.issues)
    assert np.allclose(a.issues[3].m4.map, b.issues[3].m4.map)
    d = json.loads((tmp_path / "series" / "9004.json").read_text())
    d["issues"][0]["h_s"] = [5.0] + d["issues"][0]["h_s"][1:]
    with pytest.raises(ValueError):
        CaseSeries.from_dict(d)
    with pytest.raises(ValueError):
        CaseSummary(caseid=1, status="not analysable").validate()       # a reason is required


def test_fixture_case_ids_are_not_vitaldb_ids():
    assert min(FX.CASE_IDS) > 6388


def test_case_status_and_the_not_analysed_reason():
    """SPEC §2.2: an analysable case with a neuromuscular blocker recorded is shown as not analysed."""
    assert case_status(analysable=False, nmb_recorded=True, entered=False) == "not analysable"
    assert case_status(analysable=True, nmb_recorded=True, entered=True) == "not analysed"
    assert case_status(analysable=True, nmb_recorded=False, entered=True) == "scored"
    assert case_status(analysable=True, nmb_recorded=False, entered=False) == "not scored"
    CaseSummary(caseid=1, status="not analysed", reason=NMB_REASON).validate()
    for reason in (None, "3 valid readings in the window"):
        with pytest.raises(ValueError):
            CaseSummary(caseid=1, status="not analysed", reason=reason).validate()
