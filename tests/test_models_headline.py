"""Mechanical headline selection and the pre-written sentences (SPEC §6.3, §6.5, §6.8; PREREG §6)."""
import numpy as np
import pytest

from btc.models.bootstrap import Contrast
from btc.models.headline import (HeadlineInputs, answer_line, cesarean_line, gb_line, label, select_row,
                                 sentence1, sentence2, sentence3)

M = "−"


def C(rr, dlo, dhi, comp=7.0, rr_lo=None, rr_hi=None, model="m", c="c"):
    """A contrast with relative reduction rr and Delta CI (dlo, dhi)."""
    mm = comp * (1 - rr)
    return Contrast(model, c, 100, mm, comp, mm - comp, dlo, dhi, rr,
                    rr - 0.03 if rr_lo is None else rr_lo, rr + 0.03 if rr_hi is None else rr_hi)


MEET = dict(rr=0.10, dlo=-1.0, dhi=-0.3)
MISS = dict(rr=0.01, dlo=-0.2, dhi=0.1)
NEAR = dict(rr=0.06, dlo=-0.5, dhi=0.02)      # >= 5% but CI includes zero: does not meet


def H(m4=MEET, m3=MISS, y=MISS, p1=MISS, m4p1=dict(rr=0.08, dlo=-0.9, dhi=-0.2), m3p1_m2=MISS,
      m3_m3p1=dict(rr=0.05, dlo=-0.6, dhi=-0.1)):
    return HeadlineInputs(N=210, m4_m2=C(**m4), m3_m2=C(**m3), m4_m3=C(**y), m4p1_m2=C(**p1),
                          m4_m4p1=C(**m4p1), m3p1_m2=C(**m3p1_m2), m3_m3p1=C(**m3_m3p1))


PRE = "In 210 non-cesarean spinal cases (cross-validated), "


def test_rule_needs_both_margin_and_ci():
    assert C(**MEET).meets_rule and not C(**MISS).meets_rule and not C(**NEAR).meets_rule
    assert C(rr=0.05, dlo=-1, dhi=-1e-9).meets_rule                    # unrounded, inclusive 5%
    assert not C(rr=0.0499999, dlo=-1, dhi=-0.5).meets_rule


@pytest.mark.parametrize("kw,row", [
    (dict(m4=MISS), 6), (dict(m4=NEAR, m3=MEET, y=MEET), 6),
    (dict(p1=MEET), 1), (dict(m4p1=dict(rr=0.0, dlo=-0.1, dhi=0.1)), 1),
    (dict(m3=MEET, y=MEET), 2), (dict(m3=MEET, y=MISS), 3), (dict(m3=MISS, y=MEET), 4), (dict(), 5)])
def test_row_selection(kw, row):
    assert select_row(H(**kw))["row"] == row


def test_placebo_checked_only_when_rule_met():
    s = select_row(H(m4=MISS, p1=MEET))
    assert s["row"] == 6 and s["m4_placebo_valid"] is None and s["m3_placebo_valid"] is None


def test_row1_texts_and_hr_append():
    h = H(p1=MEET, m3=MEET, m3p1_m2=MISS)
    s = select_row(h)
    t = sentence1(h, s)
    assert t.startswith(PRE + "the added signals met the rule (10.0%), but the placebo control failed "
                              "(placebo inputs: 10.0% against the cuff-only model; real minus placebo inputs: Δ ")
    assert t.endswith("so no claim is made about the added signals. Heart rate alone met the rule "
                      "(10.0%, 95% CI 7.0% to 13.0%).")
    a = answer_line(h, s)
    assert a == ("No claim. " + PRE + "forecast error fell by 10.0%, but the placebo control failed, so no claim is "
                 "made about the added signals. Heart rate alone met the rule (10.0%).")
    # M3 fails its placebo: no append, same row
    h2 = H(p1=MEET, m3=MEET, m3p1_m2=MEET)
    s2 = select_row(h2)
    assert s2["row"] == 1 and "Heart rate alone" not in sentence1(h2, s2)


def test_row2_texts_and_m3_placebo_variant():
    h = H(m3=MEET, y=MEET)
    s = select_row(h)
    assert sentence1(h, s) == (PRE + f"heart rate and pleth waveform features together lowered next-reading MAP "
                               f"forecast error by 10.0% (Δ {M}0.70 mmHg, 95% CI {M}1.00 to {M}0.30) beyond a "
                               "cuff-only third-order model; heart rate alone gave 10.0% (95% CI 7.0% to 13.0%), "
                               "and the waveform added 10.0% (95% CI 7.0% to 13.0%) beyond heart rate.")
    assert answer_line(h, s) == ("Yes. " + PRE + "heart rate and the pleth waveform together lowered next-reading "
                                 "forecast error by 10.0% (from 7.00 to 6.30 mmHg); heart rate alone gave 10.0%, "
                                 "and the waveform added 10.0% beyond heart rate.")
    h2 = H(m3=MEET, y=MEET, m3p1_m2=MEET)
    s2 = select_row(h2)
    assert s2["row"] == 2
    assert "heart rate's share did not pass its placebo check (10.0%), and the waveform" in sentence1(h2, s2)
    assert "heart rate's share did not pass its placebo check (10.0%), and the waveform" in answer_line(h2, s2)


def test_row3_texts_and_label_variant():
    h = H(m3=MEET, y=NEAR)
    s = select_row(h)
    assert s["row"] == 3
    assert sentence1(h, s).endswith(", and the gain came mainly from heart rate; pleth waveform features did not "
                                    "meet the rule beyond it (6.0%, 95% CI 3.0% to 9.0%).")
    assert answer_line(h, s).startswith("Yes, mainly from heart rate. " + PRE)
    h2 = H(m3=MEET, y=NEAR, m3_m3p1=dict(rr=0.0, dlo=-0.1, dhi=0.1))
    s2 = select_row(h2)
    assert s2["row"] == 3 and label(s2) == "Yes."
    assert ", but heart rate's share did not pass its placebo check (10.0%); pleth" in sentence1(h2, s2)
    assert ", but heart rate's share did not pass its placebo check (10.0%); pleth" in answer_line(h2, s2)


def test_rows_4_and_5_texts():
    h = H(m3=MISS, y=MEET)
    s = select_row(h)
    assert sentence1(h, s).endswith("and the gain came from the waveform features; heart rate alone did not meet "
                                    "the rule (1.0%, 95% CI −2.0% to 4.0%).")
    assert answer_line(h, s).startswith("Yes, from the pleth waveform. ")
    h5 = H()
    s5 = select_row(h5)
    assert sentence1(h5, s5).endswith("; neither heart rate alone (1.0%, 95% CI −2.0% to 4.0%) nor the waveform "
                                      "beyond it (1.0%, 95% CI −2.0% to 4.0%) met the rule.")
    assert answer_line(h5, s5).startswith("Yes, together. ")


def test_row6_bound_raised_and_labels():
    h = H(m4=dict(rr=0.01, dlo=-0.25, dhi=0.1, rr_hi=0.035))
    s = select_row(h)
    t = sentence1(h, s)
    assert t == (PRE + "heart rate and pleth waveform features did not meet the pre-specified rule for lowering "
                 "next-reading MAP forecast error beyond a cuff-only third-order model (1.0%, Δ −0.07 mmHg, "
                 "95% CI −0.25 to 0.10; rule: at least 5%, with the 95% CI excluding zero). Gains larger than 3.5% "
                 "(0.25 mmHg) are not compatible with these data (approximate 95% CI, see §6.2 of SPEC), for this "
                 "feature set and a linear model.")
    assert answer_line(h, s) == ("No. " + PRE + "heart rate and the pleth waveform did not meet the pre-specified "
                                 "rule (1.0%, from 7.00 to 6.93 mmHg; rule: at least 5%, with the 95% CI excluding "
                                 "zero). Gains larger than 3.5% (0.25 mmHg) are not compatible with these data "
                                 "(approximate 95% CI), for this feature set and a linear model.")
    # G >= 5: "Not shown."
    s2 = select_row(H(m4=dict(rr=0.03, dlo=-0.6, dhi=0.05, rr_hi=0.08)))
    assert label(s2) == "Not shown."
    # CI of Delta above zero: "; they raised it", label, and G <= 0 omits the bound sentence
    h3 = H(m4=dict(rr=-0.04, dlo=0.1, dhi=0.5, rr_lo=-0.07, rr_hi=-0.01))
    s3 = select_row(h3)
    t3 = sentence1(h3, s3)
    assert "third-order model; they raised it (−4.0%, Δ 0.28 mmHg" in t3 and "Gains larger" not in t3
    a3 = answer_line(h3, s3)
    assert a3.startswith("No: the added signals raised the error. " + PRE)
    assert "rule; they raised it (−4.0%, from 7.00 to 7.28 mmHg;" in a3 and "Gains larger" not in a3
    # heart-rate append in row 6
    h4 = H(m4=MISS, m3=MEET)
    s4 = select_row(h4)
    assert sentence1(h4, s4).endswith(" Heart rate alone met the rule (10.0%, 95% CI 7.0% to 13.0%).")
    assert answer_line(h4, s4).endswith(" Heart rate alone met the rule (10.0%).")


def test_sentence2_and_cesarean_line_with_coverage_rule():
    x = C(rr=0.02, dlo=-0.3, dhi=0.2)
    assert sentence2(20, x, 0.84) == ("In the 20 cesarean cases (external, frozen models): 2.0% (Δ −0.14 mmHg, "
                                      "95% CI −0.30 to 0.20), waveform coverage 84.0%; this is a case series and "
                                      "neither confirms nor refutes the first sentence.")
    assert cesarean_line(20, x, 0.84) == ("In the 20 cesarean cases (external, frozen models): 2.0% (95% CI −1.0% to "
                                          "5.0%); this is a case series and neither confirms nor refutes the answer above.")
    v = ("In the 20 cesarean cases (external, frozen models): insufficient signal coverage (45.0%); 2.0% "
         "(95% CI −1.0% to 5.0%) is shown for completeness.")
    assert sentence2(20, x, 0.45) == v and cesarean_line(20, x, 0.45) == v
    assert sentence2(20, x, 0.50).endswith("first sentence.")           # 50% exactly: rule does not apply


def test_sentence3_variants():
    x = C(rr=0.03, dlo=-0.4, dhi=0.1)
    yard = {"F": 0.25, "F_lo": -0.1, "F_hi": np.inf, "F_reported": True}
    base = "In the 34 fast-cycling development cases (median cuff interval < 90 s): 3.0% (Δ −0.21 mmHg, 95% CI −0.40 to 0.10); "
    assert sentence3(34, x, yard, True) == base + ("one extra cuff reading's value recovered by the added signals: "
                                                   "25.0% (95% CI −10.0% to ∞%).")
    assert sentence3(34, x, yard, False).endswith("(95% CI −10.0% to ∞%) (not distinguishable from placebo inputs).")
    assert sentence3(34, x, {**yard, "F_reported": False}, False) == base + (
        "one extra cuff reading had no detectable value at this interval.")


def test_gb_line_only_when_answers_differ():
    x = C(rr=0.07, dlo=-0.6, dhi=-0.1)
    assert gb_line(True, 6, x).startswith("An exploratory gradient-boosting check on the same inputs gave a different "
                                          "answer (7.0%, 95% CI 4.0% to 10.0%); see `REPORT.md`.")
    assert gb_line(True, 1, x) is not None and gb_line(False, 3, x) is not None
    assert gb_line(True, 2, x) is None and gb_line(False, 6, x) is None and gb_line(False, 1, x) is None
