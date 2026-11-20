"""Decision rule, placebo validity and the mechanically chosen headline
(SPEC §6.3, §6.5, §6.8; PREREG §6).

Number formats are [judgment]: percentages with one decimal, mmHg with two
decimals, intervals written "95% CI a to b", negative numbers with a Unicode
minus. The CI after "Δ … mmHg" is the CI of Δ; the CI after "Z% (…" or
"X% (95% CI …)" is the CI of that relative reduction [judgment].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import Contrast
from .constants import COVERAGE_MIN, QUESTION, RULE_MIN_RELATIVE_REDUCTION

MINUS = "−"


def _num(x: float, nd: int) -> str:
    if x == np.inf:
        return "∞"
    if x == -np.inf:
        return MINUS + "∞"
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", MINUS)


def pct(x: float) -> str:
    """A fraction as a percentage number (no % sign)."""
    return _num(100.0 * x, 1) if np.isfinite(x) else _num(x, 1)


def mm(x: float) -> str:
    return _num(x, 2)


def ci_mm(lo, hi) -> str:
    return f"95% CI {mm(lo)} to {mm(hi)}"


def ci_pct(lo, hi) -> str:
    return f"95% CI {pct(lo)}% to {pct(hi)}%"


def placebo_valid(p1_vs_m2: Contrast, model_vs_p1: Contrast) -> bool:
    """SPEC §6.3: P1 does not meet the rule against M2, and the 95% CI of
    RMSE(model) - RMSE(model-P1) lies below zero."""
    return (not p1_vs_m2.meets_rule) and model_vs_p1.delta_hi < 0


@dataclass
class HeadlineInputs:
    N: int                     # cases entering the primary outcome (S-dev)
    m4_m2: Contrast            # X
    m3_m2: Contrast            # Z
    m4_m3: Contrast            # Y
    m4p1_m2: Contrast          # P
    m4_m4p1: Contrast          # real minus placebo (M4)
    m3p1_m2: Contrast
    m3_m3p1: Contrast


def select_row(h: HeadlineInputs) -> dict:
    """Order of checks: M4 vs M2, then M4 placebo validity, then M3 vs M2 and M4 vs M3 (§6.3).
    Placebo validity is checked only for a contrast that meets the rule."""
    m4 = h.m4_m2.meets_rule
    m4_placebo = placebo_valid(h.m4p1_m2, h.m4_m4p1) if m4 else None
    m3 = h.m3_m2.meets_rule
    m3_placebo = placebo_valid(h.m3p1_m2, h.m3_m3p1) if m3 else None
    y = h.m4_m3.meets_rule
    if not m4:
        row = 6
    elif not m4_placebo:
        row = 1
    elif m3 and y:
        row = 2
    elif m3:
        row = 3
    elif y:
        row = 4
    else:
        row = 5
    g_upper = h.m4_m2.rr_hi                                    # G (fraction)
    return {"row": row, "m4_meets": m4, "m4_placebo_valid": m4_placebo, "m3_meets": m3,
            "m3_placebo_valid": m3_placebo, "m4_m3_meets": y,
            "raised": bool(h.m4_m2.delta_lo > 0),                # CI of Δ above zero
            "G": g_upper, "g": -h.m4_m2.delta_lo}


def _hr_append(h: HeadlineInputs, s: dict, plain: bool) -> str:
    if s["m3_meets"] and s["m3_placebo_valid"]:
        z = h.m3_m2
        return (f" Heart rate alone met the rule ({pct(z.rr)}%)." if plain else
                f" Heart rate alone met the rule ({pct(z.rr)}%, {ci_pct(z.rr_lo, z.rr_hi)}).")
    return ""


def sentence1(h: HeadlineInputs, s: dict) -> str:
    """SPEC §6.3 / PREREG §6, Sentence 1 (S-dev)."""
    X, Z, Y, P = h.m4_m2, h.m3_m2, h.m4_m3, h.m4p1_m2
    pre = f"In {h.N} non-cesarean spinal cases (cross-validated), "
    xd = f"{pct(X.rr)}% (Δ {mm(X.delta)} mmHg, {ci_mm(X.delta_lo, X.delta_hi)})"
    m3_ok = s["m3_placebo_valid"] is not False
    row = s["row"]
    if row == 1:
        d = h.m4_m4p1
        return (pre + f"the added signals met the rule ({pct(X.rr)}%), but the placebo control failed "
                f"(placebo inputs: {pct(P.rr)}% against the cuff-only model; real minus placebo inputs: "
                f"Δ {mm(d.delta)} mmHg, {ci_mm(d.delta_lo, d.delta_hi)}), so no claim is made about the "
                f"added signals." + _hr_append(h, s, plain=False))
    if row == 2:
        hr = (f"heart rate alone gave {pct(Z.rr)}% ({ci_pct(Z.rr_lo, Z.rr_hi)})" if m3_ok else
              f"heart rate's share did not pass its placebo check ({pct(Z.rr)}%)")
        return (pre + f"heart rate and pleth waveform features together lowered next-reading MAP forecast "
                f"error by {xd} beyond a cuff-only third-order model; {hr}, and the waveform added "
                f"{pct(Y.rr)}% ({ci_pct(Y.rr_lo, Y.rr_hi)}) beyond heart rate.")
    if row == 3:
        hr = ("and the gain came mainly from heart rate" if m3_ok else
              f"but heart rate's share did not pass its placebo check ({pct(Z.rr)}%)")
        return (pre + f"the added signals lowered next-reading MAP forecast error by {xd}, {hr}; pleth "
                f"waveform features did not meet the rule beyond it ({pct(Y.rr)}%, {ci_pct(Y.rr_lo, Y.rr_hi)}).")
    if row == 4:
        return (pre + f"the added signals lowered next-reading MAP forecast error by {xd}, and the gain came "
                f"from the waveform features; heart rate alone did not meet the rule "
                f"({pct(Z.rr)}%, {ci_pct(Z.rr_lo, Z.rr_hi)}).")
    if row == 5:
        return (pre + f"heart rate and pleth waveform features together lowered next-reading MAP forecast "
                f"error by {xd}; neither heart rate alone ({pct(Z.rr)}%, {ci_pct(Z.rr_lo, Z.rr_hi)}) nor the "
                f"waveform beyond it ({pct(Y.rr)}%, {ci_pct(Y.rr_lo, Y.rr_hi)}) met the rule.")
    raised = "; they raised it" if s["raised"] else ""
    out = (pre + "heart rate and pleth waveform features did not meet the pre-specified rule for lowering "
           f"next-reading MAP forecast error beyond a cuff-only third-order model{raised} ({pct(X.rr)}%, "
           f"Δ {mm(X.delta)} mmHg, {ci_mm(X.delta_lo, X.delta_hi)}; rule: at least 5%, with the 95% CI "
           "excluding zero).")
    if s["G"] > 0:
        out += (f" Gains larger than {pct(s['G'])}% ({mm(s['g'])} mmHg) are not compatible with these data "
                "(approximate 95% CI, see §6.2 of SPEC), for this feature set and a linear model.")
    return out + _hr_append(h, s, plain=False)


ROW_LABELS = {1: "No claim.", 2: "Yes.", 3: "Yes, mainly from heart rate.", 4: "Yes, from the pleth waveform.",
              5: "Yes, together.", 6: "No."}


def label(s: dict) -> str:
    """Row label (§6.3 plain lines), with the row-3 and row-6 variants [judgment: in row 6 the
    "raised" label is checked before the "G >= 5" label; both cannot hold at once in practice]."""
    row = s["row"]
    if row == 3 and s["m3_placebo_valid"] is False:
        return "Yes."
    if row == 6:
        if s["raised"]:
            return "No: the added signals raised the error."
        if s["G"] >= RULE_MIN_RELATIVE_REDUCTION:
            return "Not shown."
    return ROW_LABELS[row]


def answer_line(h: HeadlineInputs, s: dict) -> str:
    """Plain answer line (§6.3): the label, then "In N ... (cross-validated), ..." and the row's
    continuation. A and B are the patient-level mean RMSE of M2 and M4."""
    X, Z, Y = h.m4_m2, h.m3_m2, h.m4_m3
    A, B = mm(X.mean_comp), mm(X.mean_model)
    pre = f"{label(s)} In {h.N} non-cesarean spinal cases (cross-validated), "
    m3_ok = s["m3_placebo_valid"] is not False
    row = s["row"]
    if row == 1:
        return (pre + f"forecast error fell by {pct(X.rr)}%, but the placebo control failed, so no claim is "
                "made about the added signals." + _hr_append(h, s, plain=True))
    if row == 2:
        hr = (f"heart rate alone gave {pct(Z.rr)}%" if m3_ok else
              f"heart rate's share did not pass its placebo check ({pct(Z.rr)}%)")
        return (pre + f"heart rate and the pleth waveform together lowered next-reading forecast error by "
                f"{pct(X.rr)}% (from {A} to {B} mmHg); {hr}, and the waveform added {pct(Y.rr)}% beyond heart rate.")
    if row == 3:
        hr = ("and the gain came mainly from heart rate" if m3_ok else
              f"but heart rate's share did not pass its placebo check ({pct(Z.rr)}%)")
        return (pre + f"the added signals lowered next-reading forecast error by {pct(X.rr)}% (from {A} to {B} "
                f"mmHg), {hr}; pleth waveform features did not meet the rule beyond it ({pct(Y.rr)}%).")
    if row == 4:
        return (pre + f"the added signals lowered next-reading forecast error by {pct(X.rr)}% (from {A} to {B} "
                f"mmHg), and the gain came from the waveform features; heart rate alone did not meet the "
                f"rule ({pct(Z.rr)}%).")
    if row == 5:
        return (pre + f"heart rate and the pleth waveform together lowered next-reading forecast error by "
                f"{pct(X.rr)}% (from {A} to {B} mmHg); neither heart rate alone ({pct(Z.rr)}%) nor the "
                f"waveform beyond it ({pct(Y.rr)}%) met the rule.")
    raised = "; they raised it" if s["raised"] else ""
    out = (pre + f"heart rate and the pleth waveform did not meet the pre-specified rule{raised} "
           f"({pct(X.rr)}%, from {A} to {B} mmHg; rule: at least 5%, with the 95% CI excluding zero).")
    if s["G"] > 0:
        out += (f" Gains larger than {pct(s['G'])}% ({mm(s['g'])} mmHg) are not compatible with these data "
                "(approximate 95% CI), for this feature set and a linear model.")
    return out + _hr_append(h, s, plain=True)


def coverage_rule_applies(coverage: float) -> bool:
    """SPEC §6.5: fewer than 50% of S-ob primary forecasts with waveform features."""
    return bool(coverage < COVERAGE_MIN)


def sentence2(n: int, x: Contrast, coverage: float) -> str:
    """SPEC §6.3 Sentence 2 (S-ob, frozen models), with the coverage-rule variant (§6.5)."""
    pre = f"In the {n} cesarean cases (external, frozen models): "
    if coverage_rule_applies(coverage):
        return pre + (f"insufficient signal coverage ({pct(coverage)}%); {pct(x.rr)}% "
                      f"({ci_pct(x.rr_lo, x.rr_hi)}) is shown for completeness.")
    return pre + (f"{pct(x.rr)}% (Δ {mm(x.delta)} mmHg, {ci_mm(x.delta_lo, x.delta_hi)}), waveform coverage "
                  f"{pct(coverage)}%; this is a case series and neither confirms nor refutes the first sentence.")


def cesarean_line(n: int, x: Contrast, coverage: float) -> str:
    """SPEC §6.3 plain cesarean line; the coverage variant of Sentence 2 if the rule applies."""
    if coverage_rule_applies(coverage):
        return sentence2(n, x, coverage)
    return (f"In the {n} cesarean cases (external, frozen models): {pct(x.rr)}% ({ci_pct(x.rr_lo, x.rr_hi)}); "
            "this is a case series and neither confirms nor refutes the answer above.")


def sentence3(K: int, x: Contrast, yard: dict | None, t_placebo_ok: bool | None) -> str:
    """SPEC §6.3 Sentence 3 (fast-cycling S-dev cases). F is reported only if the 95% CI of D lies
    above zero; if T fails its placebo guard, "(not distinguishable from placebo inputs)" follows F
    [judgment: after F's CI]."""
    out = (f"In the {K} fast-cycling development cases (median cuff interval < 90 s): {pct(x.rr)}% "
           f"(Δ {mm(x.delta)} mmHg, {ci_mm(x.delta_lo, x.delta_hi)}); ")
    if yard is None or not yard["F_reported"]:
        return out + "one extra cuff reading had no detectable value at this interval."
    out += (f"one extra cuff reading's value recovered by the added signals: {pct(yard['F'])}% "
            f"({ci_pct(yard['F_lo'], yard['F_hi'])})")
    if not t_placebo_ok:
        out += " (not distinguishable from placebo inputs)"
    return out + "."


def gb_line(gb_meets: bool, row: int, x: Contrast) -> str | None:
    """SPEC §6.8: the fixed README line if the gradient-boosting answer differs from Sentence 1
    (rows 2-5 = rule met; rows 1 and 6 = not met) [judgment: X% is GB-full vs GB-cuff with the CI of
    that relative reduction]."""
    s1_meets = row in (2, 3, 4, 5)
    if gb_meets == s1_meets:
        return None
    return (f"An exploratory gradient-boosting check on the same inputs gave a different answer "
            f"({pct(x.rr)}%, {ci_pct(x.rr_lo, x.rr_hi)}); see `REPORT.md`.")


def question() -> str:
    return QUESTION
