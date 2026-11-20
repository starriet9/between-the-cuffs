"""Synthetic fixtures for building and testing the site before any real result exists.

Nothing here reads VitalDB data. Case IDs 9001-9021 do not exist in VitalDB (about 6,400 cases).
The headline texts are produced by `btc/models/headline.py` from invented contrasts, and written in
the pipeline's own file layout with its `Writer` (meta "data" = "synthetic(...)"), so the site reads
fixtures through exactly the path it will use for real results. Every render from these files is
watermarked "SYNTHETIC — NOT A RESULT", and the publish build refuses them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from btc.models import headline as H
from btc.models.bootstrap import Contrast, SubjectBootstrap, contrast
from btc.models.outputs import Writer

from .config import SiteConfig
from .series import CaseSeries, CaseSummary, Curve, IssuePoint, write_cases

N_SDEV, K_FAST = 203, 35        # invented; N = analysed S-dev patients entering the primary outcome (SPEC §2.2)
SDEV_M2 = 6.84                    # invented S-dev mean patient RMSE of M2, mmHg
CASE_IDS = tuple(range(9001, 9022))
NOT_ANALYSABLE = {9015: "3 valid readings in the window"}
PUMP_CASES = (9004, 9008, 9011, 9017, 9020)
NEITHER_CASES = (9002, 9010, 9014, 9019)


def C(model, comp, rr, lo, hi, base=SDEV_M2, n=N_SDEV) -> Contrast:
    """A contrast with relative reduction rr (CI lo..hi) against a comparator mean `base`."""
    mm_ = base * (1 - rr)
    return Contrast(model, comp, n, mm_, base, mm_ - base, -base * hi, -base * lo, rr, lo, hi)


MEET = (0.082, 0.047, 0.118)
MEET_HR = (0.061, 0.031, 0.090)
MISS = (0.014, -0.012, 0.039)
NEAR = (0.052, -0.006, 0.108)
P1_OK = (0.004, -0.011, 0.019)
P1_BAD = (0.071, 0.040, 0.101)
REAL_VS_P1_OK = (0.078, 0.044, 0.110)
REAL_VS_P1_BAD = (0.011, -0.020, 0.041)

VARIANTS = {
    # name: (m4, m3, y, m4p1, m4_vs_m4p1, m3p1, m3_vs_m3p1, extra)
    "row1": (MEET, MISS, MEET, P1_BAD, REAL_VS_P1_BAD, P1_OK, REAL_VS_P1_OK, {}),
    "row1-hr": (MEET, MEET_HR, MISS, P1_BAD, REAL_VS_P1_BAD, P1_OK, REAL_VS_P1_OK, {}),
    "row2": (MEET, MEET_HR, (0.055, 0.021, 0.088), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {}),
    "row2-hr-placebo": (MEET, MEET_HR, (0.055, 0.021, 0.088), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_BAD, {}),
    "row2-coverage": (MEET, MEET_HR, (0.055, 0.021, 0.088), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK,
                      {"sob_coverage": 0.42}),
    "row3": (MEET, MEET_HR, (0.023, -0.004, 0.049), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {}),
    "row3-hr-placebo": (MEET, MEET_HR, (0.023, -0.004, 0.049), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_BAD, {}),
    "row3-deviation": (MEET, MEET_HR, (0.023, -0.004, 0.049), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK,
                       {"deviations": 2}),
    "row4": (MEET, MISS, (0.069, 0.040, 0.097), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {}),
    "row5": (MEET, NEAR, (0.031, -0.002, 0.063), P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {}),
    "row6": ((0.012, -0.011, 0.034), MISS, MISS, P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {"no_F": True}),
    "row6-not-shown": ((0.021, -0.019, 0.061), MISS, MISS, P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {}),
    "row6-raised": ((-0.031, -0.052, -0.010), MISS, (-0.045, -0.066, -0.024), P1_OK, REAL_VS_P1_OK, P1_OK,
                    REAL_VS_P1_OK, {"no_F": True}),
    "row6-hr": ((0.043, -0.006, 0.091), MEET_HR, (-0.019, -0.044, 0.006), P1_OK, REAL_VS_P1_OK, P1_OK,
                REAL_VS_P1_OK, {"F_placebo_fail": True}),
    "row6-gb": ((0.012, -0.011, 0.034), MISS, MISS, P1_OK, REAL_VS_P1_OK, P1_OK, REAL_VS_P1_OK, {"gb_meets": True}),
}
SCREENSHOT_VARIANTS = ("row2", "row3", "row6")


def headline_inputs(variant: str) -> tuple:
    m4, m3, y, m4p1, m4v, m3p1, m3v, extra = VARIANTS[variant]
    x = C("M4", "M2", *m4)
    z = C("M3", "M2", *m3)
    yy = C("M4", "M3", *y, base=z.mean_model)
    hi = H.HeadlineInputs(N=N_SDEV, m4_m2=x, m3_m2=z, m4_m3=yy, m4p1_m2=C("M4-P1", "M2", *m4p1),
                          m4_m4p1=C("M4", "M4-P1", *m4v, base=x.mean_model / (1 - m4v[0])),
                          m3p1_m2=C("M3-P1", "M2", *m3p1),
                          m3_m3p1=C("M3", "M3-P1", *m3v, base=z.mean_model / (1 - m3v[0])))
    return hi, extra


def config_for(variant: str, **over) -> SiteConfig:
    extra = VARIANTS[variant][-1] if variant in VARIANTS else {}
    d = dict(frozen_date="2026-12-01", frozen_tag="models-frozen", scored_date="2026-12-02",
             scored_ref="sob-scored", deviations=extra.get("deviations", 0),
             reproduce_command="python scripts/reproduce.py", checksums_path="results/checksums.csv",
             last_updated="2026-12-02", repo_url="https://github.com/starriet9/between-the-cuffs",
             page_url="https://starriet9.github.io/between-the-cuffs/")
    d.update(over)
    return SiteConfig(**d)


# ---------------------------------------------------------------------------
# invented cesarean cases
# ---------------------------------------------------------------------------

H_GRID = np.arange(15.0, 400.0 + 1e-9, 5.0)
BIN_EDGES = (15.0, 90.0, 150.0, 210.0)
HALF90_M2 = (6.4, 7.9, 8.8, 9.9)


def _half(h, scale=1.0):
    idx = np.searchsorted(BIN_EDGES, h, side="right") - 1
    return np.asarray(HALF90_M2, dtype=float)[np.clip(idx, 0, 3)] * scale


def _truth(rng, base):
    """An invented MAP course: a post-spinal fall with partial recoveries."""
    t_drop = rng.uniform(150, 600)
    depth = rng.uniform(14, 34)
    tau = rng.uniform(100, 220)
    kicks = sorted(rng.uniform(t_drop + 120, 1750, size=rng.integers(2, 5)))
    amp = rng.uniform(5, 11, size=len(kicks))

    def f(t):
        t = np.asarray(t, dtype=float)
        v = base - depth * (1 - np.exp(-np.clip(t - t_drop, 0, None) / tau)) * (t > t_drop)
        for k, a in zip(kicks, amp):
            v = v + a * (1 - np.exp(-np.clip(t - k, 0, None) / 60)) * np.exp(-np.clip(t - k, 0, None) / 700)
        return v + 2.0 * np.sin(t / 97.0)
    return f, kicks


def _forecast(t_r, m_r, i, h):
    """Damped-slope forecast from the last three readings (stands in for the frozen M2)."""
    tt, mm_ = t_r[i - 2:i + 1], m_r[i - 2:i + 1]
    slope = np.polyfit(tt / 60.0, mm_, 1)[0]
    return m_r[i] + 0.45 * slope * (np.asarray(h) / 60.0) * np.exp(-np.asarray(h) / 400.0)


def make_case(caseid: int) -> tuple:
    rng = np.random.default_rng(caseid)
    status = "not analysable" if caseid in NOT_ANALYSABLE else "scored"
    pump = caseid in PUMP_CASES
    if caseid in NEITHER_CASES:
        phe, eph = 0.0, 0.0
    elif pump:
        phe, eph = float(rng.choice([0, 0, 100])), 0.0
    else:
        phe, eph = float(rng.choice([50, 100, 150, 200, 300, 400])), float(rng.choice([0, 0, 5, 10, 15]))
    if status != "scored":
        return CaseSummary(caseid=caseid, status=status, reason=NOT_ANALYSABLE[caseid], phenylephrine_ug=phe,
                           ephedrine_mg=eph, pump_track=pump), None
    base = rng.uniform(84, 102)
    f, kicks = _truth(rng, base)
    iv = rng.uniform(58, 92)
    t, ts = rng.uniform(15, 60), []
    while t <= 1800:
        ts.append(round(t, 1))
        t += iv + rng.normal(0, 6)
    t_r = np.asarray(ts)
    m_r = np.round(f(t_r) + rng.normal(0, 2.6, len(t_r)))
    baseline = float(np.mean(m_r[:3]))
    hr_t = np.arange(0.0, 1800.0, 1.0)
    hr = 84 + rng.normal(0, 1.2, len(hr_t)).cumsum() * 0.15 - 0.35 * (f(hr_t) - base)
    gap = rng.uniform(300, 1500)
    hr[(hr_t > gap) & (hr_t < gap + rng.uniform(10, 50))] = np.nan
    pl_s = np.arange(0.0, 1800.0, 10.0)
    u, us = True, []
    stay, back = rng.uniform(0.78, 0.96), rng.uniform(0.25, 0.6)
    for _ in pl_s:
        u = (rng.random() < stay) if u else (rng.random() < back)
        us.append(u)
    pl_u = np.asarray(us)
    pump_t = pump_v = None
    if pump:
        n_ch = int(rng.integers(2, 6))
        pump_t = np.concatenate([[0.0], np.sort(rng.uniform(200, 1700, n_ch))])
        pump_v = np.concatenate([[0.0], rng.choice([15.0, 30.0, 45.0, 60.0, 90.0], n_ch)])
        for j in range(1, len(pump_v)):
            if pump_v[j] == pump_v[j - 1]:
                pump_v[j] += 15.0
    issues, err = [], {"M2": [], "M3": [], "M4": [], "M4-P1": []}
    k4 = rng.uniform(-0.08, 0.2)          # this case's invented M4 gain
    onset, flag = 0, {"M2": 0, "M4": 0}
    thr = 0.8 * baseline
    covered = []
    for i in range(2, len(t_r)):
        nxt = i + 1 < len(t_r)
        t_next = float(t_r[i + 1]) if nxt else None
        y_next = float(m_r[i + 1]) if nxt else None
        m2 = _forecast(t_r, m_r, i, H_GRID)
        pull = k4 * (f(t_r[i] + H_GRID) - m2)
        m4 = m2 + pull + 0.3 * np.sin(H_GRID / 50 + caseid)
        prev = t_r[i - 1]
        win = (pl_s >= max(prev, t_r[i] - 120) - 10) & (pl_s + 10 <= t_r[i])
        cov = int(pl_u[win].sum()) >= 2
        covered.append(cov)
        if not cov:
            m4 = m2 + 0.6 * pull
        scored = nxt and (t_next - t_r[i] <= 400)
        pat = None
        if scored:
            h = t_next - t_r[i]
            p2 = float(_forecast(t_r, m_r, i, h))
            p4 = float(p2 + k4 * (1.0 if cov else 0.6) * (f(t_next) - p2) + 0.3 * math.sin(h / 50 + caseid))
            pat = {"M2": round(p2, 3), "M4": round(p4, 3)}
            err["M2"].append(p2 - y_next)
            err["M4"].append(p4 - y_next)
            err["M3"].append(0.55 * (p2 - y_next) + 0.45 * (p4 - y_next))
            err["M4-P1"].append(p2 - y_next + rng.normal(0, 0.9))
            if y_next < thr <= m_r[i]:
                onset += 1
                flag["M2"] += int(p2 < thr)
                flag["M4"] += int(p4 < thr)
        issues.append(IssuePoint(float(t_r[i]), t_next, y_next, bool(scored), H_GRID.copy(),
                                 Curve(np.round(m2, 3), _half(H_GRID)), Curve(np.round(m4, 3), _half(H_GRID, 0.96)),
                                 m4_fallback=not cov, pred_at_target=pat))
    s = CaseSeries(caseid=caseid, window_s=(0.0, 1800.0), baseline_map=round(baseline, 3),
                   incision_s=float(rng.uniform(900, 2300)), cuff_t=t_r, cuff_map=m_r, hr_t=hr_t,
                   hr_bpm=np.round(hr, 1), pleth_start=pl_s, pleth_usable=pl_u, pump_t=pump_t, pump_ml_h=pump_v,
                   issues=issues)
    s.validate()
    rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}
    changes = 0 if not pump else int(np.sum(np.diff(pump_v[pump_t <= 1800]) != 0))
    cs = CaseSummary(caseid=caseid, status="scored", n_readings=len(t_r),
                     median_interval_s=float(np.median(np.diff(t_r))), n_forecasts=len(err["M2"]), rmse=rmse,
                     waveform_coverage=float(np.mean(covered)), new_onset_80=onset, flagged_80=flag,
                     phenylephrine_ug=phe, ephedrine_mg=eph, pump_track=pump, pump_changes=changes)
    cs.validate()
    return cs, s


def make_cases() -> tuple:
    cases, series = [], {}
    for cid in CASE_IDS:
        c, s = make_case(cid)
        cases.append(c)
        if s is not None:
            series[cid] = s
    return cases, series


# ---------------------------------------------------------------------------
# the results tree, in the pipeline's layout
# ---------------------------------------------------------------------------

def _model_summary(means: dict, n: int, rng) -> pd.DataFrame:
    rows = []
    for m, v in means.items():
        se = v * 0.33 / math.sqrt(n)
        rows.append({"model": m, "n_patients_scored": n, "rmse_mean": v, "rmse_mean_ci_lo": v - 1.96 * se,
                     "rmse_mean_ci_hi": v + 1.96 * se})
    return pd.DataFrame(rows)


def _per_patient(means: dict, caseids, rng) -> pd.DataFrame:
    base = rng.lognormal(0, 0.33, len(caseids))
    out = []
    for m, v in means.items():
        x = base * (1 + rng.normal(0, 0.05, len(caseids)))
        x = x * v / x.mean()
        out.append(pd.DataFrame({"caseid": caseids, "model": m, "n": 7, "rmse": x}))
    return pd.concat(out, ignore_index=True)


def _hyp(events: int, flagged: dict, fa: dict) -> pd.DataFrame:
    return pd.DataFrame([{"definition": "below 80% of record-start baseline", "model": m, "n_events": events,
                          "flagged": flagged[m], "false_alarms_per_30min": fa[m]} for m in ("M2", "M3", "M4", "M4-P1")])


def write_results_tree(root: Path, variant: str, cases: list | None = None, series: dict | None = None) -> dict:
    """Write `<root>/results/...` and `<root>/cases/...` for one variant; returns the two paths."""
    root = Path(root)
    rng = np.random.default_rng(20261118)
    hi, extra = headline_inputs(variant)
    sel = H.select_row(hi)
    meta = {"commit": None, "describe": None, "data": f"synthetic(site fixture {variant})"}
    res = root / "results"
    # S-dev confirmatory
    if extra.get("no_F"):
        yard = {"n_patients": 31, "F_reported": False}
    else:
        yard = {"n_patients": 31, "F_reported": True, "F": 0.31, "F_lo": 0.06, "F_hi": 0.64}
    xf = C("M4", "M2", 0.064, 0.012, 0.113, base=4.91, n=K_FAST)
    s3 = H.sentence3(K_FAST, xf, yard, not extra.get("F_placebo_fail", False))
    con = {"M4 vs M2": hi.m4_m2, "M3 vs M2": hi.m3_m2, "M4 vs M3": hi.m4_m3, "M4-P1 vs M2": hi.m4p1_m2,
           "M3-P1 vs M2": hi.m3p1_m2, "M4 vs M4-P1": hi.m4_m4p1, "M3 vs M3-P1": hi.m3_m3p1}
    w = Writer(res / "sdev", {**meta, "stage": "sdev"})
    w.json("confirmatory/summary.json", {
        "stage": "confirmatory S-dev cross-validation (SPEC §9 steps 4-5)", "N": hi.N, "waveform_coverage": 0.874,
        "contrasts": con, "decision": sel, "headline_row": sel["row"], "fast_cycling": {"K": K_FAST, "m4_vs_m2": xf},
        "question": H.question(), "answer_line": H.answer_line(hi, sel), "sentence_1": H.sentence1(hi, sel),
        "sentence_3": s3})
    dev_means = {"M00": SDEV_M2 * 1.31, "M0": SDEV_M2 * 1.12, "M1": SDEV_M2 * 1.09, "M2": SDEV_M2,
                 "M3": hi.m3_m2.mean_model, "M4": hi.m4_m2.mean_model, "M3-P1": hi.m3p1_m2.mean_model,
                 "M4-P1": hi.m4p1_m2.mean_model}
    w.csv("confirmatory/model_summary.csv", _model_summary(dev_means, hi.N, rng))
    w.csv("confirmatory/per_patient_primary.csv", _per_patient(dev_means, np.arange(1, hi.N + 1) * 3, rng))
    gb_meets = extra.get("gb_meets", sel["row"] in (2, 3, 4, 5))
    gbx = C("GB-full", "GB-cuff", 0.058, 0.021, 0.094, base=6.95)
    w.json("secondary/summary.json", {"stage": "S-dev secondary analyses (SPEC §6.6-§6.8)",
                                      "gradient_boosting": {"meets_rule_with_placebo": gb_meets,
                                                            "readme_line": H.gb_line(gb_meets, sel["row"], gbx)}})
    w.csv("secondary/hypotension.csv", _hyp(118, {"M2": 41, "M3": 44, "M4": 47, "M4-P1": 40},
                                            {"M2": 0.62, "M3": 0.66, "M4": 0.64, "M4-P1": 0.63}))
    # S-ob frozen scoring, computed from the invented cases with the pipeline's bootstrap
    if cases is None or series is None:
        cases, series = make_cases()
    sc = sorted((c for c in cases if c.scored), key=lambda c: c.caseid)
    ids = np.array([c.caseid for c in sc])
    bs = SubjectBootstrap(ids, ids)
    v = {m: np.array([c.rmse[m] for c in sc]) for m in ("M2", "M3", "M4", "M4-P1")}
    v["M0"] = v["M2"] * 1.08
    v["M00"] = v["M2"] * 1.27
    ob = {"M4 vs M2": contrast(bs, v["M4"], v["M2"], "M4", "M2"), "M3 vs M2": contrast(bs, v["M3"], v["M2"], "M3", "M2"),
          "M4 vs M3": contrast(bs, v["M4"], v["M3"], "M4", "M3"),
          "M4-P1 vs M2": contrast(bs, v["M4-P1"], v["M2"], "M4-P1", "M2")}
    cov = extra.get("sob_coverage", float(np.mean([c.waveform_coverage for c in sc])))
    x = ob["M4 vs M2"]
    w2 = Writer(res / "sob", {**meta, "stage": "frozen S-ob scoring"})
    w2.json("frozen_scoring/summary.json", {
        "stage": "frozen S-ob scoring (SPEC §9 step 7)", "n": len(sc), "n_analysable_cases": len(sc),
        "n_nmb_recorded_cases": 0, "n_analysed_cases": len(sc),
        "waveform_coverage": cov, "coverage_rule_applies": H.coverage_rule_applies(cov), "contrasts": ob,
        "sentence_2": H.sentence2(len(sc), x, cov), "cesarean_line": H.cesarean_line(len(sc), x, cov)})
    ob_means = {m: float(v[m].mean()) for m in ("M00", "M0", "M2", "M3", "M4", "M4-P1")}
    w2.csv("frozen_scoring/model_summary.csv", _model_summary(ob_means, len(sc), rng))
    w2.csv("frozen_scoring/per_patient_primary.csv", pd.concat(
        [pd.DataFrame({"caseid": ids, "model": m, "n": [c.n_forecasts for c in sc], "rmse": v[m]})
         for m in ("M00", "M0", "M2", "M3", "M4", "M4-P1")], ignore_index=True))
    ev = int(sum(c.new_onset_80 for c in sc))
    fl = {m: int(sum(c.flagged_80[m] for c in sc)) for m in ("M2", "M4")}
    fl["M3"], fl["M4-P1"] = round((fl["M2"] + fl["M4"]) / 2), fl["M2"]
    w2.csv("after_frozen_scoring/hypotension.csv", _hyp(ev, fl, {"M2": 0.81, "M3": 0.85, "M4": 0.83, "M4-P1": 0.80}))
    write_cases(root / "cases", cases, series, synthetic=True)
    return {"results": res, "cases": root / "cases"}
