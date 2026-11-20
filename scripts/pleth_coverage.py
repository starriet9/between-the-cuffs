#!/usr/bin/env python3
"""Pleth coverage and signal-quality summary of one cohort (SPEC.md §5 coverage, §6.6 item 5, §6.7).

Signal-level summaries only: coverage of forecasts by usable windows, usable-window shares
overall and by time relative to each reading's display time (-60 to +60 s), per-criterion
failure shares, agreement of the R-peak rate and the pleth pulse rate with the monitor HR,
the distribution of pulse arrival time, and the notch-detector check behind SPEC §4.3 (the
notch quantities are data quality only). No blood-pressure value is read.

Scope: analysable cases (`<cohort>_cases.parquet`). Reading times come from
`<cohort>_readings.parquet` (times and validity only). S-ob (`--cohort sob`) runs only after
the frozen scoring, with BTC_FROZEN_SCORING=1 (SPEC §9).

Writes `data/derived/<cohort>_pleth_coverage.md` and `data/derived/<cohort>_pleth_coverage.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import beats, io, paths, provenance, rpeaks  # noqa: E402
from btc.constants import PRIMARY_END_S  # noqa: E402

SCRIPT = "scripts/pleth_coverage.py"
REL_BINS = np.arange(-60, 61, 10)   # window midpoint minus reading time, s [judgment]
TOL_BPM = 5.0
STRATUM_CUT = 0.80          # SPEC §6.6 item 5
NOTCH_EARLY_S = (0.08, 0.12)        # SPEC §4.3: the detector's early point after the systolic peak
NOTCH_TROUGH_S = 0.14               # trough between the two modes of the window notch time [judgment]
RATE_BANDS = (40, 60, 70, 80, 90, 100, 180)   # pleth pulse rate bands, beats/min [judgment]


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f}%"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=paths.COHORTS_KEYS, default="sdev")
    ap.add_argument("--in-dir", type=Path, default=paths.DERIVED)
    ap.add_argument("--out", type=Path, default=None, help="default: <in-dir>/<cohort>_pleth_coverage.md")
    provenance.add_allow_dirty(ap)
    args = ap.parse_args(argv)
    io.check_cohort_access(args.cohort)
    state = provenance.require_clean(args.allow_dirty)
    co, label = args.cohort, paths.COHORT_LABEL[args.cohort]
    out_md = args.out or args.in_dir / f"{co}_pleth_coverage.md"

    cases = pd.read_parquet(args.in_dir / f"{co}_cases.parquet", columns=["caseid", "analysable"])
    an = set(cases.loc[cases.analysable, "caseid"].astype(int))
    rd = pd.read_parquet(args.in_dir / f"{co}_readings.parquet", columns=["caseid", "t", "valid"])
    rd = rd[rd.valid & rd.caseid.isin(an)]
    win_all = pd.read_parquet(args.in_dir / f"{co}_windows.parquet")
    win = win_all[win_all.caseid.isin(an)].copy()
    win["mid"] = 0.5 * (win.t_start + win.t_end)
    feat = pd.read_parquet(args.in_dir / f"{co}_features_wf.parquet")
    res: dict = {"script": SCRIPT, "git": state, "cohort": label, "analysable_cases": len(an),
                 "cases_with_windows": int(win.caseid.nunique()),
                 "rpeak_errors_windows": int(win.r_failed.sum()) if "r_failed" in win else None}
    L: list[str] = [f"# {label} pleth coverage and signal quality", "",
                    "Signal-level summaries only (SPEC §5, §6.6, §6.7); no blood-pressure value "
                    "was read. Analysable cases: "
                    f"{len(an)} ({res['cases_with_windows']} with pleth windows; R-peak detector "
                    f"raised on {res['rpeak_errors_windows']} windows, which then have no R peaks). "
                    f"R-peak detector: `{rpeaks.DETECTOR}`; PAT pairing range "
                    f"[{beats.PAT_LAG[0]:.2f}, {beats.PAT_LAG[1]:.2f}) s.", ""]

    # 1. forecast coverage
    L += ["## 1. Forecasts with >= 2 usable windows (SPEC §5 coverage, §4.3 fallback)", "",
          "Usable = primary index passed and >= 4 beats; strict adds template correlation "
          ">= 0.85. Windows lie between the previous reading (at most 120 s back) and the "
          "cutoff.", "",
          "| window | variant | forecasts | cases | with >= 2 usable windows | cases with all forecasts covered |",
          "|---|---|---|---|---|---|"]
    res["forecast_coverage"] = {}
    for (w, v), g in feat.groupby(["window", "variant"], sort=True):
        allc = g.groupby("caseid").wf_available.all().mean()
        res["forecast_coverage"][f"{w}/{v}"] = {"forecasts": len(g), "share": float(g.wf_available.mean()),
                                                "cases_all_covered": float(allc)}
        L.append(f"| {w} | {v} | {len(g)} | {g.caseid.nunique()} | {pct(g.wf_available.mean())} | {pct(allc)} |")
    real = feat[(feat.window == "primary") & (feat.variant == "real")]
    per_case = real.groupby("caseid").wf_available.mean()
    L += ["", f"Primary/real, per case: median {pct(per_case.median())}, 10th percentile "
          f"{pct(per_case.quantile(0.1))}, cases with no covered forecast: {int((per_case == 0).sum())}.", ""]

    # quality strata (SPEC §6.6 item 5)
    s = real.wf_usable_share
    strata = {"fewer than 2 usable": int((~real.wf_available).sum()),
              "2+ usable, < 80% usable": int((real.wf_available & (s < STRATUM_CUT)).sum()),
              "2+ usable, >= 80% usable": int((real.wf_available & (s >= STRATUM_CUT)).sum())}
    res["quality_strata_primary_real"] = strata
    L += ["Quality strata (SPEC §6.6 item 5), primary window, real signals: " +
          "; ".join(f"{k}: {v}" for k, v in strata.items()) + ".", ""]

    # 2. usable-window shares
    L += ["## 2. Usable-window share", "",
          "| windows | n | passed primary index | usable (index + >= 4 beats) | strict |",
          "|---|---|---|---|---|"]
    res["window_share"] = {}
    for name, g in (("whole record", win), ("start in 0-30 min", win[win.t_start < PRIMARY_END_S]),
                    ("ending by 30 min", win[win.t_end <= PRIMARY_END_S])):
        res["window_share"][name] = {"n": len(g), "pqi": float(g.pqi_pass.mean()),
                                     "usable": float(g.usable.mean()), "strict": float(g.usable_strict.mean())}
        L.append(f"| {name} | {len(g)} | {pct(g.pqi_pass.mean())} | {pct(g.usable.mean())} | "
                 f"{pct(g.usable_strict.mean())} |")
    pc = win[win.t_end <= PRIMARY_END_S].groupby("caseid").usable.mean()
    L += ["", f"Per case (windows ending by 30 min): median usable share {pct(pc.median())}, "
          f"10th percentile {pct(pc.quantile(0.1))}, minimum {pct(pc.min())}.", ""]

    # 3. per-criterion failures
    L += ["## 3. Failure shares by criterion (windows ending by 30 min)", "",
          "A window can fail several criteria; 'only' counts windows failing that criterion alone. "
          "The beat and template rows are shares of the windows that reached that step.", "",
          "| criterion | fails | fails only this |", "|---|---|---|"]
    g = win[win.t_end <= PRIMARY_END_S]
    crit = ["fail_missing", "fail_flat", "fail_freq", "fail_peak", "fail_amp", "fail_const"]
    labels = {"fail_missing": "missing samples (discarded)", "fail_flat": "flat (discarded)",
              "fail_freq": "1. dominant frequency outside 0.67-3 Hz",
              "fail_peak": "2. spectral peak < 15% of in-band power", "fail_amp": "3. max |z| > 6",
              "fail_const": "4. constant runs >= 10%"}
    nfail = g[crit].sum(axis=1)
    res["criterion_failures"] = {}
    for c in crit:
        a, o = float(g[c].mean()), float((g[c] & (nfail == 1)).mean())
        res["criterion_failures"][c] = {"fails": a, "only": o}
        L.append(f"| {labels[c]} | {pct(a)} | {pct(o)} |")
    pq = g[g.pqi_pass]
    us = g[g.usable]
    res["criterion_failures"]["fewer_than_4_beats_of_pqi_pass"] = float((pq.n_beats < beats.MIN_BEATS).mean())
    res["criterion_failures"]["template_below_085_of_usable"] = float((~us.usable_strict).mean())
    L += [f"| added: < 4 beats (of windows passing the index) | {pct(res['criterion_failures']['fewer_than_4_beats_of_pqi_pass'])} | |",
          f"| strict: template correlation < 0.85 (of usable windows) | {pct(res['criterion_failures']['template_below_085_of_usable'])} | |",
          ""]

    # 4. usable share around reading times
    L += ["## 4. Usable-window share around reading display times (SPEC §5, §6.6)", "",
          "Each window is placed by its midpoint relative to each valid reading's display time "
          "t_r in the primary window, in 10-s bins; a window can count for two nearby readings. "
          "A window ends by t_r if its midpoint is at least 5 s before t_r; the cuff inflation "
          "precedes t_r.", "",
          "| midpoint - t_r (s) | windows | passed index | usable | strict |", "|---|---|---|---|---|"]
    rel = []
    wg = {c: d for c, d in win.groupby("caseid")}
    for c, r in rd[rd.t <= PRIMARY_END_S].groupby("caseid"):
        if c not in wg:
            continue
        d = wg[c]
        m = d.mid.to_numpy()
        for t in r.t.to_numpy():
            k = (m >= t + REL_BINS[0]) & (m < t + REL_BINS[-1])
            if k.any():
                rel.append(pd.DataFrame({"rel": m[k] - t, "pqi": d.pqi_pass.to_numpy()[k],
                                         "usable": d.usable.to_numpy()[k],
                                         "strict": d.usable_strict.to_numpy()[k]}))
    rel = pd.concat(rel, ignore_index=True)
    rel["bin"] = pd.cut(rel.rel, REL_BINS, right=False)
    res["around_readings"] = {}
    for b, gb in rel.groupby("bin", observed=True):
        res["around_readings"][str(b)] = {"n": len(gb), "pqi": float(gb.pqi.mean()),
                                          "usable": float(gb.usable.mean()), "strict": float(gb.strict.mean())}
        L.append(f"| [{int(b.left)}, {int(b.right)}) | {len(gb)} | {pct(gb.pqi.mean())} | "
                 f"{pct(gb.usable.mean())} | {pct(gb.strict.mean())} |")
    L.append("")

    # 5. agreement with the monitor HR
    L += ["## 5. Agreement with the monitor heart rate (`Solar8000/HR`)", "",
          "Rate within each 10-s window against the median monitor HR (values > 0) in the same "
          "window. Analysable cases, whole record.", "",
          "| rate | windows | within 5 bpm | median abs diff (bpm) | per-case within 5 bpm, median / 10th pct |",
          "|---|---|---|---|---|"]
    res["hr_agreement"] = {}
    for name, col, sel in (("R peaks, complete ECG windows", "r_rate", win.ecg_ok),
                           ("R peaks, usable pleth windows", "r_rate", win.ecg_ok & win.usable),
                           ("pleth pulse, usable windows", "pleth_rate", win.usable),
                           ("pleth pulse, strict windows", "pleth_rate", win.usable_strict)):
        g = win[sel & win.hr_monitor.notna()]
        ad = (g[col] - g.hr_monitor).abs()
        within = ad <= TOL_BPM
        pcw = within.groupby(g.caseid).mean()
        res["hr_agreement"][name] = {"n": len(g), "within5": float(within.mean()),
                                     "median_absdiff": float(ad.median()),
                                     "case_median": float(pcw.median()), "case_p10": float(pcw.quantile(0.1))}
        L.append(f"| {name} | {len(g)} | {pct(within.mean())} | {ad.median():.2f} | "
                 f"{pct(pcw.median())} / {pct(pcw.quantile(0.1))} |")
    L.append("")

    # 6. pulse arrival time and per-window features
    u = win[win.usable]
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    L += ["## 6. Per-window feature distributions (usable windows; SPEC §6.7 data quality)", "",
          "Times in s. Pulse arrival time includes a device delay (see `rpeak_selection.md`) and "
          "enters the models only as within-case differences.", "",
          "| feature | windows with a value | " + " | ".join(f"{int(100 * x)}th" for x in q) + " |",
          "|---|---|" + "---|" * len(q)]
    res["window_features"] = {}
    for f in beats.WINDOW_FEATURES:
        v = u[f].dropna()
        qq = v.quantile(q)
        res["window_features"][f] = {"share_with_value": float(u[f].notna().mean()),
                                     **{str(k): float(x) for k, x in qq.items()}}
        name = f"{f} (data quality only)" if f in beats.QC_FEATURES else f
        L.append(f"| {name} | {pct(u[f].notna().mean())} | " + " | ".join(f"{x:.3f}" for x in qq) + " |")
    pc_pat = u.groupby("caseid").pat.median().dropna()
    res["pat_case_median"] = {"cases": int(len(pc_pat)), **{str(k): float(x) for k, x in
                              pc_pat.quantile([0.0, 0.05, 0.5, 0.95, 1.0]).items()}}
    L += ["", f"Pulse arrival time, median per case ({len(pc_pat)} cases): median {pc_pat.median():.3f} s, "
          f"5th-95th percentile {pc_pat.quantile(0.05):.3f}-{pc_pat.quantile(0.95):.3f} s, "
          f"range {pc_pat.min():.3f}-{pc_pat.max():.3f} s. Windows with a PAT: {pct(u.pat.notna().mean())}.", ""]

    # 6b. notch detector check (SPEC §4.3: notch quantities are not model inputs)
    L += ["## 6b. Notch detector check (SPEC §4.3; signal level only)", "",
          f"Usable windows by where the detected point lies after the systolic peak. 'Early' = window "
          f"median notch time in [{NOTCH_EARLY_S[0]:.2f}, {NOTCH_EARLY_S[1]:.2f}] s. The notch time "
          "is listed by pleth pulse-rate band; a dicrotic notch comes earlier at higher rates.", "",
          "| pulse rate (beats/min) | windows with a notch value | early share | median notch time, early (s) | "
          "median notch time, other (s) |", "|---|---|---|---|---|"]
    nv = u[u.notch_t.notna()].copy()
    early = nv.notch_t.between(*NOTCH_EARLY_S)
    res["notch_check"] = {"usable_windows": int(len(u)), "with_value": int(len(nv)),
                          "early_share_of_usable": float(early.sum() / len(u)) if len(u) else float("nan"),
                          "early_share_of_with_value": float(early.mean()) if len(nv) else float("nan"),
                          "by_rate": {}}
    nv["band"] = pd.cut(nv.pleth_rate, RATE_BANDS, right=False)
    for b_, g in nv.groupby("band", observed=True):
        e = g.notch_t.between(*NOTCH_EARLY_S)
        res["notch_check"]["by_rate"][str(b_)] = {"n": int(len(g)), "early_share": float(e.mean()),
                                                  "median_early": float(g.notch_t[e].median()),
                                                  "median_other": float(g.notch_t[~e].median())}
        L.append(f"| [{int(b_.left)}, {int(b_.right)}) | {len(g)} | {pct(e.mean())} | "
                 f"{g.notch_t[e].median():.3f} | {g.notch_t[~e].median():.3f} |")
    per_case = early.groupby(nv.caseid).mean()
    mixed = int(((per_case >= 0.1) & (per_case <= 0.9)).sum())
    both = int(((per_case > 0) & (per_case < 1)).sum())
    below = float((u.notch_t < NOTCH_TROUGH_S).sum() / len(u)) if len(u) else float("nan")
    ua = win_all[win_all.usable]
    res["notch_check"].update({
        "cases": int(len(per_case)), "cases_both_kinds": both, "cases_mixed_10_90": mixed,
        "below_trough_share_of_usable": below,
        "all_cases_with_windows": {"cases": int(ua.caseid.nunique()), "usable_windows": int(len(ua)),
                                   "early_share": float(ua.notch_t.between(*NOTCH_EARLY_S).mean()),
                                   "below_trough_share": float((ua.notch_t < NOTCH_TROUGH_S).mean())}})
    aa = res["notch_check"]["all_cases_with_windows"]
    L += ["", f"Early windows: {pct(res['notch_check']['early_share_of_usable'])} of usable windows "
          f"({pct(res['notch_check']['early_share_of_with_value'])} of those with a notch value); below "
          f"{NOTCH_TROUGH_S:.2f} s, the trough between the two modes: {pct(below)}. Cases with windows of both "
          f"kinds: {both} of {len(per_case)}; with 10-90% early windows: {mixed}. All {aa['cases']} cases with "
          f"windows (analysable or not; {aa['usable_windows']} usable windows): early {pct(aa['early_share'])}, "
          f"below {NOTCH_TROUGH_S:.2f} s {pct(aa['below_trough_share'])}.", ""]

    # 7. missing share of each feature group per variant
    L += ["## 7. Missing share of each waveform feature (primary window)", "",
          "Level form (i) / change form (ii), over all forecasts (including fallbacks).", "",
          "| variant | " + " | ".join(beats.FEATURES) + " |", "|---|" + "---|" * len(beats.FEATURES)]
    res["missing_primary"] = {}
    for v, g in feat[feat.window == "primary"].groupby("variant"):
        cells = []
        res["missing_primary"][v] = {}
        for f in beats.FEATURES:
            a, b = g[f"wf_{f}_lvl"].isna().mean(), g[f"wf_{f}_chg"].isna().mean()
            res["missing_primary"][v][f] = [float(a), float(b)]
            cells.append(f"{pct(a)} / {pct(b)}")
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L.append("")
    out_md.write_text(provenance.with_credit("\n".join(L)))
    provenance.write_json(out_md.with_suffix(".json"), res)
    print("\n".join(L))


if __name__ == "__main__":
    main()
