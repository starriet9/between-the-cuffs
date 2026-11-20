#!/usr/bin/env python3
"""Choose the fixed ECG R-peak detector on S-dev (SPEC.md §4.3, §9 step 3).

Signal-level criteria only: the detector's heart rate is compared with the monitor heart
rate (`Solar8000/HR`). No blood-pressure value is read.

Written in this script before the comparison was first run (the order is not independently
time-stamped):
- Candidates: the neurokit2 methods in `btc.rpeaks.CANDIDATES` (neurokit, pantompkins1985,
  hamilton2002, elgendi2010), each with its own neurokit2 cleaning step, run on each 10-s
  ECG window on its own, as in the feature pipeline.
- Cases: every 5th S-dev case in ascending case ID (the 1st, 6th, 11th, ...) among the cases
  whose pleth and ECG are both present under SPEC §2.3. This uses no cuff data. [judgment]
- Windows: every 10-s window of the pleth grid (SPEC §5) whose ECG window is complete, has
  no missing sample and is not flat, and that holds at least one monitor HR value > 0.
- Detector rate: 60 (n - 1) / (last R - first R) within the window; monitor rate: the
  median of the monitor HR values in the window.
- Primary criterion: the share of windows whose detector rate is within 5 beats/min of the
  monitor rate (a window with fewer than 2 R peaks counts as a miss). Ties: lower median
  absolute difference, then the order of the candidate list.
- Also reported, descriptively: the median absolute difference, the share of windows with a
  rate, the per-case spread, the within-window SD of pulse arrival time (timing jitter), the
  number of windows on which a detector raised (no peaks), and the foot-to-R lag histogram
  that documents `btc.beats.PAT_LAG`.
- Only the `neurokit` method has a powerline filter step, set to 60 Hz (`btc.rpeaks`).

Writes `data/derived/rpeak_selection.md`, `rpeak_selection_windows.parquet` and
`rpeak_selection_log.json`. The chosen method must then be set as `btc.rpeaks.DETECTOR`.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import beats, io, pqi, provenance, rpeaks, waveform as W  # noqa: E402
from btc.paths import DERIVED  # noqa: E402

SCRIPT = "scripts/select_rpeak_detector.py"
TOL_BPM = 5.0
STEP = 5
LAG_MAX = 1.6


def selection_cases() -> list[int]:
    st = io.integrity_table().sort_values("caseid")      # SPEC §2.3, one implementation
    ok = [int(c) for c, p, e in zip(st.caseid, st.pleth_status, st.ecg_status)
          if W.present(p) and W.present(e)]
    return ok[::STEP]


def run_case(caseid: int) -> tuple[list[dict], list[np.ndarray]]:
    d = W.load_case(caseid)
    p, e = d["pleth"], d["ecg"]
    hr_t, hr_v = d["hr"] if d["hr"] is not None else (np.zeros(0), np.zeros(0))
    nw = pqi.n_windows(p.x.size)
    starts, ends = pqi.window_times(p.t0, nw, p.srate)
    rows, lags = [], []
    for w in range(nw):
        ts, te = float(starts[w]), float(ends[w])
        ew = W._ecg_window(e, ts, te)
        if ew is None:
            continue
        i0, xe = ew
        if not (np.all(np.isfinite(xe)) and xe.std() > 0):
            continue
        m = (hr_t >= ts) & (hr_t < te) & np.isfinite(hr_v) & (hr_v > 0)
        if not m.any():
            continue
        hr_mon = float(np.median(hr_v[m]))
        # pleth beats (method-independent), for the descriptive PAT measures
        b = None
        seg = p.x[w * pqi.WIN_N_RAW:(w + 1) * pqi.WIN_N_RAW]
        if pqi.assess_window(pqi.window_125(p.x, w))["pqi_pass"] and np.all(np.isfinite(seg)):
            b = beats.delineate(beats.bandpass(seg), p.srate)
            if b["foot"].size < beats.MIN_BEATS:
                b = None
        for method in rpeaks.CANDIDATES:
            t1 = time.perf_counter()
            pk, failed = rpeaks.detect_checked(xe, e.srate, method)
            dt = time.perf_counter() - t1
            rate = rpeaks.rate_bpm(pk, e.srate)
            r = {"caseid": caseid, "w": w, "method": method, "hr_monitor": hr_mon, "failed": bool(failed),
                 "n_r": int(pk.size), "rate": rate, "sec": dt, "pat_sd": np.nan, "n_pat": 0}
            if b is not None:
                rt = e.times(i0 + pk)
                pat = beats.pair_pat(b, rt, ts, p.srate)
                pat = pat[np.isfinite(pat)]
                r["n_pat"] = int(pat.size)
                if pat.size >= beats.MIN_PAT_BEATS:
                    r["pat_sd"] = float(pat.std(ddof=1))
                if method == rpeaks.DETECTOR:
                    ft = ts + b["foot"] / p.srate
                    for f in ft:
                        lg = f - rt
                        lags.append(lg[(lg > 0) & (lg < LAG_MAX)])
            rows.append(r)
    return rows, lags


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=12)
    provenance.add_allow_dirty(ap)
    args = ap.parse_args(argv)
    state = provenance.require_clean(args.allow_dirty)
    t0 = time.time()
    cases = selection_cases()
    with Pool(args.jobs) as pool:
        res = pool.map(run_case, cases)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    lags = np.concatenate([l for _, ls in res for l in ls]) if any(ls for _, ls in res) else np.zeros(0)
    meta = {"git": state, "spec": provenance.SPEC_VERSION, "cohort": "S-dev", "script": SCRIPT}
    provenance.write_parquet(df, DERIVED / "rpeak_selection_windows.parquet", meta)

    df["absdiff"] = (df["rate"] - df["hr_monitor"]).abs()
    df["within"] = df["absdiff"] <= TOL_BPM          # NaN rate -> False (counts as a miss)
    summ = []
    for m in rpeaks.CANDIDATES:
        g = df[df.method == m]
        per_case = g.groupby("caseid")["within"].mean()
        summ.append({
            "method": m, "windows": len(g), "cases": g.caseid.nunique(),
            "share_within_5": g["within"].mean(),
            "median_absdiff": g["absdiff"].median(),
            "share_with_rate": g["rate"].notna().mean(),
            "case_within_5_p10": per_case.quantile(0.10),
            "case_within_5_median": per_case.median(),
            "pat_sd_median_ms": 1000 * g["pat_sd"].median(),
            "ms_per_window": 1000 * g["sec"].mean(),
            "errors": int(g["failed"].sum()),
        })
    s = pd.DataFrame(summ)
    order = {m: i for i, m in enumerate(rpeaks.CANDIDATES)}
    s["_o"] = s["method"].map(order)
    best = s.sort_values(["share_within_5", "median_absdiff", "_o"],
                         ascending=[False, True, True]).iloc[0]["method"]
    s = s.drop(columns="_o")

    h, edges = np.histogram(lags, bins=np.arange(0, LAG_MAX + 1e-9, 0.05))
    lines = [
        "# R-peak detector selection (S-dev, signal-level only)",
        "",
        "Rule written in `scripts/select_rpeak_detector.py` before the comparison was first run (SPEC §4.3, §9; "
        "the order is not independently time-stamped). Only `neurokit` has a powerline filter step (60 Hz):",
        f"every {STEP}th S-dev case by case ID among cases with pleth and ECG present under SPEC §2.3; "
        "every 10-s window with a complete, non-flat ECG window and a monitor HR value; "
        f"primary criterion: share of windows with the detector rate within {TOL_BPM:g} beats/min "
        "of the median monitor HR (`Solar8000/HR`) in the window (fewer than 2 R peaks = miss); "
        "ties: lower median absolute difference, then candidate order. No blood-pressure value is used.",
        "",
        f"Cases: {len(cases)} ({', '.join(map(str, cases))}).",
        "",
        "| method | windows | share within 5 bpm | median abs diff (bpm) | share with a rate | "
        "per-case share within 5 bpm, 10th pct | per-case median | within-window SD of PAT, median (ms) | "
        "ms per window | detector errors (windows) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in s.iterrows():
        lines.append(f"| {r.method} | {r.windows} | {r.share_within_5:.4f} | {r.median_absdiff:.2f} | "
                     f"{r.share_with_rate:.4f} | {r.case_within_5_p10:.3f} | {r.case_within_5_median:.3f} | "
                     f"{r.pat_sd_median_ms:.1f} | {r.ms_per_window:.1f} | {r.errors} |")
    lines += [
        "",
        f"**Chosen detector: `{best}`** (set as `btc.rpeaks.DETECTOR`).",
        "",
        "## Foot-to-R lag survey (documents `btc.beats.PAT_LAG`)",
        "",
        f"Lags from each pleth beat foot (windows that pass the primary quality index, >= 4 beats) "
        f"to every R peak of detector `{rpeaks.DETECTOR}` in the same window that precedes the foot "
        f"by less than {LAG_MAX:g} s. The pairing range used for pulse arrival time is "
        f"[{beats.PAT_LAG[0]:.2f}, {beats.PAT_LAG[1]:.2f}) s.",
        "",
        "| lag from (s) | count |",
        "|---|---|",
    ]
    lines += [f"| {a:.2f} | {n} |" for a, n in zip(edges[:-1], h)]
    lines += ["", f"Run time: {time.time() - t0:.0f} s.", ""]
    (DERIVED / "rpeak_selection.md").write_text(provenance.with_credit("\n".join(lines)))
    provenance.write_json(DERIVED / "rpeak_selection_log.json", {
        "script": SCRIPT, "git": state, "cases": cases, "chosen": best, "current_detector": rpeaks.DETECTOR,
        "summary": s.to_dict(orient="records"), "lag_histogram": {f"{a:.2f}": int(n) for a, n in zip(edges[:-1], h)},
        "run_seconds": round(time.time() - t0, 1)})
    print("\n".join(lines[:16 + len(s)]))
    if best != rpeaks.DETECTOR:
        print(f"\nNOTE: set btc.rpeaks.DETECTOR = {best!r} (currently {rpeaks.DETECTOR!r}) and rerun "
              "so that the lag survey uses the chosen detector.")


if __name__ == "__main__":
    main()
