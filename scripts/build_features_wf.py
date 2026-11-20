#!/usr/bin/env python3
"""Build the per-forecast waveform features of one cohort (SPEC.md §4.3, §3, §5, §6.6).

One row per (caseid, window, issue_n, variant) for every issue point in
`<cohort>_instances.parquet`, variants real, p1, p2, ia and strict (btc/SCHEMA.md §4). The
features are computed from the window table (`scripts/build_windows.py`) with
`btc.waveform.forecast_features`, from reading times only: the issue time, the previous
valid reading and the valid reading before it (`<cohort>_readings.parquet`). P1 uses the
donor in `<cohort>_cases.parquet`. No cuff value, target or forecast error is read. The notch
quantities of the window table are not carried into this table (SPEC §4.3).

S-ob (`--cohort sob`) runs only at the frozen scoring, with BTC_FROZEN_SCORING=1 (SPEC §9).
Writes `data/derived/<cohort>_features_wf.parquet` and `..._log.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import beats, io, paths, provenance, rpeaks, waveform as W  # noqa: E402

SCRIPT = "scripts/build_features_wf.py"
INPUTS = ("instances", "readings", "cases")

OUT_COLS = (["caseid", "window", "issue_n", "variant", "signal_caseid", "t_issue", "t_prev",
             "time_shift_s", "t_end", "t_lo", "wf_n_windows", "wf_n_usable", "wf_usable_share",
             "wf_available"] + W.FEATURE_COLS)


def load_inputs(in_dir: Path, windows_path: Path, cohort: str = "sdev"):
    inst = pd.read_parquet(in_dir / f"{cohort}_instances.parquet",
                           columns=["caseid", "window", "issue_n", "t_issue", "t_prev"])
    issues = inst.drop_duplicates(["caseid", "window", "issue_n"]).sort_values(
        ["caseid", "window", "issue_n"]).reset_index(drop=True)
    rd = pd.read_parquet(in_dir / f"{cohort}_readings.parquet", columns=["caseid", "n", "t", "valid"])
    rd = rd[rd["valid"]].dropna(subset=["n"])
    t_of = {(int(c), int(n)): float(t) for c, n, t in zip(rd.caseid, rd.n, rd.t)}
    cases = pd.read_parquet(in_dir / f"{cohort}_cases.parquet")
    win = pd.read_parquet(windows_path)
    return issues, t_of, cases, win


def check_waveform_status(cases: pd.DataFrame, windows_path: Path, win: pd.DataFrame) -> None:
    """Guard against stale tables: the window table and `<cohort>_cases` must carry the same
    SPEC §2.3 status (both come from `btc.io.waveform_status`), and a case whose pleth is
    absent must have no windows."""
    log = json.loads((windows_path.parent / (windows_path.stem + "_log.json")).read_text())
    st = {int(c["caseid"]): (c["pleth"], c["ecg"]) for c in log["cases"]}
    with_windows = set(win.caseid.astype(int))
    for r in cases.itertuples(index=False):
        c = int(r.caseid)
        if st.get(c) != (r.pleth_status, r.ecg_status):
            raise ValueError(f"case {c}: waveform status differs between tables ({st.get(c)})")
        if not W.present(r.pleth_status) and c in with_windows:
            raise ValueError(f"case {c}: pleth absent but windows present")


def build(issues, t_of, cases, win) -> pd.DataFrame:
    cw = {int(c): W.CaseWindows(g) for c, g in win.groupby("caseid")}
    empty = W.CaseWindows(None)
    donor = dict(zip(cases.caseid.astype(int), cases.donor_caseid))
    rows = []
    for r in issues.itertuples(index=False):
        c, n = int(r.caseid), int(r.issue_n)
        tp_check = t_of.get((c, n - 1))
        if tp_check is None or abs(tp_check - r.t_prev) > 1e-6:
            raise ValueError(f"case {c} issue {n}: t_prev does not match reading {n - 1}")
        t_pp = t_of[(c, n - 2)]
        for v in W.VARIANTS:
            src = c
            if v == "p1":
                d = donor.get(c, np.nan)
                src = int(d) if pd.notna(d) else None
            ti, tp, tpp, te, mask = W.variant_times(v, float(r.t_issue), float(r.t_prev), t_pp)
            f = W.forecast_features(cw.get(src, empty) if src is not None else empty,
                                    ti, tp, tpp, te, mask)
            rows.append({"caseid": c, "window": r.window, "issue_n": n, "variant": v,
                         "signal_caseid": float(src) if src is not None else np.nan,
                         "t_issue": float(r.t_issue), "t_prev": float(r.t_prev),
                         "time_shift_s": ti - float(r.t_issue), "t_end": te, **f})
    out = pd.DataFrame(rows).reindex(columns=OUT_COLS)
    out["wf_available"] = out["wf_available"].astype(bool)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=paths.COHORTS_KEYS, default="sdev")
    ap.add_argument("--in-dir", type=Path, default=paths.DERIVED,
                    help="folder with <cohort>_instances / _readings / _cases")
    ap.add_argument("--windows", type=Path, default=None, help="default: <in-dir>/<cohort>_windows.parquet")
    ap.add_argument("--out", type=Path, default=None, help="default: <in-dir>/<cohort>_features_wf.parquet")
    provenance.add_allow_dirty(ap)
    args = ap.parse_args(argv)
    io.check_cohort_access(args.cohort)
    state = provenance.require_clean(args.allow_dirty)
    windows = args.windows or args.in_dir / f"{args.cohort}_windows.parquet"
    out_path = args.out or args.in_dir / f"{args.cohort}_features_wf.parquet"
    t0 = time.time()
    issues, t_of, cases, win = load_inputs(args.in_dir, windows, args.cohort)
    check_waveform_status(cases, windows, win)
    out = build(issues, t_of, cases, win)
    inputs = {p.name: provenance.read_meta(p).get("git", {}).get("commit")
              for p in [args.in_dir / f"{args.cohort}_{n}.parquet" for n in INPUTS] + [windows]}
    meta = {"git": state, "spec": provenance.SPEC_VERSION, "cohort": paths.COHORT_LABEL[args.cohort],
            "script": SCRIPT, "rpeak_detector": rpeaks.DETECTOR, "inputs_commit": inputs}
    provenance.write_parquet(out, out_path, meta)
    summ = {}
    for (w, v), g in out.groupby(["window", "variant"]):
        summ[f"{w}/{v}"] = {
            "forecasts": int(len(g)), "cases": int(g.caseid.nunique()),
            "share_available": round(float(g.wf_available.mean()), 4),
            "missing_share_lvl": {f: round(float(g[f"wf_{f}_lvl"].isna().mean()), 4)
                                  for f in beats.FEATURES},
            "missing_share_chg": {f: round(float(g[f"wf_{f}_chg"].isna().mean()), 4)
                                  for f in beats.FEATURES},
        }
    log = {"script": SCRIPT, "git": state, "cohort": paths.COHORT_LABEL[args.cohort],
           "rpeak_detector": rpeaks.DETECTOR, "inputs_commit": inputs,
           "n_rows": int(len(out)), "variants": list(W.VARIANTS), "features": list(beats.FEATURES),
           "summary": summ, "run_seconds": round(time.time() - t0, 1)}
    provenance.write_json(out_path.parent / (out_path.stem + "_log.json"), log)
    print(json.dumps({k: (v if k != "summary" else {kk: vv["share_available"] for kk, vv in v.items()})
                      for k, v in log.items() if k != "git"}, indent=1))


if __name__ == "__main__":
    main()
