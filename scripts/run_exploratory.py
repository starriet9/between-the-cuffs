#!/usr/bin/env python3
"""Exploratory lead-lag and detection-limit analyses on S-dev (EXPLORATORY_PLAN.md). Both are exploratory,
computed after the registered results were known; they change no registered outcome, rule or sentence.
The lead-lag analysis is an item of SPEC.md §6.8; the detection limit is new (post hoc exploratory).

Commands (S-dev only):
  lead-lag         plan §1: within-case correlations of feature changes with cuff MAP changes (concurrent,
                   next, next-reading M2 residual), real and P1 donor features, subject-bootstrap intervals.
                   Writes results/sdev/exploratory_lead_lag/ (summary.json, summary.md, lead_lag.csv,
                   change_identity.csv, MANIFEST.json).
  detection-limit  plan §2: a planted column of known strength in the waveform block of the real design,
                   the registered chain M2 -> M3 -> M4 refitted by the unchanged nested cross-validation at
                   each strength, noise seed 20261118 and ten replicate seeds. Writes
                   results/sdev/exploratory_detection_limit/ (summary.json, summary.md, detection_limit.csv,
                   replicates.csv, replicate_summary.csv, penalties.csv, MANIFEST.json).

Both write aggregate tables only: no per-forecast or per-case value, and never a planted value.

Guards (each command refuses otherwise): a clean work tree (untracked files count); EXPLORATORY_PLAN.md
committed at HEAD (its SHA-256 and last commit are recorded in every result); BTC_FROZEN_SCORING unset; the
S-dev derived tables equal the inputs of the confirmatory run (inputs_sha256); the fold plan equals the
confirmatory folds.csv; the command's output folder does not exist ("the first complete run is the result",
plan §3). Every attempt goes to a git-ignored log, data/derived/sdev_exploratory_run_log.jsonl. Nothing is
written before the computation has finished. No network is needed (run it with the network off).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc.models import exploratory as X  # noqa: E402
from btc.models.constants import N_BOOT, SEED  # noqa: E402
from btc.models.data import load_tables  # noqa: E402
from btc.models.folds import make_fold_plan  # noqa: E402
from btc.models.outputs import Writer, git_meta  # noqa: E402
from btc.models.pipeline import software_versions  # noqa: E402
from btc.paths import DERIVED, ROOT  # noqa: E402

RESULTS = ROOT / "results"
CONF = RESULTS / "sdev" / "confirmatory"
OUT_REL = {"lead-lag": "sdev/exploratory_lead_lag", "detection-limit": "sdev/exploratory_detection_limit"}
RUN_LOG = DERIVED / "sdev_exploratory_run_log.jsonl"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _log(entry: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry}
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def folds_equal(T) -> bool:
    p = CONF / "folds.csv"
    if not p.exists():
        return False
    folds = pd.read_csv(p, skiprows=1)
    pt = make_fold_plan(T.analysed, SEED).table()
    return len(folds) == len(pt) and all(c in folds.columns and np.array_equal(folds[c].to_numpy(), pt[c].to_numpy())
                                         for c in pt.columns)


def _f(x, nd=3) -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}".replace("-", "−")


def _p(x, nd=2) -> str:
    return _f(100.0 * x, nd) + "%"


def lead_lag_md(t: pd.DataFrame, s: dict) -> str:
    out = ["# Exploratory lead–lag, S-dev (EXPLORATORY_PLAN.md §1)", "",
           f"**Label.** {X.LABEL}. SPEC.md §6.8 lists this item; its details were fixed in the plan.", "",
           "Within-case (case-centred, pooled) Pearson correlation of the feature change at reading r with the "
           "cuff MAP change, on the primary-window next-reading forecasts of the analysed cases; a case enters "
           f"with at least {s['min_pairs_per_case']} usable pairs. 95% percentile intervals from the subject "
           f"bootstrap of the primary outcome ({s['n_bootstrap_resamples']} resamples, seed {s['bootstrap_seed']}, "
           f"{s['n_bootstrap_patients']} patients). Concurrent: MAP_r − MAP_(r−1). Next: MAP_(r+1) − MAP_r. "
           "Next residual: MAP_(r+1) minus the cross-validated M2 forecast. P1: another patient's signals at this "
           "patient's reading times.", "", f"Expectation (not a claim): {s['expectation']}.", "",
           "| Feature | MAP change | Real r (95% CI) | Cases / pairs | P1 r (95% CI) | Cases / pairs | Real − P1 (95% CI) |",
           "|---|---|---|---|---|---|---|"]
    for feat in t["feature"].unique():
        for mc in X.MAP_CHANGES:
            g = t.loc[(t["feature"] == feat) & (t["map_change"] == mc)].set_index("variant")
            cell = lambda v: f"{_f(g.loc[v, 'r'])} ({_f(g.loc[v, 'r_ci_lo'])} to {_f(g.loc[v, 'r_ci_hi'])})"
            n = lambda v: f"{int(g.loc[v, 'n_cases'])} / {int(g.loc[v, 'n_pairs'])}"
            out.append(f"| {feat} | {mc} | {cell('real')} | {n('real')} | {cell('p1')} | {n('p1')} | "
                       f"{cell('real minus p1')} |")
    out += ["", "Thirty correlations and fifteen differences; unadjusted 95% intervals, descriptive. A correlation "
            "is not an increment over the cuff.", ""]
    return "\n".join(out)


def detection_md(s: dict, main: pd.DataFrame, rsum: pd.DataFrame) -> str:
    out = ["# Exploratory detection limit, S-dev (EXPLORATORY_PLAN.md §2)", "",
           f"**Label.** {X.LABEL}. New analysis (post hoc exploratory). It tests {s['tests']}.", "",
           f"Planted column: {s['planted_column']}. Oracle: {s['oracle']}. Rule: {s['rule']}. "
           f"{s['N']} patients, {s['n_forecasts']} forecasts; noise seed {s['noise_seed_main']}.", "",
           "| Nominal q | Oracle reduction (95% CI) | M4+planted vs M2: relative reduction (95% CI) | Δ, mmHg (95% CI) "
           "| CI of Δ below 0 | Meets rule | Waveform penalty, folds 0–4 |", "|---|---|---|---|---|---|---|"]
    for r in main.to_dict("records"):
        pen = ", ".join(v if v == "dropped" else f"{v:.3g}" for v in (r[f"wf_penalty_fold{j}"] for j in range(5)))
        out.append(f"| {_p(r['strength_nominal'], 0)} | {_p(r['oracle_rr'])} ({_p(r['oracle_rr_ci_lo'])} to "
                   f"{_p(r['oracle_rr_ci_hi'])}) | {_p(r['rr'])} ({_p(r['rr_ci_lo'])} to {_p(r['rr_ci_hi'])}) | "
                   f"{_f(r['delta'])} ({_f(r['delta_ci_lo'])} to {_f(r['delta_ci_hi'])}) | "
                   f"{'yes' if r['delta_ci_below_zero'] else 'no'} | {'yes' if r['meets_rule'] else 'no'} | {pen} |")
    if len(rsum):
        out += ["", f"Ten further noise draws (seeds {s['noise_seeds_replicates'][0]}–{s['noise_seeds_replicates'][-1]}):", "",
                "| Nominal q | Draws meeting the rule | Draws with CI of Δ below 0 | Relative reduction, min / median / max |",
                "|---|---|---|---|"]
        for r in rsum.to_dict("records"):
            out.append(f"| {_p(r['strength_nominal'], 0)} | {r['n_meets_rule']} of {r['n_draws']} | "
                       f"{r['n_delta_ci_below_zero']} of {r['n_draws']} | {_p(r['rr_min'])} / {_p(r['rr_median'])} / "
                       f"{_p(r['rr_max'])} |")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(OUT_REL))
    ap.add_argument("--n-jobs", type=int, default=4, help="parallel model fits (does not change any result)")
    a = ap.parse_args(argv)
    out = RESULTS / OUT_REL[a.command]
    g, plan = git_meta(), X.plan_record()
    conf_p = CONF / "summary.json"
    conf = json.loads(conf_p.read_text()) if conf_p.exists() else None
    T = load_tables(DERIVED, "sdev")
    probs = X.start_problems(g, plan, conf, T.sha256, folds_equal(T), out)
    if probs:
        _log({"event": "refused", "command": a.command, "commit": g.get("commit"), "reasons": probs})
        print("refused: " + "; ".join(probs), file=sys.stderr)
        return 2
    _log({"event": "started", "command": a.command, "commit": g["commit"], "plan_sha256": plan["sha256"]})
    t0 = time.time()
    meta = {**g, "data": "VitalDB derived tables (derived/sdev_*.parquet)",
            "stage": f"exploratory {a.command} (EXPLORATORY_PLAN.md)", "label": X.LABEL, "inputs_sha256": T.sha256,
            "plan": {"file": X.PLAN_FILE, "sha256": plan["sha256"], "last_commit": plan["last_commit"]},
            "confirmatory_summary_sha256": _sha(conf_p), "software": software_versions()}
    if a.command == "lead-lag":
        res = X.lead_lag(T, conf, n_jobs=1, n_boot=N_BOOT)
        w = Writer(out, meta)
        w.csv("lead_lag.csv", res["table"])
        w.csv("change_identity.csv", res["identity"])
        w.json("summary.json", res["summary"])
        w.text("summary.md", lead_lag_md(res["table"], res["summary"]))
        brief = res["table"].loc[res["table"]["variant"] != "real minus p1",
                                 ["feature", "variant", "map_change", "n_cases", "r", "r_ci_lo", "r_ci_hi"]]
    else:
        res = X.detection_limit(T, conf, n_jobs=a.n_jobs, n_boot=N_BOOT)
        w = Writer(out, meta)
        w.csv("detection_limit.csv", res["main"])
        w.csv("replicates.csv", res["replicates"])
        w.csv("replicate_summary.csv", res["replicate_summary"])
        w.csv("penalties.csv", res["penalties"])
        w.json("summary.json", res["summary"])
        w.text("summary.md", detection_md(res["summary"], res["main"], res["replicate_summary"]))
        brief = res["main"][["strength_nominal", "oracle_rr", "rr", "rr_ci_lo", "rr_ci_hi", "delta_ci_hi", "meets_rule"]]
    w.manifest()
    secs = round(time.time() - t0, 1)
    _log({"event": "completed", "command": a.command, "commit": g["commit"], "seconds": secs,
          "files": sorted(w.files)})
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(brief.to_string(index=False))
    print(f"{a.command}: {secs} s; outputs in results/{OUT_REL[a.command]}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
