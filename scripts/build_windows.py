#!/usr/bin/env python3
"""Build the 10-s window table of every case of one cohort (SPEC.md §4.3, §5).

For each case: the primary pleth quality index, the beat counts, the template correlation and
the per-window waveform features of every 10-s window of the stored pleth record
(`btc.waveform.case_windows`). Signals are loaded with `btc.io.load_case`; waveforms it treats
as absent under SPEC §2.3 are not used (no pleth: no windows; no ECG: no pulse arrival time).
A missing case archive is an error. The log records each case's pleth and ECG status and the
number of windows on which the R-peak detector raised.

Signal-level only: no cuff value is read and no model is fitted. S-ob (`--cohort sob`) runs
only at the frozen scoring, with BTC_FROZEN_SCORING=1 (SPEC §9).

Writes `data/derived/<cohort>_windows.parquet` and `data/derived/<cohort>_windows_log.json`.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import beats, io, paths, provenance, rpeaks, waveform as W  # noqa: E402

SCRIPT = "scripts/build_windows.py"


def run_case(caseid: int, cohort: str) -> tuple[pd.DataFrame, dict]:
    t1 = time.time()
    d = W.load_case(caseid, cohort)
    df = W.case_windows(caseid, d["pleth"], d["ecg"], d["hr"])
    info = {"caseid": int(caseid), "pleth": d["status"]["pleth"], "ecg": d["status"]["ecg"],
            "n_windows": int(len(df)), "n_usable": int(df["usable"].sum()) if len(df) else 0,
            "n_rpeak_errors": int(df["r_failed"].sum()) if len(df) else 0,
            "seconds": round(time.time() - t1, 2)}
    return df, info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=paths.COHORTS_KEYS, default="sdev")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--cases", type=str, default="", help="comma-separated case IDs (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="default: data/derived/<cohort>_windows.parquet")
    provenance.add_allow_dirty(ap)
    a = ap.parse_args(argv)
    io.check_cohort_access(a.cohort)
    state = provenance.require_clean(a.allow_dirty)
    out = a.out or paths.derived(a.cohort, "windows.parquet")
    t0 = time.time()
    cases = [int(c) for c in a.cases.split(",")] if a.cases else io.caseids(a.cohort)
    missing = [c for c in cases if not (paths.raw_dir(a.cohort) / f"{c}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {a.cohort} case archive(s) missing, e.g. {missing[:5]}; "
                                "run the download first")
    with Pool(a.jobs) as pool:
        res = pool.map(partial(run_case, cohort=a.cohort), cases, chunksize=1)
    df = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    df = df.sort_values(["caseid", "w"]).reset_index(drop=True)
    import neurokit2
    import scipy
    meta = {"git": state, "spec": provenance.SPEC_VERSION, "cohort": paths.COHORT_LABEL[a.cohort],
            "script": SCRIPT, "rpeak_detector": rpeaks.DETECTOR}
    provenance.write_parquet(df, out, meta)
    log = {
        "script": SCRIPT,
        "git": state,
        "cohort": paths.COHORT_LABEL[a.cohort],
        "rpeak_detector": rpeaks.DETECTOR,
        "pat_lag_range_s": list(beats.PAT_LAG),
        "n_cases": len(cases),
        "n_cases_with_windows": int(df["caseid"].nunique()),
        "n_windows": int(len(df)),
        "n_usable": int(df["usable"].sum()),
        "n_usable_strict": int(df["usable_strict"].sum()),
        "n_ecg_ok_windows": int(df["ecg_ok"].sum()),
        "n_rpeak_errors": int(df["r_failed"].sum()),
        "cases": [r[1] for r in res],
        "run_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "neurokit2": neurokit2.__version__},
    }
    provenance.write_json(out.parent / (out.stem + "_log.json"), log)
    print({k: v for k, v in log.items() if k not in ("cases", "git")})


if __name__ == "__main__":
    main()
