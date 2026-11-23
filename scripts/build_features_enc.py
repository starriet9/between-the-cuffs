#!/usr/bin/env python3
"""Per-forecast encoder features of S-dev (ENCODER_PLAN.md §2, §4; variants real and p1, primary window).

One row per (caseid, primary, issue_n, variant) for every primary-window issue point of
`sdev_instances.parquet`, variants real (own signals) and p1 (the donor's windows and embeddings with
the recipient's times, `donor_caseid`), built by `btc.encoder.build_features` from reading times only
(the issue time, the previous valid reading and the one before it), with the window rules of
`btc.waveform` (`variant_times`, the 120-s look-back, the 2-window minimum, the 6-window reference).
The inputs are read with `scripts/build_features_wf.py::load_inputs` and checked with its
`check_waveform_status`. No cuff value, target or forecast error is read.

Hard check (ENCODER_PLAN.md §2 "only the feature extractor changes"): enc_n_windows, enc_n_usable and
enc_available equal wf_n_windows, wf_n_usable and wf_available of `sdev_features_wf.parquet` on every
row. The script refuses to start unless the work tree is clean, ENCODER_PLAN.md is committed at HEAD
and BTC_FROZEN_SCORING is unset, and unless the embedding array matches the SHA-256 of its log.

Writes (git-ignored): data/derived/sdev_enc_features.parquet and data/derived/sdev_enc_features_log.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import encoder as E, paths, provenance  # noqa: E402

SCRIPT = "scripts/build_features_enc.py"
COHORT = "sdev"
OUT = paths.derived(COHORT, "enc_features.parquet")
OUT_LOG = paths.derived(COHORT, "enc_features_log.json")
EMB = paths.derived(COHORT, "enc_window_embeddings.npy")
EMB_INDEX = paths.derived(COHORT, "enc_window_index.parquet")
EMB_LOG = paths.derived(COHORT, "enc_window_embeddings_log.json")


def wf_builder():
    """`scripts/build_features_wf.py` as a module (its input reader and status check are reused)."""
    p = Path(__file__).resolve().parent / "build_features_wf.py"
    spec = importlib.util.spec_from_file_location("build_features_wf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_embeddings(npy: Path = EMB, index: Path = EMB_INDEX, log: Path = EMB_LOG) -> tuple:
    """(matrix, index, log) of the window embeddings; the array must match the SHA-256 in its log."""
    lg = json.loads(Path(log).read_text())
    sha = hashlib.sha256(Path(npy).read_bytes()).hexdigest()
    if sha != lg["embeddings_sha256"]:
        raise ValueError("the embedding array differs from the SHA-256 recorded in its log")
    Emat = np.load(npy, allow_pickle=False)
    return Emat, pd.read_parquet(index), lg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    g, plan = provenance.git_state(), E.plan_record()
    probs = E.start_problems(g, plan)
    if probs:
        print("refused: " + "; ".join(probs), file=sys.stderr)
        return 2
    if OUT.exists():
        ap.error(f"{OUT.name} exists already; it is built once from the embeddings")
    t0 = time.time()
    wfb = wf_builder()
    windows = paths.derived(COHORT, "windows.parquet")
    issues, t_of, cases, win = wfb.load_inputs(paths.DERIVED, windows, COHORT)
    wfb.check_waveform_status(cases, windows, win)
    Emat, index, elog = load_embeddings(EMB, EMB_INDEX, EMB_LOG)
    ew = E.case_enc_windows(win, index, Emat)
    out = E.build_features(issues, t_of, cases, ew)
    wf_path = paths.derived(COHORT, "features_wf.parquet")
    wf = pd.read_parquet(wf_path, columns=["caseid", "window", "issue_n", "variant", "wf_n_windows", "wf_n_usable",
                                           "wf_available"])
    check = E.check_same_windows(out, wf)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs = {f"{n}.parquet": sha(paths.derived(COHORT, f"{n}.parquet"))
              for n in ("instances", "readings", "cases", "windows", "features_wf")}
    meta = {"git": g, "plan": plan, "script": SCRIPT, "cohort": paths.COHORT_LABEL[COHORT],
            "embeddings_sha256": elog["embeddings_sha256"], "weights_sha256": elog["weights"]["sha256"],
            "inputs_sha256": inputs}
    provenance.write_parquet(out, OUT, meta)
    summ = {}
    for v, gr in out.groupby("variant"):
        summ[v] = {"forecasts": int(len(gr)), "cases": int(gr["caseid"].nunique()),
                   "share_available": float(gr["enc_available"].mean()),
                   "share_level_missing_when_available": float(gr.loc[gr["enc_available"], "enc_lvl_000"].isna().mean()),
                   "share_change_missing_when_available": float(gr.loc[gr["enc_available"], "enc_chg_000"].isna().mean())}
    log = {**meta, "n_rows": int(len(out)), "variants": list(E.VARIANTS), "window": "primary",
           "same_windows_as_m4": check, "summary": summ, "features_sha256": sha(OUT),
           "run_seconds": round(time.time() - t0, 1)}
    provenance.write_json(OUT_LOG, log)
    print(json.dumps({k: v for k, v in log.items() if k not in ("git",)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
