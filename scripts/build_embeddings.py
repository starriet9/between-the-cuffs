#!/usr/bin/env python3
"""Window embeddings of the frozen PaPaGei-S encoder for S-dev (ENCODER_PLAN.md §1-§2).

Every usable 10-s window of `data/derived/sdev_windows.parquet` (SPEC §5; donors and cases with a
recorded blocker included, since P1 donors come from all analysable cases) is processed from its own
5,000 pleth samples (`btc.encoder.preprocess_window`) and passed through the model alone
(`btc.encoder.embed`, one window per forward pass, eval mode, float32, one thread per worker).
Signals are read through `btc.io.load_case` (cohort access and SPEC §2.3 integrity rule); the pleth
window grid must equal that of the window table.

Signal-level only: no cuff value is read and no model is fitted. S-ob is not handled here (ENCODER_PLAN.md
§7 runs after the S-dev result). The script refuses to start unless the work tree is clean, ENCODER_PLAN.md
is committed at HEAD and BTC_FROZEN_SCORING is unset; it needs no network (run it with the network off).

Writes (git-ignored, VitalDB-derived):
  data/derived/sdev_enc_window_embeddings.npy        float32 (n windows x 512), rows in (caseid, w) order
  data/derived/sdev_enc_window_index.parquet         caseid, w, row, dom_freq, processed_dom_freq
  data/derived/sdev_enc_window_embeddings_log.json   git state, plan SHA-256, weights, versions, counts,
                                                     SHA-256 of the array, raw files read, seconds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import encoder as E, paths, provenance, waveform as W  # noqa: E402

SCRIPT = "scripts/build_embeddings.py"
COHORT = "sdev"
WIN_COLS = ["caseid", "w", "t_start", "t_end", "usable", "dom_freq"]
DOM_TOL_HZ = 0.2            # signal-level sanity check [judgment]: processed vs raw dominant frequency
OUT_NPY = paths.derived(COHORT, "enc_window_embeddings.npy")
OUT_INDEX = paths.derived(COHORT, "enc_window_index.parquet")
OUT_LOG = paths.derived(COHORT, "enc_window_embeddings_log.json")

_MODEL = None


def _init_worker(weights: str):
    global _MODEL
    E.configure_torch()
    _MODEL, _ = E.load_model(Path(weights))


def run_case(args) -> dict:
    caseid, win_case = args
    t1 = time.time()
    d = W.load_case(caseid, COHORT)
    if d["pleth"] is None:
        raise ValueError(f"case {caseid}: usable windows but no pleth")
    res = E.embed_case(caseid, d["pleth"], win_case, lambda X: E.embed(_MODEL, X))
    res["seconds"] = round(time.time() - t1, 2)
    res["runtime"] = E.runtime_info()              # this worker's settings (threads, deterministic algorithms)
    res["raw_file"] = f"data/raw/{COHORT}/{int(caseid)}.npz"
    return res


def rel(p: Path) -> str:
    """Repository-relative path (not resolved: `data/` may be a link); the bare name outside the repository."""
    try:
        return Path(p).relative_to(paths.ROOT).as_posix()
    except ValueError:
        return Path(p).name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--cases", type=str, default="", help="comma-separated case IDs (checks only; output not written)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write the three outputs here instead (checks, e.g. the determinism rebuild)")
    a = ap.parse_args(argv)
    g, plan = provenance.git_state(), E.plan_record()
    probs = E.start_problems(g, plan)
    if probs:
        print("refused: " + "; ".join(probs), file=sys.stderr)
        return 2
    if a.cases and a.out_dir is None:
        ap.error("--cases needs --out-dir (a subset never replaces the full output)")
    out_npy, out_index, out_log = ((a.out_dir / OUT_NPY.name, a.out_dir / OUT_INDEX.name, a.out_dir / OUT_LOG.name)
                                   if a.out_dir else (OUT_NPY, OUT_INDEX, OUT_LOG))
    if a.out_dir is None and any(p.exists() for p in (OUT_NPY, OUT_INDEX, OUT_LOG)):
        ap.error("the S-dev embeddings exist already; a rebuild goes to --out-dir and is compared, not replaced")
    t0 = time.time()
    weights = E.check_weights(E.WEIGHTS)
    win_path = paths.derived(COHORT, "windows.parquet")
    win = pd.read_parquet(win_path, columns=WIN_COLS)
    win["usable"] = win["usable"].astype(bool)
    has = sorted(int(c) for c in win.loc[win["usable"], "caseid"].unique())
    if a.cases:
        want = {int(c) for c in a.cases.split(",")}
        has = [c for c in has if c in want]
    tasks = [(c, win.loc[win["caseid"] == c].reset_index(drop=True)) for c in has]
    if a.jobs == 1:                       # in-process (tests; same per-case code)
        _init_worker(str(E.WEIGHTS))
        res = [run_case(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.jobs, initializer=_init_worker, initargs=(str(E.WEIGHTS),)) as pool:
            res = pool.map(run_case, tasks, chunksize=1)
    res = sorted(res, key=lambda r: r["caseid"])
    Emat = np.concatenate([r["E"] for r in res], axis=0).astype(np.float32)
    idx = pd.DataFrame({"caseid": np.concatenate([np.full(len(r["w"]), r["caseid"]) for r in res]).astype(np.int64),
                        "w": np.concatenate([r["w"] for r in res]).astype(np.int64)})
    idx["row"] = np.arange(len(idx), dtype=np.int64)
    idx["dom_freq"] = np.concatenate([r["dom_freq"] for r in res])
    idx["processed_dom_freq"] = np.concatenate([r["processed_dom_freq"] for r in res])
    rts = {json.dumps(r["runtime"], sort_keys=True) for r in res}
    if len(rts) != 1:
        raise RuntimeError(f"workers ran with different settings: {rts}")
    worker_rt = json.loads(rts.pop())
    n_usable = int(win.loc[win["caseid"].isin(has), "usable"].sum())
    if len(idx) != n_usable or Emat.shape != (n_usable, E.EMB_DIM) or not np.all(np.isfinite(Emat)):
        raise RuntimeError("embedding count, shape or values unexpected")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, Emat, allow_pickle=False)
    arr_sha = hashlib.sha256(out_npy.read_bytes()).hexdigest()
    agree = np.abs(idx["processed_dom_freq"] - idx["dom_freq"]) <= DOM_TOL_HZ + 1e-9
    info = {"git": g, "plan": plan, "script": SCRIPT, "cohort": paths.COHORT_LABEL[COHORT],
            "weights": {"file": rel(E.WEIGHTS), **weights, "zenodo_record": E.ZENODO_RECORD, "doi": E.ZENODO_DOI},
            "model": {"class": "ResNet1DMoE", "kwargs": E.MODEL_KW, "upstream_commit": E.UPSTREAM_COMMIT,
                      "output_index": E.OUTPUT_INDEX, "windows_per_forward_pass": E.BATCH},
            "preprocessing": {"band_hz": list(E.BAND), "cheby2_order": E.CHEBY_ORDER, "cheby2_rs_db": E.CHEBY_RS_DB,
                              "smooth_samples": E.SMOOTH_N, "z_eps": E.Z_EPS, "resample": [E.RESAMPLE_UP, E.RESAMPLE_DOWN],
                              "note": E.FILTER_NOTE},
            "runtime": {**worker_rt, "workers": a.jobs, "start_method": "spawn" if a.jobs > 1 else "in-process",
                        "recorded_in": "each worker, after configure_torch (all workers agree, checked)"},
            "inputs": {"windows": rel(win_path), "windows_sha256": hashlib.sha256(win_path.read_bytes()).hexdigest(),
                       "windows_commit": provenance.read_meta(win_path).get("git", {}).get("commit")},
            "n_cases": len(res), "n_windows_embedded": int(len(idx)), "embeddings_sha256": arr_sha,
            "dom_freq_check": {"tolerance_hz": DOM_TOL_HZ, "n": int(len(idx)), "n_agree": int(agree.sum()),
                               "share_agree": float(agree.mean())},
            "raw_files_read": [r["raw_file"] for r in res],
            "all_raw_files_under": f"data/raw/{COHORT}/",
            "run_seconds": round(time.time() - t0, 1), "cpu_count": mp.cpu_count()}
    if not all(f.startswith(f"data/raw/{COHORT}/") for f in info["raw_files_read"]):
        raise RuntimeError("a raw file outside the S-dev folder was read")
    meta = {k: info[k] for k in ("git", "plan", "script", "cohort", "weights", "model", "embeddings_sha256")}
    provenance.write_parquet(idx, out_index, meta)
    provenance.write_json(out_log, info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("git", "raw_files_read")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
