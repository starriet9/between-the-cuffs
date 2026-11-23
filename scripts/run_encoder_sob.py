#!/usr/bin/env python3
"""Exploratory encoder check on S-ob, after the frozen scoring (ENCODER_PLAN.md §7). Runs once.

Written after the S-dev encoder result was committed (commit `SDEV_RESULT_COMMIT`); §7 runs "whatever it
shows". It adds code only (`ADDED_FILES`); the S-dev code is reused unchanged, and the run refuses if any
other file changed since the S-dev result.

Steps, in one run (S-ob pleth is read only here, with BTC_FROZEN_SCORING=1 set for this process):
1. S-ob window embeddings: the usable windows of the analysed S-ob cases and of their P1 donors (the donor
   mapping of sob_cases.parquet, as in the frozen scoring), processed and embedded exactly as for S-dev
   (`btc.encoder`: one window per forward pass, one thread per worker). Written to the git-ignored
   data/derived/sob_enc_window_embeddings.npy, sob_enc_window_index.parquet and _log.json.
2. S-ob encoder features (primary window, real and p1) of the analysed cases' issue points
   (`btc.encoder.build_features`), with the hard check against sob_features_wf.parquet; written to
   data/derived/sob_enc_features.parquet and _log.json.
3. M4-enc and M4-enc-P1 refitted on all analysed S-dev cases (`btc.models.encoder_sob.refit_encoder`,
   penalties from the outer folds as in the freeze step), checked against the frozen models.
4. One scoring of the analysed S-ob cases (`btc.models.encoder_sob.encoder_sob`) into
   results/sob/after_frozen_scoring/exploratory_encoder/ (MANIFEST.json; the per-forecast file is git-ignored),
   then the per-patient public table with scripts/export_public_tables.py's `export_stage`.

Guards (refused otherwise): a clean work tree; ENCODER_PLAN.md committed at HEAD and unchanged since
`PLAN_PUBLIC_COMMIT`; the S-dev result commit an ancestor of HEAD, its result folder unchanged since, and no
file outside results/ and `ADDED_FILES` changed since; BTC_FROZEN_SCORING=1; the frozen S-ob scoring
completed (STATE.json); the S-ob and S-dev derived tables equal the inputs of the frozen scoring and of the
confirmatory run; the stored frozen forecasts match the SHA-256 in results/sob/frozen_scoring/MANIFEST.json;
the S-dev encoder features match their log; the weights match their size and MD5; no output folder yet. A
rerun after a crash (output folder without summary.json) needs `--rerun-after-crash` and is logged with
whether the code changed. Every attempt goes to the git-ignored data/derived/sob_enc_run_log.jsonl.
No network is needed (run it with the network off).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc import encoder as E, io, paths, provenance, waveform as W  # noqa: E402
from btc.models import encoder_sob as MS  # noqa: E402
from btc.models.constants import N_BOOT  # noqa: E402
from btc.models.data import load_tables  # noqa: E402
from btc.models.outputs import Writer, git_meta  # noqa: E402
from btc.models.pipeline import load_frozen, software_versions  # noqa: E402
from btc.paths import ROOT  # noqa: E402

COHORT = "sob"
RESULTS = ROOT / "results"
OUT_REL = "sob/after_frozen_scoring/exploratory_encoder"
FROZEN_DIR = RESULTS / "sob" / "frozen_scoring"
CONF = RESULTS / "sdev" / "confirmatory"
SDEV_RESULT_COMMIT = "0626dd297e7b5cd63ad6d481f9bd8712730436cb"   # the S-dev encoder result (ENCODER_PLAN.md §6)
PLAN_PUBLIC_COMMIT = "29db44bb87fb766f190184003920b86486fa7d19"
ADDED_FILES = ("btc/models/encoder_sob.py", "scripts/run_encoder_sob.py", "tests/test_encoder_sob.py",
               "btc/SCHEMA.md")
WIN_COLS = ["caseid", "w", "t_start", "t_end", "usable", "dom_freq"]
DOM_TOL_HZ = 0.2

_MODEL = None


def _git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def derived(name: str) -> Path:
    return paths.derived(COHORT, name)


def SOB_DERIVED() -> list:
    return [derived(n) for n in ("enc_window_embeddings.npy", "enc_window_index.parquet",
                                 "enc_window_embeddings_log.json", "enc_features.parquet", "enc_features_log.json")]


def run_log() -> Path:
    return paths.DERIVED / "sob_enc_run_log.jsonl"


def _log(entry: dict) -> None:
    p = run_log()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry},
                           default=str) + "\n")


# --------------------------------------------------------------------------- guards
def history_problems(sdev_commit: str) -> list:
    """The S-dev result commit is an ancestor of HEAD, its result folder is unchanged since, the plan is
    unchanged since its public commit, and only results/ and ADDED_FILES changed since the S-dev result."""
    probs = []
    if _git("merge-base", "--is-ancestor", sdev_commit, "HEAD").returncode != 0:
        probs.append(f"the S-dev result commit {sdev_commit[:7]} is not an ancestor of HEAD")
        return probs
    if _git("diff", "--quiet", sdev_commit, "HEAD", "--", "results/sdev/exploratory_encoder/").returncode != 0:
        probs.append("results/sdev/exploratory_encoder/ changed since the S-dev result commit")
    if _git("diff", "--quiet", PLAN_PUBLIC_COMMIT, "HEAD", "--", E.PLAN_FILE).returncode != 0:
        probs.append(f"{E.PLAN_FILE} changed since its public commit {PLAN_PUBLIC_COMMIT[:7]}")
    r = _git("diff", "--name-only", sdev_commit, "HEAD")
    other = [p for p in r.stdout.split() if not p.startswith("results/") and p not in ADDED_FILES]
    if r.returncode != 0 or other:
        probs.append(f"files other than the S-ob additions changed since the S-dev result: {other[:5]}")
    return probs


def sob_problems(g: dict, plan: dict, out: Path, rerun: bool, state: dict | None, tables_sha: dict | None,
                 frozen_meta_sha: dict | None, dev_sha: dict | None, conf_sha: dict | None, stored_ok: bool,
                 feat_ok: bool, derived_exist: list | None = None) -> list:
    probs = []
    if derived_exist and not rerun:
        probs.append(f"S-ob encoder tables exist already ({', '.join(derived_exist)}); the S-ob step runs once")
    if g.get("state") != "clean":
        probs.append(f"the work tree is not clean (git state {g.get('state')}; untracked files count)")
    if not plan.get("committed_at_head"):
        probs.append(f"{E.PLAN_FILE} is not committed at HEAD, or differs from the committed file")
    if os.environ.get(io.FROZEN_SCORING_ENV) != "1":
        probs.append(f"{io.FROZEN_SCORING_ENV}=1 must be set for this run (S-ob signals, SPEC §9)")
    if not state or state.get("status") != "completed":
        probs.append("the frozen S-ob scoring has not completed (results/sob/frozen_scoring/STATE.json)")
    if tables_sha is None or tables_sha != frozen_meta_sha:
        probs.append("the S-ob derived tables differ from the inputs of the frozen scoring (inputs_sha256)")
    if dev_sha is None or dev_sha != conf_sha:
        probs.append("the S-dev derived tables differ from the inputs of the confirmatory run (inputs_sha256)")
    if not stored_ok:
        probs.append("the stored frozen-scoring predictions_primary.csv is missing or differs from the SHA-256 in "
                     "results/sob/frozen_scoring/MANIFEST.json")
    if not feat_ok:
        probs.append("the S-dev encoder features are missing or differ from the SHA-256 in their log")
    if out.exists():
        if not rerun:
            probs.append(f"results/{OUT_REL}/ exists: the S-ob step runs once (a rerun after a crash needs "
                         "--rerun-after-crash)")
        elif (out / "summary.json").exists():
            probs.append("the S-ob step has completed already (summary.json exists)")
    return probs


# --------------------------------------------------------------------------- S-ob embeddings and features
def _init_worker(weights: str):
    global _MODEL
    E.configure_torch()
    _MODEL, _ = E.load_model(Path(weights))


def run_case(args) -> dict:
    caseid, win_case = args
    d = W.load_case(caseid, COHORT)
    if d["pleth"] is None:
        raise ValueError(f"case {caseid}: usable windows but no pleth")
    res = E.embed_case(caseid, d["pleth"], win_case, lambda X: E.embed(_MODEL, X))
    res["raw_file"] = f"data/raw/{COHORT}/{int(caseid)}.npz"
    res["runtime"] = E.runtime_info()
    return res


def build_sob_tables(T_ob, jobs: int, g: dict, plan: dict) -> tuple:
    """Steps 1-2: (encoder table of the analysed S-ob cases, record)."""
    t0 = time.time()
    wfb_spec = importlib.util.spec_from_file_location("build_features_wf",
                                                      Path(__file__).resolve().parent / "build_features_wf.py")
    wfb = importlib.util.module_from_spec(wfb_spec)
    wfb_spec.loader.exec_module(wfb)
    win_path = derived("windows.parquet")
    issues, t_of, cases, win_all = wfb.load_inputs(paths.DERIVED, win_path, COHORT)
    wfb.check_waveform_status(cases, win_path, win_all)
    analysed = sorted(int(c) for c in T_ob.analysed["caseid"])
    donors = cases.set_index("caseid").loc[analysed, "donor_caseid"].dropna().astype(int).tolist()
    want = sorted(set(analysed) | set(donors))
    win = win_all.loc[win_all["caseid"].isin(want), WIN_COLS].reset_index(drop=True)
    win["usable"] = win["usable"].astype(bool)
    has = sorted(int(c) for c in win.loc[win["usable"], "caseid"].unique())
    tasks = [(c, win.loc[win["caseid"] == c].reset_index(drop=True)) for c in has]
    weights = E.check_weights(E.WEIGHTS)
    if jobs == 1:                                            # in-process (tests)
        _init_worker(str(E.WEIGHTS))
        res = [run_case(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(jobs, initializer=_init_worker, initargs=(str(E.WEIGHTS),)) as pool:
            res = pool.map(run_case, tasks, chunksize=1)
    res = sorted(res, key=lambda r: r["caseid"])
    rts = {json.dumps(r["runtime"], sort_keys=True) for r in res}
    if len(rts) != 1:
        raise RuntimeError(f"workers ran with different settings: {rts}")
    Emat = np.concatenate([r["E"] for r in res], axis=0).astype(np.float32)
    idx = pd.DataFrame({"caseid": np.concatenate([np.full(len(r["w"]), r["caseid"]) for r in res]).astype(np.int64),
                        "w": np.concatenate([r["w"] for r in res]).astype(np.int64)})
    idx["row"] = np.arange(len(idx), dtype=np.int64)
    idx["dom_freq"] = np.concatenate([r["dom_freq"] for r in res])
    idx["processed_dom_freq"] = np.concatenate([r["processed_dom_freq"] for r in res])
    if len(idx) != int(win["usable"].sum()) or not np.all(np.isfinite(Emat)):
        raise RuntimeError("embedding count or values unexpected")
    np.save(derived("enc_window_embeddings.npy"), Emat, allow_pickle=False)
    emb_sha = _sha(derived("enc_window_embeddings.npy"))
    agree = np.abs(idx["processed_dom_freq"] - idx["dom_freq"]) <= DOM_TOL_HZ + 1e-9
    raw = [r["raw_file"] for r in res]
    if not all(f.startswith(f"data/raw/{COHORT}/") for f in raw):
        raise RuntimeError("a raw file outside the S-ob folder was read")
    emb_log = {"git": g, "plan": plan, "script": "scripts/run_encoder_sob.py", "cohort": "S-ob",
               "cases_embedded": "the analysed S-ob cases and their P1 donors",
               "n_analysed_cases": len(analysed), "donors": sorted(set(donors)), "n_cases": len(res),
               "n_windows_embedded": int(len(idx)), "embeddings_sha256": emb_sha,
               "weights": {"file": "data/external/papagei/papagei_s.pt", **weights},
               "model": {"class": "ResNet1DMoE", "kwargs": E.MODEL_KW, "upstream_commit": E.UPSTREAM_COMMIT,
                         "output_index": E.OUTPUT_INDEX, "windows_per_forward_pass": E.BATCH},
               "preprocessing_note": E.FILTER_NOTE,
               "runtime": {**json.loads(rts.pop()), "workers": jobs},
               "inputs": {"windows": "data/derived/sob_windows.parquet", "windows_sha256": _sha(win_path)},
               "dom_freq_check": {"tolerance_hz": DOM_TOL_HZ, "n": int(len(idx)), "n_agree": int(agree.sum()),
                                  "share_agree": float(agree.mean()) if len(idx) else None},
               "raw_files_read": raw, "seconds": round(time.time() - t0, 1)}
    provenance.write_parquet(idx, derived("enc_window_index.parquet"),
                             {k: emb_log[k] for k in ("git", "plan", "script", "cohort", "embeddings_sha256")})
    provenance.write_json(derived("enc_window_embeddings_log.json"), emb_log)
    ew = E.case_enc_windows(win, idx, Emat)
    iss = issues.loc[issues["caseid"].isin(analysed)]
    table = E.build_features(iss, t_of, cases, ew)
    wf = pd.read_parquet(derived("features_wf.parquet"),
                         columns=["caseid", "window", "issue_n", "variant", "wf_n_windows", "wf_n_usable", "wf_available"])
    same = E.check_same_windows(table, wf)
    provenance.write_parquet(table, derived("enc_features.parquet"),
                             {"git": g, "plan": plan, "script": "scripts/run_encoder_sob.py", "cohort": "S-ob",
                              "embeddings_sha256": emb_sha})
    feat_sha = _sha(derived("enc_features.parquet"))
    provenance.write_json(derived("enc_features_log.json"),
                          {"git": g, "embeddings_sha256": emb_sha, "features_sha256": feat_sha, "n_rows": int(len(table)),
                           "same_windows_as_m4": same})
    record = {"n_cases_embedded": len(res), "n_windows_embedded": int(len(idx)), "donors": sorted(set(donors)),
              "embeddings_sha256": emb_sha, "encoder_features_sha256": feat_sha, "same_windows_as_m4": same,
              "dom_freq_check": emb_log["dom_freq_check"], "embedding_runtime": emb_log["runtime"],
              "weights": emb_log["weights"], "seconds_embeddings_and_features": round(time.time() - t0, 1)}
    return table, record


def export_module():
    spec = importlib.util.spec_from_file_location("export_public_tables",
                                                  Path(__file__).resolve().parent / "export_public_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DEPARTURES = (
    "S-dev embeddings were built twice: the first build's log recorded the torch settings of the parent "
    "process instead of the workers'; the record was fixed (commit a07e0d2) and the embeddings rebuilt before "
    "any blood-pressure model. Both builds gave byte-identical arrays (SHA-256 7e1c56b2...d812).",
    "The S-ob code (this step) was written after the S-dev encoder result was known (ENCODER_PLAN.md §7 fixes "
    "what it does and says it runs whatever the S-dev result shows).",
    "The fast-cycling row of ENCODER_PLAN.md §6 is omitted for S-ob (an S-dev restriction).",
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun-after-crash", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="embedding workers")
    ap.add_argument("--n-jobs", type=int, default=2, help="refit jobs")
    a = ap.parse_args(argv)
    out = RESULTS / OUT_REL
    g, plan = git_meta(), E.plan_record()
    sdev_commit = _git("rev-parse", "--verify", "--quiet", SDEV_RESULT_COMMIT + "^{commit}").stdout.strip()
    st_p = FROZEN_DIR / "STATE.json"
    state = json.loads(st_p.read_text()) if st_p.exists() else None
    fman = json.loads((FROZEN_DIR / "MANIFEST.json").read_text())
    fsum = json.loads((FROZEN_DIR / "summary.json").read_text())
    conf = json.loads((CONF / "summary.json").read_text())
    T_ob, T_dev = load_tables(paths.DERIVED, COHORT), load_tables(paths.DERIVED, "sdev")
    sp = FROZEN_DIR / "predictions_primary.csv"
    stored_ok = sp.exists() and fman["files"].get("frozen_scoring/predictions_primary.csv") == _sha(sp)
    flog_p, fpar = paths.derived("sdev", "enc_features_log.json"), paths.derived("sdev", "enc_features.parquet")
    flog = json.loads(flog_p.read_text()) if flog_p.exists() else {}
    feat_ok = fpar.exists() and flog.get("features_sha256") == _sha(fpar)
    exist = [p.name for p in SOB_DERIVED() if p.exists()]
    probs = sob_problems(g, plan, out, a.rerun_after_crash, state, T_ob.sha256, fman["meta"].get("inputs_sha256"),
                         T_dev.sha256, (conf.get("meta") or {}).get("inputs_sha256"), stored_ok, feat_ok, exist)
    probs += history_problems(sdev_commit) if sdev_commit else [f"commit {SDEV_RESULT_COMMIT[:7]} not found"]
    if probs:
        _log({"event": "refused", "commit": g.get("commit"), "reasons": probs})
        print("refused: " + "; ".join(probs), file=sys.stderr)
        return 2
    rerun = out.exists()
    if rerun:
        prev = [json.loads(l) for l in run_log().read_text().splitlines()] if run_log().exists() else []
        prev_commit = next((e.get("commit") for e in reversed(prev) if e.get("event") == "started"), None)
        k, dest = 1, paths.DERIVED / "sob_enc_crashed_output_1"
        while dest.exists():
            k += 1
            dest = paths.DERIVED / f"sob_enc_crashed_output_{k}"
        shutil.move(str(out), str(dest))                      # kept, git-ignored
        _log({"event": "rerun_after_crash", "previous_commit": prev_commit, "commit": g["commit"],
              "code_changed": prev_commit != g["commit"], "crashed_output_moved_to": dest.name})
    _log({"event": "started", "commit": g["commit"], "plan_sha256": plan["sha256"]})
    t0 = time.time()
    try:
        frozen = load_frozen(RESULTS)
        table, record = build_sob_tables(T_ob, a.jobs, g, plan)
        t1 = time.time()
        dev_table = pd.read_parquet(fpar)
        m2 = conf["m2_structure"]
        fits = MS.refit_encoder(T_dev, dev_table, m2["form"], m2["structure"], n_jobs=a.n_jobs)
        t2 = time.time()
        meta = {**g, "data": "VitalDB derived tables (derived/sob_*, derived/sdev_*) and encoder tables "
                             "(derived/sob_enc_*, derived/sdev_enc_*)",
                "stage": MS.SOB_LABEL, "inputs_sha256": T_ob.sha256, "sdev_inputs_sha256": T_dev.sha256,
                "plan": {"file": E.PLAN_FILE, "sha256": plan["sha256"], "last_commit": plan["last_commit"]},
                "plan_public_commit": PLAN_PUBLIC_COMMIT, "plan_unchanged_since_public_commit": True,
                "written_after_sdev_result": {"sdev_result_commit": sdev_commit,
                                              "note": "this step's code was written after the S-dev encoder result "
                                                      "was committed (ENCODER_PLAN.md §7)"},
                "frozen_models_sha256": json.loads((RESULTS / "frozen" / "FROZEN.json").read_text())["models_sha256"],
                "frozen_predictions_sha256": _sha(sp), "frozen_summary_sha256": _sha(FROZEN_DIR / "summary.json"),
                "sdev_encoder_features_sha256": flog["features_sha256"],
                "sdev_embeddings_sha256": flog["embeddings_sha256"],
                "weights": record["weights"], "embedding_runtime": record["embedding_runtime"],
                "preprocessing_note": E.FILTER_NOTE, "software": software_versions(), "rerun_after_crash": rerun}
        w = Writer(out, meta)
        stored = pd.read_csv(sp, skiprows=1)
        extra = {k: record[k] for k in ("n_cases_embedded", "n_windows_embedded", "donors", "same_windows_as_m4",
                                        "dom_freq_check")}
        extra["departures"] = list(DEPARTURES)
        s = MS.encoder_sob(T_ob, table, stored, fsum, fits, frozen, w, n_boot=N_BOOT, extra=extra)
        w.text("summary.md",
               "# Exploratory encoder check, S-ob (ENCODER_PLAN.md §7)\n\n"
               f"**Label.** {MS.SOB_LABEL}\n\n"
               f"**One-line summary.** {s['one_line_summary']}\n\n"
               "**Departures from ENCODER_PLAN.md (S-dev and S-ob steps).**\n\n" +
               "".join(f"- {d}\n" for d in DEPARTURES) + "\n" +
               f"**Plan.** {E.PLAN_FILE}, SHA-256 {plan['sha256']}, unchanged since commit {PLAN_PUBLIC_COMMIT}.\n")
        w.manifest()
        man = export_module().export_stage(RESULTS, OUT_REL, RESULTS, git_meta())
    except Exception as ex:
        _log({"event": "crashed", "commit": g["commit"], "error": repr(ex)[:500]})
        raise
    secs = {"embeddings_and_features": round(t1 - t0, 1), "refit": round(t2 - t1, 1), "total": round(time.time() - t0, 1)}
    _log({"event": "completed", "commit": g["commit"], "seconds": secs, "public_tables": list(man["files"]) if man else None})
    print(json.dumps({"n": s["n"], "one_line_summary": s["one_line_summary"],
                      "stages_not_equal_to_frozen": s["refit"]["stages_not_equal_to_frozen"],
                      "encoder_block_dropped_in_refit": s["refit"]["encoder_block_dropped"], "seconds": secs},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
