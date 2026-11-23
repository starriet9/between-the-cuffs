#!/usr/bin/env python3
"""Exploratory encoder check on S-dev (ENCODER_PLAN.md §5, §6, §9, §11; SPEC.md §6.8).

Commands (S-dev only; the S-ob step of ENCODER_PLAN.md §7 is not part of this script):
  build-check  §5, before any blood-pressure model: out-of-fold R^2 of ridge (penalty 1) from the 512
               window embeddings to the pleth rate of the same window, usable windows of the analysed
               S-dev cases, registered outer folds. Writes results/sdev/exploratory_encoder/build_check.json.
               Commit that file before `sdev`.
  sdev         §6: M4-enc and M4-enc-P1 by S-dev outer cross-validation, the M2 / M3 / M3-P1 stages
               checked against the stored confirmatory forecasts, every contrast of §6 with the paired
               subject bootstrap, penalties per outer fold, the one-line summary of §9. Writes
               results/sdev/exploratory_encoder/ (summary.json, summary.md, contrasts.csv, model_summary.csv,
               penalties.csv, dropped_blocks_outer_folds.csv, per_patient_primary.csv, the git-ignored
               predictions_primary.csv, MANIFEST.json) and then, with scripts/export_public_tables.py's
               `export_stage`, per_patient_primary_wide.csv and public_tables.json.

Guards (both commands refuse otherwise): a clean work tree (untracked files count); ENCODER_PLAN.md
committed at HEAD (its SHA-256 and last commit are recorded); BTC_FROZEN_SCORING unset; the S-dev
derived tables equal the inputs of the confirmatory run (inputs_sha256); the fold plan equals the
confirmatory folds.csv; the encoder tables match the SHA-256 in their logs and were built by the same code
(any commit between theirs and HEAD changed only files under results/); no earlier output of the command.
`sdev` also needs build_check.json committed at HEAD and computed from the same embeddings; if its R^2 is
below 0.5, `--not-interpretable` must be given and labels the result (ENCODER_PLAN.md §5). The stored
confirmatory forecasts (results/sdev/confirmatory/predictions_primary.csv, git-ignored) must match the
SHA-256 in results/sdev/MANIFEST.json. "The first complete S-dev run after this file is committed is the
result" (§6): `sdev` refuses to overwrite summary.json. Every attempt goes to a git-ignored log,
data/derived/sdev_enc_run_log.jsonl. No network is needed (run it with the network off).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc import encoder as E, paths, provenance  # noqa: E402
from btc.models import encoder as ME  # noqa: E402
from btc.models.constants import N_BOOT, SEED  # noqa: E402
from btc.models.data import load_tables  # noqa: E402
from btc.models.folds import make_fold_plan  # noqa: E402
from btc.models.outputs import Writer, git_meta  # noqa: E402
from btc.models.pipeline import software_versions  # noqa: E402
from btc.paths import DERIVED, ROOT  # noqa: E402

RESULTS = ROOT / "results"
OUT_REL = "sdev/exploratory_encoder"
CONF = RESULTS / "sdev" / "confirmatory"
SDEV_MANIFEST = RESULTS / "sdev" / "MANIFEST.json"
RUN_LOG = DERIVED / "sdev_enc_run_log.jsonl"
FEATURES = paths.derived("sdev", "enc_features.parquet")
FEATURES_LOG = paths.derived("sdev", "enc_features_log.json")
# Publication record of the plan: the push of its commit to origin/main, as GitHub's activity record of the
# repository gives it. Recorded in every result's meta, with whether the plan changed since that commit.
PLAN_PUBLIC = {"commit": "29db44bb87fb766f190184003920b86486fa7d19", "branch": "origin/main",
               "pushed_utc": "2026-11-23T03:54:36Z", "source": "GitHub activity record of the repository"}


def _git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def is_committed_at_head(rel: str) -> bool:
    """True if the repository file `rel` equals the file committed at HEAD (as run_frozen_scoring.py)."""
    if not (ROOT / rel).is_file():
        return False
    head, here = _git("rev-parse", "--verify", "--quiet", f"HEAD:{rel}"), _git("hash-object", "--", rel)
    return head.returncode == 0 and here.returncode == 0 and head.stdout.strip() == here.stdout.strip()


def code_changes_since(commit: str | None) -> list | None:
    """Paths changed between `commit` and HEAD outside results/ (None if the commit is unknown)."""
    if not commit:
        return None
    r = _git("diff", "--name-only", f"{commit}", "HEAD")
    if r.returncode != 0:
        return None
    return [p for p in r.stdout.split() if not p.startswith("results/")]


def plan_unchanged_since_public() -> bool:
    return _git("diff", "--quiet", PLAN_PUBLIC["commit"], "HEAD", "--", E.PLAN_FILE).returncode == 0


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _log(entry: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry}
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def common_problems(g: dict, plan: dict, conf: dict | None, tables_sha: dict | None, plan_ok: bool | None,
                    emb_log: dict | None, emb_sha: str | None) -> list:
    """Problems shared by both commands (see the module docstring)."""
    probs = E.start_problems(g, plan)
    if conf is None:
        probs.append("no confirmatory summary.json")
    elif tables_sha is not None and (conf.get("meta") or {}).get("inputs_sha256") != tables_sha:
        probs.append("the S-dev derived tables differ from the inputs of the confirmatory run (inputs_sha256)")
    if plan_ok is False:
        probs.append("the fold plan differs from the confirmatory folds.csv")
    if emb_log is None:
        probs.append("no window embeddings (scripts/build_embeddings.py)")
    else:
        if emb_sha != emb_log.get("embeddings_sha256"):
            probs.append("the embedding array differs from the SHA-256 in its log")
        if (emb_log.get("plan") or {}).get("sha256") != plan.get("sha256"):
            probs.append("the embeddings were built under another ENCODER_PLAN.md")
        ch = code_changes_since((emb_log.get("git") or {}).get("commit"))
        if ch is None or ch:
            probs.append(f"code changed since the embeddings were built: {(ch or ['unknown commit'])[:5]}")
        if (emb_log.get("git") or {}).get("state") != "clean":
            probs.append("the embeddings were built from a work tree that was not clean")
    return probs


def sdev_problems(out: Path, feat_log: dict | None, feat_sha: str | None, emb_sha: str | None, bc: dict | None,
                  bc_committed: bool, not_interpretable: bool, stored_ok: bool) -> list:
    """Problems specific to `sdev` (see the module docstring)."""
    probs = []
    if (out / "summary.json").exists():
        probs.append("an S-dev encoder result exists; a rerun is a departure (ENCODER_PLAN.md §6): write it "
                     "elsewhere and report both")
    if feat_log is None:
        probs.append("no encoder feature table (scripts/build_features_enc.py)")
    else:
        if feat_sha != feat_log.get("features_sha256"):
            probs.append("the encoder feature table differs from the SHA-256 in its log")
        if feat_log.get("embeddings_sha256") != emb_sha:
            probs.append("the encoder feature table was built from other embeddings")
        ch = code_changes_since((feat_log.get("git") or {}).get("commit"))
        if ch is None or ch:
            probs.append(f"code changed since the encoder features were built: {(ch or ['unknown commit'])[:5]}")
    if bc is None or not bc_committed:
        probs.append("build_check.json is missing or not committed at HEAD (ENCODER_PLAN.md §5 comes first)")
    else:
        if (bc.get("meta") or {}).get("embeddings_sha256") != emb_sha:
            probs.append("build_check.json was computed from other embeddings")
        if not bc.get("passed") and not not_interpretable:
            probs.append("the build check is below 0.5: check the build by signal-level means; to report the "
                         "result anyway, pass --not-interpretable (ENCODER_PLAN.md §5)")
    if not stored_ok:
        probs.append("the stored confirmatory predictions_primary.csv is missing or differs from the SHA-256 in "
                     "results/sdev/MANIFEST.json")
    return probs


def plan_section(n: int) -> str:
    """Section n of ENCODER_PLAN.md, verbatim (printed with the result)."""
    t = (ROOT / E.PLAN_FILE).read_text()
    m = re.search(rf"^## {n}\. .*?(?=^## {n + 1}\. |\Z)", t, re.S | re.M)
    return m.group(0).strip() if m else ""


def export_module():
    p = Path(__file__).resolve().parent / "export_public_tables.py"
    spec = importlib.util.spec_from_file_location("export_public_tables", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-check", "sdev"])
    ap.add_argument("--not-interpretable", action="store_true",
                    help="sdev: run although the build check is below 0.5; labels the result (ENCODER_PLAN.md §5)")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    out = RESULTS / OUT_REL
    g, plan = git_meta(), E.plan_record()
    conf_p = CONF / "summary.json"
    conf = json.loads(conf_p.read_text()) if conf_p.exists() else None
    T = load_tables(DERIVED, "sdev")
    pl = make_fold_plan(T.analysed, SEED)
    folds = pd.read_csv(CONF / "folds.csv", skiprows=1) if (CONF / "folds.csv").exists() else None
    pt = pl.table()
    plan_ok = folds is not None and len(folds) == len(pt) and all(
        c in folds.columns and np.array_equal(folds[c].to_numpy(), pt[c].to_numpy()) for c in pt.columns)
    emb_log_p = paths.derived("sdev", "enc_window_embeddings_log.json")
    emb_log = json.loads(emb_log_p.read_text()) if emb_log_p.exists() else None
    npy = paths.derived("sdev", "enc_window_embeddings.npy")
    emb_sha = _sha(npy) if npy.exists() else None
    probs = common_problems(g, plan, conf, T.sha256, plan_ok, emb_log, emb_sha)
    if a.command == "build-check" and (out / "build_check.json").exists():
        probs.append("build_check.json exists already")
    feat_log = feat_sha = bc = None
    stored = None
    if a.command == "sdev":
        feat_log = json.loads(FEATURES_LOG.read_text()) if FEATURES_LOG.exists() else None
        feat_sha = _sha(FEATURES) if FEATURES.exists() else None
        bcp = out / "build_check.json"
        bc = json.loads(bcp.read_text()) if bcp.exists() else None
        sp = CONF / "predictions_primary.csv"
        man = json.loads(SDEV_MANIFEST.read_text()) if SDEV_MANIFEST.exists() else {}
        stored_ok = sp.exists() and man.get("files", {}).get("confirmatory/predictions_primary.csv") == _sha(sp)
        probs += sdev_problems(out, feat_log, feat_sha, emb_sha, bc, is_committed_at_head(f"results/{OUT_REL}/build_check.json"),
                               a.not_interpretable, stored_ok)
        if stored_ok:
            stored = pd.read_csv(sp, skiprows=1)
    if probs:
        _log({"event": "refused", "command": a.command, "commit": g.get("commit"), "reasons": probs})
        print("refused: " + "; ".join(probs), file=sys.stderr)
        return 2
    _log({"event": "started", "command": a.command, "commit": g["commit"], "plan_sha256": plan["sha256"]})
    t0 = time.time()
    meta = {**g, "data": "VitalDB derived tables (derived/sdev_*.parquet) and encoder tables (derived/sdev_enc_*)",
            "stage": f"exploratory encoder check ({a.command}, ENCODER_PLAN.md)", "inputs_sha256": T.sha256,
            "plan": {"file": E.PLAN_FILE, "sha256": plan["sha256"], "last_commit": plan["last_commit"]},
            "plan_public": PLAN_PUBLIC, "plan_unchanged_since_public_commit": plan_unchanged_since_public(),
            "weights": {k: emb_log["weights"][k] for k in ("file", "size", "md5", "sha256", "zenodo_record", "doi")},
            "model": emb_log["model"], "embedding_runtime": emb_log["runtime"],
            "preprocessing": emb_log.get("preprocessing"), "preprocessing_note": E.FILTER_NOTE,
            "software": software_versions(), "embeddings_sha256": emb_sha,
            "embeddings_built_at_commit": emb_log["git"]["commit"]}
    if a.command == "build-check":
        Emat = np.load(npy, allow_pickle=False)
        index = pd.read_parquet(paths.derived("sdev", "enc_window_index.parquet"))
        win = pd.read_parquet(paths.derived("sdev", "windows.parquet"), columns=["caseid", "w", "usable", "pleth_rate"])
        res = ME.build_check(Emat, index, win, pl)
        res["dom_freq_check"] = emb_log.get("dom_freq_check")
        w = Writer(out, meta)
        w.json("build_check.json", {"stage": "build check (ENCODER_PLAN.md §5), before any blood-pressure model",
                                    **res})
        _log({"event": "completed", "command": a.command, "commit": g["commit"], "r2": res["r2_out_of_fold"],
              "seconds": round(time.time() - t0, 1)})
        print(f"build check: out-of-fold R^2 = {res['r2_out_of_fold']:.4f} over {res['n_windows']} windows of "
              f"{res['n_cases']} cases (threshold {res['threshold']}; passed {res['passed']}). Commit "
              f"results/{OUT_REL}/build_check.json before `sdev`.")
        return 0
    # sdev
    table = pd.read_parquet(FEATURES)
    wf = T.features_wf[["caseid", "window", "issue_n", "variant", "wf_n_windows", "wf_n_usable", "wf_available"]]
    same = E.check_same_windows(table, wf)
    label = None
    if not bc["passed"]:
        label = "not interpretable (build check below 0.5, ENCODER_PLAN.md §5)"
    meta.update(encoder_features_sha256=feat_sha, encoder_features_built_at_commit=feat_log["git"]["commit"],
                confirmatory_predictions_sha256=_sha(CONF / "predictions_primary.csv"),
                confirmatory_summary_sha256=_sha(conf_p),
                build_check={"r2_out_of_fold": bc["r2_out_of_fold"], "passed": bc["passed"],
                             "sha256": _sha(out / "build_check.json")})
    w = Writer(out, meta)
    extra = {"same_windows_as_m4": same, "n_windows_embedded": emb_log["n_windows_embedded"],
             "n_cases_embedded": emb_log["n_cases"], "build_check": {k: bc[k] for k in (
                 "r2_out_of_fold", "passed", "threshold", "n_windows", "n_cases")}}
    s = ME.encoder_sdev(T, table, stored, conf, w, n_jobs=a.n_jobs, n_boot=N_BOOT, conf_folds=folds, label=label,
                        extra=extra)
    w.text("summary.md",
           "# Exploratory encoder check, S-dev (ENCODER_PLAN.md)\n\n"
           f"{('**Label.** ' + label + chr(10) + chr(10)) if label else ''}"
           f"**One-line summary (fixed wording, ENCODER_PLAN.md §9).** {s['one_line_summary']}\n\n"
           f"**Build check (§5).** Out-of-fold R² {bc['r2_out_of_fold']:.4f} (threshold 0.5) over "
           f"{bc['n_windows']} usable windows of {bc['n_cases']} analysed cases.\n\n"
           f"**Plan.** {E.PLAN_FILE}, SHA-256 {plan['sha256']}, last changed in commit {plan['last_commit']}; "
           f"public on {PLAN_PUBLIC['branch']} at {PLAN_PUBLIC['pushed_utc']} (commit {PLAN_PUBLIC['commit']}).\n\n"
           "The disclosures of ENCODER_PLAN.md §8, printed with the result:\n\n" + plan_section(8) + "\n")
    w.files["build_check.json"] = _sha(out / "build_check.json")     # committed before this run (checked)
    w.manifest()
    man = export_module().export_stage(RESULTS, OUT_REL, RESULTS, git_meta())
    secs = round(time.time() - t0, 1)
    _log({"event": "completed", "command": a.command, "commit": g["commit"], "seconds": secs,
          "public_tables": list(man["files"]) if man else None})
    print(json.dumps({"N": s["N"], "one_line_summary": s["one_line_summary"],
                      "meets_rule_with_placebo": s["meets_rule_with_placebo"],
                      "dropped": s["encoder_block_outer_folds"], "seconds": secs}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
