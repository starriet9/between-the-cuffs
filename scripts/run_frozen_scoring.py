#!/usr/bin/env python3
"""Freeze the models and score S-ob once (SPEC §9 steps 6-8; PREREG §4, §6, §9).

Commands:
  freeze  refit every confirmatory model on all S-dev (penalties chosen with the outer folds,
          §4.5) and write `<out>/frozen/` (pickled models, SHA-256, coefficients, penalty path).
          No S-ob access. On real data it refuses a work tree with a modified tracked file or an
          untracked file outside the output folder, an existing `<out>/frozen/`, and S-dev tables
          whose SHA-256 differ from those of the confirmatory run. After fitting it checks that the
          frozen models were fitted on exactly the analysed cases of the confirmatory run (count and
          case IDs, SPEC §2.2) with the same inputs; if not, `<out>/frozen/` is moved to
          `<out>/frozen_rejected*/` and the command fails.
  score   the one frozen S-ob scoring (§9 step 7): loads the frozen models (SHA-256 checked),
          scores every one of them on the analysed S-ob cases (SPEC §2.2), writes Sentence 2 and the
          cesarean line to `<out>/sob/frozen_scoring/`, and in the same run scores them on all
          analysable S-ob cases, including those with a recorded neuromuscular blocker
          (`frozen_scoring/sensitivity_including_nmb/`, SPEC §2.2). On real data it needs
          `--frozen-tag`: HEAD must carry that tag, the work tree must be clean as for `freeze`,
          and `<out>/frozen/FROZEN.json` must be the file committed at HEAD. It refuses to run a
          second time. A rerun after a crash
          (a run that started and did not complete) needs `--rerun-after-crash` and is logged with
          both commits; a code change between the two runs must be reported with its diff (§9).
  after   S-ob analyses after the frozen scoring (§9 step 8), labelled as such, with models fitted
          on S-dev alone; writes `<out>/sob/after_frozen_scoring/` and the README table block.
          On real data it needs `--frozen-tag` and passes the same checks as `score`.
  series  the per-case files of the case browser, the summary figure and the replays (§8.1-§8.3,
          §9 step 8, labelled as computed after the frozen scoring), in the format of
          btc/site/series.py: loads the frozen models (SHA-256 checked, no refit) and writes
          `--cases-dir` (default data/derived/site/, git-ignored: VitalDB-derived series). Needs a
          completed `score` and `after`. On real data it needs `--frozen-tag` and passes the same
          checks as `score`; it reads the S-ob heart-rate and pump tracks through btc.io, so
          BTC_FROZEN_SCORING=1 must be set for this run, and the window table
          `<sob-dir>/sob_windows.parquet`.
  all     freeze, score and after in one go (synthetic tests only: on real data `score` and
          `after` need `<out>/frozen/FROZEN.json` committed at the frozen-model tag, which a
          `freeze` in the same run cannot provide).

Every attempt is appended to `<out>/sob/frozen_scoring_log.jsonl`. `--synthetic` generates
synthetic S-dev and S-ob tables and never reads `--data-dir` or `--sob-dir`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc.models import pipeline as P  # noqa: E402
from btc.models.constants import N_BOOT  # noqa: E402
from btc.models.data import load_tables, tables_from_frames  # noqa: E402
from btc.models.outputs import Writer, git_meta, tracked_files_modified, untracked_files  # noqa: E402
from btc.paths import DERIVED, ROOT  # noqa: E402


def _log(out: Path, entry: dict):
    p = out / "sob" / "frozen_scoring_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry}
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _rel_dir(out: Path) -> str | None:
    """`out` relative to the repository root, with a trailing slash; None if outside it."""
    try:
        return Path(out).resolve().relative_to(ROOT.resolve()).as_posix().rstrip("/") + "/"
    except ValueError:
        return None


def tree_problems(g: dict, out: Path) -> list:
    """Why the work tree is not clean enough for `freeze`, `score` or a rerun with unchanged code
    (SPEC §9 steps 6-7): an unknown git state, a modified, staged or deleted tracked file, a truncated
    status, or an untracked file outside the output folder `out` (untracked outputs inside it, such
    as a crashed run's own files, are allowed). Reads both `git_meta` formats (`state`/`status`, and
    `tracked_files_modified` from before untracked files were recorded)."""
    mod = tracked_files_modified(g)
    if mod is None:
        return ["the git state is unknown"]
    probs = []
    status = g.get("status") or []
    if mod:
        lines = [x for x in status if not (x.startswith("?? ") or x == "...")]
        probs.append("modified, staged or deleted tracked files: " + (", ".join(lines[:5]) or "(not listed)"))
    if "..." in status:
        probs.append("more changed paths than the recorded status lists")
    rel = _rel_dir(out)
    for p in untracked_files(g) or []:
        q = p if p.endswith("/") else p + "/"
        if not (rel and q.startswith(rel)):
            probs.append(f"untracked file outside the output folder: {p}")
    return probs


def code_changed(g: dict, prev_commit: str | None, out: Path) -> bool:
    """For a rerun after a crash (SPEC §9 step 7): a different commit, an unknown git state, or any
    modified or untracked file other than untracked outputs inside the output folder `out` (the crashed
    run's own files)."""
    return prev_commit != g.get("commit") or bool(tree_problems(g, out))


def is_committed_at_head(path: Path) -> bool:
    """True if `path` is inside the repository and its content equals the file committed at HEAD."""
    d = _rel_dir(Path(path).parent)
    if d is None or not Path(path).is_file():
        return False
    rel = (d + Path(path).name).lstrip("/")
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    head, here = run("rev-parse", "--verify", "--quiet", f"HEAD:{rel}"), run("hash-object", "--", rel)
    return head.returncode == 0 and here.returncode == 0 and head.stdout.strip() == here.stdout.strip()


def freeze_input_problems(Td, conf: dict) -> list:
    """Before fitting: the S-dev tables must be those of the confirmatory run, which used the analysed
    cases only; on real data that run must record its feature tag at HEAD and no modified tracked
    file (its `meta`, in either `git_meta` format)."""
    cm = conf.get("meta") or {}
    probs = []
    if cm.get("inputs_sha256") != Td.sha256:
        probs.append("the S-dev tables differ from the inputs of the confirmatory run (inputs_sha256)")
    if conf.get("include_nmb") is not False:
        probs.append("the confirmatory summary does not record a run on the analysed cases only (include_nmb)")
    if not str(cm.get("data", "")).startswith("synthetic"):
        if tracked_files_modified(cm) is not False:
            probs.append("the confirmatory run does not record a clean tracked tree")
        if not cm.get("feature_tag") or cm.get("feature_tag") not in (cm.get("tags_at_head") or []):
            probs.append("the confirmatory run does not record its feature tag at HEAD")
    return probs


def freeze_output_problems(out: Path, Td, conf: dict) -> list:
    """After fitting (SPEC §2.2, §9 step 6): the frozen models were fitted on the analysed S-dev cases
    of the confirmatory run (the count from the tables, from the confirmatory summary and from the
    pickled case list; the case IDs from the tables and from the confirmatory folds.csv when present),
    with its inputs (SHA-256) and its M2 structure."""
    fz = json.loads((out / "frozen" / "FROZEN.json").read_text())
    man = json.loads((out / "frozen" / "MANIFEST.json").read_text())
    frozen = P.load_frozen(out)
    cases = sorted(int(c) for c in frozen["sdev_cases"])
    probs = []
    n, n_tables, n_conf = fz.get("n_sdev_cases"), int(len(Td.analysed)), conf.get("n_analysed_cases")
    if not (n == n_tables == n_conf == len(cases)):
        probs.append(f"n_sdev_cases = {n}, but the tables have {n_tables} analysed cases, the confirmatory run "
                     f"{n_conf} and the frozen case list {len(cases)}")
    if (fz.get("sdev_case_counts") or {}).get("include_nmb") is not False:
        probs.append("the frozen models were not fitted on the analysed cases only (include_nmb)")
    if cases != sorted(Td.analysed["caseid"].astype(int).tolist()):
        probs.append("the frozen S-dev case IDs differ from the analysed cases of the tables")
    folds = out / "sdev" / "confirmatory" / "folds.csv"
    if folds.exists() and sorted(pd.read_csv(folds, skiprows=1)["caseid"].astype(int).tolist()) != cases:
        probs.append("the frozen S-dev case IDs differ from those of the confirmatory folds.csv")
    if man["meta"].get("inputs_sha256") != (conf.get("meta") or {}).get("inputs_sha256"):
        probs.append("inputs_sha256 of the freeze differs from that of the confirmatory summary.json")
    m = conf["m2_structure"]
    if fz.get("m2_structure") != {"form": m["form"], "structure": m["structure"]}:
        probs.append("the M2 structure differs from that of the confirmatory run")
    return probs


def score_problems(g: dict, out: Path, frozen_tag: str) -> list:
    """Why an S-ob step (`score`, `after` or `series`) cannot start on real data (SPEC §9 steps 7-8:
    every S-ob step runs at the frozen tag): HEAD does not carry `frozen_tag`, the work tree is not
    clean (`tree_problems`), or `<out>/frozen/FROZEN.json` (which holds the models' SHA-256) is not
    the file committed at HEAD."""
    probs = []
    if frozen_tag not in (g.get("tags_at_head") or []):
        probs.append(f"HEAD does not carry tag {frozen_tag!r}")
    probs += tree_problems(g, out)
    if not is_committed_at_head(out / "frozen" / "FROZEN.json"):
        probs.append("frozen/FROZEN.json is not the file committed at HEAD")
    return probs


def _reject_frozen(out: Path) -> Path:
    """Move a freeze that failed its checks out of the way, so that `score` cannot load it."""
    k, dest = 1, out / "frozen_rejected"
    while dest.exists():
        k += 1
        dest = out / f"frozen_rejected_{k}"
    shutil.move(str(out / "frozen"), str(dest))
    return dest


def _state(out: Path):
    p = out / "sob" / "frozen_scoring" / "STATE.json"
    return (json.loads(p.read_text()) if p.exists() else None), p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["freeze", "score", "after", "series", "all"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-seed", type=int, default=1)
    ap.add_argument("--wf-signal", type=float, default=0.0)
    ap.add_argument("--hr-signal", type=float, default=0.0)
    ap.add_argument("--data-dir", type=Path, default=DERIVED, help="folder with sdev_*.parquet")
    ap.add_argument("--sob-dir", type=Path, default=DERIVED, help="folder with sob_*.parquet")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cases-dir", type=Path, default=None,
                    help="series: output folder (default data/derived/site/; <out>/site_cases with --synthetic)")
    ap.add_argument("--rerun-after-crash", action="store_true")
    ap.add_argument("--frozen-tag", default=None,
                    help="score, after, series: the frozen-model tag (SPEC §9 steps 6-8); required on real data, "
                         "HEAD must carry it")
    ap.add_argument("--no-gb", action="store_true", help=argparse.SUPPRESS)       # synthetic tests only
    ap.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help=argparse.SUPPRESS)  # synthetic tests only
    a = ap.parse_args(argv)
    if not a.synthetic and (a.command == "all" or a.n_boot != N_BOOT or a.no_gb):
        ap.error("'all', --n-boot and --no-gb are for synthetic tests only")
    g = git_meta()
    out = a.out_dir or (ROOT / ("results_synthetic" if a.synthetic else "results"))
    if a.synthetic:
        from btc.models.synthetic import make_synthetic
        syn = make_synthetic(a.synthetic_seed, a.wf_signal, a.hr_signal)
        sdev = lambda: tables_from_frames("sdev", syn["sdev"])
        sob = lambda: tables_from_frames("sob", syn["sob"])
        data = f"synthetic(seed={a.synthetic_seed}, wf_signal={a.wf_signal}, hr_signal={a.hr_signal})"
    else:
        sdev = lambda: load_tables(a.data_dir, "sdev")
        sob = lambda: load_tables(a.sob_dir, "sob")
        data = "VitalDB derived tables"
    meta = {**g, "data": data}
    cmds = ["freeze", "score", "after"] if a.command == "all" else [a.command]

    if "freeze" in cmds:
        conf_p = out / "sdev" / "confirmatory" / "summary.json"
        if not a.synthetic:
            probs = tree_problems(g, out)
            if probs:
                ap.error("freeze on real data needs a clean work tree (SPEC §9 step 6): " + "; ".join(probs))
            if (out / "frozen").exists():
                ap.error(f"{out / 'frozen'} exists; a second freeze is a deviation (SPEC §9 step 6): write it "
                         "to another --out-dir and report both")
            if not conf_p.exists():
                ap.error("no confirmatory result found; run scripts/run_confirmatory.py first (SPEC §9)")
        conf = json.loads(conf_p.read_text()) if conf_p.exists() else None
        Td = sdev()
        exp = None
        if conf is not None:
            exp = (conf["m2_structure"]["form"], conf["m2_structure"]["structure"])
            probs = freeze_input_problems(Td, conf)
            if probs:
                print("refused: " + "; ".join(probs), file=sys.stderr)
                return 2
        w = Writer(out, {**meta, "stage": "freeze", "inputs_sha256": Td.sha256})
        P.freeze(Td, w, n_jobs=a.n_jobs, expected_structure=exp)
        w.manifest("frozen/MANIFEST.json")
        if conf is not None:
            probs = freeze_output_problems(out, Td, conf)
            if probs:
                dest = _reject_frozen(out)
                print("FREEZE CHECK FAILED (SPEC §2.2, §9 step 6): " + "; ".join(probs) +
                      f". The outputs were moved to {dest}; nothing is frozen.", file=sys.stderr)
                return 3
            print(f"checked: {json.loads((out / 'frozen' / 'FROZEN.json').read_text())['n_sdev_cases']} analysed "
                  "S-dev cases and the inputs of the confirmatory run")
        print(f"frozen models written to {out / 'frozen'}")

    tagged = [c for c in cmds if c in ("score", "after", "series")]
    if tagged and not a.synthetic:
        # every S-ob step runs at the frozen-model tag (SPEC §9 steps 7-8), checked before any S-ob read
        if not a.frozen_tag:
            ap.error(f"--frozen-tag is required for {a.command} on real data "
                     "(SPEC §9 steps 7-8: every S-ob step runs at the frozen-model tag)")
        probs = score_problems(g, out, a.frozen_tag)
        if probs:
            _log(out, {"event": "refused", "command": a.command, "reason": "; ".join(probs), "commit": g["commit"],
                       "frozen_tag": a.frozen_tag})
            print("refused: " + "; ".join(probs), file=sys.stderr)
            return 2

    if "score" in cmds:
        st, sp = _state(out)
        if st is not None and st.get("status") == "completed":
            _log(out, {"event": "refused", "reason": "already completed", "commit": g["commit"]})
            print("refused: the frozen S-ob scoring has already run (SPEC §9 step 7)", file=sys.stderr)
            return 2
        if st is not None and not a.rerun_after_crash:
            _log(out, {"event": "refused", "reason": "earlier run did not complete; --rerun-after-crash needed",
                       "commit": g["commit"]})
            print("refused: an earlier run started and did not complete; rerun with --rerun-after-crash "
                  "(logged; a code change must be reported with its diff)", file=sys.stderr)
            return 2
        rerun = st is not None
        if rerun:
            changed = code_changed(g, st.get("commit"), out)
            _log(out, {"event": "rerun_after_crash", "previous_commit": st.get("commit"), "commit": g["commit"],
                       "code_changed": changed})
            if changed:
                print("warning: code differs from the crashed run; report the change with its diff (SPEC §9)",
                      file=sys.stderr)
        frozen = P.load_frozen(out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"status": "started", "commit": g["commit"], "rerun_after_crash": rerun}) + "\n")
        _log(out, {"event": "started", "commit": g["commit"], "describe": g["describe"], "data": data,
                   "frozen_tag": a.frozen_tag, "rerun_after_crash": rerun})
        try:
            To = sob()
            w = Writer(out / "sob", {**meta, "stage": "frozen S-ob scoring", "frozen_tag": a.frozen_tag,
                                     "inputs_sha256": To.sha256,
                                     "frozen_models_sha256": json.loads(
                                         (out / "frozen" / "FROZEN.json").read_text())["models_sha256"]})
            s = P.score_sob(frozen, To, w, n_boot=a.n_boot)
            w.manifest("frozen_scoring/MANIFEST.json")
        except Exception:
            _log(out, {"event": "crashed", "commit": g["commit"], "error": traceback.format_exc(limit=3)})
            raise
        sp.write_text(json.dumps({"status": "completed", "commit": g["commit"], "rerun_after_crash": rerun}) + "\n")
        _log(out, {"event": "completed", "commit": g["commit"]})
        print("Sentence 2:", s["sentence_2"])
        print("Cesarean line:", s["cesarean_line"])
        sn = s["sensitivity_including_nmb"]
        print(f"Sensitivity ({sn['analysis']}): {sn['n_nmb_recorded_cases']} such case(s), n = {sn['n']}")

    if "after" in cmds:
        st, _ = _state(out)
        if st is None or st.get("status") != "completed":
            ap.error("the frozen S-ob scoring has not completed")
        frozen = P.load_frozen(out)
        Td, To = sdev(), sob()
        w = Writer(out / "sob", {**meta, "stage": P.AFTER_LABEL})
        P.after_frozen(Td, To, frozen, w, n_jobs=a.n_jobs, n_boot=a.n_boot, with_gb=not a.no_gb)
        if (out / "sdev" / "confirmatory" / "model_summary.csv").exists():
            w.text("after_frozen_scoring/readme_block5.md", P.readme_blocks(out / "sdev", out / "sob"))
        w.manifest("after_frozen_scoring/MANIFEST.json")
        _log(out, {"event": "after_frozen_scoring_written", "commit": g["commit"], "frozen_tag": a.frozen_tag})
        print(f"post-freeze S-ob analyses written to {out / 'sob' / 'after_frozen_scoring'}")

    if "series" in cmds:
        st, _ = _state(out)
        if st is None or st.get("status") != "completed":
            ap.error("the frozen S-ob scoring has not completed")
        if not (out / "sob" / "after_frozen_scoring" / "conformal.csv").exists():
            ap.error("run 'after' first: the interval half-widths come from its conformal.csv (SPEC §6.7)")
        frozen = P.load_frozen(out)
        sha = json.loads((out / "frozen" / "FROZEN.json").read_text())["models_sha256"]
        To = sob()
        if a.synthetic:
            from btc.models.synthetic import SyntheticSignals
            source = SyntheticSignals(To.cases, a.synthetic_seed)
        else:
            from btc.models.site_cases import RawSignals
            win = a.sob_dir / "sob_windows.parquet"
            if not win.exists():
                ap.error(f"missing window table {win.name} (scripts/build_windows.py --cohort sob)")
            try:
                source = RawSignals(To.cases, pd.read_parquet(win))
            except PermissionError as ex:
                ap.error(f"{ex} (series reads the S-ob heart-rate and pump tracks)")
        cases_dir = a.cases_dir or (out / "site_cases" if a.synthetic else DERIVED / "site")
        man = P.site_cases(frozen, To, source, out, cases_dir, {**meta, "frozen_models_sha256": sha},
                           synthetic=a.synthetic)
        _log(out, {"event": "site_cases_written", "commit": g["commit"], "frozen_tag": a.frozen_tag,
                   "n_files": len(man["files"]),
                   "cases_json_sha256": man["files"]["cases.json"]})
        print(f"site case files ({len(man['files']) - 1} series) written to {cases_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
