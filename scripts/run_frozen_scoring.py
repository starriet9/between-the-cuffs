#!/usr/bin/env python3
"""Freeze the models and score S-ob once (SPEC §9 steps 6-8; PREREG §4, §6, §9).

Commands:
  freeze  refit every confirmatory model on all S-dev (penalties chosen with the outer folds,
          §4.5) and write `<out>/frozen/` (pickled models, SHA-256, coefficients). No S-ob access.
  score   the one frozen S-ob scoring (§9 step 7): loads the frozen models (SHA-256 checked),
          scores every one of them on the analysed S-ob cases (SPEC §2.2), writes Sentence 2 and the
          cesarean line to `<out>/sob/frozen_scoring/`, and in the same run scores them on all
          analysable S-ob cases, including those with a recorded neuromuscular blocker
          (`frozen_scoring/sensitivity_including_nmb/`, SPEC §2.2). It refuses to run a second time. A rerun after a crash
          (a run that started and did not complete) needs `--rerun-after-crash` and is logged with
          both commits; a code change between the two runs must be reported with its diff (§9).
  after   S-ob analyses after the frozen scoring (§9 step 8), labelled as such, with models fitted
          on S-dev alone; writes `<out>/sob/after_frozen_scoring/` and the README table block.
  all     freeze, score and after in one go (synthetic tests only).

Every attempt is appended to `<out>/sob/frozen_scoring_log.jsonl`. `--synthetic` generates
synthetic S-dev and S-ob tables and never reads `--data-dir` or `--sob-dir`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc.models import pipeline as P  # noqa: E402
from btc.models.constants import N_BOOT  # noqa: E402
from btc.models.data import load_tables, tables_from_frames  # noqa: E402
from btc.models.outputs import Writer, git_meta  # noqa: E402
from btc.paths import DERIVED, ROOT  # noqa: E402


def _log(out: Path, entry: dict):
    p = out / "sob" / "frozen_scoring_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry}
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _state(out: Path):
    p = out / "sob" / "frozen_scoring" / "STATE.json"
    return (json.loads(p.read_text()) if p.exists() else None), p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["freeze", "score", "after", "all"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-seed", type=int, default=1)
    ap.add_argument("--wf-signal", type=float, default=0.0)
    ap.add_argument("--hr-signal", type=float, default=0.0)
    ap.add_argument("--data-dir", type=Path, default=DERIVED, help="folder with sdev_*.parquet")
    ap.add_argument("--sob-dir", type=Path, default=DERIVED, help="folder with sob_*.parquet")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rerun-after-crash", action="store_true")
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
        Td = sdev()
        conf = out / "sdev" / "confirmatory" / "summary.json"
        exp = None
        if conf.exists():
            m = json.loads(conf.read_text())["m2_structure"]
            exp = (m["form"], m["structure"])
        elif not a.synthetic:
            ap.error("no confirmatory result found; run scripts/run_confirmatory.py first (SPEC §9)")
        w = Writer(out, {**meta, "stage": "freeze", "inputs_sha256": Td.sha256})
        P.freeze(Td, w, n_jobs=a.n_jobs, expected_structure=exp)
        w.manifest("frozen/MANIFEST.json")
        print(f"frozen models written to {out / 'frozen'}")

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
            changed = st.get("commit") != g["commit"] or bool(g["tracked_files_modified"])
            _log(out, {"event": "rerun_after_crash", "previous_commit": st.get("commit"), "commit": g["commit"],
                       "code_changed": changed})
            if changed:
                print("warning: code differs from the crashed run; report the change with its diff (SPEC §9)",
                      file=sys.stderr)
        frozen = P.load_frozen(out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"status": "started", "commit": g["commit"], "rerun_after_crash": rerun}) + "\n")
        _log(out, {"event": "started", "commit": g["commit"], "describe": g["describe"], "data": data,
                   "rerun_after_crash": rerun})
        try:
            To = sob()
            w = Writer(out / "sob", {**meta, "stage": "frozen S-ob scoring", "inputs_sha256": To.sha256,
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
        _log(out, {"event": "after_frozen_scoring_written", "commit": g["commit"]})
        print(f"post-freeze S-ob analyses written to {out / 'sob' / 'after_frozen_scoring'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
