#!/usr/bin/env python3
"""Confirmatory S-dev cross-validation run (SPEC §9 steps 4-5; PREREG §4, §6).

Fits, by S-dev cross-validation at the frozen feature tag: the M2 structure (§4.1), M2-M4,
M3-P1, M4-P1 and the yardstick models L, U, T, T-P1 (§3), on the analysed cases (SPEC §2.2:
analysable, no neuromuscular blocker recorded). Writes Sentences 1 and 3 and the plain
answer line (§6.3) to `<out>/sdev/confirmatory/`. With `--with-secondary` it then runs the S-dev
secondary analyses and the gradient-boosting check (§6.6-§6.8) into `<out>/sdev/secondary/`.

"The first complete S-dev cross-validation run of all of these at the frozen feature tag is the
confirmatory result" (§9 step 5). On real data this script therefore refuses to run unless HEAD
carries the given feature tag and no tracked file is modified, and it refuses to overwrite an
existing confirmatory output. Every result file records the commit (§10). The run is
deterministic.

`--synthetic` generates synthetic tables (btc/models/synthetic.py) and never reads `--data-dir`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc.models.constants import N_BOOT  # noqa: E402
from btc.models.data import load_tables, tables_from_frames  # noqa: E402
from btc.models.outputs import Writer, git_meta  # noqa: E402
from btc.models.pipeline import confirmatory, secondary_sdev  # noqa: E402
from btc.paths import DERIVED, ROOT  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic tables (tests only)")
    ap.add_argument("--synthetic-seed", type=int, default=1)
    ap.add_argument("--wf-signal", type=float, default=0.0, help="synthetic planted waveform signal")
    ap.add_argument("--hr-signal", type=float, default=0.0, help="synthetic planted heart-rate signal")
    ap.add_argument("--data-dir", type=Path, default=DERIVED, help="folder with sdev_*.parquet (btc/SCHEMA.md)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output root (default: results/ for real data, results_synthetic/ with --synthetic)")
    ap.add_argument("--feature-tag", default=None, help="the frozen feature tag (required on real data)")
    ap.add_argument("--with-secondary", action="store_true", help="also run the S-dev secondary analyses")
    ap.add_argument("--no-gb", action="store_true", help="skip the gradient-boosting check (synthetic tests only)")
    ap.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help=argparse.SUPPRESS)   # tests only
    a = ap.parse_args(argv)

    g = git_meta()
    if a.synthetic:
        from btc.models.synthetic import make_synthetic
        out = a.out_dir or (ROOT / "results_synthetic")
        tables = tables_from_frames("sdev", make_synthetic(a.synthetic_seed, a.wf_signal, a.hr_signal)["sdev"])
        data = f"synthetic(seed={a.synthetic_seed}, wf_signal={a.wf_signal}, hr_signal={a.hr_signal})"
    else:
        out = a.out_dir or (ROOT / "results")
        if a.n_boot != N_BOOT or a.no_gb:
            ap.error("--n-boot and --no-gb are for synthetic tests only (SPEC §6.2, §6.8)")
        if not a.feature_tag:
            ap.error("--feature-tag is required on real data (SPEC §9 step 3)")
        if a.feature_tag not in g["tags_at_head"] or g["tracked_files_modified"]:
            ap.error(f"HEAD must carry tag {a.feature_tag!r} with no modified tracked file (SPEC §9 step 5)")
        if (out / "sdev" / "confirmatory" / "summary.json").exists():
            ap.error("a confirmatory result already exists; a rerun is a deviation (SPEC §9 step 5): "
                     "write it to another --out-dir and report both")
        tables = load_tables(a.data_dir, "sdev")
        data = f"VitalDB derived tables ({a.data_dir.name}/sdev_*.parquet)"
    if a.n_boot != N_BOOT:
        data += f", n_boot={a.n_boot}"
    meta = {**g, "feature_tag": a.feature_tag, "data": data, "inputs_sha256": tables.sha256,
            "stage": "sdev"}
    w = Writer(out / "sdev", meta)
    t0 = time.time()
    ctx = confirmatory(tables, w, n_jobs=a.n_jobs, n_boot=a.n_boot)
    t1 = time.time()
    if a.with_secondary:
        secondary_sdev(ctx, w, n_jobs=a.n_jobs, n_boot=a.n_boot, with_gb=not a.no_gb)
    t2 = time.time()
    w.manifest()
    s = json.loads((out / "sdev" / "confirmatory" / "summary.json").read_text())
    print(f"confirmatory: {t1 - t0:.1f} s; secondary: {t2 - t1:.1f} s; outputs in {out / 'sdev'}")
    print(f"M2 structure: form {ctx.form}, {ctx.structure}; headline row {s['headline_row']}")
    print("Answer line:", s["answer_line"])
    print("Sentence 1:", s["sentence_1"])
    print("Sentence 3:", s["sentence_3"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
