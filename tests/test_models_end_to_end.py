"""End-to-end runs on synthetic tables only (SPEC §9 step 5: no real data before the confirmatory run)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc.models.data import tables_from_frames
from btc.models.outputs import Writer
from btc.models.pipeline import NMB_ANALYSIS, NMB_DIR, confirmatory
from btc.models.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp, seed, wf=0.0, hr=0.0):
    T = tables_from_frames("sdev", make_synthetic(seed=seed, wf_signal=wf, hr_signal=hr)["sdev"])
    confirmatory(T, Writer(tmp / f"s{seed}_{wf}_{hr}", {"commit": "test", "data": "synthetic"}))
    return json.loads((tmp / f"s{seed}_{wf}_{hr}" / "confirmatory" / "summary.json").read_text())


def test_null_signal_gives_row_6_most_of_the_time(tmp_path):
    rows = [_run(tmp_path, s)["headline_row"] for s in (1, 2, 3, 4, 5)]
    assert sum(r == 6 for r in rows) >= 4, rows


def test_planted_waveform_signal_is_detected_and_placebo_is_not(tmp_path):
    s = _run(tmp_path, 2, wf=4.0)
    c = s["contrasts"]
    assert c["M4 vs M2"]["meets_rule"] and c["M4 vs M3"]["meets_rule"]
    assert not c["M4-P1 vs M2"]["meets_rule"] and not c["M3-P1 vs M2"]["meets_rule"]
    assert c["M4 vs M4-P1"]["delta_hi"] < 0
    assert s["decision"]["m4_placebo_valid"] is True
    assert s["headline_row"] == 4 and s["answer_line"].startswith("Yes, from the pleth waveform.")
    assert s["yardstick_all"]["placebo_guard_passed"] is True


def test_planted_heart_rate_signal_is_attributed_to_heart_rate(tmp_path):
    s = _run(tmp_path, 3, hr=4.0)
    assert s["contrasts"]["M3 vs M2"]["meets_rule"] and not s["contrasts"]["M3-P1 vs M2"]["meets_rule"]
    assert s["headline_row"] == 3


def _check_nmb_sensitivity_sdev(sdev):
    """SPEC §2.2 sensitivity in the S-dev secondary outputs: the ladder refitted on all analysable cases
    with its own folds; the confirmatory run used the analysed cases only."""
    conf = json.loads((sdev / "confirmatory" / "summary.json").read_text())
    sec = json.loads((sdev / "secondary" / "summary.json").read_text())
    nmb = set(conf["nmb_recorded_caseids"])
    assert len(nmb) == 4 and conf["n_analysed_cases"] == conf["n_analysable_cases"] - 4
    inc = sec["including_nmb"]
    assert inc["analysis"] == NMB_ANALYSIS and inc["n_analysed_cases"] == conf["n_analysable_cases"]
    assert sec["data_quality"]["n_analysed"] == conf["n_analysed_cases"]
    vc = pd.read_csv(sdev / "secondary" / "variants_contrasts.csv", skiprows=1)
    v = vc.loc[vc["analysis"] == NMB_ANALYSIS]
    assert sorted(zip(v["model"], v["comparator"])) == [("M3", "M2"), ("M4", "M2"), ("M4", "M3")]
    assert (v["n_patients"] == conf["N"] + 4).all()          # the 4 synthetic nmb cases enter here only
    assert NMB_ANALYSIS in set(pd.read_csv(sdev / "secondary" / "variants_summary.csv", skiprows=1)["analysis"])
    folds = pd.read_csv(sdev / "secondary" / f"{NMB_DIR}_folds.csv", skiprows=1)
    assert nmb <= set(folds["caseid"])
    assert not nmb & set(pd.read_csv(sdev / "confirmatory" / "folds.csv", skiprows=1)["caseid"])


def _script(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_scripts_end_to_end_deterministic_and_run_once(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        r = _script("scripts/run_confirmatory.py", "--synthetic", "--wf-signal", "2", "--with-secondary", "--no-gb",
                    "--out-dir", str(out), "--n-jobs", "2")
        assert r.returncode == 0, r.stderr
    ma = json.loads((a / "sdev" / "MANIFEST.json").read_text())["files"]
    mb = json.loads((b / "sdev" / "MANIFEST.json").read_text())["files"]
    assert ma == mb and len(ma) > 20                          # byte-identical outputs
    _check_nmb_sensitivity_sdev(a / "sdev")
    r = _script("scripts/run_frozen_scoring.py", "all", "--synthetic", "--wf-signal", "2", "--no-gb",
                "--out-dir", str(a), "--n-jobs", "2")
    assert r.returncode == 0, r.stderr
    s = json.loads((a / "sob" / "frozen_scoring" / "summary.json").read_text())
    # 21 synthetic S-ob cases: 1 not analysable, 1 with a neuromuscular blocker recorded (SPEC §2.2)
    assert (s["n_analysable_cases"], s["n_nmb_recorded_cases"], s["n_analysed_cases"]) == (20, 1, 19)
    assert s["sentence_2"].startswith("In the 19 cesarean cases (external, frozen models): ")
    sens = json.loads((a / "sob" / "frozen_scoring" / NMB_DIR / "summary.json").read_text())
    assert sens["analysis"] == NMB_ANALYSIS and sens["n"] == 20 and sens["n_nmb_recorded_cases"] == 1
    assert f"frozen_scoring/{NMB_DIR}/contrasts.csv" in json.loads(
        (a / "sob" / "frozen_scoring" / "MANIFEST.json").read_text())["files"]
    dq = json.loads((a / "sob" / "after_frozen_scoring" / "summary.json").read_text())["data_quality"]
    assert (dq["n_analysable"], dq["n_nmb_recorded"], dq["n_analysed"]) == (20, 1, 19)
    # per-patient public tables in place of the withheld per-forecast files, for S-dev and S-ob
    pub = tmp_path / "public"
    r = _script("scripts/export_public_tables.py", "--results", str(a), "--out", str(pub))
    assert r.returncode == 0, r.stderr
    for rel in ("sdev/confirmatory", "sob/frozen_scoring", f"sob/frozen_scoring/{NMB_DIR}"):
        man = json.loads((pub / rel / "public_tables.json").read_text())
        assert man["checks"] and all(man["checks"].values()), rel
    y = pd.read_csv(pub / "sob" / "frozen_scoring" / "per_patient_yardstick.csv", skiprows=1)
    assert {"rmse_T-ia", "rmse_LH-ia", "rmse_M2 2-ahead"} <= set(y.columns)
    w = pd.read_csv(pub / "sob" / "frozen_scoring" / NMB_DIR / "per_patient_primary_wide.csv", skiprows=1)
    assert int(w["nmb_recorded"].sum()) == 1 and int(w["entered"].sum()) == sens["n"]
    # a second scoring is refused, also with --rerun-after-crash
    for extra in ([], ["--rerun-after-crash"]):
        r = _script("scripts/run_frozen_scoring.py", "score", "--synthetic", "--wf-signal", "2", "--out-dir", str(a),
                    *extra)
        assert r.returncode == 2 and "refused" in r.stderr
    # a crashed run (started, not completed) needs --rerun-after-crash, which is logged
    state = a / "sob" / "frozen_scoring" / "STATE.json"
    state.write_text(json.dumps({"status": "started", "commit": "x"}))
    r = _script("scripts/run_frozen_scoring.py", "score", "--synthetic", "--wf-signal", "2", "--out-dir", str(a))
    assert r.returncode == 2
    r = _script("scripts/run_frozen_scoring.py", "score", "--synthetic", "--wf-signal", "2", "--out-dir", str(a),
                "--rerun-after-crash")
    assert r.returncode == 0, r.stderr
    log = [json.loads(x) for x in (a / "sob" / "frozen_scoring_log.jsonl").read_text().splitlines()]
    events = [e["event"] for e in log]
    assert events.count("completed") == 2 and "rerun_after_crash" in events and events.count("refused") == 3
    rr = [e for e in log if e["event"] == "rerun_after_crash"][0]
    assert rr["previous_commit"] == "x" and rr["code_changed"] is True


def test_real_mode_requires_feature_tag(tmp_path):
    r = _script("scripts/run_confirmatory.py", "--out-dir", str(tmp_path), "--data-dir", str(tmp_path))
    assert r.returncode != 0 and "--feature-tag is required" in r.stderr
