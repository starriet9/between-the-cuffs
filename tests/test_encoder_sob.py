"""The S-ob step of the encoder check (ENCODER_PLAN.md §7): refit against the frozen models, the one scoring,
its public table, the S-ob embedding/feature step and the script's guards. Synthetic tables and signals
only: no real S-ob (or S-dev) data is read.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from btc import encoder as E
from btc import paths as bpaths
from btc import provenance
from btc import waveform as W
from btc.models import encoder_sob as MS
from btc.models import pipeline as P
from btc.models.constants import N_BOOT, SEED
from btc.models.data import tables_from_frames
from btc.models.outputs import Writer
from btc.models.synthetic import make_synthetic
from enc_synth import enc_cols, encoder_table
from wf_synth import Beat

ROOT = Path(__file__).resolve().parents[1]
DIM = 6
COLS = enc_cols(DIM)
CLEAN = {"commit": "c" * 40, "describe": "x", "state": "clean", "dirty": False, "status": [], "tags_at_head": []}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic S-dev and S-ob tables, a freeze and one frozen S-ob scoring in a results layout."""
    syn = make_synthetic(2, wf_signal=4.0)
    Td, To = tables_from_frames("sdev", syn["sdev"]), tables_from_frames("sob", syn["sob"])
    results = tmp_path_factory.mktemp("sob") / "results"
    meta = {"commit": "f" * 40, "describe": "test", "data": "synthetic(seed=2, wf_signal=4.0)"}
    wf = Writer(results, {**meta, "stage": "freeze"})
    frozen = P.freeze(Td, wf, n_jobs=2)
    wf.manifest("frozen/MANIFEST.json")
    ws = Writer(results / "sob", {**meta, "stage": "frozen S-ob scoring", "inputs_sha256": To.sha256})
    P.score_sob(frozen, To, ws)
    ws.manifest("frozen_scoring/MANIFEST.json")
    (results / "sob" / "frozen_scoring" / "STATE.json").write_text(json.dumps({"status": "completed"}))
    conf = results / "sdev" / "confirmatory"
    conf.mkdir(parents=True)
    (conf / "summary.json").write_text(json.dumps({"meta": {"inputs_sha256": Td.sha256},
                                                   "m2_structure": {"form": "a", "structure": "pooled"}}))
    return SimpleNamespace(Td=Td, To=To, results=results, frozen=frozen,
                           fsum=json.loads((results / "sob" / "frozen_scoring" / "summary.json").read_text()),
                           stored=pd.read_csv(results / "sob" / "frozen_scoring" / "predictions_primary.csv", skiprows=1))


@pytest.fixture(scope="module")
def fits(world):
    return MS.refit_encoder(world.Td, encoder_table(world.Td, DIM, planted=True, seed=2), "a", "pooled", cols=COLS)


def test_refit_equals_the_frozen_models(world, fits):
    eq, bad = MS.frozen_equality(fits, world.frozen)
    assert bad == [] and len(eq) == 4
    assert all(v["equal"] and v["penalty_equal"] for v in eq.values())
    assert all(all(v.get("parts", {"x": True}).values()) for v in eq.values())


def test_frozen_equality_reports_a_difference(world, fits):
    other = copy.deepcopy(world.frozen)
    st = other["chains"]["main"].nodes["pooled"].stages[0]          # M2 (never dropped)
    st.beta = st.beta.copy()
    st.beta[0] += 1e-9
    eq, bad = MS.frozen_equality(fits, other)
    assert bad == ["M2 (refit chain enc vs frozen chain main)"]
    assert eq[bad[0]]["max_abs_coefficient_diff"] == pytest.approx(1e-9, rel=1e-3)


def test_sob_scoring_and_public_table(world, fits, tmp_path):
    out_rel = "sob/after_frozen_scoring/exploratory_encoder"
    results = tmp_path / "results"
    w = Writer(results / out_rel, {"commit": "t" * 40, "data": "synthetic", "stage": MS.SOB_LABEL})
    tab = encoder_table(world.To, DIM, planted=True, seed=3)
    s = MS.encoder_sob(world.To, tab, world.stored, world.fsum, fits, world.frozen, w, n_boot=N_BOOT, cols=COLS)
    w.manifest()
    assert s["n"] == world.fsum["n"] and s["label"] == MS.SOB_LABEL
    assert all(v["bit_identical"] for v in s["reproduction_of_frozen_forecasts"].values())
    assert s["fallback"]["bit_identical"] and s["refit"]["stages_not_equal_to_frozen"] == []
    assert set(s["contrasts"]) == {"M4-enc vs M2", "M4-enc vs M3", "M4-enc vs M4", "M4-enc-P1 vs M2",
                                   "M4-enc vs M4-enc-P1", "M4 vs M2"}
    assert s["contrasts"]["M4 vs M2"].delta == pytest.approx(world.fsum["contrasts"]["M4 vs M2"]["delta"], abs=1e-12)
    assert s["one_line_summary"].startswith("Exploratory, computed after the frozen scoring: in the ")
    ex = _script("export_public_tables")
    man = ex.export_stage(results, out_rel, results, {"commit": "t" * 40, "describe": "test"})
    assert man["checks"]["entering_patients_forecasts_and_delta_equal_to_summary"]
    wide = pd.read_csv(results / out_rel / "per_patient_primary_wide.csv", skiprows=1)
    assert not {"y_map", "last_map"} & set(wide.columns)
    e = wide.loc[wide["entered"]].sort_values("caseid")
    from btc.models.bootstrap import SubjectBootstrap, contrast
    bs = SubjectBootstrap(e["caseid"].to_numpy(), e["subjectid"].to_numpy(), N_BOOT, SEED)
    c = contrast(bs, e["rmse_M4-enc"], e["rmse_M2"], "M4-enc", "M2")
    assert c.rr_lo == pytest.approx(s["contrasts"]["M4-enc vs M2"].rr_lo, abs=1e-12)


def test_scoring_refuses_other_stored_forecasts(world, fits, tmp_path):
    bad = world.stored.copy()
    bad.loc[3, "pred_M3"] += 1e-6
    w = Writer(tmp_path / "x", {"commit": "t", "data": "synthetic"})
    with pytest.raises(RuntimeError, match="refitted stages differ"):
        MS.encoder_sob(world.To, encoder_table(world.To, DIM, seed=3), bad, world.fsum, fits, world.frozen, w,
                       n_boot=50, cols=COLS)


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------

def _ok(m, out, **kw):
    args = dict(g=CLEAN, plan={"committed_at_head": True}, out=out, rerun=False, state={"status": "completed"},
                tables_sha={"a": 1}, frozen_meta_sha={"a": 1}, dev_sha={"b": 2}, conf_sha={"b": 2}, stored_ok=True,
                feat_ok=True, derived_exist=[])
    args.update(kw)
    return m.sob_problems(**args)


def test_sob_guards(tmp_path, monkeypatch):
    m = _script("run_encoder_sob")
    out = tmp_path / "enc"
    monkeypatch.setenv("BTC_FROZEN_SCORING", "1")
    assert _ok(m, out) == []
    for kw, word in ((dict(g={**CLEAN, "state": "dirty"}), "not clean"), (dict(plan={}), "not committed"),
                     (dict(state=None), "not completed"), (dict(tables_sha={"a": 2}), "frozen scoring"),
                     (dict(dev_sha={"b": 3}), "confirmatory"), (dict(stored_ok=False), "predictions_primary"),
                     (dict(feat_ok=False), "S-dev encoder features"), (dict(derived_exist=["x"]), "runs once")):
        assert any(word in p for p in _ok(m, out, **kw)), word
    monkeypatch.delenv("BTC_FROZEN_SCORING")
    assert any("BTC_FROZEN_SCORING=1" in p for p in _ok(m, out))
    monkeypatch.setenv("BTC_FROZEN_SCORING", "1")
    out.mkdir()
    assert any("runs once" in p for p in _ok(m, out))
    assert _ok(m, out, rerun=True) == []
    (out / "summary.json").write_text("{}")
    assert any("completed already" in p for p in _ok(m, out, rerun=True))


def test_history_guard(monkeypatch):
    m = _script("run_encoder_sob")
    calls = {"ancestor": 0, "sdev_dir": 0, "plan": 0, "names": "results/x.csv\nbtc/models/encoder_sob.py"}

    def fake(*a):
        if a[0] == "merge-base":
            return SimpleNamespace(returncode=calls["ancestor"], stdout="")
        if a[0] == "diff" and a[1] == "--quiet":
            return SimpleNamespace(returncode=calls["plan"] if a[-1] == "ENCODER_PLAN.md" else calls["sdev_dir"], stdout="")
        return SimpleNamespace(returncode=0, stdout=calls["names"])
    monkeypatch.setattr(m, "_git", fake)
    assert m.history_problems("a" * 40) == []
    calls["names"] += "\nbtc/encoder.py"
    assert any("other than the S-ob additions" in p for p in m.history_problems("a" * 40))
    calls.update(names="", plan=1, sdev_dir=1)
    probs = m.history_problems("a" * 40)
    assert any("ENCODER_PLAN.md changed" in p for p in probs) and any("exploratory_encoder" in p for p in probs)
    calls["ancestor"] = 1
    assert any("not an ancestor" in p for p in m.history_problems("a" * 40))
    assert m.SDEV_RESULT_COMMIT.startswith("0626dd2") and len(m.SDEV_RESULT_COMMIT) == 40


# ----------------------------------------------------------------------------
# the S-ob embedding and feature step, on two synthetic cases
# ----------------------------------------------------------------------------

def test_build_sob_tables(tmp_path, monkeypatch):
    from test_encoder_features import READS, _signals, _wf_builder, stub512
    m = _script("run_encoder_sob")
    der = tmp_path / "derived"
    der.mkdir()
    monkeypatch.setattr(bpaths, "DERIVED", der)
    monkeypatch.setenv("BTC_FROZEN_SCORING", "1")
    sig = {11: _signals(900, Beat(), 0.62, 1.0, 1), 22: _signals(900, Beat(T=0.75, N=0.18), 0.6, 0.4, 2),
           33: _signals(300, Beat(T=0.7), 0.6, 0.4, 3)}
    win = pd.concat([W.case_windows(c, s["pleth"], s["ecg"]) for c, s in sig.items()], ignore_index=True)
    meta = {"git": CLEAN}
    provenance.write_parquet(win, der / "sob_windows.parquet", meta)
    (der / "sob_windows_log.json").write_text(json.dumps({"cases": [{"caseid": c, "pleth": "ok", "ecg": "ok"}
                                                                    for c in sig]}))
    t_of = {(11, i + 1): r for i, r in enumerate(READS)}
    issues = pd.DataFrame([{"caseid": 11, "window": "primary", "issue_n": n, "t_issue": READS[n - 1],
                            "t_prev": READS[n - 2], "k": 1} for n in range(3, len(READS) + 1)])
    cases = pd.DataFrame({"caseid": [11, 22, 33], "donor_caseid": [22.0, 11.0, 11.0], "pleth_status": "ok",
                          "ecg_status": "ok"})
    reads = pd.DataFrame({"caseid": 11, "n": np.arange(1, len(READS) + 1, dtype=float), "t": READS, "valid": True})
    for name, df in (("instances", issues), ("readings", reads), ("cases", cases)):
        provenance.write_parquet(df, der / f"sob_{name}.parquet", meta)
    wf = _wf_builder().build(issues[["caseid", "window", "issue_n", "t_issue", "t_prev"]], t_of, cases, win)
    provenance.write_parquet(wf, der / "sob_features_wf.parquet", meta)
    monkeypatch.setattr(m.E, "check_weights", lambda p: {"size": 1, "md5": "m", "sha256": "w"})
    monkeypatch.setattr(m.E, "configure_torch", lambda: {})
    monkeypatch.setattr(m.E, "runtime_info", lambda: {"torch": "stub"})
    monkeypatch.setattr(m.E, "load_model", lambda p: (None, {}))
    monkeypatch.setattr(m.E, "embed", lambda model, X: stub512(X))
    read = []
    monkeypatch.setattr(m.W, "load_case", lambda c, cohort: read.append((c, cohort)) or {"pleth": sig[c]["pleth"]})
    T_ob = SimpleNamespace(analysed=pd.DataFrame({"caseid": [11]}))
    table, rec = m.build_sob_tables(T_ob, 1, CLEAN, {"sha256": "p"})
    assert sorted(read) == [(11, "sob"), (22, "sob")]             # the analysed case and its donor only
    assert rec["donors"] == [22] and rec["n_cases_embedded"] == 2
    assert rec["same_windows_as_m4"]["rows_compared"] == 2 * len(issues)
    assert len(table) == 2 * len(issues) and len(table.columns) == 11 + 1024
    log = json.loads((der / "sob_enc_window_embeddings_log.json").read_text())
    assert log["raw_files_read"] == ["data/raw/sob/11.npz", "data/raw/sob/22.npz"]
    assert log["preprocessing_note"] == E.FILTER_NOTE


# ----------------------------------------------------------------------------
# the script end to end (synthetic; the embedding step replaced by a synthetic encoder table)
# ----------------------------------------------------------------------------

def test_script_end_to_end(world, tmp_path, monkeypatch):
    import shutil
    m = _script("run_encoder_sob")
    results = tmp_path / "results"
    shutil.copytree(world.results, results)
    der = tmp_path / "derived"
    der.mkdir()
    dev_tab = encoder_table(world.Td, E.EMB_DIM, planted=True, seed=2)
    dev_tab.to_parquet(der / "sdev_enc_features.parquet")
    (der / "sdev_enc_features_log.json").write_text(json.dumps(
        {"features_sha256": E.file_digests(der / "sdev_enc_features.parquet")["sha256"], "embeddings_sha256": "E"}))
    ob_tab = encoder_table(world.To, E.EMB_DIM, planted=True, seed=3)
    rec = {"n_cases_embedded": 20, "n_windows_embedded": 999, "donors": [1], "same_windows_as_m4": {"rows_compared": 1},
           "dom_freq_check": {"share_agree": 1.0}, "embedding_runtime": {"torch": "stub"}, "weights": {"sha256": "w"}}
    monkeypatch.setattr(bpaths, "DERIVED", der)
    for k, v in {"RESULTS": results, "FROZEN_DIR": results / "sob" / "frozen_scoring",
                 "CONF": results / "sdev" / "confirmatory"}.items():
        monkeypatch.setattr(m, k, v)
    monkeypatch.setattr(m, "git_meta", lambda: dict(CLEAN))
    monkeypatch.setattr(m.E, "plan_record", lambda: {"file": "ENCODER_PLAN.md", "sha256": "p", "committed_at_head": True,
                                                     "last_commit": "a" * 40})
    monkeypatch.setattr(m, "load_tables", lambda d, cohort: world.To if cohort == "sob" else world.Td)
    monkeypatch.setattr(m, "history_problems", lambda c: [])
    built = {"n": 0}

    def fake_build(T_ob, jobs, g, plan):
        built["n"] += 1
        return ob_tab, rec
    monkeypatch.setattr(m, "build_sob_tables", fake_build)
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    assert m.main([]) == 2 and built["n"] == 0                      # S-ob switch not set
    monkeypatch.setenv("BTC_FROZEN_SCORING", "1")
    assert m.main([]) == 0
    out = results / "sob" / "after_frozen_scoring" / "exploratory_encoder"
    s = json.loads((out / "summary.json").read_text())
    assert s["n"] == world.fsum["n"] and s["refit"]["stages_not_equal_to_frozen"] == []
    assert len(s["departures"]) == 3 and "byte-identical" in s["departures"][0]
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["meta"]["written_after_sdev_result"]["sdev_result_commit"].startswith("0626dd2")
    assert {"summary.json", "summary.md", "predictions_primary.csv", "refit_penalties.csv"} <= set(man["files"])
    assert (out / "per_patient_primary_wide.csv").exists() and (out / "public_tables.json").exists()
    assert "Departures" in (out / "summary.md").read_text()
    assert m.main([]) == 2                                          # once
    assert m.main(["--rerun-after-crash"]) == 2                     # completed: no rerun
    (out / "summary.json").unlink()                                 # as after a crash
    assert m.main(["--rerun-after-crash"]) == 0 and (der / "sob_enc_crashed_output_1").exists()
    ev = [json.loads(l)["event"] for l in (der / "sob_enc_run_log.jsonl").read_text().splitlines()]
    assert ev.count("completed") == 2 and "rerun_after_crash" in ev
