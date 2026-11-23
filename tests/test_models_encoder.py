"""The M4-enc / M4-enc-P1 chain, the build check, the S-dev run and its guards (ENCODER_PLAN.md §3-§6, §11).

Synthetic derived tables (`btc.models.synthetic`) with synthetic encoder columns (`tests/enc_synth.py`);
most tests use 6 embedding dimensions for speed, one checks the full-width design. The real-data guards of
`scripts/run_encoder_check.py` and of the two build scripts are run with stand-in git states; no derived
table, signal or network is read.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc import encoder as E
from btc.models import encoder as ME
from btc.models import pipeline as P
from btc.models.analyses import Scored, ladder, run_cv
from btc.models.bootstrap import Contrast
from btc.models.constants import SEED
from btc.models.data import build_frame, tables_from_frames
from btc.models.design import Cols
from btc.models.folds import make_fold_plan
from btc.models.outputs import Writer
from btc.models.ridge import hr_block
from btc.models.synthetic import make_synthetic
from enc_synth import enc_cols, encoder_table

ROOT = Path(__file__).resolve().parents[1]
DIM = 6
COLS = enc_cols(DIM)
CLEAN = {"commit": "c" * 40, "describe": "x", "state": "clean", "dirty": False, "status": [], "tags_at_head": []}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _runs(T, tab, n_jobs=2):
    frame = ME.attach_encoder(build_frame(T), tab, COLS)
    F = Cols(frame)
    plan = make_fold_plan(T.analysed, SEED)
    cv = run_cv({"enc": (F, ME.enc_ladder("a", "pooled", cols=COLS), plan),
                 "enc_p1": (F, ME.enc_ladder("a", "pooled", "p1", cols=COLS), plan),
                 "main": (F, ladder("a", "pooled"), plan),
                 "p1": (F, ladder("a", "pooled", "p1", names=("M2", "M3-P1", "M4-P1")), plan)}, n_jobs)
    return frame, cv


@pytest.fixture(scope="module")
def planted():
    T = tables_from_frames("sdev", make_synthetic(2, wf_signal=4.0)["sdev"])
    tab = encoder_table(T, DIM, planted=True, seed=2)
    return (T, tab, *_runs(T, tab))


@pytest.fixture(scope="module")
def null():
    T = tables_from_frames("sdev", make_synthetic(1)["sdev"])
    tab = encoder_table(T, DIM, seed=1)
    return (T, tab, *_runs(T, tab))


def test_chain_layout():
    spec = ME.enc_ladder("a", "pooled")
    assert spec.stages == ("M2", "M3", "M4-enc") and spec.structure == "pooled"
    hr, enc = spec.blocks
    assert hr == hr_block("real") and enc.requires == "wf_available@real" and enc.hcol == "h_min"
    assert len(enc.cols) == 1024 and enc.cols[0] == "enc_lvl_000@real" and enc.cols[-1] == "enc_chg_511@real"
    p1 = ME.enc_ladder("a", "pooled", "p1")
    assert p1.stages == ("M2", "M3-P1", "M4-enc-P1") and all(c.endswith("@p1") for b in p1.blocks for c in b.cols)
    assert p1.blocks[1].requires == "wf_available@p1"


@pytest.mark.parametrize("which", ["planted", "null"])
def test_chain_reproduces_m2_and_m3(which, request):
    """§3 "Check": the M2 and M3 stages re-derived in the encoder chain equal those of the plain ladder."""
    T, tab, frame, cv = request.getfixturevalue(which)
    assert np.array_equal(cv["enc"].stage("M2"), cv["main"].stage("M2"))
    assert np.array_equal(cv["enc"].stage("M3"), cv["main"].stage("M3"))
    assert np.array_equal(cv["enc_p1"].stage("M2"), cv["p1"].stage("M2"))
    assert np.array_equal(cv["enc_p1"].stage("M3-P1"), cv["p1"].stage("M3-P1"))
    for a, b in zip(cv["enc"].lam, cv["main"].lam):
        assert a["pooled"][:2] == b["pooled"][:2]                       # same cuff and heart-rate penalties


@pytest.mark.parametrize("which", ["planted", "null"])
def test_fallback_is_m4_fallback(which, request):
    """§4: with fewer than 2 usable windows M4-enc returns the M3 forecast, exactly M4's fallback rows."""
    T, tab, frame, cv = request.getfixturevalue(which)
    off = ~frame["wf_available@real"].to_numpy(bool)
    assert off.sum() > 20
    e4, m3, m4 = cv["enc"].stage("M4-enc"), cv["enc"].stage("M3"), cv["main"].stage("M4")
    assert np.array_equal(e4[off], m3[off]) and np.array_equal(m4[off], m3[off])
    offp = ~frame["wf_available@p1"].to_numpy(bool)
    assert np.array_equal(cv["enc_p1"].stage("M4-enc-P1")[offp], cv["enc_p1"].stage("M3-P1")[offp])
    # a fold in which the encoder block was dropped returns M3 on all its rows
    for j, lam in enumerate(cv["enc"].lam):
        if lam["pooled"][2] is None:
            te = cv["enc"].fold == j
            assert np.array_equal(e4[te], m3[te])


def test_planted_signal_is_found_and_placebo_is_not(planted):
    T, tab, frame, cv = planted
    preds = {"M2": cv["main"].stage("M2"), "M3": cv["main"].stage("M3"), "M4-enc": cv["enc"].stage("M4-enc"),
             "M4-enc-P1": cv["enc_p1"].stage("M4-enc-P1")}
    sc = Scored(frame, preds, n_boot=300)
    c, p, d = sc.contrast("M4-enc", "M2"), sc.contrast("M4-enc-P1", "M2"), sc.contrast("M4-enc", "M4-enc-P1")
    assert c.meets_rule and not p.meets_rule and d.delta_hi < 0
    assert ME.one_line(c, p, d, False).startswith(ME.LEAD + " It met the rule: ")
    assert all(l["pooled"][2] is not None for l in cv["enc"].lam)


def test_attach_requires_the_same_windows(null):
    T, tab = null[0], null[1]
    bad = tab.copy()
    i = bad.index[(bad["variant"] == "real") & bad["enc_available"]][0]
    bad.loc[i, "enc_available"] = False
    with pytest.raises(ValueError):
        ME.attach_encoder(build_frame(T), bad, COLS)


def test_full_width_design(null):
    """The design of M4-enc with all 1,024 columns: features, x h columns and merged missing-indicators."""
    T = null[0]
    frame = ME.attach_encoder(build_frame(T), encoder_table(T, E.EMB_DIM, seed=3))
    F = Cols(frame)
    spec = ME.enc_ladder("a", "pooled")
    rows = np.flatnonzero(frame["wf_available@real"].to_numpy(bool))[:900]
    d = spec.design(list(spec.blocks), True).fit(F, rows)
    Z = d.transform(F, rows)
    enc_names = [n for n, b in zip(d.names, d.block_of) if b == 2]
    assert sum(not n.startswith("miss:") and not n.endswith("*h") for n in enc_names) == 1024
    assert sum(n.endswith("*h") for n in enc_names) == 1024
    assert 1 <= sum(n.startswith("miss:") for n in enc_names) <= 2   # one indicator per missingness pattern
    assert Z.shape == (900, len(d.names)) and np.all(np.isfinite(Z))
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)


def test_one_line_wording():
    def con(rr, lo, hi, dlo, dhi):
        return Contrast("A", "B", 200, 8.0, 8.0, 0.0, dlo, dhi, rr, lo, hi)
    no = con(0.01, -0.02, 0.04, -0.05, 0.02)
    s = ME.one_line(no, no, no, True)
    assert s == (ME.LEAD + " It did not meet the rule either: 1.0% against the cuff-only model "
                 "(95% CI −2.0% to 4.0%). The cross-validation dropped the embedding block in every fold.")
    assert ME.one_line(no, no, no, False).endswith("(95% CI −2.0% to 4.0%).")
    yes, p1, gap = con(0.08, 0.06, 0.10, -0.9, -0.5), con(0.001, -0.01, 0.01, -0.1, 0.1), con(0.07, 0.05, 0.09, -0.8, -0.4)
    assert ME.one_line(yes, p1, gap, False) == (
        ME.LEAD + " It met the rule: 8.0% against the cuff-only model (95% CI 6.0% to 10.0%; placebo inputs 0.1%). "
        "The encoder was pretrained on unlabelled VitalDB pleth, the check is exploratory, and the registered "
        "answer is unchanged.")
    assert ME.one_line(yes, yes, gap, False) == \
        ME.LEAD + " It met the rule (8.0%), but its placebo control failed, so no claim is made from it."


def test_build_check(null):
    T = null[0]
    plan = make_fold_plan(T.analysed, SEED)
    rng = np.random.default_rng(0)
    ids = np.repeat(plan.caseids, 30)
    win = pd.DataFrame({"caseid": ids, "w": np.tile(np.arange(30), len(plan.caseids)), "usable": True,
                        "pleth_rate": rng.normal(75, 12, len(ids))})
    # windows of a case outside the analysed set are embedded but not used
    win = pd.concat([win, pd.DataFrame({"caseid": [999999] * 3, "w": [0, 1, 2], "usable": True,
                                        "pleth_rate": [70.0, 71.0, 72.0]})], ignore_index=True)
    index = win[["caseid", "w"]].assign(row=np.arange(len(win)))
    noise = rng.standard_normal((len(win), 512)).astype(np.float32)
    good = noise.copy()
    good[:, :5] += (win["pleth_rate"].to_numpy()[:, None] - 75) / 12 * 2.0
    g = ME.build_check(good, index, win, plan)
    assert g["passed"] and g["r2_out_of_fold"] > 0.8 and g["n_windows"] == 30 * len(plan.caseids)
    assert g["n_cases"] == len(plan.caseids) and len(g["per_fold"]) == 5
    b = ME.build_check(noise, index, win, plan)
    assert not b["passed"] and b["r2_out_of_fold"] < 0.1
    with pytest.raises(ValueError):
        ME.build_check(good, index, win.assign(usable=False), plan)


# ----------------------------------------------------------------------------
# The S-dev run on synthetic tables, its public table, and the integrity checks
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run(tmp_path_factory, planted):
    T, tab = planted[0], planted[1]
    root = tmp_path_factory.mktemp("enc") / "results"
    meta = {"commit": "t" * 40, "describe": "test", "data": "synthetic(seed=2, wf_signal=4.0)", "stage": "sdev",
            "inputs_sha256": T.sha256}
    cw = Writer(root / "sdev", meta)
    P.confirmatory(T, cw, n_jobs=2, n_boot=200)
    cw.manifest()
    conf = json.loads((root / "sdev" / "confirmatory" / "summary.json").read_text())
    stored = pd.read_csv(root / "sdev" / "confirmatory" / "predictions_primary.csv", skiprows=1)
    folds = pd.read_csv(root / "sdev" / "confirmatory" / "folds.csv", skiprows=1)
    w = Writer(root / "sdev" / "exploratory_encoder", {**meta, "stage": "exploratory encoder check"})
    s = ME.encoder_sdev(T, tab, stored, conf, w, n_jobs=2, n_boot=200, cols=COLS, conf_folds=folds)
    w.manifest()
    return root, conf, stored, s


def test_sdev_run_outputs(run):
    root, conf, stored, s = run
    out = root / "sdev" / "exploratory_encoder"
    assert s["N"] == conf["N"] and s["m2_structure"] == {"form": "a", "structure": "pooled"}
    assert all(v["close"] and v["bit_identical"] for v in s["reproduction_of_confirmatory_stages"].values())
    assert s["fallback"]["bit_identical_to_M4"] and s["fallback"]["n_forecasts"] > 20
    c = s["contrasts"]
    assert set(c) == {"M4-enc vs M2", "M4-enc vs M3", "M4-enc vs M4", "M4-enc-P1 vs M2", "M4-enc vs M4-enc-P1",
                      "M4 vs M2"}
    assert c["M4 vs M2"].delta == pytest.approx(conf["contrasts"]["M4 vs M2"]["delta"], abs=1e-12)
    assert s["meets_rule_with_placebo"] and s["one_line_summary"].startswith(ME.LEAD)
    assert s["encoder_block_outer_folds"]["M4-enc"]["n_folds"] == 5
    for f in ("summary.json", "contrasts.csv", "model_summary.csv", "penalties.csv", "dropped_blocks_outer_folds.csv",
              "per_patient_primary.csv", "predictions_primary.csv", "MANIFEST.json"):
        assert (out / f).exists(), f
    pen = pd.read_csv(out / "penalties.csv", skiprows=1)
    assert set(pen["chain"]) == {"enc:real", "enc:p1"} and len(pen) == 2 * 5 * 3
    con = pd.read_csv(out / "contrasts.csv", skiprows=1)
    assert list(con["subset"]).count("fast_cycling") == 1


def test_public_table_recomputes_the_contrasts(run):
    """scripts/export_public_tables.py's `export_stage` on the encoder folder: per-patient RMSE only, and
    every contrast recomputes from it with the paired subject bootstrap."""
    root, conf, stored, s = run
    ex = _script("export_public_tables")
    man = ex.export_stage(root, "sdev/exploratory_encoder", root, {"commit": "t" * 40, "describe": "test"})
    assert man["checks"]["rmse_equal_to_per_patient_primary"]
    wide = pd.read_csv(root / "sdev" / "exploratory_encoder" / "per_patient_primary_wide.csv", skiprows=1)
    assert not any(c in wide.columns for c in ("y_map", "last_map", "t_issue"))
    assert {f"rmse_{m}" for m in ME.REPORTED_MODELS} <= set(wide.columns)
    e = wide.loc[wide["entered"]].sort_values("caseid")
    from btc.models.bootstrap import SubjectBootstrap, contrast
    bs = SubjectBootstrap(e["caseid"].to_numpy(), e["subjectid"].to_numpy(), 200, SEED)
    for k, v in s["contrasts"].items():
        a, b = k.split(" vs ")
        r = contrast(bs, e[f"rmse_{a}"], e[f"rmse_{b}"], a, b)
        assert r.delta == pytest.approx(v.delta, abs=1e-12) and r.delta_hi == pytest.approx(v.delta_hi, abs=1e-12)
        assert r.rr_lo == pytest.approx(v.rr_lo, abs=1e-12)


def test_reproduction_check_fails_on_other_forecasts(planted, run):
    T, tab = planted[0], planted[1]
    root, conf, stored, _ = run
    bad = stored.copy()
    bad.loc[5, "pred_M3"] += 1e-6
    w = Writer(root / "other", {"commit": "t", "data": "synthetic"})
    with pytest.raises(RuntimeError, match="re-derived"):
        ME.encoder_sdev(T, tab, bad, conf, w, n_jobs=2, n_boot=50, cols=COLS)
    with pytest.raises(ValueError):
        ME.encoder_sdev(T, tab, stored.iloc[1:], conf, w, n_jobs=2, n_boot=50, cols=COLS)


# ----------------------------------------------------------------------------
# Guards (ENCODER_PLAN.md §6, §11)
# ----------------------------------------------------------------------------

def test_start_problems(monkeypatch):
    ok = {"sha256": "s", "committed_at_head": True}
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    assert E.start_problems(CLEAN, ok) == []
    assert any("not clean" in p for p in E.start_problems({**CLEAN, "state": "dirty"}, ok))
    assert any("not committed" in p for p in E.start_problems(CLEAN, {**ok, "committed_at_head": False}))
    monkeypatch.setenv("BTC_FROZEN_SCORING", "1")
    assert any("BTC_FROZEN_SCORING" in p for p in E.start_problems(CLEAN, ok))
    monkeypatch.setenv("BTC_FROZEN_SCORING", "0")
    assert any("BTC_FROZEN_SCORING" in p for p in E.start_problems(CLEAN, ok))


def test_plan_record_of_this_repository():
    r = E.plan_record()
    assert r["file"] == "ENCODER_PLAN.md" and len(r["sha256"]) == 64 and r["last_commit"]


@pytest.mark.parametrize("name", ["build_embeddings", "build_features_enc"])
def test_build_scripts_refuse_before_reading(name, monkeypatch, capsys):
    m = _script(name)
    monkeypatch.setattr(m.provenance, "git_state", lambda: {**CLEAN, "state": "dirty"})
    monkeypatch.setattr(m.E, "plan_record", lambda: {"sha256": "s", "committed_at_head": False})
    monkeypatch.setattr(m.pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read")))
    assert m.main([]) == 2
    err = capsys.readouterr().err
    assert "not clean" in err and "not committed" in err


def test_run_guards(tmp_path, monkeypatch):
    m = _script("run_encoder_check")
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    monkeypatch.setattr(m, "code_changes_since", lambda c: [] if c == "e" * 40 else ["btc/encoder.py"])
    plan = {"sha256": "p", "committed_at_head": True}
    conf = {"meta": {"inputs_sha256": {"a": 1}}}
    elog = {"embeddings_sha256": "E", "plan": {"sha256": "p"}, "git": {"commit": "e" * 40, "state": "clean"}}
    assert m.common_problems(CLEAN, plan, conf, {"a": 1}, True, elog, "E") == []
    probs = m.common_problems({**CLEAN, "state": "dirty"}, {**plan, "committed_at_head": False}, conf, {"a": 2},
                              False, {**elog, "git": {"commit": "f" * 40, "state": "dirty"}, "plan": {"sha256": "q"}}, "X")
    for k in ("not clean", "not committed", "inputs_sha256", "fold plan", "SHA-256 in its log", "another ENCODER_PLAN",
              "code changed", "not clean"):
        assert any(k in p for p in probs), k
    assert any("no window embeddings" in p for p in m.common_problems(CLEAN, plan, conf, {"a": 1}, True, None, None))
    out = tmp_path / "enc"
    out.mkdir()
    flog = {"features_sha256": "F", "embeddings_sha256": "E", "git": {"commit": "e" * 40}}
    bc = {"meta": {"embeddings_sha256": "E"}, "passed": True}
    assert m.sdev_problems(out, flog, "F", "E", bc, True, False, True) == []
    assert any("not committed" in p for p in m.sdev_problems(out, flog, "F", "E", bc, False, False, True))
    low = {**bc, "passed": False}
    assert any("below 0.5" in p for p in m.sdev_problems(out, flog, "F", "E", low, True, False, True))
    assert m.sdev_problems(out, flog, "F", "E", low, True, True, True) == []
    assert any("predictions_primary" in p for p in m.sdev_problems(out, flog, "F", "E", bc, True, False, False))
    assert any("other embeddings" in p for p in m.sdev_problems(out, {**flog, "embeddings_sha256": "Z"}, "F", "E", bc,
                                                                True, False, True))
    (out / "summary.json").write_text("{}")
    assert any("rerun is a departure" in p for p in m.sdev_problems(out, flog, "F", "E", bc, True, False, True))


def test_plan_section_is_verbatim():
    m = _script("run_encoder_check")
    s8 = m.plan_section(8)
    assert s8.startswith("## 8. Disclosures") and "Pretraining overlap" in s8 and "## 9." not in s8
    assert s8 in (ROOT / "ENCODER_PLAN.md").read_text()


# ----------------------------------------------------------------------------
# scripts/run_encoder_check.py end to end, on synthetic tables in a temporary layout
# ----------------------------------------------------------------------------

def test_run_script_end_to_end(tmp_path, monkeypatch, planted):
    """`build-check` then `sdev` through the real-data branch of the script: synthetic tables, a synthetic
    confirmatory run (2,000 resamples, as the script uses), synthetic encoder tables at full width; git state,
    plan record and HEAD checks replaced by stand-ins. No real table is read."""
    from btc import paths as bpaths
    T = planted[0]
    m = _script("run_encoder_check")
    results, derived = tmp_path / "results", tmp_path / "derived"
    derived.mkdir()
    meta = {"commit": "t" * 40, "describe": "test", "data": "synthetic(seed=2, wf_signal=4.0)", "stage": "sdev",
            "inputs_sha256": T.sha256}
    cw = Writer(results / "sdev", meta)
    P.confirmatory(T, cw, n_jobs=2)
    cw.manifest()
    # encoder tables as the build scripts write them
    plan = make_fold_plan(T.analysed, SEED)
    rng = np.random.default_rng(5)
    ids = np.repeat(T.analysable["caseid"].to_numpy(int), 8)
    win = pd.DataFrame({"caseid": ids, "w": np.tile(np.arange(8), len(ids) // 8), "usable": True,
                        "pleth_rate": rng.normal(75, 12, len(ids))})
    win.to_parquet(derived / "sdev_windows.parquet")
    Emat = rng.standard_normal((len(win), 512)).astype(np.float32)
    Emat[:, :8] += ((win["pleth_rate"].to_numpy() - 75) / 6)[:, None].astype(np.float32)
    np.save(derived / "sdev_enc_window_embeddings.npy", Emat)
    sha = E.file_digests(derived / "sdev_enc_window_embeddings.npy")["sha256"]
    win[["caseid", "w"]].assign(row=np.arange(len(win))).to_parquet(derived / "sdev_enc_window_index.parquet")
    elog = {"embeddings_sha256": sha, "plan": {"sha256": "p"}, "git": {"commit": "e" * 40, "state": "clean"},
            "weights": {"file": "data/external/papagei/papagei_s.pt", "size": 1, "md5": "m", "sha256": "w",
                        "zenodo_record": 1, "doi": "d"}, "model": {"class": "ResNet1DMoE"},
            "runtime": {"torch": "x"}, "n_windows_embedded": len(win), "n_cases": int(len(ids) // 8),
            "dom_freq_check": {"share_agree": 1.0}}
    (derived / "sdev_enc_window_embeddings_log.json").write_text(json.dumps(elog))
    tab = encoder_table(T, E.EMB_DIM, planted=True, seed=2)
    tab.to_parquet(derived / "sdev_enc_features.parquet")
    flog = {"features_sha256": E.file_digests(derived / "sdev_enc_features.parquet")["sha256"],
            "embeddings_sha256": sha, "git": {"commit": "e" * 40}}
    (derived / "sdev_enc_features_log.json").write_text(json.dumps(flog))
    # the script's paths and checks
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    monkeypatch.setattr(bpaths, "DERIVED", derived)
    for k, v in {"RESULTS": results, "CONF": results / "sdev" / "confirmatory",
                 "SDEV_MANIFEST": results / "sdev" / "MANIFEST.json", "RUN_LOG": derived / "sdev_enc_run_log.jsonl",
                 "FEATURES": derived / "sdev_enc_features.parquet", "DERIVED": derived,
                 "FEATURES_LOG": derived / "sdev_enc_features_log.json"}.items():
        monkeypatch.setattr(m, k, v)
    monkeypatch.setattr(m, "git_meta", lambda: dict(CLEAN))
    monkeypatch.setattr(m, "load_tables", lambda d, cohort: T)
    monkeypatch.setattr(m.E, "plan_record", lambda: {"file": "ENCODER_PLAN.md", "sha256": "p",
                                                     "committed_at_head": True, "last_commit": "a" * 40})
    monkeypatch.setattr(m, "code_changes_since", lambda c: [])
    monkeypatch.setattr(m, "plan_unchanged_since_public", lambda: True)
    committed = {"ok": False}
    monkeypatch.setattr(m, "is_committed_at_head", lambda rel: committed["ok"])
    out = results / "sdev" / "exploratory_encoder"
    assert m.main(["sdev"]) == 2                                   # no build check yet
    assert m.main(["build-check"]) == 0
    bc = json.loads((out / "build_check.json").read_text())
    assert bc["passed"] and bc["n_cases"] == len(plan.caseids) and bc["meta"]["embeddings_sha256"] == sha
    assert bc["meta"]["plan_public"]["pushed_utc"] == "2026-11-23T03:54:36Z"
    assert m.main(["build-check"]) == 2                            # once
    assert m.main(["sdev"]) == 2                                   # build_check.json not committed
    committed["ok"] = True
    assert m.main(["sdev", "--n-jobs", "2"]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["N"] == json.loads((results / "sdev" / "confirmatory" / "summary.json").read_text())["N"]
    assert s["meets_rule_with_placebo"] is True and s["build_check"]["passed"] is True
    man = json.loads((out / "MANIFEST.json").read_text())
    assert {"summary.json", "build_check.json", "predictions_primary.csv", "summary.md"} <= set(man["files"])
    assert man["meta"]["weights"]["sha256"] == "w" and man["meta"]["plan"]["sha256"] == "p"
    assert man["meta"]["plan_public"]["commit"].startswith("29db44b")
    pub = json.loads((out / "public_tables.json").read_text())
    assert pub["checks"]["entering_patients_forecasts_and_delta_equal_to_summary"]
    md = (out / "summary.md").read_text()
    assert s["one_line_summary"] in md and "## 8. Disclosures" in md
    assert m.main(["sdev"]) == 2                                   # the first complete run is the result
    events = [json.loads(l)["event"] for l in (derived / "sdev_enc_run_log.jsonl").read_text().splitlines()]
    assert events.count("completed") == 2 and events.count("refused") == 4
