"""Guards of `scripts/run_frozen_scoring.py` (SPEC §2.2, §9 steps 6-8), the frozen coefficient and
penalty-path tables, and the portability of the pickled models.

Synthetic tables only. The real-data branches of `freeze`, `score`, `after` and `series` are run
in-process with `git_meta`, `load_tables` and the HEAD check replaced by stand-ins, so no derived
table, signal or network is read; the data folders passed are empty temporary folders.
"""
import dataclasses
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc.models import pipeline as P
from btc.models.analyses import ladder, run_refit
from btc.models.data import build_frame, build_triples, tables_from_frames
from btc.models.design import Cols
from btc.models.folds import make_fold_plan
from btc.models.outputs import Writer, tracked_files_modified, untracked_files
from btc.models.secondary import coefficient_table, penalty_path_table
from btc.models.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
SEED, WF = 1, 2.0
TAG = "models-frozen"
CLEAN = {"commit": "c" * 40, "describe": TAG, "state": "clean", "dirty": False, "status": [],
         "tags_at_head": [TAG]}
CONF_META = {"commit": "f" * 40, "describe": "feature-freeze", "tracked_files_modified": False,   # earlier format
             "tags_at_head": ["feature-freeze"], "feature_tag": "feature-freeze",
             "data": "VitalDB derived tables (derived/sdev_*.parquet)", "stage": "sdev"}


def _module():
    spec = importlib.util.spec_from_file_location("run_frozen_scoring", ROOT / "scripts" / "run_frozen_scoring.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def syn():
    s = make_synthetic(SEED, WF)
    return {k: tables_from_frames(k, v) for k, v in s.items()}


@pytest.fixture(scope="module")
def conf_root(tmp_path_factory, syn):
    """A confirmatory run on synthetic S-dev whose `meta` reads like a real run at the feature tag
    (earlier `git_meta` format, as written at tag feature-freeze)."""
    root = tmp_path_factory.mktemp("conf") / "results"
    T = syn["sdev"]
    w = Writer(root / "sdev", {**CONF_META, "inputs_sha256": T.sha256})
    P.confirmatory(T, w, n_jobs=2)
    w.manifest()
    return root


def _real_mode(monkeypatch, m, syn, g=CLEAN, committed=True):
    """The real-data branch of the script on synthetic tables: no derived table is read."""
    monkeypatch.setattr(m, "git_meta", lambda: dict(g))
    monkeypatch.setattr(m, "load_tables", lambda d, cohort: syn[cohort])
    monkeypatch.setattr(m, "is_committed_at_head", lambda p: committed)


def _copy(conf_root, tmp_path) -> Path:
    out = tmp_path / "results"
    shutil.copytree(conf_root, out)
    return out


@pytest.fixture(scope="module")
def frozen_root(tmp_path_factory, conf_root, syn):
    """`freeze` through the real-data branch, after the synthetic confirmatory run."""
    m = _module()
    out = tmp_path_factory.mktemp("frz") / "results"
    shutil.copytree(conf_root, out)
    mp = pytest.MonkeyPatch()
    try:
        _real_mode(mp, m, syn)
        rc = m.main(["freeze", "--out-dir", str(out), "--data-dir", str(out.parent), "--n-jobs", "2"])
    finally:
        mp.undo()
    assert rc == 0
    return out


# ----------------------------------------------------------------------------------------- git state
def test_git_state_readers_accept_both_formats():
    assert tracked_files_modified({"tracked_files_modified": False}) is False
    assert tracked_files_modified({"tracked_files_modified": True}) is True
    assert tracked_files_modified({"tracked_files_modified": None}) is None
    assert tracked_files_modified({}) is None and tracked_files_modified(None) is None
    assert tracked_files_modified(CLEAN) is False
    assert tracked_files_modified({"state": "unknown", "status": []}) is None
    assert tracked_files_modified({"state": "dirty", "status": ["?? results/x"]}) is False
    assert tracked_files_modified({"state": "dirty", "status": ["?? a", " M btc/models/ridge.py"]}) is True
    assert tracked_files_modified({"state": "dirty", "status": ["D  y.txt"]}) is True
    assert tracked_files_modified({"state": "dirty", "dirty": True}) is True          # no status: conservative
    assert untracked_files({"status": ["?? a/", " M b", '?? "c d"']}) == ["a/", "c d"]
    assert untracked_files({"tracked_files_modified": False}) is None


def test_tree_problems_and_code_changed():
    m = _module()
    out = ROOT / "results"
    assert m.tree_problems(CLEAN, out) == []
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": ["?? results/sob/"]}, out) == []
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": ["?? results/"]}, out) == []
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": ["?? results/"]}, out / "x")   # collapsed dir
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": ["?? scripts/new.py"]}, out)
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": ["?? notes.txt"]}, out)
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": [" M btc/models/ridge.py"]}, out)
    assert m.tree_problems({**CLEAN, "state": "dirty", "status": ["?? results/a"] * 3 + ["..."]}, out)
    assert m.tree_problems({**CLEAN, "state": "unknown", "status": []}, out) == ["the git state is unknown"]
    # earlier format (tracked files only)
    old = {"commit": "c" * 40, "tracked_files_modified": False}
    assert m.tree_problems(old, out) == [] and m.code_changed(old, "c" * 40, out) is False
    assert m.code_changed({**old, "tracked_files_modified": True}, "c" * 40, out) is True
    assert m.code_changed({**old, "tracked_files_modified": None}, "c" * 40, out) is True
    assert m.code_changed(old, "d" * 40, out) is True


def test_is_committed_at_head():
    m = _module()
    assert m.is_committed_at_head(ROOT / "SPEC.md") is (
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "SPEC.md"], cwd=ROOT).returncode == 0)
    assert m.is_committed_at_head(ROOT / "no-such-file.json") is False
    assert m.is_committed_at_head(Path("/") / "etc" / "hostname") is False                # outside the repository


# ----------------------------------------------------------------------------------------- score
def test_score_on_real_data_needs_frozen_tag(tmp_path):
    r = subprocess.run([sys.executable, "scripts/run_frozen_scoring.py", "score", "--out-dir", str(tmp_path),
                        "--data-dir", str(tmp_path), "--sob-dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--frozen-tag is required" in r.stderr
    assert not (tmp_path / "sob").exists()


REFUSALS = [                                     # (git state, FROZEN.json committed at HEAD, logged reason)
    ({**CLEAN, "tags_at_head": ["other"]}, True, "does not carry tag"),
    ({**CLEAN, "state": "dirty", "status": [" M btc/models/ridge.py"]}, True, "tracked files"),
    ({**CLEAN, "state": "dirty", "status": ["?? scripts/extra.py"]}, True, "untracked file outside"),
    ({**CLEAN, "state": "unknown", "status": []}, True, "unknown"),
    (CLEAN, False, "FROZEN.json is not the file committed at HEAD"),
]


@pytest.mark.parametrize("g, committed, reason", REFUSALS)
def test_score_refuses_unless_at_the_frozen_tag_with_a_clean_tree(tmp_path, monkeypatch, frozen_root, syn,
                                                                   g, committed, reason):
    m = _module()
    out = _copy(frozen_root, tmp_path)
    _real_mode(monkeypatch, m, syn, g, committed)
    monkeypatch.setattr(m, "load_tables", lambda d, c: pytest.fail("score read tables before its checks"))
    assert m.main(["score", "--frozen-tag", TAG, "--out-dir", str(out), "--sob-dir", str(tmp_path)]) == 2
    assert not (out / "sob" / "frozen_scoring" / "STATE.json").exists()
    log = [json.loads(x) for x in (out / "sob" / "frozen_scoring_log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["refused"] and reason in log[0]["reason"]


def test_score_runs_at_the_frozen_tag(tmp_path, monkeypatch, frozen_root, syn):
    m = _module()
    out = _copy(frozen_root, tmp_path)
    _real_mode(monkeypatch, m, syn)
    assert m.main(["score", "--frozen-tag", TAG, "--out-dir", str(out), "--sob-dir", str(tmp_path)]) == 0
    s = json.loads((out / "sob" / "frozen_scoring" / "summary.json").read_text())
    assert s["meta"]["frozen_tag"] == TAG
    assert s["n_analysed_cases"] == s["n_analysable_cases"] - s["n_nmb_recorded_cases"] and s["n"] <= s["n_analysed_cases"]
    log = [json.loads(x) for x in (out / "sob" / "frozen_scoring_log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["started", "completed"] and log[0]["frozen_tag"] == TAG


# ----------------------------------------------------------------------------------------- after, series, all
class _Reached(Exception):
    """Raised by a stand-in table loader: the command got past its checks to its first table read."""


def _scored(frozen_root, tmp_path) -> Path:
    """A copy of the frozen root that reads as if `score` and `after` had completed (stand-in STATE.json
    and conformal.csv), so that only the tag guard stands between `after`/`series` and the tables."""
    out = _copy(frozen_root, tmp_path)
    st = out / "sob" / "frozen_scoring" / "STATE.json"
    st.parent.mkdir(parents=True)
    st.write_text(json.dumps({"status": "completed", "commit": CLEAN["commit"], "rerun_after_crash": False}) + "\n")
    cf = out / "sob" / "after_frozen_scoring" / "conformal.csv"
    cf.parent.mkdir(parents=True)
    cf.write_text("")
    return out


@pytest.mark.parametrize("command", ["after", "series"])
def test_after_and_series_on_real_data_need_frozen_tag(tmp_path, command):
    cases = tmp_path / "cases"
    r = subprocess.run([sys.executable, "scripts/run_frozen_scoring.py", command, "--out-dir", str(tmp_path),
                        "--data-dir", str(tmp_path), "--sob-dir", str(tmp_path), "--cases-dir", str(cases)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--frozen-tag is required" in r.stderr
    assert not (tmp_path / "sob").exists() and not cases.exists()


def test_all_stays_synthetic_only(tmp_path):
    r = subprocess.run([sys.executable, "scripts/run_frozen_scoring.py", "all", "--frozen-tag", TAG,
                        "--out-dir", str(tmp_path), "--data-dir", str(tmp_path), "--sob-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "for synthetic tests only" in r.stderr
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("command", ["after", "series"])
@pytest.mark.parametrize("g, committed, reason", REFUSALS)
def test_after_and_series_refuse_unless_at_the_frozen_tag_with_a_clean_tree(tmp_path, monkeypatch, frozen_root,
                                                                            syn, command, g, committed, reason):
    m = _module()
    out = _scored(frozen_root, tmp_path)
    cases = tmp_path / "cases"
    _real_mode(monkeypatch, m, syn, g, committed)
    monkeypatch.setattr(m, "load_tables", lambda d, c: pytest.fail(f"{command} read tables before its checks"))
    monkeypatch.setattr(P, "load_frozen", lambda root: pytest.fail(f"{command} loaded the models before its checks"))
    assert m.main([command, "--frozen-tag", TAG, "--out-dir", str(out), "--sob-dir", str(tmp_path),
                   "--cases-dir", str(cases)]) == 2
    log = [json.loads(x) for x in (out / "sob" / "frozen_scoring_log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["refused"] and reason in log[0]["reason"]
    assert log[0]["command"] == command and log[0]["frozen_tag"] == TAG
    assert [p.name for p in (out / "sob" / "after_frozen_scoring").iterdir()] == ["conformal.csv"]
    assert not cases.exists()


@pytest.mark.parametrize("command", ["after", "series"])
def test_after_and_series_pass_the_guard_at_the_frozen_tag(tmp_path, monkeypatch, frozen_root, syn, command):
    m = _module()
    out = _scored(frozen_root, tmp_path)
    _real_mode(monkeypatch, m, syn)

    def reached(d, cohort):
        raise _Reached(cohort)

    monkeypatch.setattr(m, "load_tables", reached)
    with pytest.raises(_Reached):
        m.main([command, "--frozen-tag", TAG, "--out-dir", str(out), "--sob-dir", str(tmp_path),
                "--cases-dir", str(tmp_path / "cases")])
    assert not (out / "sob" / "frozen_scoring_log.jsonl").exists()          # nothing refused


# ----------------------------------------------------------------------------------------- freeze
def test_freeze_checks_and_records(frozen_root, syn):
    fz = json.loads((frozen_root / "frozen" / "FROZEN.json").read_text())
    conf = json.loads((frozen_root / "sdev" / "confirmatory" / "summary.json").read_text())
    assert fz["n_sdev_cases"] == conf["n_analysed_cases"] == len(syn["sdev"].analysed)
    assert fz["n_sdev_cases"] < conf["n_analysable_cases"]                 # the nmb_recorded cases are out
    assert fz["sdev_case_counts"]["include_nmb"] is False
    assert set(fz["software"]) == {"python", "numpy", "pandas", "scikit-learn", "scipy"}
    man = json.loads((frozen_root / "frozen" / "MANIFEST.json").read_text())
    assert man["meta"]["inputs_sha256"] == conf["meta"]["inputs_sha256"]
    assert {"frozen/coefficients.csv", "frozen/penalty_path.csv", "frozen/FROZEN.json", "frozen/models.pkl"} <= set(
        man["files"])
    m = _module()
    assert m.freeze_output_problems(frozen_root, syn["sdev"], conf) == []


def test_freeze_output_problems_detect_tampering(tmp_path, frozen_root, syn):
    m = _module()
    out = _copy(frozen_root, tmp_path)
    conf = json.loads((out / "sdev" / "confirmatory" / "summary.json").read_text())
    assert m.freeze_output_problems(out, syn["sdev"], {**conf, "n_analysed_cases": conf["n_analysable_cases"]})
    bad_meta = {**conf, "meta": {**conf["meta"], "inputs_sha256": {"x": "0"}}}
    assert any("inputs_sha256" in p for p in m.freeze_output_problems(out, syn["sdev"], bad_meta))
    folds = out / "sdev" / "confirmatory" / "folds.csv"
    lines = folds.read_text().splitlines(keepends=True)
    folds.write_text("".join(lines[:-1]))                                   # one case fewer
    assert any("folds.csv" in p for p in m.freeze_output_problems(out, syn["sdev"], conf))


def test_freeze_refusals_before_fitting(tmp_path, monkeypatch, conf_root, syn):
    m = _module()
    out = _copy(conf_root, tmp_path)
    monkeypatch.setattr(P, "freeze", lambda *a, **k: pytest.fail("fitted despite a failed check"))
    _real_mode(monkeypatch, m, syn, {**CLEAN, "state": "dirty", "status": [" M btc/models/ridge.py"]})
    with pytest.raises(SystemExit):
        m.main(["freeze", "--out-dir", str(out), "--data-dir", str(tmp_path)])
    _real_mode(monkeypatch, m, syn)
    p = out / "sdev" / "confirmatory" / "summary.json"
    conf = json.loads(p.read_text())
    for meta in ({**conf["meta"], "inputs_sha256": {"sdev_cases": "0"}},        # other inputs
                 {**conf["meta"], "tracked_files_modified": True},            # confirmatory run not clean
                 {**conf["meta"], "tags_at_head": []}):                       # not at its feature tag
        p.write_text(json.dumps({**conf, "meta": meta}))
        assert m.main(["freeze", "--out-dir", str(out), "--data-dir", str(tmp_path)]) == 2
    p.write_text(json.dumps(conf))
    (out / "frozen").mkdir()                                                   # a second freeze
    with pytest.raises(SystemExit):
        m.main(["freeze", "--out-dir", str(out), "--data-dir", str(tmp_path)])
    shutil.rmtree(out / "frozen")
    p.unlink()                                                                 # no confirmatory result
    with pytest.raises(SystemExit):
        m.main(["freeze", "--out-dir", str(out), "--data-dir", str(tmp_path)])


def test_freeze_on_analysable_instead_of_analysed_cases_is_rejected(tmp_path, monkeypatch, conf_root, syn, capsys):
    """A wrong merge that fits on all analysable cases (with the nmb_recorded ones) fails loudly and
    leaves nothing that `score` could load."""
    m = _module()
    out = _copy(conf_root, tmp_path)
    real = P.freeze
    monkeypatch.setattr(P, "_analysed_only", lambda *t: None)
    monkeypatch.setattr(P, "freeze", lambda T, w, n_jobs=1, expected_structure=None:
                        real(dataclasses.replace(T, include_nmb=True), w, n_jobs=n_jobs))
    _real_mode(monkeypatch, m, syn)
    assert m.main(["freeze", "--out-dir", str(out), "--data-dir", str(tmp_path), "--n-jobs", "2"]) == 3
    err = capsys.readouterr().err
    assert "FREEZE CHECK FAILED" in err and "n_sdev_cases" in err and "include_nmb" in err
    assert not (out / "frozen").exists() and (out / "frozen_rejected" / "FROZEN.json").exists()


# ----------------------------------------------------------------------------------------- frozen tables
def _apply(tab: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Forecasts of every stage of one chain from its coefficient rows alone (the formula in the
    docstring of `coefficient_table`)."""
    stages = list(dict.fromkeys(tab["model"]))
    nodes = set(tab["node"])
    hb = frame["h_bin"].to_numpy(int)
    node_of = np.array([f"bin{b}" if f"bin{b}" in nodes else "pooled" for b in hb])
    out = np.full((len(frame), len(stages)), np.nan)
    for node in nodes:
        r = node_of == node
        f = frame.loc[r].reset_index(drop=True)
        prev = np.zeros(int(r.sum()))
        for s, st in enumerate(stages):
            rows = tab.loc[(tab["node"] == node) & (tab["model"] == st)]
            if (rows["column"] == "(dropped)").any():
                out[r, s] = prev
                continue
            b0 = float(rows.loc[rows["column"] == "(intercept)", "coef"].iloc[0])
            ch = np.full(len(f), b0)
            for x in rows.loc[rows["column"] != "(intercept)"].itertuples(index=False):
                c = x.column
                if x.block == "cuff":
                    if c == "h":
                        v = f[x.h_input]
                    elif "*h^" in c:
                        base, p = c.split("*h^")
                        v = f[base] * f[x.h_input] ** int(p)
                    else:
                        v = f[c]
                elif c.startswith("miss:"):
                    v = f[c[5:]].isna().astype(float)
                elif c.endswith("*h"):
                    v = (f[c[:-2]].fillna(x.impute_median) - x.impute_median) * f[x.h_input]
                else:
                    v = f[c].fillna(x.impute_median)
                ch = ch + x.coef * (v.to_numpy(float) - x.scaler_mean) / x.scaler_sd
            req = rows["requires"].dropna().unique()
            if len(req):
                ok = f[req[0]].astype(bool).to_numpy()
                ch = np.where(ok, ch, prev)
            out[r, s] = ch
            prev = ch
    return frame["last_map"].to_numpy(float)[:, None] + out


def test_coefficients_csv_reproduces_the_frozen_models(frozen_root, syn):
    tab = pd.read_csv(frozen_root / "frozen" / "coefficients.csv", skiprows=1)
    frozen = P.load_frozen(frozen_root)
    T = syn["sdev"]
    tri, u = build_triples(T)
    frames = {"main": build_frame(T, "primary", 1), "p1": build_frame(T, "primary", 1), "yard_T": tri,
              "yard_TP1": tri, "yard_Tia": tri, "yard_U": u, "m2_two_ahead": build_frame(T, "primary", 2)}
    assert set(tab["frozen_chain"]) == set(frozen["chains"]) == set(frames)
    for k, fr in frames.items():
        got = _apply(tab.loc[tab["frozen_chain"] == k], fr)
        want = frozen["chains"][k].predict(Cols(fr))
        assert np.allclose(got, want, rtol=0, atol=1e-8), k
    # each column carries its own block's penalty; the cuff block keeps M2's in every stage
    for (k, node), g in tab.loc[tab["column"].ne("(intercept)") & tab["column"].ne("(dropped)")].groupby(
            ["frozen_chain", "node"]):
        lam = frozen["chains"][k].nodes[node].lam
        cuff = g.loc[g["block"] == "cuff", "penalty"].astype(float)
        assert np.allclose(cuff, lam[0]), (k, node)


def test_coefficient_table_per_bin_and_dropped_block(syn):
    """A per-bin structure (nodes bin<i> and pooled) and a dropped heart-rate block (no planted HR
    signal): the waveform columns then carry the waveform penalty, not a stage index."""
    T = syn["sdev"]
    fr = build_frame(T, "primary", 1)
    fit = run_refit({"x": (Cols(fr), ladder("a", "perbin"), make_fold_plan(T.analysed, 20261118))})["x"]
    tab = coefficient_table(fit)
    assert {"pooled"} < set(tab["node"])
    assert np.allclose(_apply(tab, fr), fit.predict(Cols(fr)), rtol=0, atol=1e-8)
    for node, nf in fit.nodes.items():
        g = tab.loc[(tab["node"] == node) & (tab["model"] == "M4") & tab["block"].eq("waveform")]
        if nf.lam[2] is not None and len(g):
            assert np.allclose(g["penalty"].astype(float), nf.lam[2])
    path = penalty_path_table(fit)
    ch = path.loc[path["chosen"]]
    assert len(ch) == len(path.groupby(["node", "model"]))                 # one chosen candidate each
    for r in ch.itertuples(index=False):
        grp = path.loc[(path["node"] == r.node) & (path["model"] == r.model)]
        assert r.inner_cv_patient_mean_rmse <= grp["inner_cv_patient_mean_rmse"].min() * (1 + 1e-12)
        lam = fit.nodes[r.node].lam[fit.spec.stages.index(r.model)]
        assert r.candidate == ("dropped" if lam is None else lam)


def test_penalty_path_csv_matches_the_frozen_selection(frozen_root):
    path = pd.read_csv(frozen_root / "frozen" / "penalty_path.csv", skiprows=1)
    frozen = P.load_frozen(frozen_root)
    for (k, node, model), g in path.groupby(["frozen_chain", "node", "model"]):
        ch = frozen["chains"][k]
        lam = ch.nodes[node].lam[ch.spec.stages.index(model)]
        c = g.loc[g["chosen"], "candidate"]
        assert len(c) == 1 and str(c.iloc[0]) == ("dropped" if lam is None else str(lam))
        n_cand = 13 + (0 if model == ch.spec.stages[0] else 1)             # grid, plus "dropped" for added blocks
        assert len(g) == n_cand


# ----------------------------------------------------------------------------------------- pickle
def test_frozen_pickle_holds_no_path(frozen_root):
    blob = (frozen_root / "frozen" / "models.pkl").read_bytes()
    P.check_portable(blob)
    for s in (str(ROOT), str(Path.home())):
        assert s.encode() not in blob
    with pytest.raises(RuntimeError, match="paths"):
        P.check_portable(pickle.dumps({"source": "/data/derived/sdev_cases.parquet"}))
    with pytest.raises(RuntimeError, match="paths"):
        P.check_portable(pickle.dumps({"p": Path("relative") / "x", "q": Path("/") / "x"}))
    with pytest.raises(RuntimeError, match="paths"):
        P.check_portable(pickle.dumps(r"C:\data\x"))
