"""Per-case site files written after the frozen scoring (SPEC §9 step 8; btc/site/series.py), the publish
build on a fully synthetic chain, `git_meta` and `scripts/fill_site_config.py`.

Synthetic data only. `BTC_FROZEN_SCORING=1` is set only in `test_real_signals_on_fake_cohort`, with every
data path pointed at the fake cases of `test_cohorts_provenance.fake_sob`.
"""
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc import io, paths
from btc.models import analyses, pipeline as P, ridge, site_cases as SC
from btc.models.data import h_bin_of, load_tables, tables_from_frames
from btc.models.design import Cols
from btc.models.outputs import git_meta
from btc.models.secondary import BIN_LABELS
from btc.models.synthetic import SyntheticSignals, make_synthetic
from btc.site import fixtures as FX
from btc.site.featured import MIN_FORECASTS_FEATURED, select_featured
from btc.site.results import consistency_problems, load_site_data
from btc.site.series import NMB_REASON, load_cases
from btc.site.theme import SYNTHETIC_MARK

from test_cohorts_provenance import _script as builder, fake_sob  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
SEED, WF = 1, 2.0                       # planted waveform signal, so that M4 differs from M2
SYN = ["--synthetic", "--synthetic-seed", str(SEED), "--wf-signal", str(WF)]


def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, skiprows=1)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """synthetic S-dev + S-ob -> run_confirmatory -> run_frozen_scoring all -> series."""
    tmp = tmp_path_factory.mktemp("chain")
    out, cases = tmp / "results", tmp / "cases"
    r = _run("scripts/run_confirmatory.py", *SYN, "--with-secondary", "--no-gb", "--n-boot", "200",
             "--out-dir", str(out), "--n-jobs", "2")
    assert r.returncode == 0, r.stderr
    r = _run("scripts/run_frozen_scoring.py", "all", *SYN, "--no-gb", "--n-boot", "200", "--out-dir", str(out),
             "--n-jobs", "2")
    assert r.returncode == 0, r.stderr
    frozen_files = {p: _sha(out / "frozen" / p) for p in ("models.pkl", "FROZEN.json")}
    r = _run("scripts/run_frozen_scoring.py", "series", *SYN, "--out-dir", str(out), "--cases-dir", str(cases))
    assert r.returncode == 0, r.stderr
    syn = make_synthetic(SEED, WF)
    return {"tmp": tmp, "out": out, "cases": cases, "frozen_files": frozen_files, "syn": syn,
            "T": tables_from_frames("sob", syn["sob"]), "loaded": load_cases(cases)}


def _scored_issue(s):
    return [ip for ip in s.issues if ip.scored]


# ----------------------------------------------------------------------------------------- curves
def test_curve_at_realised_h_equals_pred_at_target(chain):
    pr = _csv(chain["out"] / "sob/frozen_scoring/predictions_primary.csv")
    _, cases, series = chain["loaded"]
    n_checked = 0
    for cid, s in series.items():
        rows = pr.loc[pr["caseid"] == cid].set_index("t_issue")
        sc = _scored_issue(s)
        assert len(sc) == len(rows) == next(c for c in cases if c.caseid == cid).n_forecasts
        for ip in sc:
            r = rows.loc[ip.t_issue_s]
            assert ip.pred_at_target == {"M2": r["pred_M2"], "M4": r["pred_M4"]}       # the scored forecasts
            k = np.flatnonzero(ip.h_s == r["h_s"])
            assert len(k) == 1, "the realised h is on the curve's grid"
            assert ip.m2.map[k[0]] == ip.pred_at_target["M2"] and ip.m4.map[k[0]] == ip.pred_at_target["M4"]
            assert ip.y_next == r["y_map"] and ip.t_next_s == r["t_target"]
            n_checked += 1
    assert n_checked == len(pr)


def test_curves_are_the_frozen_models_with_only_h_varied(chain):
    """Re-evaluate the frozen chain on the S-ob frame with h set to grid values (independent path)."""
    frozen = P.load_frozen(chain["out"])
    frame = P.frozen_predictions(frozen, chain["T"])["frame"]
    _, _, series = chain["loaded"]
    ch = frozen["chains"]["main"]
    for h in (15.0, 60.0, 155.0, 400.0):
        g = frame.copy()
        g["h_s"], g["h_min"], g["h_bin"] = h, h / 60.0, int(h_bin_of(h))
        p = ch.predict(Cols(g))
        for i, (cid, t) in enumerate(zip(frame["caseid"], frame["t_issue"])):
            if cid not in series:
                continue
            ip = next(x for x in series[cid].issues if x.t_issue_s == t)
            k = np.flatnonzero(ip.h_s == h)[0]
            assert abs(ip.m2.map[k] - p[i, 0]) < 1e-9 and abs(ip.m4.map[k] - p[i, 2]) < 1e-9
            assert ip.m4_fallback == (not bool(frame["wf_available@real"].iloc[i]))
    # M4 differs from M2 somewhere (planted signal)
    assert any(np.any(ip.m4.map != ip.m2.map) for s in series.values() for ip in s.issues)


def test_every_issue_point_has_a_curve_and_registered_half_widths(chain):
    _, cases, series = chain["loaded"]
    conf = _csv(chain["out"] / "sob/after_frozen_scoring/conformal.csv")
    w = {(m, b): hw for m, b, hw in zip(conf["model"], conf["bin"], conf["half_width"])}
    syn = chain["syn"]["sob"]
    rd = syn["readings"]
    unscored = 0
    for cid, s in series.items():
        v = rd.loc[(rd["caseid"] == cid) & rd["valid"]].sort_values("n")
        t = v["t"].to_numpy()
        n_issue = int(((v["n"] >= 3) & (v["t"] <= 1800)).sum())
        assert len(s.issues) == n_issue                       # §3: every issue point, scored or not
        np.testing.assert_array_equal(s.cuff_t, t[t <= 1800])
        for ip in s.issues:
            assert ip.h_s[0] == 15.0 and ip.h_s[-1] == 400.0 and np.all(np.isfinite(ip.m4.map))
            for h, a, b in zip(ip.h_s, ip.m2.half90, ip.m4.half90):
                lab = BIN_LABELS[int(h_bin_of(h))]
                assert a == w[("M2", lab)] and b == w[("M4", lab)]
            unscored += not ip.scored
    assert unscored >= len(series)                          # at least the last issue point of each case


# ----------------------------------------------------------------------------------------- no refit
def test_no_refit_and_frozen_models_unchanged(chain, monkeypatch, tmp_path):
    out = chain["out"]
    assert {p: _sha(out / "frozen" / p) for p in ("models.pkl", "FROZEN.json")} == chain["frozen_files"]

    def boom(*a, **k):
        raise AssertionError("a model was fitted")
    for mod, names in ((ridge, ("fit_chain", "select_node", "fit_node", "ridge_fit", "ridge_fit_path",
                                "cv_chain", "refit_chain")),
                       (analyses, ("run_cv", "run_refit")), (P, ("run_cv", "run_refit"))):
        for n in names:
            monkeypatch.setattr(mod, n, boom)
    frozen = P.load_frozen(out)
    T = chain["T"]
    man = P.site_cases(frozen, T, SyntheticSignals(T.cases, SEED), out, tmp_path / "c", {"data": "synthetic"},
                       synthetic=True)
    cli = json.loads((chain["cases"] / "MANIFEST.json").read_text())["files"]
    series = {k: v for k, v in man["files"].items() if k.startswith("series/")}
    assert series and series == {k: v for k, v in cli.items() if k.startswith("series/")}   # same bytes
    assert {p: _sha(out / "frozen" / p) for p in ("models.pkl", "FROZEN.json")} == chain["frozen_files"]


# ----------------------------------------------------------------------------------------- cases.json
def test_cases_json_covers_every_cohort_case(chain):
    syn_flag, cases, series = chain["loaded"]
    d = json.loads((chain["cases"] / "cases.json").read_text())
    assert syn_flag is True and d["meta"]["stage"] == P.AFTER_LABEL
    sob = chain["syn"]["sob"]["cases"].set_index("caseid")
    assert sorted(c.caseid for c in cases) == sorted(sob.index) and len(cases) == 21
    frz = json.loads((chain["out"] / "sob/frozen_scoring/summary.json").read_text())
    for c in cases:
        if not bool(sob.loc[c.caseid, "analysable"]):
            assert c.status == "not analysable" and c.reason
        elif bool(sob.loc[c.caseid, "nmb_recorded"]):                 # SPEC §2.2: analysable, not analysed
            assert c.status == "not analysed" and c.reason == NMB_REASON and c.n_forecasts is None
        else:
            assert c.status in ("scored", "not scored")
    assert sum(c.status == "not analysed" for c in cases) == frz["n_nmb_recorded_cases"] == 1
    scored = [c for c in cases if c.scored]
    assert len(scored) == frz["n"] == len(series)
    pp = _csv(chain["out"] / "sob/frozen_scoring/per_patient_primary.csv")
    for m in ("M2", "M3", "M4", "M4-P1"):
        ref = pp.loc[pp["model"] == m].set_index("caseid")["rmse"]
        for c in scored:
            assert abs(c.rmse[m] - ref.loc[c.caseid]) < 1e-12
    hyp = _csv(chain["out"] / "sob/after_frozen_scoring/hypotension.csv")
    hyp = hyp.loc[hyp["definition"] == "below 80% of record-start baseline"].set_index("model")
    assert sum(c.new_onset_80 for c in scored) == hyp.loc["M2", "n_events"]
    for m in ("M2", "M4"):
        assert sum(c.flagged_80[m] for c in scored) == hyp.loc[m, "flagged"]


def test_ug_and_mg_kept_apart(chain, tmp_path):
    _, cases, _ = chain["loaded"]
    src = SyntheticSignals(chain["T"].cases, SEED)
    for c in cases:
        ch = src.chart(c.caseid)
        for got, want in ((c.phenylephrine_ug, ch["phenylephrine_ug"]), (c.ephedrine_mg, ch["ephedrine_mg"])):
            assert (got is None and np.isnan(want)) or got == want
    assert any(c.stratum == "pump" for c in cases) and any((c.ephedrine_mg or 0) > 0 for c in cases)

    class Odd(SyntheticSignals):                 # totals whose sum (130) would be visible if added
        def chart(self, caseid):
            return {**super().chart(caseid), "phenylephrine_ug": 123.0, "ephedrine_mg": 7.0}
    frozen = P.load_frozen(chain["out"])
    P.site_cases(frozen, chain["T"], Odd(chain["T"].cases, SEED), chain["out"], tmp_path, {}, synthetic=True)
    d = json.loads((tmp_path / "cases.json").read_text())
    for c in d["cases"]:
        assert c["phenylephrine_ug"] == 123.0 and c["ephedrine_mg"] == 7.0
        assert 130.0 not in [v for v in c.values() if isinstance(v, (int, float))]
        assert not any(("ug" in k and "mg" in k) or "vasopressor" in k for k in c)


# ----------------------------------------------------------------------------------------- featured cases
def test_featured_selection_uses_the_series_files(chain):
    _, cases, series = chain["loaded"]
    rebuilt = []
    for c in cases:
        if not c.scored:
            rebuilt.append(c)
            continue
        s = series[c.caseid]
        sc = _scored_issue(s)
        y = np.array([ip.y_next for ip in sc])
        rmse = {m: float(np.sqrt(np.mean((np.array([ip.pred_at_target[m] for ip in sc]) - y) ** 2)))
                for m in ("M2", "M4")}
        thr = 0.8 * s.baseline_map
        last = dict(zip(s.cuff_t, s.cuff_map))
        onsets = sum(1 for ip in sc if last[ip.t_issue_s] >= thr and ip.y_next < thr)
        pump = SC.pump_changes(s.pump_t, s.pump_ml_h) if s.has_pump else 0
        assert len(sc) == c.n_forecasts and onsets == c.new_onset_80 and pump == c.pump_changes
        assert abs(rmse["M4"] - c.rmse["M4"]) < 1e-9 and abs(rmse["M2"] - c.rmse["M2"]) < 1e-9
        rebuilt.append(dataclasses.replace(c, rmse={**c.rmse, **rmse}, new_onset_80=onsets, pump_changes=pump,
                                           n_forecasts=len(sc)))
    f = select_featured(cases)
    assert f == select_featured(rebuilt) and f[1] is not None
    r1 = next(c for c in cases if c.caseid == f[1])
    assert r1.n_forecasts >= MIN_FORECASTS_FEATURED and r1.new_onset_80 >= 1
    if f[3] is not None:
        assert next(c for c in cases if c.caseid == f[3]).pump_track
    site = load_site_data(chain["out"], chain["cases"], FX.config_for("row6"))
    assert site.featured == f and consistency_problems(site) == []


def test_pump_changes_rule():
    assert SC.pump_changes([0, 300, 600, 900, 2000], [0, 30, 30, 15, 60]) == 2     # 2000 s is outside
    assert SC.pump_changes([0, 100, 200], [30, np.nan, 30]) == 0                    # missing samples skipped
    assert SC.pump_changes([200, 100], [15, 30]) == 1                               # time order, not file order
    assert SC.pump_changes([], []) == 0


# ----------------------------------------------------------------------------------------- publish
def _config(path: Path) -> Path:
    c = FX.config_for("row6", author_name="A. Tester", contact="tester@example.org")
    path.write_text(json.dumps({k: v for k, v in dataclasses.asdict(c).items() if k != "extra"}))
    return path


def test_publish_runs_on_the_synthetic_chain_only_with_the_test_flag(chain):
    tmp = chain["tmp"]
    cfg = _config(tmp / "site.json")
    base = ["scripts/build_site.py", "--publish", "--results", str(chain["out"]), "--cases-dir", str(chain["cases"]),
            "--config", str(cfg), "--jobs", "2"]
    r = _run(*base, "--readme", str(tmp / "refused" / "README.md"))
    assert r.returncode == 2 and "synthetic" in r.stderr                    # the real publish refuses
    r = _run(*base, "--test-allow-synthetic", "--out", str(ROOT / "docs"), "--readme", str(tmp / "x.md"))
    assert r.returncode != 0 and "outside docs/" in r.stderr
    out = tmp / "pub" / "docs"
    r = _run(*base, "--test-allow-synthetic", "--out", str(out), "--readme", str(tmp / "pub" / "README.md"))
    assert r.returncode == 0, r.stderr
    md = (tmp / "pub" / "README.md").read_text()
    assert SYNTHETIC_MARK in md and "](docs/assets/img/summary.png)" in md
    _, cases, _ = chain["loaded"]
    for c in cases:
        if c.scored:
            assert (out / f"assets/cases/case-{c.caseid}.png").exists()
    f = select_featured(cases)
    assert (out / f"assets/replay/replay-{f[1]}.mp4").exists() and (out / "assets/img/replay.gif").exists()
    assert not (tmp / "refused").exists()


# ----------------------------------------------------------------------------------------- real path, fake data
def test_real_signals_on_fake_cohort(chain, fake_sob, monkeypatch, tmp_path):
    """The real-data path (btc.io tracks, window table, recomputed-feature check) on fake S-ob cases."""
    monkeypatch.delenv(io.FROZEN_SCORING_ENV, raising=False)
    with pytest.raises(PermissionError):
        SC.RawSignals(pd.DataFrame(columns=["caseid", "pump_present", "intraop_phe_ug", "intraop_eph_mg"]),
                      pd.DataFrame(columns=["caseid"]))
    monkeypatch.setenv(io.FROZEN_SCORING_ENV, "1")                 # this test only, fake data only
    for name, extra in (("build_instances", []), ("build_windows", ["--jobs", "2"]), ("build_features_wf", [])):
        builder(name).main(["--cohort", "sob", "--allow-dirty", *extra])
    derived = fake_sob / "derived"
    T = load_tables(derived, "sob")
    frozen = P.load_frozen(chain["out"])
    widths = SC.read_widths(chain["out"] / "sob/after_frozen_scoring/conformal.csv")
    src = SC.RawSignals(T.cases, pd.read_parquet(derived / "sob_windows.parquet"))
    cases, series = SC.build_site_cases(frozen, T, src, widths)
    by = {c.caseid: c for c in cases}
    assert set(by) == set(T.cases["caseid"]) and by[90005].status == "not analysable"
    assert "ECG" in by[90005].reason
    assert by[90001].pump_track and by[90001].pump_changes == 1               # 30 -> 15 mL/h at 900 s
    assert series[90001].has_pump and list(series[90001].pump_ml_h) == [30.0, 15.0]
    assert by[90002].phenylephrine_ug == 100.0 and by[90003].ephedrine_mg == 10.0
    assert by[90004].phenylephrine_ug is None and by[90004].ephedrine_mg is None
    s = series[90002]
    assert np.all(np.diff(s.hr_t) > 0) and np.nanmin(s.hr_bpm) > 0 and len(s.pleth_start) > 0
    assert any(not ip.scored for ip in s.issues)
    # a derived feature that the frozen feature code does not reproduce stops the export
    bad = T.features_hr.copy()
    i = bad.index[(bad["variant"] == "real") & (bad["window"] == "primary") & (bad["caseid"] == 90002)
                  & bad["hr_level"].notna()][0]
    bad.loc[i, "hr_level"] += 1.0
    with pytest.raises(SC.ExportError, match="hr_level"):
        SC.build_site_cases(frozen, dataclasses.replace(T, features_hr=bad), src, widths)


# ----------------------------------------------------------------------------------------- git_meta
def _git(repo, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false",
                    "-c", "tag.gpgsign=false", *a], cwd=repo, check=True, capture_output=True)


def test_git_meta_counts_untracked_files(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "a.txt").write_text("a")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "c")
    _git(tmp_path, "tag", "v1")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    g = git_meta()
    assert g["state"] == "clean" and g["dirty"] is False and g["tags_at_head"] == ["v1"]
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "x").write_text("x")
    assert git_meta()["state"] == "clean"
    (tmp_path / "new.py").write_text("x")                              # untracked, not ignored
    g = git_meta()
    assert g["state"] == "dirty" and g["dirty"] is True and any("new.py" in s for s in g["status"])


# ----------------------------------------------------------------------------------------- fill_site_config
def _fill_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fill_site_config", ROOT / "scripts" / "fill_site_config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


REPORT = """# report

## Registration

| Version | Commit | Date | `PREREG.md` SHA-256 | `SPEC.md` SHA-256 |
|---|---|---|---|---|
| 1.5 | `{c}` | 2026-11-20 | `{p}` | `{s}` |

## Deviations

{dev}

---

credit
"""


def test_fill_site_config(tmp_path, monkeypatch):
    fs = _fill_module()
    repo = tmp_path
    env = {"GIT_COMMITTER_DATE": "2026-11-20T10:00:00+09:00", "GIT_AUTHOR_DATE": "2026-11-20T10:00:00+09:00"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _git(repo, "init", "-q")
    (repo / "PREREG.md").write_text("prereg v1.5\n")
    (repo / "SPEC.md").write_text("spec v1.5\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "registration")
    reg = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    sha = {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in ("PREREG.md", "SPEC.md")}
    rep = REPORT.format(c=reg, p=sha["PREREG.md"], s=sha["SPEC.md"], dev="None so far.")
    (repo / "REPORT.md").write_text(rep)
    (repo / "results" / "frozen").mkdir(parents=True)
    (repo / "results" / "frozen" / "FROZEN.json").write_text("{}")
    _git(repo, "add", ".")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-12-01T09:00:00+09:00")
    _git(repo, "commit", "-q", "-m", "freeze")
    _git(repo, "tag", "-a", "models-frozen", "-m", "frozen models")
    (repo / "results" / "sob" / "frozen_scoring").mkdir(parents=True)
    (repo / "results" / "sob" / "frozen_scoring" / "summary.json").write_text("{}")
    (repo / "REPORT.md").write_text(REPORT.format(c=reg, p=sha["PREREG.md"], s=sha["SPEC.md"],
                                                  dev="- 2026-12-02: a fault fixed.\n- 2026-12-03: another."))
    _git(repo, "add", ".")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-12-02T09:00:00+09:00")
    _git(repo, "commit", "-q", "-m", "scored")
    cfg = json.loads((ROOT / "btc/site/site.json").read_text())
    new, notes = fs.fill(repo, cfg, reproduce_command="python scripts/x.py", last_updated="2026-12-04")
    assert new["registration_commit"] == reg and new["registered_date"] == "2026-11-20"
    assert (new["frozen_tag"], new["frozen_date"]) == ("models-frozen", "2026-12-01")
    assert new["scored_date"] == "2026-12-02" and len(new["scored_ref"]) == 7      # no tag yet: the commit
    assert new["deviations"] == 2 and new["reproduce_command"] == "python scripts/x.py"
    assert new["author_name"] == cfg["author_name"] == "{{AUTHOR_NAME}}" and new["contact"] == cfg["contact"]
    assert any("checksums_path" in n for n in notes)
    assert fs.count_deviations(rep) == 0
    # the CLI writes the file; author and contact stay placeholders
    p = tmp_path / "site.json"
    p.write_text(json.dumps(cfg))
    assert fs.main(["--repo", str(repo), "--config", str(p), "--scored-ref", "HEAD"]) == 0
    d = json.loads(p.read_text())
    assert d["frozen_tag"] == "models-frozen" and d["author_name"] == "{{AUTHOR_NAME}}"
    # a registration commit whose files do not match REPORT.md is refused
    (repo / "REPORT.md").write_text(rep.replace(sha["SPEC.md"], "0" * 64))
    with pytest.raises(fs.FillError, match="SPEC.md"):
        fs.fill(repo, cfg)


def test_rerun_code_change_ignores_the_runs_own_outputs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_frozen_scoring", ROOT / "scripts" / "run_frozen_scoring.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    g = {"commit": "c" * 40, "state": "dirty", "status": ["?? results/sob/"]}
    out = ROOT / "results"
    assert m.code_changed(g, "c" * 40, out) is False                    # only the crashed run's outputs
    assert m.code_changed(g, "d" * 40, out) is True                     # another commit
    assert m.code_changed({**g, "status": ["?? results/sob/", " M btc/models/ridge.py"]}, "c" * 40, out) is True
    assert m.code_changed({**g, "status": ["?? scripts/new.py"]}, "c" * 40, out) is True
    assert m.code_changed({**g, "status": ["?? results/"]}, "c" * 40, ROOT / "results" / "x") is True
    assert m.code_changed({**g, "state": "unknown", "status": []}, "c" * 40, out) is True
