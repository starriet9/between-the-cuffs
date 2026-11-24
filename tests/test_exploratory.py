"""Exploratory lead-lag and detection-limit analyses (EXPLORATORY_PLAN.md; btc/models/exploratory.py,
scripts/run_exploratory.py).

Synthetic derived tables only (`btc.models.synthetic`); no real table, signal or network is read. The
synthetic generator draws the change features independently of the level features, so `consistent` rebuilds
each change as the level at r minus the level at r-1 wherever both exist, as in the real tables (plan §1.1).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc.models import exploratory as X
from btc.models import pipeline as P
from btc.models.analyses import Scored, m2_chain, run_cv
from btc.models.bootstrap import SubjectBootstrap
from btc.models.constants import N_BOOT, SEED, WF_COLS
from btc.models.data import build_frame, tables_from_frames
from btc.models.design import Cols
from btc.models.folds import make_fold_plan
from btc.models.outputs import Writer
from btc.models.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
CLEAN = {"commit": "c" * 40, "describe": "x", "state": "clean", "dirty": False, "status": [], "tags_at_head": []}


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def consistent(frames: dict) -> dict:
    """Synthetic frames whose change columns equal the level at r minus the level at r-1 where both exist."""
    out = dict(frames)
    for name, pairs in (("features_hr", [("hr_level", "hr_change")]),
                        ("features_wf", [(f"wf_{f}_lvl", f"wf_{f}_chg") for f in ("pat", "rise", "width", "bav")])):
        d = frames[name].copy()
        keys = ["caseid", "window", "variant"]
        prev = d[keys + ["issue_n"] + [p[0] for p in pairs]].assign(issue_n=d["issue_n"] + 1)
        m = d[keys + ["issue_n"]].merge(prev, on=keys + ["issue_n"], how="left", suffixes=("", "_prev"))
        for lvl, chg in pairs:
            diff = d[lvl].to_numpy(float) - m[lvl].to_numpy(float)
            d[chg] = np.where(np.isfinite(diff), diff, d[chg].to_numpy(float))
        out[name] = d
    return out


def _tables(seed=1, wf_signal=0.0):
    return tables_from_frames("sdev", consistent(make_synthetic(seed, wf_signal=wf_signal)["sdev"]))


def _conf(T, root: Path, n_boot: int):
    meta = {"commit": "t" * 40, "describe": "test", "data": "synthetic", "stage": "sdev", "inputs_sha256": T.sha256}
    cw = Writer(root / "sdev", meta)
    P.confirmatory(T, cw, n_jobs=2, n_boot=n_boot)
    cw.manifest()
    return json.loads((root / "sdev" / "confirmatory" / "summary.json").read_text())


@pytest.fixture(scope="module")
def signal(tmp_path_factory):
    T = _tables(2, wf_signal=4.0)
    return T, _conf(T, tmp_path_factory.mktemp("sig") / "results", 200)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    T = _tables(1)
    return T, _conf(T, tmp_path_factory.mktemp("null") / "results", 200)


# ----------------------------------------------------------------------------
# Lead-lag: deltas, centring, bootstrap
# ----------------------------------------------------------------------------

def test_case_sums_centring_and_minimum():
    rng = np.random.default_rng(0)
    cid = np.repeat([1, 2, 3, 4], [6, 5, 2, 4])
    x = rng.normal(size=len(cid))
    y = 0.5 * x + rng.normal(size=len(cid))
    x[3] = np.nan                                   # case 1 keeps 5 pairs
    cases = np.array([1, 2, 3, 4, 5])
    cs = X.case_sums(cid, x, y, cases)
    assert list(cs["n"]) == [5, 5, 2, 4, 0] and list(cs["enter"]) == [True, True, False, True, False]
    assert cs["n_cases"] == 3 and cs["n_pairs"] == 14 and cs["n_cases_below_min"] == 1 and cs["n_pairs_below_min"] == 2
    assert np.all(cs["S"][[2, 4]] == 0)
    # the statistic by hand: centre each entering case at its own means, one Pearson correlation of the pool
    xs, ys = [], []
    for c in (1, 2, 4):
        m = (cid == c) & np.isfinite(x)
        xs.append(x[m] - x[m].mean()); ys.append(y[m] - y[m].mean())
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    assert X.within_r(cs["S"]) == pytest.approx(np.sum(xs * ys) / np.sqrt(np.sum(xs ** 2) * np.sum(ys ** 2)), abs=1e-12)
    # case-centring removes any per-case offset in x and y
    off = {1: 40.0, 2: -15.0, 3: 7.0, 4: 100.0}
    shift = np.array([off[c] for c in cid])
    cs2 = X.case_sums(cid, x + shift, y - 3 * shift, cases)
    assert X.within_r(cs2["S"]) == pytest.approx(X.within_r(cs["S"]), abs=1e-12)
    # between-case structure alone gives no within-case correlation
    xb = np.array([off[c] for c in cid]) + rng.normal(0, 1e-3, len(cid))
    assert abs(X.within_r(X.case_sums(cid, xb, xb + rng.normal(size=len(cid)), cases)["S"])) < 0.5
    with pytest.raises(RuntimeError):
        X.case_sums(cid, x, y, np.array([2, 3, 4]))          # case 1 has 5 pairs but no bootstrap column


def test_bootstrap_resample_equals_explicit_resample():
    """A bootstrap weight vector applied to per-case sums equals the statistic on the explicitly resampled
    data (a drawn case brings all its pairs; its centring is unchanged)."""
    rng = np.random.default_rng(1)
    cid = np.repeat(np.arange(1, 31), rng.integers(3, 9, 30))
    x = rng.normal(size=len(cid))
    y = 0.3 * x + rng.normal(size=len(cid))
    cases = np.arange(1, 31)
    bs = SubjectBootstrap(cases, cases * 10, n_boot=20, seed=SEED)
    cs = X.case_sums(cid, x, y, cases)
    rb = X.within_r_boot(bs.W, cs["S"])
    for b in range(20):
        xs, ys = [], []
        for j, c in enumerate(cases):
            for _ in range(int(bs.W[b, j])):
                m = cid == c
                xs.append(x[m] - x[m].mean()); ys.append(y[m] - y[m].mean())
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        assert rb[b] == pytest.approx(np.sum(xs * ys) / np.sqrt(np.sum(xs ** 2) * np.sum(ys ** 2)), abs=1e-12)


def test_change_identity():
    T = _tables(1)
    rows = X.change_identity(T)
    assert {(r["feature"], r["variant"]) for r in rows} == {(f[0], v) for f in X.LL_FEATURES for v in X.LL_VARIANTS}
    assert all(r["max_abs_difference"] == 0 and r["n_level_difference_without_change"] == 0 and r["n_both"] > 100
               for r in rows)
    raw = tables_from_frames("sdev", make_synthetic(1)["sdev"])     # independent synthetic changes
    with pytest.raises(RuntimeError, match="not the level difference"):
        X.change_identity(raw)
    assert max(r["max_abs_difference"] for r in X.change_identity(raw, require=False)) > 0.1


def test_map_changes(null):
    T, conf = null
    frame = build_frame(T)
    pred = frame["last_map"].to_numpy(float) + 1.5
    d = X.map_changes(T, frame, pred)
    r = T.readings.loc[T.readings["valid"]].set_index(["caseid", "n"])["map"]
    i = 17
    c, n = int(frame["caseid"].iloc[i]), int(frame["issue_n"].iloc[i])
    assert d["concurrent"][i] == pytest.approx(r.loc[(c, n)] - r.loc[(c, n - 1)], abs=1e-12)
    assert np.allclose(d["next"], frame["y_map"] - frame["last_map"])
    assert np.allclose(d["next_residual_m2"], frame["y_map"] - pred)
    bad = frame.copy()
    bad.loc[bad.index[3], "cuff_lag1"] += 1.0
    with pytest.raises(RuntimeError):
        X.map_changes(T, bad, pred)


def test_lead_lag_finds_a_planted_tracking_signal(signal):
    """With a synthetic pulse arrival time that tracks the latent MAP (negatively), the concurrent within-case
    correlation is negative and clearly so; the donor version is not; the bootstrap patients are those of the
    primary outcome."""
    T, conf = signal
    res = X.lead_lag(T, conf, n_boot=200)
    t = res["table"].set_index(["feature", "variant", "map_change"])
    assert len(t) == 5 * 3 * 3
    pat = t.loc[("pat", "real", "concurrent")]
    assert pat["r"] < -0.2 and pat["r_ci_hi"] < 0 and pat["n_cases"] > 150
    plc = t.loc[("pat", "p1", "concurrent")]
    assert plc["r_ci_lo"] < 0 < plc["r_ci_hi"]
    assert t.loc[("pat", "real minus p1", "concurrent")]["r_ci_hi"] < 0
    for feat in ("width", "bav"):                               # pure noise in the synthetic tables
        z = t.loc[(feat, "real", "concurrent")]
        assert abs(z["r"]) < 0.15
    s = res["summary"]
    assert s["n_bootstrap_patients"] == conf["N"] and s["m2_reproduction"]["patient_mean_rmse"] == pytest.approx(
        conf["contrasts"]["M4 vs M2"]["mean_comp"], abs=1e-12)
    # same resamples as the primary outcome: Scored on the confirmatory forecasts' patients, same seed
    frame = build_frame(T)
    F = Cols(frame)
    pm2 = run_cv({"m": (F, m2_chain("a", "pooled"), make_fold_plan(T.analysed, SEED))})["m"].stage("M2")
    sc = Scored(frame, {"M2": pm2}, n_boot=200)
    cs = X.case_sums(frame["caseid"].to_numpy(int), frame["wf_pat_chg@real"].to_numpy(float),
                     X.map_changes(T, frame, pm2)["concurrent"], sc.enter)
    from btc.models.bootstrap import percentile_ci
    lo, hi = percentile_ci(X.within_r_boot(sc.bs.W, cs["S"]))
    assert (lo, hi) == pytest.approx((pat["r_ci_lo"], pat["r_ci_hi"]), abs=1e-12)


def test_lead_lag_refuses_other_confirmatory_values(null):
    T, conf = null
    bad = json.loads(json.dumps(conf))
    bad["contrasts"]["M4 vs M2"]["mean_comp"] += 1e-6
    with pytest.raises(RuntimeError, match="M2"):
        X.lead_lag(T, bad, n_boot=50)
    bad = {**conf, "N": conf["N"] + 1}
    with pytest.raises(RuntimeError, match="patients"):
        X.lead_lag(T, bad, n_boot=50)


# ----------------------------------------------------------------------------
# Detection limit: construction, oracle, detection
# ----------------------------------------------------------------------------

def test_strength_constants():
    for q in X.STRENGTHS[1:]:
        k2, b = X.kappa(q) ** 2, X.beta(q)
        assert b == pytest.approx(1 / (1 + k2), abs=1e-12)
        assert k2 / (1 + k2) == pytest.approx((1 - q) ** 2, abs=1e-12)      # expected MSE factor of the oracle
    assert X.plant_name(SEED, 0.05) == "plant_s20261118_q050" and X.plant_name(SEED, 0.2) == "plant_s20261118_q200"


def test_planted_column_is_per_row_and_per_patient():
    """s depends only on the row's own target, its own case's sigma_c and the noise (plan §2.3); with a cuff that
    explains nothing, the oracle lowers each case's RMSE by about q."""
    rng = np.random.default_rng(3)
    cid = np.repeat(np.arange(20), 4000)
    scale = np.repeat(rng.uniform(2, 20, 20), 4000)
    y = rng.normal(0, 1, len(cid)) * scale
    sig = X.patient_rms(cid, y)
    assert np.allclose(sig[:4000], np.sqrt(np.mean(y[:4000] ** 2)))
    z = X.noise(len(cid), SEED)
    assert np.array_equal(z, np.random.default_rng(SEED).standard_normal(len(cid)))
    for q in (0.05, 0.2):
        s = X.planted_column(y, sig, z, q)
        b = X.beta(q)
        e = y - b * s
        ratio = np.sqrt(pd.Series(e ** 2).groupby(cid).mean() / pd.Series(y ** 2).groupby(cid).mean())
        assert np.allclose(ratio, 1 - q, atol=0.03)                 # the same strength in every case
        y2 = y.copy()
        y2[cid == 7] *= 3.0                                         # another case's targets change ...
        s2 = X.planted_column(y2, X.patient_rms(cid, y2), z, q)
        assert np.array_equal(s2[cid != 7], s[cid != 7])            # ... and no other case's planted value does
    assert np.array_equal(X.planted_column(y, sig, z, 0.0), sig * z)


def test_oracle_forecast():
    frame = pd.DataFrame({"last_map": [80.0, 90.0, 70.0], "wf_available@real": [True, False, True]})
    m2 = np.array([82.0, 88.0, 71.0])
    s = np.array([4.0, -9.0, -2.0])
    o = X.oracle_forecast(frame, m2, s, 0.2)
    b = X.beta(0.2)
    assert o[0] == pytest.approx(80 + (1 - b) * 2 + b * 4) and o[1] == 88.0 and o[2] == pytest.approx(70 + (1 - b) + b * -2)
    assert np.array_equal(X.oracle_forecast(frame, m2, s, 0.0), m2)


def test_planted_chain_layout():
    spec = X.planted_chain("a", "pooled", "plant_x")
    assert spec.stages == ("M2", "M3", X.PLANTED)
    hr, wf = spec.blocks
    assert wf.cols == tuple(f"{c}@real" for c in WF_COLS) + ("plant_x@real",) and wf.requires == "wf_available@real"
    assert hr.requires is None and hr.cols == ("hr_level@real", "hr_change@real", "hr_slope@real")


def test_detection_found_at_high_strength_not_at_zero(null):
    T, conf = null
    res = X.detection_limit(T, conf, n_jobs=2, n_boot=200, strengths=(0.0, 0.05, 0.2), seeds=(SEED, 20261119))
    m = res["main"].set_index("strength_nominal")
    assert list(m.index) == [0.0, 0.05, 0.2]
    zero, high = m.loc[0.0], m.loc[0.2]
    assert zero["oracle_rr"] == 0.0 and not zero["meets_rule"] and not zero["delta_ci_below_zero"]
    assert 0.12 < high["oracle_rr"] < 0.25
    assert high["meets_rule"] and high["delta_ci_below_zero"] and high["rr"] > 0.10
    assert all(high[f"wf_penalty_fold{j}"] != "dropped" for j in range(5))
    assert m.loc[0.05, "oracle_rr"] > 0.02
    r = res["replicate_summary"].set_index("strength_nominal")
    assert r.loc[0.2, "n_draws"] == 1 and r.loc[0.2, "n_meets_rule"] == 1 and r.loc[0.0, "n_meets_rule"] == 0
    assert res["summary"]["checks"]["plain_m4_vs_m2"]["delta"] == pytest.approx(conf["contrasts"]["M4 vs M2"]["delta"],
                                                                               abs=1e-12)
    pen = res["penalties"]
    assert len(pen) == 5 * 3 * (1 + 3 * 2) and set(pen["model"]) == {"M2", "M3", "M4", X.PLANTED}
    # no per-forecast value in any output table
    for k in ("main", "replicates", "replicate_summary", "penalties"):
        assert not {"caseid", "issue_n", "y_map", "last_map"} & set(res[k].columns)


def test_detection_refuses_other_confirmatory_values(null):
    T, conf = null
    bad = json.loads(json.dumps(conf))
    bad["contrasts"]["M4 vs M2"]["delta_hi"] += 1e-6
    with pytest.raises(RuntimeError, match="confirmatory"):
        X.detection_limit(T, bad, n_boot=200, strengths=(0.0,), seeds=(SEED,))


# ----------------------------------------------------------------------------
# Guards and the script end to end
# ----------------------------------------------------------------------------

def test_start_problems(tmp_path, monkeypatch):
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    plan = {"sha256": "p", "committed_at_head": True}
    conf = {"meta": {"inputs_sha256": {"a": 1}}}
    out = tmp_path / "out"
    assert X.start_problems(CLEAN, plan, conf, {"a": 1}, True, out) == []
    probs = X.start_problems({**CLEAN, "state": "dirty"}, {**plan, "committed_at_head": False}, conf, {"a": 2},
                             False, out)
    for k in ("not clean", "not committed", "inputs_sha256", "fold plan"):
        assert any(k in p for p in probs), k
    assert any("no confirmatory" in p for p in X.start_problems(CLEAN, plan, None, {"a": 1}, True, out))
    out.mkdir()
    assert any("rerun is a departure" in p for p in X.start_problems(CLEAN, plan, conf, {"a": 1}, True, out))
    monkeypatch.setenv("BTC_FROZEN_SCORING", "0")
    assert any("BTC_FROZEN_SCORING" in p for p in X.start_problems(CLEAN, plan, conf, {"a": 1}, True, tmp_path / "x"))


def test_plan_record_of_this_repository():
    r = X.plan_record()
    assert r["file"] == "EXPLORATORY_PLAN.md" and len(r["sha256"]) == 64 and r["last_commit"]


def test_run_script_end_to_end(tmp_path, monkeypatch):
    """Both commands through the real-data branch of the script, on synthetic tables and a synthetic
    confirmatory run (2,000 resamples, as the script uses); git state and plan record replaced by stand-ins."""
    T = _tables(1)
    m = _script("run_exploratory")
    results, derived = tmp_path / "results", tmp_path / "derived"
    _conf(T, results, N_BOOT)
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    for k, v in {"RESULTS": results, "CONF": results / "sdev" / "confirmatory", "DERIVED": derived,
                 "RUN_LOG": derived / "sdev_exploratory_run_log.jsonl"}.items():
        monkeypatch.setattr(m, k, v)
    git = {"g": dict(CLEAN)}
    monkeypatch.setattr(m, "git_meta", lambda: dict(git["g"]))
    monkeypatch.setattr(m, "load_tables", lambda d, cohort: T)
    monkeypatch.setattr(m.X, "plan_record", lambda: {"file": "EXPLORATORY_PLAN.md", "sha256": "p" * 64,
                                                     "committed_at_head": True, "last_commit": "a" * 40})
    git["g"] = {**CLEAN, "state": "dirty"}
    assert m.main(["lead-lag"]) == 2
    git["g"] = dict(CLEAN)
    assert m.main(["lead-lag"]) == 0 and m.main(["detection-limit", "--n-jobs", "2"]) == 0
    assert m.main(["lead-lag"]) == 2 and m.main(["detection-limit"]) == 2        # the first run is the result
    ll, dl = results / "sdev" / "exploratory_lead_lag", results / "sdev" / "exploratory_detection_limit"
    for d, names in ((ll, {"lead_lag.csv", "change_identity.csv", "summary.json", "summary.md"}),
                     (dl, {"detection_limit.csv", "replicates.csv", "replicate_summary.csv", "penalties.csv",
                           "summary.json", "summary.md"})):
        man = json.loads((d / "MANIFEST.json").read_text())
        assert set(man["files"]) == names and {p.name for p in d.iterdir()} == names | {"MANIFEST.json"}
        assert man["meta"]["plan"]["sha256"] == "p" * 64 and man["meta"]["commit"] == "c" * 40
        for n in names:
            text = (d / n).read_text()
            assert "y_map" not in text and "last_map" not in text, n
        for n in (x for x in names if x.endswith(".csv")):
            cols = set(pd.read_csv(d / n, skiprows=1).columns)
            assert not {"caseid", "issue_n", "t_issue"} & cols, n
    s = json.loads((dl / "summary.json").read_text())
    assert [r["strength_nominal"] for r in s["main"]] == list(X.STRENGTHS)
    assert len(pd.read_csv(dl / "replicates.csv", skiprows=1)) == len(X.STRENGTHS) * len(X.REPLICATE_SEEDS)
    assert "Derived from VitalDB" in (ll / "summary.md").read_text()
    events = [json.loads(l)["event"] for l in (derived / "sdev_exploratory_run_log.jsonl").read_text().splitlines()]
    assert events == ["refused", "started", "completed", "started", "completed", "refused", "refused"]
