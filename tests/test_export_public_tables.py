"""`scripts/export_public_tables.py`: per-patient tables in place of the withheld per-forecast files.

Synthetic tables only. The numbers of the run are recomputed from the exported tables with a
bootstrap written here (not `btc.models.bootstrap`), as a reader of the public tables would.
"""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc.models.data import tables_from_frames
from btc.models.outputs import Writer
from btc.models.pipeline import confirmatory
from btc.models.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
WITHHELD = ("predictions_primary.csv", "yardstick_predictions.csv")
PER_FORECAST_COLS = {"y_map", "last_map", "t_issue", "t_target", "t_r1", "h_s", "issue_n", "outer_fold"}


def _module():
    spec = importlib.util.spec_from_file_location("export_public_tables", ROOT / "scripts" / "export_public_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, skiprows=1)


def _hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp") / "results"
    T = tables_from_frames("sdev", make_synthetic(seed=2, wf_signal=2.0)["sdev"])
    w = Writer(root / "sdev", {"commit": "test", "describe": "test", "data": "synthetic(seed=2)",
                               "inputs_sha256": T.sha256, "stage": "sdev"})
    confirmatory(T, w, n_jobs=2)
    w.manifest()
    return root


@pytest.fixture(scope="module")
def exported(run_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("pub")
    assert _module().main(["--results", str(run_root), "--out", str(out)]) == 0
    return out / "sdev" / "confirmatory"


# an independent subject bootstrap: subjects drawn with replacement, a patient weighted by the count of its subject
def _weights(subj, n_boot=2000, seed=20261118):
    subjects = sorted(set(subj.tolist()))
    pos = {s: i for i, s in enumerate(subjects)}
    draws = np.random.default_rng(seed).integers(0, len(subjects), size=(n_boot, len(subjects)))
    cnt = np.stack([np.bincount(d, minlength=len(subjects)) for d in draws]).astype(float)
    return cnt[:, [pos[s] for s in subj]]


def _bmean(W, v):
    return (W * v).sum(1) / W.sum(1)


def _ci(x):
    x = np.where(np.isposinf(x), 1e300, np.where(np.isneginf(x), -1e300, x))
    return tuple(np.percentile(x, [2.5, 97.5]))


def test_tables_hold_no_per_forecast_values(exported, run_root):
    src = run_root / "sdev" / "confirmatory"
    prim, yard = _csv(exported / "per_patient_primary_wide.csv"), _csv(exported / "per_patient_yardstick.csv")
    for t in (prim, yard):
        assert not PER_FORECAST_COLS & set(t.columns)
        assert not any(c.startswith("pred_") for c in t.columns)
        assert t["caseid"].is_unique and t["caseid"].is_monotonic_increasing
    pr, yp = _csv(src / "predictions_primary.csv"), _csv(src / "yardstick_predictions.csv")
    assert len(prim) == pr["caseid"].nunique() and int(prim["n_forecasts"].sum()) == len(pr)
    assert len(yard) == yp["caseid"].nunique() and int(yard["n_triples"].sum()) == len(yp)
    assert {f"rmse_{m}" for m in ("M00", "M0", "M1", "M2", "M3", "M4", "M3-P1", "M4-P1")} <= set(prim.columns)
    assert {f"rmse_{m}" for m in ("L", "LH", "T", "LH-P1", "T-P1", "U")} <= set(yard.columns)
    man = json.loads((exported / "public_tables.json").read_text())
    assert all(man["checks"].values()) and len(man["checks"]) == 3
    sdev_man = json.loads((run_root / "sdev" / "MANIFEST.json").read_text())["files"]
    for f in WITHHELD:
        assert man["sources"][f]["sha256"] == sdev_man[f"confirmatory/{f}"]
        assert man["sources"][f]["manifest"] == "sdev/MANIFEST.json"


def test_tables_reproduce_the_run(exported, run_root):
    S = json.loads((run_root / "sdev" / "confirmatory" / "summary.json").read_text())
    prim = _csv(exported / "per_patient_primary_wide.csv")
    e = prim.loc[prim["entered"]].reset_index(drop=True)
    assert len(e) == S["N"] and int(e["n_forecasts"].sum()) == S["n_forecasts"]
    assert np.isclose(prim["n_forecasts_waveform"].sum() / prim["n_forecasts"].sum(), S["waveform_coverage"])
    for sub, ref in ((e, S["contrasts"]["M4 vs M2"]),
                     (e.loc[e["fast_cycling"]].reset_index(drop=True), S["fast_cycling"]["m4_vs_m2"])):
        W = _weights(sub["subjectid"].to_numpy())
        a, b = sub["rmse_M4"].to_numpy(), sub["rmse_M2"].to_numpy()
        sa, sb = _bmean(W, a), _bmean(W, b)
        got = [a.mean() - b.mean(), *_ci(sa - sb), 1 - a.mean() / b.mean(), *_ci(1 - sa / sb)]
        want = [ref[k] for k in ("delta", "delta_lo", "delta_hi", "rr", "rr_lo", "rr_hi")]
        assert np.allclose(got, want, rtol=0, atol=1e-10)
    assert np.isclose(S["decision"]["G"], S["contrasts"]["M4 vs M2"]["rr_hi"])
    yard = _csv(exported / "per_patient_yardstick.csv")
    for sub, ref in ((yard.loc[yard["entered"]], S["yardstick_all"]),
                     (yard.loc[yard["entered"] & yard["fast_cycling"]], S["yardstick_fast_cycling"])):
        sub = sub.reset_index(drop=True)
        assert len(sub) == ref["n_patients"] and int(sub["n_triples"].sum()) == ref["n_triples"]
        W = _weights(sub["subjectid"].to_numpy())
        L, T, U = (sub[f"rmse_{m}"].to_numpy() for m in ("L", "T", "U"))
        Ns, Ds = _bmean(W, L) - _bmean(W, T), _bmean(W, L) - _bmean(W, U)
        with np.errstate(divide="ignore", invalid="ignore"):
            Fs = np.where(Ds > 0, Ns / Ds, np.where(Ns > 0, np.inf, -np.inf))
        N, D = L.mean() - T.mean(), L.mean() - U.mean()
        got = [N, *_ci(Ns), D, *_ci(Ds), N / D, *_ci(Fs)]
        want = [ref[k] for k in ("N", "N_lo", "N_hi", "D", "D_lo", "D_hi", "F", "F_lo", "F_hi")]
        assert np.allclose(got, want, rtol=0, atol=1e-10)


def test_in_place_export_changes_no_file_of_the_run(run_root, tmp_path):
    root = tmp_path / "results"
    shutil.copytree(run_root, root)
    before = _hashes(root)
    assert _module().main(["--results", str(root)]) == 0
    after = _hashes(root)
    assert {k: after[k] for k in before} == before
    assert set(after) - set(before) == {f"sdev/confirmatory/{f}" for f in
                                        ("per_patient_primary_wide.csv", "per_patient_yardstick.csv",
                                         "public_tables.json")}


def test_export_is_deterministic(run_root, tmp_path):
    m = _module()
    for d in ("a", "b"):
        assert m.main(["--results", str(run_root), "--out", str(tmp_path / d)]) == 0
    assert _hashes(tmp_path / "a") == _hashes(tmp_path / "b")


def test_export_refuses_changed_or_unrecorded_sources(run_root, tmp_path, capsys):
    m = _module()
    root = tmp_path / "results"
    shutil.copytree(run_root, root)
    p = root / "sdev" / "confirmatory" / "predictions_primary.csv"
    p.write_text(p.read_text() + "\n")
    assert m.main(["--results", str(root), "--out", str(tmp_path / "o1")]) == 2
    assert "SHA-256 differs" in capsys.readouterr().err
    (root / "sdev" / "MANIFEST.json").unlink()
    assert m.main(["--results", str(root), "--out", str(tmp_path / "o2")]) == 2
    assert "not listed in any MANIFEST.json" in capsys.readouterr().err
    assert m.main(["--results", str(tmp_path / "empty"), "--out", str(tmp_path / "o3")]) == 2


def test_export_refuses_tables_that_disagree_with_the_run(run_root, tmp_path, capsys):
    m = _module()
    root = tmp_path / "results"
    shutil.copytree(run_root, root)
    s = root / "sdev" / "confirmatory" / "summary.json"
    d = json.loads(s.read_text())
    d["contrasts"]["M4 vs M2"]["delta"] += 1e-6
    s.write_text(json.dumps(d))
    assert m.main(["--results", str(root), "--out", str(tmp_path / "o")]) == 2
    assert "differs from summary.json" in capsys.readouterr().err


def test_gitignore_withholds_only_the_per_forecast_files():
    ign = lambda p: subprocess.run(["git", "check-ignore", "-q", "--no-index", p], cwd=ROOT).returncode == 0
    for d in ("results/sdev/confirmatory", "results/sob/frozen_scoring",
              "results/sob/frozen_scoring/sensitivity_including_nmb"):
        assert ign(f"{d}/predictions_primary.csv")
        for f in ("per_patient_primary.csv", "per_patient_primary_wide.csv", "summary.json", "MANIFEST.json"):
            assert not ign(f"{d}/{f}")
    assert ign("results/sob/frozen_scoring/yardstick_predictions.csv")
    assert ign("results/sdev/confirmatory/yardstick_predictions.csv")
    for f in ("results/sdev/confirmatory/per_patient_yardstick.csv", "results/frozen/models.pkl",
              "results/frozen/coefficients.csv", "results/README.md"):
        assert not ign(f)
