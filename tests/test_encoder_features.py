"""Encoder input processing and per-forecast encoder features (ENCODER_PLAN.md §2, §4). No torch: the
encoder is replaced by a stub. Synthetic signals and tables only.

- the per-window processing: filter, z-score and resampling, their order, shapes and values;
- the aggregation (current, previous, reference) and its missing rules;
- equality with M4's window selection (counts and availability per forecast);
- the leakage test of SPEC §4.4 extended to the encoder features: every pleth and ECG sample at or after
  the cutoff is replaced with random values (and some missing ones), the window table, the processed
  windows, the embeddings and the features are recomputed, and every encoder feature must be unchanged,
  for the real and P1 variants.
"""
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import resample_poly, sosfreqz

from btc import encoder as E
from btc import provenance
from btc import waveform as W
from wf_synth import FS, Beat, make_signals

ROOT = Path(__file__).resolve().parents[1]
DIM = 16


def stub_embed(X: np.ndarray) -> np.ndarray:
    """Stand-in encoder: the first 16 FFT magnitudes of each processed window."""
    return np.abs(np.fft.rfft(np.asarray(X, dtype=np.float64), axis=1))[:, 1:DIM + 1]


# --------------------------------------------------------------------------- processing (§2)
def test_filter_design():
    assert E.SOS.shape == (4, 6)                         # order-4 band-pass = 8 poles = 4 sections
    assert E.SMOOTH_N == 25 and np.isclose(E.MA.sum(), 1.0) and np.all(E.MA == 1 / 25)
    f, h = sosfreqz(E.SOS, worN=[0.1, 2.0, 5.0, 30.0, 100.0], fs=500.0)
    g = np.abs(h)
    assert g[1] > 0.95 and g[2] > 0.95                   # pass band
    assert g[0] <= 0.1 + 1e-9 and g[3] <= 0.1 + 1e-9 and g[4] <= 0.1 + 1e-9   # >= 20 dB stop band
    assert (E.FS_OUT, E.WIN_N_IN, E.WIN_N_OUT) == (125.0, 5000, 1250)


def test_zscore_and_resample_are_as_specified():
    rng = np.random.default_rng(0)
    x = 3.0 + 2.0 * rng.standard_normal(5000)
    z = E.zscore(x)
    assert np.array_equal(z, (x - x.mean()) / (x.std(ddof=0) + 1e-7))
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1.0) < 1e-6
    assert np.array_equal(E.resample(z), resample_poly(z, 1, 4))
    assert E.resample(z).shape == (1250,)


def test_preprocess_order_shape_and_values():
    t = np.arange(5000) / FS
    x = 900.0 + 40.0 * np.sin(2 * np.pi * 1.2 * t) + 30.0 * np.sin(2 * np.pi * 0.05 * t)
    r = E.preprocess_window(x)
    assert r.shape == (1250,) and r.dtype == np.float32 and np.all(np.isfinite(r))
    # the steps in the order of the plan: filter, z-score at 500 Hz, then resample
    want = resample_poly(E.zscore(E.smooth(E.bandpass(x))), 1, 4).astype(np.float32)
    assert np.array_equal(r, want)
    other = E.zscore(resample_poly(E.smooth(E.bandpass(x)), 1, 4)).astype(np.float32)
    assert not np.allclose(r, other, atol=1e-6)
    # a 1.2-Hz pulse passes; the 0.05-Hz drift and the offset do not (zero phase: no lag)
    ref = np.sin(2 * np.pi * 1.2 * t[::4])
    mid = slice(125, 1125)
    assert np.corrcoef(r[mid], ref[mid])[0, 1] > 0.995
    assert abs(float(r.std()) - 1.0) < 0.05
    # invariant to the display scale and offset of the stored pleth
    assert np.allclose(E.preprocess_window(5.0 * x - 200.0), r, atol=1e-5)


def test_preprocess_rejects_bad_windows():
    with pytest.raises(ValueError):
        E.preprocess_window(np.zeros(4999))
    x = np.sin(np.arange(5000) / 50.0)
    x[100] = np.nan
    with pytest.raises(ValueError):
        E.preprocess_window(x)


def test_processed_dom_freq_on_a_synthetic_pulse():
    p, *_ = make_signals(30, Beat(T=0.8), noise=0.1, seed=3)
    r = E.preprocess_window(p[:5000])
    assert abs(E.processed_dom_freq(r) - 1.25) <= 0.1 + 1e-9


# --------------------------------------------------------------------------- aggregation (§2, §4)
def _table(n=20, t0=1.0, unusable=()):
    w = np.arange(n)
    return pd.DataFrame({"caseid": 1, "w": w, "t_start": t0 + 10.0 * w, "t_end": t0 + 10.0 * w + 10.0,
                         "usable": ~np.isin(w, list(unusable)), "dom_freq": 1.2})


def test_aggregation_current_previous_reference():
    win = _table(30, unusable=(2, 9))
    rng = np.random.default_rng(1)
    Eu = rng.standard_normal((int(win.usable.sum()), 4))
    ew = E.EncWindows(win, Eu, 4)
    full = np.full((30, 4), np.nan)
    full[win.usable.to_numpy()] = Eu
    t_issue, t_prev, t_pp = 251.0, 171.0, 91.0
    f = E.forecast_embedding_features(ew, t_issue, t_prev, t_pp, t_issue)
    ts, te, u = win.t_start.to_numpy(), win.t_end.to_numpy(), win.usable.to_numpy()
    cur = full[(ts >= t_prev) & (te <= t_issue) & u].mean(axis=0)            # 120-s look-back not binding
    prv = full[(ts >= t_pp) & (te <= t_prev) & u].mean(axis=0)
    ref = full[np.flatnonzero(u)[:6]].mean(axis=0)                          # windows 0,1,3,4,5,6
    assert f["enc_available"] and f["enc_n_windows"] == 8 and f["enc_n_usable"] == 8
    assert np.allclose(f["lvl"], cur - ref) and np.allclose(f["chg"], cur - prv)
    # the 120-s look-back
    g = E.forecast_embedding_features(ew, 281.0, 101.0, 21.0, 281.0)
    assert g["enc_n_windows"] == 12 and np.allclose(g["lvl"], full[(ts >= 161.0) & (te <= 281.0) & u].mean(0) - ref)


def test_missing_rules():
    # windows w span [1 + 10 w, 11 + 10 w); w = 7-10 unusable
    win = _table(15, unusable=(7, 8, 9, 10))
    ew = E.EncWindows(win, np.random.default_rng(2).standard_normal((11, 4)), 4)
    # reference: the 6th usable window (w = 5) ends at 61 s; before that the level is missing
    f = E.forecast_embedding_features(ew, 60.5, 30.5, 0.5, 60.5)
    assert f["enc_available"] and f["enc_n_usable"] == 2
    assert np.all(np.isnan(f["lvl"])) and np.all(np.isfinite(f["chg"]))
    f = E.forecast_embedding_features(ew, 61.0, 30.5, 0.5, 61.0)
    assert f["enc_n_usable"] == 3 and np.all(np.isfinite(f["lvl"]))
    # fewer than 2 usable windows: nothing (M4-enc falls back to M3)
    f = E.forecast_embedding_features(ew, 111.0, 70.0, 50.0, 111.0)
    assert f["enc_n_windows"] == 4 and f["enc_n_usable"] == 0 and not f["enc_available"]
    assert np.all(np.isnan(f["lvl"])) and np.all(np.isnan(f["chg"]))
    # previous interval without 2 usable windows: change missing, level present
    f = E.forecast_embedding_features(ew, 131.0, 110.5, 95.0, 131.0)
    assert f["enc_available"] and f["enc_n_usable"] == 2
    assert np.all(np.isfinite(f["lvl"])) and np.all(np.isnan(f["chg"]))


def test_inputs_are_times_only():
    assert list(inspect.signature(E.forecast_embedding_features).parameters) == \
        ["ew", "t_issue", "t_prev", "t_prev2", "t_end"]


def test_embeddings_must_match_usable_windows():
    win = _table(10, unusable=(3,))
    with pytest.raises(ValueError):
        E.EncWindows(win, np.ones((10, 4)), 4)


# --------------------------------------------------------------------------- same windows as M4
def _signals(duration, beat, pat, t0, seed):
    p, e, *_ = make_signals(duration, beat, pat=pat, t0=t0, noise=0.3, seed=seed)
    return {"pleth": W.Wave(p, t0, FS), "ecg": W.Wave(e, t0, FS)}


@pytest.fixture(scope="module")
def two_cases():
    own = _signals(900, Beat(), 0.62, 1.0, 1)
    donor = _signals(900, Beat(T=0.75, N=0.18), 0.6, 0.4, 2)
    out = {}
    for cid, s in ((11, own), (22, donor)):
        win = W.case_windows(cid, s["pleth"], s["ecg"])
        out[cid] = (s, win)
    return out


# valid reading times of the synthetic case; issue n (from the 3rd reading) has t_issue = reading n
READS = [3.0, 12.0, 45.3, 150.2, 211.9, 300.4, 380.0, 455.1, 612.7, 700.0, 800.0, 832.0, 845.3]
ALL_ISSUES = [(READS[n - 1], READS[n - 2], READS[n - 3]) for n in range(3, len(READS) + 1)]
ISSUES = [ALL_ISSUES[i] for i in (0, 3, 5, 6, 8, 10)]     # early (no reference), 13-s interval, late


def _enc_windows(cid, s, win):
    r = E.embed_case(cid, s["pleth"], win, stub_embed)
    return E.EncWindows(win, r["E"], DIM)


def test_counts_equal_m4_window_selection(two_cases):
    for cid, (s, win) in two_cases.items():
        ew, cw = _enc_windows(cid, s, win), W.CaseWindows(win)
        n_avail = 0
        for ti, tp, tpp in ALL_ISSUES:
            f = E.forecast_embedding_features(ew, ti, tp, tpp, ti)
            g = W.forecast_features(cw, ti, tp, tpp, ti, "usable")
            assert (f["enc_n_windows"], f["enc_n_usable"], f["enc_available"]) == \
                (g["wf_n_windows"], g["wf_n_usable"], g["wf_available"])
            n_avail += f["enc_available"]
        assert n_avail >= 6


def _wf_builder():
    spec = importlib.util.spec_from_file_location("build_features_wf", ROOT / "scripts" / "build_features_wf.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_feature_table_and_hard_check(two_cases):
    """`build_features` against `scripts/build_features_wf.py::build` on the same issues and windows: the
    hard check passes, and fails when a count is changed."""
    t_of = {(11, i + 1): r for i, r in enumerate(READS)}
    issues = pd.DataFrame([{"caseid": 11, "window": "primary", "issue_n": n, "t_issue": READS[n - 1],
                            "t_prev": READS[n - 2]} for n in range(3, len(READS) + 1)])
    cases = pd.DataFrame({"caseid": [11, 22], "donor_caseid": [22.0, np.nan]})
    win = pd.concat([two_cases[11][1], two_cases[22][1]], ignore_index=True)
    wf = _wf_builder().build(issues, t_of, cases, win)
    ews = {cid: _enc_windows(cid, s, w) for cid, (s, w) in two_cases.items()}
    enc = E.build_features(issues, t_of, cases, ews, dim=DIM)
    assert len(enc) == 2 * len(issues) and set(enc["variant"]) == {"real", "p1"}
    assert enc.loc[enc["variant"] == "p1", "signal_caseid"].eq(22.0).all()
    chk = E.check_same_windows(enc, wf)
    assert chk["rows_compared"] == len(enc) and chk["rows_available"] >= 12
    bad = enc.copy()
    bad.loc[0, "enc_n_usable"] += 1
    with pytest.raises(ValueError):
        E.check_same_windows(bad, wf)


def test_case_inputs_checks_the_window_grid(two_cases):
    s, win = two_cases[11]
    with pytest.raises(ValueError):
        E.case_inputs(11, s["pleth"], win.iloc[:-1])
    shifted = win.assign(t_start=win["t_start"] + 0.002)
    with pytest.raises(ValueError):
        E.case_inputs(11, s["pleth"], shifted)
    ws, X = E.case_inputs(11, s["pleth"], win)
    assert X.shape == (int(win["usable"].sum()), 1250) and np.array_equal(ws, win.loc[win.usable, "w"].to_numpy())


# --------------------------------------------------------------------------- leakage (SPEC §4.4)
def _perturb(w, t_end, rng):
    x = w.x.copy()
    k = w.t0 + np.arange(x.size) / w.srate >= t_end
    sd = np.nanstd(w.x) or 1.0
    x[k] = np.nanmean(w.x) + 3 * sd * rng.standard_normal(int(k.sum()))
    x[k & (rng.random(x.size) < 0.001)] = np.nan
    return W.Wave(x, w.t0, w.srate)


def _enc_features(src, times, variant, embed_fn=stub_embed, dim=DIM):
    ti, tp, tpp, te, mask = W.variant_times(variant, *times)
    assert mask == "usable"
    win = W.case_windows(0, src["pleth"], src["ecg"])
    r = E.embed_case(0, src["pleth"], win, embed_fn)
    return E.forecast_embedding_features(E.EncWindows(win, r["E"], dim), ti, tp, tpp, te), te


def check_leakage(own, donor, issues, embed_fn=stub_embed, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    n_avail = n_lvl = 0
    for times in issues:
        for variant in E.VARIANTS:
            src = donor if variant == "p1" else own
            before, te = _enc_features(src, times, variant, embed_fn, dim)
            pert = {"pleth": _perturb(src["pleth"], te, rng), "ecg": _perturb(src["ecg"], te, rng)}
            after, _ = _enc_features(pert, times, variant, embed_fn, dim)
            for k in before:
                a, b = np.asarray(before[k], dtype=float), np.asarray(after[k], dtype=float)
                assert np.array_equal(a, b, equal_nan=True), (variant, times, k)
            n_avail += before["enc_available"]
            n_lvl += bool(np.all(np.isfinite(before["lvl"])) and np.all(np.isfinite(before["chg"])))
    return n_avail, n_lvl


def test_leakage_synthetic(two_cases):
    own, donor = two_cases[11][0], two_cases[22][0]
    n_avail, n_lvl = check_leakage(own, donor, ISSUES)
    assert n_avail >= 8 and n_lvl >= 6          # the test exercises real feature values


# --------------------------------------------------------------------------- the two build scripts
def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stub512(X):
    return np.abs(np.fft.rfft(np.asarray(X, dtype=np.float64), axis=1))[:, 1:E.EMB_DIM + 1].astype(np.float32)


def test_build_scripts_end_to_end(tmp_path, monkeypatch, two_cases):
    """scripts/build_embeddings.py (in-process) and scripts/build_features_enc.py on two synthetic cases in a
    temporary derived folder: stub encoder, stand-in git state and plan record, no real data."""
    from btc import paths as bpaths
    clean = {"commit": "c" * 40, "describe": "x", "state": "clean", "dirty": False, "status": []}
    der = tmp_path / "derived"
    der.mkdir()
    monkeypatch.setattr(bpaths, "DERIVED", der)
    monkeypatch.delenv("BTC_FROZEN_SCORING", raising=False)
    win = pd.concat([two_cases[11][1], two_cases[22][1]], ignore_index=True)
    bpaths_meta = {"git": clean}
    provenance.write_parquet(win, der / "sdev_windows.parquet", bpaths_meta)
    (der / "sdev_windows_log.json").write_text(json.dumps({"cases": [{"caseid": c, "pleth": "ok", "ecg": "ok"}
                                                                     for c in (11, 22)]}))
    t_of = {(11, i + 1): r for i, r in enumerate(READS)}
    issues = pd.DataFrame([{"caseid": 11, "window": "primary", "issue_n": n, "t_issue": READS[n - 1],
                            "t_prev": READS[n - 2], "k": 1} for n in range(3, len(READS) + 1)])
    cases = pd.DataFrame({"caseid": [11, 22], "donor_caseid": [22.0, np.nan], "pleth_status": "ok",
                          "ecg_status": "ok"})
    reads = pd.DataFrame({"caseid": 11, "n": np.arange(1, len(READS) + 1, dtype=float), "t": READS, "valid": True})
    for name, df in (("instances", issues), ("readings", reads), ("cases", cases)):
        provenance.write_parquet(df, der / f"sdev_{name}.parquet", bpaths_meta)
    wfb = _wf_builder()
    wf = wfb.build(issues[["caseid", "window", "issue_n", "t_issue", "t_prev"]], t_of, cases, win)
    provenance.write_parquet(wf, der / "sdev_features_wf.parquet", bpaths_meta)
    # build_embeddings
    be = _script("build_embeddings")
    plan = {"file": "ENCODER_PLAN.md", "sha256": "p", "committed_at_head": True, "last_commit": "a" * 40}
    monkeypatch.setattr(be.provenance, "git_state", lambda: dict(clean))
    monkeypatch.setattr(be.E, "plan_record", lambda: dict(plan))
    monkeypatch.setattr(be.E, "check_weights", lambda p: {"size": 1, "md5": "m", "sha256": "w"})
    monkeypatch.setattr(be.E, "configure_torch", lambda: {})
    monkeypatch.setattr(be.E, "runtime_info", lambda: {"torch": "stub"})
    monkeypatch.setattr(be.E, "load_model", lambda p: (None, {}))
    monkeypatch.setattr(be.E, "embed", lambda model, X: stub512(X))
    signals = {11: two_cases[11][0], 22: two_cases[22][0]}
    monkeypatch.setattr(be.W, "load_case", lambda c, cohort: {"pleth": signals[c]["pleth"]})
    for k in ("OUT_NPY", "OUT_INDEX", "OUT_LOG"):
        monkeypatch.setattr(be, k, der / getattr(be, k).name)
    assert be.main(["--jobs", "1"]) == 0
    log = json.loads((der / "sdev_enc_window_embeddings_log.json").read_text())
    assert log["n_windows_embedded"] == int(win["usable"].sum()) and log["n_cases"] == 2
    assert log["raw_files_read"] == ["data/raw/sdev/11.npz", "data/raw/sdev/22.npz"]
    assert log["dom_freq_check"]["share_agree"] > 0.9
    assert log["preprocessing"]["note"] == E.FILTER_NOTE and log["runtime"]["torch"] == "stub"
    with pytest.raises(SystemExit):
        be.main(["--jobs", "1"])                                     # never replaced in place
    # build_features_enc
    bf = _script("build_features_enc")
    monkeypatch.setattr(bf.provenance, "git_state", lambda: dict(clean))
    monkeypatch.setattr(bf.E, "plan_record", lambda: dict(plan))
    for k in ("OUT", "OUT_LOG", "EMB", "EMB_INDEX", "EMB_LOG"):
        monkeypatch.setattr(bf, k, der / getattr(bf, k).name)
    assert bf.main([]) == 0
    flog = json.loads((der / "sdev_enc_features_log.json").read_text())
    assert flog["same_windows_as_m4"]["rows_compared"] == 2 * len(issues)
    out = pd.read_parquet(der / "sdev_enc_features.parquet")
    assert list(out.columns[:11]) == E.KEY_COLS and len(out.columns) == 11 + 1024
    assert out["enc_available"].sum() >= 12

