"""No test-fold leakage in the scaler, imputer, indicator merging or penalty choice (SPEC §4.1, §4.5),
and no use of reading r+1's values in the yardstick models L and T (SPEC §3, §4.4)."""
import numpy as np
import pandas as pd
import pytest

from btc.models.analyses import ladder, yard_T_chain
from btc.models.data import build_frame, build_triples, tables_from_frames
from btc.models.design import Cols
from btc.models.folds import make_fold_plan
from btc.models.ridge import fit_chain, own_bins
from btc.models.synthetic import _instances, make_synthetic


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(seed=11, wf_signal=2.0, hr_signal=1.0)


def _state(cf):
    out = []
    for key, node in sorted(cf.nodes.items()):
        out.append((key, tuple(node.lam)))
        for st in node.stages:
            if st.dropped:
                out.append("dropped")
                continue
            out.append((st.b0, st.beta.tobytes(), st.design.mean.tobytes(), st.design.sd.tobytes(),
                        tuple(m.tobytes() if m is not None else None for m in st.design.med),
                        tuple(map(tuple, st.design.keep))))
    return out


@pytest.mark.parametrize("structure", ["pooled", "perbin"])
def test_fold_model_ignores_its_test_fold(synth, structure):
    T = tables_from_frames("sdev", synth["sdev"])
    frame = build_frame(T)
    plan = make_fold_plan(T.analysed)
    spec = ladder("a", structure)
    F = Cols(frame)
    bins = own_bins(F, spec)
    j = 2
    outer = plan.outer_of(F.caseid)
    tr, te = np.flatnonzero(outer != j), np.flatnonzero(outer == j)
    inner = np.full(F.n, -1)
    inner[tr] = plan.inner_of(j, F.caseid[tr])
    ref = fit_chain(F, spec, tr, inner, bins)
    # perturb every target and feature of the test fold (h kept: per-bin counts are over all S-dev, §4.1)
    keep = {"caseid", "subjectid", "window", "issue_n", "k", "h_s", "h_min", "h_bin", "hmed_bin", "sex"}
    pert = frame.copy()
    rng = np.random.default_rng(0)
    for c in pert.columns:
        if c in keep:
            continue
        if pert[c].dtype == bool:
            pert.loc[te, c] = rng.random(len(te)) < 0.5
        elif np.issubdtype(pert[c].dtype, np.number):
            pert[c] = pert[c].astype(float)
            pert.loc[te, c] = rng.normal(500, 300, len(te))
    Fp = Cols(pert)
    alt = fit_chain(Fp, spec, tr, inner, bins)
    assert _state(ref) == _state(alt)
    assert np.array_equal(ref.predict(F, te), alt.predict(F, te))


def test_inner_folds_use_training_cases_only(synth):
    T = tables_from_frames("sdev", synth["sdev"])
    plan = make_fold_plan(T.analysed)
    for j in range(5):
        assert np.all(plan.inner[j][plan.outer == j] == -1)
        assert np.all(plan.inner[j][plan.outer != j] >= 0)
    # inner folds of fold j do not change when the test cases' classes change
    a = T.analysed.copy()
    te = plan.outer == 0
    a2 = a.copy()
    a2.loc[te, "fast_cycling"] = ~a2.loc[te, "fast_cycling"].astype(bool)
    p2 = make_fold_plan(a2)
    if np.array_equal(p2.outer, plan.outer):
        assert np.array_equal(p2.inner[0], plan.inner[0])


def test_triples_do_not_use_values_of_reading_r_plus_1(synth):
    """Perturb the values of one reading q; the L/T inputs of the triple with r + 1 = q must not
    change (only U and targets may)."""
    S = {k: v.copy() for k, v in synth["sdev"].items()}
    T = tables_from_frames("sdev", S)
    tri, _ = build_triples(T)
    cid = int(tri["caseid"].iloc[len(tri) // 2])
    q = int(tri.loc[tri["caseid"] == cid, "issue_n"].iloc[1]) + 1      # reading r + 1
    rd = S["readings"]
    c = S["cases"].set_index("caseid").loc[cid]
    idx = rd.index[(rd["caseid"] == cid) & (rd["n"] == q)][0]
    rd2 = rd.copy()
    rd2.loc[idx, ["map", "sbp", "dbp"]] = [140.0, 190.0, 110.0]
    new = pd.DataFrame(_instances(cid, int(c["subjectid"]), bool(c["fast_cycling"]),
                                  rd2.loc[rd2["caseid"] == cid], float(c["opstart"])))
    inst = pd.concat([S["instances"].loc[S["instances"]["caseid"] != cid], new], ignore_index=True)
    T2 = tables_from_frames("sdev", {**S, "readings": rd2, "instances": inst})
    tri2, u2 = build_triples(T2)
    key = lambda d: d.loc[(d["caseid"] == cid) & (d["issue_n"] == q - 1)]
    a, b = key(tri), key(tri2)
    spec = yard_T_chain("a")
    des_cols = ["cuff_level", "cuff_slope", "cuff_curv", "cuff_pp", "cuff_tsr_min", "cuff_lag0", "cuff_lag1",
                "cuff_lag2", "cuff_gap1_min", "cuff_gap2_min", "h_min", "elapsed_min", "h2_min", "last_map",
                *[c for blk in spec.blocks for c in blk.cols], "wf_available@real"]
    pd.testing.assert_frame_equal(a[des_cols].reset_index(drop=True), b[des_cols].reset_index(drop=True))
    ub = u2.loc[(u2["caseid"] == cid) & (u2["issue_n"] == q)]
    assert ub["last_map"].iloc[0] == 140.0                              # U does see reading r + 1
