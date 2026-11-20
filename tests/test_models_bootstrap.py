"""Paired subject bootstrap (SPEC §6.2, §3) and folds (SPEC §4.5)."""
import numpy as np
import pandas as pd
import pytest

from btc.models.bootstrap import SubjectBootstrap, contrast, fraction_F, percentile_ci
from btc.models.constants import N_BOOT, SEED
from btc.models.folds import assign_folds, make_fold_plan, subject_class


def _pat(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(np.arange(1, 6000), n, replace=False))
    subj = np.arange(n) + 100
    subj[-1] = subj[3]                             # one subject with two cases
    return ids, subj, rng.uniform(4, 9, n), rng.uniform(4, 9, n)


def test_bootstrap_is_deterministic_and_registered():
    ids, subj, a, b = _pat()
    b1, b2 = SubjectBootstrap(ids, subj), SubjectBootstrap(ids, subj)
    assert b1.n_boot == N_BOOT == 2000 and b1.seed == SEED == 20261118
    assert np.array_equal(b1.W, b2.W)
    assert contrast(b1, a, b, "x", "y") == contrast(b2, a, b, "x", "y")


def test_bootstrap_resamples_subjects():
    ids, subj, a, b = _pat()
    bs = SubjectBootstrap(ids, subj)
    i, j = 3, len(ids) - 1                          # the two cases of one subject move together
    assert np.array_equal(bs.W[:, i], bs.W[:, j])
    assert np.allclose(bs.W.sum(1) - bs.W[:, j], bs.n_subjects)   # n_subjects draws per resample
    assert bs.n_subjects == len(ids) - 1


def test_paired_contrast_matches_direct_computation():
    ids, subj, a, b = _pat()
    bs = SubjectBootstrap(ids, subj)
    c = contrast(bs, a, b, "x", "y")
    sa, sb = bs.mean(a), bs.mean(b)
    assert np.isclose(c.delta, a.mean() - b.mean())
    assert np.isclose(c.rr, 1 - a.mean() / b.mean())
    assert (c.delta_lo, c.delta_hi) == pytest.approx(tuple(np.percentile(sa - sb, [2.5, 97.5])))
    assert (c.rr_lo, c.rr_hi) == pytest.approx(tuple(np.percentile(1 - sa / sb, [2.5, 97.5])))


def test_F_infinity_rule():
    ids, subj, _, _ = _pat()
    rng = np.random.default_rng(1)
    L = rng.uniform(6, 8, len(ids))
    U = L + rng.normal(0, 0.5, len(ids))            # D near zero: many resamples with D* <= 0
    T = L - 0.3
    bs = SubjectBootstrap(ids, subj)
    r = fraction_F(bs, L, T, U)
    sL, sT, sU = bs.mean(L), bs.mean(T), bs.mean(U)
    N, D = sL - sT, sL - sU
    assert r["n_resamples_D_le_0"] == int((D <= 0).sum()) > 0
    Fs = np.where(D > 0, N / np.where(D > 0, D, 1), np.where(N > 0, np.inf, -np.inf))
    assert np.isinf(Fs).sum() == r["n_resamples_D_le_0"]
    lo, hi = percentile_ci(Fs)
    assert (r["F_lo"], r["F_hi"]) == (lo, hi)
    assert r["F_reported"] is bool(r["D_lo"] > 0)


def test_percentile_ci_with_infinities():
    x = np.r_[np.full(97, 1.0), np.full(3, np.inf)]
    assert percentile_ci(x) == (1.0, np.inf)
    x = np.r_[np.full(3, -np.inf), np.arange(97.0)]
    assert percentile_ci(x)[0] == -np.inf


def test_folds_grouped_stratified_deterministic():
    rng = np.random.default_rng(2)
    n = 120
    cases = pd.DataFrame({"caseid": np.sort(rng.choice(np.arange(1, 6000), n, replace=False)),
                          "subjectid": np.arange(n), "fast_cycling": rng.random(n) < 0.16})
    cases.loc[n - 1, "subjectid"] = cases.loc[5, "subjectid"]
    cases.loc[n - 1, "fast_cycling"] = not cases.loc[5, "fast_cycling"]
    p1, p2 = make_fold_plan(cases, SEED), make_fold_plan(cases, SEED)
    assert np.array_equal(p1.outer, p2.outer)
    assert p1.outer[5] == p1.outer[n - 1]                     # grouped by subject
    # the subject's class is that of its lowest-numbered case
    assert p1.labels[n - 1] == int(cases.loc[5, "fast_cycling"])
    fast_per_fold = np.bincount(p1.outer[p1.labels == 1], minlength=5)
    assert fast_per_fold.max() - fast_per_fold.min() <= 2     # stratified
    assert not np.array_equal(make_fold_plan(cases, SEED + 1).outer, p1.outer)
    with pytest.raises(ValueError):
        assign_folds(cases["caseid"].to_numpy()[::-1], cases["subjectid"], p1.labels, SEED)


def test_subject_class_uses_lowest_case():
    lab = subject_class([10, 20, 30], [1, 2, 1], [False, True, True])
    assert list(lab) == [0, 1, 0]
