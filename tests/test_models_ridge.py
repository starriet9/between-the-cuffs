"""Block-penalised ridge (SPEC §4.5) against closed forms, and the design-matrix rules."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from btc.models.constants import LAMBDA_GRID
from btc.models.design import Block, Cols, Design
from btc.models.ridge import GRID_DESC, choose, ridge_fit, ridge_fit_path


def _data(n=200, p=7, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p)) * rng.uniform(0.5, 3, p) + rng.normal(size=p)
    y = Z @ rng.normal(size=p) + 3.0 + rng.normal(size=n)
    return Z, y


def test_single_block_equals_sklearn_ridge_alpha_n_lambda():
    Z, y = _data()
    for lam in (1e-3, 0.1, 1.0, 1e3):
        b0, beta = ridge_fit(Z, y, np.full(Z.shape[1], lam))
        sk = Ridge(alpha=len(y) * lam, fit_intercept=True, solver="cholesky").fit(Z, y)
        assert np.allclose(beta, sk.coef_, atol=1e-9)
        assert np.isclose(b0, sk.intercept_, atol=1e-9)


def test_block_penalties_equal_rescaled_ridge_and_direct_minimum():
    Z, y = _data(p=9)
    n = len(y)
    pen = np.array([0.01] * 4 + [3.0] * 3 + [100.0] * 2)        # three blocks
    b0, beta = ridge_fit(Z, y, pen)
    # (1) direct closed form of the normal equations
    Zc, yc = Z - Z.mean(0), y - y.mean()
    beta_cf = np.linalg.solve(Zc.T @ Zc + n * np.diag(pen), Zc.T @ yc)
    assert np.allclose(beta, beta_cf, atol=1e-10)
    # (2) column rescaling: Ridge(alpha = n) on Z_j / sqrt(pen_j), coefficients scaled back
    s = 1.0 / np.sqrt(pen)
    sk = Ridge(alpha=n, solver="cholesky").fit(Z * s, y)
    assert np.allclose(beta, sk.coef_ * s, atol=1e-9)
    assert np.isclose(b0, sk.intercept_, atol=1e-9)
    # (3) numerical minimum of the stated objective (intercept unpenalised)
    obj = lambda th: np.mean((y - th[0] - Z @ th[1:]) ** 2) + np.sum(pen * th[1:] ** 2)
    r = minimize(obj, np.zeros(Z.shape[1] + 1), method="BFGS", options={"gtol": 1e-10})
    assert np.allclose(r.x[1:], beta, atol=1e-5)
    assert np.isclose(r.x[0], b0, atol=1e-5)


def test_intercept_is_not_penalised():
    Z, y = _data()
    b0, beta = ridge_fit(Z, y + 1000.0, np.full(Z.shape[1], 1e3))
    b0_ref, beta_ref = ridge_fit(Z, y, np.full(Z.shape[1], 1e3))
    assert np.allclose(beta, beta_ref) and np.isclose(b0 - b0_ref, 1000.0)


def test_path_matches_single_fits():
    Z, y = _data(p=6)
    fixed = np.array([0.1, 0.1, 0.1, 0, 0, 0.0])
    vary = np.array([False, False, False, True, True, True])
    for (b0, beta), lam in zip(ridge_fit_path(Z, y, fixed, vary, LAMBDA_GRID), LAMBDA_GRID):
        b0r, betar = ridge_fit(Z, y, np.where(vary, lam, fixed))
        assert np.allclose(beta, betar) and np.isclose(b0, b0r)


def test_grid_is_registered():
    assert len(LAMBDA_GRID) == 13 and np.isclose(LAMBDA_GRID[0], 1e-3) and np.isclose(LAMBDA_GRID[-1], 1e3)
    assert GRID_DESC[0] == pytest.approx(1e3)


def test_ties_go_to_dropped_then_larger_penalty():
    assert choose([1.0, 1.0, 1.0], [None, 10.0, 1.0]) is None
    assert choose([2.0, 1.0, 1.0], [None, 10.0, 1.0]) == 10.0
    assert choose([2.0, 1.5, 1.0], [None, 10.0, 1.0]) == 1.0


def _frame():
    rng = np.random.default_rng(3)
    n = 60
    d = pd.DataFrame({"caseid": np.repeat(np.arange(1, 13), 5), "subjectid": np.repeat(np.arange(1, 13), 5),
                      "y_map": rng.normal(80, 5, n), "last_map": rng.normal(80, 5, n),
                      "cuff_level": rng.normal(0, 5, n), "cuff_slope": rng.normal(0, 2, n),
                      "cuff_curv": rng.normal(0, 1, n), "cuff_pp": rng.normal(45, 5, n),
                      "cuff_tsr_min": rng.uniform(2, 30, n), "h_min": rng.uniform(0.5, 6, n),
                      "a@real": rng.normal(70, 10, n), "b@real": rng.normal(1, 1, n), "c@real": rng.normal(0, 1, n)})
    d.loc[[1, 5, 9], "a@real"] = np.nan
    d.loc[[1, 5, 9], "b@real"] = np.nan          # same pattern as a -> merged indicator
    d.loc[[2], "c@real"] = np.nan
    return d


def test_design_interactions_imputation_indicators():
    d = _frame()
    F = Cols(d)
    cols = ("cuff_level", "cuff_slope", "cuff_curv", "cuff_pp", "cuff_tsr_min")
    inter = (("cuff_slope", 1), ("cuff_curv", 2), ("cuff_level", 1))
    blk = Block("x", ("a@real", "b@real", "c@real"), None, "h_min")
    tr = np.arange(40)
    des = Design(cols, inter, "h_min", True, (), (blk,)).fit(F, tr)
    raw = des._raw(F, np.arange(len(d)))
    names = des.names
    H = d["h_min"].to_numpy()
    # cuff interactions from unstandardised features, h in minutes (§4.5)
    assert np.allclose(raw[:, names.index("cuff_slope*h^1")], d["cuff_slope"] * H)
    assert np.allclose(raw[:, names.index("cuff_curv*h^2")], d["cuff_curv"] * H ** 2)
    assert np.allclose(raw[:, names.index("cuff_level*h^1")], d["cuff_level"] * H)
    # imputation by the training-fold median, before the interaction; centring at that median
    med = np.nanmedian(d["a@real"].to_numpy()[tr])
    a = raw[:, names.index("a@real")]
    assert np.allclose(a[[1, 5, 9]], med)
    assert np.allclose(raw[:, names.index("a@real*h")], (a - med) * H)
    # one indicator per column that can be missing, identical indicators merged
    ind = [x for x in names if x.startswith("miss:")]
    assert ind == ["miss:a@real", "miss:c@real"]
    # indicators and interactions belong to the feature's block
    assert set(des.block_of[[names.index(x) for x in ("a@real", "a@real*h", "miss:a@real")]]) == {1}
    # standardised on the training rows
    Zt = des.transform(F, tr)
    nz = Zt.std(0) > 0
    assert np.allclose(Zt.mean(0), 0, atol=1e-10) and np.allclose(Zt.std(0)[nz], 1)


def test_per_bin_design_has_no_interactions():
    d = _frame()
    F = Cols(d)
    blk = Block("x", ("a@real", "c@real"), None, "h_min")
    des = Design(("cuff_level", "cuff_slope"), (("cuff_slope", 1),), "h_min", False, (), (blk,)).fit(F, np.arange(40))
    assert not any("*h" in x for x in des.names)
    assert "h" in des.names and "miss:a@real" in des.names


def test_design_statistics_come_from_training_rows_only():
    d = _frame()
    tr = np.arange(40)
    d2 = d.copy()
    num = [c for c in d.columns if c not in ("caseid", "subjectid")]
    d2.loc[40:, num] = np.random.default_rng(9).normal(1e3, 1e3, (20, len(num)))
    blk = Block("x", ("a@real", "b@real", "c@real"), None, "h_min")
    mk = lambda F: Design(("cuff_level", "cuff_slope"), (("cuff_slope", 1),), "h_min", True, (), (blk,)).fit(F, tr)
    a, b = mk(Cols(d)), mk(Cols(d2))
    assert all(np.array_equal(x, y) for x, y in zip(a.med, b.med))
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.sd, b.sd) and a.keep == b.keep
