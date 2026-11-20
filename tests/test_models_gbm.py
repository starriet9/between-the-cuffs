"""Gradient-boosting check (SPEC §6.8)."""
import numpy as np
from threadpoolctl import threadpool_limits

from btc.models.data import build_frame, tables_from_frames
from btc.models.design import Cols
from btc.models.gbm import CANDIDATES, _fit_staged, _model, arm_design
from btc.models.synthetic import make_synthetic


def test_grid_and_arms():
    assert len(CANDIDATES) == 18
    assert {c[0] for c in CANDIDATES} == {0.03, 0.1} and {c[1] for c in CANDIDATES} == {3, 7, 15}
    assert {c[2] for c in CANDIDATES} == {30, 100, 300}
    m = _model(0.1, 7, 30)
    assert m.early_stopping is False and m.random_state == 20261118 and m.loss == "squared_error"
    d = arm_design("GB-full-P1", "b")
    assert all("@p1" in c for b in d.blocks for c in b.cols) and not d.pooled
    assert all(b.requires is None for b in d.blocks)          # no fallback in GB-full
    assert arm_design("GB-cuff", "a").blocks == ()


def test_staged_predictions_equal_separate_fits():
    T = tables_from_frames("sdev", make_synthetic(seed=5, wf_signal=2.0)["sdev"])
    F = Cols(build_frame(T))
    d = arm_design("GB-full", "a")
    fit, val = np.arange(0, 900), np.arange(900, 1200)
    staged = _fit_staged(F, d, fit, val, 0.1, 7)
    d2 = arm_design("GB-full", "a").fit(F, fit)
    for it, p in zip((30, 100, 300), staged):
        with threadpool_limits(1):
            sep = _model(0.1, 7, it).fit(d2.transform(F, fit), F.y[fit]).predict(d2.transform(F, val))
        assert np.array_equal(p, sep)
