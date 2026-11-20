"""Gradient-boosting check (SPEC §6.8; PREREG §7). Exploratory; always run; never changes the
primary outcome, the rule or Sentences 1-3.

Arms: GB-cuff (M2's features in the chosen form, with h, no interaction columns), GB-hr (+ heart
rate), GB-full (+ waveform; a forecast without waveform stays in with every waveform group marked
missing, no fallback), GB-full-P1 (GB-full on the P1 donor features). Target, forecasts, imputation,
missing-indicators and outer folds are those of the ridge models (the same `Design` without
interactions; standardisation does not change a tree model). Learner: scikit-learn
`HistGradientBoostingRegressor`, squared error, early stopping off, seed 20261118, other settings
default. Grid: learning rate 0.03 / 0.1; maximum leaf nodes 3 / 7 / 15; iterations 30 / 100 / 300,
chosen per arm by the inner CV of §4.5. The three iteration counts are read from one 300-iteration
fit with `staged_predict` (identical to separate fits with early stopping off; tested).
Ties go to the first setting in the order learning rate, leaves, iterations, each ascending
[judgment: the most shrunken setting, by analogy with "ties go to the larger penalty"].
"""
from __future__ import annotations

from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .constants import GB_LEARNING_RATES, GB_MAX_ITER, GB_MAX_LEAF_NODES, HR_COLS, N_SPLITS, SEED, WF_COLS
from .design import Block, Cols, Design, cuff_columns
from .ridge import choose, patient_mean_rmse

GB_ARMS = ("GB-cuff", "GB-hr", "GB-full", "GB-full-P1")
SETTINGS = [(lr, nl) for lr in GB_LEARNING_RATES for nl in GB_MAX_LEAF_NODES]
CANDIDATES = [(lr, nl, it) for lr, nl, it in product(GB_LEARNING_RATES, GB_MAX_LEAF_NODES, GB_MAX_ITER)]


def arm_design(arm: str, form: str) -> Design:
    cols, inter = cuff_columns(form, "map")
    v = "p1" if arm.endswith("P1") else "real"
    blocks = []
    if arm != "GB-cuff":
        blocks.append(Block(f"hr@{v}", tuple(f"{c}@{v}" for c in HR_COLS)))
    if arm in ("GB-full", "GB-full-P1"):
        blocks.append(Block(f"wf@{v}", tuple(f"{c}@{v}" for c in WF_COLS)))   # no `requires`: no fallback
    return Design(cols, inter, "h_min", pooled=False, blocks=tuple(blocks))


def _model(lr, nl, it):
    return HistGradientBoostingRegressor(loss="squared_error", learning_rate=lr, max_leaf_nodes=nl,
                                         max_iter=it, early_stopping=False, random_state=SEED)


def _fit_staged(F: Cols, design: Design, fit_rows, pred_rows, lr, nl):
    """Predictions after 30, 100 and 300 iterations of one fit."""
    with threadpool_limits(1):
        d = design.__class__(design.cuff_cols, (), design.hcol, False, (), design.blocks).fit(F, fit_rows)
        m = _model(lr, nl, max(GB_MAX_ITER)).fit(d.transform(F, fit_rows), F.y[fit_rows])
        Xp = d.transform(F, pred_rows)
        want = {it - 1: it for it in GB_MAX_ITER}
        out, last = {}, None
        for i, p in enumerate(m.staged_predict(Xp)):
            last = p
            if i in want:
                out[want[i]] = p
        for it in GB_MAX_ITER:                     # stopped early (no split possible): last stage
            out.setdefault(it, last)
        return [out[it] for it in GB_MAX_ITER]


def _fit_final(F: Cols, design: Design, fit_rows, pred_rows, setting):
    with threadpool_limits(1):
        d = design.__class__(design.cuff_cols, (), design.hcol, False, (), design.blocks).fit(F, fit_rows)
        m = _model(*setting).fit(d.transform(F, fit_rows), F.y[fit_rows])
        return m.predict(d.transform(F, pred_rows)), (d, m)


def _select(F, rows, fold_of_row, results_for_rows):
    """Criterion per candidate from val predictions; returns the chosen (lr, nl, it)."""
    pos = np.full(F.n, -1)
    pos[rows] = np.arange(len(rows))
    P = np.empty((len(rows), len(CANDIDATES)))
    for (i, lr, nl), preds in results_for_rows.items():
        val = rows[fold_of_row[rows] == i]
        for k, it in enumerate(GB_MAX_ITER):
            P[pos[val], CANDIDATES.index((lr, nl, it))] = preds[k]
    vals = [patient_mean_rmse(F.y[rows], P[:, c], F.code[rows], F.ncode) for c in range(len(CANDIDATES))]
    return choose(vals, CANDIDATES), dict(zip([str(c) for c in CANDIDATES], vals))


def cv_gb(F: Cols, form: str, plan, n_jobs: int = 1, arms=GB_ARMS) -> dict:
    """Outer-CV predictions (absolute) of every arm, and the chosen settings per fold."""
    outer = plan.outer_of(F.caseid)
    inner_by_j = []
    for j in range(N_SPLITS):
        tr = np.flatnonzero(outer != j)
        inn = np.full(F.n, -1)
        inn[tr] = plan.inner_of(j, F.caseid[tr])
        inner_by_j.append((tr, inn))
    designs = {a: arm_design(a, form) for a in arms}
    tasks = []
    for a in arms:
        for j, (tr, inn) in enumerate(inner_by_j):
            for i in range(N_SPLITS):
                for lr, nl in SETTINGS:
                    tasks.append((a, j, i, lr, nl, tr[inn[tr] != i], tr[inn[tr] == i]))
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_staged)(F, designs[t[0]], t[5], t[6], t[3], t[4]) for t in tasks)
    by = {}
    for t, r in zip(tasks, res):
        by.setdefault((t[0], t[1]), {})[(t[2], t[3], t[4])] = r
    chosen = {}
    for a in arms:
        for j, (tr, inn) in enumerate(inner_by_j):
            chosen[(a, j)] = _select(F, tr, inn, by[(a, j)])
    final = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_final)(F, designs[a], inner_by_j[j][0], np.flatnonzero(outer == j), chosen[(a, j)][0])
        for a in arms for j in range(N_SPLITS))
    pred = {a: np.full(F.n, np.nan) for a in arms}
    k = 0
    for a in arms:
        for j in range(N_SPLITS):
            pred[a][outer == j] = F.last[outer == j] + final[k][0]
            k += 1
    return {"pred": pred, "chosen": {f"{a}|fold{j}": list(chosen[(a, j)][0]) for a in arms for j in range(N_SPLITS)}}


def refit_gb(F: Cols, form: str, plan, n_jobs: int = 1, arms=GB_ARMS) -> dict:
    """Refit every arm on all S-dev, settings chosen with the outer folds (§6.8, §4.5)."""
    outer = plan.outer_of(F.caseid)
    rows = np.arange(F.n)
    designs = {a: arm_design(a, form) for a in arms}
    tasks = [(a, i, lr, nl) for a in arms for i in range(N_SPLITS) for lr, nl in SETTINGS]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_staged)(F, designs[a], rows[outer != i], rows[outer == i], lr, nl) for a, i, lr, nl in tasks)
    by = {}
    for t, r in zip(tasks, res):
        by.setdefault(t[0], {})[(t[1], t[2], t[3])] = r
    out = {}
    for a in arms:
        setting, crit = _select(F, rows, outer, by[a])
        _, (d, m) = _fit_final(F, designs[a], rows, rows[:1], setting)
        out[a] = {"setting": list(setting), "design": d, "model": m, "crit": crit}
    return out


def predict_gb(frozen: dict, F: Cols) -> dict:
    rows = np.arange(F.n)
    with threadpool_limits(1):
        return {a: F.last + v["model"].predict(v["design"].transform(F, rows)) for a, v in frozen.items()}
