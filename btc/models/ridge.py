"""Block-penalised ridge and the model ladder (SPEC §4.1, §4.5; PREREG §4).

Objective (SPEC §4.5): mean squared error over the training forecasts plus, for
each block b, lambda_b times the sum of that block's squared coefficients on
the standardised columns; the intercept is not penalised. With one block this
is scikit-learn `Ridge(alpha = n_train * lambda)`. With several blocks it is
the same estimator on columns rescaled by 1/sqrt(lambda_b); here it is solved
in closed form, (Zc'Zc + n diag(lambda)) beta = Zc'yc, which
`tests/test_models_ridge.py` checks against the rescaled `Ridge` and against a
direct minimisation of the objective.

A chain is a ladder of nested models (e.g. M2 -> M3 -> M4). Stage 0 is the cuff
model; stage s adds block s with its own penalty from the grid or "dropped",
keeping the penalties of the stages below (SPEC §4.1). A dropped stage returns
the forecast of the stage below and its block stays out of later stages
(§4.5). A block with `requires` is fitted on the rows where it holds; other
rows get the stage below (M4 -> M3 fallback, §4.3, §4.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import solve

from .constants import (DROPPED, HR_COLS, LAMBDA_GRID, N_BINS, N_SPLITS, PERBIN_MIN_TARGETS, TIE_RTOL,
                        WF_COLS)
from .design import Block, Cols, Design, cuff_columns


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------

def ridge_fit(Z: np.ndarray, y: np.ndarray, pen: np.ndarray):
    """argmin_{b0, beta} mean((y - b0 - Z beta)^2) + sum_j pen_j beta_j^2 (intercept unpenalised)."""
    n = len(y)
    zm, ym = Z.mean(axis=0), y.mean()
    Zc = Z - zm
    A = Zc.T @ Zc + n * np.diag(pen)
    beta = solve(A, Zc.T @ (y - ym), assume_a="pos")
    return ym - zm @ beta, beta


def ridge_fit_path(Z, y, pen_fixed, vary_mask, lams):
    """Same solution for each value in `lams` of the penalty on the `vary_mask` columns."""
    n = len(y)
    zm, ym = Z.mean(axis=0), y.mean()
    Zc = Z - zm
    G, r = Zc.T @ Zc, Zc.T @ (y - ym)
    out = []
    for lam in lams:
        pen = np.where(vary_mask, lam, pen_fixed)
        beta = solve(G + n * np.diag(pen), r, assume_a="pos")
        out.append((ym - zm @ beta, beta))
    return out


# ----------------------------------------------------------------------------
# Chain specification
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainSpec:
    name: str
    stages: tuple                 # model names, stage 0 first, e.g. ("M2", "M3", "M4")
    form: str = "a"               # M2 form (SPEC §4.1)
    structure: str = "pooled"     # "pooled" or "perbin" (SPEC §4.1)
    target: str = "map"           # "map" or "sbp" (SBP models, §6.7)
    hcol: str = "h_min"           # horizon in minutes (h-replacement sensitivity: "hmed_min")
    bincol: str = "h_bin"         # per-bin routing column
    extra_cuff: tuple = ()        # extra cuff-block inputs (yardstick L: elapsed time)
    blocks: tuple = ()            # Block per added stage

    def __post_init__(self):
        if len(self.blocks) != len(self.stages) - 1:
            raise ValueError("one block per added stage")

    def design(self, active_blocks, pooled: bool) -> Design:
        cols, inter = cuff_columns(self.form, self.target)
        return Design(cols, inter, self.hcol, pooled, self.extra_cuff, active_blocks)


@dataclass
class StageFit:
    dropped: bool
    design: Design | None = None
    b0: float = 0.0
    beta: np.ndarray | None = None
    requires: str | None = None


@dataclass
class NodeFit:
    """A fitted chain on one node (the pooled model, or one h-bin of a per-bin structure)."""
    pooled: bool
    lam: list                     # chosen penalty per stage (None = dropped)
    stages: list                  # StageFit per stage
    crit: list = field(default_factory=list)   # inner-CV criterion per candidate, per stage


# ----------------------------------------------------------------------------
# Criterion and choice (SPEC §4.5)
# ----------------------------------------------------------------------------

def patient_mean_rmse(y, pred, code, ncode) -> float:
    """Mean over patients of each patient's RMSE, no minimum count (§4.5 inner criterion)."""
    e2 = (y - pred) ** 2
    sse = np.bincount(code, e2, ncode)
    cnt = np.bincount(code, None, ncode)
    m = cnt > 0
    return float(np.mean(np.sqrt(sse[m] / cnt[m])))


def choose(values: list, candidates: list):
    """First candidate (in preference order) whose criterion equals the minimum.

    Candidates are ordered "dropped" first, then penalties from largest to smallest, so ties
    go to "dropped" and to the larger penalty (§4.5)."""
    v = np.asarray(values, dtype=float)
    best = np.nanmin(v)
    for c, x in zip(candidates, v):
        if x <= best + TIE_RTOL * abs(best):
            return c
    raise RuntimeError("no candidate")


GRID_DESC = [float(x) for x in LAMBDA_GRID[::-1]]


# ----------------------------------------------------------------------------
# Fitting one node
# ----------------------------------------------------------------------------

def _active(spec: ChainSpec, lam: list, s: int) -> list:
    """Blocks of stages 1..s that were not dropped (stage s included if s >= 1)."""
    return [spec.blocks[i - 1] for i in range(1, s + 1) if (i == s or lam[i] is not DROPPED)]


def _lam_vector(lam: list, s: int) -> list:
    """Penalty per active block index (0 = cuff), the stage-s penalty last (placeholder)."""
    v = [lam[0]] + [lam[i] for i in range(1, s) if lam[i] is not DROPPED]
    return v + [np.nan] if s >= 1 else v


def _fit_predict(F: Cols, spec, active, pooled, lamvec, fit_rows, pred_rows, lams):
    """Fit on fit_rows for each candidate penalty of the last block; predictions on pred_rows."""
    d = spec.design(active, pooled).fit(F, fit_rows)
    Zf, Zp = d.transform(F, fit_rows), d.transform(F, pred_rows)
    nb = len(lamvec)
    vary = d.block_of == (nb - 1)
    pen_fixed = np.asarray(lamvec, dtype=float)[d.block_of]
    sols = ridge_fit_path(Zf, F.y[fit_rows], np.nan_to_num(pen_fixed), vary, lams)
    return [b0 + Zp @ beta for b0, beta in sols]


def select_node(F: Cols, spec: ChainSpec, rows: np.ndarray, pooled: bool, sel_folds: list):
    """Choose the penalties of every stage by CV over `sel_folds` (list of (fit, val) row arrays
    partitioning `rows`). Returns (lam, crit) (SPEC §4.1, §4.5)."""
    n_st = len(spec.stages)
    pos = np.full(F.n, -1)
    pos[rows] = np.arange(len(rows))
    y = F.y[rows]
    code = F.code[rows]
    lam, crit = [], []
    prev = None
    for s in range(n_st):
        if s == 0:
            cands, lams = GRID_DESC, GRID_DESC
            req = np.ones(F.n, dtype=bool)
        else:
            cands, lams = [DROPPED] + GRID_DESC, GRID_DESC
            req = F.mask(spec.blocks[s - 1].requires)
        active = _active(spec, lam + [0.0], s) if s else []
        lamvec = _lam_vector(lam + [0.0], s) if s else [np.nan]
        P = np.empty((len(rows), len(lams)))
        for fit, val in sel_folds:
            fr, vr = fit[req[fit]], val[req[val]]
            vo = val[~req[val]]
            if len(vo):
                P[pos[vo], :] = prev[pos[vo]][:, None]
            if len(vr) == 0:
                continue
            if len(fr) < 2:                       # [judgment] nothing to fit: the model below
                P[pos[vr], :] = (prev[pos[vr]][:, None] if s else 0.0)
                continue
            preds = _fit_predict(F, spec, active, pooled, lamvec, fr, vr, lams)
            for i, p in enumerate(preds):
                P[pos[vr], i] = p
        vals = [patient_mean_rmse(y, P[:, i], code, F.ncode) for i in range(len(lams))]
        if s:
            vals = [patient_mean_rmse(y, prev, code, F.ncode)] + vals
            cols = [None] + list(range(len(lams)))
        else:
            cols = list(range(len(lams)))
        c = choose(vals, cands)
        lam.append(c)
        crit.append(dict(zip(["dropped" if x is None else x for x in cands], vals)))
        prev = prev if c is DROPPED else P[:, cols[cands.index(c)]]
    return lam, crit


def fit_node(F: Cols, spec: ChainSpec, rows: np.ndarray, pooled: bool, lam: list) -> NodeFit:
    """Fit every stage on `rows` with the chosen penalties."""
    stages = []
    for s in range(len(spec.stages)):
        if s and lam[s] is DROPPED:
            stages.append(StageFit(dropped=True))
            continue
        req = spec.blocks[s - 1].requires if s else None
        fr = rows[F.mask(req)[rows]]
        active = _active(spec, lam, s) if s else []
        d = spec.design(active, pooled)
        if len(fr) < 2:
            stages.append(StageFit(dropped=True))   # [judgment] nothing to fit: the model below
            continue
        d.fit(F, fr)
        lamvec = [lam[0]] + [lam[i] for i in range(1, s + 1) if lam[i] is not DROPPED] if s else [lam[0]]
        b0, beta = ridge_fit(d.transform(F, fr), F.y[fr], d.penalties(lamvec))
        stages.append(StageFit(dropped=False, design=d, b0=b0, beta=beta, requires=req))
    return NodeFit(pooled=pooled, lam=lam, stages=stages)


def predict_node(F: Cols, node: NodeFit, rows: np.ndarray) -> np.ndarray:
    """Predicted change, one column per stage, with fallbacks to the stage below."""
    out = np.empty((len(rows), len(node.stages)))
    for s, st in enumerate(node.stages):
        if st.dropped:
            out[:, s] = out[:, s - 1] if s else 0.0
            continue
        m = F.mask(st.requires)[rows]
        col = out[:, s - 1].copy() if s else np.zeros(len(rows))
        if m.any():
            r = rows[m]
            col[m] = st.b0 + st.design.transform(F, r) @ st.beta
        out[:, s] = col
    return out


# ----------------------------------------------------------------------------
# Structures: pooled, or per-bin with the pooled fallback (SPEC §4.1)
# ----------------------------------------------------------------------------

def own_bins(F: Cols, spec: ChainSpec) -> tuple:
    """Bins fitted on their own: >= 300 training targets in all of S-dev (§4.1). Counted on the
    analysis frame (all S-dev forecasts of the analysis at hand) [judgment]."""
    if spec.structure == "pooled":
        return ()
    b = F[spec.bincol].astype(int)
    return tuple(i for i in range(N_BINS) if int((b == i).sum()) >= PERBIN_MIN_TARGETS)


def _nodes(F: Cols, spec: ChainSpec, rows: np.ndarray, bins: tuple) -> list:
    """(key, rows, pooled) for each node; rows routed to the pooled model unless their bin is own."""
    if spec.structure == "pooled":
        return [("pooled", rows, True)]
    b = F[spec.bincol][rows].astype(int)
    nodes = [(f"bin{i}", rows[b == i], False) for i in bins]
    # The pooled model serves the bins with < 300 targets [judgment: fitted on all rows]. It is always
    # fitted, so that a row outside 15-400 s (possible only when h is replaced by the median previous
    # interval, §3) also has a model [judgment].
    nodes.append(("pooled", rows, True))
    return nodes


def _route(F: Cols, spec: ChainSpec, rows: np.ndarray, bins: tuple) -> dict:
    if spec.structure == "pooled":
        return {"pooled": rows}
    b = F[spec.bincol][rows].astype(int)
    r = {f"bin{i}": rows[b == i] for i in bins}
    r["pooled"] = rows[~np.isin(b, bins)]
    return r


def _folds_within(rows: np.ndarray, fold_of_row: np.ndarray) -> list:
    f = fold_of_row[rows]
    return [(rows[f != i], rows[f == i]) for i in range(N_SPLITS)]


@dataclass
class ChainFit:
    """A chain fitted on one training set (all nodes)."""
    spec: ChainSpec
    bins: tuple
    nodes: dict                   # key -> NodeFit

    def predict_change(self, F: Cols, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(F.n) if rows is None else rows
        out = np.full((len(rows), len(self.spec.stages)), np.nan)
        pos = np.full(F.n, -1)
        pos[rows] = np.arange(len(rows))
        for key, r in _route(F, self.spec, rows, self.bins).items():
            if len(r):
                out[pos[r]] = predict_node(F, self.nodes[key], r)
        return out

    def predict(self, F: Cols, rows=None) -> np.ndarray:
        rows = np.arange(F.n) if rows is None else rows
        return F.last[rows][:, None] + self.predict_change(F, rows)


def fit_chain(F: Cols, spec: ChainSpec, rows: np.ndarray, sel_fold_of_row: np.ndarray, bins: tuple,
              keep_crit: bool = False) -> ChainFit:
    """Select penalties on `rows` with CV folds `sel_fold_of_row`, then fit every node on `rows`."""
    nodes = {}
    for key, r, pooled in _nodes(F, spec, rows, bins):
        lam, crit = select_node(F, spec, r, pooled, _folds_within(r, sel_fold_of_row))
        nf = fit_node(F, spec, r, pooled, lam)
        nf.crit = crit if keep_crit else []
        nodes[key] = nf
    return ChainFit(spec=spec, bins=bins, nodes=nodes)


# ----------------------------------------------------------------------------
# Outer cross-validation and the refit on all S-dev
# ----------------------------------------------------------------------------

@dataclass
class CVResult:
    spec: ChainSpec
    seed: int
    bins: tuple
    pred: np.ndarray              # absolute forecasts, one column per stage (outer-CV predictions)
    fold: np.ndarray              # outer fold per row
    lam: list                     # per outer fold: {node: [penalty per stage]}

    def stage(self, name: str) -> np.ndarray:
        return self.pred[:, self.spec.stages.index(name)]


def cv_chain(F: Cols, spec: ChainSpec, plan) -> CVResult:
    """Outer-CV predictions of every stage (§4.1, §4.5)."""
    bins = own_bins(F, spec)
    outer = plan.outer_of(F.caseid)
    pred = np.full((F.n, len(spec.stages)), np.nan)
    lams = []
    for j in range(N_SPLITS):
        tr = np.flatnonzero(outer != j)
        te = np.flatnonzero(outer == j)
        inner = np.full(F.n, -1)
        inner[tr] = plan.inner_of(j, F.caseid[tr])
        cf = fit_chain(F, spec, tr, inner, bins)
        if len(te):
            pred[te] = cf.predict(F, te)
        lams.append({k: list(v.lam) for k, v in cf.nodes.items()})
    return CVResult(spec=spec, seed=plan.seed, bins=bins, pred=pred, fold=outer, lam=lams)


def refit_chain(F: Cols, spec: ChainSpec, plan) -> ChainFit:
    """Refit on all rows, penalties chosen with the outer folds as CV folds (§4.5)."""
    bins = own_bins(F, spec)
    return fit_chain(F, spec, np.arange(F.n), plan.outer_of(F.caseid), bins, keep_crit=True)


def with_structure(spec: ChainSpec, form: str, structure: str) -> ChainSpec:
    return replace(spec, form=form, structure=structure)


def hr_block(variant: str = "real", hcol: str = "h_min", name: str | None = None) -> Block:
    """Heart-rate block (§4.2) of a variant; fitted on every row (§4.5)."""
    return Block(name or f"hr@{variant}", tuple(f"{c}@{variant}" for c in HR_COLS), None, hcol)


def wf_block(variant: str = "real", hcol: str = "h_min", cols=None, name: str | None = None) -> Block:
    """Waveform block (§4.3) of a variant; fitted on rows with >= 2 usable windows (§4.3, §4.5)."""
    cols = WF_COLS if cols is None else cols
    return Block(name or f"wf@{variant}", tuple(f"{c}@{variant}" for c in cols),
                 f"wf_available@{variant}", hcol)
