"""Design matrices for the ridge models (SPEC §4.1, §4.5; PREREG §4).

Everything that is estimated (imputation medians, which indicator columns are
merged, scaler mean and SD) is estimated on the rows passed to `Design.fit`,
which are always the training rows of the model at hand (§4.1: "the scaler and
the imputation medians are fitted on training folds only").

Column order: cuff block (M2 inputs, h, pooled interactions, extra timing
inputs), then each added block (features, feature x h, missing-indicators).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# M2 inputs (SPEC §4.1, btc/SCHEMA.md §3). h is added from the chain's horizon column.
CUFF_FORM = {
    "a": ("cuff_level", "cuff_slope", "cuff_curv", "cuff_pp", "cuff_tsr_min"),
    "b": ("cuff_lag0", "cuff_lag1", "cuff_lag2", "cuff_gap1_min", "cuff_gap2_min", "cuff_pp", "cuff_tsr_min"),
}
# Pooled horizon interactions (SPEC §4.1 structure (b)): (column, power of h)
CUFF_INTERACTIONS = {
    "a": (("cuff_slope", 1), ("cuff_curv", 2), ("cuff_level", 1)),
    "b": tuple((f"cuff_lag{i}", p) for i in range(3) for p in (1, 2)),
}
# SBP models (§6.7): SBP in place of MAP as lags and baseline; pulse pressure, time and gaps unchanged
SBP_RENAME = {"cuff_level": "cuff_sbp_level", "cuff_slope": "cuff_sbp_slope", "cuff_curv": "cuff_sbp_curv",
              "cuff_lag0": "cuff_sbp_lag0", "cuff_lag1": "cuff_sbp_lag1", "cuff_lag2": "cuff_sbp_lag2"}


def cuff_columns(form: str, target: str):
    cols, inter = CUFF_FORM[form], CUFF_INTERACTIONS[form]
    if target == "sbp":
        cols = tuple(SBP_RENAME.get(c, c) for c in cols)
        inter = tuple((SBP_RENAME.get(c, c), p) for c, p in inter)
    elif target != "map":
        raise ValueError(target)
    return cols, inter


@dataclass(frozen=True)
class Block:
    """An added block (SPEC §4.1: heart rate, waveform; §6.6 placebo and ablation versions)."""
    name: str
    cols: tuple
    requires: str | None = None     # boolean column; other rows get the model below (§4.3, §4.5)
    hcol: str = "h_min"             # horizon (minutes) for feature x h (§4.5; yardstick T: remaining horizon)
    kind: str = "feature"           # "feature" or "indicator" (0/1 columns used as is, §6.6 item 4)


class Cols:
    """Column store: numpy float arrays of a frame, plus target, case codes and last value."""

    def __init__(self, frame: pd.DataFrame, target: str = "map"):
        self.frame = frame
        self.target = target
        self._c = {}
        self.n = len(frame)
        self.caseid = frame["caseid"].to_numpy().astype(int)
        self.subjectid = frame["subjectid"].to_numpy()
        ucase, self.code = np.unique(self.caseid, return_inverse=True)
        self.ncode = len(ucase)
        self.last = frame[f"last_{target}"].to_numpy(dtype=float)
        self.yabs = frame[f"y_{target}"].to_numpy(dtype=float)
        self.y = self.yabs - self.last          # all models except M00 predict the change (§4.1)

    def __getitem__(self, col: str) -> np.ndarray:
        a = self._c.get(col)
        if a is None:
            a = self.frame[col].to_numpy(dtype=float)
            self._c[col] = a
        return a

    def mask(self, col: str | None) -> np.ndarray:
        if col is None:
            return np.ones(self.n, dtype=bool)
        return self.frame[col].to_numpy().astype(bool)


def _unique_columns(M: np.ndarray) -> list:
    """Indices of the first occurrence of each distinct column (identical indicators merged, §4.5)."""
    keep = []
    for j in range(M.shape[1]):
        if not any(np.array_equal(M[:, j], M[:, k]) for k in keep):
            keep.append(j)
    return keep


class Design:
    """One model's design matrix (SPEC §4.5)."""

    def __init__(self, cuff_cols, cuff_inter, hcol: str, pooled: bool, extra_cols=(), blocks=()):
        self.cuff_cols = tuple(cuff_cols)
        self.cuff_inter = tuple(cuff_inter) if pooled else ()
        self.hcol = hcol
        self.pooled = pooled
        self.extra_cols = tuple(extra_cols)
        self.blocks = tuple(blocks)
        self.fitted = False

    # -- fitting -----------------------------------------------------------------
    def fit(self, F: Cols, rows: np.ndarray) -> "Design":
        self.med, self.keep = [], []
        for b in self.blocks:
            X = np.column_stack([F[c][rows] for c in b.cols]) if len(rows) else np.zeros((0, len(b.cols)))
            if b.kind == "indicator":
                self.med.append(None)
                self.keep.append(_unique_columns(X))
                continue
            miss = np.isnan(X)
            with np.errstate(all="ignore"):
                med = np.array([np.median(X[~miss[:, j], j]) if (~miss[:, j]).any() else 0.0
                                for j in range(X.shape[1])])   # [judgment] all missing -> 0
            self.med.append(med)
            self.keep.append(_unique_columns(miss))
        self.fitted = True
        Z = self._raw(F, rows)
        self.mean = Z.mean(axis=0) if len(rows) else np.zeros(Z.shape[1])
        sd = Z.std(axis=0) if len(rows) else np.ones(Z.shape[1])
        self.sd = np.where(sd > 0, sd, 1.0)    # zero-variance column: scale 1, as scikit-learn
        return self

    # -- transforming -------------------------------------------------------------
    def _raw(self, F: Cols, rows: np.ndarray) -> np.ndarray:
        H = F[self.hcol][rows]
        cols, names, blk = [], [], []
        for c in self.cuff_cols:
            cols.append(F[c][rows]); names.append(c); blk.append(0)
        cols.append(H); names.append("h"); blk.append(0)
        for c, p in self.cuff_inter:                       # unstandardised features, h in minutes
            cols.append(F[c][rows] * H ** p); names.append(f"{c}*h^{p}"); blk.append(0)
        for c in self.extra_cols:
            cols.append(F[c][rows]); names.append(c); blk.append(0)
        cuff = np.column_stack(cols) if cols else np.zeros((len(rows), 0))
        if np.isnan(cuff).any():
            raise ValueError("missing cuff-block value (cuff features are never imputed)")
        mats = [cuff]
        for bi, b in enumerate(self.blocks, start=1):
            X = np.column_stack([F[c][rows] for c in b.cols])
            if b.kind == "indicator":
                if np.isnan(X).any():
                    raise ValueError("indicator columns cannot be missing")
                k = self.keep[bi - 1]
                mats.append(X[:, k]); names += [b.cols[j] for j in k]; blk += [bi] * len(k)
                continue
            miss = np.isnan(X)
            med = self.med[bi - 1]
            Xi = np.where(miss, med, X)                     # imputed before the interactions (§4.5)
            mats.append(Xi); names += list(b.cols); blk += [bi] * len(b.cols)
            if self.pooled:                                 # centred at the training median, x h (§4.5)
                Hb = F[b.hcol][rows]
                mats.append((Xi - med) * Hb[:, None]); names += [f"{c}*h" for c in b.cols]
                blk += [bi] * len(b.cols)
            k = self.keep[bi - 1]
            mats.append(miss[:, k].astype(float)); names += [f"miss:{b.cols[j]}" for j in k]
            blk += [bi] * len(k)
        self.names = names
        self.block_of = np.array(blk, dtype=int)
        return np.column_stack(mats)

    def transform(self, F: Cols, rows: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Design.fit first")
        return (self._raw(F, rows) - self.mean) / self.sd

    def penalties(self, lam_by_block) -> np.ndarray:
        """Per-column penalty vector from one penalty per block (index 0 = cuff)."""
        return np.asarray(lam_by_block, dtype=float)[self.block_of]
