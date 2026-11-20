"""Outer and inner folds (SPEC §4.1, §4.5; PREREG §4).

scikit-learn `StratifiedGroupKFold` (5 splits, shuffled, seeded) applied to one
row per analysed case (SPEC §2.2) in ascending case-ID order, stratified by cycling class
and grouped by subject. A subject's class is that of its lowest-numbered case.
The outer folds are made once per seed and reused by every analysis. Inner
folds are "built like the outer folds inside the training data": the same
procedure on the training cases of an outer fold, with the same seed
[judgment: same seed as the outer folds].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import N_SPLITS, SEED


def subject_class(caseids, subjectids, fast) -> np.ndarray:
    """Class of each case = class of its subject's lowest-numbered case (§4.5)."""
    df = pd.DataFrame({"caseid": np.asarray(caseids), "subjectid": np.asarray(subjectids),
                       "fast": np.asarray(fast, dtype=bool)}).sort_values("caseid")
    first = df.groupby("subjectid")["fast"].first()
    return pd.Series(np.asarray(subjectids)).map(first).to_numpy().astype(int)


def assign_folds(caseids, subjectids, labels, seed: int, n_splits: int = N_SPLITS) -> np.ndarray:
    """Fold index (0..n_splits-1) of each case. Input order must be ascending case ID."""
    caseids = np.asarray(caseids)
    if np.any(np.diff(caseids) <= 0):
        raise ValueError("cases must be in strictly ascending case-ID order (§4.5)")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(caseids), -1, dtype=int)
    for j, (_, test) in enumerate(sgkf.split(np.zeros(len(caseids)), np.asarray(labels),
                                             groups=np.asarray(subjectids))):
        fold[test] = j
    if np.any(fold < 0):
        raise RuntimeError("a case was not assigned to a fold")
    return fold


@dataclass
class FoldPlan:
    """Outer folds and the inner folds of every outer training set, for one seed."""
    seed: int
    caseids: np.ndarray                 # ascending
    subjectids: np.ndarray
    labels: np.ndarray                  # subject class
    outer: np.ndarray                   # outer fold per case
    inner: list = field(default_factory=list)   # per outer fold j: inner fold per case (-1 = test case)

    def outer_of(self, caseid_rows: np.ndarray) -> np.ndarray:
        return _lookup(self.caseids, self.outer, caseid_rows)

    def inner_of(self, j: int, caseid_rows: np.ndarray) -> np.ndarray:
        return _lookup(self.caseids, self.inner[j], caseid_rows)

    def table(self) -> pd.DataFrame:
        d = pd.DataFrame({"caseid": self.caseids, "subjectid": self.subjectids,
                          "subject_class_fast": self.labels, "outer_fold": self.outer})
        for j, inn in enumerate(self.inner):
            d[f"inner_fold_outer{j}"] = inn
        return d


def _lookup(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(keys, query)
    if np.any(pos >= len(keys)) or np.any(keys[np.minimum(pos, len(keys) - 1)] != query):
        raise KeyError("case not in the fold plan")
    return values[pos]


def make_fold_plan(cases: pd.DataFrame, seed: int = SEED) -> FoldPlan:
    """`cases`: the analysed cases (§2.2; all analysable ones in the blocker sensitivity analysis), with caseid, subjectid, fast_cycling (§4.5)."""
    c = cases.sort_values("caseid")
    ids = c["caseid"].to_numpy().astype(int)
    subj = c["subjectid"].to_numpy()
    lab = subject_class(ids, subj, c["fast_cycling"].to_numpy())
    outer = assign_folds(ids, subj, lab, seed)
    inner = []
    for j in range(N_SPLITS):
        tr = outer != j
        inn = np.full(len(ids), -1, dtype=int)
        inn[tr] = assign_folds(ids[tr], subj[tr], lab[tr], seed)
        inner.append(inn)
    return FoldPlan(seed=seed, caseids=ids, subjectids=subj, labels=lab, outer=outer, inner=inner)
