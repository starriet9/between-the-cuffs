"""Paired patient-level bootstrap (SPEC §6.2, §3 yardstick; PREREG §6, §7).

2,000 resamples, seed 20261118; resampling is over subjects (`subjectid`); a
drawn subject brings all of its cases (§4.5: "the bootstrap resamples
subjects"). Every contrast of one analysis uses the same resamples, so the
bootstrap is paired. Confidence intervals are percentile intervals
[judgment: numpy's default linear interpolation between order statistics].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import CI_PERCENTILES, N_BOOT, RULE_MIN_RELATIVE_REDUCTION, SEED


class SubjectBootstrap:
    """Resampling weights (n_boot x n_patients) over the patients that enter an analysis."""

    def __init__(self, caseids, subjectids, n_boot: int = N_BOOT, seed: int = SEED):
        caseids, subjectids = np.asarray(caseids), np.asarray(subjectids)
        order = np.argsort(caseids, kind="stable")
        if not np.array_equal(order, np.arange(len(caseids))):
            raise ValueError("patients must be in ascending case-ID order")
        subjects = np.unique(subjectids)                 # ascending subject ID
        sidx = np.searchsorted(subjects, subjectids)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
        flat = (draws + np.arange(n_boot)[:, None] * len(subjects)).ravel()
        cnt = np.bincount(flat, minlength=n_boot * len(subjects)).reshape(n_boot, len(subjects))
        self.caseids = caseids
        self.n_boot, self.seed, self.n_subjects = n_boot, seed, len(subjects)
        self.W = cnt[:, sidx].astype(float)

    def mean(self, v) -> np.ndarray:
        return (self.W @ np.asarray(v, dtype=float)) / self.W.sum(axis=1)

    def ratio(self, num, den) -> np.ndarray:
        return (self.W @ np.asarray(num, dtype=float)) / (self.W @ np.asarray(den, dtype=float))


def percentile_ci(x: np.ndarray) -> tuple:
    """Percentile CI; +/-inf entries are allowed (F, §3) and handled as limits."""
    x = np.asarray(x, dtype=float)
    big = 1e300
    z = np.clip(np.where(np.isposinf(x), big, np.where(np.isneginf(x), -big, x)), -big, big)
    lo, hi = np.percentile(z, CI_PERCENTILES)
    conv = lambda v: np.inf if v > 1e299 else (-np.inf if v < -1e299 else float(v))
    return conv(lo), conv(hi)


@dataclass
class Contrast:
    """model vs comparator: Delta = mean RMSE(model) - mean RMSE(comparator) (§6.2);
    relative reduction = 1 - mean RMSE(model) / mean RMSE(comparator)."""
    model: str
    comparator: str
    n_patients: int
    mean_model: float
    mean_comp: float
    delta: float
    delta_lo: float
    delta_hi: float
    rr: float
    rr_lo: float
    rr_hi: float

    @property
    def meets_rule(self) -> bool:
        """SPEC §6.3: relative reduction >= 5% and upper 95% CI limit of Delta below zero
        (unrounded values)."""
        return bool(self.rr >= RULE_MIN_RELATIVE_REDUCTION and self.delta_hi < 0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["meets_rule"] = self.meets_rule
        return d


def contrast(bs: SubjectBootstrap, a, b, model: str, comparator: str) -> Contrast:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ma, mb = float(a.mean()), float(b.mean())
    sa, sb = bs.mean(a), bs.mean(b)
    dlo, dhi = percentile_ci(sa - sb)
    rlo, rhi = percentile_ci(1.0 - sa / sb)
    return Contrast(model, comparator, len(a), ma, mb, ma - mb, dlo, dhi, 1.0 - ma / mb, rlo, rhi)


def mean_ci(bs: SubjectBootstrap, a) -> tuple:
    a = np.asarray(a, dtype=float)
    return (float(a.mean()), *percentile_ci(bs.mean(a)))


def fraction_F(bs: SubjectBootstrap, rmse_L, rmse_T, rmse_U) -> dict:
    """Withheld-cuff yardstick (§3): N = RMSE_L - RMSE_T, D = RMSE_L - RMSE_U, F = N / D, all
    patient-level means. CI of F: percentile CI of N*/D*; a resample with D* <= 0 counts as +inf
    if N* > 0 and as -inf otherwise. F is not truncated."""
    L, T, U = (np.asarray(x, dtype=float) for x in (rmse_L, rmse_T, rmse_U))
    N, D = L.mean() - T.mean(), L.mean() - U.mean()
    sL, sT, sU = bs.mean(L), bs.mean(T), bs.mean(U)
    Ns, Ds = sL - sT, sL - sU
    with np.errstate(divide="ignore", invalid="ignore"):
        Fs = np.where(Ds > 0, Ns / Ds, np.where(Ns > 0, np.inf, -np.inf))
        F = N / D if D != 0 else (np.inf if N > 0 else -np.inf)
    nlo, nhi = percentile_ci(Ns)
    dlo, dhi = percentile_ci(Ds)
    flo, fhi = percentile_ci(Fs)
    return {"N": float(N), "N_lo": nlo, "N_hi": nhi, "D": float(D), "D_lo": dlo, "D_hi": dhi,
            "F": float(F), "F_lo": flo, "F_hi": fhi, "F_reported": bool(dlo > 0),
            "n_resamples_D_le_0": int((Ds <= 0).sum())}
