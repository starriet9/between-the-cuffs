"""Per-patient metrics (SPEC §6.1; PREREG §6, §7).

Each metric is computed per patient ("per patient" means per case, §4.5) over
that patient's scored forecasts in the analysis at hand, then summarised as
mean +/- SD across patients. RMSE, MAE and mean signed error need >= 3
forecasts; R^2 and FIT need >= 5 forecasts and a non-zero SST. Patients
dropped by these rules are counted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MIN_FORECASTS_R2, MIN_FORECASTS_RMSE


def per_patient(caseid: np.ndarray, y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """One row per case: n, rmse, mae, mse_signed (mean of forecast - observed), r2, fit, sse.

    R^2 = 1 - SSR/SST and FIT = 100 (1 - ||y - yhat|| / ||y - ybar||) with ybar the patient's
    in-window mean of the observed target readings (§6.1) [judgment: the mean of the targets
    scored in the analysis at hand]. Entries failing the minimum-count rules are NaN."""
    d = pd.DataFrame({"caseid": caseid, "y": y, "e": pred - y})
    if d["e"].isna().any():
        raise ValueError("missing forecast: every forecast is scored (§6.2)")
    d["e2"] = d["e"] ** 2
    d["ae"] = d["e"].abs()
    g = d.groupby("caseid", sort=True)
    out = pd.DataFrame({"n": g.size(), "sse": g["e2"].sum(), "sae": g["ae"].sum(), "se": g["e"].sum()})
    ybar = g["y"].transform("mean")
    d["dev2"] = (d["y"] - ybar) ** 2
    out["sst"] = d.groupby("caseid", sort=True)["dev2"].sum()
    ok3 = out["n"] >= MIN_FORECASTS_RMSE
    ok5 = (out["n"] >= MIN_FORECASTS_R2) & (out["sst"] > 0)
    out["rmse"] = np.where(ok3, np.sqrt(out["sse"] / out["n"]), np.nan)
    out["mae"] = np.where(ok3, out["sae"] / out["n"], np.nan)
    out["mse_signed"] = np.where(ok3, out["se"] / out["n"], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["r2"] = np.where(ok5, 1.0 - out["sse"] / out["sst"], np.nan)
        out["fit"] = np.where(ok5, 100.0 * (1.0 - np.sqrt(out["sse"]) / np.sqrt(out["sst"])), np.nan)
    return out


def summarise(pp: pd.DataFrame) -> dict:
    """Mean +/- SD across patients, with the counts entering and dropped (§6.1)."""
    res = {"n_patients_scored": int(len(pp)), "n_forecasts": int(pp["n"].sum())}
    for m in ("rmse", "mae", "mse_signed", "r2", "fit"):
        v = pp[m].dropna()
        res[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
        res[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        res[f"{m}_n_patients"] = int(len(v))
        res[f"{m}_n_dropped"] = int(len(pp) - len(v))
    res["pooled_rmse"] = float(np.sqrt(pp["sse"].sum() / pp["n"].sum())) if len(pp) else np.nan
    return res
