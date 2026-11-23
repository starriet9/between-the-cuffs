"""Synthetic per-forecast encoder tables (btc/encoder.py layout) on top of the synthetic derived tables of
`btc.models.synthetic`, for testing the model code only. No real data.

Each (caseid, primary, issue_n, variant) row of the synthetic waveform table gets one encoder row with the
same window counts and availability. The embedding differences are standard normal noise; with
`planted`, column enc_lvl_000 carries the synthetic `wf_pat_lvl` and enc_chg_000 / enc_chg_001 carry
`wf_pat_chg` / `wf_rise_chg` (which hold the planted MAP signal when `make_synthetic(wf_signal > 0)`),
each of the variant itself, so P1 rows carry the donor's values. Level columns are missing where the
synthetic reference is not yet established (wf_rise_lvl missing), change columns where wf_rise_chg is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc import encoder as E


def encoder_table(T, dim: int = E.EMB_DIM, planted: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wf = T.features_wf
    wf = wf.loc[(wf["window"] == "primary") & wf["variant"].isin(E.VARIANTS)].reset_index(drop=True)
    n = len(wf)
    lvl = rng.standard_normal((n, dim))
    chg = rng.standard_normal((n, dim))
    if planted:
        for M, j, c in ((lvl, 0, "wf_pat_lvl"), (chg, 0, "wf_pat_chg"), (chg, 1, "wf_rise_chg")):
            v = wf[c].to_numpy(float)
            M[:, j] = np.where(np.isfinite(v), v, M[:, j])
    avail = wf["wf_available"].to_numpy(bool)
    lvl[~avail | wf["wf_rise_lvl"].isna().to_numpy()] = np.nan
    chg[~avail | wf["wf_rise_chg"].isna().to_numpy()] = np.nan
    keys = pd.DataFrame({"caseid": wf["caseid"].to_numpy(), "window": "primary", "issue_n": wf["issue_n"].to_numpy(),
                         "variant": wf["variant"].to_numpy(), "signal_caseid": np.nan,
                         "t_issue": wf["t_issue"].to_numpy(float), "t_prev": wf["t_prev"].to_numpy(float),
                         "t_end": wf["t_end"].to_numpy(float), "enc_n_windows": wf["wf_n_windows"].to_numpy(),
                         "enc_n_usable": wf["wf_n_usable"].to_numpy(), "enc_available": avail})
    names = [f"enc_lvl_{k:03d}" for k in range(dim)] + [f"enc_chg_{k:03d}" for k in range(dim)]
    return pd.concat([keys, pd.DataFrame(np.hstack([lvl, chg]), columns=names)], axis=1)


def enc_cols(dim: int) -> tuple:
    return tuple(f"enc_lvl_{k:03d}" for k in range(dim)) + tuple(f"enc_chg_{k:03d}" for k in range(dim))
