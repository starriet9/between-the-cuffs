"""Synthetic derived tables in the btc/SCHEMA.md format, for testing the model code only.

SPEC §9 step 5: before the confirmatory run, the fitting code is tested only on
M2 and on synthetic added features. Nothing here reads real data.

Each case has a latent MAP on a 1-s grid (baseline, Ornstein-Uhlenbeck
wander and, in most cases, a post-spinal drop with partial recovery). Cuff
readings sample it with measurement noise at 60-90 s (fast cyclers) or
~100-320 s (slow cyclers). Signal features are generated at the feature level:
with `wf_signal` > 0 the pulse-arrival-time and rise-time features carry the
latent MAP level, its change and, as a leading signal, the latent change over
the next 2 min (information no cuff-only model has); with `hr_signal` > 0 the
heart-rate features carry the leading signal. With both 0 the added features are
pure noise. P1 features use the donor's latent processes at the recipient's
times, P2 the patient's own at times shifted back by 600 s, `ia` stops 30 s
early, and `strict` has fewer usable windows (btc/SCHEMA.md §4).

A few analysable cases get `nmb_recorded` = True (SPEC §2.2), drawn from a separate random stream
after everything else, so the rest of the cohort (readings, features, P1 donors) does not depend on
the number of such cases.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .constants import HR_COLS, IA_EXCLUDE_S, P2_SHIFT_S, WF_COLS, WF_FEATURES
from ..constants import MAX_H_S as MAX_H
from ..constants import PRIMARY_END_S as PRIMARY_END
from ..constants import SECONDARY_CAP_S as SECONDARY_CAP

T_MAX = 3700


@dataclass(frozen=True)
class SynthConfig:
    cohort: str = "sdev"
    n_cases: int = 226
    frac_fast: float = 0.16
    wf_signal: float = 0.0
    hr_signal: float = 0.0
    n_not_analysable: int = 8
    n_late_waveform: int = 2
    two_case_subject: bool = True
    p_hr_missing: float = 0.04
    meas_sd: float = 3.5
    female_share: float = 0.29
    n_cerclage: int = 6
    n_nmb: int = 4                     # analysable cases with a recorded neuromuscular blocker (SPEC §2.2)


SDEV = SynthConfig()
SOB = SynthConfig(cohort="sob", n_cases=21, frac_fast=16 / 21, n_not_analysable=1, n_late_waveform=0,
                  two_case_subject=False, female_share=1.0, n_cerclage=0, n_nmb=1)
NMB_STREAM = 2202                      # separate seed stream for the nmb_recorded draw


# ----------------------------------------------------------------------------
# Latent processes
# ----------------------------------------------------------------------------

def _ou(rng, n, tau, sd):
    a = np.exp(-1.0 / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _latent_case(rng):
    t = np.arange(T_MAX + 1, dtype=float)
    base = rng.normal(88, 10)
    m = base + _ou(rng, len(t), rng.uniform(200, 600), rng.uniform(3, 7))
    if rng.random() < 0.7:                                       # post-spinal drop, partial recovery
        t0, depth = rng.uniform(60, 1300), rng.uniform(8, 35)
        w, rho, trec = rng.uniform(20, 90), rng.uniform(0.2, 0.7), rng.uniform(300, 1500)
        s = 1.0 / (1.0 + np.exp(-(t - t0) / w))
        rec = rho + (1 - rho) * np.exp(-np.maximum(0.0, t - t0) / trec)
        m = m - depth * s * rec
    m = np.clip(m, 40, 150)
    hr0 = rng.normal(75, 12)
    hr_noise = _ou(rng, len(t), 300.0, 4.0)
    p_usable = rng.beta(5, 2) if rng.random() < 0.85 else rng.beta(1, 5)    # some cases with poor pleth
    return {"m": m, "hr0": hr0, "hr_noise": hr_noise, "pp": rng.uniform(35, 60), "p_usable": p_usable}


# ----------------------------------------------------------------------------
# Cuff readings and instances (SPEC §2.4, §2.5, §3; btc/SCHEMA.md §2-§3)
# ----------------------------------------------------------------------------

def _reading_times(rng, fast: bool, start: float):
    base = rng.uniform(58, 88) if fast else float(np.clip(rng.lognormal(np.log(185), 0.3), 95, 330))
    t, out = start, []
    while t < T_MAX - 50:
        out.append(t)
        step = base * rng.uniform(0.9, 1.1)
        u = rng.random()
        if u < 0.05:
            step *= 0.5                              # an extra reading
        elif u < 0.08:
            step *= 2.5                              # a long gap
        t += max(step, 20.0)
    return np.array(out)


def _readings(rng, caseid, lat, fast, start, meas_sd):
    t = np.round(_reading_times(rng, fast, start), 1)
    idx = np.clip(np.round(t).astype(int), 0, T_MAX)
    mp = np.round(lat["m"][idx] + rng.normal(0, meas_sd, len(t)))
    pp = np.round(lat["pp"] + rng.normal(0, 3, len(t)))
    sbp = mp + np.round(2 * pp / 3)
    dbp = mp - np.round(pp / 3)
    drop = np.array([""] * len(t), dtype=object)
    u = rng.random(len(t))
    bad_range, bad_order = u < 0.01, (u >= 0.01) & (u < 0.015)
    mp = np.where(bad_range, 175.0, mp)
    sbp = np.where(bad_order, mp - 5, sbp)
    drop[bad_range] = "map_range"
    drop[bad_order] = "order"
    valid = drop == ""
    n = np.full(len(t), np.nan)
    n[valid] = np.arange(1, valid.sum() + 1)
    r = pd.DataFrame({"caseid": caseid, "n_all": np.arange(1, len(t) + 1), "t": t, "sbp": sbp, "dbp": dbp,
                      "map": mp, "valid": valid, "drop_reason": np.where(valid, None, drop), "n": n})
    v = r.loc[r["valid"]]
    tp = v.loc[v["t"] <= PRIMARY_END, "t"].to_numpy()
    med = float(np.median(np.diff(tp))) if len(tp) > 1 else np.nan
    gaps = np.r_[np.nan, np.diff(v["t"].to_numpy())]
    gf = pd.Series(False, index=r.index)
    gf.loc[v.index] = gaps > 2 * med if np.isfinite(med) else False
    r["gap_flag"] = gf.to_numpy()
    return r, med


def _ls_slope(tm, y):
    tc = tm - tm.mean()
    return float(np.sum(tc * (y - y.mean())) / np.sum(tc * tc))


def _curv(tm, y):
    (t1, t2, t3), (y1, y2, y3) = tm, y
    return float(2.0 * ((y3 - y2) / (t3 - t2) - (y2 - y1) / (t2 - t1)) / (t3 - t1))


def _cuff(t, mp, sbp, dbp, n):
    t, mp, sbp, dbp = t[:n], mp[:n], sbp[:n], dbp[:n]
    bm, bs = mp[:3].mean(), sbp[:3].mean()
    t3 = t[-3:] / 60.0
    return {"t_issue": t[-1], "t_prev": t[-2], "h_med_prev_s": float(np.median(np.diff(t))),
            "last_map": mp[-1], "last_sbp": sbp[-1], "last_dbp": dbp[-1], "base_map": bm, "base_sbp": bs,
            "run_mean_map": mp.mean(),
            "cuff_level": mp[-1] - bm, "cuff_slope": _ls_slope(t3, mp[-3:]), "cuff_curv": _curv(t3, mp[-3:]),
            "cuff_pp": sbp[-1] - dbp[-1], "cuff_tsr_min": t[-1] / 60.0,
            "cuff_lag0": mp[-1] - bm, "cuff_lag1": mp[-2] - bm, "cuff_lag2": mp[-3] - bm,
            "cuff_gap1_min": (t[-1] - t[-2]) / 60.0, "cuff_gap2_min": (t[-2] - t[-3]) / 60.0,
            "cuff_sbp_level": sbp[-1] - bs, "cuff_sbp_slope": _ls_slope(t3, sbp[-3:]),
            "cuff_sbp_curv": _curv(t3, sbp[-3:]), "cuff_sbp_lag0": sbp[-1] - bs,
            "cuff_sbp_lag1": sbp[-2] - bs, "cuff_sbp_lag2": sbp[-3] - bs}


def _instances(caseid, subjectid, fast, readings, opstart):
    v = readings.loc[readings["valid"]]
    t, mp, sbp, dbp = (v[c].to_numpy(dtype=float) for c in ("t", "map", "sbp", "dbp"))
    rows = []
    for window, (lo, hi) in (("primary", (0.0, PRIMARY_END)),
                             ("secondary", (0.0, min(opstart, SECONDARY_CAP)))):
        for n in range(3, len(t) + 1):
            if not (lo <= t[n - 1] <= hi):
                continue
            feats = None
            for k in (1, 2, 3):
                j = n + k
                if j > len(t):
                    break
                h = t[j - 1] - t[n - 1]
                if not (lo <= t[j - 1] <= hi and h <= MAX_H):
                    break
                feats = feats or _cuff(t, mp, sbp, dbp, n)
                rows.append({"caseid": caseid, "window": window, "issue_n": n, "subjectid": subjectid,
                             "fast_cycling": fast, "k": k, "target_n": j, "t_target": t[j - 1],
                             "t_next": t[n] if n < len(t) else np.nan, "h_s": h, "h_min": h / 60.0,
                             "y_map": mp[j - 1], "y_sbp": sbp[j - 1], "y_dbp": dbp[j - 1], **feats})
    return rows


# ----------------------------------------------------------------------------
# Signal features (btc/SCHEMA.md §4)
# ----------------------------------------------------------------------------

def _at(a, tau):
    return a[int(np.clip(np.floor(tau) - 1, 0, T_MAX))]         # last sample strictly before tau


LEAD_S = 120.0     # the planted signal sees the latent MAP change over the next 2 min


def _lead(m, te):
    return _at(m, te + LEAD_S) - _at(m, te)


def _hr_series(lat, cfg):
    return lat["hr0"] + lat["hr_noise"]


def _hr_feats(rng, src, hr, tau, tau_prev, t_first, cfg):
    if src is None or tau - 60 < t_first or rng.random() < cfg.p_hr_missing:
        return {c: np.nan for c in HR_COLS}
    sig = -0.5 * cfg.hr_signal * _lead(src["m"], tau)          # HR rises before MAP falls
    lvl = _at(hr, tau) + sig + rng.normal(0, 0.7)
    lvl_prev = _at(hr, tau_prev) + rng.normal(0, 0.7) if tau_prev - 60 >= t_first else np.nan
    slope = (_at(hr, tau) - _at(hr, tau - 59)) + 0.5 * sig + rng.normal(0, 1.0)
    chg = lvl - lvl_prev if rng.random() > 0.02 else np.nan
    return {"hr_level": lvl, "hr_change": chg, "hr_slope": slope}


def _wf_feats(rng, src, tau, tau_prev, t_end, pleth_first, p_usable, cfg, t_ref):
    a = max(tau_prev, tau - 120.0, pleth_first)
    n_win = int(max(0.0, min(tau, t_end) - a) // 10)
    n_use = int(rng.binomial(n_win, p_usable)) if (src is not None and n_win) else 0
    out = {c: np.nan for c in WF_COLS}
    out.update(wf_n_windows=n_win, wf_n_usable=n_use, wf_usable_share=(n_use / n_win) if n_win else np.nan,
               wf_available=bool(n_use >= 2))
    if not out["wf_available"]:
        return out
    m = src["m"]
    te = min(tau, t_end)
    lvl = _at(m, te) - m[:300].mean()
    chg = _at(m, te) - _at(m, tau_prev)
    lead = _lead(m, te)
    s = cfg.wf_signal
    signal = {"wf_pat_lvl": -0.15 * s * lvl, "wf_pat_chg": -0.15 * s * chg - 0.15 * s * lead,
              "wf_rise_chg": 0.15 * s * lead}
    for f in WF_FEATURES:
        for form in ("lvl", "chg"):
            c = f"wf_{f}_{form}"
            out[c] = signal.get(c, 0.0) + rng.normal(0, 1.0)
    if te < t_ref:
        for f in WF_FEATURES:
            out[f"wf_{f}_lvl"] = np.nan                         # reference not yet established
    if rng.random() < 0.1:
        for f in WF_FEATURES:
            out[f"wf_{f}_chg"] = np.nan
    if rng.random() < 0.05:
        out["wf_pat_lvl"] = out["wf_pat_chg"] = np.nan          # no paired R peaks (SPEC §4.3)
    return out


# (time shift, cutoff relative to the shifted issue time) per variant (btc/SCHEMA.md §4)
VARIANT_TIMES = {"real": (0.0, 0.0), "p1": (0.0, 0.0), "p2": (-P2_SHIFT_S, 0.0), "ia": (0.0, -IA_EXCLUDE_S),
                 "strict": (0.0, 0.0)}


def _features(rng, case, issues, lat_by_case, info_by_case, cfg):
    hr_rows, wf_rows = [], []
    own = lat_by_case[case["caseid"]]
    donor = case["donor_caseid"]
    dlat = lat_by_case.get(int(donor)) if np.isfinite(donor) else None
    dinfo = info_by_case.get(int(donor)) if np.isfinite(donor) else None
    info = info_by_case[case["caseid"]]
    hr_own = _hr_series(own, cfg)
    hr_don = _hr_series(dlat, cfg) if dlat is not None else None
    for r in issues.itertuples(index=False):
        for variant, (shift, cut) in VARIANT_TIMES.items():
            src, inf = (dlat, dinfo) if variant == "p1" else (own, info)
            tau, tau_prev = r.t_issue + shift, r.t_prev + shift
            t_end = tau + cut
            key = {"caseid": case["caseid"], "window": r.window, "issue_n": r.issue_n, "variant": variant,
                   "t_issue": r.t_issue, "t_prev": r.t_prev, "t_end": t_end}
            if variant != "strict":
                hr = hr_don if variant == "p1" else hr_own
                hf = _hr_feats(rng, src, hr, min(tau, t_end), tau_prev, inf["hr_t_first"] if inf else 0, cfg) \
                    if src is not None else {c: np.nan for c in HR_COLS}
                hr_rows.append({**key, **hf})
            p_use = (inf["p_usable"] * (0.8 if variant == "strict" else 1.0)) if inf else 0.0
            wf = _wf_feats(rng, src, tau, tau_prev, t_end, inf["pleth_t_first"] if inf else np.inf, p_use, cfg,
                           inf["t_ref"] if inf else np.inf)
            wf_rows.append({**key, **wf})
    return hr_rows, wf_rows


# ----------------------------------------------------------------------------
# Cohort
# ----------------------------------------------------------------------------

def _assign_donors(cases: pd.DataFrame, seed: int) -> pd.Series:
    """P1 donors as in SPEC §6.6 (next covering case of another subject in a seeded permutation)."""
    a = cases.loc[cases["analysable"]].sort_values("caseid")
    perm = np.random.default_rng(seed).permutation(a["caseid"].to_numpy())
    rec = a.set_index("caseid")
    donor = pd.Series(np.nan, index=cases["caseid"].to_numpy(), dtype=float)
    for i, cid in enumerate(perm):
        r = rec.loc[cid]
        for step in range(1, len(perm)):
            d = perm[(i + step) % len(perm)]
            dr = rec.loc[d]
            if dr["subjectid"] == r["subjectid"]:
                continue
            if all(dr[f"{s}_t_first"] <= r["first_issue_t"] and dr[f"{s}_t_last"] >= r["last_issue_t"]
                   for s in ("hr", "pleth", "ecg")):
                donor[cid] = float(d)
                break
    return donor


def _mark_nmb(cases: pd.DataFrame, instances: pd.DataFrame, n: int, seed: int) -> None:
    """`nmb_recorded` for `n` analysable cases with >= 3 primary next-reading targets (so that they would
    enter the primary outcome), and `analysed` (SPEC §2.2). Changes nothing else."""
    cases["nmb_recorded"] = False
    if n:
        k1 = instances.loc[(instances["window"] == "primary") & (instances["k"] == 1)].groupby("caseid").size()
        pool = np.sort(cases.loc[cases["analysable"] & cases["caseid"].isin(k1.index[k1 >= 3]), "caseid"].to_numpy())
        pick = np.random.default_rng([seed, NMB_STREAM]).choice(pool, n, replace=False)
        cases.loc[cases["caseid"].isin(pick), "nmb_recorded"] = True
    cases["analysed"] = cases["analysable"] & ~cases["nmb_recorded"]


def make_cohort(cfg: SynthConfig, seed: int, caseids: np.ndarray, subjectids: np.ndarray) -> dict:
    rng = np.random.default_rng(seed)
    n = len(caseids)
    fast = rng.random(n) < cfg.frac_fast
    lat_by_case, info, case_rows, read_parts, inst_rows = {}, {}, [], [], []
    bad = set(rng.choice(n, cfg.n_not_analysable, replace=False)) if cfg.n_not_analysable else set()
    late = set(rng.choice(sorted(set(range(n)) - bad), cfg.n_late_waveform, replace=False)) \
        if cfg.n_late_waveform else set()
    cerclage = set(rng.choice(n, cfg.n_cerclage, replace=False)) if cfg.n_cerclage else set()
    for i, (cid, sid) in enumerate(zip(caseids, subjectids)):
        cid, sid = int(cid), int(sid)
        lat = _latent_case(rng)
        lat_by_case[cid] = lat
        no_pleth = i in bad and i % 2 == 0
        start = rng.uniform(1450, 1600) if (i in bad and not no_pleth) else rng.uniform(60, 420)
        r, med = _readings(rng, cid, lat, bool(fast[i]), start, cfg.meas_sd)
        read_parts.append(r)
        v = r.loc[r["valid"]]
        n_valid_primary = int((v["t"] <= PRIMARY_END).sum())
        issue_t = v.loc[(v["n"] >= 3) & (v["t"] <= PRIMARY_END), "t"].to_numpy()
        reasons = (["pleth_absent", "ecg_absent"] if no_pleth else []) + \
                  (["lt6_valid_readings_primary"] if n_valid_primary < 6 else [])
        pleth_first = 2400.0 if i in late else rng.uniform(0, 10)
        hr_first = rng.uniform(0, 5)
        opstart = float(np.clip(rng.normal(27, 9), 2, 46) * 60)
        fc = bool(np.isfinite(med) and med < 90)
        info[cid] = {"hr_t_first": hr_first, "pleth_t_first": pleth_first, "p_usable": lat["p_usable"],
                     "t_ref": pleth_first + 60.0 / max(lat["p_usable"], 0.2) + 10}
        row = {"caseid": cid, "subjectid": sid, "cohort": cfg.cohort, "cerclage": i in cerclage,
               "opstart": opstart, "analysable": not reasons, "not_analysable_reason": ";".join(reasons),
               "n_valid_primary": n_valid_primary, "median_interval_primary_s": med, "fast_cycling": fc,
               "pleth_present": not no_pleth, "ecg_present": not no_pleth, "waveform_flag": "",
               "hr_t_first": hr_first, "hr_t_last": float(T_MAX),
               "pleth_t_first": pleth_first if not no_pleth else np.nan,
               "pleth_t_last": float(T_MAX) if not no_pleth else np.nan,
               "ecg_t_first": pleth_first if not no_pleth else np.nan,
               "ecg_t_last": float(T_MAX) if not no_pleth else np.nan,
               "first_issue_t": issue_t[0] if len(issue_t) else np.nan,
               "last_issue_t": issue_t[-1] if len(issue_t) else np.nan,
               "sex": "F" if rng.random() < cfg.female_share else "M",
               "n_drop_map_range_primary": int(((r["drop_reason"] == "map_range") & (r["t"] <= PRIMARY_END)).sum()),
               "n_drop_order_primary": int(((r["drop_reason"] == "order") & (r["t"] <= PRIMARY_END)).sum()),
               "n_gaps_primary": int((r["gap_flag"] & (r["t"] <= PRIMARY_END)).sum())}
        if cfg.cohort == "sob":
            row["vaso_stratum"] = ["pump", "bolus_no_pump", "neither_recorded"][int(rng.integers(0, 3))]
        case_rows.append(row)
        if not reasons:
            inst_rows += _instances(cid, sid, fc, r, opstart)
    cases = pd.DataFrame(case_rows)
    cases["donor_caseid"] = _assign_donors(cases, seed=20261118).reindex(cases["caseid"]).to_numpy()
    instances = pd.DataFrame(inst_rows)
    issues = instances.drop_duplicates(["caseid", "window", "issue_n"])[
        ["caseid", "window", "issue_n", "t_issue", "t_prev"]]
    hr_rows, wf_rows = [], []
    for case in cases.loc[cases["analysable"]].to_dict("records"):
        h, w = _features(rng, case, issues.loc[issues["caseid"] == case["caseid"]], lat_by_case, info, cfg)
        hr_rows += h
        wf_rows += w
    hr = pd.DataFrame(hr_rows)
    wf = pd.DataFrame(wf_rows)
    _mark_nmb(cases, instances, cfg.n_nmb, seed)
    return {"cases": cases, "readings": pd.concat(read_parts, ignore_index=True), "instances": instances,
            "features_hr": hr, "features_wf": wf}


def make_synthetic(seed: int = 1, wf_signal: float = 0.0, hr_signal: float = 0.0,
                   sdev: SynthConfig = SDEV, sob: SynthConfig = SOB) -> dict:
    """Synthetic S-dev and S-ob tables: {"sdev": {...}, "sob": {...}} (btc/SCHEMA.md names)."""
    rng = np.random.default_rng(seed + 7919)
    sdev = replace(sdev, wf_signal=wf_signal, hr_signal=hr_signal)
    sob = replace(sob, wf_signal=wf_signal, hr_signal=hr_signal)
    ids = np.sort(rng.choice(np.arange(1, 6389), sdev.n_cases + sob.n_cases, replace=False))
    is_sob = np.zeros(len(ids), dtype=bool)
    is_sob[rng.choice(len(ids), sob.n_cases, replace=False)] = True
    sdev_ids, sob_ids = ids[~is_sob], ids[is_sob]
    subj = rng.choice(np.arange(1, 100000), len(ids), replace=False)
    sdev_subj, sob_subj = subj[~is_sob].copy(), subj[is_sob]
    if sdev.two_case_subject:
        sdev_subj[-1] = sdev_subj[len(sdev_subj) // 2]          # one subject with two cases (§2.2)
    return {"sdev": make_cohort(sdev, seed, sdev_ids, sdev_subj),
            "sob": make_cohort(sob, seed + 1, sob_ids, sob_subj)}
