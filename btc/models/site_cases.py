"""Per-case files of the case browser, the summary figure and the replays (SPEC §8.1-§8.3), written
after the frozen scoring (SPEC §9 step 8) in the format of `btc/site/series.py`.

Labelled as computed after the frozen scoring: nothing here can change Sentences 1-3. No model is
fitted. The frozen models are loaded from disk (`pipeline.load_frozen`, SHA-256 checked).

- **Scored forecasts.** `pred_at_target` and the per-case RMSE are those of the frozen scoring:
  `pipeline.frozen_predictions`, the call `pipeline.score_sob` makes.
- **Curves.** The frozen M2 and M4 (chain `main`) evaluated on an h grid of 15-400 s. Every other input
  of the issue point is unchanged, so imputation, missing-indicators, the per-bin routing and the
  M4 -> M3 fallback (§4.3, §4.5) are the scored model's; only h varies (§3). At the realised h of a
  scored target the curve value is the scored forecast; the grid path must reproduce it within 1e-9.
- **Issue points without a scored target** (next reading more than 400 s away or outside the window,
  §3) have no row in the derived tables. Their features come from the frozen feature functions on the
  same inputs: `btc.instances.cuff_features` on the valid readings, `btc.hr.case_hr_features` on the
  HR track, `btc.waveform.forecast_features` on the window table (variant `real`). On real data the
  same functions must reproduce the derived features of every scored issue point, or the export
  stops. The cuff features are checked on synthetic data too.
- **Intervals.** The registered 90% cross-conformal half-widths per h-bin (§6.7), from all S-dev
  outer-fold residuals, as written to `sob/after_frozen_scoring/conformal.csv`.
- **Hypotension counts** per case follow §4.5 and `secondary.hypotension`: a new-onset reading below 80%
  of the record-start baseline is a scored target below the threshold whose issue reading is not; it
  is flagged when the forecast is below the threshold.
- **Pump changes** (§8.2 rule 3) use the `PHEN_RATE` track only, no forecast.
- Phenylephrine (µg) and ephedrine (mg) stay in separate fields; nothing here adds them (§2.6).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import cuff, hr as HR, instances as INS, io, waveform as W
from ..constants import MAX_H_S, PRIMARY_END_S
from ..site import series as S
from .constants import CONFORMAL_LEVEL, HR_COLS, HYPO_REL_BASELINE, MIN_FORECASTS_RMSE, WF_COLS
from .data import Tables, h_bin_of
from .design import CUFF_FORM, Cols
from .metrics import per_patient
from .secondary import BIN_LABELS

H_GRID_S = np.arange(15.0, MAX_H_S + 1e-9, 5.0)      # §3: curves for h in 15-400 s
WINDOW_S = (0.0, PRIMARY_END_S)                       # §2.5 primary window
HR_DISPLAY_GAP_S = 10.0     # [judgment] display only: a heart-rate gap longer than this is drawn as a break
TOL = 1e-9
CUFF_CHECK = ("last_map", "base_map", *sorted({c for f in CUFF_FORM.values() for c in f}))
REAL = "real"
SITE_MODELS = ("M2", "M3", "M4", "M4-P1")             # btc/site/series.py CaseSummary.rmse

REASON_TEXT = {"pleth_absent": "no pleth waveform", "ecg_absent": "no ECG waveform",
               "subject_also_in_sob": "subject also in the cesarean cohort"}


class ExportError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# Signals of the frozen scoring (real data)
# ----------------------------------------------------------------------------

class RawSignals:
    """S-ob signals for the case files: heart-rate and pump tracks from the case archives through
    `btc.io.load_case` (S-ob needs BTC_FROZEN_SCORING=1, SPEC §9), the window table of the frozen
    scoring (`sob_windows.parquet`, btc/SCHEMA.md §4b) and the chart totals of `sob_cases.parquet`."""

    verify = True

    def __init__(self, cases: pd.DataFrame, windows: pd.DataFrame, cohort: str = "sob"):
        io.check_cohort_access(cohort)
        need = {"pump_present", "intraop_phe_ug", "intraop_eph_mg"} - set(cases.columns)
        if need:
            raise ExportError(f"the cases table lacks {sorted(need)} (btc/SCHEMA.md §1, S-ob)")
        self.cohort = cohort
        self.cases = cases.set_index("caseid")
        self.win = {int(c): g for c, g in windows.groupby("caseid")}
        self._loaded = (None, None)

    def _case(self, caseid: int):
        if self._loaded[0] != caseid:
            self._loaded = (caseid, io.load_case(caseid, self.cohort))
        return self._loaded[1]

    def hr(self, caseid: int):
        c = self._case(int(caseid))
        return c.hr.t, c.hr.v

    def pump(self, caseid: int):
        c = self._case(int(caseid))
        return c.phen.t, c.phen.v

    def windows(self, caseid: int) -> pd.DataFrame:
        return self.win.get(int(caseid), pd.DataFrame(columns=W.WINDOW_COLS))

    def chart(self, caseid: int) -> dict:
        r = self.cases.loc[int(caseid)]
        return {"pump_present": bool(r["pump_present"]), "phenylephrine_ug": r["intraop_phe_ug"],
                "ephedrine_mg": r["intraop_eph_mg"]}


# ----------------------------------------------------------------------------
# Small pieces
# ----------------------------------------------------------------------------

def _num(x) -> float | None:
    """A finite float, or None (JSON null) for a missing value."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def pump_changes(t, v, window=WINDOW_S) -> int:
    """PHEN_RATE changes in the primary window (§8.2 rule 3; no forecast used) [judgment]: the number
    of finite samples with 0 <= t <= 1800 s whose rate differs from the previous finite sample of the
    track. The first finite sample of the track is not a change; missing samples are skipped."""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    if len(t) < 2:
        return 0
    ch = v[1:] != v[:-1]
    inw = (t[1:] >= window[0]) & (t[1:] <= window[1])
    return int(np.sum(ch & inw))


def pump_series(t, v, window=WINDOW_S):
    """PHEN_RATE samples in the window, mL/h (each value holds until the next), or (None, None).
    Missing samples are skipped, as in `pump_changes`."""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    k = np.isfinite(t) & np.isfinite(v) & (t >= window[0]) & (t <= window[1])
    if not k.any():
        return None, None
    return t[k], v[k]


def hr_series(t, v, window=WINDOW_S):
    """Solar8000/HR for display, at most 1 Hz: samples in the window, values that are not valid heart
    rates (missing or <= 0 bpm, as in `btc.hr.clean_hr`) set to missing, the last sample of each
    second kept, and a break (missing value) inside gaps longer than HR_DISPLAY_GAP_S."""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    k = np.isfinite(t) & (t >= window[0]) & (t <= window[1])
    t, v = t[k], np.where(np.isfinite(v[k]) & (v[k] > 0), v[k], np.nan)
    if len(t):
        sec = np.floor(t)
        last = np.r_[sec[1:] != sec[:-1], True]
        t, v = t[last], v[last]
        ok = np.isfinite(v)
        tv = t[ok]
        gaps = np.flatnonzero(np.diff(tv) > HR_DISPLAY_GAP_S)
        if len(gaps):
            br = (tv[gaps] + tv[gaps + 1]) / 2
            t = np.concatenate([t, br])
            v = np.concatenate([v, np.full(len(br), np.nan)])
            o = np.argsort(t, kind="stable")
            t, v = t[o], v[o]
    return t, v


def _reason(row, n_valid: int) -> str:
    out = []
    for code in str(row.get("not_analysable_reason") or "").split(";"):
        if not code:
            continue
        if code == "lt6_valid_readings_primary":
            out.append(f"{n_valid} valid readings in the window (6 needed)")
        else:
            out.append(REASON_TEXT.get(code, code.replace("_", " ")))
    return "; ".join(out) or "not analysable"


def read_widths(conformal_csv: Path) -> dict:
    """{model: {bin index: half-width}} from `after_frozen_scoring/conformal.csv` (§6.7)."""
    d = pd.read_csv(conformal_csv, skiprows=1)
    out = {}
    for m in ("M2", "M4"):
        g = d.loc[d["model"] == m]
        if g.empty:
            raise ExportError(f"{conformal_csv.name}: no {CONFORMAL_LEVEL:.0%} half-widths for {m}")
        out[m] = {BIN_LABELS.index(b): float(w) for b, w in zip(g["bin"], g["half_width"])}
    return out


def half_widths(widths: dict, h) -> np.ndarray:
    b = h_bin_of(h)
    return np.array([widths.get(int(i), np.nan) if i >= 0 else np.nan for i in b], dtype=float)


# ----------------------------------------------------------------------------
# Issue points (§3) and their features
# ----------------------------------------------------------------------------

def issue_points(readings: pd.DataFrame, window=WINDOW_S) -> pd.DataFrame:
    """Every issue point of the window (§3: valid readings from the third of the record on, t_r in the
    window), with the times the waveform features need and the next valid reading in the window."""
    v = cuff.valid_readings(readings)
    t, mp = v["t"].to_numpy(float), v["map"].to_numpy(float)
    ip = INS.issue_points(v, window)
    rows = []
    for n in ip["n"].astype(int):
        nxt = n < len(t) and window[0] <= t[n] <= window[1]
        rows.append({"issue_n": n, "t_issue": t[n - 1], "t_prev": t[n - 2], "t_pp": t[n - 3],
                     "t_next": t[n] if nxt else np.nan, "y_next": mp[n] if nxt else np.nan})
    return pd.DataFrame(rows, columns=["issue_n", "t_issue", "t_prev", "t_pp", "t_next", "y_next"])


def _cuff_rows(readings: pd.DataFrame, ns) -> pd.DataFrame:
    v = cuff.valid_readings(readings)
    arr = [v[c].to_numpy(float) for c in ("t", "map", "sbp", "dbp")]
    return pd.DataFrame([{"issue_n": int(n), **INS.cuff_features(*arr, int(n))} for n in ns])


def _signal_rows(caseid: int, ip: pd.DataFrame, source) -> pd.DataFrame:
    """HR and waveform features (variant real) at the issue points, by the frozen feature functions."""
    iss = pd.DataFrame({"caseid": caseid, "window": "primary", "issue_n": ip["issue_n"],
                        "t_issue": ip["t_issue"], "t_prev": ip["t_prev"]})
    t, v = source.hr(caseid)
    h = HR.case_hr_features(iss, (np.asarray(t, float), np.asarray(v, float)))
    h = h.loc[h["variant"] == REAL].set_index("issue_n")[list(HR_COLS)]
    cw = W.CaseWindows(source.windows(caseid))
    rows = []
    for r in ip.itertuples(index=False):
        ti, tp, tpp, te, mask = W.variant_times(REAL, float(r.t_issue), float(r.t_prev), float(r.t_pp))
        f = W.forecast_features(cw, ti, tp, tpp, te, mask)
        rows.append({"issue_n": int(r.issue_n), "wf_available": bool(f["wf_available"]),
                     **{c: f[c] for c in WF_COLS}})
    wf = pd.DataFrame(rows).set_index("issue_n")
    out = pd.concat([h, wf], axis=1)
    out.columns = [f"{c}@{REAL}" for c in out.columns]
    return out


def _same(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOL)


def _check_scored(caseid, scored: pd.DataFrame, cuffr: pd.DataFrame, sig: pd.DataFrame | None, ip: pd.DataFrame):
    """The recomputed features of the scored issue points must equal the derived tables."""
    n = scored["issue_n"].to_numpy()
    c = cuffr.set_index("issue_n").loc[n]
    for col in CUFF_CHECK:
        if col in scored.columns and not _same(scored[col], c[col]).all():
            raise ExportError(f"case {caseid}: recomputed {col} differs from the instances table")
    i = ip.set_index("issue_n").loc[n]
    if not (_same(scored["t_target"], i["t_next"]).all() and _same(scored["y_map"], i["y_next"]).all()):
        raise ExportError(f"case {caseid}: next reading differs from the instances table")
    if sig is None:
        return
    s = sig.loc[n]
    for col in [f"{x}@{REAL}" for x in (*HR_COLS, *WF_COLS)]:
        if not _same(scored[col], s[col]).all():
            raise ExportError(f"case {caseid}: recomputed {col} differs from the feature tables")
    if not np.array_equal(scored[f"wf_available@{REAL}"].to_numpy(bool), s[f"wf_available@{REAL}"].to_numpy(bool)):
        raise ExportError(f"case {caseid}: recomputed waveform availability differs from the feature table")


# ----------------------------------------------------------------------------
# Curves of the frozen models
# ----------------------------------------------------------------------------

def _main_chain(frozen: dict):
    ch = frozen["chains"]["main"]
    sp = ch.spec
    if sp.hcol != "h_min" or sp.bincol != "h_bin" or any(b.hcol != "h_min" for b in sp.blocks):
        raise ExportError("the frozen main chain does not use h_min / h_bin as its horizon")
    return ch, sp.stages.index("M2"), sp.stages.index("M4")


def _with_h(frame: pd.DataFrame, h) -> pd.DataFrame:
    g = frame.copy()
    h = np.broadcast_to(np.asarray(h, dtype=float), (len(g),)).copy()
    g["h_s"], g["h_min"], g["h_bin"] = h, h / 60.0, h_bin_of(h)
    return g


def curves(frozen: dict, rows: pd.DataFrame, grid=H_GRID_S) -> tuple[np.ndarray, np.ndarray]:
    """(M2, M4) forecasts, rows x grid: the frozen main chain with only h set to each grid value."""
    ch, i2, i4 = _main_chain(frozen)
    m2, m4 = np.empty((len(rows), len(grid))), np.empty((len(rows), len(grid)))
    for j, g in enumerate(grid):
        p = ch.predict(Cols(_with_h(rows, g)))
        m2[:, j], m4[:, j] = p[:, i2], p[:, i4]
    return m2, m4


# ----------------------------------------------------------------------------
# The export
# ----------------------------------------------------------------------------

def build_site_cases(frozen: dict, T: Tables, source, widths: dict, fp: dict | None = None,
                     expected_n: int | None = None) -> tuple[list, dict]:
    """([CaseSummary] for every cohort case, {caseid: CaseSeries} for every scored case)."""
    from .pipeline import frozen_predictions
    fp = fp or frozen_predictions(frozen, T)
    frame, preds = fp["frame"], fp["preds"]
    if not np.array_equal(frame.index.to_numpy(), np.arange(len(frame))):
        raise ExportError("the forecast frame is not row-aligned with its predictions")
    pp = {m: per_patient(frame["caseid"].to_numpy(int), frame["y_map"].to_numpy(float), np.asarray(preds[m]))
          for m in SITE_MODELS}
    ana = set(T.analysable["caseid"].astype(int))
    need_cols = sorted(set(frame.columns) - {c for c in frame.columns if "@" in c and not c.endswith(f"@{REAL}")})
    summaries, series = [], {}
    for row in T.cases.sort_values("caseid").to_dict("records"):
        cid = int(row["caseid"])
        rd = T.readings.loc[T.readings["caseid"] == cid]
        v = cuff.valid_readings(rd)
        inw = (v["t"] >= WINDOW_S[0]) & (v["t"] <= WINDOW_S[1])
        n_valid = int(inw.sum())
        chart = source.chart(cid)
        pt, pv = source.pump(cid) if chart["pump_present"] else (np.zeros(0), np.zeros(0))
        base = dict(caseid=cid, n_readings=n_valid, median_interval_s=_num(cuff.median_interval(rd, *WINDOW_S)),
                    phenylephrine_ug=_num(chart["phenylephrine_ug"]), ephedrine_mg=_num(chart["ephedrine_mg"]),
                    pump_track=bool(chart["pump_present"]), pump_changes=pump_changes(pt, pv))
        if cid not in ana:
            summaries.append(S.CaseSummary(status="not analysable", reason=_reason(row, n_valid), **base))
            continue
        m = (frame["caseid"] == cid).to_numpy()
        n_fc = int(m.sum())
        if n_fc < MIN_FORECASTS_RMSE:
            summaries.append(S.CaseSummary(status="not scored", n_forecasts=n_fc,
                                           reason=f"{n_fc} scored forecasts (3 needed)", **base))
            continue
        cs, ser = _scored_case(frozen, cid, row, frame.loc[m], {k: np.asarray(p)[m] for k, p in preds.items()},
                               rd, source, widths, need_cols, base)
        cs.rmse = {k: float(pp[k].loc[cid, "rmse"]) for k in SITE_MODELS}
        summaries.append(cs)
        series[cid] = ser
    if expected_n is not None and len(series) != expected_n:
        raise ExportError(f"{len(series)} scored cases but the frozen scoring has n = {expected_n}")
    for c in summaries:
        c.validate()
    for s in series.values():
        s.validate()
    return summaries, series


def _scored_case(frozen, cid, row, scored: pd.DataFrame, pred: dict, rd, source, widths, need_cols, base):
    ip = issue_points(rd)
    cuffr = _cuff_rows(rd, ip["issue_n"])
    sig = _signal_rows(cid, ip, source)
    _check_scored(cid, scored, cuffr, sig if source.verify else None, ip)
    # one row per issue point: the derived row if scored, else the recomputed features
    is_scored = ip["issue_n"].isin(scored["issue_n"]).to_numpy()
    extra = ip.loc[~is_scored, ["issue_n"]].merge(cuffr, on="issue_n")
    extra = extra.join(sig, on="issue_n")
    for c in ("subjectid", "fast_cycling", "sex", "cerclage", "opstart", "vaso_stratum"):
        if c in scored.columns:
            extra[c] = scored[c].iloc[0]
    extra["caseid"] = cid
    extra["window"], extra["k"], extra["y_map"], extra["t_target"] = "primary", 1, np.nan, np.nan
    rows = pd.concat([scored[[c for c in need_cols if c in scored.columns]], extra], ignore_index=True)
    rows = rows.sort_values("issue_n").reset_index(drop=True)
    if not np.array_equal(rows["issue_n"].to_numpy(int), ip["issue_n"].to_numpy(int)):
        raise ExportError(f"case {cid}: issue points do not line up")
    rows[f"wf_available@{REAL}"] = rows[f"wf_available@{REAL}"].astype(bool)
    m2g, m4g = curves(frozen, rows)
    # the grid path at the realised h must reproduce the scored forecasts
    ch, i2, i4 = _main_chain(frozen)
    sc_rows = rows.loc[rows["issue_n"].isin(scored["issue_n"])]
    chk = ch.predict(Cols(_with_h(sc_rows, sc_rows["h_s"].to_numpy(float))))
    pos = {int(n): i for i, n in enumerate(scored["issue_n"])}
    order = [pos[int(n)] for n in sc_rows["issue_n"]]
    for k, i in (("M2", i2), ("M4", i4)):
        if np.max(np.abs(chk[:, i] - pred[k][order]), initial=0.0) > TOL:
            raise ExportError(f"case {cid}: the curve path does not reproduce the scored {k} forecasts")
    issues = []
    thr = HYPO_REL_BASELINE * float(scored["base_map"].iloc[0])
    for j, r in enumerate(ip.itertuples(index=False)):
        n = int(r.issue_n)
        h, c2, c4 = H_GRID_S.copy(), m2g[j].copy(), m4g[j].copy()
        pat = None
        if n in pos:
            i = pos[n]
            hr_ = float(scored["h_s"].iloc[i])
            p2, p4 = float(pred["M2"][i]), float(pred["M4"][i])
            k = int(np.searchsorted(h, hr_))
            if k < len(h) and h[k] == hr_:
                c2[k], c4[k] = p2, p4
            else:
                h, c2, c4 = np.insert(h, k, hr_), np.insert(c2, k, p2), np.insert(c4, k, p4)
            pat = {"M2": p2, "M4": p4}
        issues.append(S.IssuePoint(
            t_issue_s=float(r.t_issue), t_next_s=_num(r.t_next), y_next=_num(r.y_next), scored=pat is not None,
            h_s=h, m2=S.Curve(c2, half_widths(widths["M2"], h)), m4=S.Curve(c4, half_widths(widths["M4"], h)),
            m4_fallback=not bool(rows.loc[j, f"wf_available@{REAL}"]), pred_at_target=pat))
    # per-case summary numbers (§4.5 hypotension scoring; §6.5 coverage)
    last, y = scored["last_map"].to_numpy(float), scored["y_map"].to_numpy(float)
    event = (last >= thr) & (y < thr)
    flagged = {k: int(np.sum(event & (pred[k] < thr))) for k in ("M2", "M4")}
    v = cuff.valid_readings(rd)
    v = v.loc[(v["t"] >= WINDOW_S[0]) & (v["t"] <= WINDOW_S[1])]
    t_hr, v_hr = hr_series(*source.hr(cid))
    win = source.windows(cid)
    win = win.loc[(win["t_start"] >= WINDOW_S[0]) & (win["t_start"] < WINDOW_S[1])].sort_values("t_start")
    p_t, p_v = pump_series(*source.pump(cid)) if base["pump_track"] else (None, None)
    ser = S.CaseSeries(caseid=cid, window_s=WINDOW_S, baseline_map=float(scored["base_map"].iloc[0]),
                       incision_s=_num(row.get("opstart")), cuff_t=v["t"].to_numpy(float),
                       cuff_map=v["map"].to_numpy(float), hr_t=t_hr, hr_bpm=v_hr,
                       pleth_start=win["t_start"].to_numpy(float), pleth_usable=win["usable"].to_numpy(bool),
                       pump_t=p_t, pump_ml_h=p_v, issues=issues)
    cs = S.CaseSummary(status="scored", n_forecasts=len(scored),
                       waveform_coverage=float(scored[f"wf_available@{REAL}"].astype(bool).mean()),
                       new_onset_80=int(event.sum()), flagged_80=flagged, **base)
    return cs, ser


def write_site_cases(cases_dir: Path, summaries: list, series: dict, synthetic: bool, meta: dict) -> dict:
    """Write `<cases-dir>/cases.json` and `series/<caseid>.json` (btc/site/series.py), replacing the
    previous export, and `MANIFEST.json` with their SHA-256. Returns the manifest."""
    cases_dir = Path(cases_dir)
    (cases_dir / "series").mkdir(parents=True, exist_ok=True)
    for p in (cases_dir / "series").glob("*.json"):
        p.unlink()
    S.write_cases(cases_dir, summaries, series, synthetic=synthetic, meta=meta)
    files = sorted([cases_dir / "cases.json", *(cases_dir / "series").glob("*.json")])
    man = {"meta": meta, "files": {str(p.relative_to(cases_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in files}}
    (cases_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    return man
