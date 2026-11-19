"""Per-case inputs of the case browser, the summary figure and the replay (SPEC §8.1-§8.3).

The frozen-scoring step (SPEC §9 steps 7-8) writes one folder, here called `<cases-dir>`
(`scripts/run_frozen_scoring.py series`, `btc/models/site_cases.py`):

    <cases-dir>/cases.json              one entry per cesarean cohort case, analysable or not
    <cases-dir>/series/<caseid>.json    one file per scored case
    <cases-dir>/MANIFEST.json           SHA-256 of the files above and their provenance

These files hold VitalDB-derived series. Keep them in a git-ignored folder (for example
`data/derived/site/`), never in `docs/` or `results/`: the site build turns them into images only,
so no VitalDB data are redistributed (SPEC §2.1, §8.3).

cases.json
----------
    {"schema": "btc-site-cases/1", "synthetic": false, "cohort": "S-ob", "meta": {...},
     "cases": [<CaseSummary>, ...]}                  (meta: provenance, optional, not read)

CaseSummary (one per cohort case; numbers are the frozen-scoring values):
    caseid              int
    status              "scored" | "not analysable" | "not analysed" | "not scored"
                        (not analysed = analysable but a neuromuscular blocker recorded, SPEC §2.2;
                        not scored = analysed but < 3 scored forecasts, so not in the primary outcome;
                        `case_status` gives the status)
    reason              str, required unless scored, e.g. "3 valid readings in the window";
                        for "not analysed" it is NMB_REASON
    n_readings          valid cuff readings in the primary window (0-30 min)
    median_interval_s   median interval between consecutive valid readings in the window, s
    n_forecasts         scored primary forecasts (next-reading targets)
    rmse                {"M2": mmHg, "M3": mmHg, "M4": mmHg, "M4-P1": mmHg}  patient-level RMSE
    waveform_coverage   share (0-1) of this case's primary forecasts with waveform features
    new_onset_80        new-onset readings below 80% of the record-start baseline in the window: scored
                        targets below the threshold whose issue reading is not (SPEC §4.5, §6.6)
    flagged_80          {"M2": int, "M4": int}  of those, flagged one reading ahead by each model
    phenylephrine_ug    `intraop_phe` chart total in µg, or null if not recorded
    ephedrine_mg        `intraop_eph` chart total in mg, or null if not recorded
    pump_track          true if the case has an `Orchestra/PHEN_RATE` track
    pump_changes        number of PHEN_RATE changes in the primary window (§8.2 rule 3)
Only caseid, status and reason are needed for a case that is not scored. Phenylephrine (µg) and
ephedrine (mg) are kept apart; nothing here or downstream adds them (SPEC §2.6).

series/<caseid>.json (CaseSeries)
---------------------------------
    {"schema": "btc-site-series/1", "caseid": int, "window_s": [0, 1800],
     "baseline_map": mmHg,                    record-start baseline (mean of the first 3 valid readings)
     "incision_s": s or null,                 `opstart` on the record clock
     "cuff":  {"t_s": [...], "map": [...]},   every valid reading in the window (display time t_r)
     "hr":    {"t_s": [...], "bpm": [...]},   Solar8000/HR, at most 1 Hz, null for missing
     "pleth": {"window_start_s": [...], "usable": [bool, ...]},   10-s quality windows (§5)
     "pump":  {"t_s": [...], "ml_h": [...]} or null,  PHEN_RATE in mL/h; each value holds until the next
     "issues": [<IssuePoint>, ...]}

IssuePoint (one per issue point of the window, from the third valid reading on; SPEC §3):
    t_issue_s           issue time = display time of the issue reading
    t_next_s            display time of the next valid reading in the window, or null
    y_next              MAP of that reading, or null
    scored              true if that reading is a scored primary target (within 400 s, in the window)
    h_s                 horizon grid of the curves, ascending, within 15-400 s (e.g. every 5 s)
    m2, m4              {"map": [...], "half90": [...]}: the frozen model's forecast curve
                        ŷ(t_issue + h) on h_s, and the half-width of its 90% cross-conformal interval
                        at each h (constant within an h-bin; SPEC §6.7)
    m4_fallback         true if M4 returned the M3 forecast (no waveform)
    pred_at_target      {"M2": mmHg, "M4": mmHg} the scored point forecasts at the realised h, or
                        null when not scored (the error row uses these, never an interpolation)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CASES_SCHEMA = "btc-site-cases/1"
SERIES_SCHEMA = "btc-site-series/1"
STATUSES = ("scored", "not analysable", "not analysed", "not scored")
NMB_REASON = "neuromuscular blocker recorded"       # SPEC §2.2: analysable, excluded from fitting and scoring
STRATA = ("pump", "bolus total only", "neither recorded")
H_MIN_S, H_MAX_S = 15.0, 400.0


def case_status(analysable: bool, nmb_recorded: bool, entered: bool) -> str:
    """Status of a cohort case (SPEC §2.2, §6.1): not analysable (signal and cuff rule); not analysed
    (analysable, neuromuscular blocker recorded; reason NMB_REASON); scored (entered the primary
    outcome); not scored (analysed, < 3 scored forecasts)."""
    if not analysable:
        return "not analysable"
    if nmb_recorded:
        return "not analysed"
    return "scored" if entered else "not scored"


def _arr(v) -> np.ndarray:
    return np.asarray([np.nan if x is None else x for x in v], dtype=float)


def _list(a) -> list:
    return [None if (isinstance(x, float) and math.isnan(x)) else x for x in np.asarray(a, dtype=float).tolist()]


@dataclass
class CaseSummary:
    caseid: int
    status: str
    reason: str | None = None
    n_readings: int | None = None
    median_interval_s: float | None = None
    n_forecasts: int | None = None
    rmse: dict = field(default_factory=dict)
    waveform_coverage: float | None = None
    new_onset_80: int | None = None
    flagged_80: dict = field(default_factory=dict)
    phenylephrine_ug: float | None = None
    ephedrine_mg: float | None = None
    pump_track: bool = False
    pump_changes: int = 0

    @property
    def scored(self) -> bool:
        return self.status == "scored"

    @property
    def stratum(self) -> str:
        """SPEC §2.6 strata, counting phenylephrine or ephedrine (never their sum)."""
        if self.pump_track:
            return "pump"
        if (self.phenylephrine_ug or 0) > 0 or (self.ephedrine_mg or 0) > 0:
            return "bolus total only"
        return "neither recorded"

    def validate(self):
        if self.status not in STATUSES:
            raise ValueError(f"case {self.caseid}: unknown status {self.status!r}")
        if not self.scored:
            if not self.reason:
                raise ValueError(f"case {self.caseid}: a case that is not scored needs a reason")
            if self.status == "not analysed" and self.reason != NMB_REASON:
                raise ValueError(f"case {self.caseid}: a case that is not analysed has the reason {NMB_REASON!r}")
            return
        for k in ("n_readings", "median_interval_s", "n_forecasts", "waveform_coverage", "new_onset_80"):
            if getattr(self, k) is None:
                raise ValueError(f"case {self.caseid}: {k} missing")
        for m in ("M2", "M3", "M4", "M4-P1"):
            v = self.rmse.get(m)
            if v is None or not np.isfinite(v):
                raise ValueError(f"case {self.caseid}: RMSE of {m} missing")
        for m in ("M2", "M4"):
            if m not in self.flagged_80:
                raise ValueError(f"case {self.caseid}: flagged_80[{m}] missing")

    @classmethod
    def from_dict(cls, d: dict) -> "CaseSummary":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        c = cls(**known)
        c.caseid = int(c.caseid)
        return c

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Curve:
    map: np.ndarray
    half90: np.ndarray


@dataclass
class IssuePoint:
    t_issue_s: float
    t_next_s: float | None
    y_next: float | None
    scored: bool
    h_s: np.ndarray
    m2: Curve
    m4: Curve
    m4_fallback: bool = False
    pred_at_target: dict | None = None

    def end_h(self, window_end: float) -> float:
        """Horizon at which the complete-frame curve stops: the next reading, or 400 s / window end."""
        if self.t_next_s is not None and self.t_next_s - self.t_issue_s <= H_MAX_S:
            return self.t_next_s - self.t_issue_s
        return min(H_MAX_S, window_end - self.t_issue_s)


@dataclass
class CaseSeries:
    caseid: int
    window_s: tuple
    baseline_map: float
    incision_s: float | None
    cuff_t: np.ndarray
    cuff_map: np.ndarray
    hr_t: np.ndarray
    hr_bpm: np.ndarray
    pleth_start: np.ndarray
    pleth_usable: np.ndarray
    pump_t: np.ndarray | None
    pump_ml_h: np.ndarray | None
    issues: list

    @property
    def has_pump(self) -> bool:
        return self.pump_t is not None and len(self.pump_t) > 0

    @property
    def pleth_usable_share(self) -> float:
        return float(np.mean(self.pleth_usable)) if len(self.pleth_usable) else float("nan")

    def validate(self):
        w0, w1 = self.window_s
        if not (len(self.cuff_t) == len(self.cuff_map) and len(self.cuff_t) >= 3):
            raise ValueError(f"case {self.caseid}: cuff arrays")
        if np.any(np.diff(self.cuff_t) <= 0) or self.cuff_t[0] < w0 or self.cuff_t[-1] > w1:
            raise ValueError(f"case {self.caseid}: cuff times must ascend inside the window")
        if len(self.hr_t) != len(self.hr_bpm) or len(self.pleth_start) != len(self.pleth_usable):
            raise ValueError(f"case {self.caseid}: hr or pleth arrays")
        if self.has_pump and len(self.pump_t) != len(self.pump_ml_h):
            raise ValueError(f"case {self.caseid}: pump arrays")
        for ip in self.issues:
            h = ip.h_s
            if len(h) < 2 or np.any(np.diff(h) <= 0) or h[0] < H_MIN_S - 1e-9 or h[-1] > H_MAX_S + 1e-9:
                raise ValueError(f"case {self.caseid}: horizon grid must ascend within 15-400 s")
            for c in (ip.m2, ip.m4):
                if len(c.map) != len(h) or len(c.half90) != len(h) or not np.all(np.isfinite(c.map)):
                    raise ValueError(f"case {self.caseid}: forecast curve arrays")
            if ip.scored and (ip.pred_at_target is None or ip.y_next is None or ip.t_next_s is None):
                raise ValueError(f"case {self.caseid}: a scored issue point needs its target and forecasts")

    @classmethod
    def from_dict(cls, d: dict) -> "CaseSeries":
        if d.get("schema") != SERIES_SCHEMA:
            raise ValueError(f"unknown series schema {d.get('schema')!r}")
        pump = d.get("pump")
        issues = [IssuePoint(t_issue_s=float(i["t_issue_s"]),
                             t_next_s=None if i.get("t_next_s") is None else float(i["t_next_s"]),
                             y_next=None if i.get("y_next") is None else float(i["y_next"]),
                             scored=bool(i["scored"]), h_s=_arr(i["h_s"]),
                             m2=Curve(_arr(i["m2"]["map"]), _arr(i["m2"]["half90"])),
                             m4=Curve(_arr(i["m4"]["map"]), _arr(i["m4"]["half90"])),
                             m4_fallback=bool(i.get("m4_fallback", False)),
                             pred_at_target=i.get("pred_at_target"))
                  for i in d["issues"]]
        s = cls(caseid=int(d["caseid"]), window_s=tuple(d.get("window_s", (0.0, 1800.0))),
                baseline_map=float(d["baseline_map"]),
                incision_s=None if d.get("incision_s") is None else float(d["incision_s"]),
                cuff_t=_arr(d["cuff"]["t_s"]), cuff_map=_arr(d["cuff"]["map"]),
                hr_t=_arr(d["hr"]["t_s"]), hr_bpm=_arr(d["hr"]["bpm"]),
                pleth_start=_arr(d["pleth"]["window_start_s"]),
                pleth_usable=np.asarray(d["pleth"]["usable"], dtype=bool),
                pump_t=None if not pump else _arr(pump["t_s"]),
                pump_ml_h=None if not pump else _arr(pump["ml_h"]), issues=issues)
        s.validate()
        return s

    def to_dict(self) -> dict:
        return {"schema": SERIES_SCHEMA, "caseid": self.caseid, "window_s": list(self.window_s),
                "baseline_map": self.baseline_map, "incision_s": self.incision_s,
                "cuff": {"t_s": _list(self.cuff_t), "map": _list(self.cuff_map)},
                "hr": {"t_s": _list(self.hr_t), "bpm": _list(self.hr_bpm)},
                "pleth": {"window_start_s": _list(self.pleth_start), "usable": [bool(x) for x in self.pleth_usable]},
                "pump": None if not self.has_pump else {"t_s": _list(self.pump_t), "ml_h": _list(self.pump_ml_h)},
                "issues": [{"t_issue_s": i.t_issue_s, "t_next_s": i.t_next_s, "y_next": i.y_next, "scored": i.scored,
                            "h_s": _list(i.h_s), "m2": {"map": _list(i.m2.map), "half90": _list(i.m2.half90)},
                            "m4": {"map": _list(i.m4.map), "half90": _list(i.m4.half90)},
                            "m4_fallback": i.m4_fallback, "pred_at_target": i.pred_at_target}
                           for i in self.issues]}


def load_cases(cases_dir: Path) -> tuple[bool, list, dict]:
    """(synthetic flag, [CaseSummary], {caseid: CaseSeries}) from `<cases-dir>`."""
    cases_dir = Path(cases_dir)
    d = json.loads((cases_dir / "cases.json").read_text())
    if d.get("schema") != CASES_SCHEMA:
        raise ValueError(f"unknown cases schema {d.get('schema')!r}")
    cases = [CaseSummary.from_dict(c) for c in d["cases"]]
    ids = [c.caseid for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case IDs in cases.json")
    series = {}
    for c in cases:
        c.validate()
        if c.scored:
            p = cases_dir / "series" / f"{c.caseid}.json"
            s = CaseSeries.from_dict(json.loads(p.read_text()))
            if s.caseid != c.caseid:
                raise ValueError(f"{p.name}: case ID mismatch")
            series[c.caseid] = s
    return bool(d.get("synthetic", False)), cases, series


def write_cases(cases_dir: Path, cases: list, series: dict, synthetic: bool, meta: dict | None = None):
    """Write the case folder. `meta` (provenance: commit, data, stage) is stored under "meta" in
    cases.json; readers ignore it."""
    cases_dir = Path(cases_dir)
    (cases_dir / "series").mkdir(parents=True, exist_ok=True)
    head = {"schema": CASES_SCHEMA, "synthetic": synthetic, "cohort": "S-ob"}
    if meta is not None:
        head["meta"] = meta
    (cases_dir / "cases.json").write_text(json.dumps(
        {**head, "cases": [c.to_dict() for c in cases]}, indent=1, ensure_ascii=False, default=str) + "\n")
    for cid, s in series.items():
        (cases_dir / "series" / f"{cid}.json").write_text(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
