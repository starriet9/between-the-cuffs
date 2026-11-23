"""Adapter from the model pipeline's result files to everything the site shows.

The headline words come only from `btc/models/headline.py`, through the files that
`btc/models/pipeline.py` writes; nothing here re-implements the row logic. Read from a results root
(default `results/`):

    sdev/confirmatory/summary.json           question, answer_line, sentence_1, sentence_3,
                                             decision, N, contrasts, waveform_coverage
    sdev/confirmatory/model_summary.csv      README block 5 table (S-dev)
    sdev/confirmatory/per_patient_primary.csv
    sdev/secondary/summary.json              gradient_boosting.readme_line (SPEC §6.8)
    sdev/secondary/hypotension.csv
    sob/frozen_scoring/summary.json          n, sentence_2, cesarean_line, contrasts, coverage,
                                             n_analysable_cases, n_nmb_recorded_cases (SPEC §2.2)
    sob/frozen_scoring/model_summary.csv
    sob/frozen_scoring/per_patient_primary.csv
    sob/after_frozen_scoring/hypotension.csv

the case folder of `btc/site/series.py`, and the cerclage flags of the S-dev case table
(`data/derived/sdev_cases.parquet`, column `cerclage`; SPEC §2.2, §6.7). The label is split off the
answer line with `headline.label()` on the stored decision, and the answer line must start with it.
"""
from __future__ import annotations

import errno
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from btc.models import headline as H
from btc.models.secondary import hypotension_line

from .config import SiteConfig
from .featured import select_featured
from .series import load_cases

README_MODELS = ("M00", "M0", "M2", "M3", "M4", "M4-P1")          # SPEC §8.1 block 5
MODEL_DESCRIPTIONS = {
    "M00": "running mean of the readings so far",
    "M0": "last reading (persistence)",
    "M2": "cuff only, third order",
    "M3": "M2 + heart rate",
    "M4": "M3 + pleth waveform",
    "M4-P1": "M4 on another patient's signals (placebo)",
}
PANEL_CONTRASTS = (("M3", "M3 vs M2"), ("M4", "M4 vs M2"), ("M4-P1", "M4-P1 vs M2"))
COHORTS = ("S-dev", "S-ob")
COHORT_SOB_SIZE = 21          # SPEC §2.2


def _num(v) -> float:
    """Undo outputs._clean: None -> NaN, 'inf'/'-inf' -> +/-inf."""
    if v is None:
        return float("nan")
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf}[v]
    return float(v)


@dataclass
class RR:
    """One contrast, as stored by bootstrap.Contrast.as_dict()."""
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
    meets_rule: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "RR":
        return cls(model=d["model"], comparator=d["comparator"], n_patients=int(d["n_patients"]),
                   **{k: _num(d[k]) for k in ("mean_model", "mean_comp", "delta", "delta_lo", "delta_hi",
                                              "rr", "rr_lo", "rr_hi")},
                   meets_rule=bool(d.get("meets_rule", False)))


@dataclass
class Headline:
    row: int
    label: str
    question: str
    answer_line: str            # as written by headline.answer_line (label first)
    answer_rest: str            # the answer line after the label
    emphasis: list              # substrings of answer_rest set in semibold on the page
    cesarean_line: str
    sentence_1: str
    sentence_2: str
    sentence_3: str
    gb_line: str | None
    coverage_rule: bool
    N: int
    n: int


@dataclass
class ModelRow:
    model: str
    description: str
    sdev: tuple                 # (mean, lo, hi) mmHg
    sob: tuple


@dataclass
class SiteData:
    synthetic: bool
    provenance: list
    config: SiteConfig
    headline: Headline
    contrasts: dict             # cohort -> {"M3"|"M4"|"M4-P1": RR} against M2
    table: list
    coverage: dict              # cohort -> share of primary forecasts with waveform features
    hypotension_line: str
    per_patient: dict           # cohort -> DataFrame[caseid, model, rmse]
    cases: list                 # CaseSummary, whole cesarean cohort
    series: dict                # caseid -> CaseSeries (scored cases)
    featured: dict              # rule -> caseid
    n_analysable_sob: int
    n_nmb_sob: int = 0          # analysable but not analysed: neuromuscular blocker recorded (SPEC §2.2)
    cerclage: frozenset | None = None   # S-dev cerclage case IDs (SPEC §6.7); None = flags not loaded
    commit: str | None = None
    variant: str = ""
    notes: list = field(default_factory=list)

    @property
    def scored_cases(self) -> list:
        return [c for c in self.cases if c.scored]

    @property
    def cerclage_sdev(self) -> list:
        """The cerclage cases of the S-dev per-patient file, sorted: the cases the per-patient figure marks."""
        ids = set(int(c) for c in self.per_patient["S-dev"]["caseid"])
        return sorted(ids & set(self.cerclage or ()))


def answer_emphasis(row: int, rest: str, x: RR, G: float) -> list:
    """Substrings of the answer line set in semibold on the page: X% together with the mmHg pair,
    never X% alone; row 6: 'did not meet the pre-specified rule' and G%; row 1: none
    [judgment: with the placebo failed, no number is emphasised]."""
    if row == 1:
        return []
    if row == 6:
        out = ["did not meet the pre-specified rule"]
        if G > 0:
            out.append(f"{H.pct(G)}%")
    else:
        out = [f"{H.pct(x.rr)}%", f"from {H.mm(x.mean_comp)} to {H.mm(x.mean_model)} mmHg"]
    pos = 0
    for s in out:
        i = rest.find(s, pos)
        if i < 0:
            raise ValueError(f"emphasis {s!r} not found in the answer line")
        pos = i + len(s)
    return out


def headline_from(conf: dict, frz: dict, sec: dict) -> Headline:
    dec = {k: (_num(v) if k in ("G", "g") else v) for k, v in conf["decision"].items()}
    lab = H.label(dec)
    ans = conf["answer_line"]
    if not ans.startswith(lab + " "):
        raise ValueError("the answer line does not start with its row label")
    rest = ans[len(lab) + 1:]
    x = RR.from_dict(conf["contrasts"]["M4 vs M2"])
    gb = (sec.get("gradient_boosting") or {}).get("readme_line")
    return Headline(row=int(conf["headline_row"]), label=lab, question=conf["question"], answer_line=ans,
                    answer_rest=rest, emphasis=answer_emphasis(int(conf["headline_row"]), rest, x, dec["G"]),
                    cesarean_line=frz["cesarean_line"], sentence_1=conf["sentence_1"], sentence_2=frz["sentence_2"],
                    sentence_3=conf["sentence_3"], gb_line=gb, coverage_rule=bool(frz["coverage_rule_applies"]),
                    N=int(conf["N"]), n=int(frz["n"]))


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, skiprows=1)


def _is_synthetic(meta: dict | None) -> bool:
    return "synthetic" in str((meta or {}).get("data", "")).lower()


def load_cerclage(path: Path, sdev_caseids) -> frozenset:
    """S-dev case IDs flagged `cerclage` in the S-dev case table. Every case of the S-dev per-patient file
    must be in the table, so that a wrong or stale table cannot pass unnoticed."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(errno.ENOENT, "missing input", str(p))
    t = pd.read_parquet(p, columns=["caseid", "cerclage"])
    if t["cerclage"].isna().any():
        raise ValueError(f"{p.name}: missing cerclage flags")
    missing = sorted(set(int(c) for c in sdev_caseids) - set(int(c) for c in t["caseid"]))
    if missing:
        raise ValueError(f"{p.name}: {len(missing)} case(s) of the S-dev per-patient file are not in this table "
                         f"(first: {missing[0]})")
    return frozenset(int(c) for c in t.loc[t["cerclage"].astype(bool), "caseid"])


def load_site_data(results_root: Path, cases_dir: Path, config: SiteConfig, variant: str = "",
                   sdev_cases: Path | None = None) -> SiteData:
    """`sdev_cases`: the S-dev case table with the cerclage flags; without it `cerclage` stays None, which
    a preview accepts and the publish build refuses."""
    r = Path(results_root)
    rd = lambda p: json.loads((r / p).read_text())
    conf = rd("sdev/confirmatory/summary.json")
    sec = rd("sdev/secondary/summary.json")
    frz = rd("sob/frozen_scoring/summary.json")
    hl = headline_from(conf, frz, sec)
    metas = [conf.get("meta"), sec.get("meta"), frz.get("meta")]
    contrasts = {"S-dev": {k: RR.from_dict(conf["contrasts"][c]) for k, c in PANEL_CONTRASTS},
                 "S-ob": {k: RR.from_dict(frz["contrasts"][c]) for k, c in PANEL_CONTRASTS}}
    ms = {"S-dev": _csv(r / "sdev/confirmatory/model_summary.csv").set_index("model"),
          "S-ob": _csv(r / "sob/frozen_scoring/model_summary.csv").set_index("model")}
    tri = lambda d, m: (float(d.loc[m, "rmse_mean"]), float(d.loc[m, "rmse_mean_ci_lo"]),
                        float(d.loc[m, "rmse_mean_ci_hi"]))
    table = [ModelRow(m, MODEL_DESCRIPTIONS[m], tri(ms["S-dev"], m), tri(ms["S-ob"], m)) for m in README_MODELS]
    hyp = {"Non-cesarean": _csv(r / "sdev/secondary/hypotension.csv"),
           "Cesarean": _csv(r / "sob/after_frozen_scoring/hypotension.csv")}
    pp = {"S-dev": _csv(r / "sdev/confirmatory/per_patient_primary.csv"),
          "S-ob": _csv(r / "sob/frozen_scoring/per_patient_primary.csv")}
    syn_cases, cases, series = load_cases(cases_dir)
    cerclage = None if sdev_cases is None else load_cerclage(sdev_cases, pp["S-dev"]["caseid"])
    site = SiteData(
        synthetic=syn_cases or any(_is_synthetic(m) for m in metas),
        provenance=[str((m or {}).get("data")) for m in metas], config=config, headline=hl,
        contrasts=contrasts, table=table,
        coverage={"S-dev": float(conf["waveform_coverage"]), "S-ob": float(frz["waveform_coverage"])},
        hypotension_line=hypotension_line(hyp),
        per_patient={k: v[["caseid", "model", "rmse"]].copy() for k, v in pp.items()},
        cases=cases, series=series, featured=select_featured(cases),
        n_analysable_sob=int(frz["n_analysable_cases"]), n_nmb_sob=int(frz["n_nmb_recorded_cases"]),
        cerclage=cerclage, commit=(conf.get("meta") or {}).get("commit"),
        variant=variant)
    return site


def consistency_problems(site: SiteData) -> list:
    """Cross-checks between the result files and the case folder; the publish build refuses on any."""
    out = []
    scored = site.scored_cases
    if len(scored) != site.headline.n:
        out.append(f"{len(scored)} scored tiles but n = {site.headline.n}")
    if len(site.cases) != COHORT_SOB_SIZE:
        out.append(f"{len(site.cases)} cohort cases, expected {COHORT_SOB_SIZE}")
    n_na = sum(1 for c in site.cases if c.status == "not analysed")
    if n_na != site.n_nmb_sob:
        out.append(f"{n_na} cases marked not analysed but {site.n_nmb_sob} with a neuromuscular blocker "
                   "recorded (SPEC §2.2)")
    if site.featured.get(1) is None:
        out.append("no rule-1 case (SPEC §8.2)")
    pp = site.per_patient["S-ob"]
    m4 = pp.loc[pp["model"] == "M4"].set_index("caseid")["rmse"]
    for c in scored:
        if c.caseid not in site.series:
            out.append(f"case {c.caseid}: no series file")
        if c.caseid in m4.index and abs(float(m4.loc[c.caseid]) - c.rmse["M4"]) > 1e-6:
            out.append(f"case {c.caseid}: M4 RMSE differs from per_patient_primary.csv")
        elif c.caseid not in m4.index:
            out.append(f"case {c.caseid}: not in the S-ob per-patient file")
    return out
