"""One source for every word the README and the page share (SPEC §8.1, §8.3).

`blocks(site)` returns README blocks 1-8 as a small document model. `readme.py` renders it to
Markdown and `page.py` to HTML, so the page's top block repeats README block 1 word for word.
The headline sentences come unchanged from `btc/models/headline.py` (via `results.SiteData`).

Segment kinds: t (text), b (bold in both), em (semibold on the page only, plain in the README),
link (text + link key), code, sep (" · "), date (a date, optionally linked).
Link keys are resolved per medium by the renderers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from btc.models.headline import mm, pct

from .featured import RULE_TAGS

SEP = " · "
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
VITALDB_URL = "https://vitaldb.net/"
VITALDB_DOI_URL = "https://doi.org/10.1038/s41597-022-01411-5"

# SPEC §2.1: required on every public output, word for word
CREDIT_STATEMENT = ("Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; "
                    "signals resampled, filtered and segmented; features computed; plots and summary statistics "
                    "derived. No attempt was made to identify any person.")
NOT_ADVICE = "Research prototype, not medical advice."

SUMMARY_ALT = ("Left: one cesarean case (VitalDB case {cid}, chosen by a pre-set rule) with cuff readings, "
               "forecasts, heart rate and pleth. Right: error reduction against the cuff-only model, with 95% CIs.")

# Block 2 readings, taken from SPEC §7 (the pre-written outcome readings)
MEANING_POSITIVE = ("Heart rate and the pleth may partly stand in for the missing drug input. Such a result cannot "
                    "separate physiology from drug effect, and it is probably an upper estimate of what the same "
                    "signals would add to a system that already has timed doses.")
MEANING_HR = ("Heart rate may partly stand in for unrecorded vasopressor doses: it falls after phenylephrine and "
              "rises after ephedrine. Such a result cannot separate physiology from drug effect.")
MEANING_NULL = ("A null is harder to attribute to the drug proxy, but opposite heart-rate responses to the two "
                "vasopressors could also cancel in a linear model. It bounds only this feature set under a linear "
                "model; the gradient-boosting check in `REPORT.md` asks the same question without the linear form.")
MEANING_NO_CLAIM = "The placebo control failed, so no claim is made about the added signals."
EITHER_WAY = "Either way, only data with time-stamped vasopressor doses can settle the question."
MEANING_HEADING = "What this means (written before the result)"

# Block 3: the same proposal for either outcome (SPEC §8.1); a draft for the author
NEXT_EXPERIMENT = (
    "On a platform with time-stamped vasopressor doses and 1-minute cuffs, test the increment of the pleth "
    "waveform over an ARX model that includes the drug term: retrospectively first, then scored silently and "
    "prospectively, with the same placebo and echo controls. The drug-proxy confound is named in advance: heart "
    "rate falls after phenylephrine and rises after ephedrine, so only with the doses in the model can a gain from "
    "heart rate or the pleth be told apart from the drug. Two measurements this dataset lacks would separate more: "
    "the monitor's perfusion-index numeric (does pleth amplitude follow vascular tone, or only the display "
    "scaling?) and a recorded probe site (does pulse arrival time cross the blocked territory, and does inflation "
    "of a cuff on the same arm shape the pleth?). These are proposals, not results.")

LIMITS = (
    "The analysis is retrospective, with 21 cesarean cases ({n_analysable} analysable{nmb}), 19 flagged emergency.",
    "The spinal injection time is unknown, and the baseline is a record-start value, not pre-spinal.",
    "Bolus times are absent and the models have no drug input, so a heart-rate gain may reflect unrecorded drugs.",
    "This is a research prototype, not medical advice.",
)

# SPEC §2.2: shown only if an analysable cesarean case had a neuromuscular blocker recorded
LIMITS_NMB = ", {k} of them not analysed because a neuromuscular blocker was recorded"


def limits(site) -> list:
    nmb = LIMITS_NMB.format(k=site.n_nmb_sob) if site.n_nmb_sob else ""
    return [x.format(n_analysable=site.n_analysable_sob, nmb=nmb) for x in LIMITS]


AUTHOR_NOTE = ("The echo and placebo controls ask of a time-series model the question I study in imaging: does the "
               "model use the evidence in front of it (here, the pleth), or infer from context (the cuff history)?")

SHEET_CAPTION = ("Every case scored in the single frozen run; none is left out. Sorted by the full model's error, "
                 "lowest first. Single cases show what the models did; they do not decide the answer.")
VASO_NOTE = "Bolus times are not recorded. Chart totals miss pump infusions."
PER_PATIENT_ALT = ("Per-patient RMSE of M0, M2, M4 and M4-P1: one line per patient, for the non-cesarean and "
                   "the cesarean cases.")
PER_PATIENT_CAPTION = "Each grey line is one patient; the larger marks are the patient-level means."


@dataclass
class Seg:
    kind: str
    text: str
    href: str | None = None


@dataclass
class Unit:
    role: str
    segs: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Block:
    num: int
    key: str
    heading: str | None
    units: list


def t(s):
    return Seg("t", s)


def code_split(s: str) -> list:
    """Text with `backtick` spans -> t / code segments."""
    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", s)):
        if part:
            out.append(Seg("code" if i % 2 else "t", part))
    return out


def emphasised(rest: str, spans: list) -> list:
    out, pos = [], 0
    for s in spans:
        i = rest.find(s, pos)
        if i > pos:
            out.append(t(rest[pos:i]))
        out.append(Seg("em", s))
        pos = i + len(s)
    if pos < len(rest):
        out.append(t(rest[pos:]))
    return out


def registration_nodes(site) -> list:
    """The registration line as nodes (lists of segments); the page draws them as a timeline."""
    c = site.config
    nodes = [
        [Seg("link", "Pre-registered", "prereg"), t(" on "), Seg("date", c.registered_date),
         t(f" (public release {c.registered_public_date}), before any model was fitted")],
        [t("models frozen "), Seg("date", c.frozen_date, "frozen")],
        [t("cesarean cases scored once on "), Seg("date", c.scored_date, "scored")],
    ]
    if c.deviations:
        k = int(c.deviations)
        nodes.append([t(f"{k} deviation{'s' if k != 1 else ''} from the registration, listed in "),
                      Seg("link", "REPORT.md", "report")])
    chips = [[t("placebo control")], [t("open data ("), Seg("link", "VitalDB", "vitaldb"), t(")")]]
    return nodes, chips


def links_line(site) -> list:
    c, n = site.config, site.headline.n
    items = [[Seg("link", "20-s replay", "replay")]]
    if c.narrated_replay_url:
        items.append([Seg("link", "Narrated replay", "narrated")])
    items.append([Seg("link", f"All {n} cesarean cases", "cases")])
    if c.pdf_url:
        items.append([Seg("link", "One-page PDF", "pdf")])
    items.append([Seg("link", "Full report", "report")])
    items.append([Seg("link", "How to read these numbers", "reading")])
    items.append([t("Reproduce: "), Seg("code", c.reproduce_command)])
    out = []
    for i, it in enumerate(items):
        if i:
            out.append(Seg("sep", SEP))
        out.extend(it)
    return out


def block1(site) -> Block:
    h = site.headline
    nodes, chips = registration_nodes(site)
    reg = []
    for i, nd in enumerate(nodes + chips):
        if i:
            reg.append(Seg("sep", SEP))
        reg.extend(nd)
    rule1 = site.featured.get(1)
    units = [
        Unit("question", [t(h.question)]),
        Unit("answer", [Seg("b", h.label), t(" ")] + emphasised(h.answer_rest, h.emphasis),
             {"label": h.label, "rest": emphasised(h.answer_rest, h.emphasis)}),
        Unit("cesarean", [t(h.cesarean_line)]),
        Unit("registration", reg, {"nodes": nodes, "chips": chips}),
        Unit("figure", [], {"alt": SUMMARY_ALT.format(cid=rule1), "img": "img/summary.png"}),
        Unit("links", links_line(site)),
        Unit("sentence", [t(h.sentence_1)]),
        Unit("sentence", [t(h.sentence_2)]),
        Unit("sentence", [t(h.sentence_3)]),
    ]
    return Block(1, "answer", None, units)


def meaning_paragraphs(row: int) -> list:
    if row == 1:
        return [MEANING_NO_CLAIM, EITHER_WAY]
    if row == 3:
        return [MEANING_HR, EITHER_WAY]
    if row == 6:
        return [MEANING_NULL, EITHER_WAY]
    return [MEANING_POSITIVE, EITHER_WAY]


def fmt_ci_mm(v: tuple) -> str:
    return f"{mm(v[0])} ({mm(v[1])} to {mm(v[2])})"


def table_rows(site) -> tuple:
    h = site.headline
    head = ["Model",
            f"Non-cesarean, N = {h.N}: mean RMSE (95% CI), mmHg",
            f"Cesarean, n = {h.n}: mean RMSE (95% CI), mmHg"]
    rows = [[f"{r.model} · {r.description}", fmt_ci_mm(r.sdev), fmt_ci_mm(r.sob)] for r in site.table]
    rows.append(["Waveform coverage · share of forecasts with waveform features",
                 f"{pct(site.coverage['S-dev'])}%", f"{pct(site.coverage['S-ob'])}%"])
    return head, rows


def blocks(site) -> list:
    h = site.headline
    b = [block1(site)]
    meaning = [Unit("p", code_split(p)) for p in meaning_paragraphs(h.row)]
    if h.gb_line:
        meaning.append(Unit("p", code_split(h.gb_line)))
    b.append(Block(2, "meaning", MEANING_HEADING, meaning))
    b.append(Block(3, "next", "Next experiment", [Unit("p", [t(NEXT_EXPERIMENT)])]))
    rule1 = site.featured.get(1)
    b.append(Block(4, "replay", "Replay", [
        Unit("figure", [], {"alt": f"20-s replay of VitalDB case {rule1}: cuff readings, both forecasts with "
                                   "their bands, heart rate, pleth quality and the pump rate if recorded.",
                            "img": "img/replay.gif"}),
        Unit("p", [t(f"VitalDB case {rule1}, chosen by a pre-set rule ({RULE_TAGS[1]} case, "),
                   Seg("link", "PREREG.md", "prereg"), t(" §8). The first and last frames show the complete record.")]),
    ]))
    head, rows = table_rows(site)
    b.append(Block(5, "models", "Error by model", [
        Unit("table", [], {"head": head, "rows": rows}),
        Unit("p", [t(site.hypotension_line)]),
        Unit("figure", [], {"alt": PER_PATIENT_ALT, "img": "img/per-patient.png"}),
        Unit("p", [t(PER_PATIENT_CAPTION)]),
    ]))
    lim = limits(site)
    b.append(Block(6, "limits", "What this does not show", [
        Unit("list", [], {"items": [[t(x)] for x in lim]}),
        Unit("p", [t("The full list is in "), Seg("link", "REPORT.md", "report"), t(".")]),
    ]))
    b.append(Block(7, "note", "Author's note", [Unit("p", [t(AUTHOR_NOTE)])]))
    pre, post = CREDIT_STATEMENT.split("CC BY 4.0", 1)
    b.append(Block(8, "credit", "Credit", [
        Unit("p", [t(pre), Seg("link", "CC BY 4.0", "licence"), t(post)]),
        Unit("p", [t("Data: VitalDB, Lee HC, et al. Sci Data 2022;9:279, "),
                   Seg("link", "doi:10.1038/s41597-022-01411-5", "doi"),
                   t(". No VitalDB data are redistributed here; the code downloads them from the public API "
                     "at run time.")]),
        Unit("p", [t("Design, analysis and code: "), t(site.config.author_name), t(" ("),
                   Seg("link", site.config.contact, "contact"), t(").")]),
    ]))
    return b


# ---------------------------------------------------------------------------
# Facts panel and tiles (page only). Phenylephrine (µg) and ephedrine (mg) stay on separate lines.
# ---------------------------------------------------------------------------

def _amount(v, unit: str) -> str:
    if v is None:
        return "not recorded"
    return f"{v:g} {unit}"


def phenylephrine_text(c) -> str:
    return _amount(c.phenylephrine_ug, "µg")


def ephedrine_text(c) -> str:
    return _amount(c.ephedrine_mg, "mg")


def tile_error_text(c) -> str:
    return f"cuff only {mm(c.rmse['M2'])} → + HR + pleth {mm(c.rmse['M4'])} mmHg"


def facts(c, s) -> list:
    """(label, [values]) rows of the facts panel for a scored case and its series."""
    rm = c.rmse
    rows = [
        ("Readings in window", [f"{c.n_readings}"]),
        ("Median cuff interval", [f"{c.median_interval_s:.0f} s"]),
        ("Scored forecasts", [f"{c.n_forecasts}"]),
        ("RMSE, mmHg", [f"cuff only {mm(rm['M2'])}", f"+ HR {mm(rm['M3'])}", f"+ HR + pleth {mm(rm['M4'])}",
                        f"placebo {mm(rm['M4-P1'])}"]),
        ("Waveform coverage", [f"{pct(c.waveform_coverage)}% of forecasts",
                               f"{pct(s.pleth_usable_share)}% of pleth windows usable"]),
        ("New-onset readings below 80% of baseline", [
            f"{c.new_onset_80}" + ("" if not c.new_onset_80 else
                                   f"; flagged one reading ahead: cuff only {c.flagged_80['M2']}, "
                                   f"+ HR + pleth {c.flagged_80['M4']}")]),
        ("Phenylephrine (chart total)", [phenylephrine_text(c)]),
        ("Ephedrine (chart total)", [ephedrine_text(c)]),
        ("Pump track", ["yes" if c.pump_track else "none"]),
    ]
    return rows
