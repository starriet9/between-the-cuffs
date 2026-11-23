"""Results README (SPEC §8.1): blocks 1-8 in order, rendered from text.blocks()."""
from __future__ import annotations

import re

from . import text as T
from .theme import SYNTHETIC_MARK

_MD_SPECIAL = re.compile(r"([\\*_\[\]<>`])")


def md_escape(s: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", s)


def link_targets(site) -> dict:
    c = site.config
    return {"prereg": "PREREG.md", "report": "REPORT.md", "reading": "REPORT.md#how-to-read-these-numbers",
            "contact": c.contact_url, "frozen": c.frozen_url, "scored": c.scored_url,
            "vitaldb": T.VITALDB_URL, "replay": "#replay", "cases": c.page_url.rstrip("/") + "/#cases",
            "narrated": c.narrated_replay_url, "pdf": c.pdf_url, "licence": T.LICENCE_URL,
            "doi": T.VITALDB_DOI_URL}


def segs_md(segs, links: dict) -> str:
    out = []
    for s in segs:
        if s.kind in ("t", "em"):
            out.append(md_escape(s.text))
        elif s.kind == "b":
            out.append(f"**{md_escape(s.text)}**")
        elif s.kind == "code":
            out.append(f"`{s.text}`")
        elif s.kind == "sep":
            out.append(s.text)
        elif s.kind == "link":
            out.append(f"[{md_escape(s.text)}]({links[s.href]})")
        elif s.kind == "date":
            out.append(f"[{s.text}]({links[s.href]})" if s.href else s.text)
        else:
            raise ValueError(s.kind)
    return "".join(out)


def unit_md(u, links: dict, asset_prefix: str) -> str:
    if u.role == "question":
        return f"**{segs_md(u.segs, links)}**"
    if u.role == "figure":
        return f"![{md_escape(u.extra['alt'])}]({asset_prefix}{u.extra['img']})"
    if u.role == "table":
        head, rows = u.extra["head"], u.extra["rows"]
        lines = ["| " + " | ".join(md_escape(x) for x in head) + " |",
                 "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
        lines += ["| " + " | ".join(md_escape(x) for x in r) + " |" for r in rows]
        return "\n".join(lines)
    if u.role == "list":
        return "\n".join("- " + segs_md(it, links) for it in u.extra["items"])
    return segs_md(u.segs, links)


def render_readme(site, asset_prefix: str = "docs/assets/") -> str:
    links = link_targets(site)
    parts = ["# between-the-cuffs"]
    if site.synthetic:
        parts.append(f"> **{SYNTHETIC_MARK}.** Layout preview built from invented numbers and invented cases.")
    for b in T.blocks(site):
        if b.heading:
            parts.append(f"## {b.heading}")
        for u in b.units:
            parts.append(unit_md(u, links, asset_prefix))
    return "\n\n".join(parts) + "\n"
