"""Static page `index.html` (SPEC §8.3; landing design §2, §4).

Generated at build time with every number already in the HTML: no fetch(), it opens from file://
and works without JavaScript (every case is then a plain figure in a long list; the script only turns
the list into a viewer). Its top repeats README block 1 word for word from the same text model.
"""
from __future__ import annotations

from html import escape
from urllib.parse import quote

from btc.models.headline import pct

from . import text as T
from .featured import RULE_TAGS, rule_of
from .theme import PRELOAD_FONTS, SYNTHETIC_MARK

FAVICON_SVG = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><line x1='8' y1='20' x2='24' y2='12' "
               "stroke='#1B62A8' stroke-width='2.6' stroke-dasharray='1.5 3.2' stroke-linecap='round'/>"
               "<circle cx='7' cy='21' r='4.2' fill='#14181F'/><circle cx='25' cy='11' r='4.2' fill='#14181F'/></svg>")
STRATUM_LABEL = {"pump": "pump", "bolus total only": "bolus total only", "neither recorded": "neither recorded"}
TRANSPARENT_GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


def e(s) -> str:
    return escape(str(s), quote=True)


def link_targets(site) -> dict:
    c = site.config
    return {"prereg": c.blob("PREREG.md"), "report": c.blob("REPORT.md"),
            "reading": c.blob("REPORT.md") + "#how-to-read-these-numbers", "contact": c.contact_url,
            "frozen": c.frozen_url,
            "scored": c.scored_url, "vitaldb": T.VITALDB_URL, "replay": "#replay", "cases": "#cases",
            "narrated": c.narrated_replay_url, "pdf": c.pdf_url, "licence": T.LICENCE_URL, "doi": T.VITALDB_DOI_URL}


def _a(href: str, inner: str) -> str:
    rel = ' rel="noopener"' if href.startswith("http") else ""
    return f'<a href="{e(href)}"{rel}>{inner}</a>'


def segs_html(segs, links: dict) -> str:
    out = []
    for s in segs:
        if s.kind == "t":
            out.append(e(s.text))
        elif s.kind == "em":
            out.append(f'<strong class="em">{e(s.text)}</strong>')
        elif s.kind == "b":
            out.append(f"<strong>{e(s.text)}</strong>")
        elif s.kind == "code":
            out.append(f"<code>{e(s.text)}</code>")
        elif s.kind == "sep":
            out.append(f'<span class="sep">{e(s.text)}</span>')
        elif s.kind == "link":
            out.append(_a(links[s.href], e(s.text)))
        elif s.kind == "date":
            tm = f'<time datetime="{e(s.text)}">{e(s.text)}</time>'
            out.append(_a(links[s.href], tm) if s.href else tm)
        else:
            raise ValueError(s.kind)
    return "".join(out)


def _img(src, dims, alt, cls="", lazy=False):
    w, h = dims
    attrs = [f'src="{e(src)}"', f'width="{w}"', f'height="{h}"', f'alt="{e(alt)}"']
    if cls:
        attrs.append(f'class="{cls}"')
    if lazy:
        attrs.append('loading="lazy"')
    attrs.append('decoding="async"')
    return "<img " + " ".join(attrs) + ">"


# ---------------------------------------------------------------------------
# block 1
# ---------------------------------------------------------------------------

def block1_html(b, site, links, assets) -> str:
    u = {x.role: x for x in b.units}
    sentences = [x for x in b.units if x.role == "sentence"]
    reg = u["registration"]
    nodes, chips = reg.extra["nodes"], reg.extra["chips"]
    parts = []
    for i, nd in enumerate(nodes):
        if i:
            parts.append('<span class="sep"> · </span>')
        dev = " node-dev" if i == 3 else ""
        parts.append(f'<span class="node{dev}"><span class="node-text">{segs_html(nd, links)}</span></span>')
    chip_html = "".join(('<span class="sep sep-gap"> · </span>' if j else "")
                        + f'<span class="chip">{segs_html(ch, links)}</span>' for j, ch in enumerate(chips))
    parts.append(f'<span class="sep sep-gap"> · </span><span class="chips">{chip_html}</span>')
    fig = u["figure"]
    wide = assets["img/summary.png"]
    left, right = assets["img/summary-left.png"], assets["img/summary-right.png"]
    picture = (
        '<a class="summary-link" href="#cases" aria-label="Summary figure; go to all cesarean cases">'
        f'<picture><source media="(max-width: 720px)" srcset="assets/img/summary-left.png" width="{left[0]}" '
        f'height="{left[1]}">{_img("assets/img/summary.png", wide, fig.extra["alt"], "summary-main")}</picture>'
        f'<picture class="summary-second"><source media="(min-width: 721px)" srcset="{TRANSPARENT_GIF}" width="1" '
        f'height="1">{_img("assets/img/summary-right.png", right, "", "summary-right")}</picture></a>')
    links_line = segs_html(u["links"].segs, links)
    links_line += (' <button type="button" class="copy" data-copy="' + e(site.config.reproduce_command)
                   + '" data-noparity hidden>Copy</button>')
    return f"""<section id="answer" class="block block1" data-block="1" aria-labelledby="question">
<h1 id="question">{segs_html(u["question"].segs, links)}</h1>
<p class="answer-label">{e(u["answer"].extra["label"])}</p>
<p class="answer">{segs_html(u["answer"].extra["rest"], links)}</p>
<p class="cesarean">{segs_html(u["cesarean"].segs, links)}</p>
<p class="registration">{"".join(parts)}</p>
<figure class="summary">{picture}</figure>
<p class="links">{links_line}</p>
<div class="result-full">
{"".join(f"<p>{segs_html(x.segs, links)}</p>" for x in sentences)}
</div>
</section>"""


# ---------------------------------------------------------------------------
# blocks 2, 3, 5-8
# ---------------------------------------------------------------------------

def unit_html(x, links) -> str:
    if x.role == "p":
        return f"<p>{segs_html(x.segs, links)}</p>"
    if x.role == "list":
        return "<ul>" + "".join(f"<li>{segs_html(it, links)}</li>" for it in x.extra["items"]) + "</ul>"
    if x.role == "table":
        head, rows = x.extra["head"], x.extra["rows"]
        th = "".join(f'<th scope="col"{" class=num" if j >= 1 else ""}>{e(h)}</th>' for j, h in enumerate(head))
        body = []
        for r in rows:
            code, _, desc = r[0].partition(" · ")
            label = f'<span class="m">{e(code)}</span>' + (f'<span class="sep-dot"> · </span><span class="desc">{e(desc)}</span>' if desc else "")
            cells = [f'<th scope="row">{label}</th>']
            cells += [f'<td class="num">{e(v)}</td>' for v in r[1:]]
            body.append("<tr>" + "".join(cells) + "</tr>")
        return (f'<div class="table-wrap" role="region" aria-label="Error by model" tabindex="0"><table>'
                f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    raise ValueError(x.role)


def plain_block_html(b, links, assets, cls="") -> str:
    inner = []
    for x in b.units:
        if x.role == "figure":
            src = x.extra["img"]
            inner.append(f'<figure class="wide">{_img("assets/" + src, assets[src], x.extra["alt"], lazy=True)}</figure>')
        else:
            inner.append(unit_html(x, links))
    tag = "footer" if b.key == "credit" else "section"
    return (f'<{tag} id="{b.key}" class="block {cls}" data-block="{b.num}" aria-labelledby="{b.key}-h">'
            f'<h2 id="{b.key}-h">{e(b.heading)}</h2>{"".join(inner)}')


# ---------------------------------------------------------------------------
# block 4 on the page: all cesarean cases (contact sheet + viewer, replay inside)
# ---------------------------------------------------------------------------

def _rule_tag(r: int) -> str:
    return f'<span class="tag rule" title="Featured by pre-set rule {r} (PREREG §8)">rule {r}: {RULE_TAGS[r]}</span>'


def cases_html(site, assets) -> str:
    rules = rule_of(site.featured)
    scored = sorted(site.scored_cases, key=lambda c: (c.rmse["M4"], c.caseid))
    others = sorted((c for c in site.cases if not c.scored), key=lambda c: c.caseid)
    n = site.headline.n
    counts = {k: sum(1 for c in scored if c.stratum == k) for k in STRATUM_LABEL}
    tiles = []
    for i, c in enumerate(scored):
        tags = f'<span class="tag">{e(STRATUM_LABEL[c.stratum])}</span>'
        if c.caseid in rules:
            tags += _rule_tag(rules[c.caseid])
        err = T.tile_error_text(c)
        tiles.append(
            f'<li class="tile" data-case="{c.caseid}" data-err="{c.rmse["M4"]:.6f}" data-stratum="{e(c.stratum)}">'
            f'<a class="tile-btn" href="#case-{c.caseid}">'
            f'{_img(f"assets/cases/tile-{c.caseid}.png", assets[f"cases/tile-{c.caseid}.png"], f"Cuff readings of VitalDB case {c.caseid}", "tile-img")}'
            f'<span class="tile-id">case {c.caseid}</span><span class="tile-err">{e(err)}</span>'
            f'<span class="tags">{tags}</span></a></li>')
    for c in others:
        tiles.append(f'<li class="tile tile-na" data-case="{c.caseid}" data-stratum="none" data-err="Infinity">'
                     f'<span class="tile-id">case {c.caseid}</span> · {e(c.status)}: {e(c.reason)}</li>')
    chips = [f'<button type="button" class="chip-btn" data-filter="all" aria-pressed="true">all {len(scored)}</button>']
    chips += [f'<button type="button" class="chip-btn" data-filter="{e(k)}" aria-pressed="false">{e(v)} {counts[k]}</button>'
              for k, v in STRATUM_LABEL.items()]
    controls = ('<div class="sheet-controls" data-js-only hidden>'
                '<div class="ctl" role="group" aria-label="Sort"><span class="ctl-label">Sort</span>'
                '<button type="button" class="chip-btn" data-sort="error" aria-pressed="true">error</button>'
                '<button type="button" class="chip-btn" data-sort="id" aria-pressed="false">case ID</button></div>'
                f'<div class="ctl" role="group" aria-label="Filter by vasopressor record">{"".join(chips)}</div></div>')
    arts = [case_article(site, c, assets, rules) for c in scored]
    viewer_nav = ('<div class="viewer-nav" data-js-only hidden>'
                  '<button type="button" class="nav-btn" data-nav="-1" aria-label="Previous case">← Previous</button>'
                  '<span class="viewer-pos" aria-live="polite"></span>'
                  '<button type="button" class="nav-btn" data-nav="1" aria-label="Next case">Next →</button></div>')
    return (f'<section id="cases" class="block" data-block="4" aria-labelledby="cases-h">'
            f'<h2 id="cases-h">All {n} scored cesarean cases</h2>'
            f'<p class="caption">{e(T.SHEET_CAPTION)}</p>{controls}'
            f'<ol class="sheet" id="sheet">{"".join(tiles)}</ol>'
            f'<div class="viewer" id="viewer" tabindex="-1" aria-label="Case viewer">{viewer_nav}'
            f'<div class="case-list">{"".join(arts)}</div></div></section>')


def case_article(site, c, assets, rules) -> str:
    s = site.series[c.caseid]
    r = rules.get(c.caseid)
    rp = assets.get("replays", {}).get(c.caseid)
    head = f'<h3>VitalDB case {c.caseid}</h3>' + (_rule_tag(r) if r else "")
    mode = ""
    replay = ""
    if rp:
        mode = ('<div class="mode" role="group" aria-label="Still or replay" data-js-only hidden>'
                '<button type="button" class="chip-btn" data-mode="still" aria-pressed="true">Still</button>'
                '<button type="button" class="chip-btn" data-mode="replay" aria-pressed="false">Replay</button></div>')
        rid = "replay" if r == 1 else f"replay-{c.caseid}"
        w, h = rp["dims"]
        replay = (f'<div class="case-replay" id="{rid}"><video controls playsinline muted preload="none" '
                  f'poster="{e(rp["poster"])}" width="{w}" height="{h}">'
                  f'<source src="{e(rp["mp4"])}" type="video/mp4"><source src="{e(rp["webm"])}" type="video/webm">'
                  f'</video><p class="replay-cap">20-s replay of case {c.caseid}. It opens and closes on the complete '
                  f'record; nothing moves until you press play.</p></div>')
    alt = (f"VitalDB case {c.caseid}: cuff readings with both forecasts and their 90% bands, the absolute error at "
           f"each reading, heart rate, pleth quality" + (", and the pump rate in mL/h" if s.has_pump else "") + ".")
    still = _img(f"assets/cases/case-{c.caseid}.png", assets[f"cases/case-{c.caseid}.png"], alt, "case-still", lazy=True)
    rows = "".join(f'<div class="fact"><dt>{e(k)}</dt>' + "".join(f"<dd>{e(v)}</dd>" for v in vals) + "</div>"
                   for k, vals in T.facts(c, s))
    return (f'<article class="case" id="case-{c.caseid}" data-case="{c.caseid}" aria-label="VitalDB case {c.caseid}">'
            f'<header class="case-head">{head}{mode}</header><div class="case-body">'
            f'<figure class="case-fig">{still}{replay}</figure>'
            f'<div class="case-side"><dl class="facts">{rows}</dl><p class="facts-note">{e(T.VASO_NOTE)}</p></div>'
            f'</div></article>')


# ---------------------------------------------------------------------------
# check-it-yourself, header, footer, page
# ---------------------------------------------------------------------------

def check_html(site) -> str:
    c = site.config
    dev = ("Deviations: none, see REPORT.md" if c.deviations == 0 else
           f"Deviations: {c.deviations}, listed in REPORT.md")
    items = [(c.registration_url, f"Registration commit ({c.registration_commit}, {c.registered_date})"),
             (c.frozen_url, f"Frozen-model tag ({c.frozen_tag}, {c.frozen_date})"),
             (c.blob("REPORT.md"), dev), (c.tree("tests"), "Tests"), (c.blob(c.checksums_path), "Data checksums")]
    lis = "".join(f"<li>{_a(h, e(t_))}</li>" for h, t_ in items)
    cmd = e(c.reproduce_command)
    return (f'<section id="check" class="block" data-noparity aria-labelledby="check-h"><h2 id="check-h">Check it '
            f'yourself</h2><ul class="check-list">{lis}</ul><p class="cmd"><code>{cmd}</code> '
            f'<button type="button" class="copy" data-copy="{cmd}" hidden>Copy</button></p></section>')


def render_page(site, assets: dict, css_href: str, js_href: str) -> str:
    links = link_targets(site)
    bl = T.blocks(site)
    h, c = site.headline, site.config
    desc = f"{h.label} {h.answer_rest}"
    og = c.page_url.rstrip("/") + "/assets/img/og.png"
    pre = "".join(f'<link rel="preload" href="assets/fonts/{f}" as="font" type="font/woff" crossorigin>'
                  for f in PRELOAD_FONTS)
    banner = ""
    if site.synthetic:
        banner = (f'<div class="synthetic-banner" role="note"><strong>{SYNTHETIC_MARK}.</strong> Layout preview '
                  f'built from invented numbers and invented cases{f" (variant {e(site.variant)})" if site.variant else ""}.'
                  '</div>')
    mark = FAVICON_SVG.replace("<svg ", '<svg class="mark" aria-hidden="true" width="22" height="22" ', 1)
    body = [block1_html(bl[0], site, links, assets),
            plain_block_html(bl[1], links, assets) + "</section>",
            plain_block_html(bl[2], links, assets, "card") + "</section>",
            cases_html(site, assets),
            plain_block_html(bl[4], links, assets) + "</section>",
            plain_block_html(bl[5], links, assets) + "</section>",
            plain_block_html(bl[6], links, assets) + "</section>",
            check_html(site)]
    footer = (plain_block_html(bl[7], links, assets, "site-footer")
              + f'<p class="updated" data-noparity>Last updated {e(c.last_updated)}</p></footer>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>between-the-cuffs</title>
<meta name="description" content="{e(desc)}">
<meta property="og:type" content="website">
<meta property="og:title" content="{e(h.question)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:image" content="{e(og)}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,{quote(FAVICON_SVG)}">
{pre}
<link rel="stylesheet" href="{e(css_href)}">
</head>
<body>
<a class="skip" href="#main">Skip to the answer</a>
{banner}
<header class="site-header"><div class="wrap head-row"><a class="brand" href="#answer">{mark}<span>between-the-cuffs</span></a>
<nav class="head-nav" aria-label="Author and code"><span>{e(c.author_name)}</span><span class="dot" aria-hidden="true"> · </span>{_a(c.repo_url, "GitHub")}</nav></div></header>
<main id="main" class="wrap">
{chr(10).join(body)}
</main>
<div class="wrap">{footer}</div>
<script src="{e(js_href)}" defer></script>
</body>
</html>
"""
