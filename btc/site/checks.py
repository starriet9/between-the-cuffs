"""Checks run on every build. If any fails, the publish build stops and leaves docs/ and the README as they were.

- placeholders: `{{`, `[X]`-style tokens, NaN/None/inf leaks, TODO/TBD, NNNN, lorem;
- forbidden strings: absolute paths and file URLs, plus the git-ignored private patterns;
- phenylephrine (µg) and ephedrine (mg) never in one text run, and no summed vasopressor;
- parity: the page's top block equals README block 1 word for word (and blocks 2, 3, 5-8);
- no external request from the page (only links), no fetch(); page weight before case images.
"""
from __future__ import annotations

import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

PLACEHOLDERS = [
    (r"\{\{", "{{ placeholder"),
    (r"\[(?:X|N|n|A|B|Z|Y|G|g|K|C|F|P|lo|hi|Xob|date|url|path|command|Label|label|Full Name)\]", "[X]-style placeholder"),
    (r"\bNaN\b|\bnan\b", "NaN"),
    (r"\bNone\b", "None"),
    (r"(?<![\w.])-?inf\b", "inf"),
    (r"TODO|\bTBD\b|\bFIXME\b", "TODO"),
    (r"NNNN", "NNNN"),
    (r"(?i)lorem ipsum", "lorem"),
]
FORBIDDEN = [
    (r"/(?:home|Users|root)/|(?<![\w.])/(?:tmp|mnt)/|\b[A-Za-z]:\\+[A-Za-z]", "absolute path"),
    (r"file://", "file URL"),
]
# Context-specific words (names and folders) live in a git-ignored file, so that the public code does
# not carry them; every build and the tests load it when it is present.
PRIVATE_PATTERNS = Path(__file__).resolve().parents[2] / "private" / "site_leak_patterns.txt"


def private_patterns(path: Path | None = None) -> list:
    p = Path(path) if path else PRIVATE_PATTERNS
    if not p.exists():
        return []
    return [(ln.strip(), "private leak pattern") for ln in p.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


VASO_SUM = re.compile(r"(?i)phenylephrine\s*(\+|and|plus)\s*ephedrine\s*(total|sum|dose)|vasopressor (score|total|sum)")
MG = re.compile(r"(?<![µu])\bmg\b")


def scan_text(text: str, name: str, patterns) -> list:
    out = []
    for pat, what in patterns:
        for m in re.finditer(pat, text):
            ctx = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
            out.append(f"{name}: {what}: …{ctx}…")
    return out


def user_email() -> str | None:
    try:
        r = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------

BLOCK_TAGS = {"p", "h1", "h2", "h3", "li", "figure", "div", "tr", "td", "th", "dt", "dd", "section", "ul", "ol",
              "table", "br", "header", "footer", "article", "picture", "dl", "figcaption"}
VOID = {"img", "br", "source", "meta", "link", "input", "hr"}


class _Text(HTMLParser):
    """Visible text per data-block, skipping [data-noparity] subtrees; img alt counts as text.
    Also collects every text run (for the unit check) and every requested resource URL."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []            # (tag, block, skip)
        self.blocks = {}
        self.runs = []
        self.requests = []
        self.scripts_inline = []
        self._in_script = False

    def _ctx(self):
        return (self.stack[-1][1], self.stack[-1][2]) if self.stack else (None, False)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        blk, skip = self._ctx()
        blk = a.get("data-block", blk)
        skip = skip or ("data-noparity" in a)
        if tag in ("img", "source", "script", "video", "audio", "iframe", "embed"):
            for k in ("src", "srcset", "poster"):
                if a.get(k):
                    self.requests.append((tag, a[k]))
        if tag == "link" and a.get("rel") not in ("canonical",):
            if a.get("href"):
                self.requests.append(("link", a["href"]))
        if tag == "script" and not a.get("src"):
            self._in_script = True
        if blk is not None and not skip:
            if tag in BLOCK_TAGS:
                self.blocks.setdefault(blk, []).append(" ")
            if tag == "img" and a.get("alt"):
                self.blocks.setdefault(blk, []).append(" " + a["alt"] + " ")
        if tag not in VOID:
            self.stack.append((tag, blk, skip))

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
        while self.stack:
            t, blk, skip = self.stack.pop()
            if blk is not None and not skip and t in BLOCK_TAGS:
                self.blocks.setdefault(blk, []).append(" ")
            if t == tag:
                break

    def handle_data(self, data):
        if self._in_script:
            self.scripts_inline.append(data)
            return
        if data.strip():
            self.runs.append(data)
        blk, skip = self._ctx()
        if blk is not None and not skip:
            self.blocks.setdefault(blk, []).append(data)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_html(html: str) -> _Text:
    p = _Text()
    p.feed(html)
    p.close()
    return p


def html_blocks(html: str) -> dict:
    return {k: norm("".join(v)) for k, v in parse_html(html).blocks.items()}


# ---------------------------------------------------------------------------
# Markdown text extraction (the README renderer's subset of Markdown)
# ---------------------------------------------------------------------------

def md_plain(md: str) -> str:
    lines = []
    for ln in md.splitlines():
        if re.match(r"^\|[-:| ]+\|$", ln.strip()):
            continue
        if ln.startswith("|"):
            ln = " ".join(c.strip() for c in ln.strip().strip("|").split("|"))
        ln = re.sub(r"^- ", "", ln)
        ln = re.sub(r"^#+ ", "", ln)
        lines.append(ln)
    s = "\n".join(lines)
    s = re.sub(r"!\[((?:\\.|[^\]])*)\]\([^)]*\)", r" \1 ", s)
    s = re.sub(r"\[((?:\\.|[^\]])*)\]\([^)]*\)", r"\1", s)
    s = s.replace("**", "")
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\\([\\*_\[\]<>`])", r"\1", s)
    return norm(s)


def md_blocks(md: str, headings: dict) -> dict:
    """{block number (str): plain text}; block 1 is everything between the title (and the synthetic note)
    and the first `## ` heading. `headings` maps heading text -> block number."""
    parts = re.split(r"(?m)^## ", md)
    head = parts[0].splitlines()
    body1 = [ln for ln in head if not ln.startswith("# ") and not ln.startswith("> ")]
    out = {"1": md_plain("\n".join(body1))}
    for p in parts[1:]:
        p = re.split(r"(?m)^---$", p)[0]          # the author line after the credit is not a block
        title = p.splitlines()[0].strip()
        num = headings.get(title)
        if num is not None:
            out[str(num)] = md_plain("## " + p)
    return out


def parity_problems(readme: str, html: str, blocks) -> list:
    """README block text vs page block text, for every block except 4 (replay vs case browser)."""
    heads = {b.heading: b.num for b in blocks if b.heading}
    md = md_blocks(readme, heads)
    pg = html_blocks(html)
    out = []
    for b in blocks:
        if b.num == 4:
            continue
        k = str(b.num)
        if md.get(k) != pg.get(k):
            a, c = md.get(k) or "", pg.get(k) or ""
            i = next((j for j in range(min(len(a), len(c))) if a[j] != c[j]), min(len(a), len(c)))
            out.append(f"block {k} differs between README and page at char {i}: README …{a[i - 40:i + 40]}… "
                       f"page …{c[i - 40:i + 40]}…")
    return out


# ---------------------------------------------------------------------------
# whole-output checks
# ---------------------------------------------------------------------------

def unit_problems(html: str, readme: str) -> list:
    out = []
    runs = parse_html(html).runs + readme.splitlines()
    for r in runs:
        if "µg" in r and MG.search(r):
            out.append(f"µg and mg in one text run: {r.strip()[:120]}")
        if VASO_SUM.search(r):
            out.append(f"summed vasopressor: {r.strip()[:120]}")
    return out


def request_problems(html: str, js: str) -> list:
    p = parse_html(html)
    out = []
    for tag, url in p.requests:
        u = url.strip()
        if re.match(r"^(https?:)?//", u):
            out.append(f"external request from <{tag}>: {u}")
    if re.search(r"\bfetch\s*\(|XMLHttpRequest|import\s*\(", js + "".join(p.scripts_inline)):
        out.append("script makes a network request")
    return out


def content_problems(texts: dict, email: str | None = None, extra_patterns: list | None = None) -> list:
    """texts: {name: text} for HTML, README, CSS and JS outputs. The private patterns are added unless
    `extra_patterns` is given."""
    out = []
    forbidden = FORBIDDEN + (private_patterns() if extra_patterns is None else extra_patterns)
    for name, t in texts.items():
        out += scan_text(t, name, PLACEHOLDERS)
        out += scan_text(t, name, forbidden)
        if email and email in t:
            out.append(f"{name}: contains the git user e-mail address")
    return out


def page_weight(out_dir: Path, html: str) -> dict:
    """Bytes loaded before any case image: HTML, CSS, JS, fonts, summary figure (desktop or phone pair),
    tiles and the per-patient figure. Case images, posters, videos and og.png are excluded."""
    out_dir = Path(out_dir)
    p = parse_html(html)
    req = [u for _, u in p.requests if not u.startswith("data:")]
    size = lambda rel: (out_dir / rel).stat().st_size if (out_dir / rel).exists() else 0
    base = size("index.html")
    css = [u for u in req if u.endswith(".css")]
    fonts = list((out_dir / "assets" / "fonts").glob("*.woff"))
    common = base + sum(size(u) for u in css) + sum(size(u) for u in req if u.endswith(".js"))
    common += sum(f.stat().st_size for f in fonts)
    common += sum(size(u) for u in req if "/tile-" in u) + size("assets/img/per-patient.png")
    desk = common + size("assets/img/summary.png")
    phone = common + size("assets/img/summary-left.png") + size("assets/img/summary-right.png")
    return {"desktop": desk, "phone": phone}
