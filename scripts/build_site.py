#!/usr/bin/env python3
"""Build the results page and README (SPEC §8.1-§8.3).

  --synthetic   preview from synthetic fixtures (all headline variants, or --variant NAME ...) into
                build/preview/<variant>/ (index.html, assets/, README.md), plus build/preview/index.html.
                Every page and image is marked "SYNTHETIC — NOT A RESULT".
  --publish     build docs/ and README.md from the real results (--results, --cases-dir, --config).
                Refuses synthetic inputs, inconsistent case files and any leftover placeholder
                ({{, [X], NaN, TODO, ...); docs/ is only replaced when every check passes.
"""
from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc.paths import DERIVED, ROOT  # noqa: E402
from btc.site import fixtures as FX  # noqa: E402
from btc.site.build import PublishRefused, build_site  # noqa: E402
from btc.site.config import DEFAULT_CONFIG, load_config  # noqa: E402
from btc.site.results import load_site_data  # noqa: E402


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def preview(variants, out: Path, jobs: int, replays) -> int:
    out.mkdir(parents=True, exist_ok=True)
    cache = out.parent / ".site-cache"
    cases, series = FX.make_cases()
    rows = []
    for v in variants:
        t0 = time.time()
        paths = FX.write_results_tree(out / "_fixtures" / v, v, cases, series)
        site = load_site_data(paths["results"], paths["cases"], FX.config_for(v), variant=v)
        res = build_site(site, out / v, publish=False, cache_dir=cache, replay_rules=replays, jobs=jobs)
        w = res.weight
        print(f"{v:18s} row {site.headline.row}  {site.headline.label:42s} desktop {w['desktop'] / 1e3:6.0f} kB  "
              f"phone {w['phone'] / 1e3:6.0f} kB  {time.time() - t0:5.1f} s")
        real = [p for p in res.problems if "placeholder" not in p]
        for p in res.problems:
            print("   note:" if p not in real else "   PROBLEM:", p)
        rows.append((v, site.headline.row, site.headline.label))
    items = "".join(f'<li><a href="{html.escape(v)}/index.html">{html.escape(v)}</a> — row {r}: {html.escape(lab)}'
                    f' · <a href="{html.escape(v)}/README.md">README.md</a></li>' for v, r, lab in rows)
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>between-the-cuffs previews</title><body style="font:16px/1.5 sans-serif;margin:24px">'
        '<p style="background:#B3261E;color:#fff;padding:8px">SYNTHETIC — NOT A RESULT. Layout previews.</p>'
        f'<ul>{items}</ul></body>')
    print(f"previews: {rel(out / 'index.html')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--publish", action="store_true")
    ap.add_argument("--variant", action="append", choices=sorted(FX.VARIANTS), help="with --synthetic")
    ap.add_argument("--out", type=Path, default=None, help="preview folder (default build/preview)")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--cases-dir", type=Path, default=DERIVED / "site")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--replays", choices=["rule1", "all"], default="rule1",
                    help="featured replays as video: rule 1 only, or rules 1-3")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    replays = (1,) if a.replays == "rule1" else (1, 2, 3)
    if a.synthetic:
        return preview(a.variant or list(FX.VARIANTS), a.out or ROOT / "build" / "preview", a.jobs, replays)
    if a.out or a.variant:
        ap.error("--out and --variant are for --synthetic only; --publish always writes docs/")
    try:
        site = load_site_data(a.results, a.cases_dir, load_config(a.config))
        res = build_site(site, ROOT / "docs", publish=True, readme_path=a.readme,
                         cache_dir=ROOT / "build" / ".site-cache", replay_rules=replays, jobs=a.jobs)
    except PublishRefused as ex:
        print(ex, file=sys.stderr)
        return 2
    except FileNotFoundError as ex:
        print(f"refused: missing input {rel(Path(ex.filename))}", file=sys.stderr)
        return 2
    w = res.weight
    print(f"published: docs/index.html and {rel(res.readme_path)}; page weight before case images "
          f"{w['desktop'] / 1e3:.0f} kB (desktop), {w['phone'] / 1e3:.0f} kB (phone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
