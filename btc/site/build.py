"""Build the static page and the results README from one SiteData.

preview (`build_site(..., publish=False)`): any input, synthetic included; writes `<out>/index.html`,
    `<out>/assets/...` and `<out>/README.md` (images referenced as `assets/...`).
publish (`publish=True`): refuses synthetic inputs, inconsistent case files, missing cerclage flags and,
    after rendering into a staging folder, any placeholder, forbidden string, unit mix, parity failure,
    external request or a page heavier than 1 MB before case images. Only then are `docs/index.html`,
    `docs/assets/` and `docs/.nojekyll` replaced and the README written (images referenced as `docs/assets/...`).
    `allow_synthetic=True` lifts only the synthetic refusal, for end-to-end tests that publish a fully
    synthetic chain into a temporary folder; every other check still applies and the watermark stays.
Expensive per-case images and replays are cached by content hash.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import checks, readme as R, text as T
from .featured import rule_of
from .page import render_page
from .results import consistency_problems
from .theme import FONT_DIR, FONT_LICENCES, STATIC_DIR, WEB_FONTS

MAX_WEIGHT = 1_000_000
HERE = Path(__file__).resolve().parent


class PublishRefused(RuntimeError):
    pass


@dataclass
class BuildResult:
    out_dir: Path
    readme_path: Path | None
    weight: dict
    problems: list = field(default_factory=list)
    files: list = field(default_factory=list)


def _code_hash() -> str:
    h = hashlib.sha256()
    for f in ("figures.py", "theme.py", "series.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


def _key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


def _cached(cache: Path, key: str, make, dest: Path) -> dict:
    """Run make(dir) -> {file: dims} once per key; copy the files to dest."""
    d = cache / key
    meta = d / "_dims.json"
    if not meta.exists():
        tmp = Path(tempfile.mkdtemp(dir=cache))
        dims = make(tmp)
        (tmp / "_dims.json").write_text(json.dumps(dims))
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)
    dims = json.loads(meta.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for f in dims:
        shutil.copy2(d / f, dest / f)
    return {k: tuple(v) for k, v in dims.items()}


def _case_job(args):
    kind, sdict, synthetic, extra, out = args
    from . import figures as Fg
    from .series import CaseSeries
    s = CaseSeries.from_dict(sdict)
    out = Path(out)
    if kind == "case":
        Fg.case_image(s, out / f"case-{s.caseid}.png", synthetic)
        return {f"case-{s.caseid}.png": (1040, 620)}
    if kind == "tile":
        Fg.tile_image(s, out / f"tile-{s.caseid}.png", synthetic, tuple(extra))
        return {f"tile-{s.caseid}.png": (200, 76)}
    if kind == "replay":
        return Fg.replay_outputs(s, out, synthetic, out, gif=extra)
    raise ValueError(kind)


def render_assets(site, adir: Path, cache: Path, replay_rules=(1,), jobs: int = 8) -> dict:
    """All images into `adir` (= <out>/assets); returns {relative path: (w, h)} plus 'replays'."""
    from . import figures as Fg
    cache.mkdir(parents=True, exist_ok=True)
    code = _code_hash()
    syn = bool(site.synthetic)
    assets = {}
    img = adir / "img"
    img.mkdir(parents=True, exist_ok=True)
    dims = Fg.summary_figures(site, img, syn)
    assets.update({f"img/{k}": v for k, v in dims.items()})
    Fg.per_patient_figure(site, img / "per-patient.png", syn)
    assets["img/per-patient.png"] = (1040, 400)
    rng = Fg.tile_map_range(site.series)
    tasks = []
    for cid, s in site.series.items():
        sd = s.to_dict()
        tasks.append(("case", sd, syn, None, adir / "cases"))
        tasks.append(("tile", sd, syn, list(rng), adir / "cases"))
    rules = rule_of(site.featured)
    for cid, r in rules.items():
        if r in replay_rules:
            tasks.append(("replay", site.series[cid].to_dict(), syn, r == 1, adir / "replay"))

    def make_for(t):
        kind, sd, s_, extra, dest = t
        key = _key(kind, sd, s_, extra, code)
        return key, dest

    todo, hits = [], []
    for t in tasks:
        key, dest = make_for(t)
        if (cache / key / "_dims.json").exists():
            hits.append((t, key, dest))
        else:
            todo.append((t, key, dest))
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, min(jobs, len(todo))), mp_context=mp.get_context("spawn")) as ex:
            futs = []
            for t, key, dest in todo:
                tmp = Path(tempfile.mkdtemp(dir=cache))
                futs.append((ex.submit(_case_job, (t[0], t[1], t[2], t[3], str(tmp))), tmp, key))
            for fu, tmp, key in futs:
                d = fu.result()
                (tmp / "_dims.json").write_text(json.dumps(d))
                final = cache / key
                if final.exists():
                    shutil.rmtree(final)
                tmp.rename(final)
    for t, key, dest in todo + hits:
        d = _cached(cache, key, None, dest)
        sub = dest.name
        for f, v in d.items():
            assets[f"{sub}/{f}"] = v
    # README GIF lives with the other README images
    if (adir / "replay" / "replay.gif").exists():
        shutil.move(adir / "replay" / "replay.gif", img / "replay.gif")
        assets["img/replay.gif"] = assets.pop("replay/replay.gif")
    assets["replays"] = {}
    for cid, r in rules.items():
        if r in replay_rules:
            assets["replays"][cid] = {"mp4": f"assets/replay/replay-{cid}.mp4",
                                      "webm": f"assets/replay/replay-{cid}.webm",
                                      "poster": f"assets/replay/replay-{cid}.png",
                                      "dims": assets[f"replay/replay-{cid}.mp4"]}
    return assets


def _copy_static(adir: Path) -> tuple:
    (adir / "fonts").mkdir(parents=True, exist_ok=True)
    for f in list(WEB_FONTS) + list(FONT_LICENCES):
        shutil.copy2(FONT_DIR / f, adir / "fonts" / f)
    out = []
    for name in ("site.css", "site.js"):
        b = (STATIC_DIR / name).read_bytes()
        stem, ext = name.rsplit(".", 1)
        hashed = f"{stem}.{hashlib.sha256(b).hexdigest()[:10]}.{ext}"
        (adir / hashed).write_bytes(b)
        out.append(f"assets/{hashed}")
    return tuple(out)


def run_checks(site, out_dir: Path, html: str, readme: str) -> tuple:
    adir = out_dir / "assets"
    css = "".join(p.read_text() for p in adir.glob("site.*.css"))
    js = "".join(p.read_text() for p in adir.glob("site.*.js"))
    texts = {"index.html": html, "README.md": readme, "site.css": css, "site.js": js}
    problems = checks.content_problems(texts, checks.user_email())
    problems += checks.unit_problems(html, readme)
    problems += checks.parity_problems(readme, html, T.blocks(site))
    problems += checks.request_problems(html, js)
    if T.CREDIT_STATEMENT not in checks.norm(checks.md_plain(readme)):
        problems.append("README lacks the SPEC §2.1 credit statement")
    if T.CREDIT_STATEMENT not in checks.html_blocks(html).get("8", ""):
        problems.append("page lacks the SPEC §2.1 credit statement")
    w = checks.page_weight(out_dir, html)
    if max(w.values()) >= MAX_WEIGHT:
        problems.append(f"page weight before case images {max(w.values())} bytes >= {MAX_WEIGHT}")
    return problems, w


def build_site(site, out_dir: Path, *, publish: bool = False, readme_path: Path | None = None,
               cache_dir: Path | None = None, replay_rules=(1,), jobs: int = 8,
               allow_synthetic: bool = False) -> BuildResult:
    out_dir = Path(out_dir)
    cache = Path(cache_dir) if cache_dir else out_dir.parent / ".site-cache"
    if publish:
        if site.synthetic and not allow_synthetic:
            raise PublishRefused("refused: the inputs are synthetic (fixtures); publish needs the real results")
        cp = consistency_problems(site)
        if site.config.deviations is None:
            cp.append("deviations count not set in the site settings (see REPORT.md)")
        if site.cerclage is None:
            cp.append("no cerclage flags: the per-patient figure marks the cerclage cases (SPEC §6.7); "
                      "give the S-dev case table (data/derived/sdev_cases.parquet)")
        if cp:
            raise PublishRefused("refused: " + "; ".join(cp))
    cache.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=cache, prefix="stage-")) if publish else out_dir
    if not publish and stage.exists():
        for p in ("index.html", "README.md", ".nojekyll"):
            (stage / p).unlink(missing_ok=True)
        shutil.rmtree(stage / "assets", ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    adir = stage / "assets"
    css_href, js_href = _copy_static(adir)
    assets = render_assets(site, adir, cache / "figures", replay_rules, jobs)
    html = render_page(site, assets, css_href, js_href)
    (stage / "index.html").write_text(html)
    (stage / ".nojekyll").write_text("")
    readme_stage = R.render_readme(site, asset_prefix="assets/")
    readme_final = R.render_readme(site, asset_prefix="docs/assets/") if publish else readme_stage
    problems, weight = run_checks(site, stage, html, readme_stage)
    problems += [p.replace("README.md", "README.md (publish)") for p in
                 checks.content_problems({"README.md": readme_final}) if publish]
    if publish and problems:
        shutil.rmtree(stage, ignore_errors=True)
        raise PublishRefused("refused:\n  " + "\n  ".join(problems))
    if publish:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out_dir / "assets", ignore_errors=True)
        shutil.move(str(stage / "assets"), str(out_dir / "assets"))
        for p in ("index.html", ".nojekyll"):
            shutil.move(str(stage / p), str(out_dir / p))
        shutil.rmtree(stage, ignore_errors=True)
        rp = Path(readme_path) if readme_path else None
        if rp:
            rp.write_text(readme_final)
    else:
        rp = stage / "README.md"
        rp.write_text(readme_stage)
    files = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    return BuildResult(out_dir, rp, weight, problems, files)
