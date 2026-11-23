"""Preview and publish builds: files, watermarks, replay, page weight, no-JS fallback and refusal rules."""
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgb
from PIL import Image, ImageChops

from btc.site import checks, figures as Fg, fixtures as FX
from btc.site.build import MAX_WEIGHT, PublishRefused, build_site
from btc.site.results import load_cerclage
from btc.site.theme import CERCLAGE, IMAGE_CREDIT, SYNTHETIC_MARK

from site_support import site_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def preview(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("site_build")
    site = site_for("row3", tmp / "fx")
    res = build_site(site, tmp / "out", publish=False, cache_dir=tmp / "cache", replay_rules=(1,), jobs=4)
    return site, res


def test_preview_files(preview):
    site, res = preview
    out = res.out_dir
    for f in ("index.html", "README.md", ".nojekyll", "assets/img/summary.png", "assets/img/summary-left.png",
              "assets/img/summary-right.png", "assets/img/og.png", "assets/img/per-patient.png",
              "assets/img/replay.gif", "assets/fonts/OFL-SourceSans3.txt", "assets/fonts/OFL-SourceSerif4.txt"):
        assert (out / f).exists(), f
    r1 = site.featured[1]
    for ext in ("mp4", "webm", "png"):
        assert (out / f"assets/replay/replay-{r1}.{ext}").exists()
    assert len(list((out / "assets/cases").glob("tile-*.png"))) == len(site.scored_cases)
    assert len(list((out / "assets/cases").glob("case-*.png"))) == len(site.scored_cases)
    assert len(list((out / "assets/fonts").glob("*.woff"))) == 4


def test_only_author_placeholders_remain_in_preview(preview):
    site, res = preview
    assert res.problems and all("placeholder" in p for p in res.problems)
    assert all("AUTHOR_NAME" in p or "CONTACT" in p for p in res.problems)


def test_every_png_is_marked(preview):
    site, res = preview
    pngs = list(res.out_dir.rglob("*.png"))
    assert len(pngs) > 40
    for p in pngs:
        im = Image.open(p)
        assert im.mode in ("P", "RGB")
        assert im.info.get("Comment") == SYNTHETIC_MARK, p.name


def test_synthetic_banner_on_page_and_readme(preview):
    site, res = preview
    html = (res.out_dir / "index.html").read_text()
    md = (res.out_dir / "README.md").read_text()
    assert 'class="synthetic-banner"' in html and SYNTHETIC_MARK in html
    assert md.splitlines()[2].startswith(f"> **{SYNTHETIC_MARK}.**")


def test_gif_opens_and_closes_on_the_complete_frame(preview):
    site, res = preview
    im = Image.open(res.out_dir / "assets/img/replay.gif")
    n = im.n_frames
    total = 0
    for i in range(n):
        im.seek(i)
        total += im.info["duration"]
    im.seek(0)
    first = im.convert("RGB").copy()
    im.seek(n - 1)
    last = im.convert("RGB").copy()
    im.seek(n // 2)
    mid = im.convert("RGB").copy()
    assert abs(total - 20000) <= 200 and im.size[0] == 830
    assert ImageChops.difference(first, last).getbbox() is None
    assert ImageChops.difference(first, mid).getbbox() is not None
    assert (res.out_dir / "assets/img/replay.gif").stat().st_size < 3_000_000


def test_video_is_not_autoplayed(preview):
    site, res = preview
    html = (res.out_dir / "index.html").read_text()
    vids = re.findall(r"<video[^>]*>", html)
    assert len(vids) == 1
    v = vids[0]
    assert "controls" in v and 'preload="none"' in v and "poster=" in v
    assert "autoplay" not in v and "loop" not in v


def test_page_weight_and_requests(preview):
    site, res = preview
    assert max(res.weight.values()) < MAX_WEIGHT
    html = (res.out_dir / "index.html").read_text()
    js = "".join(p.read_text() for p in (res.out_dir / "assets").glob("site.*.js"))
    assert checks.request_problems(html, js) == []
    assert "fetch(" not in js


def test_works_without_javascript(preview):
    site, res = preview
    html = (res.out_dir / "index.html").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    for c in site.scored_cases:
        assert f"case-{c.caseid}" in ids
        assert f'href="#case-{c.caseid}"' in html
        assert f'src="assets/cases/case-{c.caseid}.png"' in html
    assert "replay" in ids and "cases" in ids
    for c in site.cases:
        if not c.scored:
            assert f"case {c.caseid}</span> · not analysable: {c.reason}" in html
    # controls that need the script are hidden until it runs
    for m in re.finditer(r"<div[^>]*data-js-only[^>]*>", html):
        assert "hidden" in m.group(0)


def test_preview_readme_and_page_pass_parity(preview):
    site, res = preview
    assert not [p for p in res.problems if "differs" in p]


# --- cerclage cases in the per-patient figure (SPEC §6.7) ---------------------------------------------------

def _drawn(monkeypatch, site, tmp_path):
    figs = []
    monkeypatch.setattr(Fg, "save_png", lambda fig, path, synthetic: figs.append(fig))
    Fg.per_patient_figure(site, tmp_path / "pp.png", True)
    fig = figs[0]
    mine = lambda ln: to_hex(ln.get_color()) == to_hex(CERCLAGE)
    by_colour = [sorted(round(float(ln.get_ydata()[0]), 9) for ln in ax.lines if mine(ln)) for ax in fig.axes]
    keys = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
    return by_colour, keys


def test_per_patient_figure_marks_the_cerclage_cases_in_sdev_only(preview, monkeypatch, tmp_path):
    site, _ = preview
    assert site.cerclage == frozenset(FX.CERCLAGE_SDEV) and site.cerclage_sdev == sorted(FX.CERCLAGE_SDEV)
    pp = site.per_patient["S-dev"]
    m0 = pp.loc[pp["model"] == "M0"].set_index("caseid")["rmse"]
    (dev, ob), keys = _drawn(monkeypatch, site, tmp_path)
    assert dev == sorted(round(float(m0.loc[c]), 9) for c in FX.CERCLAGE_SDEV)       # exactly those patients
    assert ob == [] and keys == [f"cerclage ({len(FX.CERCLAGE_SDEV)})"]
    # an S-ob case ID among the flags changes nothing: the S-ob panel marks nothing
    sob_id = site.scored_cases[0].caseid
    (dev2, ob2), keys2 = _drawn(monkeypatch, dataclasses.replace(site, cerclage=site.cerclage | {sob_id}), tmp_path)
    assert (dev2, ob2, keys2) == (dev, [], keys)
    # no flags (preview without the case table): no orange line and no key
    assert _drawn(monkeypatch, dataclasses.replace(site, cerclage=None), tmp_path) == ([[], []], [])


def test_per_patient_png_has_the_cerclage_colour_in_the_left_panel_only(preview):
    site, res = preview
    a = np.asarray(Image.open(res.out_dir / "assets/img/per-patient.png").convert("RGB")).astype(int)
    near = np.abs(a - np.array(to_rgb(CERCLAGE)) * 255).sum(axis=2) < 40
    w = a.shape[1]
    assert near[:, : w // 2].sum() > 100 and near[:, w // 2:].sum() == 0


def test_cerclage_table_must_cover_the_sdev_per_patient_cases(preview, tmp_path):
    site, _ = preview
    ids = sorted(site.per_patient["S-dev"]["caseid"].unique())
    t = tmp_path / "sdev_cases.parquet"
    pd.DataFrame({"caseid": ids[1:], "cerclage": False}).to_parquet(t, index=False)
    with pytest.raises(ValueError, match="1 case"):
        load_cerclage(t, ids)
    with pytest.raises(FileNotFoundError):
        load_cerclage(tmp_path / "absent.parquet", ids)


# --- publish refusals ----------------------------------------------------------------------------------

def _real_like(site, **config):
    """A copy marked non-synthetic, for testing the publish path only (never written to docs/)."""
    cfg = dataclasses.replace(site.config, **config)
    return dataclasses.replace(site, synthetic=False, config=cfg)


def test_publish_refuses_synthetic(preview, tmp_path):
    site, _ = preview
    out = tmp_path / "docs"
    with pytest.raises(PublishRefused, match="synthetic"):
        build_site(site, out, publish=True, readme_path=tmp_path / "README.md", cache_dir=tmp_path / "c")
    assert not out.exists() and not (tmp_path / "README.md").exists()


def test_publish_refuses_placeholders(preview, tmp_path):
    site, res = preview
    out = tmp_path / "docs"
    with pytest.raises(PublishRefused, match=r"\{\{"):
        build_site(_real_like(site), out, publish=True, readme_path=tmp_path / "README.md",
                   cache_dir=res.out_dir.parent / "cache", replay_rules=())
    assert not (out / "index.html").exists() and not (tmp_path / "README.md").exists()


def test_publish_refuses_nan_and_unknown_deviations(preview, tmp_path):
    site, res = preview
    ok = dict(author_name="A. Author", contact="author@example.org")
    bad = _real_like(site, **ok)
    bad = dataclasses.replace(bad, headline=dataclasses.replace(bad.headline, sentence_3="X% (Δ NaN mmHg)"))
    with pytest.raises(PublishRefused, match="NaN"):
        build_site(bad, tmp_path / "docs", publish=True, cache_dir=res.out_dir.parent / "cache", replay_rules=())
    with pytest.raises(PublishRefused, match="deviations"):
        build_site(_real_like(site, deviations=None, **ok), tmp_path / "docs2", publish=True,
                   cache_dir=res.out_dir.parent / "cache", replay_rules=())


def test_publish_refuses_without_cerclage_flags(preview, tmp_path):
    site, res = preview
    good = _real_like(site, author_name="A. Author", contact="author@example.org")
    with pytest.raises(PublishRefused, match="cerclage"):
        build_site(dataclasses.replace(good, cerclage=None), tmp_path / "docs", publish=True,
                   readme_path=tmp_path / "README.md", cache_dir=res.out_dir.parent / "cache", replay_rules=())
    assert not (tmp_path / "docs").exists() and not (tmp_path / "README.md").exists()


def test_publish_succeeds_when_everything_is_filled(preview, tmp_path):
    site, res = preview
    good = _real_like(site, author_name="A. Author", contact="author@example.org")
    out = tmp_path / "docs"
    r = build_site(good, out, publish=True, readme_path=tmp_path / "README.md", cache_dir=res.out_dir.parent / "cache",
                   replay_rules=(1,), jobs=4)
    assert r.problems == []
    md = (tmp_path / "README.md").read_text()
    assert "](docs/assets/img/summary.png)" in md and SYNTHETIC_MARK not in md
    html = (out / "index.html").read_text()
    assert "synthetic-banner" not in html
    assert Image.open(out / "assets/img/summary.png").info.get("Comment") == IMAGE_CREDIT


def test_cli_publish_refuses_synthetic_results(preview, tmp_path):
    site, res = preview
    fx = res.out_dir.parent / "fx" / "row3"
    before = sorted(p.name for p in (ROOT / "docs").iterdir()) if (ROOT / "docs").exists() else None
    cmd = [sys.executable, str(ROOT / "scripts/build_site.py"), "--publish", "--results", str(fx / "results"),
           "--cases-dir", str(fx / "cases"), "--readme", str(tmp_path / "README.md")]
    r = subprocess.run([*cmd, "--sdev-cases", str(fx / "sdev_cases.parquet")], capture_output=True, text=True)
    assert r.returncode == 2 and "synthetic" in r.stderr
    r = subprocess.run([*cmd, "--sdev-cases", str(tmp_path / "absent.parquet")], capture_output=True, text=True)
    assert r.returncode == 2 and "missing input" in r.stderr and "absent.parquet" in r.stderr
    after = sorted(p.name for p in (ROOT / "docs").iterdir()) if (ROOT / "docs").exists() else None
    assert before == after and not (tmp_path / "README.md").exists()
