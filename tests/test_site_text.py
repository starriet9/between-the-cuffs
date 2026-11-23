"""README and page text: one source, word-for-word parity, every headline row and variant (SPEC §6.3, §8.1, §8.3)."""
import dataclasses
import re

import pytest

from btc.models import headline as H
from btc.site import checks, fixtures as FX, text as T
from btc.site.page import render_page
from btc.site.readme import md_escape, render_readme
from btc.site.results import README_MODELS, consistency_problems
from btc.site.series import NMB_REASON, CaseSummary

from site_support import fake_assets, site_for

EXPECTED = {   # variant: (row, label) — labels are the registered texts (SPEC §6.3 plain lines)
    "row1": (1, "No claim."), "row1-hr": (1, "No claim."), "row2": (2, "Yes."), "row2-hr-placebo": (2, "Yes."),
    "row2-coverage": (2, "Yes."), "row3": (3, "Yes, mainly from heart rate."), "row3-hr-placebo": (3, "Yes."),
    "row3-deviation": (3, "Yes, mainly from heart rate."), "row4": (4, "Yes, from the pleth waveform."),
    "row5": (5, "Yes, together."), "row6": (6, "No."), "row6-not-shown": (6, "Not shown."),
    "row6-raised": (6, "No: the added signals raised the error."), "row6-hr": (6, "Not shown."), "row6-gb": (6, "No."),
}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("site_text")
    out = {}
    for v in FX.VARIANTS:
        site = site_for(v, tmp)
        out[v] = (site, render_readme(site, asset_prefix="assets/"), render_page(site, fake_assets(site), "a.css", "a.js"))
    return out


def test_every_variant_is_covered():
    assert set(EXPECTED) == set(FX.VARIANTS)
    assert {r for r, _ in EXPECTED.values()} == {1, 2, 3, 4, 5, 6}


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_row_and_label(rendered, variant):
    site, md, html = rendered[variant]
    row, lab = EXPECTED[variant]
    h = site.headline
    assert (h.row, h.label) == (row, lab)
    assert h.answer_line == f"{h.label} {h.answer_rest}"
    assert f"**{lab}** In {h.N} non-cesarean spinal cases (cross-validated), " in md
    assert f'<p class="answer-label">{lab}</p>' in html.replace("&#x27;", "'")


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_page_top_block_equals_readme_block_1(rendered, variant):
    site, md, html = rendered[variant]
    assert checks.parity_problems(md, html, T.blocks(site)) == []
    md1 = checks.md_blocks(md, {})["1"]
    pg1 = checks.html_blocks(html)["1"]
    assert md1 == pg1 and len(md1) > 500


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_sentences_come_verbatim_from_the_headline_module(rendered, variant):
    site, md, html = rendered[variant]
    h = site.headline
    md_text, page_text = checks.md_plain(md), checks.html_blocks(html)["1"]
    for s in (h.question, h.answer_rest, h.cesarean_line, h.sentence_1, h.sentence_2, h.sentence_3):
        assert s in md_text and s in page_text
    assert h.question == H.question()


def test_parity_check_detects_a_change(rendered):
    site, md, html = rendered["row2"]
    bad = md.replace("before any model was fitted", "before any model was fit", 1)
    assert checks.parity_problems(bad, html, T.blocks(site))
    bad_html = html.replace(site.headline.sentence_3[:40], site.headline.sentence_3[:39] + "X", 1)
    assert checks.parity_problems(md, bad_html, T.blocks(site))


def test_variants_change_only_what_they_should(rendered):
    cov = rendered["row2-coverage"][0].headline
    assert cov.coverage_rule and cov.cesarean_line == cov.sentence_2 and "insufficient signal coverage" in cov.sentence_2
    assert "neither confirms nor refutes the answer above" in rendered["row2"][0].headline.cesarean_line
    for v, (site, md, html) in rendered.items():
        gb = "An exploratory gradient-boosting check on the same inputs gave a different answer"
        assert (gb in md and gb in html) == (v == "row6-gb")
        dev = "deviations from the registration, listed in"
        assert (dev in md) == (v == "row3-deviation")
    assert "Heart rate alone met the rule (6.1%)." in rendered["row6-hr"][0].headline.answer_rest
    assert "Gains larger than" not in rendered["row6-raised"][0].headline.answer_rest
    assert "heart rate's share did not pass its placebo check" in rendered["row3-hr-placebo"][0].headline.answer_rest


def test_emphasis(rendered):
    for v, (site, md, html) in rendered.items():
        ems = re.findall(r'<strong class="em">([^<]*)</strong>', html)
        h = site.headline
        if h.row == 1:
            assert ems == []
        elif h.row == 6:
            assert ems[0] == "did not meet the pre-specified rule"
        else:
            assert len(ems) == 2 and ems[0].endswith("%") and ems[1].startswith("from ") and ems[1].endswith(" mmHg")
        answer_md = next(ln for ln in md.splitlines() if ln.startswith(f"**{h.label}** In "))
        assert answer_md.count("**") == 2          # the README bolds only the label


def md_escape_s1(site):
    return md_escape(site.headline.sentence_1)


def test_readme_block_order_and_page_order(rendered):
    site, md, html = rendered["row6-hr"]
    heads = re.findall(r"(?m)^## (.+)$", md)
    assert heads == [T.MEANING_HEADING, "Next experiment", "Replay", "Error by model", "What this does not show",
                     "Author's note", "Credit"]
    order = re.findall(r'data-block="(\d)"', html)
    assert order == ["1", "2", "3", "4", "5", "6", "7", "8"]
    assert html.index('id="cases"') < html.index('id="models"') and html.index('id="check"') < html.index('id="credit"')
    assert md.index("![Left, an illustrative case") < md.index(md_escape_s1(site))


def test_credit_on_both(rendered):
    for site, md, html in rendered.values():
        assert T.CREDIT_STATEMENT in checks.md_plain(md)
        assert T.CREDIT_STATEMENT in checks.html_blocks(html)["8"]
        assert "https://creativecommons.org/licenses/by/4.0/" in md and "https://creativecommons.org/licenses/by/4.0/" in html


def test_table_models_match_the_pipeline():
    from btc.models.pipeline import README_MODELS as PIPE
    assert tuple(README_MODELS) == tuple(PIPE)


def test_no_forbidden_text_in_rendered_outputs(rendered):
    for v, (site, md, html) in rendered.items():
        probs = [p for p in checks.content_problems({"README": md, "page": html}) if "placeholder" not in p]
        assert probs == [], probs
        assert checks.unit_problems(html, md) == []
        assert checks.request_problems(html, "") == []


def test_case_with_neuromuscular_blocker_shows_as_not_analysed(tmp_path):
    """SPEC §2.2: an analysable cesarean case with a neuromuscular blocker recorded is listed as not analysed,
    with the count in the limits line; nothing changes when there is none."""
    site = site_for("row3", tmp_path)
    lim0 = "with 21 cesarean cases (20 analysable), 19 flagged emergency."
    assert lim0 in checks.md_plain(render_readme(site, asset_prefix="assets/"))
    featured = set(site.featured.values())
    cid = next(c.caseid for c in site.scored_cases if c.caseid not in featured)
    cases = [CaseSummary(caseid=c.caseid, status="not analysed", reason=NMB_REASON) if c.caseid == cid else c
             for c in site.cases]
    nmb = dataclasses.replace(site, cases=cases, series={k: v for k, v in site.series.items() if k != cid},
                              n_nmb_sob=1)
    md, html = render_readme(nmb, asset_prefix="assets/"), render_page(nmb, fake_assets(nmb), "a.css", "a.js")
    assert f"case {cid}</span> · not analysed: neuromuscular blocker recorded" in html
    lim1 = ("with 21 cesarean cases (20 analysable, 1 of them not analysed because a neuromuscular blocker "
            "was recorded), 19 flagged emergency.")
    assert lim1 in checks.md_plain(md) and lim1 in html
    assert checks.parity_problems(md, html, T.blocks(nmb)) == []
    assert not [p for p in consistency_problems(nmb) if "neuromuscular" in p]
    assert [p for p in consistency_problems(dataclasses.replace(nmb, n_nmb_sob=0)) if "neuromuscular" in p]
