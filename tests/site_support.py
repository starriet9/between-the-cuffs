"""Shared helpers for the site tests: fixture results trees and a fake asset table (no images)."""
from functools import lru_cache
from pathlib import Path

from btc.site import fixtures as FX
from btc.site.featured import rule_of
from btc.site.results import load_site_data


@lru_cache(maxsize=1)
def cases_and_series():
    return FX.make_cases()


def site_for(variant: str, tmp: Path, **config):
    cases, series = cases_and_series()
    paths = FX.write_results_tree(Path(tmp) / variant, variant, cases, series)
    return load_site_data(paths["results"], paths["cases"], FX.config_for(variant, **config), variant=variant,
                          sdev_cases=paths["sdev_cases"])


def fake_assets(site) -> dict:
    a = {"img/summary.png": (1040, 430), "img/summary-left.png": (540, 660), "img/summary-right.png": (540, 430),
         "img/per-patient.png": (1040, 400), "img/og.png": (1200, 630), "img/replay.gif": (830, 498)}
    for c in site.scored_cases:
        a[f"cases/tile-{c.caseid}.png"] = (200, 76)
        a[f"cases/case-{c.caseid}.png"] = (1040, 620)
    a["replays"] = {cid: {"mp4": f"assets/replay/replay-{cid}.mp4", "webm": f"assets/replay/replay-{cid}.webm",
                          "poster": f"assets/replay/replay-{cid}.png", "dims": (1040, 624)}
                    for cid, r in rule_of(site.featured).items() if r == 1}
    return a
