"""Shared visual tokens for the page and every figure.

One meaning for the accent: heart rate, pleth and the model that uses them. Grey = cuff only.
Black = measured cuff readings. Orange marks the cerclage cases in the per-patient figure, and nothing
else. No red or green in any published image.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
STATIC_DIR = HERE / "static"

# Colour tokens (light only)
BG = "#FFFFFF"
BG_SOFT = "#F4F6F8"
INK = "#14181F"
INK_2 = "#4A5361"
RULE = "#D9DEE5"
MUTE = "#8A929D"
ACCENT = "#1B62A8"
ACCENT_TINT = "#DCE8F4"
CERCLAGE = "#B85C00"       # per-patient figure only: the S-dev cerclage cases (SPEC §6.7)
SYNTH = "#B3261E"          # synthetic watermark only; never in a published image

SANS = "Source Sans 3"
SERIF = "Source Serif 4"

# Web font files (full glyph set, WOFF container; font data unchanged) and their licences
WEB_FONTS = {
    "SourceSerif4-SemiBold.woff": (SERIF, 600),
    "SourceSerif4-Bold.woff": (SERIF, 700),
    "SourceSans3-Regular.woff": (SANS, 400),
    "SourceSans3-SemiBold.woff": (SANS, 600),
}
FONT_LICENCES = ("OFL-SourceSans3.txt", "OFL-SourceSerif4.txt")
PRELOAD_FONTS = ("SourceSerif4-SemiBold.woff", "SourceSans3-Regular.woff")

SYNTHETIC_MARK = "SYNTHETIC — NOT A RESULT"
IMAGE_CREDIT = "Derived from VitalDB (CC BY 4.0)"


def px(n: float) -> float:
    """CSS pixels at 1x (figure dpi 100) to points."""
    return n * 0.72


_registered = False


def use_matplotlib_theme():
    """Register the self-hosted TTFs with matplotlib and set the shared rc parameters."""
    global _registered
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    if not _registered:
        for f in sorted(FONT_DIR.glob("*.ttf")):
            font_manager.fontManager.addfont(str(f))
        _registered = True
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [SANS, "DejaVu Sans"],
        "font.size": px(13),
        "text.color": INK,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "axes.linewidth": 0.6,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "hatch.color": MUTE,
        "hatch.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
