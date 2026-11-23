"""Every image of the README and the page (SPEC §8.1).

All layouts are written in CSS pixels at 1x (figure dpi 100) and saved at 2x as opaque white PNGs,
except the replay, whose frames are encoded to MP4/WebM (page) and GIF (README). One set of drawing
functions serves the case viewer image, the summary figure's left panel, the tiles and every replay
frame, so they cannot drift apart. Each image carries the short credit; synthetic renders carry a
diagonal "SYNTHETIC — NOT A RESULT" watermark and the same text in the PNG metadata.
"""
from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .theme import (ACCENT, ACCENT_TINT, BG, CERCLAGE, IMAGE_CREDIT, INK, INK_2, MUTE, RULE, SYNTH, SYNTHETIC_MARK,
                    px, use_matplotlib_theme)

use_matplotlib_theme()

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from btc.models.headline import pct  # noqa: E402

UG_PER_ML = 100.0                 # SPEC §2.6: VitalDB-documented syringe concentration
PUMP_SCALE_NOTE = "right scale: µg/min at the VitalDB-documented concentration of 100 µg/mL"
SCALE = 2                         # pixel density of the PNGs


@dataclass(frozen=True)
class Style:
    tick: float = 12.0
    label: float = 13.0
    title: float = 15.0
    credit: float = 10.0
    dot: float = 7.0              # cuff dot diameter, px
    lw_m2: float = 1.5
    lw_m4: float = 2.0
    left: float = 92.0            # left margin (row titles + ticks), px
    right: float = 104.0          # right margin (reference-line labels, pump µg/min scale), px


DESKTOP = Style()
PHONE = Style(tick=17, label=18, title=20, credit=14, dot=9, lw_m2=2.0, lw_m4=2.6, left=112, right=124)
REPLAY = Style(tick=12, label=13, title=15, credit=10, dot=7, left=92, right=104)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def new_fig(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(BG)
    fig._css = (w, h)
    return fig


def add_axes(fig, x, y, w, h):
    W, H = fig._css
    return fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H))


def ftext(fig, x, y, s, size, **kw):
    """Text at CSS px (x, y) from the top-left; va defaults to top."""
    W, H = fig._css
    kw.setdefault("va", "top")
    return fig.text(x / W, 1 - y / H, s, fontsize=px(size), **kw)


def credit(fig, style: Style, y_from_bottom: float = 5):
    W, H = fig._css
    fig.text(1 - 8 / W, y_from_bottom / H, IMAGE_CREDIT, ha="right", va="bottom", fontsize=px(style.credit),
             color=INK_2)


def watermark(fig, synthetic: bool, size: float | None = None):
    if not synthetic:
        return
    W, H = fig._css
    ang = math.degrees(math.atan2(H, W))
    size = size or max(14.0, min(W, H * 2.2) / 14)
    fig.text(0.5, 0.5, SYNTHETIC_MARK, rotation=ang, ha="center", va="center", fontsize=px(size), color=SYNTH,
             alpha=0.28, fontweight="semibold", zorder=1000, rotation_mode="anchor")


def write_png(img, path: Path, synthetic: bool):
    """Opaque PNG, quantised to a 128-colour palette (flat line art: no visible change, about 4x smaller),
    with the credit or the synthetic mark in its metadata."""
    from PIL import Image, PngImagePlugin
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", SYNTHETIC_MARK if synthetic else IMAGE_CREDIT)
    q = img.convert("RGB").quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    q.save(path, pnginfo=info, optimize=True)


def save_png(fig, path: Path, synthetic: bool, scale: int = SCALE):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100 * scale, facecolor=BG, format="png")
    buf.seek(0)
    write_png(Image.open(buf), path, synthetic)


def _style_axis(ax, style: Style, grid=True):
    ax.tick_params(labelsize=px(style.tick), length=0, pad=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    if grid:
        ax.grid(axis="y", color=RULE, linewidth=0.5)
        ax.set_axisbelow(True)


def _nice_range(lo, hi, step):
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


# ---------------------------------------------------------------------------
# the case strip: MAP (+ error) + heart rate + pleth + pump
# ---------------------------------------------------------------------------

def case_limits(s) -> dict:
    w1 = s.window_s[1]
    vals = [s.cuff_map, [65.0, 0.8 * s.baseline_map]]
    for ip in s.issues:
        k = ip.h_s <= ip.end_h(w1) + 1e-9
        for c in (ip.m2, ip.m4):
            vals += [c.map[k] - c.half90[k], c.map[k] + c.half90[k]]
    v = np.concatenate([np.asarray(x, dtype=float) for x in vals])
    lo, hi = _nice_range(np.nanmin(v) - 3, np.nanmax(v) + 3, 10)
    hr = s.hr_bpm[np.isfinite(s.hr_bpm)]
    hlo, hhi = _nice_range((hr.min() if len(hr) else 60) - 4, (hr.max() if len(hr) else 100) + 4, 10)
    err = [abs(ip.pred_at_target[m] - ip.y_next) for ip in s.issues if ip.scored for m in ("M2", "M4")]
    pmax = float(np.nanmax(s.pump_ml_h)) if s.has_pump else 0.0
    return {"map": (lo, hi), "hr": (hlo, hhi), "err": (0, max(4.0, math.ceil(max(err or [1]) * 1.1))),
            "pump": (0, max(10.0, math.ceil(pmax * 1.18 / 10) * 10)), "x": (s.window_s[0] / 60, w1 / 60)}


def case_rows(has_pump: bool, error_row: bool, top: float, bottom: float, map_share: float | None = None,
              style: Style = DESKTOP) -> list:
    """Row geometry (name, y, h) in CSS px between `top` and `bottom`."""
    k = style.tick / 12
    fixed = [("err", 56 * k, 18 * k)] if error_row else []
    fixed += [("hr", 52 * k, 18 * k), ("pleth", 15 * k, 10 * k)]
    if has_pump:
        fixed.append(("pump", 46 * k, (30 if style is PHONE else 16) * k))
    axis = 30 * k
    avail = bottom - top - axis
    used = sum(h + g for _, h, g in fixed)
    map_h = max(avail - used, 80) if map_share is None else map_share
    rows, y = [("map", top, map_h)], top + map_h
    for name, h, g in fixed:
        y += g
        rows.append((name, y, h))
        y += h
    return rows


def build_case_axes(fig, s, x0, x1, top, bottom, *, error_row, style=DESKTOP, map_share=None):
    lim = case_limits(s)
    rows = case_rows(s.has_pump, error_row, top, bottom, map_share, style)
    axes = {}
    left, right = x0 + style.left, x1 - style.right
    titles = {"map": "MAP\nmmHg", "err": "absolute\nerror, mmHg", "hr": "heart rate\n/min", "pleth": "pleth",
              "pump": "pump\nmL/h"}
    for name, y, h in rows:
        ax = add_axes(fig, left, y, right - left, h)
        ax.set_xlim(*lim["x"])
        _style_axis(ax, style, grid=name not in ("pleth",))
        ax.tick_params(axis="x", labelbottom=False)
        ftext(fig, x0 + 4, y + h / 2, titles[name], style.label * 0.92, va="center", ha="left", color=INK_2,
              linespacing=1.05)
        axes[name] = ax
    a = axes
    a["map"].set_ylim(*lim["map"])
    a["map"].yaxis.set_major_locator(_ticker_multiple(10 if lim["map"][1] - lim["map"][0] <= 60 else 20))
    a["hr"].set_ylim(*lim["hr"])
    a["hr"].yaxis.set_major_locator(_ticker_maxn(3))
    a["pleth"].set_ylim(0, 1)
    a["pleth"].set_yticks([])
    a["pleth"].spines["bottom"].set_visible(False)
    if "err" in a:
        a["err"].set_ylim(*lim["err"])
        a["err"].yaxis.set_major_locator(_ticker_maxn(3))
    if "pump" in a:
        a["pump"].set_ylim(*lim["pump"])
        a["pump"].yaxis.set_major_locator(_ticker_maxn(3))
        sec = a["pump"].secondary_yaxis("right", functions=(lambda v: v * UG_PER_ML / 60, lambda v: v * 60 / UG_PER_ML))
        sec.tick_params(labelsize=px(style.tick * 0.92), length=0, pad=3, colors=INK_2)
        sec.spines["right"].set_visible(False)
        sec.yaxis.set_major_locator(_ticker_maxn(3))
        note = PUMP_SCALE_NOTE if style is not PHONE else PUMP_SCALE_NOTE.replace(" concentration", "\nconcentration")
        a["pump"].text(1.0, 1.0, note, transform=a["pump"].transAxes, ha="right", va="bottom",
                       fontsize=px(style.tick * 0.88), color=INK_2, linespacing=1.05)
    bottom_ax = a[rows[-1][0]]
    bottom_ax.tick_params(axis="x", labelbottom=True)
    bottom_ax.set_xticks(np.arange(0, lim["x"][1] + 0.1, 5))
    bottom_ax.set_xlabel("minutes since record start", fontsize=px(style.label), color=INK_2, labelpad=3)
    if bottom_ax is a["pleth"]:
        a["pleth"].tick_params(axis="x", length=0)
    _static_case(a, s, lim, style)
    return a, lim


def _ticker_multiple(m):
    from matplotlib.ticker import MultipleLocator
    return MultipleLocator(m)


def _ticker_maxn(n):
    from matplotlib.ticker import MaxNLocator
    return MaxNLocator(n, integer=True, min_n_ticks=2)


def _static_case(a, s, lim, style):
    """Reference lines and labels that never move."""
    ax = a["map"]
    b80 = 0.8 * s.baseline_map
    for yv, lab in ((65.0, "65 mmHg"), (b80, "80% of record-start baseline")):
        ax.axhline(yv, color=INK_2, lw=px(1), ls=(0, (4, 3)), zorder=1)
    # labels at the right end of each line, in the margin; nudged apart if the two lines are close
    lo, hi = lim["map"]
    ypx = ax.get_position().height * ax.figure._css[1]
    min_gap = (hi - lo) * (style.tick * 2.3) / ypx
    y65, y80 = 65.0, b80
    if abs(y80 - y65) < min_gap:
        mid = (y80 + y65) / 2
        y80, y65 = (mid + min_gap / 2, mid - min_gap / 2) if b80 >= 65 else (mid - min_gap / 2, mid + min_gap / 2)
    for yv, lab in ((y65, "65 mmHg"), (y80, "80% of record-\nstart baseline")):
        ax.annotate(lab, xy=(1.0, yv), xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=px(style.tick * 0.88), color=INK_2, linespacing=1.0,
                    annotation_clip=False)
    if s.incision_s is not None and s.window_s[0] <= s.incision_s <= s.window_s[1]:
        xi = s.incision_s / 60
        ax.plot([xi, xi], [hi, hi - (hi - lo) * 0.06], color=INK_2, lw=px(1.2), zorder=3, clip_on=False)
        ax.text(xi, hi, " incision", ha="left", va="top", fontsize=px(style.tick * 0.88), color=INK_2)


def _xoff(ax, px_off):
    """Data-x offset equivalent to `px_off` CSS px on this axis."""
    x0, x1 = ax.get_xlim()
    wpx = ax.get_position().width * ax.figure._css[0]
    return (x1 - x0) * px_off / wpx


def draw_case_dynamic(a, s, lim, style, now: float | None = None) -> list:
    """Everything that depends on time. now=None draws the complete frame. Returns the artists."""
    out = []
    w1 = s.window_s[1]
    t_now = w1 if now is None else now
    ax = a["map"]
    for ip in s.issues:
        if ip.t_issue_s > t_now:
            continue
        landed = ip.t_next_s is not None and ip.t_next_s <= t_now and ip.t_next_s - ip.t_issue_s <= 400
        h_end = ip.t_next_s - ip.t_issue_s if landed else min(400.0, w1 - ip.t_issue_s)
        k = ip.h_s <= h_end + 1e-9
        if k.sum() < 2:
            continue
        x = (ip.t_issue_s + ip.h_s[k]) / 60
        for c, col, lw, ls, z in ((ip.m2, MUTE, style.lw_m2, (0, (4, 2.5)), 3), (ip.m4, ACCENT, style.lw_m4, "-", 4)):
            out.append(ax.fill_between(x, c.map[k] - c.half90[k], c.map[k] + c.half90[k], color=col, alpha=0.10,
                                       lw=0, zorder=1))
            out += ax.plot(x, c.map[k], color=col, lw=px(lw), ls=ls, zorder=z, solid_capstyle="round")
        if landed and ip.scored:
            d = _xoff(ax, 2.2)
            xt = ip.t_next_s / 60
            for m, col, sgn in (("M2", MUTE, -1), ("M4", ACCENT, 1)):
                out += ax.plot([xt + sgn * d] * 2, [ip.pred_at_target[m], ip.y_next], color=col, lw=px(1.1), zorder=5)
            if "err" in a:
                bw = _xoff(a["err"], 3.0)
                for m, col, sgn in (("M2", MUTE, -1), ("M4", ACCENT, 1)):
                    e = abs(ip.pred_at_target[m] - ip.y_next)
                    out.append(a["err"].bar(xt + sgn * bw * 0.6, e, width=bw, color=col, lw=0, zorder=3))
    kc = s.cuff_t <= t_now
    out += ax.plot(s.cuff_t[kc] / 60, s.cuff_map[kc], ls="none", marker="o", ms=px(style.dot), mfc=INK, mec=BG,
                   mew=px(1), zorder=7)
    kh = s.hr_t <= t_now
    out += a["hr"].plot(s.hr_t[kh] / 60, s.hr_bpm[kh], color=ACCENT, lw=px(1.3), zorder=3)
    # pleth strip: usable windows in accent tint, unusable hatched grey
    st = s.pleth_start
    kp = st < t_now
    good = [((x0 / 60), (min(x0 + 10, t_now) - x0) / 60) for x0, u in zip(st[kp], s.pleth_usable[kp]) if u]
    bad = [((x0 / 60), (min(x0 + 10, t_now) - x0) / 60) for x0, u in zip(st[kp], s.pleth_usable[kp]) if not u]
    if good:
        out.append(a["pleth"].broken_barh(_merge(good), (0, 1), facecolors=ACCENT_TINT, lw=0))
    if bad:
        out.append(a["pleth"].broken_barh(_merge(bad), (0, 1), facecolors=BG, edgecolors=MUTE, hatch="////",
                                          lw=0))
    if now is None:
        share = s.pleth_usable_share
        out.append(a["pleth"].text(1.005, 0.5, f"{pct(share)}% usable", transform=a["pleth"].transAxes, ha="left",
                                   va="center", fontsize=px(style.tick * 0.88), color=INK_2))
    if s.has_pump:
        kq = s.pump_t <= t_now
        if kq.any():
            xs = np.append(s.pump_t[kq], t_now) / 60
            ys = np.append(s.pump_ml_h[kq], s.pump_ml_h[kq][-1])
            out += a["pump"].step(xs, ys, where="post", color=INK_2, lw=px(1.5), zorder=3)
    if now is not None:
        for k_ax in a.values():
            out.append(k_ax.axvline(now / 60, color=INK_2, lw=px(0.9), zorder=8))
    return out


def _merge(ranges):
    out = []
    for x, w in ranges:
        if out and abs(out[-1][0] + out[-1][1] - x) < 1e-9:
            out[-1] = (out[-1][0], out[-1][1] + w)
        else:
            out.append((x, w))
    return out


def case_key(fig, x, y, style, ncol=4):
    W, H = fig._css
    handles = [Line2D([], [], ls="none", marker="o", ms=px(style.dot), mfc=INK, mec=BG),
               Line2D([], [], color=MUTE, lw=px(style.lw_m2), ls=(0, (4, 2.5))),
               Line2D([], [], color=ACCENT, lw=px(style.lw_m4)),
               Patch(facecolor=ACCENT, alpha=0.18, lw=0)]
    labels = ["cuff reading", "forecast, cuff only", "forecast, + HR + pleth", "90% interval"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x / W, 1 - y / H), ncol=ncol, frameon=False,
               fontsize=px(style.tick), handlelength=1.9, handletextpad=0.5, columnspacing=1.3, borderaxespad=0,
               borderpad=0, labelcolor=INK_2)


# ---------------------------------------------------------------------------
# case viewer image and tiles
# ---------------------------------------------------------------------------

def case_image(s, path: Path, synthetic: bool, title: str | None = None):
    """Viewer image, 1040 x 620 CSS px: MAP, error, heart rate, pleth, pump if recorded."""
    W, Hh = 1040, 620
    fig = new_fig(W, Hh)
    ftext(fig, 12, 10, title or f"VitalDB case {s.caseid}", DESKTOP.title, fontweight="semibold", color=INK)
    case_key(fig, 12, 34, DESKTOP)
    a, lim = build_case_axes(fig, s, 0, W, 62, Hh - 16, error_row=True)
    draw_case_dynamic(a, s, lim, DESKTOP)
    credit(fig, DESKTOP)
    watermark(fig, synthetic)
    save_png(fig, path, synthetic)


def tile_image(s, path: Path, synthetic: bool, map_range: tuple):
    """Contact-sheet tile, 200 x 76 CSS px: cuff dots with the 65 mmHg and 80%-of-baseline lines."""
    W, Hh = 200, 76
    fig = new_fig(W, Hh)
    ax = add_axes(fig, 4, 4, W - 8, Hh - 22)
    ax.set_xlim(s.window_s[0] / 60, s.window_s[1] / 60)
    ax.set_ylim(*map_range)
    ax.axis("off")
    for yv in (65.0, 0.8 * s.baseline_map):
        ax.axhline(yv, color=INK_2, lw=px(0.8), ls=(0, (3, 2)))
    ax.plot(s.cuff_t / 60, s.cuff_map, ls="none", marker="o", ms=px(4.2), mfc=INK, mec=BG, mew=px(0.6))
    credit(fig, replace(DESKTOP, credit=9.5), y_from_bottom=2)
    if synthetic:
        fig.text(0.5, 0.55, "SYNTHETIC", ha="center", va="center", rotation=15, fontsize=px(22), color=SYNTH,
                 alpha=0.3, fontweight="semibold")
    save_png(fig, path, synthetic)


def tile_map_range(series: dict) -> tuple:
    v = np.concatenate([s.cuff_map for s in series.values()] + [[65.0]])
    return _nice_range(np.nanmin(v) - 4, np.nanmax(v) + 4, 10)


# ---------------------------------------------------------------------------
# summary figure (README block 1 and page top)
# ---------------------------------------------------------------------------

ROWS = (("M3", "+ heart rate", INK, False), ("M4", "+ heart rate + pleth", ACCENT, True),
        ("M4-P1", "another patient's signals (placebo)", MUTE, False))


def _ci_text(r) -> str:
    return f"{pct(r.rr)}% ({pct(r.rr_lo)} to {pct(r.rr_hi)})"


def result_panel(fig, site, x0, x1, top, bottom, style: Style):
    """Relative reduction vs M2 with 95% CIs: filled = S-dev (decides), hollow = S-ob (case series)."""
    k = style.tick / 12
    num_w = 140 * k
    ftext(fig, x0, top, "Error reduction vs cuff only", style.title, fontweight="semibold", color=INK)
    legend_h = 44 * k             # two legend rows
    label_h = 44 * k              # tick labels + axis label line
    ax_top, ax_w = top + 46 * k, (x1 - x0) - num_w
    ax_h = bottom - ax_top - label_h - legend_h
    ax = add_axes(fig, x0, ax_top, ax_w, ax_h)
    dev, ob = site.contrasts["S-dev"], site.contrasts["S-ob"]
    lo = min([-5.0] + [100 * dev[m].rr_lo for m, *_ in ROWS])
    hi = max([10.0] + [100 * dev[m].rr_hi for m, *_ in ROWS])
    span = hi - lo
    xmin, xmax = math.floor(lo - 0.06 * span), math.ceil(hi + 0.06 * span)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(2.62, -0.7)
    _style_axis(ax, style, grid=False)
    ax.set_yticks([])
    ax.axvline(0, color=INK_2, lw=px(1.1), zorder=1)
    ax.axvline(5, color=INK_2, lw=px(1), ls=(0, (4, 3)), zorder=1)
    ax.text(5, -0.68, " pre-set margin", ha="left", va="top", fontsize=px(style.tick * 0.92), color=INK_2)
    for i, (m, lab, col, primary) in enumerate(ROWS):
        ax.text(xmin, i - 0.36, lab, ha="left", va="center", fontsize=px(style.label),
                fontweight="semibold" if primary else "normal", color=INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none"))
        for r, y, filled in ((dev[m], i - 0.02, True), (ob[m], i + 0.30, False)):
            a_, b_, c_ = 100 * r.rr_lo, 100 * r.rr, 100 * r.rr_hi
            ca, cb = max(a_, xmin), min(c_, xmax)
            if cb > ca:
                ax.plot([ca, cb], [y, y], color=col, lw=px(2.0 if filled else 1.4), solid_capstyle="butt", zorder=3)
            for edge, clipped, d in ((xmin, a_ < xmin, -1), (xmax, c_ > xmax, 1)):
                if clipped:
                    ax.annotate("", xy=(edge, y), xytext=(edge - d * span * 0.05, y),
                                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5", color=col,
                                                lw=px(1.4)), zorder=4)
            if xmin <= b_ <= xmax:
                ax.plot([b_], [y], ls="none", marker="o", ms=px(9 * k), mfc=col if filled else BG, mec=col,
                        mew=px(1.6 * k), zorder=5)
            ax.annotate(_ci_text(r), xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(10 * k * 0.72, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=px(style.tick),
                        color=INK if filled else INK_2, annotation_clip=False)
    ax.tick_params(axis="x", labelsize=px(style.tick))
    y_lab = ax_top + ax_h + 22 * k
    ftext(fig, x0, y_lab, "% lower error than cuff only", style.tick, ha="left", color=INK_2)
    ftext(fig, x0 + ax_w, y_lab, "better →", style.tick, ha="right", color=INK_2)
    h = site.headline
    handles = [Line2D([], [], ls="none", marker="o", ms=px(9 * k), mfc=INK_2, mec=INK_2),
               Line2D([], [], ls="none", marker="o", ms=px(9 * k), mfc=BG, mec=INK_2, mew=px(1.6 * k))]
    labels = [f"{h.N} non-cesarean cases (decides)", f"{h.n} cesarean cases (case series)"]
    Wc, Hc = fig._css
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(x0 / Wc, 1 - (bottom - 2) / Hc), ncol=1,
               frameon=False, fontsize=px(style.tick), handletextpad=0.4, borderaxespad=0, borderpad=0,
               labelcolor=INK_2, labelspacing=0.35)


def summary_left(fig, site, s, x0, x1, top, bottom, style: Style, ncol_key=4):
    k = style.tick / 12
    title = f"One cesarean case (VitalDB case {s.caseid}, chosen by a pre-set rule)"
    lines = 1
    if ncol_key != 4:
        title, lines = title.replace(" (", "\n(", 1), 2
    ftext(fig, x0 + 4, top, title, style.title, fontweight="semibold", color=INK, linespacing=1.15)
    key_y = top + (24 + 19 * (lines - 1)) * k
    case_key(fig, x0 + 4, key_y, style, ncol=ncol_key)
    key_rows = 1 if ncol_key == 4 else 2
    t0 = key_y + (28 + 19 * (key_rows - 1)) * k
    a, lim = build_case_axes(fig, s, x0, x1, t0, bottom, error_row=False, style=style,
                             map_share=(bottom - top) * 0.55 - (t0 - top) if style is DESKTOP else None)
    draw_case_dynamic(a, s, lim, style)


def summary_figures(site, out_dir: Path, synthetic: bool) -> dict:
    """summary.png (1040 x 430), summary-left.png / summary-right.png (540 wide, phone), og.png (1200 x 630)."""
    s = site.series[site.featured[1]]
    out = {}
    W, Hh = 1040, 430
    fig = new_fig(W, Hh)
    split = W * 0.62
    summary_left(fig, site, s, 0, split, 10, Hh - 18, DESKTOP)
    result_panel(fig, site, split + 26, W - 6, 10, Hh - 20, DESKTOP)
    credit(fig, DESKTOP)
    watermark(fig, synthetic, size=34)
    save_png(fig, out_dir / "summary.png", synthetic)
    out["summary.png"] = (W, Hh)
    # phone: the same two panels, stacked, each 540 CSS px wide (shown at ~360 px; text >= 11 px)
    Hl = 660 if s.has_pump else 590
    fig = new_fig(540, Hl)
    summary_left(fig, site, s, 0, 540, 10, Hl - 34, PHONE, ncol_key=2)
    credit(fig, PHONE)
    watermark(fig, synthetic, size=34)
    save_png(fig, out_dir / "summary-left.png", synthetic)
    out["summary-left.png"] = (540, Hl)
    fig = new_fig(540, 430)
    result_panel(fig, site, 12, 528, 10, 430 - 26, PHONE)
    credit(fig, PHONE)
    watermark(fig, synthetic, size=34)
    save_png(fig, out_dir / "summary-right.png", synthetic)
    out["summary-right.png"] = (540, 430)
    _og(out_dir / "summary.png", out_dir / "og.png", synthetic)
    out["og.png"] = (1200, 630)
    return out


def _og(src: Path, dst: Path, synthetic: bool):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w = 1160
    im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    can = Image.new("RGB", (1200, 630), BG)
    can.paste(im, ((1200 - w) // 2, (630 - im.height) // 2))
    write_png(can, dst, synthetic)


# ---------------------------------------------------------------------------
# per-patient figure (README block 5)
# ---------------------------------------------------------------------------

PP_MODELS = (("M0", "M0\nlast reading", INK, True), ("M2", "M2\ncuff only", MUTE, True),
             ("M4", "M4\n+ HR + pleth", ACCENT, True), ("M4-P1", "M4-P1\nplacebo", MUTE, False))


def per_patient_figure(site, path: Path, synthetic: bool):
    """One line per patient. In the S-dev panel the cerclage cases are drawn in their own colour, with a key
    (SPEC §6.7); the S-ob panel marks nothing (every S-ob case is a cesarean section)."""
    W, Hh = 1040, 400
    fig = new_fig(W, Hh)
    h = site.headline
    panels = (("S-dev", f"Non-cesarean cases (N = {h.N}, cross-validated)"),
              ("S-ob", f"Cesarean cases (n = {h.n}, frozen models)"))
    pw = (W - 40) / 2
    for j, (coh, title) in enumerate(panels):
        x0 = 20 + j * pw
        ftext(fig, x0 + 4, 10, title, DESKTOP.title, fontweight="semibold", color=INK)
        ax = add_axes(fig, x0 + 52, 44, pw - 80, Hh - 44 - 62)
        _style_axis(ax, DESKTOP)
        d = site.per_patient[coh]
        wide = d.pivot_table(index="caseid", columns="model", values="rmse").reindex(columns=[m for m, *_ in PP_MODELS])
        wide = wide.dropna()
        xs = np.arange(len(PP_MODELS))
        alpha = 0.22 if len(wide) > 60 else 0.55
        mark = [c for c in site.cerclage_sdev if c in wide.index] if coh == "S-dev" else []
        rest = wide.drop(index=mark)
        for _, r in rest.iterrows():
            ax.plot(xs, r.to_numpy(), color=MUTE, lw=px(0.7), alpha=alpha, zorder=1)
        ax.plot(np.repeat(xs[None, :], len(rest), 0).ravel(), rest.to_numpy().ravel(), ls="none", marker="o",
                ms=px(3), mfc=MUTE, mec="none", alpha=min(1, alpha * 1.6), zorder=2)
        for c in mark:
            ax.plot(xs, wide.loc[c].to_numpy(), color=CERCLAGE, lw=px(1.4), marker="o", ms=px(4), mfc=CERCLAGE,
                    mec="none", zorder=3)
        if mark:
            key = Line2D([], [], color=CERCLAGE, lw=px(1.4), marker="o", ms=px(4), mfc=CERCLAGE, mec="none")
            fig.legend([key], [f"cerclage ({len(mark)})"], loc="upper right",
                       bbox_to_anchor=((x0 + pw - 28) / W, 1 - 14 / Hh), frameon=False, fontsize=px(DESKTOP.tick),
                       handlelength=1.9, handletextpad=0.5, borderaxespad=0, borderpad=0, labelcolor=INK_2)
        means = wide.mean()
        for i, (m, lab, col, filled) in enumerate(PP_MODELS):
            ax.plot([i], [means[m]], ls="none", marker="o", ms=px(11), mfc=col if filled else BG, mec=col if
                    filled else INK_2, mew=px(1.6), zorder=4)
            ax.annotate(f"{means[m]:.2f}", xy=(i, means[m]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=px(12), color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=0.85))
        ax.set_xticks(xs, [lab for _, lab, *_ in PP_MODELS], fontsize=px(12))
        ax.set_xlim(-0.35, len(xs) - 0.5)
        ax.set_ylabel("patient RMSE, mmHg", fontsize=px(13), color=INK_2)
    credit(fig, DESKTOP)
    watermark(fig, synthetic, size=34)
    save_png(fig, path, synthetic)


# ---------------------------------------------------------------------------
# replay: frames -> MP4 + WebM (page) and GIF (README)
# ---------------------------------------------------------------------------

REPLAY_CSS = (830, 498)           # GIF size; the video is the same layout at 1040 x 624
VIDEO_W = 1040
FPS_VIDEO, FPS_GIF = 12, 10
HOLD_S, PLAY_S = 1.0, 18.0


def replay_frame_times(fps: int, window_s: tuple) -> list:
    """None = the complete frame. 1 s complete, 18 s of play over the window, 1 s complete (20 s)."""
    hold = [None] * round(HOLD_S * fps)
    n = round(PLAY_S * fps)
    w0, w1 = window_s
    play = [w0 + (w1 - w0) * (i + 1) / n for i in range(n)]
    return hold + play + hold


class ReplayRenderer:
    def __init__(self, s, synthetic: bool, dpi: float):
        self.s, self.dpi = s, dpi
        W, Hh = REPLAY_CSS
        fig = new_fig(W, Hh)
        ftext(fig, 12, 10, f"VitalDB case {s.caseid}", REPLAY.title, fontweight="semibold", color=INK)
        self.clock = ftext(fig, W - 12, 10, "", REPLAY.title, ha="right", color=INK_2, fontweight="semibold")
        case_key(fig, 12, 34, REPLAY)
        self.a, self.lim = build_case_axes(fig, s, 0, W, 62, Hh - 16, error_row=False, style=REPLAY)
        credit(fig, REPLAY)
        watermark(fig, synthetic, size=30)
        self.fig = fig

    def frame(self, now) -> np.ndarray:
        arts = draw_case_dynamic(self.a, self.s, self.lim, REPLAY, now)
        t = self.s.window_s[1] if now is None else now
        self.clock.set_text(f"{int(t // 60):02d}:{int(t % 60):02d} since record start")
        buf = io.BytesIO()
        self.fig.savefig(buf, dpi=self.dpi, facecolor=BG, format="png")
        for x in arts:
            x.remove()
        buf.seek(0)
        from PIL import Image
        return np.asarray(Image.open(buf).convert("RGB"))


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def replay_outputs(s, out_dir: Path, synthetic: bool, work_dir: Path, gif: bool = True) -> dict:
    """Writes replay-<id>.mp4, replay-<id>.webm and the poster replay-<id>.png (complete frame, 1040 px);
    with gif=True also replay.gif (830 px, 10 frames/s). Returns {file: (w, h)}."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = ffmpeg_exe()
    dpi_v = 100 * VIDEO_W / REPLAY_CSS[0]
    rv = ReplayRenderer(s, synthetic, dpi_v)
    times = replay_frame_times(FPS_VIDEO, s.window_s)
    res = {}
    with tempfile.TemporaryDirectory(dir=work_dir) as td:
        td = Path(td)
        cache = {}
        for i, now in enumerate(times):
            key = "full" if now is None else now
            if key not in cache:
                cache[key] = rv.frame(now)
            img = cache[key]
            h2 = img.shape[0] - img.shape[0] % 2
            Image.fromarray(img[:h2, : img.shape[1] - img.shape[1] % 2]).save(td / f"v{i:04d}.png")
        full = cache["full"]
        poster = out_dir / f"replay-{s.caseid}.png"
        write_png(Image.fromarray(full), poster, synthetic)
        res[poster.name] = (full.shape[1], full.shape[0])
        if exe is None:
            raise RuntimeError("ffmpeg not found: install it, or `uv pip install --python .venv/bin/python "
                               "imageio-ffmpeg`")
        vin = ["-framerate", str(FPS_VIDEO), "-i", str(td / "v%04d.png")]
        mp4 = out_dir / f"replay-{s.caseid}.mp4"
        _run([exe, "-y", "-loglevel", "error", *vin, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
              "-preset", "slow", "-tune", "animation", "-movflags", "+faststart", "-an", str(mp4)])
        webm = out_dir / f"replay-{s.caseid}.webm"
        _run([exe, "-y", "-loglevel", "error", *vin, "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "38",
              "-b:v", "0", "-row-mt", "1", "-an", str(webm)])
        res[mp4.name] = res[webm.name] = (full.shape[1] - full.shape[1] % 2, full.shape[0] - full.shape[0] % 2)
        if gif:
            g = out_dir / "replay.gif"
            W, Hh = REPLAY_CSS
            vf = (f"fps={FPS_GIF},scale={W}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:"
                  "stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle")
            _run([exe, "-y", "-loglevel", "error", *vin, "-vf", vf, "-loop", "0", str(g)])
            res[g.name] = (W, Hh)
    return res
