from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
LOGO_DIR = SCRIPT_DIR.parent / "logos"
CAPTION_GAP = 26
LOGO_TARGET_PX = 28

#          x       y      model                       retriever        pos      dx    dy
POINTS = [
    (23.61,  6.27, "Qwen3.5-4B",              "BM25",          "left",    40,   30),
    (28.92,  6.62, "Qwen3.5-9B",              "BM25",          "right",  -35,   30),
    (32.65,  6.27, "GPT-OSS-20B",             "BM25",          "right",  -30,  -30),
    (41.93,  5.78, "Qwen3.5-4B",              "Qwen3-Emb-8B",  "right",  -20,  -25),
    (46.14,  9.52, "Qwen3.5-9B",              "Qwen3-Emb-8B",  "right",   -8,   -8),
    (48.00,  6.00, "GPT-OSS-20B",             "Qwen3-Emb-8B",  "right",   -5,   -5),
    (48.07, 10.85, "Qwen3.5-4B",              "AgentIR",       "left",    30,   30),
    (54.94,  8.07, "Qwen3.5-9B",              "AgentIR",       "right",   -8,   -5),
    (62.89, 11.71, "Qwen3.5-35B",             "AgentIR",       "right",  -35,   30),
    (63.30, 10.00, "GPT-OSS-20B",             "AgentIR",       "left",     8,    3),
    (68.55,  2.65, "OpenResearcher-30B",      "AgentIR",       "left",    10,    0),
    (72.50,  3.68, "Qwen3.6-35B",             "AgentIR",       "right",  -15,   20),
    (78.40,  0.10, "GPT-OSS-120B",            "AgentIR",       "left",     5,   -5),
    (80.50,  3.84, "DS-V4-Flash-Max",         "AgentIR",       "right",   -8,    0),
    (81.70, -1.10, "Tongyi-DeepResearch-30B", "AgentIR",       "right",   -8,   -3),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Lato", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_BODY       = "#CFE3C4"
COL_BODY_EDGE  = "#3F6B3A"
COL_SWEET      = "#5CB85C"
COL_SWEET_LINE = "#1B4D1F"
COL_REGIME_L   = "#E3ECF4"
COL_REGIME_R   = "#FAEEDA"
COL_TEXT_DARK  = "#1F2937"
COL_ANN_LEFT   = "#1B4D6B"
COL_ANN_RIGHT  = "#8B4513"
COL_TITLE_LEFT  = "#4A7C99"
COL_TITLE_MID   = "#4E8050"
COL_TITLE_RIGHT = "#B07A4A"
COL_HILITE_LEFT  = "#B7CEE3"
COL_HILITE_MID   = "#63B95C"
COL_HILITE_RIGHT = "#E0B66D"

retriever_colors = {
    "BM25": "#1B6E8C",
    "Qwen3-Emb-8B": "#C0392B",
    "AgentIR": "#0D3D7A",
}

_RAW_LOGOS = {
    "openai": mpimg.imread(LOGO_DIR / "gpt.png"),
    "Qwen": mpimg.imread(LOGO_DIR / "tongyi.png"),
    "nvidia": mpimg.imread(LOGO_DIR / "nvidia.png"),
    "deepseek": mpimg.imread(LOGO_DIR / "deepseek.png"),
}

model_vendor = {
    "Qwen3.5-4B": "Qwen",
    "Qwen3.5-9B": "Qwen",
    "Qwen3.5-35B": "Qwen",
    "Qwen3.6-35B": "Qwen",
    "GPT-OSS-20B": "openai",
    "GPT-OSS-120B": "openai",
    "OpenResearcher-30B": "nvidia",
    "Tongyi-DeepResearch-30B": "Qwen",
    "DS-V4-Flash-Max": "deepseek",
}


def _zoom_for(img, target_px):
    h, w = img.shape[:2]
    return target_px / max(h, w)


VENDOR_ZOOM = {
    name: _zoom_for(img, LOGO_TARGET_PX)
    for name, img in _RAW_LOGOS.items()
}

silhouette_x = np.array([18, 30, 40, 42, 48, 58, 66, 71, 76, 84, 100])
silhouette_y = np.array([5.6, 6.2, 6.6, 6.9, 11.1, 12.9, 7.5, 3.9, 1.8, 0.25, 0.0])
x_grid = np.linspace(silhouette_x.min(), silhouette_x.max(), 1500)
spline = PchipInterpolator(silhouette_x, silhouette_y)
y_grid = np.clip(spline(x_grid), 0, None)
y_grid = gaussian_filter1d(y_grid, sigma=2)

peak_idx = int(np.argmax(y_grid))
SWEET_THRESHOLD = 6.8
LEFT_VL = 40.0
RIGHT_VL = 70.0
xmax = 100.0
xmin = 18.0
ymin, ymax = -1.5, 16.0

above = y_grid > SWEET_THRESHOLD
diffs = np.diff(above.astype(int))
starts = list(np.where(diffs == 1)[0] + 1)
ends = list(np.where(diffs == -1)[0] + 1)
if above[0]:
    starts = [0] + starts
if above[-1]:
    ends = ends + [len(above)]
runs = sorted(zip(starts, ends), key=lambda r: r[1] - r[0], reverse=True)
big_start, big_end = runs[0]
bulge_left = float(x_grid[big_start])
bulge_right = float(x_grid[big_end - 1])

_old_xmax = 90.0
_old_xmin = bulge_left - (_old_xmax - bulge_right)
_old_xspan = _old_xmax - _old_xmin
_new_xspan = xmax - xmin
_old_w, _old_h = 10.0, 7.18
fig_w = _old_w * (_new_xspan / _old_xspan)
fig_h = _old_h


def draw_logo_marker(ax, x, y, vendor_key):
    im = OffsetImage(
        _RAW_LOGOS[vendor_key],
        zoom=VENDOR_ZOOM[vendor_key],
        interpolation="bilinear",
    )
    ab = AnnotationBbox(
        im,
        (x, y),
        frameon=False,
        pad=0,
        box_alignment=(0.5, 0.5),
        zorder=5,
    )
    ax.add_artist(ab)


def _anchor_for(pos):
    g = CAPTION_GAP
    if pos == "above":
        return (0, g, "center", "bottom")
    if pos == "below":
        return (0, -g, "center", "top")
    if pos == "left":
        return (-g, 0, "right", "center")
    if pos == "right":
        return (g, 0, "left", "center")
    raise ValueError(f"unknown pos: {pos}")


def add_highlight(ax, highlight):
    if highlight is None:
        return
    name, strength = highlight
    strength = float(strength)

    if name == "bottleneck":
        rect = Rectangle(
            (xmin, ymin),
            LEFT_VL - xmin,
            ymax - ymin,
            facecolor=COL_HILITE_LEFT,
            edgecolor=COL_TITLE_LEFT,
            lw=1.8 + 1.8 * strength,
            alpha=0.10 + 0.42 * strength,
            zorder=0.9,
        )
        ax.add_patch(rect)
    elif name == "saturated":
        # Start just inside the saturated regime to avoid anti-aliased spillover
        # around the 70% divider in animated GIF playback.
        sat_start = RIGHT_VL + 0.6
        sat_xs = np.linspace(sat_start, xmax, 400)
        sat_upper = np.clip(100.0 - sat_xs, ymin, ymax)
        ax.fill_between(
            sat_xs,
            ymin,
            sat_upper,
            color=COL_HILITE_RIGHT,
            alpha=0.16 + 0.55 * strength,
            edgecolor="none",
            zorder=0.9,
        )
    elif name == "sweet":
        sweet_mask = np.r_[
            np.zeros(big_start, bool),
            np.ones(big_end - big_start, bool),
            np.zeros(len(y_grid) - big_end, bool),
        ]
        ax.fill_between(
            x_grid,
            SWEET_THRESHOLD,
            y_grid,
            where=sweet_mask,
            color=COL_HILITE_MID,
            alpha=0.18 + 0.58 * strength,
            edgecolor="none",
            zorder=4.2,
        )
        ax.plot(
            x_grid[big_start:big_end],
            y_grid[big_start:big_end],
            color=COL_TITLE_MID,
            lw=2.4 + 2.2 * strength,
            alpha=0.55 + 0.40 * strength,
            zorder=4.3,
        )
    else:
        raise ValueError(f"unknown highlight: {name}")


def highlight_strength(highlight, name):
    if highlight is None:
        return 0.0
    active_name, strength = highlight
    return float(strength) if active_name == name else 0.0


def title_style(highlight, name, base_color, active_color):
    strength = highlight_strength(highlight, name)
    if strength <= 0:
        return {
            "color": base_color,
            "fontsize": 20,
            "fontweight": "semibold",
            "zorder": 4,
            "bbox": None,
        }
    return {
        "color": active_color,
        "fontsize": 20 + 2.0 * strength,
        "fontweight": "bold",
        "zorder": 9,
        "bbox": None,
    }


def render_frame(highlight=None):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.axvspan(xmin, LEFT_VL, color=COL_REGIME_L, alpha=0.6, zorder=0)
    ax.axvspan(LEFT_VL, RIGHT_VL, color="#FAFAF7", alpha=1.0, zorder=0)
    ax.axvspan(RIGHT_VL, xmax, color=COL_REGIME_R, alpha=0.6, zorder=0)

    _ub_xs = np.linspace(100.0 - ymax, xmax, 300)
    ax.fill_between(_ub_xs, 100.0 - _ub_xs, ymax, color="white", alpha=1.0, zorder=0.5)

    add_highlight(ax, highlight)

    ax.fill_between(x_grid, ymin, y_grid, color=COL_BODY, alpha=0.55, edgecolor="none", zorder=1)
    ax.fill_between(
        x_grid,
        SWEET_THRESHOLD,
        y_grid,
        where=np.r_[
            np.zeros(big_start, bool),
            np.ones(big_end - big_start, bool),
            np.zeros(len(y_grid) - big_end, bool),
        ],
        color=COL_SWEET,
        alpha=0.45,
        edgecolor="none",
        zorder=2,
    )
    ax.plot(
        [bulge_left, bulge_right],
        [SWEET_THRESHOLD, SWEET_THRESHOLD],
        color=COL_SWEET_LINE,
        lw=1.6,
        ls=(0, (5, 3)),
        alpha=0.9,
        zorder=2.5,
    )
    ax.plot(x_grid, y_grid, color=COL_BODY_EDGE, lw=2.0, alpha=0.95, zorder=3)

    ax.axvline(LEFT_VL, color=COL_ANN_LEFT, lw=1.4, ls=(0, (5, 4)), alpha=0.7, zorder=0.7)
    ax.axvline(RIGHT_VL, color=COL_ANN_RIGHT, lw=1.4, ls=(0, (5, 4)), alpha=0.7, zorder=0.7)

    line_half = 7
    for x, y, model_key, retr, pos, dx, dy in POINTS:
        vendor = model_vendor[model_key]
        y_logo = y if y >= 0.6 else y + 0.6
        draw_logo_marker(ax, x, y_logo, vendor)

        anc_dx, anc_dy, ha, va = _anchor_for(pos)
        base_dx = anc_dx + dx
        base_dy = anc_dy + dy
        retr_color = retriever_colors[retr]

        ax.annotate(
            model_key,
            xy=(x, y_logo),
            xytext=(base_dx, base_dy + line_half),
            textcoords="offset points",
            fontsize=10.5,
            color=COL_TEXT_DARK,
            ha=ha,
            va=va,
            fontweight="medium",
            zorder=7,
        )
        ax.annotate(
            f"+ {retr}",
            xy=(x, y_logo),
            xytext=(base_dx, base_dy - line_half),
            textcoords="offset points",
            fontsize=10,
            color=retr_color,
            ha=ha,
            va=va,
            fontweight="bold",
            zorder=7,
        )

    left_title = title_style(highlight, "bottleneck", COL_TITLE_LEFT, COL_ANN_LEFT)
    mid_title = title_style(highlight, "sweet", COL_TITLE_MID, COL_SWEET_LINE)
    right_title = title_style(highlight, "saturated", COL_TITLE_RIGHT, COL_ANN_RIGHT)

    ax.text(
        (xmin + LEFT_VL) / 2,
        ymax - 0.3,
        "Retriever\nBottleneck",
        ha="center",
        va="top",
        **left_title,
    )
    ax.text(
        (LEFT_VL + RIGHT_VL) / 2,
        ymax - 0.3,
        "CM Matters Most",
        ha="center",
        va="top",
        **mid_title,
    )
    ax.text(
        (RIGHT_VL + (100.0 - ymax)) / 2 + 0.5,
        ymax - 0.3,
        "Model\nSaturated",
        ha="center",
        va="top",
        **right_title,
    )

    sweet_label_x, sweet_label_y = 46.5, 13.8
    ax.text(
        sweet_label_x,
        sweet_label_y,
        "CM Sweet Spot",
        ha="center",
        va="center",
        fontsize=16,
        color=COL_SWEET_LINE,
        fontweight="bold",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor=COL_SWEET_LINE, linewidth=1.2, alpha=0.95),
    )
    sweet_arrow_target_x = 55.5
    sweet_arrow_target_y = float(spline(sweet_arrow_target_x)) - 0.8
    ax.annotate(
        "",
        xy=(sweet_arrow_target_x, sweet_arrow_target_y),
        xytext=(sweet_label_x + 4.0, sweet_label_y - 0.8),
        arrowprops=dict(arrowstyle="-|>", color=COL_SWEET_LINE, lw=1.5, connectionstyle="arc3,rad=0"),
        zorder=7,
    )

    ax.text(
        (xmin + bulge_left) / 2,
        2.6,
        "Bounded by\nnoises retrieved",
        fontsize=16,
        color=COL_ANN_LEFT,
        ha="center",
        va="center",
        style="italic",
        fontweight="bold",
        zorder=6,
    )

    text_x, text_y = 87.0, 2.25
    ax.text(
        text_x,
        text_y,
        "Pruning evicts useful signal,\nprolonging trajectories",
        fontsize=12,
        color=COL_ANN_RIGHT,
        ha="center",
        va="center",
        style="italic",
        fontweight="bold",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75),
    )

    target_x = 85.0
    target_y = float(spline(target_x)) + 0.05
    ax.add_patch(
        FancyArrowPatch(
            (89.0, text_y - 1),
            (target_x, target_y + 0.2),
            arrowstyle="-|>",
            mutation_scale=14,
            connectionstyle="arc3,rad=-0.05",
            color=COL_ANN_RIGHT,
            lw=1.5,
            alpha=0.85,
            zorder=5.5,
        )
    )

    ub_x0, ub_x1 = 100.0 - ymax, xmax
    ub_y0, ub_y1 = float(ymax), 0.0
    ax.plot([ub_x0, ub_x1], [ub_y0, ub_y1], color=COL_ANN_RIGHT, lw=1.6, ls=(0, (6, 3)), alpha=0.75, zorder=3)
    ax.text(
        93.0,
        4.5,
        "Improvement Upper Bound",
        fontsize=16,
        color=COL_ANN_RIGHT,
        ha="center",
        va="bottom",
        style="italic",
        fontweight="bold",
        rotation=-69,
        zorder=6,
        transform=ax.transData,
    )

    ax.set_xlabel("Accuracy without Context Management (%)", fontweight="medium", labelpad=8, fontsize=22)
    ax.set_ylabel(r"$\Delta$ Accuracy from Context Management (%)", fontweight="medium", labelpad=8, fontsize=20)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.tick_params(colors="#333")
    plt.tight_layout()
    return fig


def figure_to_pil(fig, dpi=160):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    rgba = Image.open(buf).convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return rgb


def save_static():
    fig = render_frame()
    fig.savefig(SCRIPT_DIR / "teaser.png", dpi=400, bbox_inches="tight")
    fig.savefig(SCRIPT_DIR / "teaser.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SCRIPT_DIR / 'teaser.png'}")
    print(f"Saved: {SCRIPT_DIR / 'teaser.pdf'}")


def save_gif():
    fade = [0.10, 0.24, 0.40, 0.58, 0.76, 0.92, 1.00, 1.00, 1.00, 0.92, 0.76, 0.58, 0.40, 0.24, 0.10]
    sequence = [
        (None, 800),
        *[(("bottleneck", v), 150) for v in fade],
        (None, 360),
        *[(("saturated", v), 150) for v in fade],
        (None, 360),
        *[(("sweet", v), 160) for v in fade],
        (None, 1000),
    ]
    frames = [figure_to_pil(render_frame(highlight), dpi=155) for highlight, _ in sequence]
    durations = [duration for _, duration in sequence]
    out = SCRIPT_DIR / "teaser.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=[2] * len(frames),
        optimize=False,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    save_static()
    save_gif()
    print(f"x range: [{xmin:.2f}, {xmax}]   y range: [{ymin}, {ymax}]")
    print(f"figsize: ({fig_w:.2f}, {fig_h})")
