"""Draw main Fig. 4 from a round-20 response-model CSV. Effects are in percentage points.
"""

from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from math import isclose
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
STEM = "VCIA_Sensitivity_Annotated"
DATASETS = {
    "PROTEINS": "PROT.",
    "NCI1": "NCI1",
    "ENZYMES": "ENZ.",
    "IMDB-BINARY": "IMDB-B",
    "MUTAG": "MUTAG",
}
FACTORS = (
    ("alpha", "Partition"),
    ("malicious_fraction", "Perturbation fraction"),
    ("audit_size", "Audit size"),
    ("candidate_budget", "Candidate budget"),
    ("threshold", "Threshold"),
    ("noise_scale", "Noise multiplier"),
    ("local_E", "Local epochs"),
)
WIDTH, HEIGHT = 346, 188  # PDF points at the paper's text width
GRID_LEFT, GRID_RIGHT, GUTTER = 32, 308.5, 8
ROW_HEIGHT = 64
ROW_BOTTOMS = (110, 22)
SCALE_LEFT, SCALE_WIDTH = 318, 6.5
LABEL_FONT_SIZE, CAPTION_FONT_SIZE, LEGEND_FONT_SIZE = 7.1, 7.65, 6.5
INK = "#000000"
TICK_COLOR = "#202b38"
OUTLINE = "#68788a"


def load_panels(input_path):
    effects = {}
    levels = defaultdict(set)
    defaults = defaultdict(set)
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        for row in DictReader(source):
            factor, dataset = row["parameter"], row["dataset"]
            level = float(row["value"])
            default = float(row["default_value"])
            effect = float(row["delta_ACC_pp"])
            key = (factor, dataset, level)
            assert key not in effects
            assert (row["attack"], row["round"], row["nested_records"]) == ("A2", "20", "15")
            assert isclose(
                effect,
                float(row["round20_mean_ACC_pct"]) - float(row["within_sweep_default_ACC_pct"]),
                abs_tol=1e-10,
            )
            assert abs(effect) <= 4
            assert level != default or effect == 0
            effects[key] = effect
            levels[factor].add(level)
            defaults[factor].add(default)

    panels = []
    for index, (factor, name) in enumerate(FACTORS):
        settings = sorted(levels[factor])
        (default,) = defaults[factor]
        matrix = [[effects[factor, dataset, setting] for setting in settings] for dataset in DATASETS]
        panels.append((f"({chr(97 + index)}) {name}", settings, default, matrix))
    assert len(effects) == sum(len(panel[1]) * len(DATASETS) for panel in panels) == 130
    return panels


def label_color(background):
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in background[:3]]
    lightness = sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
    black_contrast = (lightness + 0.05) / 0.05
    return "white" if black_contrast < 5.6 else INK


def effect_label(value):
    return "0.00" if round(value, 2) == 0 else f"{value:+.2f}".replace("-", "−")


def draw(panels):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "text.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.color": TICK_COLOR,
        "ytick.color": TICK_COLOR,
        "pdf.fonttype": 42,
    })
    norm = Normalize(vmin=-4, vmax=4)
    original_colors = plt.get_cmap("RdBu")
    cmap = LinearSegmentedColormap.from_list("delta_acc", original_colors(np.linspace(.07, .93, 257)))
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=300)
    annotations = []
    meshes = []
    guides = []
    captions = []
    rows = []

    for bottom, row in zip(ROW_BOTTOMS, (panels[:3], panels[3:])):
        row_axes = []
        rows.append(row_axes)
        grid_width = GRID_RIGHT - GRID_LEFT - (len(row) - 1) * GUTTER
        cell_width = grid_width / sum(len(panel[1]) for panel in row)
        left = GRID_LEFT
        for col, (title, levels, default, matrix) in enumerate(row):
            width = len(levels) * cell_width
            ax = fig.add_axes((left / WIDTH, bottom / HEIGHT, width / WIDTH, ROW_HEIGHT / HEIGHT))
            row_axes.append(ax)
            caption = ax.text(.5, -12.75 / ROW_HEIGHT, title, transform=ax.transAxes,
                              ha="center", va="top", fontsize=CAPTION_FONT_SIZE, fontweight="bold")
            captions.append((caption, ax))
            guides.append(caption)
            mesh = ax.pcolormesh(np.arange(len(levels) + 1), np.arange(len(DATASETS) + 1), matrix,
                                 cmap=cmap, norm=norm, edgecolors="#cfd7e0", linewidth=0.34)
            meshes.append(mesh)
            ax.set_xlim(0, len(levels))
            ax.set_ylim(len(DATASETS), 0)
            ax.tick_params(axis="both", length=0, pad=2, labelsize=LABEL_FONT_SIZE)
            ax.set_yticks(np.arange(len(DATASETS)) + .5, DATASETS.values())
            ax.tick_params(axis="y", labelleft=col == 0)
            ax.set_xticks(np.arange(len(levels)) + .5,
                          [f"{level:g}" + ("*" if level == default else "") for level in levels])
            ax.get_xticklabels()[levels.index(default)].set_fontweight("bold")
            guides.extend(ax.get_xticklabels() + ax.get_yticklabels())
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(Rectangle((levels.index(default), 0), 1, len(DATASETS),
                                   fill=False, edgecolor=OUTLINE, linewidth=.66, clip_on=False))
            for i, values in enumerate(matrix):
                for j, value in enumerate(values):
                    label = ax.text(j + .5, i + .5, effect_label(value), ha="center", va="center",
                                    fontsize=LABEL_FONT_SIZE, color=label_color(cmap(norm(value))))
                    annotations.append((label, ax, i, j))
            left += width + GUTTER

    grid_top = ROW_BOTTOMS[0] + ROW_HEIGHT
    legend = fig.legend([Line2D([], [], linestyle="none")], ["* Default setting"],
                        loc="lower center", bbox_to_anchor=(.5, (grid_top + 2.3) / HEIGHT),
                        fontsize=LEGEND_FONT_SIZE, handlelength=0, handletextpad=0, borderpad=.22,
                        borderaxespad=0, frameon=True, framealpha=1, edgecolor="#9ba7b4",
                        facecolor="white", fancybox=False)
    legend.get_frame().set_linewidth(.50)

    grid_height = grid_top - ROW_BOTTOMS[1]
    scale = fig.add_axes((SCALE_LEFT / WIDTH, ROW_BOTTOMS[1] / HEIGHT,
                          SCALE_WIDTH / WIDTH, grid_height / HEIGHT))
    colorbar = fig.colorbar(meshes[0], cax=scale, ticks=[-4, -2, 0, 2, 4])
    colorbar.set_ticklabels(["−4", "−2", "0", "+2", "+4"])
    colorbar.outline.set_linewidth(.5)
    colorbar.outline.set_edgecolor(OUTLINE)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.solids.set_linewidth(.3)
    colorbar.ax.tick_params(length=2, width=.45, pad=2, labelsize=LABEL_FONT_SIZE)
    scale_title = scale.set_title("ΔACC (pp)", pad=5.5, fontsize=7.4)
    guides.extend([scale_title, *scale.get_yticklabels()])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(annotations) == 130
    for label, ax, i, j in annotations:
        left, upper = ax.transData.transform((j, i))
        right, lower = ax.transData.transform((j + 1, i + 1))
        bbox = label.get_window_extent(renderer)
        assert left < bbox.x0 < bbox.x1 < right, label.get_text()
        assert lower < bbox.y0 < bbox.y1 < upper
        assert isclose((bbox.x0 + bbox.x1) / 2, (left + right) / 2, abs_tol=.01)
        assert isclose((bbox.y0 + bbox.y1) / 2, (lower + upper) / 2, abs_tol=.01)

    for caption, ax in captions:
        bottom_of_ticks = min(tick.get_window_extent(renderer).y0 for tick in ax.get_xticklabels())
        caption_box = caption.get_window_extent(renderer)
        assert caption_box.y1 + renderer.points_to_pixels(2) <= bottom_of_ticks, caption.get_text()
        if ax in rows[0]:
            next_row_top = rows[1][0].bbox.y1
            assert caption_box.y0 >= next_row_top + renderer.points_to_pixels(4), caption.get_text()

    legend_box = legend.get_window_extent(renderer)
    assert isclose((legend_box.x0 + legend_box.x1) / 2, fig.bbox.width / 2, abs_tol=.01)
    assert legend_box.y0 >= rows[0][0].bbox.y1 + renderer.points_to_pixels(2)
    guide_boxes = [(artist.get_text(), artist.get_window_extent(renderer)) for artist in guides]
    guide_boxes.append(("Default legend", legend_box))
    for i, (name, box) in enumerate(guide_boxes):
        assert 0 < box.x0 < box.x1 < fig.bbox.width, name
        assert 0 < box.y0 < box.y1 < fig.bbox.height, name
        for other_name, other_box in guide_boxes[i + 1:]:
            assert not box.overlaps(other_box), (name, other_name)

    output = HERE.parent / "runs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    main_pdf = output / f"{STEM}.pdf"
    fig.savefig(main_pdf)
    fig.savefig(output / "VCIA_Sensitivity_Preview.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "Sensitivity_Delta_ACC.csv",
                        help="Path to the one-factor sensitivity CSV")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Plot CSV missing: {args.input}. Supply it with --input.")
    draw(load_panels(args.input))
    print("130 centered labels, 7 panels, and the shared vertical color scale verified.")
