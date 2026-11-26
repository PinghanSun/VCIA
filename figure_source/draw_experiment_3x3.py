"""Render main Fig. 3 from response-model CSV inputs.
The CSV key PCOC is the archived identifier of the VCIA policy profile.
"""

from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_source" / "experiment_3x3"
STEM = "VCIA_Experiment_3x3"
WIDTH, HEIGHT = 346, 289
PANEL_WIDTH, PANEL_HEIGHT = 74, 50
LEGEND_FONT_SIZE = 6.5
X = (30, 137, 245)
Y = (225, 129, 33)
DATASETS = ("PROTEINS", "NCI1", "ENZYMES", "IMDB-BINARY", "MUTAG")
SHORT_DATASETS = ("PROT.", "NCI1", "ENZ.", "IMDB-B", "MUTAG")
ACC_TICKS = ((55, 65, 75), (55, 65, 75), (30, 40, 50),
             (55, 65, 75), (60, 70, 80))
ACC_LIMITS = ((50, 80), (50, 80), (25, 51), (50, 80), (59, 83))
POLICIES = ("PCOC", "JRank", "Unscreened", "CensusAudit")
POLICY_NAMES = ("VCIA", "J-Rank", "Unscreened", "Census-Audit")
POLICY_COLOR = ("#6E4B96", "#3477AD", "#A47423", "#358B7A")
LINE_STYLES = ("solid", (0, (3.4, 1.5)), (0, (1, 1.3)), (0, (4, 1.3, 1, 1.3)))
VARIANTS = ("StaleNorm", "ReuseV", "NonAtomic")
SEVERITIES = ("A0", "A1", "A2", "A3")
PERTURBATION_MULTIPLIERS = ("0", "0.65", "1.0", "1.3")
INK = "#000000"
TICK_COLOR = "#202b38"
GRID = "#e4e9ef"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(DictReader(source))


def load_trajectories(input_dir):
    curves = defaultdict(list)
    for row in read_csv(input_dir / "trajectories.csv"):
        assert row["attack"] == "A3"
        curves[row["dataset"], row["method"]].append(
            (int(row["round"]), float(row["mean_ACC_pct"]), float(row["SD_ACC_pct"]))
        )
    assert set(curves) == {(d, p) for d in DATASETS for p in POLICIES}
    for values in curves.values():
        values.sort()
        assert [v[0] for v in values] == list(range(1, 21))
    return curves


def load_intervals(path, series, key, estimate, low, high, *, severity=None):
    by_key = {}
    for row in read_csv(path):
        if severity is not None and row["attack"] != severity:
            continue
        pair = row["dataset"], row[key]
        assert pair not in by_key
        value, left, right = (float(row[col]) for col in (estimate, low, high))
        assert left <= value <= right
        by_key[pair] = value, left, right
    assert set(by_key) == {(d, s) for d in DATASETS for s in series}
    return by_key


def style_axis(ax, *, categorical=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", pad=2)
    if categorical:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)


def trajectory(ax, index, curves):
    dataset = DATASETS[index]
    ax.axvspan(5, 12, color="#edf1f5", lw=0, zorder=0)
    for idx, (policy, color, linestyle) in enumerate(zip(POLICIES, POLICY_COLOR, LINE_STYLES)):
        rounds, mean, sd = zip(*curves[dataset, policy])
        lower = [m - s for m, s in zip(mean, sd)]
        upper = [m + s for m, s in zip(mean, sd)]
        assert min(lower) >= ACC_LIMITS[index][0] and max(upper) <= ACC_LIMITS[index][1]
        ax.fill_between(rounds, lower, upper, facecolor=color, alpha=.095, lw=0, zorder=1)
        ax.plot(rounds, mean, color=color, ls=linestyle, lw=.89 if idx == 0 else .72,
                zorder=4 if idx == 0 else 3)
    ax.set(xlim=(1, 20), xticks=[1, 5, 10, 15, 20], ylim=ACC_LIMITS[index],
           yticks=ACC_TICKS[index], xlabel="Round", ylabel="ACC (%)")
    ax.grid(axis="y", color=GRID, lw=.4)
    style_axis(ax)


def framed_legend(owner, handles, *, loc, anchor, handlelength):
    legend = owner.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncols=len(handles),
                          frameon=True, fancybox=False, framealpha=1, facecolor="white",
                          edgecolor="#9ba7b4", fontsize=LEGEND_FONT_SIZE, handlelength=handlelength,
                          handletextpad=.25, columnspacing=.42, borderaxespad=0, borderpad=.26)
    legend.get_frame().set_linewidth(.55)
    return legend


def panel_key(ax, handles):
    legend = framed_legend(ax, handles, loc="lower center", anchor=(.5, 1.045), handlelength=.90)
    legend.set_in_layout(False)


def forest(ax, values, keys, names, colors, markers, limits, ticks, label):
    for dataset_index in (0, 2, 4):
        ax.axhspan(dataset_index - .50, dataset_index + .50, color="#f1f4f7", lw=0)
    ax.axvline(0, color="#667382", lw=.55, zorder=2)
    handles = []
    spacing = .70 / (len(keys) - 1)
    for index, (key, name, color, marker) in enumerate(zip(keys, names, colors, markers)):
        offset = -.35 + index * spacing
        for dataset_index, dataset in enumerate(DATASETS):
            value, low, high = values[dataset, key]
            assert limits[0] <= low <= high <= limits[1]
            ax.errorbar(value, dataset_index + offset, xerr=[[value - low], [high - value]],
                        fmt=marker, color=color, ms=2, elinewidth=.67, capsize=1.05,
                        capthick=.57, mew=.15, mec=color, zorder=3)
        handles.append(Line2D([], [], color=color, lw=.72, marker=marker, ms=2.4, label=name))
    ax.set(xlim=limits, xticks=ticks, ylim=(4.53, -.53), yticks=range(5), xlabel=label)
    ax.set_yticklabels(SHORT_DATASETS)
    ax.grid(axis="x", color=GRID, lw=.4)
    style_axis(ax, categorical=True)
    panel_key(ax, handles)


def recovery(ax, input_dir):
    rows = {r["method"]: r for r in read_csv(input_dir / "recovery.csv")}
    methods = (*POLICIES, "NoVerify")
    assert set(rows) == set(methods)
    statuses = ("not_disrupted", "recovered", "censored_at60")
    names = ("Not disrupted", "Recovered", "Censored")
    colors = ("#a1c7df", "#6eae8e", "#bcc4d0")
    for i, method in enumerate(methods):
        counts = [int(rows[method][status]) for status in statuses]
        assert sum(counts) == 225
        start = 0
        for status, count, color in zip(statuses, counts, colors):
            ax.barh(i, count, left=start, height=.57, color=color, edgecolor="white", lw=.45)
            if 0 < count < 21 and status == "censored_at60":
                ax.annotate(str(count), xy=(start + count, i), xytext=(2, 0), textcoords="offset points",
                            fontsize=6.2, ha="left", va="center")
            elif 0 < count < 21:
                ax.annotate(str(count), xy=(start + count / 2, i), xytext=(-1.4, 0),
                            textcoords="offset points", fontsize=5.7, ha="center", va="center")
            elif count >= 21:
                ax.text(start + count / 2, i, str(count), ha="center", va="center", fontsize=6.2)
            start += count
    ax.set(xlim=(0, 243), xticks=[0, 75, 150, 225], ylim=(4.53, -.53),
           yticks=range(5), xlabel="Trajectories")
    ax.set_yticklabels(("VCIA", "J-Rank", "Unscr.", "C-Audit", "No-Ver."))
    ax.grid(axis="x", color=GRID, lw=.4)
    style_axis(ax, categorical=True)
    ax.tick_params(axis="y", pad=4)
    handles = [Patch(facecolor=color, label=name) for color, name in zip(colors, names)]
    panel_key(ax, handles)


def verify_bounds(fig, axes, captions, legend):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    side_margin = renderer.points_to_pixels(2)
    caption_clearance = renderer.points_to_pixels(4)
    legend_panels = [(i, ax.get_legend()) for i, ax in enumerate(axes) if ax.get_legend() is not None]
    panel_legends = [key for _, key in legend_panels]
    for artist in [*axes, *captions, *panel_legends, legend]:
        bounds = artist.get_tightbbox(renderer)
        assert side_margin < bounds.x0 < bounds.x1 < frame.width - side_margin, (artist, bounds)
        assert 0 < bounds.y0 < bounds.y1 < frame.height, (artist, bounds)
    for i, key in enumerate(panel_legends):
        bounds = key.get_tightbbox(renderer)
        others = [*axes, *captions, *panel_legends[i + 1:], legend]
        assert not any(bounds.overlaps(other.get_tightbbox(renderer)) for other in others), key
    for panel, key in legend_panels:
        bounds = key.get_window_extent(renderer)
        ax = axes[panel]
        assert abs((bounds.x0 + bounds.x1) / 2 - (ax.bbox.x0 + ax.bbox.x1) / 2) < .01
        assert bounds.y0 >= ax.bbox.y1 + side_margin
        assert captions[panel - 3].get_window_extent(renderer).y0 >= bounds.y1 + caption_clearance
    for row in range(3):
        boxes = [ax.get_tightbbox(renderer) for ax in axes[row * 3: row * 3 + 3]]
        assert all(left.x1 + 4 < right.x0 for left, right in zip(boxes, boxes[1:])), boxes
    for i, caption in enumerate(captions):
        bounds = caption.get_tightbbox(renderer)
        assert bounds.y1 < axes[i].get_tightbbox(renderer).y0, (caption.get_text(), bounds)
        if i < 6:
            assert bounds.y0 > axes[i + 3].get_tightbbox(renderer).y1 + 2
    first_row_top = max(ax.get_tightbbox(renderer).y1 for ax in axes[:3])
    assert legend.get_tightbbox(renderer).y0 >= first_row_top + renderer.points_to_pixels(1.5)
    recovery_axis = axes[-1]
    count_boxes = [text.get_window_extent(renderer) for text in recovery_axis.texts]
    tick_boxes = [tick.get_window_extent(renderer) for tick in recovery_axis.get_yticklabels()]
    for i, box in enumerate(count_boxes):
        assert not any(box.overlaps(other) for other in count_boxes[i + 1:] + tick_boxes), box


def main(input_dir):
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 7.4, "pdf.fonttype": 42,
        "axes.linewidth": .48, "axes.edgecolor": "#778593", "axes.labelcolor": INK,
        "axes.labelsize": 7.4, "axes.labelpad": 1.5, "text.color": INK,
        "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.color": TICK_COLOR, "ytick.color": TICK_COLOR,
        "xtick.major.width": .45, "ytick.major.width": .45,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.unicode_minus": True,
    })
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=144)
    axes = [fig.add_axes([x / WIDTH, y / HEIGHT, PANEL_WIDTH / WIDTH, PANEL_HEIGHT / HEIGHT])
            for y in Y for x in X]
    curves = load_trajectories(input_dir)
    for index, axis in enumerate(axes[:5]):
        trajectory(axis, index, curves)
    policy = load_intervals(input_dir / "policy_contrasts.csv", POLICIES[1:], "comparator", "delta_ACC_pct",
                            "ACC_CI95_low", "ACC_CI95_high", severity="A3")
    mechanism = load_intervals(input_dir / "mechanism_contrasts.csv", VARIANTS, "comparator", "delta_ACC_pct",
                               "ACC_CI95_low", "ACC_CI95_high", severity="A3")
    interaction = load_intervals(input_dir / "gx_interactions.csv", SEVERITIES, "attack", "interaction_pp",
                                 "CI95_low", "CI95_high")
    forest(axes[5], policy, POLICIES[1:], POLICY_NAMES[1:],
           POLICY_COLOR[1:], ("s", "^", "D"), (-5.7, 12.4), [-5, 0, 5, 10], "ΔACC (pp)")
    forest(axes[6], mechanism, VARIANTS, VARIANTS, ("#3477ad", "#358b7a", "#a55264"),
           ("s", "^", "D"), (-5.5, 6.5), [-4, 0, 4], "ΔACC (pp)")
    forest(axes[7], interaction, SEVERITIES, PERTURBATION_MULTIPLIERS, ("#718397", "#4b94ba", "#346da7", "#343c70"),
           ("o", "s", "^", "D"), (-15.8, 12.5), [-15, -10, -5, 0, 5, 10], "Interaction (pp)")
    recovery(axes[8], input_dir)
    captions = []
    for i, title in enumerate((*DATASETS, "Policy contrasts", "Mechanism variants",
                               "G/X interaction", "Status at round 60")):
        row, col = divmod(i, 3)
        captions.append(fig.text((X[col] + PANEL_WIDTH / 2) / WIDTH, (Y[row] - 23) / HEIGHT,
                                 f"({chr(97 + i)}) {title}", ha="center", va="top",
                                 fontsize=7.65, fontweight="bold"))
    handles = [Line2D([], [], color=color, ls=style, lw=1.05, label=name)
               for name, color, style in zip(POLICY_NAMES, POLICY_COLOR, LINE_STYLES)]
    legend = framed_legend(fig, handles, loc="upper center", anchor=(.5, 1 - 2 / HEIGHT),
                           handlelength=2.1)
    verify_bounds(fig, axes, captions, legend)
    output = ROOT / "runs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{STEM}.pdf"
    fig.savefig(pdf)
    fig.savefig(output / f"{STEM}_Preview.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA, help="Directory containing the five plot CSVs")
    args = parser.parse_args()
    required = ("trajectories.csv", "policy_contrasts.csv", "mechanism_contrasts.csv",
                "gx_interactions.csv", "recovery.csv")
    missing = [name for name in required if not (args.input_dir / name).is_file()]
    if missing:
        parser.error(f"Missing plot inputs in {args.input_dir}: {', '.join(missing)}. Use --input-dir.")
    main(args.input_dir)
