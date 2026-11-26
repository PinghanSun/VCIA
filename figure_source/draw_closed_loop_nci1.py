"""Build the experimental figures from the numeric publication inputs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "runs" / "figures"
DATA = json.loads((HERE / "closed_loop_nci1_plot_data.json").read_text(encoding="utf-8"))
SERIF = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman" if "Times New Roman" in SERIF else "DejaVu Serif"],
    "font.size": 9,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

BLUE = "#245c89"
ORANGE = "#a65227"
GREEN = "#347b68"


def draw_closed_loop():
    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.19), sharey=True)
    contrasts = ["No-Composition", "Reassignment only", "Drop both", "J-Rank-Feasible"]
    datasets = {"PROTEINS": (BLUE, "o", 0.13), "ENZYMES": (ORANGE, "s", -0.13)}
    for axis, metric, label in zip(
        axes, ["replacements", "nmi"], ["(a) Replacement difference", "(b) Held-out NMI difference (pp)"]
    ):
        axis.axvline(0, color="0.55", lw=0.8, ls="--", zorder=0)
        for row, contrast in enumerate(contrasts[::-1]):
            for dataset, (color, marker, offset) in datasets.items():
                values = DATA["closed_loop"][contrast][dataset][metric]
                axis.scatter(values, [row + offset] * len(values), s=18, c=color, marker=marker,
                             alpha=0.76, linewidths=0, label=dataset if row == 0 else None)
        axis.set_yticks(range(4), contrasts[::-1])
        axis.set_ylim(-0.43, 3.43)
        axis.tick_params(axis="y", length=0)
        axis.set_xlabel(label, fontweight="bold", labelpad=6)
        axis.grid(axis="x", color="0.88", lw=0.55)
        axis.set_axisbelow(True)
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, edgecolor="0.64",
               bbox_to_anchor=(0.60, 1.00), handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.77, bottom=0.22, wspace=0.26)
    fig.savefig(FIGURES / "VCIA_ClosedLoop_Split_Contrasts.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def draw_nci1():
    fig, (time_ax, yield_ax) = plt.subplots(1, 2, figsize=(5.3, 2.4))
    sizes = DATA["nci1"]["sizes"]
    x = range(len(sizes))
    for method, color in [("IID 2048", BLUE), ("IID 8192", ORANGE)]:
        for cache, cache_label, marker, line in [("Cold", "cold", "o", "--"), ("Design-warm", "cached", "s", "-")]:
            series = DATA["nci1"][cache][method]
            time_ax.plot(x, series["time"], color=color, marker=marker, lw=1, ls=line, ms=4,
                         label=f"{int(method[4:]):,} · {cache_label}")
        yield_ax.plot(x, DATA["nci1"]["Design-warm"][method]["yield"], color=color,
                      marker="o", lw=1.15, ms=4, label=f"{int(method[4:]):,} draws")
    yield_ax.plot(x, [1] * len(sizes), color=GREEN, marker="D", lw=1.1, ms=3.5, label="Census")
    time_ax.axhline(0, color="0.55", lw=0.75)
    time_ax.set_ylabel("Paired time vs. census (ms)")
    yield_ax.set_ylabel("Feasible nominees admitted")
    yield_ax.set_ylim(0, 1.12)
    yield_ax.yaxis.set_major_formatter(PercentFormatter(1))
    for axis, panel in [(time_ax, "(a) Audit time"), (yield_ax, "(b) Feasible admissions")]:
        axis.set_xticks(x, [f"{value:,}" for value in sizes])
        axis.set_xlabel("Graphs in pool", labelpad=3)
        axis.annotate(panel, (0.5, -0.40), xycoords="axes fraction", ha="center",
                      fontweight="bold")
        axis.grid(axis="y", color="0.88", lw=0.55)
        axis.set_axisbelow(True)
        axis.legend(loc="lower left", frameon=True,
                    framealpha=0.96, edgecolor="0.64", fontsize=7, handlelength=1.5,
                    borderpad=0.35, labelspacing=0.34)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.96, wspace=0.43)
    fig.savefig(FIGURES / "VCIA_NCI1_Cost_and_Admission.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_closed_loop()
    draw_nci1()
