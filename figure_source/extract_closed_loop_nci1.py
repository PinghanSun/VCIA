"""Reproduce the new tables and figure inputs from the experiment workbook."""

import json
import sys
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import mean, median

from openpyxl import load_workbook


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "runs" / "workbook-export"
DATASETS = ("PROTEINS", "ENZYMES")
RANKERS = ("VCIA", "Drop both", "J-Rank-Feasible")
CONSTRAINT_ARMS = ("No-Composition", "Drift only", "Reassignment only")
CONTRASTS = ("No-Composition", "Reassignment only", "Drop both", "J-Rank-Feasible")
SIZES = (256, 493, 1024, 1315)
CACHES = ("Cold", "Design-warm")
AUDITS = ("Census", "IID 2048", "IID 8192")


def closed_loop(sheets):
    summary = {(r[0], r[1], r[2]): r for r in sheets[1]}
    by_trajectory = defaultdict(list)
    for row in sheets[7]:
        by_trajectory[row[0]].append(row)
    by_group = defaultdict(list)
    by_split = defaultdict(list)
    assert len(by_trajectory) == len(sheets[6]) == 216
    for r in sheets[6]:
        rounds = by_trajectory[r[0]]
        assert {x[1] for x in rounds} == set(range(1, 61))
        nominees = [x for x in rounds if x[7] != "no_candidate"]
        admitted = [x for x in rounds if x[7] == "committed"]
        for x in rounds:
            assert x[14] == (0 if x[7] == "no_candidate" else 8192)
            if x[7] != "no_candidate":
                assert int(x[8]).bit_count() == 6
                assert x[9] == (x[17] <= x[21] and x[18] <= x[22])
        assert (len(rounds), len(nominees), sum(x[9] for x in nominees),
                len(admitted), sum(x[9] for x in admitted), sum(x[10] for x in admitted if x[9]),
                sum(x[14] for x in rounds)) == (r[6], r[8], r[9], r[11], r[12], r[14], r[23])
        assert r[13] == 0 and r[14] == r[15]
        by_group[r[2], r[1], r[5]].append(r)
        by_group[r[2], "All", r[5]].append(r)
        key = (r[2], r[1], r[3], r[5])
        by_split[key].append(r)

    for key, entries in by_group.items():
        r = summary[key]
        for col, trajectory_col in zip((4, 5, 6, 7, 8, 9, 10, 11, 16), (6, 8, 9, 10, 11, 12, 13, 14, 23)):
            assert r[col] == sum(x[trajectory_col] for x in entries), (key, col)
        assert isclose(r[14], mean(x[21] for x in entries), abs_tol=1e-9)
        assert isclose(r[15], mean(x[22] for x in entries), abs_tol=1e-9)

    table = [
        r"\begin{table}[!t]",
        r"\caption{\textbf{Closed-loop results by dataset, perturbation level, and policy.} Each row covers 720 scheduled rounds across 12 trajectories. Over limit counts nominees that exceed either tolerance under the exact census. All admitted compositions met both finite-pool tolerances. Replacements are committed operator changes. Draws are summed over all nominees. NMI and ARI are mean round-60 held-out scores.}",
        r"\label{supp:tab-expanded-closed-loop}",
        r"\centering\small",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrrr@{}}",
        r"\toprule",
        r"Dataset & Policy & Nom. & Over limit & Adm. & Repl. & Draws (M) & NMI & ARI \\",
    ]
    ending = r" \\"
    conditions = sorted({key[0] for key in by_group})
    for condition in conditions:
        title = "No perturbation" if condition == "Clean" else "Moderate perturbation"
        table.extend((r"\midrule", r"\multicolumn{9}{@{}l}{\textbf{" + title + "}}" + ending))
        arms = RANKERS if condition == "Clean" else CONSTRAINT_ARMS + RANKERS
        for dataset in DATASETS:
            for i, arm in enumerate(arms):
                r = summary[condition, dataset, arm]
                table.append(f"{dataset if i == 0 else ''} & {arm} & {r[5]:,} & {r[7]:,} & {r[9]:,} & "
                             f"{r[11]:,} & {r[16] / 1e6:.3f} & {r[14]:.2f} & {r[15]:.2f}" + ending)
    table.extend((r"\bottomrule", r"\end{tabular*}", r"\end{table}"))

    moderate, = [condition for condition in conditions if condition != "Clean"]
    plot = {}
    for arm in CONTRASTS:
        plot[arm] = {}
        for dataset in DATASETS:
            splits = sorted(k[2] for k in by_split if k[:2] == (moderate, dataset) and k[3] == "VCIA")
            diffs = []
            for split in splits:
                ours, baseline = (sorted(by_split[moderate, dataset, split, a], key=lambda r: r[4])
                                  for a in ("VCIA", arm))
                assert len(ours) == len(baseline) == 2
                assert [(r[4], r[31]) for r in ours] == [(r[4], r[31]) for r in baseline]
                diffs.append([mean(o[col] - b[col] for o, b in zip(ours, baseline)) for col in (14, 21, 22)])
            assert len(diffs) == 6
            plot[arm][dataset] = dict(zip(("replacements", "nmi", "ari"), map(list, zip(*diffs))))
    print("Closed-loop: 216 matched trajectories; 12960 rounds verified.")
    for arm in CONSTRAINT_ARMS + RANKERS:
        r = summary[moderate, "All", arm]
        print(arm, [r[i] for i in (5, 7, 9, 11, 16)])
    return table, plot


def nci1(sheets):
    summary = {(r[0], r[1], r[2]): r for r in sheets[2] if isinstance(r[0], int)}
    repeats = defaultdict(list)
    for r in sheets[10]:
        repeats[r[:4]].append(r[9])
    by_group = defaultdict(list)
    for r in sheets[9]:
        assert len(repeats[r[:4]]) == 3
        assert isclose(median(repeats[r[:4]]), r[26], abs_tol=1e-7)
        assert r[9] == (r[16] <= r[14] and r[17] <= r[15])
        assert r[12] == 0
        by_group[r[1:4]].append(r)
    census = {r[:3]: r[26] for r in sheets[9] if r[3] == "Census"}
    assert len(repeats) == 2712 and len(census) == 904
    plot = {"sizes": SIZES, **{cache: {audit: {"time": [], "yield": []} for audit in AUDITS} for cache in CACHES}}
    ending = r" \\"
    table = [
        r"\begin{table}[!htbp]",
        r"\caption{\textbf{Audit cost and feasible admissions on NCI1.} All rows use the same 113 fixed nominees. Feasible admissions are reported relative to the number feasible on the indicated pool. The paired difference $\Delta$ is audit time minus census time on the same nominee and cache state. Median times use three repeats per cell. No method admitted an over-limit nominee.}",
        r"\label{supp:tab-nci1-audit-cost}",
        r"\centering\small",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rlccccc@{}}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Cold cache} & \multicolumn{2}{c}{Cached design evaluations} \\",
        r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Pool size & Audit & Feasible admitted & Median (ms) & $\Delta$ (ms) & Median (ms) & $\Delta$ (ms) \\",
    ]
    for size in SIZES:
        table.append(r"\midrule")
        for i, audit in enumerate(AUDITS):
            times = []
            for cache in CACHES:
                key = (size, cache, audit)
                entries = by_group[key]
                r = summary[key]
                t = [(x[26], x[26] - census[x[:3]]) for x in entries]
                assert (len(entries), sum(x[9] for x in entries), sum(x[11] for x in entries)) == (r[3], r[4], r[5])
                assert r[3] == len({x[0] for x in entries}) == 113
                assert all(isclose(median(x[j] for x in t), r[11 + j], abs_tol=1e-7) for j in (0, 1))
                assert sum(x[1] < 0 for x in t) == r[13]
                plot[cache][audit]["time"].append(r[12])
                plot[cache][audit]["yield"].append(r[5] / r[4])
                times.extend((f"{r[11]:.2f}", "--" if audit == "Census" else f"${r[12]:+.2f}$"))
            a = summary[size, CACHES[0], audit]
            b = summary[size, CACHES[1], audit]
            assert a[4:9] == b[4:9]
            label = {"Census": "Census", "IID 2048": "IID 2,048", "IID 8192": "IID 8,192"}[audit]
            table.append(f"{f'{size:,}' if i == 0 else ''} & {label} & {a[5]}/{a[4]} & " + " & ".join(times) + ending)
    table.extend((r"\bottomrule", r"\end{tabular*}", r"\end{table}"))
    for size in (1024, 1315):
        r = [s for s in sheets[3] if s[0] == size and s[1] == "Design-warm" and s[2] in AUDITS]
        assert len(r) == 3 and {x[4] for x in r} == {5 if size == 1024 else 7}
        assert all(x[5] == 0 for x in r if x[2] != "Census")
    print("NCI1: 2712 candidate/pool/cache/method cells and 8136 timings verified.")
    return table, plot


def main():
    workbook = load_workbook(sys.argv[1], read_only=True, data_only=True)
    try:
        sheets = {i: list(workbook.worksheets[i].iter_rows(min_row=5, values_only=True))
                  for i in (1, 2, 3, 6, 7, 9, 10)}
    finally:
        workbook.close()
    closed_table, closed_plot = closed_loop(sheets)
    nci1_table, nci1_plot = nci1(sheets)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ClosedLoop_Expanded_Table.tex", closed_table), ("NCI1_Audit_Cost_Table.tex", nci1_table)):
        (DATA / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    (DATA / "closed_loop_nci1_plot_data.json").write_text(
        json.dumps({"closed_loop": closed_plot, "nci1": nci1_plot}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
