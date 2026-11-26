"""Summarize the full WL-TFIDF and scalar MMFD ablation reruns."""

import argparse
import csv
from collections import Counter
from pathlib import Path


REFERENCE_ARMS = ("VCIA", "No-Composition")
WL_ARMS = ("No-Composition", "Drift only", "Reassignment only",
           "Drop both", "J-Rank-Feasible", "VCIA")


def summarize(title, files, conditions, rounds_per_arm, arms):
    totals = {arm: Counter() for arm in arms}
    for file in files:
        with file.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                arm = row["arm"]
                if arm not in totals or row["condition"] not in conditions:
                    continue
                committed = row["status"] == "committed"
                feasible = row["finite_feasible"]
                totals[arm].update(
                    rounds=1, nominees=int(feasible in {"0", "1"}),
                    nominees_above_tolerance=int(feasible == "0"),
                    replacements=int(row["operator_updates"]), draws=int(row["audit_draws"]),
                    admissions=int(committed), excess=int(committed and feasible == "0"),
                )

    vcia, control = (totals[arm] for arm in REFERENCE_ARMS)
    if any(counts["rounds"] != rounds_per_arm for counts in totals.values()):
        raise ValueError(f"{title} requires a full run with {rounds_per_arm:,} scheduled rounds per arm.")
    if control["replacements"] == 0 or control["draws"] == 0:
        raise ValueError(f"{title} has zero reference replacements or audit draws.")
    print(title)
    for arm, counts in totals.items():
        print(f"  {arm}: {counts['rounds']:,} scheduled rounds, {counts['nominees']:,} nominees, "
              f"{counts['nominees_above_tolerance']:,} nominees above tolerance, "
              f"{counts['admissions']:,} admissions, "
              f"{counts['replacements']:,} replacements, {counts['draws']:,} audit draws, "
              f"{counts['excess']} admissions above tolerance")
    print(f"  Replacement change: {100 * (vcia['replacements'] / control['replacements'] - 1):+.1f}%")
    print(f"  Audit draw change: {100 * (vcia['draws'] / control['draws'] - 1):+.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl", type=Path, required=True, help="Full run_closed_ablation.py output directory")
    parser.add_argument("--mmfd", type=Path, required=True, help="Full run_mmfd_ablation.py output directory")
    args = parser.parse_args()
    summarize("WL-TFIDF, moderate perturbation", sorted(args.wl.glob("*/rounds.csv")), {"Moderate"}, 1440, WL_ARMS)
    summarize("Scalar MMFD, clean and moderate", [args.mmfd / "rounds.csv"], {"Clean", "Moderate"}, 2160, REFERENCE_ARMS)


if __name__ == "__main__":
    main()
