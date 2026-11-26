"""Summarize detectable updates on frozen WL states and along each policy history."""

import csv
from argparse import ArgumentParser
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median


def summarize(rows, arm):
    nominated = [row for row in rows if row["arm"] == arm]
    admitted = [row for row in nominated if row["status"] == "committed"]
    frozen = Counter()
    direction = 1 if arm == "VCIA" else -1
    for row in nominated:
        count_gain = direction * (int(row["H"]) - int(row["other_H"]))
        age_gain = direction * (int(row["A"]) - int(row["other_A"]))
        if count_gain > 0:
            frozen["count"] += 1
        elif count_gain == 0 and age_gain > 0:
            frozen["age"] += 1
        else:
            assert count_gain == 0 and age_gain == 0 and row["mask"] == row["other_mask"]
            frozen["same"] += 1
    detectable = sum(int(row["H"]) for row in admitted)
    ages = sorted(int(age) for row in admitted for age in row["ages"].split())
    assert len(ages) == detectable
    return {
        "history": arm,
        "nominated_rounds": len(nominated),
        "same_state_full_higher_H": frozen["count"],
        "same_state_full_higher_A_at_equal_H": frozen["age"],
        "same_state_identical_mask": frozen["same"],
        "admitted_rounds": len(admitted),
        "committed_detectable_updates": detectable,
        "committed_operator_replacements": sum(int(row["operator_updates"]) for row in admitted),
        "mean_detectable_per_admission": detectable / len(admitted),
        "detectable_claim_only": sum(int(row["detectable_claim_only"]) for row in admitted),
        "detectable_both_operator_and_claim": sum(int(row["detectable_joint"]) for row in admitted),
        "detectable_update_age_median": median(ages),
        "detectable_update_age_p90_nearest_rank": ages[ceil(0.9 * len(ages)) - 1],
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to ranking_replay.csv")
    parser.add_argument("--output", type=Path, required=True, help="Path for the ranking summary CSV")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1104
    records = [summarize(rows, arm) for arm in ("VCIA", "Drop both")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
