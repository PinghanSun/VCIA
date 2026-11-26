"""Summarize the two policies' IID-audited fixed-snapshot replay."""

import argparse
import csv
from pathlib import Path


POLICIES = {"VCIA": "VCIA", "J-Rank": "J-Rank-Feasible"}
SNAPSHOT_FIELDS = ("dataset", "outer_seed", "alpha", "condition", "round")


def summarize(path):
    records = {policy: {} for policy in POLICIES}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            policy = row["ranking"]
            if row["audit"] != "IID" or policy not in records:
                continue
            snapshot = tuple(row[field] for field in SNAPSHOT_FIELDS)
            if snapshot in records[policy]:
                raise ValueError(f"Duplicate IID replay record for {policy}: {snapshot}")
            records[policy][snapshot] = row

    vcia = records["VCIA"]
    rank = records["J-Rank"]
    if len(vcia) != 324 or vcia.keys() != rank.keys():
        raise ValueError("The main-paper summary requires the same 324 snapshots for both policies.")

    passes = {
        policy: {snapshot for snapshot, row in rows.items() if row["passed"] == "1"}
        for policy, rows in records.items()
    }
    shared = passes["VCIA"] & passes["J-Rank"]
    print(f"{len(vcia)} matched snapshots; {len(shared)} passed under both policies")
    full_totals = {}
    for policy, rows in records.items():
        replacements = {snapshot: int(rows[snapshot]["committed_operator_updates"])
                        for snapshot in passes[policy]}
        full_totals[policy] = sum(replacements.values())
        shared_total = sum(replacements[snapshot] for snapshot in shared)
        print(f"{POLICIES[policy]}: {len(passes[policy])} passes, {full_totals[policy]} prospective "
              f"replacements across its passes, {shared_total} across common passes")

    if full_totals["J-Rank"]:
        change = 100 * (full_totals["VCIA"] / full_totals["J-Rank"] - 1)
        print(f"VCIA replacement change over each policy's own passes: {change:+.2f}%")
    feasible = sum(row["finite_feasible"] == "1" for row in vcia.values())
    withheld = sum(row["finite_feasible"] == "1" and row["passed"] == "0" for row in vcia.values())
    nominees = sum(row["finite_feasible"] in {"0", "1"} for row in vcia.values())
    print(f"VCIA: {feasible} of {nominees} nominees feasible on their finite pools; "
          f"{withheld} feasible nominees withheld by the IID audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="Full run_checks.py replay.csv")
    args = parser.parse_args()
    summarize(args.replay)


if __name__ == "__main__":
    main()
