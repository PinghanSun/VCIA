"""Recompute the two abstract comparisons from the released records."""

from collections import defaultdict
from csv import DictReader
from math import fsum
from pathlib import Path


RECORDS = Path(__file__).resolve().parent


def read_records(filename):
    with (RECORDS / filename).open(encoding="utf-8-sig", newline="") as stream:
        yield from DictReader(stream)


def pair_rows(rows, match_fields, arm_field, arms, expected_pairs):
    pairs = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in match_fields)
        arm = row[arm_field]
        if arm in pairs[key]:
            raise ValueError(f"Duplicate row for {key}, {arm}")
        pairs[key][arm] = row
    if len(pairs) != expected_pairs or any(set(pair) != set(arms) for pair in pairs.values()):
        raise ValueError(f"Expected {expected_pairs} complete pairs for {arms}")
    return pairs


def report_timing():
    pairs = pair_rows(
        read_records("timing_rounds.csv"),
        ("dataset", "seed", "condition", "repeat", "round"),
        "arm", ("VCIA", "Unscreened"), 720,
    )
    matched_fields = (
        "executed_state_sha256", "unique_graph_operator_misses",
        "design_draws", "verification_draws", "status",
    )
    groups = defaultdict(list)
    for key, pair in pairs.items():
        dataset, seed, condition, repeat, round_number = key
        vcia, baseline = pair["VCIA"], pair["Unscreened"]
        if any(vcia[field] != baseline[field] for field in matched_fields):
            raise ValueError(f"Unmatched timing conditions for {key}")
        switched = pairs[dataset, seed, condition, str(3 - int(repeat)), round_number]
        if vcia["order"] == baseline["order"] or switched["VCIA"]["order"] != baseline["order"]:
            raise ValueError(f"Unexpected execution order for {key}")
        groups[dataset, condition].append(
            (float(vcia["end_to_end_ms"]), float(baseline["end_to_end_ms"]))
        )
    if len(groups) != 4 or any(len(rows) != 180 for rows in groups.values()):
        raise ValueError("Expected 180 paired rounds per dataset and condition")
    for (dataset, condition), rows in sorted(groups.items()):
        if condition == "Perturbed":
            vcia, baseline = map(fsum, zip(*rows, strict=True))
            reduction = 100 * (1 - vcia / baseline)
            print(f"{dataset}: {reduction:.1f}% lower cumulative online runtime vs. Unscreened "
                  f"({vcia:.1f} vs. {baseline:.1f} ms across {len(rows)} paired rounds)")


def report_replay():
    arms = ("VCIA", "J-Rank")
    pairs = pair_rows(
        (row for row in read_records("replay.csv") if row["audit"] == "IID"),
        ("dataset", "outer_seed", "alpha", "condition", "round"),
        "ranking", arms, 324,
    )
    replacements = dict.fromkeys(arms, 0)
    passes = dict.fromkeys(arms, 0)
    eligible = 0
    for key, pair in pairs.items():
        vcia, baseline = (pair[arm] for arm in arms)
        for field in ("candidate_family", "feasible_mask_count", "prior_token"):
            if vcia[field] != baseline[field]:
                raise ValueError(f"Unmatched replay conditions for {key}, {field}")
        eligible += int(vcia["feasible_mask_count"]) > 0
        for arm, row in pair.items():
            passed = row["passed"] == "1"
            committed = int(row["committed_operator_updates"])
            expected = int(row["possible_operator_updates"]) if passed else 0
            if committed != expected:
                raise ValueError(f"Inconsistent operator count for {key}, {arm}")
            replacements[arm] += committed
            passes[arm] += passed
    vcia, baseline = (replacements[arm] for arm in arms)
    increase = 100 * (vcia / baseline - 1)
    print(f"Fixed-snapshot IID replay: {increase:.1f}% more operator replacements vs. J-Rank "
          f"({vcia} vs. {baseline}; {passes['VCIA']} vs. {passes['J-Rank']} audit passes; "
          f"{eligible}/{len(pairs)} snapshots eligible)")


if __name__ == "__main__":
    report_timing()
    report_replay()
