"""Recompute the expanded closed-loop and NCI1 results from the archived records."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

DATASETS = ("PROTEINS", "ENZYMES")
METRIC_ROLES = ("train", "audit", "test")
ARMS = {
    "Clean": ("VCIA", "Drop both", "J-Rank-Feasible"),
    "Moderate": (
        "No-Composition", "Drift only", "Reassignment only",
        "VCIA", "Drop both", "J-Rank-Feasible",
    ),
}
SIZES = (256, 493, 1024, 1315)
CACHES = ("Cold", "Design-warm")
METHODS = ("Census", "IID 2048", "IID 8192")
AUDIT_LABELS = {"Census": "Census", "IID 2048": "IID 2,048", "IID 8192": "IID 8,192"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def close(expected, actual, label):
    require(
        math.isclose(expected, actual, rel_tol=0, abs_tol=1e-8),
        f"{label}: {expected} != {actual}",
    )


def table_rows(path, columns, numeric_cell):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = tuple(cell.strip().strip("$") for cell in line.strip().rstrip("\\").split("&"))
        if len(cells) == columns and re.fullmatch(numeric_cell, cells[2]):
            rows.append(cells)
    return rows


def check_table(name, expected, published):
    require(len(expected) == len(published), f"{name}: unexpected row count")
    for row, (left, right) in enumerate(zip(expected, published), start=1):
        require(left == right, f"{name}, row {row}: {left} != {right}")


def endpoint_scores(labels, predictions):
    """Return arithmetic-normalized NMI and ARI on their percentage scale."""
    pairs = Counter(zip(labels, predictions))
    left = Counter(labels)
    right = Counter(predictions)
    n = len(labels)
    mutual_information = sum(
        count / n * math.log(count * n / (left[a] * right[b]))
        for (a, b), count in pairs.items()
    )
    entropy = -sum(
        count / n * math.log(count / n)
        for counts in (left, right) for count in counts.values()
    )
    nmi = max(0, 2 * mutual_information / entropy) if entropy else 1
    joint_pairs = sum(math.comb(count, 2) for count in pairs.values())
    left_pairs = sum(math.comb(count, 2) for count in left.values())
    right_pairs = sum(math.comb(count, 2) for count in right.values())
    total_pairs = math.comb(n, 2)
    numerator = 2 * (joint_pairs * total_pairs - left_pairs * right_pairs)
    denominator = (left_pairs + right_pairs) * total_pairs - 2 * left_pairs * right_pairs
    return 100 * nmi, 100 * (numerator / denominator if denominator else 1)


def check_predictions(split_dir, endpoints, splits):
    import numpy as np

    score_file = split_dir / "sealed_endpoints.npz"
    seal = load_json(split_dir / "prediction_seal.json")
    require(hashlib.sha256(score_file.read_bytes()).hexdigest() == seal["sha256"], score_file)
    labels_by_rep = load_json(split_dir / "test_labels.json")
    test_by_rep = {str(split["rep"]): split["test"] for split in splits}
    with np.load(score_file, allow_pickle=False) as bank:
        require(set(bank.files) == {row["token"] for row in endpoints}, score_file)
        for row in endpoints:
            rep = str(row["rep"])
            labels = labels_by_rep[rep]
            values = bank[row["token"]]
            require(values.ndim == 2 and np.isfinite(values).all(), row["trajectory_id"])
            require(len(labels) == len(values) == len(test_by_rep[rep]) == row["test_size"],
                    row["trajectory_id"])
            nmi, ari = endpoint_scores(labels, np.argmax(values, axis=1).tolist())
            close(nmi, row["nmi_pp"], row["trajectory_id"] + " NMI")
            close(ari, row["ari_pp"], row["trajectory_id"] + " ARI")


def check_main_ablation(groups, source_path):
    source = source_path.read_text(encoding="utf-8")
    body = source.split(r"\label{tab:composition-closed-loop}", 1)[1].split(r"\end{table}", 1)[0]
    published = []
    for line in body.splitlines():
        cells = [cell.strip() for cell in line.strip().rstrip("\\").split("&")]
        if len(cells) == 6 and re.fullmatch(r"[0-9,]+", cells[1]):
            cells[0] = cells[0].removeprefix(r"\textbf{").removesuffix("}")
            published.append(tuple(cells))

    expected = []
    for arm in ("No-Composition", "Drift only", "Reassignment only",
                "Drop both", "J-Rank-Feasible", "VCIA"):
        totals = Counter()
        for dataset in DATASETS:
            for trajectory in groups[dataset, "Moderate", arm]:
                totals.update(trajectory)
        counts = (totals[key] for key in ("nominations", "violations", "admissions", "replacements"))
        expected.append((arm, *(f"{value:,}" for value in counts), f"{totals['draws'] / 1e6:.3f}"))
    check_table("main closed-loop ablation", expected, published)


def closed_loop(data, plots, score_predictions, main_experiments):
    endpoints = {}
    rounds = defaultdict(list)
    for split_dir in sorted((data / "reproduction" / "closed").glob("*_*")):
        if not split_dir.is_dir():
            continue
        splits = load_json(split_dir / "splits.json")
        require({split["rep"] for split in splits} == {1, 2} and len(splits) == 2, split_dir)
        for split in splits:
            assigned = [index for role in METRIC_ROLES for index in split[role]]
            peer_train = [index for peer in split["peer_train"] for index in peer]
            require(len(assigned) == len(set(assigned)), f"{split_dir}: overlapping data roles")
            require(sorted(peer_train) == sorted(split["train"]),
                    f"{split_dir}: inconsistent peer training indices")
        split_endpoints = load_json(split_dir / "endpoints.json")
        for endpoint in split_endpoints:
            identity = endpoint["trajectory_id"]
            require(identity not in endpoints, f"duplicate trajectory {identity}")
            endpoints[identity] = endpoint
        for row in load_csv(split_dir / "rounds.csv"):
            rounds[row["trajectory_id"]].append(row)
        if score_predictions:
            check_predictions(split_dir, split_endpoints, splits)

    require(len(endpoints) == 216 and endpoints.keys() == rounds.keys(), "closed-loop coverage")
    groups = defaultdict(list)
    paired = {}
    matched_rounds = defaultdict(dict)
    for identity, trajectory in rounds.items():
        endpoint = endpoints[identity]
        trajectory.sort(key=lambda row: int(row["round"]))
        require([int(row["round"]) for row in trajectory] == list(range(1, 61)), identity)
        counts = Counter()
        previous = endpoint["initial_sha"]
        for row in trajectory:
            nominated = bool(row["mask"])
            admitted = row["status"] == "committed"
            draws = int(row["audit_draws"])
            require(previous == row["previous_sha"] and int(row["design_draws"]) == 512, identity)
            require(draws == (8192 if nominated else 0), identity)
            if nominated:
                feasible = (
                    float(row["exact_ell"]) <= float(row["chi"])
                    and float(row["exact_phi"]) <= float(row["gamma"])
                )
                require(int(row["finite_feasible"]) == feasible, identity)
                counts["nominations"] += 1
                counts["violations"] += not feasible
            if admitted:
                require(nominated and feasible and row["candidate_sha"] == row["committed_sha"],
                        f"{identity}: invalid commitment")
                counts["admissions"] += 1
                counts["replacements"] += int(row["operator_updates"])
            else:
                require(previous == row["committed_sha"], f"{identity}: state changed on hold")
            counts["draws"] += draws
            previous = row["committed_sha"]
            if endpoint["condition"] == "Moderate" and endpoint["arm"] in ("VCIA", "Reassignment only"):
                key = (endpoint["dataset"], endpoint["outer_seed"], endpoint["rep"], int(row["round"]))
                arms = matched_rounds[key]
                require(endpoint["arm"] not in arms and previous, f"invalid matched round: {key}")
                arms[endpoint["arm"]] = (row["mask"], previous)

        dataset, condition, arm = (endpoint[key] for key in ("dataset", "condition", "arm"))
        counts.update(nmi=endpoint["nmi_pp"], ari=endpoint["ari_pp"])
        groups[dataset, condition, arm].append(counts)
        key = (dataset, condition, arm, endpoint["outer_seed"], endpoint["rep"])
        require(key not in paired, f"duplicate pairing key: {key}")
        paired[key] = counts

    table = []
    for condition, arms in ARMS.items():
        for dataset in DATASETS:
            for index, arm in enumerate(arms):
                values = groups[dataset, condition, arm]
                require(len(values) == 12, f"{dataset}, {condition}, {arm}: pairing coverage")
                totals = sum(values, Counter())
                table.append((
                    dataset if index == 0 else "", arm,
                    f"{totals['nominations']:,}", f"{totals['violations']:,}",
                    f"{totals['admissions']:,}", f"{totals['replacements']:,}",
                    f"{totals['draws'] / 1_000_000:.3f}",
                    f"{mean(row['nmi'] for row in values):.2f}",
                    f"{mean(row['ari'] for row in values):.2f}",
                ))
    check_table("closed-loop table", table, table_rows(data / "ClosedLoop_Expanded_Table.tex", 9, r"[0-9,]+"))

    if main_experiments.is_file():
        check_main_ablation(groups, main_experiments)
        print("Main closed-loop table: verified all six policy rows.")
    require(len(matched_rounds) == 1440, "moderate round pairing")
    mask_differences = 0
    for key, arms in matched_rounds.items():
        require(set(arms) == {"VCIA", "Reassignment only"}, f"incomplete pair: {key}")
        full_mask, full_state = arms["VCIA"]
        other_mask, other_state = arms["Reassignment only"]
        require(full_state == other_state, f"end-of-round states differ: {key}")
        mask_differences += full_mask != other_mask
    require(mask_differences == 4, "nomination mask differences")
    print("Four nomination masks differ "
          "and all 1,440 end-of-round state pairs match for VCIA and Reassignment only.")

    for arm, series in plots.items():
        for dataset, metrics in series.items():
            for metric, expected in metrics.items():
                actual = [
                    mean(
                        paired[dataset, "Moderate", "VCIA", seed, rep][metric]
                        - paired[dataset, "Moderate", arm, seed, rep][metric]
                        for rep in (1, 2)
                    )
                    for seed in range(31501, 31507)
                ]
                require(len(expected) == len(actual), f"{dataset}, {arm}, {metric}: plot coverage")
                for left, right in zip(expected, actual):
                    close(left, right, f"{dataset}, {arm}, {metric}")

    print("S21 / S1: verified 12,960 round decisions and 216 endpoints across 12 outer splits.")
    if score_predictions:
        print("Verified all 216 sealed prediction arrays and recalculated their NMI and ARI.")


def nci1(data, plots):
    root = data / "reproduction" / "nci1"
    scheduled = load_csv(root / "scheduled_snapshots.csv")
    nominees = {row["snapshot_id"]: row for row in scheduled if row["candidate_mask"]}
    require(len({row["snapshot_id"] for row in scheduled}) == len(scheduled) == 180, "snapshots")
    require(len(nominees) == 113, "nominated snapshots")

    def key(row):
        return row["snapshot_id"], int(row["pool_size"]), row["cache"], row["method"]

    repeats = defaultdict(list)
    for row in load_csv(root / "timings.csv"):
        repeats[key(row)].append(row)
        close(sum(float(row[name]) for name in ("prep_ms", "inference_ms", "decision_ms")),
              float(row["audit_ms"]), key(row))
    records = load_csv(root / "candidate_results.csv")
    by_key = {key(row): row for row in records}
    require(len(by_key) == len(records) == len(repeats) == 2712, "NCI1 candidate coverage")
    require(sum(map(len, repeats.values())) == 8136, "NCI1 timing coverage")
    require(by_key.keys() == repeats.keys(), "NCI1 candidate / timing join")
    pool_membership = {
        (row["partition"], row["pool"]): set(row["ids"]) for row in load_json(root / "pools.json")
    }
    for partition in {int(row["partition"]) for row in scheduled}:
        smaller = set()
        for size in SIZES:
            members = pool_membership[partition, size]
            require(len(members) == size and smaller < members, (partition, size))
            smaller = members

    for k, row in by_key.items():
        require(k[0] in nominees and row["mask"] == nominees[k[0]]["candidate_mask"], k)
        require({int(rep["repeat"]) for rep in repeats[k]} == {1, 2, 3}, k)
        close(median(float(rep["audit_ms"]) for rep in repeats[k]), float(row["audit_ms"]), k)
        feasible = (
            float(row["finite_ell"]) <= float(row["chi"])
            and float(row["finite_phi"]) <= float(row["gamma"])
        )
        passed = bool(int(row["passed"]))
        require(feasible == bool(int(row["finite_feasible"])), k)
        require(int(row["feasible_pass"]) == (feasible and passed), k)
        require(int(row["feasible_held"]) == (feasible and not passed), k)
        require(int(row["excess_pass"]) == 0 and (not passed or feasible), k)
        if row["method"] == "Census":
            require(passed == feasible and int(row["audit_draws"]) == 0, k)
        else:
            require(int(row["audit_draws"]) == int(row["method"].split()[1]), k)

    table = []
    require(plots["sizes"] == list(SIZES), "plotted pool sizes")
    for size_index, size in enumerate(SIZES):
        for method in METHODS:
            timings = []
            fractions = []
            for cache in CACHES:
                rows = [by_key[snapshot, size, cache, method] for snapshot in nominees]
                admitted = sum(int(row["feasible_pass"]) for row in rows)
                eligible = sum(int(row["finite_feasible"]) for row in rows)
                fractions.append(f"{admitted}/{eligible}")
                deltas = [
                    float(row["audit_ms"])
                    - float(by_key[row["snapshot_id"], size, cache, "Census"]["audit_ms"])
                    for row in rows
                ]
                paired_median = median(deltas)
                timings.extend((
                    f"{median(float(row['audit_ms']) for row in rows):.2f}",
                    "--" if method == "Census" else f"{paired_median:+.2f}",
                ))
                close(paired_median, plots[cache][method]["time"][size_index], (size, cache, method))
                close(admitted / eligible, plots[cache][method]["yield"][size_index], (size, cache, method))
                if size == 1315 and cache == "Design-warm" and method == "IID 2048":
                    require(sum(delta < 0 for delta in deltas) == 109, "NCI1 faster cases")
                if size == 1315 and cache == "Design-warm" and method != "Census":
                    expected = 1038 if method == "IID 2048" else 1313
                    require(median(int(row["unique_graphs"]) for row in rows) == expected,
                            "NCI1 distinct graphs")
                if size in (1024, 1315) and method != "Census":
                    near_threshold = [
                        row for row in rows if int(row["finite_feasible"])
                        and 0.9 <= max(float(row["finite_ell"]) / float(row["chi"]),
                                       float(row["finite_phi"]) / float(row["gamma"])) <= 1.1
                    ]
                    require(len(near_threshold) == {1024: 5, 1315: 7}[size]
                            and all(int(row["feasible_held"]) for row in near_threshold),
                            "NCI1 near-threshold feasible nominees")
            require(fractions[0] == fractions[1], (size, method, fractions))
            table.append((
                f"{size:,}" if method == "Census" else "", AUDIT_LABELS[method],
                fractions[0], *timings,
            ))
    check_table("NCI1 table", table, table_rows(data / "NCI1_Audit_Cost_Table.tex", 7, r"\d+/\d+"))
    print("S22 / S2: verified 180 snapshots, 2,712 decisions and 8,136 timing receipts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-predictions", action="store_true", help="Recalculate NMI and ARI (NumPy).")
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted supplement/data directory.")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    archive = data.parents[1]
    plots = load_json(archive / "figure_source" / "closed_loop_nci1_plot_data.json")
    main_experiments = archive / "main" / "sections" / "04_Experiments.tex"
    closed_loop(data, plots["closed_loop"], args.score_predictions, main_experiments)
    nci1(data, plots["nci1"])


if __name__ == "__main__":
    main()
