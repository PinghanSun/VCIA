"""Check the optional records archive and recompute manuscript summaries."""

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_hashes(root):
    manifest = (root / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in manifest.splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Invalid record path: {name}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == digest, f"SHA-256 mismatch: {name}")
    print(f"Verified {len(manifest.splitlines())} archived files.", flush=True)


def verify_timing(path):
    pairs = defaultdict(dict)
    keys = ("dataset", "condition", "seed", "repeat", "round")
    for row in read_rows(path):
        key = tuple(row[column] for column in keys)
        require(row["arm"] not in pairs[key], f"Duplicate timing arm: {key}")
        pairs[key][row["arm"]] = row

    totals = defaultdict(Counter)
    for key, arms in pairs.items():
        require(arms.keys() == {"VCIA", "Unscreened"}, f"Unpaired timing: {key}")
        screened, unscreened = arms["VCIA"], arms["Unscreened"]
        for column in ("executed_state_sha256", "status", "unique_graph_operator_misses"):
            require(screened[column] == unscreened[column], f"Different {column}: {key}")
        totals[key[:2]].update({"pairs": 1})
        for arm, row in arms.items():
            totals[key[:2]][arm] += float(row["end_to_end_ms"])

    for dataset, expected in (("PROTEINS", 4.40), ("IMDB-BINARY", 5.91)):
        for condition in ("Clean", "Perturbed"):
            require(totals[dataset, condition]["pairs"] == 180, "Expected 180 paired rounds per condition")
        group = totals[dataset, "Perturbed"]
        saving = 100 * (group["Unscreened"] - group["VCIA"]) / group["Unscreened"]
        require(round(saving, 2) == expected, f"Unexpected {dataset} timing reduction")
        print(f"{dataset}: {group['pairs']} perturbed pairs, {saving:.2f}% lower cumulative admission time.")


def verify_replay(path):
    arms = defaultdict(dict)
    columns = ("dataset", "outer_seed", "alpha", "condition", "round")
    for row in read_rows(path):
        key = tuple(row[column] for column in columns)
        arm = row["ranking"], row["audit"]
        require(key not in arms[arm], f"Duplicate replay row: {arm}, {key}")
        arms[arm][key] = row

    expected = {"VCIA": (130, 723, 161, 31), "J-Rank": (134, 678, 164, 30)}
    baseline = arms["VCIA", "IID"].keys()
    require(len(arms) == 4 and len(baseline) == 324, "Unexpected replay design")
    require(all(group.keys() == baseline for group in arms.values()), "Replay snapshot keys differ")
    iid_passes = {}
    for ranking, counts in expected.items():
        display = "J-Rank-Feasible" if ranking == "J-Rank" else ranking
        group = arms[ranking, "IID"]
        feasible = {key for key, row in group.items() if row["finite_feasible"] == "1"}
        passed = {key: row for key, row in group.items() if row["passed"] == "1"}
        replacements = sum(int(row["committed_operator_updates"]) for row in passed.values())
        actual = len(passed), replacements, len(feasible), len(feasible - passed.keys())
        require(passed.keys() <= feasible, f"{display}: passing nominee exceeds a finite-pool tolerance")
        require(actual == counts, f"{display}: unexpected replay totals {actual}")
        iid_passes[ranking] = passed
        print(f"{display}: {actual[0]}/324 passes, {replacements} prospective replacements, "
              f"{actual[3]}/{actual[2]} finite-pool feasible nominations withheld.")
    improvement = 100 * (expected["VCIA"][1] - expected["J-Rank"][1]) / expected["J-Rank"][1]
    print(f"Prospective increase on each policy's passing snapshots: {improvement:.2f}%.")
    shared = iid_passes["VCIA"].keys() & iid_passes["J-Rank"].keys()
    shared_updates = {
        ranking: sum(int(passes[key]["committed_operator_updates"]) for key in shared)
        for ranking, passes in iid_passes.items()
    }
    print(f"Jointly passing snapshots: {len(shared)}, with {shared_updates['VCIA']} "
          f"and {shared_updates['J-Rank']} prospective replacements, respectively.")


def verify_original_closed_loop(root):
    keys = ("dataset", "outer_seed", "rep", "peers", "q", "topology", "condition", "arm")
    by_path = defaultdict(dict)
    for row in read_rows(root / "closed_loop_rounds.csv"):
        key = tuple(row[name] for name in keys)
        index = int(row["round"])
        require(index not in by_path[key], f"Duplicate closed-loop round: {key}, {index}")
        by_path[key][index] = row

    by_arm = defaultdict(Counter)
    for key, rounds in by_path.items():
        require(rounds.keys() == set(range(1, 61)), f"Incomplete closed-loop trajectory: {key}")
        commits = [row for row in rounds.values() if row["status"] == "committed"]
        by_arm[key[-1]].update(paths=1, rounds=len(rounds), commits=len(commits),
                               replacements=sum(int(row["operator_updates"]) for row in commits),
                               overlimit=sum(row["finite_feasible"] != "1" for row in commits))

    expected = {
        "VCIA": (1449, 9994, 0),
        "Unscreened": (1449, 9994, 0),
        "J-Rank-Feasible": (2045, 12583, 0),
        "No-Verify": (4491, 31149, 318),
    }
    require(by_arm.keys() == expected.keys(), "Unexpected original closed-loop arms")
    for arm, counts in by_arm.items():
        require((counts["paths"], counts["rounds"]) == (144, 8640), f"Unexpected grid size: {arm}")
        actual = counts["commits"], counts["replacements"], counts["overlimit"]
        require(actual == expected[arm], f"Unexpected closed-loop totals for {arm}: {actual}")
        print(f"Original closed loop, {arm}: {actual[0]} admissions, {actual[1]} replacements, "
              f"{actual[2]} admissions above tolerance.")

    endpoint_groups = defaultdict(dict)
    for row in read_rows(root / "closed_loop_endpoints.csv"):
        key = tuple(row[name] for name in keys)
        require(key in by_path and key[-1] not in endpoint_groups[key[:-1]],
                f"Unknown or duplicate closed-loop endpoint: {key}")
        require(row["final_state"] == by_path[key][60]["committed_sha"], f"Endpoint state differs: {key}")
        endpoint_groups[key[:-1]][key[-1]] = float(row["nmi"])

    by_condition = defaultdict(list)
    for key, arms in endpoint_groups.items():
        require(arms.keys() == expected.keys(), f"Unpaired closed-loop endpoints: {key}")
        by_condition[(key[0], *key[3:])].append(100 * (arms["VCIA"] - arms["J-Rank-Feasible"]))
    require(len(endpoint_groups) == 144, "Unexpected endpoint count")
    require(len(by_condition) == 24 and all(len(group) == 6 for group in by_condition.values()),
            "Unexpected closed-loop test conditions")
    nmi_means = [mean(group) for group in by_condition.values()]
    min_nmi, max_nmi = min(nmi_means), max(nmi_means)
    require((round(min_nmi, 2), round(max_nmi, 2)) == (-0.72, 0.56),
            "Unexpected range of held-out NMI contrasts")
    print(f"Original closed loop: held-out NMI contrasts {min_nmi:+.2f} to {max_nmi:+.2f} pp.")


def verify_earlier_ablations(root):
    studies = (
        ("closed_loop", ("dataset", "outer_seed", "rep", "arm"), "commits", "updates", 48),
        ("mmfd_composition", ("dataset", "outer_seed", "alpha", "condition", "arm"),
         "accepted_rounds", "operator_updates", 72),
    )
    for study, fields, commits_column, updates_column, count in studies:
        rounds_by_path = defaultdict(list)
        for row in read_rows(root / f"{study}_rounds.csv"):
            rounds_by_path[tuple(row[field] for field in fields)].append(row)
        trajectories = read_rows(root / f"{study}_trajectories.csv")
        require(len(rounds_by_path) == len(trajectories) == count, f"{study}: unexpected trajectory count")
        seen = set()
        for row in trajectories:
            key = tuple(row[field] for field in fields)
            require(key not in seen, f"{study}: duplicate trajectory {key}")
            seen.add(key)
            rounds = rounds_by_path[key]
            require(len(rounds) == 60, f"{study}: incomplete round log {key}")
            require({int(round_row["round"]) for round_row in rounds} == set(range(1, 61)),
                    f"{study}: invalid round indices {key}")
            commits = [round_row for round_row in rounds if round_row["status"] == "committed"]
            updates = sum(int(round_row["operator_updates"]) for round_row in commits)
            require(len(commits) == int(row[commits_column]), f"{study}: commit count differs {key}")
            require(updates == int(row[updates_column]), f"{study}: update count differs {key}")
        print(f"{study}: checked commit and replacement totals on {count} trajectories.")


def verify_probe_reuse(root):
    paired = read_rows(root / "paired_probe_audits.csv")
    near = read_rows(root / "near_threshold_audits.csv")
    require(len(paired) == 2592 and len(near) == 594, "Unexpected probe-audit record counts")
    overlimit = [row for row in paired if row["equal_draws"] == "512" and row["exact_phi"]
                 and float(row["exact_phi"]) > float(row["gamma"])
                 and float(row["exact_ell"]) <= float(row["chi"])]
    require(len(overlimit) == 47, "Unexpected count of reassignment-overlimit nominees")
    biases = [100 * mean(float(row[column]) - float(row["exact_phi"]) for row in overlimit)
              for column in ("reuse_phi", "fresh_phi")]
    require(tuple(round(bias, 3) for bias in biases) == (-0.823, -0.079), "Unexpected probe bias")
    print(f"Reassignment bias on 47 nominees: design {biases[0]:+.3f} pp, fresh {biases[1]:+.3f} pp.")

    expected = {"2048": ((20280, 20296), (4.41e-7, 3.67e-7)),
                "8192": ((36224, 36012), (1.19e-9, 4.06e-10))}
    for budget, (expected_feasible, expected_risks) in expected.items():
        groups = [row for row in near if row["audit_draws"] == budget]
        feasible = [row for row in groups if row["cohort"] == "Feasible"]
        excess = [row for row in groups if row["cohort"] != "Feasible"]
        require((len(feasible), len(excess)) == (250, 47), f"Unexpected near-threshold cohorts: {budget}")
        require(all(int(row["repetitions"]) == 256 for row in groups), "Unexpected replay repetitions")
        feasible_passes = tuple(sum(int(row[f"{arm}_pass"]) for row in feasible) for arm in ("reuse", "fresh"))
        false_passes = tuple(sum(int(row[f"{arm}_pass"]) for row in excess) for arm in ("reuse", "fresh"))
        joint_risks = tuple(float(f"{mean(float(row[f'{arm}_joint_probability']) for row in excess):.2e}")
                            for arm in ("reuse", "fresh"))
        require(feasible_passes == expected_feasible and false_passes == (0, 0)
                and joint_risks == expected_risks, f"Unexpected probe-reuse replay results: {budget}")
        print(f"Probe reuse, budget {budget}: zero false admissions per arm across 12,032 trials; "
              f"feasible admissions {feasible_passes[0]:,} and {feasible_passes[1]:,} of 64,000.")


def verify_abstract(data):
    arms = ("VCIA", "No-Composition")
    closed_paths = sorted((data / "reproduction" / "closed").glob("*/rounds.csv"))
    studies = (
        ("WL-TFIDF, moderate perturbation", closed_paths, {"Moderate"},
         ((1440, 1632, 4390912, 301, 0), (1440, 1361, 11796480, 242, 0))),
        ("Scalar MMFD, both conditions", [data / "mmfd_composition_rounds.csv"], {"Clean", "Moderate"},
         ((2160, 9950, 16146432, 1780, 0), (2160, 7509, 17694720, 1321, 0))),
    )
    for name, paths, conditions, expected in studies:
        counts = {arm: Counter() for arm in arms}
        for path in paths:
            for row in read_rows(path):
                arm = row["arm"]
                if arm not in counts or row["condition"] not in conditions:
                    continue
                committed = row["status"] == "committed"
                counts[arm].update(rounds=1, updates=int(row["operator_updates"]),
                                   draws=int(row["audit_draws"]), commits=int(committed),
                                   violations=int(committed and row["finite_feasible"] != "1"))
        fields = ("rounds", "updates", "draws", "commits", "violations")
        actual = tuple(tuple(counts[arm][field] for field in fields) for arm in arms)
        require(actual == expected, f"{name}: unexpected aggregate counts {actual}")
        vcia, control = (counts[arm] for arm in arms)
        gain = 100 * (vcia["updates"] / control["updates"] - 1)
        savings = 100 * (1 - vcia["draws"] / control["draws"])
        print(f"{name}: {vcia['rounds']:,} scheduled rounds per arm; "
              f"{vcia['updates']:,} vs {control['updates']:,} committed replacements (+{gain:.1f}%); "
              f"{vcia['draws']:,} vs {control['draws']:,} audit draws ({savings:.2f}% fewer).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="Extracted verification archive root")
    parser.add_argument("--score-predictions", action="store_true", help="Recompute held-out NMI and ARI")
    options = parser.parse_args()
    root = options.records.resolve()
    verify_hashes(root)
    verify_timing(root / "main_results" / "timing_rounds.csv")
    verify_replay(root / "main_results" / "replay.csv")
    verify_original_closed_loop(root / "main_results")
    data = root / "supplement" / "data"
    verify_earlier_ablations(data)
    verify_probe_reuse(data)
    verify_abstract(data)
    print("Verifying the expanded closed-loop and NCI1 records.", flush=True)
    command = [sys.executable, str(Path(__file__).with_name("verify_reproduction.py")), "--data-dir", str(data)]
    if options.score_predictions:
        command.append("--score-predictions")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

