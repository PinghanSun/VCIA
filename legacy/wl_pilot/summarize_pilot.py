"""Export the historical pilot's matched screening and fixed-state ranking tables."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

ARMS = ("PCOC", "Unscreened")
RANKERS = (
    "PCOC",
    "J-Rank-Feasible",
    "Drift-First",
    "Random-Feasible",
    "No-Count",
    "No-Age",
)
CACHE_MISSES = "new_graph_operator_pairs"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mask(policy):
    candidate = policy["candidate"]
    return tuple(candidate["selected"]) if candidate else None


def content_counts(policy, field):
    category = policy.get(field, {})
    joint = category.get("joint", 0)
    return category.get("operator_only", 0) + joint, category.get(
        "claim_only", 0
    ) + joint


def read_logs(root):
    manifest = read_json(root / "manifest.json")
    index = read_json(root / "run_index.json")
    expected = set(product(manifest["seeds"], manifest["regimes"], manifest["arms"]))
    seen = set()
    trajectories = {}
    for run in index:
        key = (run["seed"], run["regime"], run["arm"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicated pilot trajectory {key}")
        seen.add(key)
        log = root / run["path"] / "rounds.jsonl"
        with log.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if [row["round"] for row in rows] != list(range(1, manifest["rounds"] + 1)):
            raise ValueError(f"Incomplete or unordered rounds in {log}")
        if any((row["seed"], row["regime"], row["arm"]) != key for row in rows):
            raise ValueError(f"Mixed trajectories in {log}")
        trajectories[key] = rows
    if seen != expected or set(manifest["arms"]) != set(ARMS):
        raise ValueError(f"Incomplete matched screening archive in {root}")
    return manifest, trajectories


def summarize(root):
    manifest, trajectories = read_logs(root)
    dataset = manifest.get("dataset_name", "PROTEINS")
    screening = defaultdict(Counter)
    ranking = defaultdict(Counter)
    for seed, regime in product(manifest["seeds"], manifest["regimes"]):
        screened = trajectories[seed, regime, "PCOC"]
        unscreened = trajectories[seed, regime, "Unscreened"]
        for left, right in zip(screened, unscreened, strict=True):
            online_screened = left["policies"]["PCOC"]
            online_unscreened = right["policies"]["Unscreened"]
            matched_mask = mask(online_screened) == mask(online_unscreened)
            matched_status = online_screened["status"] == online_unscreened["status"]
            matched_state = left["executed_state_hash"] == right["executed_state_hash"]
            matched_quality = (
                online_screened["post_decision_inner_validation"]
                == online_unscreened["post_decision_inner_validation"]
            )
            for arm, trace in zip(ARMS, (left, right), strict=True):
                decision = trace["policies"][arm]
                operators, claims = content_counts(
                    decision, "committed_content_updates"
                )
                counts = screening[regime, arm]
                verification_passed = decision["status"] == "accepted"
                counts["scheduled_rounds"] += 1
                counts["composition_evaluations"] += trace[
                    "online_selection_compositions"
                ]
                counts["excluded_feasible_masks"] += sum(
                    candidate["unscreened_rejection"] is None
                    and not candidate["screened"]
                    for candidate in trace["candidates"]
                )
                counts["design_operator_evaluations"] += trace["design_cache"][
                    CACHE_MISSES
                ]
                counts["verification_operator_evaluations"] += decision.get(
                    "verify_cache_delta", {}
                ).get(CACHE_MISSES, 0)
                counts["verification_passes"] += verification_passed
                counts["no_candidate_rounds"] += decision["status"] == "no_candidate"
                counts["verifier_holds"] += (
                    decision["status"] == "verification_rejected"
                )
                counts["finite_audit_exceedances_on_commit"] += (
                    verification_passed
                    and not decision["finite_audit_law"]["within_tolerance"]
                )
                counts["committed_operator_updates"] += operators
                counts["committed_claim_updates"] += claims
                counts["matched_mask_rounds"] += matched_mask
                counts["matched_status_rounds"] += matched_status
                counts["matched_state_rounds"] += matched_state
                counts["matched_quality_rounds"] += matched_quality
                if trace["round"] == manifest["rounds"]:
                    counts["endpoints"] += 1
                    for metric, value in decision[
                        "post_decision_inner_validation"
                    ].items():
                        counts[f"sum_endpoint_{metric}"] += value

            main_mask = mask(online_screened)
            for policy in RANKERS:
                alt = left["policies"][policy]
                alt_mask = mask(alt)
                if (main_mask is None) != (alt_mask is None):
                    raise ValueError(
                        "Rankers disagree about the shared family's emptiness"
                    )
                operators, claims = content_counts(alt, "accepted_candidate_updates")
                for group in (regime, "All"):
                    counts = ranking[group, policy]
                    counts["scheduled_contexts"] += 1
                    counts["candidate_contexts"] += main_mask is not None
                    counts["priority_count_active_contexts"] += left[
                        "priority_count_active"
                    ]
                    counts["priority_age_active_contexts"] += left[
                        "priority_age_active"
                    ]
                    counts["mask_changes_vs_pcoc"] += alt_mask != main_mask
                    counts["status_changes_vs_pcoc"] += (
                        alt["status"] != online_screened["status"]
                    )
                    counts["verification_passes"] += alt["status"] == "accepted"
                    counts["would_accept_operator_updates"] += operators
                    counts["would_accept_claim_updates"] += claims

    screening_rows = []
    for (regime, arm), counts in screening.items():
        means = {
            f"mean_endpoint_{metric}": counts.pop(f"sum_endpoint_{metric}")
            / counts["endpoints"]
            for metric in ("nmi", "ari", "acc", "macro_f1")
        }
        screening_rows.append(
            {"dataset": dataset, "regime": regime, "arm": arm, **counts, **means}
        )
    ranking_rows = [
        {"dataset": dataset, "regime": regime, "policy": policy, **counts}
        for (regime, policy), counts in ranking.items()
    ]
    return dataset, screening_rows, ranking_rows


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="+", help="Archived or regenerated pilot roots"
    )
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory")

    datasets, screening, ranking = set(), [], []
    for path in args.results:
        dataset, screen_rows, rank_rows = summarize(path)
        if dataset in datasets:
            parser.error(f"Provide only one pilot root for {dataset}")
        datasets.add(dataset)
        screening.extend(screen_rows)
        ranking.extend(rank_rows)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "matched_screening.csv", screening)
    write_csv(args.out / "fixed_state_ranking.csv", ranking)


if __name__ == "__main__":
    main()
