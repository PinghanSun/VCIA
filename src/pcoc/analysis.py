"""Summarize PCOC runs and calibrate verification on frozen inner evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.stats import binom, t

GROUP_FIELDS = ("dataset", "phase", "regime", "segment", "arm")
PAIR_FIELDS = GROUP_FIELDS[:-1]
TRAJECTORY_FIELDS = ("dataset", "phase", "partition_seed", "seed", "regime", "arm")
QUALITY_FIELDS = ("nmi", "ari", "acc", "macro_f1")
DEFAULT_DRAWS = (2048, 4096, 8192, 16384, 32768, 65536)
DEFAULT_RATES = (0.0, 0.003, 0.005, 0.008, 0.01, 0.015)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def record_key(record: dict, fields: Sequence[str]) -> tuple:
    return tuple(record[key] for key in fields)


def grouped(records: Iterable[dict], fields: Sequence[str]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in records:
        groups[record_key(row, fields)].append(row)
    return dict(sorted(groups.items()))


def read_trajectory_markers(root: Path, source_dir: Path, single_file: bool):
    planned, sources = {}, []

    def read(path):
        source = path.read_bytes()
        sources.append(
            {
                "path": path.relative_to(source_dir).as_posix(),
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        )
        return json.loads(source.decode("utf-8-sig"))

    def round_path(directory, entry):
        seed_dir = f"p{entry['partition_seed']}_i{entry['seed']}"
        return (
            directory
            / entry["dataset"]
            / seed_dir
            / entry["regime"]
            / entry["arm"]
            / "rounds.jsonl"
        )

    if not single_file:
        for path in sorted(root.rglob("manifest.json")):
            manifest = read(path)
            study = manifest["study"]
            for job in manifest["plan"]["jobs"]:
                planned[round_path(path.parent, job)] = {
                    **job,
                    "phase": study["phase"],
                    "scheduled_rounds": study["rounds"],
                    "status": "registered",
                }
        for path in sorted(root.rglob("run_index.json")):
            for job in read(path):
                rounds = round_path(path.parent, job)
                planned[rounds] = {**planned.get(rounds, {}), **job}
    failures = (
        [root.with_name("failure.json")]
        if single_file
        else sorted(root.rglob("failure.json"))
    )
    for path in failures:
        if path.is_file():
            marker = read(path)
            rounds = root if single_file else path.with_name("rounds.jsonl")
            planned[rounds] = {**planned.get(rounds, {}), **marker, "status": "failed"}
    return planned, sources


def round_log(path):
    source = path.read_bytes() if path.is_file() else None
    rows = []
    if source is not None:
        try:
            for line in source.decode("utf-8-sig").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except ValueError as error:
            return source, rows, f"Invalid round record in {path}: {error}"
    return source, rows, None


def log_issue(path, rows, entry):
    if not rows:
        return f"No round records in {path}"
    identity = record_key(entry, TRAJECTORY_FIELDS)
    if any(record_key(row, TRAJECTORY_FIELDS) != identity for row in rows):
        return f"Mixed trajectory identities in {path}"
    if [row["round"] for row in rows] != list(range(1, len(rows) + 1)):
        return f"Incomplete or inconsistent round numbering in {path}"
    if entry["scheduled_rounds"] != len(rows):
        return f"Unexpected round count in {path}"
    return None


def completion_counts(entries):
    return {
        "trajectories": len(entries),
        "statuses": dict(Counter(entry["status"] for entry in entries)),
        "scheduled_rounds_all_segments": sum(
            entry["scheduled_rounds"] for entry in entries
        ),
        "recorded_rounds_all_segments": sum(
            entry["recorded_rounds"] for entry in entries
        ),
        "included_metric_rounds": sum(
            entry["included_metric_rounds"] for entry in entries
        ),
    }


def load_runs(root: Path, *, phase: str | None = None, arms=(), segments=()):
    """Read complete trajectories and preserve unavailable jobs in the report denominator."""
    single_file = root.is_file()
    source_dir = root.parent if single_file else root
    planned, metadata_sources = read_trajectory_markers(root, source_dir, single_file)
    leaves = {root} if single_file else set(root.rglob("rounds.jsonl"))
    records, sources, completion = [], [], []
    seen = set()
    for path in sorted(leaves | planned.keys()):
        marker = planned.get(path)
        if marker and (
            (phase is not None and marker["phase"] != phase)
            or (arms and marker["arm"] not in arms)
        ):
            continue
        source, rows, issue = round_log(path)
        if marker is None and (issue or not rows):
            raise ValueError(issue or f"No round records in {path}")
        entry = marker if marker is not None else rows[0]
        if (phase is not None and entry["phase"] != phase) or (
            arms and entry["arm"] not in arms
        ):
            continue
        identity = record_key(entry, TRAJECTORY_FIELDS)
        if identity in seen:
            raise ValueError(f"Duplicated trajectory identity {identity}")
        seen.add(identity)
        scheduled_rounds = entry.get("scheduled_rounds", len(rows))
        issue = issue or log_issue(
            path, rows, {**entry, "scheduled_rounds": scheduled_rounds}
        )
        status = "complete"
        if marker is not None and marker["status"] == "failed":
            status = "failed"
            issue = marker.get("failure") or issue
        elif (
            marker is not None
            and marker["status"] == "registered"
            and not (path.parent / "summary.json").is_file()
        ):
            status = "incomplete" if path.parent.is_dir() else "unstarted"
            issue = (
                "Run completion was not recorded" if status == "incomplete" else None
            )
        elif issue:
            if marker is None:
                raise ValueError(issue)
            status = "incomplete"
        selected = (
            [row for row in rows if not segments or row["segment"] in segments]
            if status == "complete"
            else []
        )
        records.extend(selected)
        relative = path.relative_to(source_dir).as_posix()
        if source is not None:
            sources.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(source).hexdigest(),
                    "rounds": len(rows),
                }
            )
        completion.append(
            {
                **dict(zip(TRAJECTORY_FIELDS, identity, strict=True)),
                "job": entry.get("job"),
                "path": relative,
                "status": status,
                "reason": issue,
                "scheduled_rounds": scheduled_rounds,
                "recorded_rounds": len(rows),
                "included_metric_rounds": len(selected),
            }
        )
    if not completion:
        raise ValueError(f"No matching trajectories under {root}")
    completion_fields = ("dataset", "phase", "regime", "arm")
    report = {
        "scope": {
            "phase": phase,
            "arms": list(arms),
            "metric_segments": list(segments) or ["all"],
        },
        "metric_population": "complete trajectories only",
        **completion_counts(completion),
        "groups": [
            {
                **dict(zip(completion_fields, key, strict=True)),
                **completion_counts(values),
            }
            for key, values in grouped(completion, completion_fields).items()
        ],
        "trajectory_details": completion,
        "metadata_sources": metadata_sources,
    }
    return records, sources, report


def with_all_segments(records):
    return records + [
        dict(row, segment="all") for row in records if row["segment"] != "all"
    ]


def nominated(row):
    return row["selected"] is not None


def passed(row):
    return row["status"] == "accepted"


def total(records, field):
    return math.fsum(row[field] for row in records)


def cost_totals(records):
    costs = defaultdict(list)
    for row in records:
        for name, value in row["cost"].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                costs[name].append(value)
    return {name: math.fsum(values) for name, values in sorted(costs.items())}


def endpoints(records):
    last = max(records, key=lambda row: row["round"])
    costs = cost_totals(records)
    measured = (
        "screen_seconds",
        "composition_seconds",
        "composition_evaluations",
        "design_seconds",
        "verification_seconds",
        "verification_attempts",
        "verify_draws",
        "verify_unique_graphs",
        "new_graph_operator_pairs",
        "transport_seconds",
        "wire_bytes",
    )
    return {
        **{f"end_{name}": (last["quality"] or {}).get(name) for name in QUALITY_FIELDS},
        "operator_updates": total(records, "operator_updates"),
        "claim_updates": total(records, "claim_updates"),
        "hold_rate": 1 - sum(map(passed, records)) / len(records),
        "screen_and_composition_seconds": (
            costs["screen_seconds"] + costs["composition_seconds"]
            if {"screen_seconds", "composition_seconds"} <= costs.keys()
            else None
        ),
        **{name: costs.get(name) for name in measured},
        "online_round_seconds": (
            costs["round_seconds"] - costs["offline_seconds"] - costs["quality_seconds"]
            if {"round_seconds", "offline_seconds", "quality_seconds"} <= costs.keys()
            else None
        ),
    }


def paired_statistics(pairs):
    """Average nested initializations within partition before inference."""
    groups = defaultdict(list)
    for partition, difference in pairs:
        groups[partition].append(difference)
    means = {p: fmean(groups[p]) for p in sorted(groups)}
    n = len(means)
    mean = fmean(means.values()) if n else None
    interval = None
    if n > 1:
        error = float(t.ppf(0.975, n - 1)) * stdev(means.values()) / math.sqrt(n)
        interval = [mean - error, mean + error]
    return {
        "pairs": len(pairs),
        "partitions": n,
        "partition_deltas": means,
        "mean_difference": mean,
        "partition_t_95": interval,
    }


def paired_contrasts(records, reference, other=None, *, factor="arm"):
    contexts = defaultdict(dict)
    context_fields = tuple(field for field in GROUP_FIELDS if field != factor)
    dimensions = (*context_fields, "partition_seed", "seed", factor)
    for key, rows in grouped(records, dimensions).items():
        group, partition_init, condition = key[:-3], key[-3:-1], key[-1]
        contexts[group].setdefault(condition, {})[partition_init] = rows
    contrasts = []
    for group, conditions in sorted(contexts.items()):
        if reference not in conditions:
            continue
        for condition, current in sorted(conditions.items()):
            if condition == reference or other is not None and condition != other:
                continue
            baseline = conditions[reference]
            common = set(current) & set(baseline)
            differences = defaultdict(list)
            changes = Counter()
            schedule_mismatches = []
            for key in sorted(common):
                rr = {row["round"]: row for row in baseline[key]}
                cr = {row["round"]: row for row in current[key]}
                if rr.keys() != cr.keys():
                    schedule_mismatches.append(list(key))
                    continue
                ref = endpoints(baseline[key])
                for name, value in endpoints(current[key]).items():
                    if ref[name] is not None and value is not None:
                        differences[name].append((key[0], value - ref[name]))
                for step, left in rr.items():
                    right = cr[step]
                    diff = left["selected"] != right["selected"]
                    left_hash = left.get("nominated_state_hash")
                    right_hash = right.get("nominated_state_hash")
                    changes["rounds"] += 1
                    changes["mask"] += diff
                    changes["execution"] += (
                        left["executed_state_hash"] != right["executed_state_hash"]
                    )
                    changes["decision"] += left["status"] != right["status"]
                    if nominated(left) or nominated(right):
                        changes["nonempty"] += 1
                        changes["nonempty_mask"] += diff
                        left_known = not nominated(left) or left_hash is not None
                        right_known = not nominated(right) or right_hash is not None
                        if left_known and right_known:
                            changes["comparable_nomination"] += 1
                            changes["different_nomination_state"] += (
                                left_hash != right_hash
                            )
                            if left_hash == right_hash:
                                changes["same_nomination_state"] += 1
                                changes["same_state_different_disposition"] += (
                                    left["status"] != right["status"]
                                )
            contrasts.append(
                {
                    **dict(zip(context_fields, group, strict=True)),
                    factor: condition,
                    "reference": reference,
                    "difference_direction": f"{factor} minus reference",
                    "pairs_with_equal_schedule": len(common) - len(schedule_mismatches),
                    f"{factor}_only_partition_init": sorted(
                        set(current) - set(baseline)
                    ),
                    "reference_only_partition_init": sorted(
                        set(baseline) - set(current)
                    ),
                    "schedule_mismatches": schedule_mismatches,
                    "comparable_rounds": changes["rounds"],
                    "different_nomination_masks": changes["mask"],
                    "different_executed_states": changes["execution"],
                    "different_dispositions": changes["decision"],
                    "nomination_mask_change_rate": ratio(
                        changes["mask"], changes["rounds"]
                    ),
                    "nomination_mask_change_rate_nonempty": ratio(
                        changes["nonempty_mask"], changes["nonempty"]
                    ),
                    "comparable_nominated_states": changes["comparable_nomination"],
                    "different_nominated_states": changes["different_nomination_state"],
                    "nominated_state_change_rate": ratio(
                        changes["different_nomination_state"],
                        changes["comparable_nomination"],
                    ),
                    "same_nominated_state_different_disposition": changes[
                        "same_state_different_disposition"
                    ],
                    "same_nominated_states": changes["same_nomination_state"],
                    "endpoints": {
                        name: paired_statistics(v)
                        for name, v in sorted(differences.items())
                    },
                }
            )
    return contrasts


def rejection_kind(row):
    evidence = row.get("verification") or {}
    names = [name for name in ("drift", "assignment") if evidence.get(f"{name}_failed")]
    return "joint" if len(names) == 2 else names[0] if names else "unclassified"


def measured_summary(records):
    statuses = Counter(row["status"] for row in records)
    nominations = [row for row in records if nominated(row)]
    verified = [
        row
        for row in records
        if row.get("verification") is not None
        and row["verification"].get("route") != "No-Verify"
    ]
    exact = [row for row in nominations if row.get("nominated_exact") is not None]
    admissible = [row for row in exact if row["nominated_exact"]["within_tolerance"]]
    outside = [row for row in exact if not row["nominated_exact"]["within_tolerance"]]
    admissible_commits = [row for row in admissible if passed(row)]
    exact_commits = sum(map(passed, exact))
    complete_commit_coverage = exact_commits == statuses["accepted"]
    outside_commits = sum(map(passed, outside))
    admissible_changes = [
        sum(row["content_updates"].values()) for row in admissible_commits
    ]
    design_counts = [
        row["nominated_H"] for row in nominations if row.get("nominated_H") is not None
    ]
    size = len(records)
    ends = [
        endpoints(rows)
        for rows in grouped(records, ("partition_seed", "seed")).values()
    ]
    by_quality = {
        name: [end[f"end_{name}"] for end in ends if end[f"end_{name}"] is not None]
        for name in QUALITY_FIELDS
    }
    families = [
        row
        for row in records
        if (row.get("eligible_family_size", row.get("family_size")) or 0) > 0
    ]
    ages = np.asarray(
        [age for row in records for age in row.get("content_ages", [])], dtype=float
    )
    measured_attempts = sum(
        row["cost"].get("verification_attempts", 0) for row in records
    )
    content = Counter()
    for row in records:
        content.update(row.get("content_updates", {}))
    return {
        "scheduled_rounds": size,
        "trajectories": len(ends),
        "partitions": len({row["partition_seed"] for row in records}),
        "outcomes": dict(statuses),
        "nominations": len(nominations),
        "nominations_per_scheduled_round": len(nominations) / size,
        "holds": size - statuses["accepted"],
        "successful_commits": statuses["accepted"],
        "hold_rate": 1 - statuses["accepted"] / size,
        "no_candidate_reasons": dict(
            Counter(
                row.get("no_candidate_reason")
                or row.get("hold_reason")
                or "unspecified"
                for row in records
                if row["status"] == "no_candidate"
            )
        ),
        "verifier_attempts": measured_attempts,
        "audited_rounds": len(verified),
        "verifier_rejection_rate": ratio(
            statuses["verification_rejected"], len(verified)
        ),
        "verifier_rejections": dict(
            Counter(
                rejection_kind(row)
                for row in records
                if row["status"] == "verification_rejected"
            )
        ),
        "exact_audited_nominations": len(exact),
        "exact_admissible_nominations": len(admissible),
        "exact_admissible_nominations_held": sum(not passed(row) for row in admissible),
        "exact_admissible_nominations_rejected": sum(
            row["status"] == "verification_rejected" for row in admissible
        ),
        "exact_audited_commits": exact_commits,
        "out_of_tolerance_commits": outside_commits,
        "r_all": outside_commits / size if complete_commit_coverage else None,
        "r_commit": ratio(outside_commits, exact_commits)
        if complete_commit_coverage
        else None,
        "U_adm": sum(admissible_changes) if complete_commit_coverage else None,
        "C_adm": sum(count > 0 for count in admissible_changes)
        if complete_commit_coverage
        else None,
        "design_detectable_nominations": len(design_counts),
        "nomination_H_distribution": dict(sorted(Counter(design_counts).items())),
        "operator_updates": total(records, "operator_updates"),
        "operator_updates_per_scheduled_round": total(records, "operator_updates")
        / size,
        "claim_updates": total(records, "claim_updates"),
        "canonical_content_update_types": dict(content),
        "canonical_content_updates": content.total(),
        "canonical_content_updates_per_scheduled_round": content.total() / size,
        "observable_operator_updates": sum(
            row.get("observable_operator_updates", 0) for row in records
        )
        if all(
            not row["operator_updates"] or "observable_operator_updates" in row
            for row in records
        )
        else None,
        "content_age": {
            "mean": float(ages.mean()),
            "p95": float(np.quantile(ages, 0.95)),
            "max": int(ages.max()),
        }
        if ages.size
        else None,
        "accepted_without_content_update": sum(
            passed(row) and row["operator_updates"] == 0 and row["claim_updates"] == 0
            for row in records
        ),
        "admissible_committed_operator_updates": total(
            admissible_commits, "operator_updates"
        ),
        "nonempty_eligible_family_rounds": len(families),
        "count_priority_active_rate": ratio(
            sum(bool(row.get("priority_count_active")) for row in families),
            len(families),
        ),
        "age_priority_active_rate": ratio(
            sum(bool(row.get("priority_age_active")) for row in families), len(families)
        ),
        "trajectory_end_quality": {
            name: {"n": len(scores), "mean": fmean(scores) if scores else None}
            for name, scores in by_quality.items()
        },
        "cost_totals": cost_totals(records),
    }


def fixed_ranking(records, reference):
    """Compare decisions made on the same frozen state and audit draw."""
    report = []
    logged = [row for row in records if row.get("fixed_ranking") is not None]
    for key, rows in grouped(logged, GROUP_FIELDS).items():
        choices = defaultdict(Counter)
        for row in rows:
            decisions = row["fixed_ranking"]
            if reference not in decisions:
                raise ValueError(
                    f"Fixed ranking trace is missing reference {reference}"
                )
            baseline = decisions[reference]
            for arm, decision in decisions.items():
                counts = choices[arm]
                selected = decision["selected"]
                accepted = bool(decision["accepted"])
                if accepted and selected is None:
                    raise ValueError("Fixed ranking accepted an empty nomination")
                counts["contexts"] += 1
                counts["nominations"] += selected is not None
                counts["changed_nomination_masks"] += selected != baseline["selected"]
                counts["changed_verification_decisions"] += (
                    accepted != baseline["accepted"]
                )
                counts["verified_passes"] += accepted
                counts["would_pass_operator_updates"] += (
                    decision["operator_updates"] if accepted else 0
                )
                counts["would_pass_claim_updates"] += (
                    decision.get("claim_updates", 0) if accepted else 0
                )
        for arm, counts in sorted(choices.items()):
            if counts["contexts"] != len(rows):
                raise ValueError(f"Fixed ranking trace is incomplete for {arm}")
            report.append(
                {
                    **dict(
                        zip((*PAIR_FIELDS, "snapshot_source_arm"), key, strict=True)
                    ),
                    "ranking": arm,
                    "reference": reference,
                    **dict(counts),
                }
            )
    return report


def summarize(records, reference):
    segment_rows = with_all_segments(records)
    return {
        "groups": [
            {**dict(zip(GROUP_FIELDS, key, strict=True)), **measured_summary(rows)}
            for key, rows in grouped(segment_rows, GROUP_FIELDS).items()
        ],
        "paired_contrasts": paired_contrasts(segment_rows, reference),
        "paired_regime_contrasts": paired_contrasts(
            segment_rows, "clean", factor="regime"
        ),
        "fixed_snapshot_ranking": fixed_ranking(segment_rows, "Unscreened"),
    }


def mask_losses(masks, predicate):
    family = [mask for mask in masks if mask["H"] > 0 and predicate(mask)]
    retained = [mask for mask in family if mask["screened"]]
    maximum = max((mask["operator_updates"] for mask in family), default=0)
    retained_maximum = max((mask["operator_updates"] for mask in retained), default=0)
    return {
        "reference_masks": len(family),
        "retained_masks": len(retained),
        "removed_masks": len(family) - len(retained),
        "nonempty_reference_rounds": int(bool(family)),
        "all_reference_masks_removed": int(bool(family and not retained)),
        "maximum_canonical_updates_reference": maximum,
        "maximum_canonical_updates_screened": retained_maximum,
        "maximum_canonical_updates_lost": maximum - retained_maximum,
    }


def screening(records, screened_arm, unscreened_arm):
    """Report offline opportunity losses and online paired costs."""
    records = [row for row in records if row["arm"] in {screened_arm, unscreened_arm}]
    families = {
        "design_feasible": lambda mask: mask["feasible"],
        "census_admissible": lambda mask: mask["exact"]["within_tolerance"],
        "both": lambda mask: mask["feasible"] and mask["exact"]["within_tolerance"],
    }
    exact_rows = []
    for row in records:
        masks = row.get("candidates")
        if masks is None:
            continue
        if any(mask.get("exact") is None for mask in masks):
            raise ValueError(
                "All-mask screening analysis requires exact evidence for every enumerated mask"
            )
        if len({tuple(mask["selected"]) for mask in masks}) != len(masks):
            raise ValueError(
                "Duplicate masks in the census of an all-mask screening round"
            )
        exact_rows.append(
            {
                **{
                    key: row[key]
                    for key in (*GROUP_FIELDS, "partition_seed", "seed", "round")
                },
                "evaluated_masks": len(masks),
                "offline_seconds": row["cost"].get("offline_seconds"),
                "families": {
                    name: mask_losses(masks, predicate)
                    for name, predicate in families.items()
                },
            }
        )
    aggregate = []
    for key, rows in grouped(with_all_segments(exact_rows), GROUP_FIELDS).items():
        combined = {}
        for name in families:
            counts = Counter()
            for row in rows:
                counts.update(row["families"][name])
            combined[name] = {
                **dict(counts),
                "removed_mask_rate": ratio(
                    counts["removed_masks"], counts["reference_masks"]
                ),
                "all_removed_rate_given_nonempty": ratio(
                    counts["all_reference_masks_removed"],
                    counts["nonempty_reference_rounds"],
                ),
                "mean_maximum_updates_lost_given_nonempty": ratio(
                    counts["maximum_canonical_updates_lost"],
                    counts["nonempty_reference_rounds"],
                ),
            }
        times = [
            row["offline_seconds"] for row in rows if row["offline_seconds"] is not None
        ]
        aggregate.append(
            {
                **dict(zip(GROUP_FIELDS, key, strict=True)),
                "offline_census_rounds": len(rows),
                "evaluated_masks": sum(row["evaluated_masks"] for row in rows),
                "offline_seconds_total": math.fsum(times) if times else None,
                "families": combined,
            }
        )
    return {
        "offline_all_mask_census": aggregate,
        "offline_rounds": exact_rows,
        "paired_execution": paired_contrasts(
            with_all_segments(records), unscreened_arm, other=screened_arm
        ),
    }


def binary_cutoff(draws, gamma, alpha):
    """Return the largest flip count passing the exact one-sided gate."""
    low, high = -1, draws
    while high - low > 1:
        count = (high + low) // 2
        if binom.cdf(count, draws, gamma) <= alpha:
            low = count
        else:
            high = count
    return low


def validate_settings(budgets, alpha, chi, gamma):
    if not budgets or len(set(budgets)) != len(budgets) or min(budgets) < 2:
        raise ValueError("Use distinct verification budgets with at least two draws")
    if not (0 < alpha < 1 and 0 < gamma < 1 and 0 < chi <= 1):
        raise ValueError(
            "Expected alpha and gamma in (0, 1), and normalized chi in (0, 1]"
        )


def analytic(budgets, rates, alpha, chi, gamma):
    validate_settings(budgets, alpha, chi, gamma)
    if any(not 0 <= p <= 1 for p in rates):
        raise ValueError("Analytic flip rates must lie in [0, 1]")
    results = []
    for draws in budgets:
        cutoff = binary_cutoff(draws, gamma, alpha)
        floor = 7 * math.log(2 / alpha) / (3 * (draws - 1))
        results.append(
            {
                "draws": draws,
                "binary_max_flips": cutoff,
                "normalized_bernstein_floor": floor,
                "max_constant_normalized_drift_passing": chi - floor
                if floor <= chi
                else None,
                "exact_binary_pass_probability": {
                    str(p): float(binom.cdf(cutoff, draws, p)) for p in rates
                },
            }
        )
    return {
        "alpha_per_component": alpha,
        "chi_normalized": chi,
        "gamma": gamma,
        "budgets": results,
    }


def prepare_evidence(row):
    evidence = row.get("fixed_evidence")
    if evidence is None:
        raise ValueError(
            f"Nominated inner round {row['round']} in {row['arm']} needs fixed_evidence"
        )
    scale = float(evidence["output_range"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("The fixed evidence output range must be positive")
    native_drift = np.asarray(evidence["distances"], dtype=np.float64)
    drift = native_drift / scale
    flips = np.asarray(evidence["flips"])
    if (
        drift.ndim != 1
        or drift.size == 0
        or not np.isfinite(drift).all()
        or np.any(drift < 0)
        or np.any(drift > 1 + 1e-12)
        or flips.dtype != np.bool_
        or drift.shape != flips.shape
    ):
        raise ValueError(
            "Fixed evidence needs paired distances in range and boolean flips"
        )
    native_chi = float(evidence["chi"])
    alpha, chi, gamma = (
        float(evidence["alpha"]),
        native_chi / scale,
        float(evidence["gamma"]),
    )
    validate_settings([2], alpha, chi, gamma)
    unique = evidence.get("design_unique")
    operators = evidence.get("distinct_operators")
    if unique is not None and not 0 <= unique <= len(drift):
        raise ValueError("Design graphs exceed the registered finite audit target")
    if operators is not None and operators < 1:
        raise ValueError("A nomination needs at least one distinct operator")
    return {
        "round": row,
        "drift": np.minimum(drift, 1),
        "flips": flips,
        "design_unique": unique,
        "distinct_operators": operators,
        "alpha": alpha,
        "chi": chi,
        "gamma": gamma,
        "exact_admissible": float(native_drift.mean()) <= native_chi
        and float(flips.mean()) <= gamma,
    }


def estimate_group(contexts, draws, batches, seed):
    """Use paired multinomial counts for a given support size and budget."""
    first = contexts[0]
    size = len(first["drift"])
    alpha, chi, gamma = (first[name] for name in ("alpha", "chi", "gamma"))
    cutoff = binary_cutoff(draws, gamma, alpha)
    rng = np.random.default_rng(np.random.SeedSequence([seed, size, draws]))
    log_term = math.log(2 / alpha)
    additive = 7 * log_term / (3 * (draws - 1))
    laws = defaultdict(list)
    for index, context in enumerate(contexts):
        laws[context["drift"].tobytes(), context["flips"].tobytes()].append(index)
    representatives = [contexts[indices[0]] for indices in laws.values()]
    distances = np.column_stack([c["drift"] for c in representatives])
    flips = np.column_stack([c["flips"] for c in representatives]).astype(float)
    means = distances.mean(axis=0)
    centered = distances - means
    deviations = centered**2
    passes = np.zeros((len(representatives), 3), dtype=np.int64)
    probabilities = np.full(size, 1 / size)
    for start in range(0, batches, 256):
        counts = rng.multinomial(
            draws, probabilities, size=min(256, batches - start)
        ).astype(float)
        mean_delta = counts @ centered / draws
        variance = np.maximum(
            0, (counts @ deviations - draws * mean_delta**2) / (draws - 1)
        )
        upper = np.minimum(
            1.0,
            means + mean_delta + np.sqrt(2 * log_term * variance / draws) + additive,
        )
        drift_pass = upper <= chi
        flip_pass = counts @ flips <= cutoff
        passes += np.column_stack(
            (drift_pass.sum(0), flip_pass.sum(0), (drift_pass & flip_pass).sum(0))
        )
    expanded = np.zeros((len(contexts), 3), dtype=np.int64)
    for indices, outcomes in zip(laws.values(), passes, strict=True):
        expanded[indices] = outcomes
    return expanded, cutoff


def estimate_candidates(
    records, budgets, batches, seed, interior_drift, interior_flip, sd_edges
):
    available = defaultdict(list)
    for row in records:
        if not nominated(row):
            continue
        context = prepare_evidence(row)
        key = (
            len(context["drift"]),
            context["alpha"],
            context["chi"],
            context["gamma"],
        )
        available[key].append(context)
    for key, contexts in sorted(available.items()):
        size, alpha, chi, gamma = key
        for draws in budgets:
            counts, cutoff = estimate_group(contexts, draws, batches, seed)
            coverage = -math.expm1(draws * math.log1p(-1 / size)) if size > 1 else 1.0
            for context, (drift_count, flip_count, joint_count) in zip(
                contexts, counts, strict=True
            ):
                row = context["round"]
                mean = float(context["drift"].mean())
                sd = float(context["drift"].std())
                flips = int(context["flips"].sum())
                phi = flips / size
                joint = int(joint_count) / batches
                exact_pass = context["exact_admissible"]
                if (
                    row.get("nominated_exact")
                    and exact_pass != row["nominated_exact"]["within_tolerance"]
                ):
                    raise ValueError(
                        "Finite evidence and logged exact admissibility disagree"
                    )
                design_unique = context["design_unique"]
                operators = context["distinct_operators"]
                yield {
                    **{
                        name: row[name]
                        for name in (*GROUP_FIELDS, "partition_seed", "seed", "round")
                    },
                    "draws": draws,
                    "alpha": alpha,
                    "chi_normalized": chi,
                    "gamma": gamma,
                    "target_size": size,
                    "flips": flips,
                    "flip_rate": phi,
                    "normalized_drift_mean": mean,
                    "normalized_drift_sd": sd,
                    "normalized_drift_sd_stratum": int(
                        np.searchsorted(sd_edges, sd, side="right")
                    ),
                    "at_operating_margin": mean <= interior_drift
                    and phi <= interior_flip,
                    "exact_admissible": exact_pass,
                    "actual_accepted": passed(row),
                    "actual_nmi": (row["quality"] or {}).get("nmi"),
                    "actual_status": row["status"],
                    "actual_operator_updates": row["operator_updates"],
                    "candidate_operator_updates": row.get("candidate_operator_updates"),
                    "exact_binary_pass_probability": float(
                        binom.cdf(cutoff, draws, phi)
                    ),
                    "drift_mc_pass_probability": int(drift_count) / batches,
                    "flip_mc_pass_probability": int(flip_count) / batches,
                    "joint_mc_pass_probability": joint,
                    "joint_mc_standard_error": math.sqrt(joint * (1 - joint) / batches),
                    "expected_distinct_graphs": size * coverage,
                    "expected_new_graph_operator_pairs": (
                        (size - design_unique) * operators * coverage
                        if design_unique is not None and operators is not None
                        else None
                    ),
                }


def calibration_summary(scheduled, candidates, strata_count, joint_target):
    admissible = [row for row in candidates if row["exact_admissible"]]
    outside = [row for row in candidates if not row["exact_admissible"]]
    interior = [row for row in candidates if row["at_operating_margin"]]
    strata = {}
    for bin_id in range(strata_count):
        sample = [
            row for row in interior if row["normalized_drift_sd_stratum"] == bin_id
        ]
        strata[bin_id] = {
            "candidates": len(sample),
            "minimum_joint_probability_estimate": min(
                (row["joint_mc_pass_probability"] for row in sample), default=None
            ),
            "fraction_meeting_joint_target_by_point_estimate": ratio(
                sum(row["joint_mc_pass_probability"] >= joint_target for row in sample),
                len(sample),
            ),
        }
    expected_passes = total(candidates, "joint_mc_pass_probability")
    certifiable_updates = [
        row for row in candidates if row["candidate_operator_updates"] is not None
    ]
    n = len(scheduled)
    return {
        "scheduled_rounds": n,
        "nominations": len(candidates),
        "without_nomination": n - len(candidates),
        "inside_operating_margin": len(interior),
        "inside_margin_per_scheduled_round": len(interior) / n,
        "exact_admissible_nominations": len(admissible),
        "actual_admissible_held": sum(not row["actual_accepted"] for row in admissible),
        "actual_out_of_tolerance_passes": sum(
            row["actual_accepted"] for row in outside
        ),
        "actual_accepted_per_scheduled_round": sum(map(passed, scheduled)) / n,
        "actual_operator_updates_per_scheduled_round": total(
            scheduled, "operator_updates"
        )
        / n,
        "mean_joint_pass_probability_given_candidate": ratio(
            expected_passes, len(candidates)
        ),
        "mean_joint_pass_probability_per_scheduled_round": expected_passes / n,
        "expected_outside_tolerance_passes_per_scheduled_round": total(
            outside, "joint_mc_pass_probability"
        )
        / n,
        "expected_new_graph_operator_pairs_per_scheduled_round": (
            total(candidates, "expected_new_graph_operator_pairs") / n
            if all(
                row["expected_new_graph_operator_pairs"] is not None
                for row in candidates
            )
            else None
        ),
        "expected_operator_updates_per_scheduled_round_on_frozen_nominations": (
            math.fsum(
                row["candidate_operator_updates"] * row["joint_mc_pass_probability"]
                for row in certifiable_updates
            )
            / n
            if len(certifiable_updates) == len(candidates)
            else None
        ),
        "counterfactual_routes_on_same_frozen_nominations": {
            "no_verify_out_of_tolerance_passes": len(outside),
            "census_nominations_passing": len(admissible),
            "census_out_of_tolerance_passes": 0,
        },
        "interior_by_normalized_drift_sd_stratum": strata,
    }


def calibrate(
    records,
    budgets,
    batches,
    seed,
    *,
    interior_drift=0.016,
    interior_flip=0.005,
    joint_target=0.95,
    drift_sd_edges=(0.005, 0.02, 0.05),
):
    """Estimate operating power on frozen inner evidence."""
    if any(row["phase"] != "inner" for row in records):
        raise ValueError("Calibration reads the inner phase only")
    if (
        not 0 <= interior_flip < 1
        or not 0 <= interior_drift < 1
        or not 0 < joint_target <= 1
    ):
        raise ValueError("Invalid operating margins or joint target")
    if list(drift_sd_edges) != sorted(set(drift_sd_edges)) or any(
        not 0 < v < 1 for v in drift_sd_edges
    ):
        raise ValueError("Drift dispersion boundaries must increase inside (0, 1)")
    if batches < 1 or seed < 0:
        raise ValueError("Expected positive Monte Carlo batches and a nonnegative seed")
    validate_settings(budgets, 0.05, 0.05, 0.05)
    estimates = list(
        estimate_candidates(
            records,
            budgets,
            batches,
            seed,
            interior_drift,
            interior_flip,
            drift_sd_edges,
        )
    )
    dims = (*GROUP_FIELDS, "draws")
    summaries = []
    by_budget = grouped(with_all_segments(estimates), dims)
    for key, scheduled in grouped(with_all_segments(records), GROUP_FIELDS).items():
        for draws in budgets:
            current_key = (*key, draws)
            candidates = by_budget.get(current_key, [])
            summaries.append(
                {
                    **dict(zip(dims, current_key, strict=True)),
                    **calibration_summary(
                        scheduled, candidates, len(drift_sd_edges) + 1, joint_target
                    ),
                }
            )
    return {
        "law": "IID with replacement on each registered uniform finite target",
        "scope": "fixed nominations from inner logs",
        "pair_cost_model": "local design and verification share the graph/operator prediction cache",
        "batches_per_candidate_and_budget": batches,
        "monte_carlo_seed": seed,
        "operating_target": {
            "interior_normalized_mean_drift": interior_drift,
            "interior_flip_rate": interior_flip,
            "joint_pass_target": joint_target,
            "normalized_drift_sd_bin_edges": list(drift_sd_edges),
        },
        "groups": summaries,
        "candidate_budgets": estimates,
    }


def write_json(destination, report):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(destination)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summarize", "screening", "calibrate", "analytic"):
        options = sub.add_parser(name)
        options.add_argument("--output", type=Path, required=True)
        if name != "analytic":
            options.add_argument("--runs", type=Path, required=True)
            options.add_argument("--arm", action="append", default=[])
            options.add_argument("--segment", action="append", default=[])
        if name in ("summarize", "screening"):
            options.add_argument("--phase", choices=("inner", "outer"))
        if name in ("calibrate", "analytic"):
            options.add_argument("--draws", nargs="+", type=int, default=DEFAULT_DRAWS)
    sub.choices["summarize"].add_argument("--reference", default="Unscreened")
    sub.choices["screening"].add_argument("--screened-arm", default="PCOC")
    sub.choices["screening"].add_argument("--unscreened-arm", default="Unscreened")
    sub.choices["analytic"].add_argument(
        "--rates", nargs="+", type=float, default=DEFAULT_RATES
    )
    sub.choices["analytic"].add_argument(
        "--alpha", type=float, default=0.05 / (30 * 2 * 2)
    )
    sub.choices["analytic"].add_argument("--chi-normalized", type=float, default=0.02)
    sub.choices["analytic"].add_argument("--gamma", type=float, default=0.01)
    sub.choices["calibrate"].add_argument("--batches", type=int, default=10_000)
    sub.choices["calibrate"].add_argument("--seed", type=int, default=271828)
    sub.choices["calibrate"].add_argument("--interior-drift", type=float, default=0.016)
    sub.choices["calibrate"].add_argument("--interior-flip", type=float, default=0.005)
    sub.choices["calibrate"].add_argument("--joint-target", type=float, default=0.95)
    sub.choices["calibrate"].add_argument(
        "--drift-sd-edges", nargs="*", type=float, default=(0.005, 0.02, 0.05)
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "analytic":
            report = analytic(
                args.draws, args.rates, args.alpha, args.chi_normalized, args.gamma
            )
        else:
            phase = "inner" if args.command == "calibrate" else args.phase
            records, sources, completion = load_runs(
                args.runs, phase=phase, arms=args.arm, segments=args.segment
            )
            if args.command == "summarize":
                report = summarize(records, args.reference)
            elif args.command == "screening":
                report = screening(records, args.screened_arm, args.unscreened_arm)
            else:
                report = calibrate(
                    records,
                    args.draws,
                    args.batches,
                    args.seed,
                    interior_drift=args.interior_drift,
                    interior_flip=args.interior_flip,
                    joint_target=args.joint_target,
                    drift_sd_edges=args.drift_sd_edges,
                )
            report["sources"] = sources
            report["completion"] = completion
        write_json(args.output, report)
    except (ArithmeticError, KeyError, OSError, ValueError) as error:
        parser.exit(2, f"pcoc.analysis: {error}\n")


if __name__ == "__main__":
    main()
