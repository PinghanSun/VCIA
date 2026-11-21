"""Validate the execution seal and score the graph study using outer labels."""

from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from csv import DictReader, DictWriter
from itertools import product
import json
from statistics import mean, median, stdev

import numpy as np

from run_checks import HERE, PLAN, base, old

RECORDS = Path("runs/replay")
PRIOR_RECORDS = Path("runs/closed-loop")
PAIR = ("dataset", "outer_seed", "alpha", "condition")
ARM = (*PAIR, "ranking", "audit")
SNAPSHOT = (*PAIR, "round")
METRICS = ("acc", "nmi", "ari", "macro_f1")


def json_from(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from DictReader(stream)


def parse(path, text_fields):
    def typed(key, value):
        if value == "":
            return None
        if key in text_fields:
            return value
        return float(value) if any(c in value for c in ".eE") else int(value)
    return [{key: typed(key, value) for key, value in r.items()} for r in read_rows(path)]


def group(rows, fields):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in fields)].append(row)
    return out


def named(fields, values):
    return dict(zip(fields, values, strict=True))


def read_index():
    index = json_from(RECORDS / "index.json")
    arms = group(index, ("cohort", "dataset", "outer_seed", "rep", "peers", "alpha", "condition", "method"))
    assert len(arms) == len(index) == 48 + 54 + 216
    return index


def write(name, rows):
    fields = list(rows[0])
    with (RECORDS / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality(prob, ids, truth):
    assert prob.shape[0] == len(ids) and np.isfinite(prob).all()
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-12, rtol=0)
    counts = np.bincount(prob.argmax(axis=1), minlength=prob.shape[1])
    return {**base.quality(prob, truth[ids]), "n_test": len(ids), "k": prob.shape[1],
            "occupied_clusters": int(np.count_nonzero(counts)),
            "smallest_occupied_share": float(min(counts[counts > 0]) / len(ids)),
            "cluster_sizes": "/".join(map(str, sorted(counts, reverse=True)))}


def med(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return median(values) if values else None


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def operations(rows):
    attempts = [r for r in rows if r["finite_feasible"] is not None]
    committed = [r for r in rows if r["status"] == "committed"]
    feasible = sum(r["finite_feasible"] for r in attempts)
    over = sum(r["finite_feasible"] == 0 for r in committed)
    refused = sum(r["finite_feasible"] == 1 and r["status"] == "rejected" for r in attempts)
    return {"planned_rounds": len(rows), "attempts": len(attempts), "commits": len(committed),
            "no_candidate": len(rows)-len(attempts), "rejects": len(attempts)-len(committed),
            "finite_feasible_attempts": feasible, "overlimit_commits": over,
            "overlimit_per_attempt": rate(over, len(attempts)), "feasible_rejections": refused,
            "feasible_rejection_rate": rate(refused, feasible), "operator_updates": sum(r["operator_updates"] for r in rows),
            "updates_per_planned_round": mean(r["operator_updates"] for r in rows),
            "median_pipeline_ms": med(rows, "pipeline_ms"),
            "median_proposal_ms": med(rows, "proposal_ms"), "median_design_prep_ms": med(rows, "design_prep_ms"),
            "median_design_inference_ms": med(rows, "design_inference_ms"), "median_search_ms": med(rows, "search_ms"),
            "median_audit_prep_ms": med(attempts, "probe_prep_ms"), "median_audit_inference_ms": med(attempts, "probe_inference_ms"),
            "median_audit_decision_ms": med(attempts, "decision_ms"), "median_audit_ms": med(attempts, "audit_total_ms"),
            "median_commit_ms": med(attempts, "commit_ms")}


def metric_diff(left, right, prefix=""):
    return {f"{prefix}{metric}_pp": 100*(left[metric]-right[metric]) for metric in METRICS}


def diagnostic_summaries(records):
    detail = []
    summarized = []
    strata = ("dataset", "outer_seed", "peers", "topology")
    for key, rows in group(records, strata).items():
        assert len(rows) == 2
        detail.append({**named(strata, key), "nested_repeats": len(rows),
                       **{item: mean(row[item] for row in rows) for item in rows[0] if item.endswith("_pp")}})
    collapse = ("dataset", "peers", "topology")
    for key, rows in group(detail, collapse).items():
        assert len(rows) == 3
        summary = {**named(collapse, key), "outer_splits": len(rows)}
        for method in ("stored", "central", "vcia"):
            for metric in (*METRICS, "occupied"):
                field = f"{method}_{metric}_pp"
                values = [r[field] for r in rows]
                summary[f"{method}_{metric}_mean"] = mean(values)
                summary[f"{method}_{metric}_split_sd"] = stdev(values)
        summarized.append(summary)
    return detail, summarized


def all_analyses():
    seal = json_from(RECORDS / "seal.json")
    assert not seal["smoke"]
    for name, digest in seal["sha256"].items():
        assert base.digest(RECORDS / name) == digest, name
    for name, digest in seal["prior_record_sha256"].items():
        assert base.digest(PRIOR_RECORDS / name) == digest, name
    prior_manifest = json_from(PRIOR_RECORDS / "score_manifest.json")
    for name in ("endpoints.csv", "recovery_pairs.csv", "paired_round_gaps.csv"):
        assert base.digest(PRIOR_RECORDS / name) == prior_manifest["sha256"][name], name
    assert base.digest(PRIOR_RECORDS / "seal.json") == prior_manifest["seal_sha256"]
    assert base.digest(PRIOR_RECORDS / "rounds.csv") == json_from(PRIOR_RECORDS / "seal.json")["sha256"]["rounds.csv"]
    sources = json_from(RECORDS / "execution.json")["source_sha"]
    for path, digest in sources.items():
        assert base.digest(HERE / path) == digest, path

    labels = {name: old.dataset(name)[1] for name in PLAN["datasets"]}
    splits = json_from(RECORDS / "splits.json")
    split_index = {(r["dataset"], r["outer_seed"], r["alpha"]): r for r in splits}
    assert len(split_index) == 27
    peers = []
    for split in splits:
        n_train = sum(map(len, split["peer_train"]))
        for user, (ids, profile) in enumerate(zip(split["peer_train"], split["peer_train_class_counts"], strict=True)):
            assert len(ids) == sum(profile) >= PLAN["min_train_graphs_per_peer"]
            peers.append({**{key: split[key] for key in ("dataset", "outer_seed", "alpha", "partition_seed",
                          "allocation_attempts", "weighted_label_tv")}, "peer": user + 1, "graphs": len(ids),
                          "train_fraction": len(ids)/n_train, "dominant_class_share": max(profile)/len(ids),
                          "class_counts_label_equals_position": "/".join(map(str, profile))})
    assert len(peers) == 216

    raw = parse(RECORDS / "online_rounds.csv", {"dataset", "condition", "ranking", "audit", "status", "committed_sha"})
    observed = group(raw, ARM)
    assert len(observed) == 216 and len(raw) == 12960
    for key, trajectory in observed.items():
        assert [r["round"] for r in trajectory] == list(range(1, 61))
        for row in trajectory:
            assert 0 <= row["operator_updates"] <= PLAN["quota"]
            assert (row["status"] == "no_candidate") == (row["mask"] is None)
            assert (row["mask"] is None) == (row["finite_feasible"] is None)
            assert (row["mask"] is None) == (row["design_feasible_masks"] == 0)
            assert row["operator_updates"] == 0 or row["status"] == "committed"
            if row["mask"] is not None:
                assert row["mask"].bit_count() == PLAN["quota"]
                assert np.isclose(row["audit_total_ms"], sum(row[p] for p in ("probe_prep_ms", "probe_inference_ms", "decision_ms")))
                if row["audit"] == "Census":
                    assert (row["status"] == "committed") == bool(row["finite_feasible"])

    reference = parse(PRIOR_RECORDS / "endpoints.csv", {"dataset", "condition", "topology", "arm", "final_state"})
    old_rounds = [
        r for r in read_rows(PRIOR_RECORDS / "rounds.csv")
        if r["peers"] == "8" and r["topology"] == "Ring" and r["rep"] == "1" and r["arm"] == "VCIA"
        and r["condition"] in PLAN["conditions"]
    ]
    for historic in old_rounds:
        new = observed[historic["dataset"], int(historic["outer_seed"]), .5, historic["condition"], "VCIA", "IID"]
        assert new[int(historic["round"])-1]["committed_sha"] == historic["committed_sha"]
    assert len(old_rounds) == 720

    historic_arm_fields = ("dataset", "outer_seed", "rep", "peers")
    new_endpoints, historical_arms, diagnostics, replay = [], [], [], []
    historical = {}
    bank = np.load(RECORDS / "sealed_predictions.npz", allow_pickle=False)
    try:
        assert len(bank.files) == seal["probability_arrays"]
        for item in read_index():
            core = {field: item[field] for field in (*PAIR, "rep", "peers", "method", "cohort", "token")}
            ids = np.asarray(item["test_ids"])
            result = {**core, **quality(bank[item["token"]], ids, labels[item["dataset"]])}
            if item["cohort"] == "historical":
                historical[tuple(item[k] for k in historic_arm_fields) + (item["method"],)] = result
                historical_arms.append(result)
            else:
                assert item["test_ids"] == split_index[tuple(item[k] for k in PAIR[:3])]["test"]
                rank, sep, audit = item["method"].partition(" / ")
                result.update(operations(observed[tuple(item[k] for k in PAIR) + (rank, audit)]) if sep else
                              dict.fromkeys(operations(observed[next(iter(observed))])))
                new_endpoints.append(result)
        for row in reference:
            if row["condition"] != "Clean" or row["arm"] != "VCIA":
                continue
            key = tuple(row[k] for k in historic_arm_fields)
            stored = historical[key + ("Stored-only",)]
            central = historical[key + ("WL + K-means",)]
            assert row["n_test"] == stored["n_test"] == central["n_test"]
            diagnostics.append({**named(historic_arm_fields, key), "topology": row["topology"], "n_test": row["n_test"],
                **{f"{m}_{key}_pp": 100*value[key] for m, value in (("stored", stored), ("central", central), ("vcia", row)) for key in METRICS},
                **{f"{m}_occupied_pp": r["occupied_clusters"] for m, r in (("stored", stored), ("central", central), ("vcia", row))},
                "stored_cluster_sizes": stored["cluster_sizes"], "central_cluster_sizes": central["cluster_sizes"],
                **metric_diff(row, stored, "vcia_minus_stored_"), **metric_diff(central, stored, "central_minus_stored_")})
        assert len(diagnostics) == 48

        for row in parse(RECORDS / "replay.csv", {"dataset", "condition", "ranking", "audit", "candidate_family", "prior_token", "candidate_token"}):
            ids = np.asarray(split_index[tuple(row[k] for k in PAIR[:3])]["test"])
            prior = quality(bank[row["prior_token"]], ids, labels[row["dataset"]])
            candidate = quality(bank[row["candidate_token"]], ids, labels[row["dataset"]]) if row["candidate_token"] else None
            commit = candidate if row["passed"] else prior
            replay.append({**row, **metric_diff(commit, prior, "committed_delta_"),
                           "pre_update_nmi": prior["nmi"], "pre_update_ari": prior["ari"],
                           "candidate_nmi": candidate["nmi"] if candidate else None, "candidate_ari": candidate["ari"] if candidate else None})
    finally:
        bank.close()

    paired_live = []
    endpoints = {(tuple(r[k] for k in PAIR), r["method"]): r for r in new_endpoints}
    controls = [("ranking / IID", "VCIA / IID", "J-Rank / IID"), ("ranking / Census", "VCIA / Census", "J-Rank / Census"),
                ("audit / VCIA", "VCIA / IID", "VCIA / Census"), ("audit / J-Rank", "J-Rank / IID", "J-Rank / Census"),
                ("update / Stored-only", "VCIA / IID", "Stored-only"), ("update / WL + K-means", "VCIA / IID", "WL + K-means")]
    for identity in group([r for r in new_endpoints if r["condition"] != "Both"], PAIR):
        for name, a, b in controls:
            left, right = endpoints[identity, a], endpoints[identity if " / " in b else (*identity[:-1], "Both"), b]
            paired_live.append({**named(PAIR, identity), "contrast": name, "left": a, "right": b,
                **metric_diff(left, right),
                **{f"delta_{k}": left[k]-right[k] if right[k] is not None else None for k in
                   ("operator_updates", "commits", "overlimit_commits", "median_pipeline_ms")}})

    assert len(replay) == 1296 and len(paired_live) == 324
    for key, rows in group(replay, SNAPSHOT).items():
        assert len(rows) == 4 and len({r["candidate_family"] for r in rows}) == 1
        assert len({r["prior_token"] for r in rows}) == 1
        reference_round = observed[key[:-1] + ("VCIA", "IID")][key[-1] - 1]
        for rank, pair in group(rows, ("ranking",)).items():
            iid, census = sorted(pair, key=lambda r: r["audit"], reverse=True)
            assert iid["audit"] == "IID" and census["audit"] == "Census"
            assert iid["mask"] == census["mask"] and iid["candidate_token"] == census["candidate_token"]
            assert iid["finite_feasible"] == census["finite_feasible"] == census["passed"]
            assert (iid["mask"] is None) == (iid["candidate_family"] is None)
            assert (iid["mask"] is None or str(iid["mask"]) in iid["candidate_family"].split(";"))
            if rank == ("VCIA",):
                assert iid["mask"] == reference_round["mask"]
                assert iid["committed_operator_updates"] == reference_round["operator_updates"]
                if iid["mask"] is not None:
                    assert bool(iid["passed"]) == (reference_round["status"] == "committed")
    split_diagnostic, overall_diagnostic = diagnostic_summaries(diagnostics)
    return dict(client_partitions=peers, historical_diagnostics=diagnostics, historical_controls=historical_arms,
                diagnostics_by_split=split_diagnostic, diagnostics_summary=overall_diagnostic,
                endpoints=new_endpoints, paired_live=paired_live, scored_replay=replay)


def recovery():
    pairs = parse(PRIOR_RECORDS / "recovery_pairs.csv", {"dataset", "topology", "condition", "arm", "status"})
    gaps = parse(PRIOR_RECORDS / "paired_round_gaps.csv", {"dataset", "topology", "condition", "arm"})
    fields = ("dataset", "outer_seed", "rep", "peers", "topology", "condition", "arm")
    gap_index = group(gaps, fields)
    assert len(pairs) == len(gap_index) == 384 and len(gaps) == 23040
    for pair in pairs:
        points = gap_index[tuple(pair[k] for k in fields)]
        assert [r["round"] for r in points] == list(range(1, 61))
        peak = max(r["tv"] for r in points[10:35])
        trigger = peak > .05
        confirmation = next((r for r in range(38, 61) if all(p["tv"] <= .05 for p in points[r-3:r])), None) if trigger else None
        assert peak == pair["peak_tv"] and confirmation == pair["confirm_round"]
        assert pair["status"] == ("not_disrupted" if not trigger else "recovered" if confirmation else "right_censored")
        assert pair["relapse"] == (int(any(p["tv"] > .05 for p in points[confirmation:])) if confirmation else None)
        pair["triggered"] = int(trigger)
        pair["never_triggered"] = int(not trigger)
        pair["recovered"] = int(confirmation is not None)
        pair["right_censored"] = int(trigger and confirmation is None)
    counts = []
    for grouping in (("condition", "arm"), ("dataset", "condition", "arm"), ("dataset", "outer_seed", "condition", "arm"),
                     ("dataset", "peers", "topology", "condition", "arm")):
        for key, rows in group(pairs, grouping).items():
            n = len(rows)
            sums = {f: sum(r[f] for r in rows) for f in ("never_triggered", "triggered", "recovered", "right_censored")}
            assert n == sums["never_triggered"] + sums["triggered"]
            assert sums["triggered"] == sums["recovered"] + sums["right_censored"]
            counts.append({**dict.fromkeys(("dataset", "outer_seed", "peers", "topology")), **named(grouping, key),
                           "pairs": n, **sums, "trigger_fraction": sums["triggered"]/n,
                           "recovery_given_trigger": rate(sums["recovered"], sums["triggered"]),
                           "relapses": sum(r["relapse"] or 0 for r in rows), "median_peak_tv": med(rows, "peak_tv"),
                           "median_cumulative_tv": med(rows, "cumulative_tv"), "median_final_tv": med(rows, "final_tv")})
    contrasted = []
    for identity, rows in group(pairs, fields[:-1]).items():
        by_arm = {r["arm"]: r for r in rows}
        a = by_arm["VCIA"]
        for arm in ("Unscreened", "J-Rank-Feasible", "No-Verify"):
            b = by_arm[arm]
            contrasted.append({**named(fields[:-1], identity), "comparator": arm,
                "vcia_peak_tv": a["peak_tv"], "reference_peak_tv": b["peak_tv"], "delta_peak_tv": a["peak_tv"]-b["peak_tv"],
                "delta_cumulative_tv": a["cumulative_tv"]-b["cumulative_tv"], "delta_final_tv": a["final_tv"]-b["final_tv"],
                "vcia_triggered": a["triggered"], "reference_triggered": b["triggered"]})
    peak_summary = []
    for grouping in (("condition", "comparator"), ("dataset", "outer_seed", "condition", "comparator")):
        for identity, rows in group(contrasted, grouping).items():
            delta = [r["delta_peak_tv"] for r in rows]
            peak_summary.append({**dict.fromkeys(("dataset", "outer_seed")), **named(grouping, identity),
                                 "pairs": len(rows), "vcia_lower": sum(v < -1e-12 for v in delta),
                                 "equal": sum(abs(v) <= 1e-12 for v in delta), "vcia_higher": sum(v > 1e-12 for v in delta),
                                 "median_vcia_peak_tv": med(rows, "vcia_peak_tv"), "median_reference_peak_tv": med(rows, "reference_peak_tv"),
                                 "median_delta_peak_tv": median(delta), "median_delta_cumulative_tv": med(rows, "delta_cumulative_tv"),
                                 "median_delta_final_tv": med(rows, "delta_final_tv")})
    return dict(recovery_counts=counts, recovery_detail=pairs, recovery_peak_comparisons=contrasted, recovery_peak_summary=peak_summary)


def main():
    tables = {**all_analyses(), **recovery()}
    for name, rows in tables.items():
        write(name, rows)
    base.write_json(RECORDS / "score_manifest.json", {"scored_at_utc": base.now(), "execution_seal": base.digest(RECORDS / "seal.json"),
        "scorer_sha256": base.digest(HERE / "score_checks.py"),
        "rows": {name: len(rows) for name, rows in tables.items()},
        "sha256": {name + ".csv": base.digest(RECORDS / f"{name}.csv") for name in tables}})


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("--data", type=Path, default=old.DATA)
    cli.add_argument("--runs", type=Path, default=RECORDS)
    cli.add_argument("--closed-loop", type=Path, default=PRIOR_RECORDS)
    args = cli.parse_args()
    old.DATA, RECORDS, PRIOR_RECORDS = args.data, args.runs, args.closed_loop
    main()
