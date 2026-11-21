"""Verify sealed predictions, score outer tests, and reconcile all counts."""

from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from csv import DictReader
from itertools import product
from math import comb
import json

import numpy as np

import run_evidence
from run_evidence import HERE, OUT, PROTOCOL, _wl_features, base, dataset

KEY = ("dataset", "outer_seed", "rep", "peers", "q", "topology", "condition", "arm")
GROUP = ("dataset", "peers", "topology", "condition")


def read(name):
    with (OUT / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(DictReader(stream))


def key(row, fields=KEY):
    return tuple(str(row[name]) for name in fields)


def grouped(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[key(row, fields)].append(row)
    return result


def sum_of(rows, field):
    return sum(int(row[field]) for row in rows)


def quantile(values, fraction=.5):
    return float(np.quantile(values, fraction)) if len(values) else None


def mean(values):
    return float(np.mean(values)) if len(values) else None


def write(name, rows):
    target = base.Rows(OUT / f"{name}.csv")
    try:
        for row in rows:
            target.add(row)
    finally:
        target.close()


def identity(label, fields):
    return dict(zip(fields, label, strict=True))


def screening(rows, timing):
    repeats = grouped(timing, (*KEY, "round"))
    assert set(repeats) == {key(row, (*KEY, "round")) for row in rows}
    per_snapshot = []
    for row in rows:
        log = repeats[key(row, (*KEY, "round"))]
        assert len(log) == 2 * PROTOCOL["timing_repetitions_per_arm"]
        for pair in grouped(log, ("repetition",)).values():
            assert {item["order"] for item in pair} == {"1", "2"}
            assert {item["policy"] for item in pair} == {"VCIA", "Unscreened"}
        paired = defaultdict(list)
        for item in log:
            paired[item["policy"]].append(item)
        metrics = {}
        for policy, prefix in (("VCIA", "vcia"), ("Unscreened", "unscreened")):
            for phase in ("setup_s", "peer_screen_or_detect_s", "composition_s", "total_s"):
                metrics[f"{prefix}_{phase[:-2]}_ms"] = quantile(
                    [1000 * float(v[phase]) for v in paired[policy]])
        per_snapshot.append({**row, **metrics,
                             "paired_saved_ms": metrics["unscreened_total_ms"] - metrics["vcia_total_ms"],
                             "paired_speedup": metrics["unscreened_total_ms"] / metrics["vcia_total_ms"]})
    summary = []
    scopes = [(identity(k, GROUP), v) for k, v in grouped(per_snapshot, GROUP).items()]
    scopes += [({"dataset": "Both", "peers": p, "topology": t, "condition": c}, v)
               for (p, t, c), v in grouped(per_snapshot, GROUP[1:]).items()]
    scopes += [({"dataset": "Both", "peers": p, "topology": "Both", "condition": c}, v)
               for (p, c), v in grouped(per_snapshot, ("peers", "condition")).items()]
    for label, group in scopes:
        for window, subset in (("All six checkpoints", group),
                               ("Rounds 11–35", [r for r in group if 11 <= int(r["round"]) <= 35])):
            full = sum_of(subset, "full_evaluations")
            kept = sum_of(subset, "screened_evaluations")
            summary.append({**label, "window": window, "snapshots": len(subset), "full_evaluations": full,
                            "screened_evaluations": kept, "saved_evaluations": full - kept,
                            "saved_pp": 100 * (full - kept) / full,
                            **{f: sum_of(subset, f) for f in ("design_feasible", "design_feasible_omitted",
                               "exact_feasible", "exact_feasible_omitted", "both_feasible",
                               "both_feasible_omitted", "same_mask")},
                            "both_no_candidate": sum(not r["vcia_mask"] and not r["unscreened_mask"] for r in subset),
                            "selected_in_both": sum(bool(r["vcia_mask"] and r["unscreened_mask"]) for r in subset),
                            "vcia_median_ms": quantile([r["vcia_total_ms"] for r in subset]),
                            "unscreened_median_ms": quantile([r["unscreened_total_ms"] for r in subset]),
                            "paired_saved_median_ms": quantile([r["paired_saved_ms"] for r in subset]),
                            "paired_speedup_median": quantile([r["paired_speedup"] for r in subset]),
                            "paired_speedup_p10": quantile([r["paired_speedup"] for r in subset], .1),
                            "paired_speedup_p90": quantile([r["paired_speedup"] for r in subset], .9)})
    return per_snapshot, summary


def fixed_audit(rows, snapshots, rounds):
    by_attempt = grouped(rows, (*KEY, "round"))
    selected = {key(r, (*KEY, "round")): r["vcia_mask"] for r in snapshots if r["vcia_mask"]}
    assert set(by_attempt) == set(selected)
    online = {key(r, (*KEY, "round")): r for r in rounds if key(r, (*KEY, "round")) in selected}
    assert set(online) == set(selected)
    for name, records in by_attempt.items():
        assert len(records) == 5
        assert len({(r["mask"], r["finite_feasible"], r["ell"], r["phi"]) for r in records}) == 1
        assert records[0]["mask"] == selected[name]
        census = next(r for r in records if r["policy"] == "Finite census")
        assert census["passed"] == census["finite_feasible"]
        actual = online[name]
        default = next(r for r in records if r["policy"] == f"IID {PROTOCOL['verification_draws']}")
        assert actual["mask"] == default["mask"] and actual["finite_feasible"] == default["finite_feasible"]
        assert (actual["status"] == "committed") == (default["passed"] == "1")
    result = []
    scopes = [(identity(k, (*GROUP, "policy")), v) for k, v in grouped(rows, (*GROUP, "policy")).items()]
    scopes += [({"dataset": "Both", "peers": "Both", "topology": "Both", "condition": "All", "policy": k[0]}, v)
               for k, v in grouped(rows, ("policy",)).items()]
    for label, records in scopes:
        feasible = sum_of(records, "finite_feasible")
        rejected = sum(r["passed"] == "0" and r["finite_feasible"] == "1" for r in records)
        violations = sum(r["passed"] == "1" and r["finite_feasible"] == "0" for r in records)
        result.append({**label, "candidates": len(records), "finite_feasible": feasible,
                       "finite_infeasible": len(records) - feasible, "passed": sum_of(records, "passed"),
                       "overlimit_passes": violations, "overlimit_rate_pct": 100 * violations / len(records),
                       "feasible_rejected": rejected, "feasible_rejected_pct": 100 * rejected / feasible if feasible else None,
                       "unique_graphs_median": quantile([int(r["unique_graphs"]) for r in records]),
                       "decision_median_ms": quantile([1000 * float(r["decision_s"]) for r in records if r["decision_s"]])})
    return result


def score():
    sealed = json.loads((OUT / "seal.json").read_text(encoding="utf-8"))
    assert all(base.digest(OUT / name) == digest for name, digest in sealed["sha256"].items())
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    expected_keys = {tuple(map(str, (ds, seed, rep, n, q, top, condition, arm)))
                     for ds, seed, rep, (n, q), top, condition, arm in product(
                         PROTOCOL["datasets"], PROTOCOL["outer_seeds"],
                         range(1, PROTOCOL["paired_repetitions_per_split"] + 1),
                         PROTOCOL["peer_quota_pairs"], PROTOCOL["topologies"], PROTOCOL["conditions"], PROTOCOL["arms"])}
    assert len(index) == len(expected_keys) and {key(row) for row in index} == expected_keys
    round_rows, snap = read("rounds"), read("snapshots")
    round_index = grouped(round_rows, KEY)
    labels, representations = {}, {}
    for name in PROTOCOL["datasets"]:
        graphs, labels[name], *_ = dataset(name)
        representations[name] = _wl_features(graphs, 2, 512)
    configs = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    config_index = {key(r, ("dataset", "outer_seed", "rep", "peers")): r for r in configs}
    assert len(configs) == len(config_index) == len({r[:4] for r in expected_keys})
    widths = {}
    for config_key, config in config_index.items():
        wl, fingerprints = representations[config["dataset"]]
        role_sets = [{fingerprints[i] for i in config[role]} for role in ("train", "audit", "test")]
        assert not (role_sets[0] & role_sets[1] or role_sets[1] & role_sets[2] or role_sets[0] & role_sets[2])
        widths[config_key] = int(np.count_nonzero(np.any(wl[config["train"]], axis=0)))
    assert set(round_index) == expected_keys
    assert {(key(r), int(r["round"])) for r in snap} == {(k, t) for k in expected_keys if k[-1] == "VCIA"
                                                          for t in PROTOCOL["replay_rounds"]}
    assert len(snap) == len(expected_keys) // len(PROTOCOL["arms"]) * len(PROTOCOL["replay_rounds"])
    for row in snap:
        assert int(row["full_evaluations"]) == comb(int(row["peers"]), int(row["q"]))
        assert int(row["full_evaluations"]) >= int(row["screened_evaluations"])
        assert int(row["same_mask"]) == int(row["vcia_mask"] == row["unscreened_mask"])
        for v in ("design_feasible", "exact_feasible", "both_feasible"):
            assert 0 <= int(row[v + "_omitted"]) <= int(row[v])
    compact_index = {key(r): r for r in index}
    recovery, gap_rows, endpoints = [], [], []
    t = PROTOCOL["rounds"]
    with np.load(OUT / "sealed_outer_probabilities.npz", allow_pickle=False) as bank:
        assert set(bank.files) == {r["token"] for r in index}
        for row in index:
            name = key(row)
            ids = row["test_ids"]
            config_key = key(row, ("dataset", "outer_seed", "rep", "peers"))
            config = config_index[config_key]
            assert ids == config["test"]
            train, audit, test = (set(config[role]) for role in ("train", "audit", "test"))
            assert not (train & audit or test & train or test & audit)
            assert len(train | audit | test) == len(labels[row["dataset"]])
            client_ids = [i for peer in config["peer_train"] for i in peer]
            assert len(client_ids) == len(train) and set(client_ids) == train
            prob = bank[row["token"]]
            assert prob.shape == (t, len(ids), 6 if row["dataset"] == "ENZYMES" else 2)
            assert (prob >= 0).all() and np.allclose(prob.sum(axis=-1), 1)
            run = round_index[name]
            assert [int(r["round"]) for r in run] == list(range(1, t + 1))
            for prev, r in zip([None, *run[:-1]], run, strict=True):
                assert r["status"] in ("committed", "rejected", "no_candidate")
                assert bool(r["mask"]) == bool(r["finite_feasible"]) == (r["status"] != "no_candidate")
                if r["mask"]:
                    assert int(r["mask"]).bit_count() == row["q"]
                if r["status"] == "committed":
                    assert r["candidate_sha"] == r["committed_sha"]
                elif prev:
                    assert r["committed_sha"] == prev["committed_sha"]
                assert 0 <= int(r["operator_updates"]) <= row["q"]
                assert int(r["uplink_app_bytes"]) == row["peers"] * (8 * (prob.shape[-1] * widths[config_key] + 1) + 2)
                assert int(r["downlink_app_bytes"]) == row["peers"] * 37
                assert int(r["composition_evaluations"]) <= comb(row["peers"], row["q"])
                assert int(r["perturbed_proposals"]) == (
                    (2 if row["condition"] == "Moderate" else row["peers"] // 2)
                    if row["condition"] != "Clean" and 11 <= int(r["round"]) <= 35 else 0)
            n_updates = sum_of(run, "operator_updates")
            tag = identity(name, KEY)
            endpoint = {**tag, "n_test": len(ids), **base.quality(prob[-1], labels[row["dataset"]][ids]),
                        "final_state": run[-1]["committed_sha"], "occupied_clusters": len(np.unique(prob[-1].argmax(axis=1))),
                        "attempts": sum(r["status"] != "no_candidate" for r in run),
                        "commits": sum(r["status"] == "committed" for r in run),
                        "verification_rejects": sum(r["status"] == "rejected" for r in run),
                        "no_candidate": sum(r["status"] == "no_candidate" for r in run),
                        "overlimit_commits": sum(r["status"] == "committed" and r["finite_feasible"] == "0" for r in run),
                        "finite_feasible_attempts": sum(r["finite_feasible"] == "1" for r in run),
                        "feasible_rejected": sum(r["status"] == "rejected" and r["finite_feasible"] == "1" for r in run),
                        "operator_updates": n_updates, "composition_evaluations": sum_of(run, "composition_evaluations"),
                        **{f"median_{k[:-2]}_ms": quantile([1000 * float(r[k]) for r in run]) for k in
                           ("proposal_s", "probe_prep_s", "search_s", "verify_s", "stage_and_commit_s", "loopback_s", "online_s")},
                        "online_total_s": sum(float(r["online_s"]) for r in run),
                        "uplink_app_bytes": sum_of(run, "uplink_app_bytes"),
                        "downlink_app_bytes": sum_of(run, "downlink_app_bytes")}
            endpoints.append(endpoint)
            if row["condition"] == "Clean":
                continue
            control = compact_index[(*name[:-2], "Clean", name[-1])]
            assert control["test_ids"] == ids
            clean = bank[control["token"]]
            np.testing.assert_array_equal(prob[:10], clean[:10])
            gap = .5 * np.abs(prob - clean).sum(axis=-1).mean(axis=1)
            assignment = (prob.argmax(axis=-1) != clean.argmax(axis=-1)).mean(axis=1)
            peak = float(gap[10:35].max())
            reached = next((r for r in range(38, t + 1) if (gap[r - 3:r] <= .05).all()), None) if peak > .05 else None
            status = "not_disrupted" if peak <= .05 else ("recovered" if reached else "right_censored")
            recovery.append({**tag, "status": status, "confirm_round": reached,
                             "relapse": int((gap[reached:] > .05).any()) if reached else None,
                             "peak_tv": peak, "mean_during_tv": float(gap[10:35].mean()),
                             "cumulative_tv": float(gap[10:].sum()), "final_tv": float(gap[-1]),
                             "peak_assignment_change_pct": float(100 * assignment[10:35].max()),
                             "final_assignment_change_pct": float(100 * assignment[-1])})
            gap_rows.extend({**tag, "round": r, "tv": float(v), "assignment_change": float(a)}
                            for r, (v, a) in enumerate(zip(gap, assignment, strict=True), 1))
    return endpoints, recovery, gap_rows


def paired_quality(rows):
    records = {key(r): r for r in rows}
    pairs = []
    for name, vcia in records.items():
        if name[-1] != "VCIA":
            continue
        for arm in PROTOCOL["arms"][1:]:
            other = records[(*name[:-1], arm)]
            pairs.append({**identity(name[:-1], KEY[:-1]), "reference": arm, "n_test": vcia["n_test"],
                          "same_final_state": int(vcia["final_state"] == other["final_state"]),
                          **{f"vcia_{k}": vcia[k] for k in ("nmi", "ari")},
                          **{f"delta_{k}_pp": 100 * (vcia[k] - other[k]) for k in ("nmi", "ari")}})
    summary = []
    for label, rows_here in grouped(pairs, (*GROUP, "reference")).items():
        splits = grouped(rows_here, ("outer_seed",))
        assert len(splits) == len(PROTOCOL["outer_seeds"])
        result = {**identity(label, (*GROUP, "reference")), "n_pairs": len(rows_here), "n_outer_splits": len(splits),
                  "same_final_state": sum_of(rows_here, "same_final_state")}
        for metric in ("nmi", "ari"):
            by_split = [mean([r[f"delta_{metric}_pp"] for r in repeats]) for repeats in splits.values()]
            result.update({f"vcia_{metric}_pp": 100 * mean([r[f"vcia_{metric}"] for r in rows_here]),
                           f"delta_{metric}_mean_pp": mean(by_split),
                           f"delta_{metric}_min_outer_pp": min(by_split), f"delta_{metric}_max_outer_pp": max(by_split)})
        summary.append(result)
    return pairs, summary


def recovery_summary(rows):
    scopes = [(identity(k, (*GROUP, "arm")), v) for k, v in grouped(rows, (*GROUP, "arm")).items()]
    scopes += [({"dataset": "Both", "peers": "Both", "topology": "Both", **identity(k, ("condition", "arm"))}, v)
               for k, v in grouped(rows, ("condition", "arm")).items()]
    result = []
    for label, group in scopes:
        cases = [r for r in group if r["status"] == "recovered"]
        censored = sum(r["status"] == "right_censored" for r in group)
        result.append({**label, "pairs": len(group), "not_disrupted": len(group) - len(cases) - censored,
                       "triggered": len(cases) + censored, "recovered": len(cases), "right_censored": censored,
                       "relapses": sum_of(cases, "relapse"),
                       "median_confirmation": quantile([r["confirm_round"] for r in cases]),
                       **{f"median_{k}": quantile([r[k] for r in group]) for k in
                          ("peak_tv", "mean_during_tv", "cumulative_tv", "final_tv",
                           "peak_assignment_change_pct", "final_assignment_change_pct")}})
    return result


def runtime_summary(endpoints, rounds):
    summaries = []
    grouped_rounds = grouped(rounds, (*GROUP, "arm"))
    for key_here, traces in grouped(endpoints, (*GROUP, "arm")).items():
        data = grouped_rounds[key_here]
        n = len(data)
        feasible = sum_of(traces, "finite_feasible_attempts")
        rejected = sum_of(traces, "feasible_rejected")
        summary = {**identity(key_here, (*GROUP, "arm")), "n_trajectories": len(traces), "planned_rounds": n,
                   **{f: sum_of(traces, f) for f in ("attempts", "commits", "verification_rejects",
                      "no_candidate", "overlimit_commits", "operator_updates", "composition_evaluations")},
                   "finite_feasible_attempts": feasible, "feasible_rejected": rejected,
                   "feasible_rejected_pct": 100 * rejected / feasible if feasible else None,
                   "operator_updates_per_planned_round": sum_of(traces, "operator_updates") / n,
                   **{f"median_{k[:-2]}_ms": quantile([1000 * float(r[k]) for r in data]) for k in
                      ("proposal_s", "probe_prep_s", "search_s", "verify_s", "stage_and_commit_s", "loopback_s", "online_s")},
                   "p90_online_ms": quantile([1000 * float(r["online_s"]) for r in data], .9),
                   "median_up_bytes_per_round": quantile([int(r["uplink_app_bytes"]) for r in data]),
                   "median_down_bytes_per_round": quantile([int(r["downlink_app_bytes"]) for r in data])}
        assert summary["commits"] + summary["verification_rejects"] + summary["no_candidate"] == n
        summaries.append(summary)
    return summaries


def main():
    endpoints, recovery, gaps = score()
    snapshot, screen_summary = screening(read("snapshots"), read("replay_timing"))
    rounds = read("rounds")
    audit_summary = fixed_audit(read("fixed_audit"), snapshot, rounds)
    pairs, pair_summary = paired_quality(endpoints)
    route_summary = runtime_summary(endpoints, rounds)
    recovery_groups = recovery_summary(recovery)
    generated = {"endpoints": endpoints, "recovery_pairs": recovery, "paired_round_gaps": gaps,
                 "screening_paired_timing": snapshot, "screening_summary": screen_summary,
                 "fixed_audit_summary": audit_summary, "quality_pairs": pairs, "quality_summary": pair_summary,
                 "recovery_summary": recovery_groups, "runtime_summary": route_summary}
    for name, rows in generated.items():
        write(name, rows)
    base.write_json(OUT / "score_manifest.json", {"scored_at_utc": base.now(),
                    "seal_sha256": base.digest(OUT / "seal.json"),
                    "scoring_code_sha256": base.digest(HERE / "score_evidence.py"),
                    "counts": {k: len(v) for k, v in generated.items()},
                    "sha256": {f"{k}.csv": base.digest(OUT / f"{k}.csv") for k in generated}})
    print(json.dumps({"runs": len(endpoints), "rounds": len(rounds), "snapshots": len(snapshot),
                      "paired_quality_rows": len(pairs), "recovery_pairs": len(recovery)}))


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("--data", type=Path, default=run_evidence.DATA)
    cli.add_argument("--runs", type=Path, default=OUT)
    args = cli.parse_args()
    run_evidence.DATA, OUT = args.data, args.runs
    main()
