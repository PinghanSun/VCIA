"""Score paired timing or sealed outer-split predictions."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from csv import DictWriter
from itertools import product
from pathlib import Path
from statistics import mean, median

import numpy as np

import run_core_gap
from run_core_gap import (ARMS, EPS, EVIDENCE, FIELDS, HERE, METRICS, NEW_SEEDS, OLD_SEEDS, base,
                          load, make_plan, original, rows)


def write_csv(name, data):
    with (HERE / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = DictWriter(file, data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def assess_timing():
    path = HERE / "timing_rounds.csv"
    assert base.digest(path) == load(HERE / "timing_seal.json")["record_sha256"]
    fields = ("dataset", "seed", "condition", "repeat", "round")
    pair_index = defaultdict(dict)
    detail = []
    for raw in rows(path):
        row = {k: int(v) if k in ("seed", "repeat", "round", "order", "composition_evaluations", "unique_graph_operator_misses", "design_draws", "verification_draws")
               else (float(v) if v else None) if k.endswith("_ms") else v
               for k, v in raw.items()}
        assert row["end_to_end_ms"] >= sum(row[k] or 0. for k in ("train_ms", "design_ms", "verify_commit_ms"))
        key = tuple(row[k] for k in fields)
        assert row["arm"] not in pair_index[key]
        pair_index[key][row["arm"]] = row
        detail.append(row)
    assert pair_index and len(detail) == 2 * len(pair_index)
    grouped = defaultdict(list)
    for key, arms in pair_index.items():
        vcia, ref = (arms[k] for k in ("VCIA", "Unscreened"))
        for field in ("unique_graph_operator_misses", "executed_state_sha256", "design_draws", "verification_draws", "status"):
            assert ref[field] == vcia[field]
        assert vcia["order"] + ref["order"] == 3
        switched_key = (*key[:3], 3 - key[3], key[4])
        assert pair_index[switched_key]["VCIA"]["order"] == ref["order"]
        grouped[key[0], key[2]].append((vcia, ref))
    results = []
    for (dataset, condition), pairs in sorted(grouped.items()):
        left = [v["end_to_end_ms"] for v, _ in pairs]
        right = [r["end_to_end_ms"] for _, r in pairs]
        compositions = [sum(pair[index]["composition_evaluations"] for pair in pairs) for index in (0, 1)]
        results.append(dict(dataset=dataset, condition=condition, trajectory_pairs=len({v["seed"] for v, _ in pairs}),
                            timing_repetitions=2, paired_rounds=len(pairs), vcia_median_ms=median(left), ref_median_ms=median(right),
                            median_paired_diff_ms=median(a - b for a, b in zip(left, right, strict=True)),
                            vcia_total_ms=sum(left), ref_total_ms=sum(right), saved_pct=100*(1-sum(left)/sum(right)),
                            vcia_faster_rounds=sum(a < b for a, b in zip(left, right, strict=True)),
                            vcia_composition_evals=compositions[0], ref_composition_evals=compositions[1],
                            composition_saved_pct=100*(1-compositions[0]/compositions[1]),
                            matching_state_and_cache_rounds=len(pairs)))
    write_csv("timing_summary", results)
    return results


def scored_new(plan):
    inputs = []
    cohort = {tuple(cell[k] for k in FIELDS) for cell in plan["eligible"]}
    for seed in NEW_SEEDS:
        path = HERE / f"split_{seed}"
        seal = load(path / "seal.json")
        assert set(seal["sha256"]) == {"index.json", "partitions.json", "rounds.csv", "sealed_predictions.npz"}
        assert all(base.digest(path / f) == h for f, h in seal["sha256"].items())
        index = load(path / "index.json")
        key = (*FIELDS, "rep", "arm")
        expected = {(*cell, rep, arm) for cell in cohort for rep, arm in product((1, 2), ARMS)}
        assert len(index) == len(expected) and {tuple(row[k] for k in key) for row in index} == expected
        partitions = {(r["dataset"], r["peers"], r["rep"]): r for r in load(path / "partitions.json")}
        archived, final = defaultdict(list), {}
        for row in rows(path / "rounds.csv"):
            for k in ("peers", "q", "rep", "outer_seed", "round"):
                row[k] = int(row[k])
            k = tuple(row[k] for k in key)
            assert row["outer_seed"] == seed and row["status"] in ("no_candidate", "rejected", "committed")
            archived[k].append(row["round"])
            if row["round"] == 60:
                final[k] = row["committed_sha"]
        assert archived.keys() == expected and all(x == list(range(1, 61)) for x in archived.values())
        for r in index:
            k = tuple(r[f] for f in key)
            assert r["outer_seed"] == seed and r["final_sha256"] == final[k]
            assert r["test_ids"] == partitions[r["dataset"], r["peers"], r["rep"]]["test"]
        inputs.append((path, index))
    # Do not load held-out labels until every new run has passed its execution seal.
    labels = {name: original.dataset(name)[1] for name in {c["dataset"] for c in plan["eligible"]}}
    endpoints = []
    for path, index in inputs:
        with np.load(path / "sealed_predictions.npz") as pred:
            assert set(pred.files) == {r["token"] for r in index}
            for row in index:
                probs = pred[row["token"]]
                ids = row["test_ids"]
                assert probs.shape[0] == len(ids) and np.isfinite(probs).all()
                np.testing.assert_allclose(probs.sum(axis=1), 1., rtol=0, atol=1e-12)
                key = {k: row[k] for k in (*FIELDS, "outer_seed", "rep", "arm")}
                endpoints.append({**key, "n_test": len(ids), **base.quality(probs, labels[row["dataset"]][ids]),
                                  "final_sha256": row["final_sha256"]})
    write_csv("quality_endpoints_new", endpoints)
    return endpoints


def signs(v):
    return dict(positive=sum(x > EPS for x in v), negative=sum(x < -EPS for x in v),
                tie=sum(abs(x) <= EPS for x in v))


def assess_quality(plan, endpoints):
    key_fields = (*FIELDS, "reference", "outer_seed", "rep")
    pair_detail = []
    for r in rows(EVIDENCE / "quality_pairs.csv"):
        if r["reference"] in ("Unscreened", "J-Rank-Feasible"):
            key = {k: int(r[k]) if k in ("peers", "q", "outer_seed", "rep") else r[k] for k in key_fields}
            deltas = {m: float(r[m]) for m in METRICS}
            left = {m: float(r[f"vcia_{m}"]) for m in ("nmi", "ari")}
            ref = {m: left[m] - deltas[f"delta_{m}_pp"] / 100 for m in left}
            pair_detail.append({**key, "cohort": "original", "n_test": int(r["n_test"]),
                                "vcia_nmi_pct": 100*left["nmi"], "ref_nmi_pct": 100*ref["nmi"], "delta_nmi_pp": deltas["delta_nmi_pp"],
                                "vcia_ari_pct": 100*left["ari"], "ref_ari_pct": 100*ref["ari"], "delta_ari_pp": deltas["delta_ari_pp"],
                                "same_final_state": int(r["same_final_state"])})
    assert len(pair_detail) == 288
    pairs = defaultdict(dict)
    for r in endpoints:
        key = (*[r[f] for f in FIELDS], "J-Rank-Feasible", r["outer_seed"], r["rep"])
        pairs[key][r["arm"]] = r
    for key, arms in sorted(pairs.items()):
        left, ref = (arms[a] for a in ARMS)
        assert left["n_test"] == ref["n_test"]
        pair_detail.append({**dict(zip(key_fields, key, strict=True)), "cohort": "extension", "n_test": left["n_test"],
                            "vcia_nmi_pct": 100*left["nmi"], "ref_nmi_pct": 100*ref["nmi"], "delta_nmi_pp": 100*(left["nmi"]-ref["nmi"]),
                            "vcia_ari_pct": 100*left["ari"], "ref_ari_pct": 100*ref["ari"], "delta_ari_pp": 100*(left["ari"]-ref["ari"]),
                            "same_final_state": int(left["final_sha256"] == ref["final_sha256"])})
    assert len(pairs) == len(plan["eligible"]) * len(NEW_SEEDS) * 2
    group = defaultdict(lambda: defaultdict(list))
    for r in pair_detail:
        group[tuple(r[k] for k in (*FIELDS, "reference"))][r["outer_seed"]].append(r)
    flags = {tuple(r[k] for k in (*FIELDS, "reference")): r["flips"] for r in plan["original_directions"]}
    summary = []
    for key, rates in sorted(group.items()):
        assert tuple(sorted(rates)) in (OLD_SEEDS, OLD_SEEDS + NEW_SEEDS)
        assert all(len(nested) == 2 for nested in rates.values())
        for metric in METRICS:
            means = {seed: mean(r[metric] for r in rr) for seed, rr in rates.items()}
            existing = [means[s] for s in OLD_SEEDS]
            new = [means[s] for s in NEW_SEEDS if s in means]
            prior = signs(existing)
            later = signs(new)
            pooled = signs(existing + new)
            old_flip = int(metric in flags[key])
            assert old_flip == bool(prior["positive"] and prior["negative"])
            summary.append({**dict(zip((*FIELDS, "reference"), key, strict=True)),
                            "metric": "NMI" if metric == METRICS[0] else "ARI", "old_sign_flip": old_flip,
                            **{f"split_{s}": means.get(s) for s in (*OLD_SEEDS, *NEW_SEEDS)},
                            "old_mean_pp": mean(existing), "new_mean_pp": mean(new) if new else None,
                            "all_mean_pp": mean(existing + new), "all_n_splits": len(existing + new),
                            **{f"new_{k}": (v if new else None) for k, v in later.items()},
                            **{f"all_{k}": v for k, v in pooled.items()}})
    write_csv("quality_split_summary", summary)
    write_csv("quality_nested_pairs", pair_detail)
    return summary, pair_detail


def section(caption, note, headings, records):
    return {"caption": caption, "note": note, "head": headings, "rows": [list(r.values()) for r in records], "table": True}



if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("timing", "quality"))
    parser.add_argument("--data", type=Path, default=original.DATA)
    parser.add_argument("--closed-loop", type=Path, default=EVIDENCE)
    parser.add_argument("--runs", type=Path, default=HERE)
    args = parser.parse_args()
    original.DATA = args.data
    EVIDENCE = run_core_gap.EVIDENCE = args.closed_loop
    HERE = run_core_gap.HERE = args.runs
    if args.action == "timing":
        assess_timing()
    else:
        plan = load(HERE / "quality_plan.json")
        current = make_plan()
        assert plan["prior_pairs_sha256"] == current["prior_pairs_sha256"] and plan["eligible"] == current["eligible"]
        assess_quality(plan, scored_new(plan))
