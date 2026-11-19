"""Rebuild pilot decisions and verify the saved state and outcome chains."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations, product
from json import dumps, loads
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from threadpoolctl import threadpool_limits

from .learner import initialize, load_proteins, predict, prepare_pilot, propose
from .provenance import IMDB_SOURCES, PROTEINS_SOURCES, validate_source_hashes
from .run_imdb_pilot import load_imdb


def read_json(path):
    return loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def state_hash(maps, claims):
    return sha256(
        np.asarray(maps, dtype="<f8").tobytes()
        + np.asarray(claims, dtype="<f8").tobytes()
    ).hexdigest()


def stream(seed, round_no, purpose):
    return np.random.default_rng(np.random.SeedSequence([seed, round_no, purpose]))


def close(expected, observed):
    np.testing.assert_allclose(expected, observed, rtol=1e-12, atol=1e-14)


def aggregate(outputs, weights):
    return np.einsum("i,ijk->jk", weights, outputs) / weights.sum()


def effects(candidate, stored):
    return np.abs(candidate - stored).mean(axis=-1), candidate.argmax(
        1
    ) != stored.argmax(1)


def bernstein(values, output_range, alpha):
    n = len(values)
    log = np.log(2 / alpha)
    return min(
        output_range,
        values.mean()
        + np.sqrt(2 * values.var(ddof=1) * log / n)
        + 7 * output_range * log / (3 * (n - 1)),
    )


def check_quality(scores, labels, logged):
    assignment = scores.argmax(1)
    truth_ids, labels = np.unique(labels, return_inverse=True)
    counts = contingency_matrix(labels, assignment)
    rows, cols = linear_sum_assignment(-counts)
    xwalk = dict(zip(np.unique(assignment)[cols], rows))
    matched = [xwalk.get(value, -1) for value in assignment]
    close(normalized_mutual_info_score(labels, assignment), logged["nmi"])
    close(adjusted_rand_score(labels, assignment), logged["ari"])
    close(counts[rows, cols].sum() / len(labels), logged["acc"])
    close(
        f1_score(
            labels,
            matched,
            labels=np.arange(len(truth_ids)),
            average="macro",
            zero_division=0,
        ),
        logged["macro_f1"],
    )


def verify_run(directory, summary, manifest, pilot):
    traces = [
        loads(line)
        for line in (directory / "rounds.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    seed, regime, arm = (summary[k] for k in ("seed", "regime", "arm"))
    cfg = manifest["decision"]
    audit_ids, validation_ids = pilot.audit_ids, pilot.test_ids
    outcome = {
        "rounds": len(traces),
        "commits": 0,
        "hold_no_candidate": 0,
        "hold_verifier": 0,
        "accepted_population_violations": 0,
        "sampled_audits": 0,
    }
    assert len(traces) == manifest["rounds"] == summary["rounds"]
    assert summary["initialization_seed"] == int(
        stream(seed, 0, 5).integers(0, 2**32, dtype=np.uint32)
    )
    maps = np.round(initialize(pilot, summary["initialization_seed"]), 8)
    maps[maps == 0] = 0
    claims = np.full(len(maps), manifest["normal_claim"])
    last_submit, last_content = np.zeros(len(maps), int), np.zeros(len(maps), int)
    indexed = list(range(len(maps)))
    masks = list(combinations(indexed, cfg["q"]))
    candidates_keys = {
        "PCOC": lambda c: (-c["H"], -c["A"], c["phi"], c["ell"], c["selected"]),
        "Drift-First": lambda c: (c["ell"], c["phi"], -c["H"], -c["A"], c["selected"]),
        "No-Count": lambda c: (-c["A"], c["phi"], c["ell"], c["selected"]),
        "No-Age": lambda c: (-c["H"], c["phi"], c["ell"], c["selected"]),
        "J-Rank-Feasible": lambda c: (c["U"], c["selected"]),
    }
    candidates_keys["Unscreened"] = candidates_keys["PCOC"]
    with (
        np.load(directory / "states_and_draws.npz") as states_file,
        np.load(directory / "finite_effects.npz") as exact_file,
        np.load(directory / "inner_validation_outputs.npz") as predictions_file,
    ):
        states, exact, predictions = (
            dict(states_file),
            dict(exact_file),
            dict(predictions_file),
        )
        np.testing.assert_array_equal(exact["audit_ids"], audit_ids)
        np.testing.assert_array_equal(predictions["graph_ids"], validation_ids)
        policy_names = exact["policy_names"].tolist()
        eps, cap, q = (cfg[k] for k in ("epsilon", "claim_cap", "q"))
        initial_hash = state_hash(maps, claims)
        warmup = []
        for offset, trace in enumerate(traces):
            t = offset + 1
            assert (trace["round"], trace["seed"], trace["regime"], trace["arm"]) == (
                t,
                seed,
                regime,
                arm,
            )
            assert trace["stored_state_hash"] == state_hash(maps, claims)
            np.testing.assert_array_equal(maps, states["stored_maps"][offset])
            np.testing.assert_array_equal(claims, states["stored_claims"][offset])
            np.testing.assert_array_equal(last_content, trace["last_content_before"])
            np.testing.assert_array_equal(last_submit, trace["last_submit_before"])
            local_seed = int(stream(seed, t, 1).integers(0, 2**32, dtype=np.uint32))
            assert trace["local_seed"] == local_seed
            current = np.round(propose(pilot, maps, local_seed), 8)
            current[current == 0] = 0
            new_claims = np.full(len(maps), manifest["normal_claim"])
            repeat = np.flatnonzero((t + 2 * np.arange(len(maps))) % 7 == 0)
            current[repeat], new_claims[repeat] = maps[repeat], claims[repeat]
            stress = manifest["perturbation"]
            if (
                regime == "Perturbed"
                and stress["rounds_inclusive"][0] <= t <= stress["rounds_inclusive"][1]
            ):
                for i in stress["zero_based_peers"]:
                    current[i] = current[i, ::-1].copy()
                    new_claims[i] = stress["claim"]
            np.testing.assert_array_equal(repeat, trace["scheduled_repeats"])
            np.testing.assert_array_equal(current, states["current_maps"][offset])
            np.testing.assert_array_equal(new_claims, states["current_claims"][offset])
            assert trace["current_state_hash"] == state_hash(current, new_claims)
            changed_operator = (current != maps).any(axis=(1, 2))
            changed_claim = new_claims != claims
            changed = changed_operator | changed_claim
            np.testing.assert_array_equal(changed, trace["content_changed_in_proposal"])
            picked_design = stream(seed, t, 2).choice(
                audit_ids, manifest["design_draws"], replace=True
            )
            np.testing.assert_array_equal(
                states["design_probe_ids"][offset], picked_design
            )
            assert trace["design_unique_graphs"] == len(np.unique(picked_design))
            design_indices = np.searchsorted(audit_ids, picked_design)
            old = np.stack([predict(pilot, bank, audit_ids) for bank in maps])
            new = np.stack([predict(pilot, bank, audit_ids) for bank in current])
            close(old, exact["audit_stored"][offset])
            close(new, exact["audit_current"][offset])
            old_w, new_w = (
                eps + np.minimum(claims, cap),
                eps + np.minimum(new_claims, cap),
            )
            all_new, all_old = aggregate(new, new_w), aggregate(old, old_w)
            scope = 2 / pilot.n_clusters
            eligible, detectable = [], []
            for i, observed in enumerate(trace["version_stats"]):
                counter = new.copy()
                counter[i] = old[i]
                weights = new_w.copy()
                weights[i] = old_w[i]
                counter_out = aggregate(counter, weights)
                others = np.delete(indexed, i)
                background = aggregate(new[others], new_w[others])
                J = effects(all_new, counter_out)[0][design_indices]
                D = (
                    effects(all_new, background)[0]
                    - effects(counter_out, background)[0]
                )[design_indices]
                bound = bernstein(J, scope, cfg["delta_screen"] / len(maps))
                close(
                    [J.mean(), J.var(ddof=1), D.mean(), D.var(ddof=1), bound],
                    [
                        observed[k]
                        for k in ("mean_J", "var_J", "mean_D", "var_D", "upper_J")
                    ],
                )
                eligible.append(bound <= cfg["kappa"])
                detectable.append(changed[i] and J.mean() > cfg["nu"])
                assert (
                    observed["eligible"] == eligible[-1]
                    and observed["detectable"] == detectable[-1]
                )
            assert len(trace["candidates"]) == comb(len(maps), q)
            masks_outputs, screened, unscreened = {}, [], []
            for mask, observed in zip(masks, trace["candidates"], strict=True):
                mask_ids = np.asarray(mask)
                outputs, weights = old.copy(), old_w.copy()
                outputs[mask_ids], weights[mask_ids] = new[mask_ids], new_w[mask_ids]
                masks_outputs[mask] = effects(aggregate(outputs, weights), all_old)
                drift, flips = (
                    values[design_indices].mean() for values in masks_outputs[mask]
                )
                hits = [i for i in mask if detectable[i]]
                A = sum(min(t - last_content[i], cfg["age_cap"]) for i in hits)
                assert list(mask) == observed["selected"]
                close(
                    [
                        drift,
                        flips,
                        len(hits),
                        A,
                        sum(trace["version_stats"][i]["upper_J"] for i in mask),
                    ],
                    [observed[k] for k in ("ell", "phi", "H", "A", "U")],
                )
                passed_screen = all(eligible[i] for i in mask)
                feasible = (
                    drift <= cfg["eta"] * cfg["chi"]
                    and flips <= cfg["eta"] * cfg["gamma"]
                )
                assert (passed_screen, feasible) == (
                    observed["screened"],
                    observed["feasible"],
                )
                if feasible and hits:
                    unscreened.append(observed)
                    if passed_screen:
                        screened.append(observed)
            assert trace["online_selection_compositions"] == (
                len(masks)
                if arm == "Unscreened"
                else sum(c["screened"] for c in trace["candidates"])
            )
            H_max = max((c["H"] for c in screened), default=0)
            assert trace["priority_count_active"] == (
                len({c["H"] for c in screened}) > 1
            )
            assert trace["priority_age_active"] == (
                len({c["A"] for c in screened if c["H"] == H_max}) > 1
            )
            selections = {
                name: min(
                    unscreened if name == "Unscreened" else screened,
                    key=key,
                    default=None,
                )
                for name, key in candidates_keys.items()
            }
            random_seed = int(stream(seed, t, 3).integers(0, 2**32, dtype=np.uint32))
            selections["Random-Feasible"] = (
                screened[np.random.default_rng(random_seed).integers(len(screened))]
                if screened
                else None
            )
            verify = any(candidate is not None for candidate in selections.values())
            n = manifest["verify_draws"]
            picked_verify = (
                stream(seed, t, 4).choice(audit_ids, n, replace=True)
                if verify
                else np.full(n, -1)
            )
            np.testing.assert_array_equal(
                states["verification_probe_ids"][offset], picked_verify
            )
            assert trace["verify_draws"] == (n if verify else 0)
            assert trace["verify_unique_graphs"] == (
                len(np.unique(picked_verify)) if verify else 0
            )
            verify_indices = np.searchsorted(audit_ids, picked_verify)
            assert set(selections) == set(trace["policies"]) == set(policy_names)
            executed_maps, executed_claims = maps, claims
            for p, name in enumerate(policy_names):
                observed = trace["policies"][name]
                candidate = selections[name]
                assert observed["candidate"] == candidate and observed[
                    "closed_loop"
                ] == (name == arm)
                assert exact["present"][offset, p] == (candidate is not None)
                if candidate is None:
                    assert (
                        observed["status"] == "no_candidate" and observed["hold_reason"]
                    )
                    if name == arm:
                        outcome["hold_no_candidate"] += 1
                    continue
                mask = tuple(candidate["selected"])
                ids = np.asarray(mask)
                staged_maps, staged_claims = maps.copy(), claims.copy()
                staged_maps[ids], staged_claims[ids] = current[ids], new_claims[ids]
                assert (
                    state_hash(staged_maps, staged_claims)
                    == observed["candidate_state_hash"]
                )
                drift, flips = masks_outputs[mask]
                close(drift, exact["drifts"][offset, p])
                np.testing.assert_array_equal(flips, exact["flips"][offset, p])
                population = observed["finite_audit_law"]
                close(drift.mean(), population["ell"])
                assert (
                    population["size"] == len(audit_ids)
                    and population["flips"] == flips.sum()
                )
                within = drift.mean() <= cfg["chi"] and flips.mean() <= cfg["gamma"]
                assert population["within_tolerance"] == within
                batch_drift, batch_flips = drift[verify_indices], flips[verify_indices]
                k = int(batch_flips.sum())
                alpha = cfg["delta_verify"] / 2
                drift_bound = bernstein(batch_drift, scope, alpha)
                flip_bound = 1.0 if k == n else beta.isf(alpha, k + 1, n - k)
                result = observed["verification"]
                close(
                    [batch_drift.mean(), k / n, drift_bound, flip_bound],
                    [
                        result[field]
                        for field in ("ell", "phi", "upper_ell", "upper_phi")
                    ],
                )
                assert result["flips"] == k and result["draws"] == n
                assert result["unique_graphs"] == len(np.unique(picked_verify))
                accepted = drift_bound <= cfg["chi"] and flip_bound <= cfg["gamma"]
                assert (
                    accepted == result["accepted"] == (observed["status"] == "accepted")
                )
                assert result["drift_failed"] == (drift_bound > cfg["chi"])
                assert result["assignment_failed"] == (flip_bound > cfg["gamma"])
                op, claim = changed_operator[ids], changed_claim[ids]
                updates = {
                    "operator_only": int((op & ~claim).sum()),
                    "claim_only": int((~op & claim).sum()),
                    "joint": int((op & claim).sum()),
                }
                assert observed["attempted_content_updates"] == updates
                zero = dict.fromkeys(updates, 0)
                assert observed["accepted_candidate_updates"] == (
                    updates if accepted else zero
                )
                assert observed["committed_content_updates"] == (
                    updates if accepted and name == arm else zero
                )
                outcome["sampled_audits"] += 1
                if name == arm and accepted:
                    executed_maps, executed_claims = staged_maps, staged_claims
                    last_submit[ids] = t
                    last_content[ids[changed[ids]]] = t
                    outcome["commits"] += 1
                    outcome["accepted_population_violations"] += int(not within)
                elif name == arm:
                    outcome["hold_verifier"] += 1
            maps, claims = executed_maps, executed_claims
            np.testing.assert_array_equal(last_submit, trace["last_submit_after"])
            np.testing.assert_array_equal(last_content, trace["last_content_after"])
            assert state_hash(maps, claims) == trace["executed_state_hash"]
            pred = aggregate(
                np.stack([predict(pilot, bank, validation_ids) for bank in maps]),
                eps + np.minimum(claims, cap),
            )
            close(pred, predictions["probabilities"][offset])
            check_quality(
                pred,
                pilot.labels[validation_ids],
                trace["policies"][arm]["post_decision_inner_validation"],
            )
            if t < stress["rounds_inclusive"][0]:
                warmup.append(
                    (trace["current_state_hash"], trace["executed_state_hash"])
                )
        np.testing.assert_array_equal(maps, states["final_maps"])
        np.testing.assert_array_equal(claims, states["final_claims"])
        assert outcome["commits"] == summary["accepted_rounds"]
        assert (
            summary["endpoint_inner_validation"]
            == traces[-1]["policies"][arm]["post_decision_inner_validation"]
        )
    return initial_hash, warmup, outcome


def main():
    if not __debug__:
        raise RuntimeError("Run the artifact checks without Python's optimization flag")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="TU source ZIP for the archived run"
    )
    parser.add_argument("--reference-manifest", type=Path)
    args = parser.parse_args()
    manifest, summary = (
        read_json(args.results / file) for file in ("manifest.json", "run_index.json")
    )
    dataset_name = manifest.get("dataset_name", "PROTEINS")
    loader, source_names = {
        "PROTEINS": (load_proteins, PROTEINS_SOURCES),
        "IMDB-BINARY": (load_imdb, IMDB_SOURCES),
    }[dataset_name]
    source_origin = validate_source_hashes(manifest["source_sha256"], source_names)
    raw = loader(args.dataset)
    assert raw.source_sha256 == manifest["dataset_sha256"]
    if args.reference_manifest:
        assert digest(args.reference_manifest) == manifest.get(
            "reference_manifest_sha256"
        )
    expected = set(product(manifest["seeds"], manifest["regimes"], manifest["arms"]))
    assert len(summary) == len(expected)
    assert {(run["seed"], run["regime"], run["arm"]) for run in summary} == expected
    output = {
        "dataset": dataset_name,
        "trajectories": len(summary),
        "source_hashes_match": True,
        "dataset_hash_matches": True,
        "source_origin": source_origin,
        "shared_schedule_regenerated": True,
        "warmup_paths_match_between_regimes": True,
        "runs": {},
    }
    with threadpool_limits(limits=1):
        for seed in manifest["seeds"]:
            pilot = prepare_pilot(
                raw, seed, sealed_outer_seed=manifest["sealed_outer_seed"]
            )
            split = read_json(args.results / f"split_{seed}.json")
            for expected_ids, key in (
                (pilot.train_ids, "train_ids"),
                (pilot.audit_ids, "audit_ids"),
                (pilot.test_ids, "inner_validation_ids"),
            ):
                np.testing.assert_array_equal(expected_ids, split[key])
            for ids, observed in zip(
                pilot.peer_train_ids, split["peer_train_ids"], strict=True
            ):
                np.testing.assert_array_equal(ids, observed)
            assert split["partition_attempts"] == pilot.partition_attempts
            hashes, warmups = set(), {}
            for run in (run for run in summary if run["seed"] == seed):
                try:
                    initial, warmup, counts = verify_run(
                        args.results / run["path"], run, manifest, pilot
                    )
                except AssertionError as error:
                    raise AssertionError(f"Inconsistent run: {run['path']}") from error
                hashes.add(initial)
                warmups.setdefault(run["arm"], []).append(warmup)
                output["runs"][run["path"]] = counts
            assert len(hashes) == 1
            assert all(
                warmup == pairs[0] for pairs in warmups.values() for warmup in pairs
            )
    output["totals"] = {
        key: sum(result[key] for result in output["runs"].values())
        for key in next(iter(output["runs"].values()))
    }
    print(dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
