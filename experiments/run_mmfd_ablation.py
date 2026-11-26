"""Rerun the scalar-MMFD composition ablation on the frozen graph splits."""

from argparse import ArgumentParser
import csv
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_checks as checks
from decision import _full_key
from score_checks import quality

old = checks.old
base = checks.base
SPLITS = Path(__file__).resolve().parents[1] / "configs" / "mmfd_splits.json"
KEY = ("dataset", "outer_seed", "alpha", "condition", "arm", "round")
ARMS = ("VCIA", "No-Composition")


def graph_mmfd(graphs):
    # Fan (ICML 2025), Algorithm 1, with the released rank-20 setting.
    values = []
    for graph in graphs:
        n = len(graph.neighbors)
        adjacency = np.eye(n)
        for node, neighbors in enumerate(graph.neighbors):
            adjacency[node, list(neighbors)] = 1
        spectrum, directions = np.linalg.eigh(adjacency)
        largest = np.argsort(-np.abs(spectrum), kind="stable")[:20]
        factor_means = directions[:, largest].mean(axis=0) * np.sqrt(np.abs(spectrum[largest]))
        values.append(np.linalg.norm(factor_means))
    return np.asarray(values)[:, None]


def normalize_block(matrix, train):
    center = matrix[train].mean(axis=0)
    centered = matrix - center
    rms = np.sqrt(np.square(centered[train]).sum(axis=1).mean())
    return centered / max(rms, 1e-12)


def fingerprint(state):
    return base.state_id(state.maps, state.claims)


def proposal(pilot, state, seed, row):
    round_no = row["round"]
    condition = row["condition"]
    peers = len(state.maps)
    maps = old.proposals(pilot, state.maps, seed, round_no, row["topology"])
    claims = np.full(peers, .25)
    repeat = np.flatnonzero((round_no + 2 * np.arange(peers)) % 7 == 0)
    maps[repeat], claims[repeat] = state.maps[repeat], state.claims[repeat]
    if 11 <= round_no <= 35 and condition == "Moderate":
        for i in (1, 1 + peers // 2):
            maps[i], claims[i] = np.roll(maps[i], 1, axis=0), 1.
    return maps, claims


def cached_predictions(pilot, state, maps, claims):
    cache = base.PredictionCache(pilot)
    return cache, (maps, state.maps, claims, state.claims)


def evaluate(cache, operators, ids):
    maps, stored, claims, old_claims = operators
    return cache.get(maps, ids), cache.get(stored, ids), claims, old_claims


def design(pilot, state, seed, round_no, config, cache, operators, draws):
    sequence, ids, index = base.draw_ids(pilot, seed, round_no, 2, draws)
    maps, _, claims, _ = operators
    changed = np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
    selection = base.design(*evaluate(cache, operators, ids), changed, round_no - state.last_content,
                            index, config, screen_only=True)
    return selection, sequence


def stage(state, maps, claims, selected):
    staged = base.state_id(*base.full_state(state, maps, claims, selected))
    counts, changed = base.counts(state, maps, claims, selected)
    return staged, counts["operator_only"] + counts["joint"], changed


def closed_step(pilot, state, seed, row, config, arm):
    r = row["round"]
    maps, claims = proposal(pilot, state, seed, row)
    cache, operators = cached_predictions(pilot, state, maps, claims)
    pool, _ = design(pilot, state, seed, r, config, cache, operators, old.PROTOCOL["design_draws"])
    available = [c for c in pool.candidates if c.H > 0]
    choice = (min(available, key=_full_key, default=None)
              if arm == "No-Composition" else pool.selections["PCOC"])
    record = dict(row, arm=arm, pre_sha=fingerprint(state), status="no_candidate",
                  mask=None, candidate_sha=None, finite_feasible=None, exact_ell=None, exact_phi=None,
                  possible_operator_updates=0, operator_updates=0, audit_draws=0,
                  no_candidate_reason=(("insufficient_eligible_peers" if
                                       sum(v.eligible for v in pool.version_stats) < config.q
                                       else "no_detectable_update") if arm == "No-Composition"
                                       else pool.hold_reasons["PCOC"]),
                  eligible_peers=sum(v.eligible for v in pool.version_stats),
                  screened_detectable_masks=len(available),
                  design_feasible_masks=sum(c.feasible for c in available),
                  upper_ell=None, upper_phi=None, new_state_sha=None)
    if choice is not None:
        draws = old.PROTOCOL["verification_draws"]
        _, ids, idx = base.draw_ids(pilot, seed, r, 4, draws)
        audited = base.verify(*evaluate(cache, operators, ids), choice, idx, config)
        exact = base.exact(*evaluate(cache, operators, pilot.audit_ids), choice, config)
        staged, n_updates, changed = stage(state, maps, claims, choice.selected)
        if audited.accepted:
            state = base.commit(state, maps, claims, choice.selected, changed, r, staged)
        record.update(status="committed" if audited.accepted else "rejected",
                      mask=old.mask(choice), candidate_sha=staged, finite_feasible=int(exact.within_tolerance),
                      exact_ell=exact.ell, exact_phi=exact.phi, possible_operator_updates=n_updates,
                      operator_updates=n_updates if audited.accepted else 0, audit_draws=draws,
                      upper_ell=audited.upper_ell, upper_phi=audited.upper_phi, no_candidate_reason=None)
    record["new_state_sha"] = fingerprint(state)
    return state, record


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rounds(rows, reference):
    def key(row):
        return tuple(str(row[name]) for name in KEY)

    with reference.open(encoding="utf-8-sig", newline="") as handle:
        archived = {key(row): row for row in csv.DictReader(handle)}
    for row in rows:
        previous = archived[key(row)]
        for name in ("status", "mask", "pre_sha", "candidate_sha", "new_state_sha",
                     "operator_updates", "audit_draws", "finite_feasible"):
            value = "" if row[name] is None else str(row[name])
            if value != previous[name]:
                raise ValueError(f"{key(row)}: {name} differs from the archived run")
    print(f"Matched {len(rows)} round decisions and state fingerprints.")


def run(output, data, smoke, reference, splits=SPLITS):
    frozen = json.loads(splits.read_text(encoding="utf-8"))
    old.DATA = data
    rounds, index, predictions, truth = [], [], {}, {}
    rounds_per_trajectory = 12 if smoke else 60
    dataset_hashes = {}
    for name in ("PROTEINS", "IMDB-BINARY"):
        graphs, labels, _, dataset_hashes[name] = old.dataset(name)
        truth[name] = labels
        _, fingerprints = old._wl_features(graphs, 2, 512)
        raw = graph_mmfd(graphs)
        partitions = [split for split in frozen if split["dataset"] == name]
        for split in partitions[:1] if smoke else partitions:
            train, audit, test = (np.asarray(split[role]) for role in ("train", "audit", "test"))
            clients = tuple(np.asarray(ids) for ids in split["peer_train"])
            roles = [set(fingerprints[i] for i in ids) for ids in (train, audit, test)]
            assert all(not (left & right) for left, right in combinations(roles, 2))
            np.testing.assert_array_equal(np.sort(np.concatenate(clients)), train)
            features = normalize_block(raw, train)
            outer, seed = split["outer_seed"], split["run_seed"]
            k = split["k"]
            pilot = old.PilotData(features, np.full(len(labels), -1), train, audit, test, clients,
                                  fingerprints, outer, split["partition_seed"], split["allocation_attempts"],
                                  0, k, 2, 512, .25)
            config = base.DecisionConfig.normalized(6, n_clusters=k, epsilon=.1)
            initial = old.initial_maps(pilot, seed)
            for condition, arm in product(("Clean", "Moderate"), ARMS):
                state = checks.start_state(initial)
                initial_sha = fingerprint(state)
                tag = dict(dataset=name, outer_seed=outer, alpha=split["alpha"], condition=condition,
                           representation="MMFD", peers=8, topology="Ring", q=6)
                trajectory = []
                for number in range(1, rounds_per_trajectory + 1):
                    state, row = closed_step(pilot, state, seed, {**tag, "round": number}, config, arm)
                    row.update(ell_tolerance=config.chi, phi_tolerance=config.gamma)
                    expected_sha = row["candidate_sha"] if row["status"] == "committed" else row["pre_sha"]
                    assert row["new_state_sha"] == expected_sha
                    assert row["audit_draws"] == 8192 * (row["mask"] is not None)
                    trajectory.append(row)
                rounds.extend(trajectory)
                token = f"p{len(index):04d}"
                predictions[token] = checks.outer_scores(pilot, state.maps, state.claims, config)
                index.append({**tag, "arm": arm, "token": token, "test_ids": test, "run_seed": seed,
                              "partition_seed": split["partition_seed"], "initial_sha": initial_sha,
                              "final_sha": fingerprint(state), "scheduled_rounds": len(trajectory),
                              "accepted_rounds": sum(r["status"] == "committed" for r in trajectory),
                              "operator_updates": sum(r["operator_updates"] for r in trajectory),
                              "audit_draws": sum(r["audit_draws"] for r in trajectory),
                              "over_limit_accepted": sum(r["status"] == "committed" and r["finite_feasible"] == 0
                                                        for r in trajectory)})
            print(f"{name} split={outer} alpha={split['alpha']:g} completed", flush=True)
    if reference is not None:
        compare_rounds(rounds, reference)

    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output / "protocol.json", dict(smoke=smoke, rounds=rounds_per_trajectory,
                    splits_sha256=base.digest(splits), datasets_sha256=dataset_hashes,
                    audit_draws_per_nominee=8192, design_draws=512, blas_threads=1))
    write_rows(output / "rounds.csv", rounds)
    base.write_json(output / "index.json", index)
    prediction_file = output / "sealed_predictions.npz"
    np.savez_compressed(prediction_file, **predictions)
    sealed = ("protocol.json", "index.json", "rounds.csv", prediction_file.name)
    base.write_json(output / "seal.json", {"sha256": {f: base.digest(output / f) for f in sealed}})
    with np.load(prediction_file, allow_pickle=False) as bank:
        scores = []
        for row in index:
            ids = row["test_ids"]
            summary = {field: value for field, value in row.items() if field != "test_ids"}
            scores.append({**summary, **quality(bank[row["token"]], ids, truth[row["dataset"]])})
    write_rows(output / "trajectories.csv", scores)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=SPLITS, help="Path to the frozen graph split file")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reference", type=Path, help="Optional archived mmfd_composition_rounds.csv")
    args = parser.parse_args()
    if not args.splits.is_file():
        parser.error(f"Frozen split file missing: {args.splits}. Supply it with --splits.")
    with threadpool_limits(limits=1):
        run(args.output, args.data, args.smoke, args.reference, args.splits)
