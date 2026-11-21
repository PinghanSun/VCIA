"""Paired diagnostics, heterogeneity sweep and fixed-state rank/audit replay."""

from argparse import ArgumentParser
from collections import Counter
from itertools import combinations, product
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
import run_evidence as old

base = old.base
SPEC = HERE / "replay_protocol.json"
PLAN = json.loads(SPEC.read_text(encoding="utf-8"))


def source_files():
    return [Path(__file__), SPEC, HERE / "run_evidence.py", HERE / "evidence_protocol.json",
            Path(base.__file__), HERE / "multiclass.py", HERE / "snapshots.py",
            HERE / "decisions.py", HERE / "requirements.txt",
            *(base.PILOT / f for f in ("learner.py", "decision.py", "run_pilot.py", "run_imdb_pilot.py"))]


def start_state(initial):
    peers = len(initial)
    return base.State(initial.copy(), np.full(peers, .25), np.zeros(peers, dtype=int),
                      np.zeros(peers, dtype=int))


def outer_scores(pilot, maps, claims, config):
    pred = base.all_peer_predictions(pilot, maps, pilot.test_ids)
    return base.aggregate(pred, pred, claims, claims, (), config)


def pooled_scores(pilot, seed):
    fit = KMeans(n_clusters=pilot.n_clusters, n_init=10, max_iter=300, tol=.0001,
                 random_state=seed, algorithm="lloyd")
    fit.fit(pilot.features[pilot.train_ids])
    return np.eye(pilot.n_clusters)[fit.predict(pilot.features[pilot.test_ids])]


def prepare_pilot(features, labels, fingerprints, roles, name, split, rep, alpha, peers, historical):
    train, audit, test = roles
    partition_seed = split * 10 + rep
    run_seed = 500000 + partition_seed
    if historical is None:
        peer_train, tries = old._partition_peers(
            train, labels, partition_seed, peers, alpha, PLAN["min_train_graphs_per_peer"],
            PLAN["allocation_attempt_limit"])
    else:
        for role, expected in zip(("train", "audit", "test"), roles, strict=True):
            np.testing.assert_array_equal(expected, historical[role])
        assert historical["partition_seed"] == partition_seed and historical["init_and_round_seed"] == run_seed
        peer_train = tuple(np.asarray(ids) for ids in historical["peer_train"])
        tries = historical["partition_attempts"]
    np.testing.assert_array_equal(np.sort(np.concatenate(peer_train)), train)
    assert min(map(len, peer_train)) >= PLAN["min_train_graphs_per_peer"]
    k = 6 if name == "ENZYMES" else 2
    pilot = old.PilotData(features, np.full(len(labels), -1), train, audit, test, peer_train,
                          fingerprints, split, partition_seed, tries, max(Counter(fingerprints).values()),
                          k, 2, 512, .25)
    label_bins = int(labels.max()) + 1
    counts = [np.bincount(labels[ids], minlength=label_bins) for ids in peer_train]
    global_dist = np.sum(counts, axis=0) / len(train)
    profiles = [.5 * np.abs(row / sum(row) - global_dist).sum() for row in counts]
    partitions = dict(train=train, audit=audit, test=test, peer_train=peer_train, k=k,
                      peer_train_class_counts=counts, partition_seed=partition_seed, run_seed=run_seed,
                      allocation_attempts=tries, weighted_label_tv=float(np.average(profiles, weights=list(map(len, peer_train)))))
    return pilot, run_seed, partitions


def audit_candidate(pilot, cache, state, maps, claims, candidate, seed, rnd, config, kind):
    start = perf_counter()
    if kind == "IID":
        _, ids, inverse = base.draw_ids(pilot, seed, rnd, 4, PLAN["iid_draws"])
    else:
        ids, inverse = pilot.audit_ids, None
    prepared = perf_counter()
    inputs = (cache.get(maps, ids), cache.get(state.maps, ids), claims, state.claims)
    inferred = perf_counter()
    if kind == "IID":
        check = base.verify(*inputs, candidate, inverse, config)
        passed = check.accepted
    else:
        check = base.exact(*inputs, candidate, config)
        passed = check.within_tolerance
    complete = perf_counter()
    return passed, check, dict(probe_prep_ms=1000*(prepared-start), probe_inference_ms=1000*(inferred-prepared),
                               decision_ms=1000*(complete-inferred), audit_total_ms=1000*(complete-start),
                               unique_audit_graphs=len(ids))


def candidate_bank(state, maps, claims, pick):
    new_maps, new_claims = base.full_state(state, maps, claims, pick.selected)
    changes, changed_ids = base.counts(state, maps, claims, pick.selected)
    return new_maps, new_claims, changes["operator_only"] + changes["joint"], changed_ids


def run_round(pilot, state, seed, config, tag, rnd, replay_logs, save):
    start = perf_counter()
    maps = old.proposals(pilot, state.maps, seed, rnd, PLAN["topology"])
    claims = np.full(len(maps), .25)
    repeat = np.flatnonzero((rnd + 2 * np.arange(len(maps))) % 7 == 0)
    maps[repeat], claims[repeat] = state.maps[repeat], state.claims[repeat]
    stressed = tag["condition"] == "Moderate" and 11 <= rnd <= 35
    if stressed:
        for peer in (1, 5):
            maps[peer] = np.roll(maps[peer], 1, axis=0)
            claims[peer] = 1.
    proposed = perf_counter()
    _, ids, design = base.draw_ids(pilot, seed, rnd, 2, PLAN["design_draws"])
    cache = base.PredictionCache(pilot)
    prepared = perf_counter()
    cur, mem = cache.get(maps, ids), cache.get(state.maps, ids)
    inferred = perf_counter()
    changes = np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
    pool = base.design(cur, mem, claims, state.claims, changes, rnd - state.last_content,
                       design, config, screen_only=True)
    selections = {"VCIA": pool.selections["PCOC"], "J-Rank": pool.selections["J-Rank-Feasible"]}
    pick = selections[tag["ranking"]]
    designed = perf_counter()
    row = {**tag, "round": rnd, "proposal_ms": 1000*(proposed-start),
           "design_prep_ms": 1000*(prepared-proposed), "design_inference_ms": 1000*(inferred-prepared),
           "search_ms": 1000*(designed-inferred), "mask": old.mask(pick),
           "evaluated_masks": len(pool.candidates), "design_feasible_masks": sum(c.rejection is None for c in pool.candidates),
           "perturbed_clients": 2 if stressed else 0, "status": "no_candidate"}
    audit_timing = dict.fromkeys(("probe_prep_ms", "probe_inference_ms", "decision_ms",
                                  "audit_total_ms", "unique_audit_graphs"))
    feasibility = ell = phi = stages = None
    n_updates = 0
    result_state = state
    if pick is not None:
        passed, check, audit_timing = audit_candidate(pilot, cache, state, maps, claims, pick, seed, rnd, config, tag["audit"])
        start_commit = perf_counter()
        row["status"] = "committed" if passed else "rejected"
        if passed:
            new_maps, new_claims, n_updates, changed_ids = candidate_bank(state, maps, claims, pick)
            result_state = base.commit(state, maps, claims, pick.selected, changed_ids, rnd,
                                       base.state_id(new_maps, new_claims))
        stages = 1000*(perf_counter()-start_commit)
        exact = check if tag["audit"] == "Census" else base.exact(
            base.all_peer_predictions(pilot, maps, pilot.audit_ids),
            base.all_peer_predictions(pilot, state.maps, pilot.audit_ids), claims, state.claims, pick, config)
        feasibility, ell, phi = int(exact.within_tolerance), exact.ell, exact.phi
    row.update(**audit_timing, finite_feasible=feasibility, finite_tv=ell, finite_flip_rate=phi,
               commit_ms=stages, operator_updates=n_updates,
               pipeline_ms=1000*(designed-start)+(audit_timing["audit_total_ms"] or 0)+(stages or 0),
               committed_sha=base.state_id(result_state.maps, result_state.claims))
    if tag["ranking"] == "VCIA" and tag["audit"] == "IID" and rnd in PLAN["replay_rounds"]:
        snap = {k: v for k, v in row.items() if k in ("dataset", "outer_seed", "alpha", "condition", "round")}
        masks = sorted(old.mask(x) for x in pool.candidates if x.rejection is None)
        prior_token = save(outer_scores(pilot, state.maps, state.claims, config))
        for rank, option in selections.items():
            candidate_token = possible_updates = None
            if option is not None:
                new_maps, new_claims, possible_updates, _ = candidate_bank(state, maps, claims, option)
                candidate_token = save(outer_scores(pilot, new_maps, new_claims, config))
            outcomes = {}
            kinds = PLAN["audit"] if rnd % 2 else PLAN["audit"][::-1]
            for kind in kinds:
                if option is not None:
                    fresh_cache = base.PredictionCache(pilot)
                    fresh_cache.get(maps, ids)
                    fresh_cache.get(state.maps, ids)
                    outcomes[kind] = audit_candidate(pilot, fresh_cache, state, maps, claims, option, seed, rnd, config, kind)
            for kind in PLAN["audit"]:
                passed, check, costs = outcomes.get(kind, (None, None, dict.fromkeys(audit_timing)))
                reference = outcomes["Census"][1] if outcomes else None
                replay_logs.add({**snap, "ranking": rank, "audit": kind, "order": kinds.index(kind) + 1,
                                 "candidate_family": ";".join(map(str, masks)), "feasible_mask_count": len(masks),
                                 "mask": old.mask(option), "finite_feasible": int(reference.within_tolerance) if reference else None,
                                 "finite_tv": reference.ell if reference else None,
                                 "finite_flip_rate": reference.phi if reference else None,
                                 "passed": int(passed) if passed is not None else None,
                                 "possible_operator_updates": possible_updates, "committed_operator_updates": possible_updates if passed else 0,
                                 "prior_token": prior_token, "candidate_token": candidate_token,
                                 **costs})
    return result_state, row


def main(out, smoke, prior):
    prior_files = ("seal.json", "score_manifest.json", "endpoints.csv", "recovery_pairs.csv", "recovery_summary.csv",
                   "splits.json", "paired_round_gaps.csv", "index.json", "sealed_outer_probabilities.npz")
    if prior is None and not smoke:
        raise ValueError("The full run requires --closed-loop with scored full-run outputs")
    if prior is not None and not all((prior / name).is_file() for name in prior_files):
        raise ValueError("--closed-loop must point to the scored full closed-loop run")
    out.mkdir(parents=True, exist_ok=False)
    files = source_files()
    names = PLAN["datasets"][:1] if smoke else PLAN["datasets"]
    execution = {"start_utc": base.now(), "smoke": smoke, "os": platform.platform(),
                 "cpu": platform.processor(),
                 "python": platform.python_version(), "BLAS_threads": 1,
                 "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                 "source_sha": {f.relative_to(HERE).as_posix(): base.digest(f) for f in files},
                 "datasets": {name: old.dataset(name)[-1] for name in names}}
    base.write_json(out / "execution.json", execution)
    history = {(row["dataset"], row["outer_seed"], row["rep"], row["peers"]): row for row in
               json.loads((prior / "splits.json").read_text(encoding="utf-8"))} if prior else {}
    probabilities, index, split_rows = {}, [], []

    def save(value):
        token = f"p{len(probabilities):05d}"
        probabilities[token] = value
        return token

    logs = {name: base.Rows(out / f"{name}.csv") for name in ("online_rounds", "replay")}
    try:
        for name in names:
            graphs, labels, _, _ = old.dataset(name)
            wl, fingerprints = old._wl_features(graphs, 2, 512)
            for outer in PLAN["outer_seeds"][:1] if smoke else PLAN["outer_seeds"]:
                roles = base._split_roles(fingerprints, outer)
                role_groups = [{fingerprints[i] for i in ids} for ids in roles]
                assert all(not a & b for a, b in combinations(role_groups, 2))
                features = old.representation(wl, None, roles[0], "WL-TFIDF")
                for rep, peers in ((1, 8), (2, 8), (1, 12), (2, 12)) if name != "IMDB-BINARY" else ():
                    historic = history.get((name, outer, rep, peers))
                    pilot, seed, split = prepare_pilot(features, labels, fingerprints, roles, name, outer, rep, .5, peers, historic)
                    config = base.DecisionConfig.normalized(6 if peers == 8 else 9, n_clusters=pilot.n_clusters, epsilon=.1)
                    init = start_state(old.initial_maps(pilot, seed))
                    for method, prob in (("Stored-only", outer_scores(pilot, init.maps, init.claims, config)),
                                         ("WL + K-means", pooled_scores(pilot, seed))):
                        index.append(dict(dataset=name, outer_seed=outer, rep=rep, peers=peers, alpha=.5, condition="Clean",
                                          method=method, cohort="historical", test_ids=split["test"], token=save(prob)))
                for alpha in ([.5] if smoke else PLAN["alphas"]):
                    rep, peers = 1, PLAN["peers"]
                    historic = history.get((name, outer, rep, peers)) if alpha == .5 else None
                    pilot, seed, split = prepare_pilot(features, labels, fingerprints, roles, name, outer, rep, alpha, peers, historic)
                    config = base.DecisionConfig.normalized(PLAN["quota"], n_clusters=pilot.n_clusters, epsilon=.1)
                    initial = old.initial_maps(pilot, seed)
                    init = start_state(initial)
                    tag = dict(dataset=name, outer_seed=outer, alpha=alpha)
                    split_rows.append({**tag, **split})
                    for method, prob in (("Stored-only", outer_scores(pilot, init.maps, init.claims, config)),
                                         ("WL + K-means", pooled_scores(pilot, seed))):
                        index.append({**tag, "rep": rep, "peers": peers, "condition": "Both", "method": method,
                                      "cohort": "online", "test_ids": split["test"], "token": save(prob)})
                    for condition, rank, audit in product(PLAN["conditions"], PLAN["ranking"], PLAN["audit"]):
                        params = {**tag, "condition": condition, "ranking": rank, "audit": audit}
                        state = start_state(initial)
                        for number in range(1, 13 if smoke else PLAN["rounds"] + 1):
                            state, row = run_round(pilot, state, seed, config, params, number, logs["replay"], save)
                            logs["online_rounds"].add(row)
                        index.append({**tag, "rep": rep, "peers": peers, "condition": condition, "method": f"{rank} / {audit}",
                                      "cohort": "online", "test_ids": split["test"],
                                      "token": save(outer_scores(pilot, state.maps, state.claims, config))})
                    print(f"{name} · {outer} · alpha={alpha:g} completed", flush=True)
    finally:
        for log in logs.values():
            log.close()
    np.savez_compressed(out / "sealed_predictions.npz", **probabilities)
    base.write_json(out / "index.json", index)
    base.write_json(out / "splits.json", split_rows)
    base.write_json(out / "seal.json", {"end_utc": base.now(), "smoke": smoke, "probability_arrays": len(probabilities),
                    "sha256": {f.name: base.digest(f) for f in out.iterdir() if f.is_file()},
                    "prior_record_sha256": {name: base.digest(prior / name) for name in prior_files} if prior else {}})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data", type=Path, default=old.DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--closed-loop", type=Path)
    args = parser.parse_args()
    old.DATA = args.data
    if not args.smoke and args.closed_loop is None:
        parser.error("the full run requires --closed-loop with scored outputs")
    output = args.output or (Path("runs/replay-smoke") if args.smoke else Path("runs/replay"))
    with threadpool_limits(1):
        main(output, args.smoke, args.closed_loop)
