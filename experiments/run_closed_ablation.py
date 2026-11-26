"""Run the locked six-arm closed-loop matrix against the original graph pipeline."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
import run_evidence as old

base = old.base
ORIGINAL_VERIFY = base.verify
ORIGINAL_SEARCH = old.search
ORIGINAL_COUNTS = base.counts
DIAG = {}
DATASETS = ("PROTEINS", "ENZYMES")
SPLITS = tuple(range(31501, 31507))
COMMON = ("VCIA", "Drop both", "J-Rank-Feasible")
ARMS = {"Clean": COMMON, "Moderate": (*COMMON, "Drift only", "Reassignment only", "No-Composition")}
NO_DRIFT = ("Reassignment only", "No-Composition")
NO_FLIPS = ("Drift only", "No-Composition")
KEYS = ("dataset", "outer_seed", "rep", "condition", "arm")


def search(args, policy):
    start = perf_counter()
    cur, mem, claims, old_claims, changed, ages, draws, config = args
    ids, weights = old._draw_counts(draws, cur.shape[1])
    n = len(draws)
    pool = old._Pool(cur, mem, claims, old_claims, ids, config)
    peers = tuple(range(len(cur)))
    setup_end = perf_counter()
    all_current = pool.composed(peers)
    upper, detectable, eligible = [], [], []
    for peer in peers:
        previous = pool.composed(tuple(i for i in peers if i != peer))
        dist = old._distance(all_current, previous)
        mean, variance = old._moments(dist, weights, n)
        u = old.empirical_bernstein_upper(mean, variance, pool.range, n, config.delta_screen / len(peers))
        upper.append(u)
        detectable.append(changed[peer] and mean > config.nu)
        if u <= config.kappa:
            eligible.append(peer)
    screen_end = perf_counter()
    baseline = pool.composed(())
    labels = baseline.argmax(axis=-1)
    capped = np.minimum(ages, config.age_cap)
    masks = tuple(combinations(eligible, config.q))
    feasible = []
    full_feasible = set()
    reasons = Counter()
    for mask in masks:
        distances, flips = old._composition_effect(pool, mask, baseline, labels, weights)
        ell = float(np.dot(distances, weights) / n)
        phi = flips / n
        useful = [i for i in mask if detectable[i]]
        if not useful:
            reasons["not_detectable"] += 1
            continue
        drift = ell > config.eta * config.chi
        flip = phi > config.eta * config.gamma
        reasons[("meets_both", "drift_only", "flip_only", "both_exceeded")[int(drift) + 2 * int(flip)]] += 1
        if not drift and not flip:
            full_feasible.add(mask)
        if (drift and policy not in NO_DRIFT) or (flip and policy not in NO_FLIPS):
            continue
        feasible.append(old.Candidate(mask, len(useful), float(capped[useful].sum()), sum(upper[i] for i in mask),
                                      phi, ell, True, True, None, None, 0.))
    if policy == "J-Rank-Feasible":
        key = lambda c: (c.U, c.selected)
    elif policy == "Drop both":
        key = lambda c: (c.phi, c.ell, c.selected)
    else:
        key = old._full_key
    selected = min(feasible, key=key, default=None)
    if selected is not None:
        reason = None
    elif not masks:
        reason = "screened_below_quota"
    elif reasons["not_detectable"] == len(masks):
        reason = "no_detectable_update"
    else:
        reason = "composition_constraints"
    feasible_masks = {candidate.selected for candidate in feasible}
    assert sum(reasons.values()) == len(masks)
    assert feasible_masks == full_feasible if policy in COMMON else feasible_masks >= full_feasible
    DIAG.update(eligible_peers=len(eligible), design_eligible_masks=len(feasible),
                design_no_detectable=reasons["not_detectable"], design_drift_only=reasons["drift_only"],
                design_reassignment_only=reasons["flip_only"], design_both_exceeded=reasons["both_exceeded"],
                design_meets_both=reasons["meets_both"], hold_reason=reason,
                design_ell=None if selected is None else selected.ell, design_phi=None if selected is None else selected.phi)
    if DIAG.pop("verify_original_search", False):
        legacy_choice, legacy_masks, legacy_feasible, _ = ORIGINAL_SEARCH(args, policy)
        assert (selected, masks, feasible_masks) == (legacy_choice, legacy_masks, legacy_feasible)
    return selected, masks, feasible_masks, (setup_end - start, screen_end - setup_end, perf_counter() - screen_end)


def audited(*args):
    result = ORIGINAL_VERIFY(*args)
    DIAG.update(audit_draws=result.draws, audit_unique_graphs=result.unique_graphs,
                audit_ell=result.ell, audit_phi=result.phi,
                audit_upper_ell=result.upper_ell, audit_upper_phi=result.upper_phi,
                audit_drift_failed=int(result.drift_failed), audit_reassignment_failed=int(result.assignment_failed))
    return result


def counted(*args):
    counts, changed = ORIGINAL_COUNTS(*args)
    DIAG.update(counts)
    return counts, changed


def run_split(name, outer, store, rounds):
    old.PROTOCOL["replay_rounds"] = []
    old.search, base.verify, base.counts = search, audited, counted
    graphs, labels, archive, digest = old.dataset(name)
    wl, fingerprints = old._wl_features(graphs, 2, 512)
    train, audit, test = base._split_roles(fingerprints, outer)
    roles = [set(fingerprints[i] for i in ids) for ids in (train, audit, test)]
    assert all(not left & right for left, right in combinations(roles, 2))
    features = old.representation(wl, None, train, "WL-TFIDF")
    k = 6 if name == "ENZYMES" else 2
    config = base.DecisionConfig.normalized(6, n_clusters=k, epsilon=.1)
    folder = store / f"{name}_{outer}"
    folder.mkdir()
    log = base.Rows(folder / "rounds.csv")
    index, splits, banks = [], [], {}
    sockets = base.connection()
    try:
        for rep in (1, 2):
            partition = outer * 10 + rep
            seed = 500000 + partition
            peers, tries = old._partition_peers(train, labels, partition, 8, .5, 10)
            assert np.array_equal(np.sort(np.concatenate(peers)), train)
            pilot = old.PilotData(features, np.full(len(labels), -1), train, audit, test, peers, fingerprints,
                                  outer, partition, tries, max(Counter(fingerprints).values()), k, 2, 512, .25)
            initial = old.initial_maps(pilot, seed)
            splits.append(dict(rep=rep, partition_seed=partition, stream_seed=seed, train=train, audit=audit,
                               test=test, peer_train=peers, decision=asdict(config), dataset_archive=archive.name,
                               dataset_sha=digest))
            for condition, arms in ARMS.items():
                for arm in arms:
                    token = f"{name}-{outer}-{rep}-{condition}-{arm}"
                    key = dict(dataset=name, outer_seed=outer, rep=rep, peers=8, q=6, topology="Ring",
                               condition=condition, arm=arm)
                    state = base.State(initial.copy(), np.full(8, .25), np.zeros(8, dtype=int), np.zeros(8, dtype=int))
                    initial_hash = base.state_id(state.maps, state.claims)
                    last_op = np.zeros(8, dtype=int)
                    for rnd in range(1, rounds + 1):
                        DIAG.clear()
                        if arm in ("VCIA", "J-Rank-Feasible") and rep == 1 and rnd <= 2:
                            DIAG["verify_original_search"] = True
                        previous = base.state_id(state.maps, state.claims)
                        next_state, prob, row, _ = old.round_step(pilot, state, {**key, "round": rnd}, seed, config, sockets)
                        last_op[np.any(state.maps != next_state.maps, axis=(1, 2))] = rnd
                        op_age, content_age = rnd - last_op, rnd - next_state.last_content
                        state = next_state
                        nominee = row["status"] != "no_candidate"
                        committed = row["status"] == "committed"
                        assert (row["committed_sha"] != previous) == committed
                        assert (row["candidate_sha"] == row["committed_sha"]) == committed
                        assert DIAG.get("audit_draws", 0) == 8192 * nominee
                        assert row["operator_updates"] == DIAG.get("operator_only", 0) + DIAG.get("joint", 0)
                        log.add({**row, "trajectory_id": token, "previous_sha": previous, "chi": config.chi,
                                 "gamma": config.gamma, "pool_size": len(audit), "design_draws": 512,
                                 "op_age_median": float(np.median(op_age)), "op_age_p90": float(np.percentile(op_age, 90)),
                                 "content_age_median": float(np.median(content_age)),
                                 "content_age_p90": float(np.percentile(content_age, 90)),
                                 **dict.fromkeys(("operator_only", "claim_only", "joint", "audit_draws", "audit_unique_graphs"), 0),
                                 **dict.fromkeys(("audit_ell", "audit_phi", "audit_upper_ell", "audit_upper_phi",
                                                  "audit_drift_failed", "audit_reassignment_failed")),
                                 **DIAG})
                    file_key = f"t{len(index):02}"
                    banks[file_key] = prob
                    index.append(dict(**key, trajectory_id=token, token=file_key, initial_sha=initial_hash,
                                      completed_rounds=rounds, test_size=len(test)))
    finally:
        for sock in sockets:
            sock.close()
        log.close()
    predictions = folder / "sealed_endpoints.npz"
    np.savez_compressed(predictions, **banks)
    digest = base.digest(predictions)
    base.write_json(folder / "index.json", index)
    base.write_json(folder / "splits.json", splits)
    base.write_json(folder / "prediction_seal.json", dict(sha256=digest, sealed_at_utc=base.now()))
    with np.load(predictions, allow_pickle=False) as sealed:
        for row in index:
            truth = labels[test]
            assigned = np.argmax(sealed[row["token"]], axis=1)
            row.update(nmi_pp=100 * normalized_mutual_info_score(truth, assigned),
                       ari_pp=100 * adjusted_rand_score(truth, assigned))
    import json
    (folder / "endpoints.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return name, outer, len(index)


def worker(name, outer, store, rounds, data):
    old.DATA = data
    with threadpool_limits(limits=1):
        return run_split(name, outer, store, rounds)


def history_match(store, history):
    columns = (*KEYS, "round")
    prior = {}
    for file in history:
        with file.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["peers"] == "8" and row["q"] == "6" and row["topology"] == "Ring":
                    prior[tuple(row[k] for k in columns)] = row
    checks = Counter()
    for file in sorted(store.glob("*/rounds.csv")):
        with file.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                previous = prior.get(tuple(row[k] for k in columns))
                if previous:
                    for col in ("status", "mask", "committed_sha", "candidate_sha", "operator_updates", "finite_feasible"):
                        if col in previous:
                            assert row[col] == previous[col], (file, tuple(row[k] for k in columns), col, row[col], previous[col])
                    checks[(row["arm"], row["outer_seed"])] += 1
    return [{"arm": arm, "split": int(split), "compared_rounds": n} for (arm, split), n in sorted(checks.items())]


def main(out, data, workers, smoke, history):
    from multiclass import __file__ as multiclass_file

    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(old.__file__), Path(base.__file__), HERE / "evidence_protocol.json",
               Path(multiclass_file), base.PILOT / "learner.py", base.PILOT / "decision.py",
               base.PILOT / "run_pilot.py"]
    jobs = [(DATASETS[0], SPLITS[0])] if smoke else list(product(DATASETS, SPLITS))
    rounds = 2 if smoke else 60
    from importlib.metadata import version
    binding = dict(start_utc=base.now(), smoke=smoke, roles="WL-TFIDF depth 2 bins 512, original fingerprint splits",
                   dataset_splits=jobs, paired_initializations=[1, 2], arms=ARMS, topology="Ring",
                   rounds=rounds, peers=8, quota=6, design_draws=512, audit_draws=8192, numpy_threads=1,
                   os=platform.platform(), cpu=platform.processor(), python=platform.python_version(), workers=workers,
                   versions={name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                   source_sha256={path.relative_to(HERE).as_posix(): base.digest(path) for path in sources})
    base.write_json(out / "execution.json", binding)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        work = {executor.submit(worker, name, outer, out, rounds, data): (name, outer) for name, outer in jobs}
        for done in as_completed(work):
            print("completed:", done.result(), flush=True)
    checks = history_match(out, history) if history else []
    if history and not checks:
        raise ValueError("No matching rounds were found in the provided history files.")
    base.write_json(out / "verification.json", dict(archived_rounds=checks, finished_utc=base.now(),
                   sha256={str(f.relative_to(out)): base.digest(f) for f in out.rglob("*") if f.is_file()}))


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("data"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--history", type=Path, action="append", default=[],
                     help="Optional prior rounds.csv to compare. Repeat for multiple files.")
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--smoke", action="store_true")
    options = cli.parse_args()
    main(options.output, options.data, options.workers, options.smoke, options.history)
