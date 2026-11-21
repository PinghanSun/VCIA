"""Run paired graph-clustering trajectories and matched per-round decision replay."""

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from importlib.metadata import version
from json import dump, dumps
from pathlib import Path
from platform import platform, python_version, processor
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from threadpoolctl import threadpool_limits

from decision import DecisionConfig, aggregate, design, verify
from learner import initialize, load_proteins, predict, prepare_pilot, propose


ROUNDS = 30
DESIGN_DRAWS = 512
VERIFY_DRAWS = 8192
REGIMES = ("Clean", "Perturbed")
ARMS = ("PCOC", "Unscreened")
STRESS_PEERS = (1, 5)
SEALED_OUTER_SEED = 27001


@dataclass
class State:
    maps: np.ndarray
    claims: np.ndarray
    last_submit: np.ndarray
    last_content: np.ndarray


def encode(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path, obj):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        dump(obj, stream, default=encode, indent=2, allow_nan=False)
        stream.write("\n")


def rng(seed, round_no, purpose):
    return np.random.default_rng(np.random.SeedSequence([seed, round_no, purpose]))


def operator_key(operator):
    return sha256(np.asarray(operator, dtype="<f8").tobytes()).hexdigest()


def canonicalize(maps):
    rounded = np.asarray(np.round(maps, 8), dtype="<f8", order="C")
    if not np.isfinite(rounded).all():
        raise ValueError("Local learner produced a nonfinite operator")
    rounded[rounded == 0] = 0
    return rounded


def state_id(maps, claims):
    return sha256(maps.tobytes() + np.asarray(claims, dtype="<f8").tobytes()).hexdigest()


class PredictionCache:
    """Count graph-operator evaluations separately from repeated probe draws."""

    def __init__(self, pilot):
        self.pilot = pilot
        self.outputs = {}
        self.seen_graphs = set()
        self.requested = 0
        self.evaluated = 0
        self.calls = 0
        self.seconds = 0.0

    def get(self, operators, ids):
        self.seen_graphs.update(ids.tolist())
        output = []
        for operator in operators:
            cached = self.outputs.setdefault(operator_key(operator), {})
            missing = np.asarray([graph_id for graph_id in ids if graph_id not in cached], dtype=int)
            self.requested += len(ids)
            if missing.size:
                start = perf_counter()
                predicted = predict(self.pilot, operator, missing)
                self.seconds += perf_counter() - start
                cached.update(zip(missing.tolist(), predicted))
                self.evaluated += len(missing)
                self.calls += 1
            output.append(np.stack([cached[graph_id] for graph_id in ids]))
        return np.stack(output)

    def stats(self):
        return {
            "requested_graph_operator_pairs": self.requested,
            "new_graph_operator_pairs": self.evaluated,
            "cache_hits": self.requested - self.evaluated,
            "vectorized_prediction_calls": self.calls,
            "prediction_seconds": self.seconds,
            "unique_graphs_seen": len(self.seen_graphs),
        }


def draw_ids(pilot, seed, round_no, phase, count):
    drawn = rng(seed, round_no, phase).choice(pilot.audit_ids, size=count, replace=True)
    unique, inverse = np.unique(drawn, return_inverse=True)
    return drawn, unique, inverse


def quality(scores, labels):
    assignments = scores.argmax(axis=1)
    classes, encoded = np.unique(labels, return_inverse=True)
    counts = contingency_matrix(encoded, assignments)
    i, j = linear_sum_assignment(counts, maximize=True)
    assigned_ids = np.unique(assignments)
    matched = np.full(scores.shape[1], -1, dtype=int)
    matched[assigned_ids[j]] = i
    return {
        "acc": float(counts[i, j].sum() / labels.size),
        "macro_f1": f1_score(encoded, matched[assignments], labels=np.arange(len(classes)),
                             average="macro", zero_division=0),
        "nmi": normalized_mutual_info_score(labels, assignments),
        "ari": adjusted_rand_score(labels, assignments),
    }


def full_state(state, maps, claims, selected):
    ii = np.asarray(selected, dtype=int)
    combined_maps, combined_claims = state.maps.copy(), state.claims.copy()
    combined_maps[ii], combined_claims[ii] = maps[ii], claims[ii]
    return combined_maps, combined_claims


def counts(state, maps, claims, selected):
    take = np.zeros(len(maps), dtype=bool)
    take[list(selected)] = True
    op = np.any(maps != state.maps, axis=(1, 2)) & take
    claim = (claims != state.claims) & take
    return {
        "operator_only": int(np.sum(op & ~claim)),
        "claim_only": int(np.sum(claim & ~op)),
        "joint": int(np.sum(op & claim)),
    }, op | claim


def audit_pair(result):
    if result is None:
        return None
    return asdict(result)


def commit(state, maps, claims, selected, changed, round_no, expected):
    new_maps, new_claims = full_state(state, maps, claims, selected)
    if state_id(new_maps, new_claims) != expected:
        raise RuntimeError("Staged payload hash changed before commit")
    last_submit, last_content = state.last_submit.copy(), state.last_content.copy()
    last_submit[list(selected)] = round_no
    last_content[changed] = round_no
    return State(new_maps, new_claims, last_submit, last_content)


def trial(pilot, seed, regime, arm, state, round_no, config, design_draws, verify_draws):
    start = perf_counter()
    local_seed = int(rng(seed, round_no, 1).integers(0, 2**32, dtype=np.uint32))
    maps = canonicalize(propose(pilot, state.maps, local_seed))
    if maps.shape != state.maps.shape:
        raise ValueError("Local learner changed the operator interface")
    claims = np.full(len(maps), 0.25, dtype=np.float64)
    repeat = np.flatnonzero((round_no + 2 * np.arange(len(maps))) % 7 == 0)
    maps[repeat], claims[repeat] = state.maps[repeat], state.claims[repeat]
    if regime == "Perturbed" and 9 <= round_no <= 20:
        for identity in STRESS_PEERS:
            maps[identity] = maps[identity, ::-1].copy()
            claims[identity] = 1.0
    training_seconds = perf_counter() - start

    affected = np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
    ages = round_no - state.last_content
    cache = PredictionCache(pilot)
    design_start = perf_counter()
    design_drawn, design_ids, design_idx = draw_ids(pilot, seed, round_no, 2, design_draws)
    cur_design, mem_design = cache.get(maps, design_ids), cache.get(state.maps, design_ids)
    selection_seed = int(rng(seed, round_no, 3).integers(0, 2**32, dtype=np.uint32))
    online = design(
        cur_design, mem_design, claims, state.claims, affected, ages, design_idx, config,
        random_seed=selection_seed, screen_only=arm != "Unscreened",
    )
    design_seconds = perf_counter() - design_start
    design_cache = cache.stats()
    replay_start = perf_counter()
    selected = online if arm == "Unscreened" else design(
        cur_design, mem_design, claims, state.claims, affected, ages, design_idx, config,
        random_seed=selection_seed,
    )
    if online.selections[arm] != selected.selections[arm]:
        primary, replayed = online.selections[arm], selected.selections[arm]
        if primary is None or replayed is None or primary.selected != replayed.selected:
            raise RuntimeError("Online selection disagrees with matched complete replay")
    policies = tuple(selected.selections)
    pending = selected.selections
    staged = {
        name: state_id(*full_state(state, maps, claims, candidate.selected))
        for name, candidate in pending.items() if candidate is not None
    }
    replay_design_seconds = perf_counter() - replay_start
    # The verification RNG is invoked only after every candidate in this replay is fixed.
    if staged:
        verify_drawn, verify_ids, verify_idx = draw_ids(pilot, seed, round_no, 4, verify_draws)
    else:
        verify_drawn, verify_ids, verify_idx = (np.empty(0, dtype=int),) * 3
    memo, records = {}, {}
    next_state = state
    for name in (arm, *[name for name in policies if name != arm]):
        started = perf_counter()
        candidate = pending[name]
        counts_before = cache.stats()
        if candidate is None:
            records[name] = {"status": "no_candidate", "candidate": None,
                             "hold_reason": selected.hold_reasons[name], "closed_loop": name == arm}
            continue
        key = candidate.selected
        if key not in memo:
            candidate_maps, _ = full_state(state, maps, claims, key)
            cur_verify = cache.get(candidate_maps, verify_ids)
            mem_verify = cache.get(state.maps, verify_ids)
            memo[key] = verify(cur_verify, mem_verify, claims, state.claims, candidate, verify_idx, config)
        attempt_counts, actual_changes = counts(state, maps, claims, key)
        result = memo[key]
        if name == arm and result.accepted:
            next_state = commit(state, maps, claims, key, actual_changes, round_no, staged[name])
        counts_after = cache.stats()
        records[name] = {
            "status": "accepted" if result.accepted else "verification_rejected",
            "candidate": candidate,
            "candidate_state_hash": staged[name],
            "verification": audit_pair(result),
            "attempted_content_updates": attempt_counts,
            "accepted_candidate_updates": attempt_counts if result.accepted else dict.fromkeys(attempt_counts, 0),
            "committed_content_updates": (attempt_counts if name == arm and result.accepted
                                          else dict.fromkeys(attempt_counts, 0)),
            "closed_loop": name == arm,
            "verify_and_commit_wall_seconds": perf_counter() - started,
            "verify_cache_delta": {stat: counts_after[stat] - counts_before[stat] for stat in counts_before},
        }
    arm_cache = cache.stats()
    return (
        next_state if records[arm]["status"] == "accepted" else state,
        {
            "seed": seed, "regime": regime, "arm": arm, "round": round_no,
            "stored_state_hash": state_id(state.maps, state.claims),
            "current_state_hash": state_id(maps, claims),
            "local_seed": local_seed,
            "local_lloyd_steps": len(maps),
            "training_seconds": training_seconds,
            "online_design_wall_seconds": design_seconds,
            "online_selection_compositions": len(online.candidates),
            "online_screen_seconds": online.screen_seconds,
            "frozen_replay_design_wall_seconds": replay_design_seconds,
            "design_draws": len(design_drawn), "verify_draws": len(verify_drawn),
            "design_unique_graphs": len(design_ids), "verify_unique_graphs": len(verify_ids),
            "design_cache": design_cache, "all_replay_cache": arm_cache,
            "priority_count_active": selected.count_active, "priority_age_active": selected.age_active,
            "version_stats": selected.version_stats,
            "candidates": selected.candidates, "policies": records,
            "last_content_before": state.last_content, "last_submit_before": state.last_submit,
            "content_changed_in_proposal": affected, "scheduled_repeats": repeat,
        },
        (maps, claims, design_drawn, verify_drawn),
    )


def score_frozen(pilot, archive, config, directory):
    started = perf_counter()
    offline = PredictionCache(pilot)
    test, probes = np.asarray(pilot.test_ids), np.asarray(pilot.audit_ids)
    policy_names = tuple(archive[0][0]["policies"])
    effects_shape = (len(archive), len(policy_names), len(probes))
    drifts, flips = np.zeros(effects_shape), np.zeros(effects_shape, dtype=bool)
    present = np.zeros(effects_shape[:2], dtype=bool)
    actual_outcomes, audit_current, audit_stored = [], [], []
    for r, (rec, old, artifacts) in enumerate(archive):
        maps, claims, _, _ = artifacts
        old_probe = offline.get(old.maps, probes)
        new_probe = offline.get(maps, probes)
        old_test = offline.get(old.maps, test)
        new_test = offline.get(maps, test)
        baseline_p = aggregate(old_probe, old_probe, old.claims, old.claims, (), config)
        baseline_t = aggregate(old_test, old_test, old.claims, old.claims, (), config)
        audit_current.append(new_probe)
        audit_stored.append(old_probe)
        rec["stored_inner_validation"] = quality(baseline_t, pilot.labels[test])
        executed = baseline_t
        for p, name in enumerate(policy_names):
            out = rec["policies"][name]
            if out["candidate"] is None:
                out["post_decision_inner_validation"] = rec["stored_inner_validation"]
                continue
            key = out["candidate"].selected
            candidate_p = aggregate(new_probe, old_probe, claims, old.claims, key, config)
            candidate_t = aggregate(new_test, old_test, claims, old.claims, key, config)
            differences = np.abs(candidate_p - baseline_p).mean(axis=1)
            flip = candidate_p.argmax(axis=1) != baseline_p.argmax(axis=1)
            drifts[r, p], flips[r, p], present[r, p] = differences, flip, True
            n_flips = int(flip.sum())
            out["finite_audit_law"] = {"ell": differences.mean(), "flips": n_flips, "size": len(probes)}
            out["finite_audit_law"]["within_tolerance"] = bool(
                differences.mean() <= config.chi and n_flips <= config.gamma * len(probes)
            )
            out["candidate_inner_validation"] = quality(candidate_t, pilot.labels[test])
            out["post_decision_inner_validation"] = (out["candidate_inner_validation"] if out["status"] == "accepted"
                                                      else rec["stored_inner_validation"])
            if name == rec["arm"] and out["status"] == "accepted":
                executed = candidate_t
        actual_outcomes.append(executed)
    np.savez_compressed(directory / "finite_effects.npz", audit_ids=probes, policy_names=policy_names,
                        rounds=np.arange(1, len(archive) + 1), drifts=drifts, flips=flips, present=present,
                        audit_current=np.stack(audit_current), audit_stored=np.stack(audit_stored))
    np.savez_compressed(directory / "inner_validation_outputs.npz", graph_ids=test,
                        probabilities=np.stack(actual_outcomes))
    return {"wall_seconds": perf_counter() - started, "cache": offline.stats()}


def run(pilot, seed, regime, arm, config, directory, rounds, design_draws, verify_draws):
    directory.mkdir(exist_ok=False)
    start = perf_counter()
    initialization_seed = int(rng(seed, 0, 5).integers(0, 2**32, dtype=np.uint32))
    initial = State(canonicalize(initialize(pilot, initialization_seed)), np.full(8, 0.25),
                    np.zeros(8, int), np.zeros(8, int))
    archive, state = [], initial
    for round_no in range(1, rounds + 1):
        next_state, rec, artifacts = trial(pilot, seed, regime, arm, state, round_no, config,
                                          design_draws, verify_draws)
        rec["executed_state_hash"] = state_id(next_state.maps, next_state.claims)
        rec["last_content_after"], rec["last_submit_after"] = next_state.last_content, next_state.last_submit
        if rec["policies"][arm]["status"] != "accepted":
            assert next_state is state
        archive.append((rec, state, artifacts))
        state = next_state
    offline = score_frozen(pilot, archive, config, directory)
    serialization_start = perf_counter()
    with (directory / "rounds.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for rec, _, _ in archive:
            stream.write(dumps(rec, default=encode, allow_nan=False) + "\n")
    np.savez_compressed(
        directory / "states_and_draws.npz",
        stored_maps=np.stack([prev.maps for _, prev, _ in archive]),
        stored_claims=np.stack([prev.claims for _, prev, _ in archive]),
        current_maps=np.stack([args[0] for _, _, args in archive]),
        current_claims=np.stack([args[1] for _, _, args in archive]),
        design_probe_ids=np.stack([args[2] for _, _, args in archive]),
        verification_probe_ids=np.stack([args[3] if len(args[3]) else np.full(verify_draws, -1, int)
                                         for _, _, args in archive]),
        final_maps=state.maps, final_claims=state.claims,
    )
    accepts = sum(r[0]["policies"][arm]["status"] == "accepted" for r in archive)
    endpoint = archive[-1][0]["policies"][arm]["post_decision_inner_validation"]
    return {"seed": seed, "regime": regime, "arm": arm, "accepted_rounds": accepts, "rounds": len(archive),
            "path": directory.name, "endpoint_inner_validation": endpoint,
            "initialization_seed": initialization_seed, "offline_scoring": offline,
            "serialization_seconds": perf_counter() - serialization_start,
            "total_wall_seconds": perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--rounds", type=int, default=ROUNDS, choices=range(1, ROUNDS + 1))
    parser.add_argument("--design-draws", type=int, default=DESIGN_DRAWS)
    parser.add_argument("--verify-draws", type=int, default=VERIFY_DRAWS)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    if min(args.design_draws, args.verify_draws) < 2:
        parser.error("probe batches must contain at least two draws")
    raw = load_proteins(args.dataset)
    config = DecisionConfig(q=6, epsilon=0.10, claim_cap=1.0)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "manifest.json", {
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest(),
        "seeds": args.seeds, "rounds": args.rounds, "regimes": REGIMES, "arms": ARMS,
        "design_draws": args.design_draws, "verify_draws": args.verify_draws,
        "evidence_scope": "exploratory WL prototype pilot inside fingerprint-grouped outer training",
        "sealed_outer_seed": SEALED_OUTER_SEED,
        "probe_law": "iid uniform with replacement on the reserved finite audit split",
        "perturbation": {"rounds_inclusive": [9, 20], "zero_based_peers": STRESS_PEERS,
                         "operator": "reverse two prototype coordinates after alignment", "claim": 1.0},
        "normal_claim": 0.25, "repeated_proposal": "(round + 2 * zero_based_peer) % 7 == 0 before stress",
        "decision": config,
        "runtime": {"platform": platform(), "python": python_version(), "processor": processor(),
                    "blas_threads": 1, "mode": "single-process CPU ring emulation"},
        "learner": {"clusters_known": 2, "wl_depth": 3, "wl_bins_per_level": 192, "temperature": 0.15,
                    "local_lloyd_steps_per_peer": 1, "step": 0.25, "batch_size_at_most": 32,
                    "ring_neighbor_weight": 0.2, "partition_alpha": 0.5, "min_peer_graphs": 8},
        "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        "source_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("run_pilot.py", "learner.py", "decision.py")},
    })
    summaries = []
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            prep_start = perf_counter()
            pilot = prepare_pilot(raw, seed, sealed_outer_seed=SEALED_OUTER_SEED)
            write_json(args.out / f"split_{seed}.json", {
                "train_ids": pilot.train_ids, "audit_ids": pilot.audit_ids,
                "inner_validation_ids": pilot.test_ids, "peer_train_ids": pilot.peer_train_ids,
                "partition_seed": pilot.partition_seed, "partition_attempts": pilot.partition_attempts,
                "largest_wl_fingerprint_group": pilot.max_fingerprint_group,
                "feature_dimension": pilot.features.shape[1], "preparation_seconds": perf_counter() - prep_start,
            })
            for regime in REGIMES:
                for arm in ARMS:
                    directory = args.out / f"seed_{seed}_{regime.lower()}_{arm.lower()}"
                    summary = run(pilot, seed, regime, arm, config, directory,
                                  args.rounds, args.design_draws, args.verify_draws)
                    summaries.append(summary)
                    print(dumps(summary), flush=True)
    write_json(args.out / "run_index.json", summaries)


if __name__ == "__main__":
    main()
