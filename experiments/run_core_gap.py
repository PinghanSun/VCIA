"""Reproduce the paper pilot timing and extend only sign-discordant ranking cells."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from csv import DictReader
from hashlib import sha256
from inspect import getsource
from itertools import combinations, product
import json
from pathlib import Path
import platform
from statistics import mean

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path("runs/core-checks")
EVIDENCE = Path("runs/closed-loop")
PILOT_RUNS = Path("runs/pilots")
PROTOCOL = Path(__file__).with_name("evidence_protocol.json")
import run_evidence as original

base = original.base
import run_pilot as pilot_code

ARMS = ("VCIA", "J-Rank-Feasible")
FIELDS = ("dataset", "peers", "q", "topology", "condition")
METRICS = ("delta_nmi_pp", "delta_ari_pp")
OLD_SEEDS = (31501, 31502, 31503)
NEW_SEEDS = (31504, 31505, 31506)
EPS = 1e-9


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        yield from DictReader(file)


def change_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected frozen source anchor: {old}")
    return text.replace(old, new, 1)


def instrument_pilot():
    body = getsource(pilot_code.trial)
    replay_begin = body.index("    replay_start = perf_counter()")
    replay_end = body.index("    pending = selected.selections", replay_begin)
    body = body[:replay_begin] + "    selected = online\n    policies = (arm,)\n" + body[replay_end:]
    body = change_once(body, "    replay_design_seconds = perf_counter() - replay_start\n", "")
    body = change_once(body, "    arm_cache = cache.stats()", "    end_to_end_s = perf_counter() - start\n    arm_cache = cache.stats()")
    body = change_once(body, '"frozen_replay_design_wall_seconds": replay_design_seconds,',
                       '"end_to_end_s": end_to_end_s,')
    scope = vars(pilot_code).copy()
    exec(compile(body, "<frozen pilot with offline replay omitted>", "exec"), scope)
    return scope["trial"], sha256(body.encode()).hexdigest()


def pilot_initial(pilot, seed):
    init = int(pilot_code.rng(seed, 0, 5).integers(0, 2**32, dtype=np.uint32))
    return base.State(base.canonicalize(base.initialize(pilot, init)), np.full(8, .25),
                      np.zeros(8, dtype=int), np.zeros(8, dtype=int))


def read_archive(archive, seed, regime, arm):
    path = archive / f"seed_{seed}_{regime.lower()}_{arm.lower()}" / "rounds.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def timing():
    trial, code_sha = instrument_pilot()
    log = base.Rows(HERE / "timing_rounds.csv")
    manifests = {}
    try:
        for name, archive_dir in (("PROTEINS", "results_proteins_inner_v1"), ("IMDB-BINARY", "results_imdb_binary_inner_v1")):
            archive = PILOT_RUNS / archive_dir
            manifest = load(archive / "manifest.json")
            manifests[name] = base.digest(archive / "manifest.json")
            for file, sha in manifest["source_sha256"].items():
                assert base.digest(base.PILOT / file) == sha, file
            loader, source_sha = base.DATASETS[name]
            path = original.DATA / f"{name}.zip"
            assert base.digest(path) == source_sha == manifest["dataset_sha256"]
            raw = loader(path)
            config = base.DecisionConfig(**manifest["decision"])
            budgets = (manifest["design_draws"], manifest["verify_draws"])
            arms = tuple(manifest["arms"])
            for seed, regime in product(manifest["seeds"], manifest["regimes"]):
                pilot = base.prepare_pilot(raw, seed, sealed_outer_seed=manifest["sealed_outer_seed"])
                frozen_split = load(archive / f"split_{seed}.json")
                for role, actual in (("train_ids", pilot.train_ids), ("audit_ids", pilot.audit_ids),
                                     ("inner_validation_ids", pilot.test_ids)):
                    np.testing.assert_array_equal(actual, frozen_split[role])
                for actual, expected in zip(pilot.peer_train_ids, frozen_split["peer_train_ids"], strict=True):
                    np.testing.assert_array_equal(actual, expected)
                archived = {arm: read_archive(archive, seed, regime, arm) for arm in arms}
                initial = pilot_initial(pilot, seed)
                for arm in arms:
                    trial(pilot, seed, regime, arm, initial, 1, config, *budgets)
                for repeat in (1, 2):
                    states = dict.fromkeys(arms, initial)
                    for rnd in range(1, manifest["rounds"] + 1):
                        order = arms if (rnd + repeat) % 2 == 0 else arms[::-1]
                        pair = []
                        for position, arm in enumerate(order, 1):
                            state, row, _ = trial(pilot, seed, regime, arm, states[arm], rnd, config, *budgets)
                            states[arm] = state
                            old = archived[arm][rnd - 1]
                            current = row["policies"][arm]
                            expected = old["policies"][arm]
                            replay_hash = base.state_id(state.maps, state.claims)
                            assert replay_hash == old["executed_state_hash"]
                            assert current["status"] == expected["status"]
                            assert current.get("candidate_state_hash") == expected.get("candidate_state_hash")
                            for key in ("round", "stored_state_hash", "current_state_hash", "online_selection_compositions", "design_draws", "verify_draws"):
                                assert row[key] == old[key], (name, seed, regime, arm, rnd, key)
                            cache_key = "new_graph_operator_pairs"
                            misses = row["design_cache"][cache_key] + current.get("verify_cache_delta", {}).get(cache_key, 0)
                            expected_misses = old["design_cache"][cache_key] + expected.get("verify_cache_delta", {}).get(cache_key, 0)
                            assert misses == expected_misses
                            pair.append((misses, replay_hash))
                            log.add(dict(dataset=name, seed=seed, condition=regime, repeat=repeat, round=rnd,
                                         order=position, arm="VCIA" if arm == "PCOC" else arm,
                                         end_to_end_ms=1000 * row["end_to_end_s"],
                                         train_ms=1000 * row["training_seconds"],
                                         design_ms=1000 * row["online_design_wall_seconds"],
                                         verify_commit_ms=None if "verify_and_commit_wall_seconds" not in current else 1000 * current["verify_and_commit_wall_seconds"],
                                         composition_evaluations=row["online_selection_compositions"],
                                         unique_graph_operator_misses=misses, design_draws=row["design_draws"],
                                         verification_draws=row["verify_draws"], status=current["status"], executed_state_sha256=replay_hash))
                        assert len(set(pair)) == 1, (name, seed, regime, rnd)
                print(f"Timing verified: {name} {seed} {regime}", flush=True)
    finally:
        log.close()
    base.write_json(HERE / "timing_seal.json", {
        "record_sha256": base.digest(HERE / "timing_rounds.csv"), "frozen_manifests": manifests,
        "instrumented_trial_sha256": code_sha,
        "boundary": "Each round from before proposal to completion of the executed arm verification/commit; original single-process pilot, no network path. Offline comparisons and scoring excluded.",
        "cache": "New round-local cache per arm, reused from design to verification, with identical cache misses and executed state for every timed pair.",
        "warmup": "One untimed round per arm on the initial state; state reset; two repetitions with opposite order per round.",
        "environment": platform.platform(), "cpu": platform.processor(), "python": platform.python_version(),
        "threadpools": threadpool_info(), "completed": base.now(),
    })


def make_plan():
    path = EVIDENCE / "quality_pairs.csv"
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows(path):
        if row["reference"] in ("Unscreened", "J-Rank-Feasible"):
            key = tuple(int(row[k]) if k in ("peers", "q") else row[k] for k in FIELDS)
            groups[(key, row["reference"])][int(row["outer_seed"])].append(row)
    current, eligible = [], []
    for (key, ref), splits in sorted(groups.items()):
        assert tuple(sorted(splits)) == OLD_SEEDS and all(len(rep) == 2 for rep in splits.values())
        data = [{"outer_seed": seed, **{m: mean(float(row[m]) for row in splits[seed]) for m in METRICS}}
                for seed in OLD_SEEDS]
        flips = [m for m in METRICS if min(r[m] for r in data) < -EPS and max(r[m] for r in data) > EPS]
        info = dict(zip(FIELDS, key, strict=True))
        current.append({**info, "reference": ref, "flips": flips, "split_means": data})
        if ref == "J-Rank-Feasible" and flips:
            eligible.append(info)
        if ref == "Unscreened":
            assert all(all(row[m] == "0.0" for m in METRICS) and row["same_final_state"] == "1"
                       for rep in splits.values() for row in rep)
    return {"prior_pairs_sha256": base.digest(path), "prior_source_sha256": base.digest(Path(original.__file__)),
            "extension_source_sha256": base.digest(Path(__file__)), "prior_protocol_sha256": base.digest(PROTOCOL),
            "old_seeds": OLD_SEEDS, "new_seeds": NEW_SEEDS, "arms": ARMS, "rounds": original.FINAL_ROUND,
            "selection": "After averaging nested reps, include a cell if NMI or ARI has both positive and negative outer-split signs; a tie is within 1e-9 percentage points. No new design settings.",
            "eligible": eligible, "original_directions": current}


def online_only():
    code = getsource(original.round_step)
    body = code[:code.index("    eval_args = (")]
    body += '    return new_state, {"status": status, "mask": mask(choice), "committed_sha": state_hash}\n'
    scope = vars(original).copy()
    exec(compile(body, "<frozen evidence round through decision acknowledgement>", "exec"), scope)
    return scope["round_step"], sha256(body.encode()).hexdigest()


def states(pilot, seed):
    maps = original.initial_maps(pilot, seed)
    return base.State(maps, np.full(len(maps), .25), np.zeros(len(maps), dtype=int), np.zeros(len(maps), dtype=int))


def extend(outer):
    plan = load(HERE / "quality_plan.json")
    assert plan["extension_source_sha256"] == base.digest(Path(__file__))
    assert plan["prior_source_sha256"] == base.digest(Path(original.__file__))
    assert plan["prior_protocol_sha256"] == base.digest(PROTOCOL)
    assert outer in plan["new_seeds"]
    step, code_sha = online_only()
    path = HERE / f"split_{outer}"
    path.mkdir(exist_ok=False)
    log = base.Rows(path / "rounds.csv")
    sockets = base.connection()
    partitions, index, probabilities = [], [], {}
    try:
        for name in sorted({c["dataset"] for c in plan["eligible"]}):
            graphs, labels, _, _ = original.dataset(name)
            wl, fp = original._wl_features(graphs, 2, 512)
            train, audit, test = base._split_roles(fp, outer)
            fingerprint_roles = [{fp[i] for i in group} for group in (train, audit, test)]
            assert all(not (a & b) for a, b in combinations(fingerprint_roles, 2))
            features = original.representation(wl, None, train, "WL-TFIDF")
            cells = [c for c in plan["eligible"] if c["dataset"] == name]
            for peers, q in sorted({(c["peers"], c["q"]) for c in cells}):
                for rep in (1, 2):
                    partition = outer * 10 + rep
                    seed = 500000 + partition
                    peer_train, tries = original._partition_peers(train, labels, partition, peers, .5, 10)
                    np.testing.assert_array_equal(np.sort(np.concatenate(peer_train)), train)
                    k = 6 if name == "ENZYMES" else 2
                    pilot = original.PilotData(features, np.full(len(labels), -1), train, audit, test, peer_train,
                                               fp, outer, partition, tries, max(Counter(fp).values()), k, 2, 512, .25)
                    config = base.DecisionConfig.normalized(q, n_clusters=k, epsilon=.1)
                    initial = states(pilot, seed)
                    partitions.append(dict(dataset=name, outer_seed=outer, peers=peers, q=q, rep=rep,
                                           partition_seed=partition, init_and_round_seed=seed, n_attempts=tries,
                                           train=train, audit=audit, test=test, peer_train=peer_train,
                                           initial_sha256=base.state_id(initial.maps, initial.claims)))
                    for cell, arm in product((c for c in cells if c["peers"] == peers), plan["arms"]):
                        state = initial
                        key = {**cell, "outer_seed": outer, "rep": rep, "arm": arm}
                        for rnd in range(1, plan["rounds"] + 1):
                            round_key = {**key, "round": rnd}
                            state, result = step(pilot, state, round_key, seed, config, sockets)
                            log.add({**round_key, **result})
                        pred = base.all_peer_predictions(pilot, state.maps, test)
                        token = f"p{len(index):04d}"
                        index.append({**key, "token": token, "test_ids": test, "final_sha256": result["committed_sha"]})
                        probabilities[token] = base.aggregate(pred, pred, state.claims, state.claims, (), config)
                    print(f"Outer split {outer}: {name}, {peers}/{q}, repetition {rep}", flush=True)
    finally:
        log.close()
        for socket in sockets:
            socket.close()
    np.savez_compressed(path / "sealed_predictions.npz", **probabilities)
    base.write_json(path / "partitions.json", partitions)
    base.write_json(path / "index.json", index)
    assert len(index) == len(plan["eligible"]) * 4
    base.write_json(path / "seal.json", {"sha256": {f.name: base.digest(f) for f in path.iterdir()},
                                        "online_step_sha256": code_sha, "sealed_before_scoring": base.now()})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("timing", "plan", "extend"))
    parser.add_argument("--outer", type=int, choices=NEW_SEEDS)
    parser.add_argument("--data", type=Path, default=original.DATA)
    parser.add_argument("--closed-loop", type=Path, default=EVIDENCE)
    parser.add_argument("--pilot-runs", type=Path, default=PILOT_RUNS)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    if args.action == "extend" and args.outer is None:
        parser.error("extend requires --outer")
    original.DATA, EVIDENCE, PILOT_RUNS, HERE = args.data, args.closed_loop, args.pilot_runs, args.output
    HERE.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.action == "timing":
            timing()
        elif args.action == "plan":
            base.write_json(HERE / "quality_plan.json", make_plan())
        else:
            extend(args.outer)
