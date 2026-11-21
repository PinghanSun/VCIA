"""Paired 60-round real-graph study and matched fixed-candidate replay."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import kmeans_plusplus
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DATA = Path("data")
import run as base
from decision import (_Pool, _composition_effect, _distance, _draw_counts, _full_key,
                      _moments, Candidate, empirical_bernstein_upper)
from learner import PilotData, _distances, _partition_peers, _wl_features
from multiclass import ARCHIVE_SHA, data as enzyme_data, representation

PROTOCOL = json.loads((HERE / "evidence_protocol.json").read_text(encoding="utf-8"))
OUT = Path("runs/closed-loop")
FINAL_ROUND = PROTOCOL["rounds"]


def dataset(name):
    path = DATA / f"{name}.zip"
    if name == "ENZYMES":
        return *enzyme_data(path), path, ARCHIVE_SHA
    loader, expected = base.DATASETS[name]
    assert base.digest(path) == expected
    raw = loader(path)
    return raw.graphs, raw.labels, path, expected


def search(args, policy):
    start = perf_counter()
    cur, mem, claims, old_claims, changed, ages, draws, config = args
    ids, weights = _draw_counts(draws, cur.shape[1])
    n = len(draws)
    pool = _Pool(cur, mem, claims, old_claims, ids, config)
    peers = tuple(range(len(cur)))
    setup_end = perf_counter()
    all_current = pool.composed(peers)
    upper, detectable, eligible = [], [], []
    for peer in peers:
        previous = pool.composed(tuple(i for i in peers if i != peer))
        dist = _distance(all_current, previous)
        mean, variance = _moments(dist, weights, n)
        u = (empirical_bernstein_upper(mean, variance, pool.range, n, config.delta_screen / len(peers))
             if policy != "Unscreened" else 0.)
        upper.append(u)
        detectable.append(changed[peer] and mean > config.nu)
        if policy == "Unscreened" or u <= config.kappa:
            eligible.append(peer)
    screen_end = perf_counter()
    baseline = pool.composed(())
    labels = baseline.argmax(axis=-1)
    capped = np.minimum(ages, config.age_cap)
    masks = tuple(combinations(eligible, config.q))
    feasible = []
    for mask in masks:
        distances, flips = _composition_effect(pool, mask, baseline, labels, weights)
        ell = float(np.dot(distances, weights) / n)
        phi = flips / n
        useful = [i for i in mask if detectable[i]]
        if useful and ell <= config.eta * config.chi and phi <= config.eta * config.gamma:
            feasible.append(Candidate(mask, len(useful), float(capped[useful].sum()), sum(upper[i] for i in mask),
                                      phi, ell, True, True, None, None, 0.))
    key = (lambda candidate: (candidate.U, candidate.selected)) if policy == "J-Rank-Feasible" else _full_key
    selected = min(feasible, key=key, default=None)
    end = perf_counter()
    return selected, masks, {c.selected for c in feasible}, (setup_end - start, screen_end - setup_end, end - screen_end)


def align(centers, reference):
    _, order = linear_sum_assignment(_distances(reference, centers))
    return centers[order]


def initial_maps(pilot, seed):
    bank = []
    for client, ids in enumerate(pilot.peer_train_ids):
        local = pilot.features[ids]
        initial, _ = kmeans_plusplus(local, n_clusters=pilot.n_clusters,
                                     random_state=int(base.rng(seed, client, 0).integers(2**32)))
        bank.append(base._lloyd(local, initial, 12, 1.))
    return base.canonicalize(np.stack([align(centers, bank[0]) for centers in bank]))


def proposals(pilot, bank, seed, round_no, topology):
    result = np.empty_like(bank)
    size = len(bank)
    for client, ids in enumerate(pilot.peer_train_ids):
        neighbors = ((client - 1) % size, (client + 1) % size)
        if topology == "Complete":
            neighbors = tuple(i for i in range(size) if i != client)
        start = .8 * bank[client] + .2 * bank[list(neighbors)].mean(axis=0)
        sample = base.rng(seed, round_no, 100 + client).choice(ids, min(32, len(ids)), replace=False)
        result[client] = align(base._lloyd(pilot.features[sample], start, 1, .25), bank[client])
    return base.canonicalize(result)


def mask(choice):
    return None if choice is None else sum(1 << i for i in choice.selected)


def finite_feasible(args, masks, config):
    pool = _Pool(*args, None, config)
    baseline = pool.composed(())
    labels = baseline.argmax(axis=-1)
    weights = np.ones(pool.n_graphs, dtype=np.int64)
    feasible = set()
    for selected in masks:
        distances, flips = _composition_effect(pool, selected, baseline, labels, weights)
        if distances.mean() <= config.chi and flips / pool.n_graphs <= config.gamma:
            feasible.add(selected)
    return feasible


def replay(key, design_args, exact_args, config, seed, full_draws, design_online):
    vcia, kept, _, _ = search(design_args, "VCIA")
    unscreened, exhaustive, design_feasible, _ = search(design_args, "Unscreened")
    native = base.design(*design_args)
    expected = native.selections
    assert mask(vcia) == mask(expected["PCOC"]) == mask(design_online)
    assert mask(unscreened) == mask(expected["Unscreened"])
    assert mask(search(design_args, "J-Rank-Feasible")[0]) == mask(expected["J-Rank-Feasible"])
    assert set(kept) == {candidate.selected for candidate in native.candidates if candidate.screened}
    assert design_feasible == {candidate.selected for candidate in native.candidates
                               if candidate.unscreened_rejection is None}
    exact_feasible = finite_feasible(exact_args, exhaustive, config)
    kept_set = set(kept)
    comparisons = []
    first = int(base.rng(seed, key["round"], 200).integers(2))
    methods = ("VCIA", "Unscreened")
    for rep in range(PROTOCOL["timing_repetitions_per_arm"]):
        for order in range(2):
            policy = methods[(first + rep + order) % 2]
            _, _, _, (setup, screen, compose) = search(design_args, policy)
            comparisons.append({**key, "policy": policy, "repetition": rep + 1, "order": order + 1,
                                "setup_s": setup, "peer_screen_or_detect_s": screen,
                                "composition_s": compose, "total_s": setup + screen + compose})
    audit = []
    if vcia is not None:
        check = base.exact(*exact_args, vcia, config)
        assert check.within_tolerance == (vcia.selected in exact_feasible)
        for budget in PROTOCOL["fixed_candidate_audit_draws"]:
            started = perf_counter()
            result = base.verify(*exact_args, vcia, full_draws[:budget], config)
            audit.append({**key, "mask": mask(vcia), "policy": f"IID {budget}", "draws": budget,
                          "unique_graphs": result.unique_graphs, "passed": int(result.accepted),
                          "finite_feasible": int(check.within_tolerance), "ell": check.ell, "phi": check.phi,
                          "decision_s": perf_counter() - started})
        started = perf_counter()
        census = base.exact(*exact_args, vcia, config)
        elapsed = perf_counter() - started
        for policy, passes, graphs, seconds in (("Finite census", census.within_tolerance, census.size, elapsed),
                                                 ("No-Verify", True, 0, None)):
            audit.append({**key, "mask": mask(vcia), "policy": policy, "draws": 0, "unique_graphs": graphs,
                          "passed": int(passes), "finite_feasible": int(check.within_tolerance),
                          "ell": check.ell, "phi": check.phi, "decision_s": seconds})
    snapshot = {**key, "full_evaluations": len(exhaustive), "screened_evaluations": len(kept),
                "design_feasible": len(design_feasible), "design_feasible_omitted": len(design_feasible - kept_set),
                "exact_feasible": len(exact_feasible), "exact_feasible_omitted": len(exact_feasible - kept_set),
                "both_feasible": len(exact_feasible & design_feasible),
                "both_feasible_omitted": len((exact_feasible & design_feasible) - kept_set),
                "vcia_mask": mask(vcia), "unscreened_mask": mask(unscreened),
                "same_mask": int(mask(vcia) == mask(unscreened))}
    return snapshot, comparisons, audit


def round_step(pilot, state, key, seed, config, sockets):
    round_no, topology, condition, policy = (key[k] for k in ("round", "topology", "condition", "arm"))
    peers = len(state.maps)
    start = perf_counter()
    maps = proposals(pilot, state.maps, seed, round_no, topology)
    claims = np.full(peers, .25)
    repeat = np.flatnonzero((round_no + 2 * np.arange(peers)) % 7 == 0)
    maps[repeat], claims[repeat] = state.maps[repeat], state.claims[repeat]
    n_stressed = 0
    if 11 <= round_no <= 35 and condition != "Clean":
        identities = (1, 1 + peers // 2) if condition == "Moderate" else tuple(range(1, peers, 2))
        for peer in identities:
            maps[peer] = np.roll(maps[peer], 1, axis=0)
            claims[peer] = 1.
        n_stressed = len(identities)
    proposal_s = perf_counter() - start
    start = perf_counter()
    up = 0
    for i, proto in enumerate(maps):
        frame = bytes((round_no, i)) + np.r_[proto.ravel(), claims[i]].astype("<f8").tobytes()
        received = base.packet(*sockets, frame)
        assert received == frame
        values = np.frombuffer(received, offset=2, dtype="<f8")
        maps[i], claims[i] = values[:-1].reshape(proto.shape), values[-1]
        up += len(received)
    tcp_s = perf_counter() - start
    online_start = perf_counter()
    cache = base.PredictionCache(pilot)
    _, ids, drawn = base.draw_ids(pilot, seed, round_no, 2, PROTOCOL["design_draws"])
    changed = np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
    args = (cache.get(maps, ids), cache.get(state.maps, ids), claims, state.claims, changed,
            round_no - state.last_content, drawn, config)
    prep_s = perf_counter() - online_start
    start = perf_counter()
    choice, selected_masks, _, _ = search(args, "VCIA" if policy == "No-Verify" else policy)
    search_s = perf_counter() - start
    new_state, staged = state, None
    status = "no_candidate"
    exact = None
    n_updates, verify_s, state_commit_s = 0, 0., 0.
    if choice is not None:
        start = perf_counter()
        candidate, candidate_claims = base.full_state(state, maps, claims, choice.selected)
        staged = base.state_id(candidate, candidate_claims)
        state_commit_s = perf_counter() - start
        accepted = True
        if policy != "No-Verify":
            start = perf_counter()
            _, ids, index = base.draw_ids(pilot, seed, round_no, 4, PROTOCOL["verification_draws"])
            result = base.verify(cache.get(maps, ids), cache.get(state.maps, ids), claims, state.claims,
                                 choice, index, config)
            accepted = result.accepted
            verify_s = perf_counter() - start
        start = perf_counter()
        status = "committed" if accepted else "rejected"
        if accepted:
            updates, changed_for_commit = base.counts(state, maps, claims, choice.selected)
            n_updates = updates["operator_only"] + updates["joint"]
            new_state = base.commit(state, maps, claims, choice.selected, changed_for_commit, round_no, staged)
        state_commit_s += perf_counter() - start
    start = perf_counter()
    state_hash = base.state_id(new_state.maps, new_state.claims)
    reply = bytes((round_no, {"no_candidate": 0, "rejected": 1, "committed": 2}[status]))
    reply += (mask(choice) or 0).to_bytes(2, "little") + bytes.fromhex(state_hash)
    down = 0
    for peer in range(peers):
        frame = bytes((peer,)) + reply
        assert base.packet(*sockets[::-1], frame) == frame
        down += len(frame)
    tcp_s += perf_counter() - start
    eval_args = (base.all_peer_predictions(pilot, maps, pilot.audit_ids),
                 base.all_peer_predictions(pilot, state.maps, pilot.audit_ids), claims, state.claims)
    if choice is not None:
        exact = base.exact(*eval_args, choice, config)
    snapshots = None
    if policy == "VCIA" and round_no in PROTOCOL["replay_rounds"]:
        full_draws = base.rng(seed, round_no, 4).integers(len(pilot.audit_ids), size=max(PROTOCOL["fixed_candidate_audit_draws"]))
        realized, _, _ = base.draw_ids(pilot, seed, round_no, 4, PROTOCOL["verification_draws"])
        assert np.array_equal(realized, pilot.audit_ids[full_draws[:len(realized)]])
        snapshots = replay(key, args, eval_args, config, seed, full_draws, choice)
    outer = base.all_peer_predictions(pilot, new_state.maps, pilot.test_ids)
    probabilities = base.aggregate(outer, outer, new_state.claims, new_state.claims, (), config)
    row = {**key, "perturbed_proposals": n_stressed, "status": status, "mask": mask(choice),
           "finite_feasible": None if exact is None else int(exact.within_tolerance),
           "exact_ell": None if exact is None else exact.ell, "exact_phi": None if exact is None else exact.phi,
           "operator_updates": n_updates, "composition_evaluations": len(selected_masks),
           "online_graph_operator_evals": cache.evaluated, "proposal_s": proposal_s, "probe_prep_s": prep_s,
           "search_s": search_s, "verify_s": verify_s, "stage_and_commit_s": state_commit_s,
           "loopback_s": tcp_s, "online_s": proposal_s + prep_s + search_s + verify_s + state_commit_s + tcp_s,
           "uplink_app_bytes": up, "downlink_app_bytes": down, "candidate_sha": staged, "committed_sha": state_hash}
    return new_state, probabilities, row, snapshots


def execute(root=OUT, smoke=False):
    root.mkdir(parents=True, exist_ok=False)
    source = [HERE / "evidence_protocol.json", Path(__file__), Path(base.__file__), HERE / "multiclass.py",
              HERE / "snapshots.py", HERE / "decisions.py"]
    source.extend(base.PILOT / name for name in ("learner.py", "decision.py", "run_pilot.py", "run_imdb_pilot.py"))
    source.append(HERE / "requirements.txt")
    from importlib.metadata import version
    import subprocess
    try:
        cpu = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
                                       "(Get-CimInstance Win32_Processor).Name"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    info = {"start_utc": base.now(), "smoke": smoke, "os": platform.platform(), "cpu": cpu,
            "python": platform.python_version(), "BLAS_threads": 1,
            "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
            "file_sha256": {p.name: base.digest(p) for p in source}}
    names = PROTOCOL["datasets"][:1] if smoke else PROTOCOL["datasets"]
    for name in names:
        _, _, path, digest = dataset(name)
        info["file_sha256"][path.name] = digest
    base.write_json(root / "execution.json", info)
    index, splits = [], []
    sockets = base.connection()
    logs = {name: base.Rows(root / f"{name}.csv") for name in ("rounds", "snapshots", "replay_timing", "fixed_audit")}
    banks = {}
    try:
        for name in names:
            graphs, labels, _, _ = dataset(name)
            wl, fingerprints = _wl_features(graphs, 2, 512)
            for outer in PROTOCOL["outer_seeds"][:1] if smoke else PROTOCOL["outer_seeds"]:
                train, audit, test = base._split_roles(fingerprints, outer)
                roles = [set(fingerprints[i] for i in ids) for ids in (train, audit, test)]
                assert all(not a & b for a, b in combinations(roles, 2))
                features = representation(wl, None, train, "WL-TFIDF")
                for repeat in range(1, (1 if smoke else PROTOCOL["paired_repetitions_per_split"]) + 1):
                    partition = outer * 10 + repeat
                    seed = 500000 + partition
                    for peers, q in PROTOCOL["peer_quota_pairs"][1:] if smoke else PROTOCOL["peer_quota_pairs"]:
                        peer_train, n_attempt = _partition_peers(train, labels, partition, peers, .5, 10)
                        assert np.array_equal(np.sort(np.concatenate(peer_train)), train)
                        k = 6 if name == "ENZYMES" else 2
                        pilot = PilotData(features, np.full(len(labels), -1), train, audit, test, peer_train,
                                          fingerprints, outer, partition, n_attempt,
                                          max(Counter(fingerprints).values()), k, 2, 512, .25)
                        config = base.DecisionConfig.normalized(q, n_clusters=k, epsilon=.1)
                        initial = initial_maps(pilot, seed)
                        splits.append({"dataset": name, "outer_seed": outer, "rep": repeat,
                                       "partition_seed": partition, "init_and_round_seed": seed, "peers": peers,
                                       "partition_attempts": n_attempt, "train": train, "audit": audit, "test": test,
                                       "peer_train": peer_train, "decision": asdict(config),
                                       "peer_train_class_counts": [dict(Counter(map(int, labels[ids]))) for ids in peer_train]})
                        for topology, condition, policy in product(PROTOCOL["topologies"], PROTOCOL["conditions"], PROTOCOL["arms"]):
                            base_key = dict(dataset=name, outer_seed=outer, rep=repeat, peers=peers, q=q,
                                            topology=topology, condition=condition, arm=policy)
                            state = base.State(initial.copy(), np.full(peers, .25), np.zeros(peers, dtype=int),
                                               np.zeros(peers, dtype=int))
                            results = []
                            for r in range(1, (12 if smoke else FINAL_ROUND) + 1):
                                state, prob, row, details = round_step(
                                    pilot, state, {**base_key, "round": r}, seed, config, sockets)
                                results.append(prob)
                                logs["rounds"].add(row)
                                if details:
                                    snapshot, timing, audit_rows = details
                                    logs["snapshots"].add(snapshot)
                                    for item in timing:
                                        logs["replay_timing"].add(item)
                                    for item in audit_rows:
                                        logs["fixed_audit"].add(item)
                            token = f"p{len(index):04d}"
                            index.append({**base_key, "token": token, "test_ids": test})
                            banks[token] = np.stack(results)
                    print(f"Completed {name}, split {outer}, repetition {repeat}, {len(index)} trajectories", flush=True)
    finally:
        for sock in sockets:
            sock.close()
        for log in logs.values():
            log.close()
    np.savez_compressed(root / "sealed_outer_probabilities.npz", **banks)
    base.write_json(root / "index.json", index)
    base.write_json(root / "splits.json", splits)
    base.write_json(root / "seal.json", {"sealed_at_utc": base.now(),
                                        "sha256": {p.name: base.digest(p) for p in root.iterdir() if p.is_file()}})


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--data", type=Path, default=DATA)
    cli.add_argument("--output", type=Path)
    options = cli.parse_args()
    DATA = options.data
    output = options.output or (Path("runs/closed-loop-smoke") if options.smoke else OUT)
    with threadpool_limits(limits=1):
        execute(output, options.smoke)
