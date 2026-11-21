"""Frozen, paired graph clustering experiments and independent outer scoring."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import socket
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "reference"
sys.path.insert(0, str(PILOT))

from decision import DecisionConfig, aggregate, design, exact, verify
from learner import (
    _align, _initial_centers, _lloyd, _split_roles, all_peer_predictions,
    initialize, load_proteins, prepare_pilot, propose,
)
from run_imdb_pilot import load_imdb
from run_pilot import (
    PredictionCache, State, canonicalize, commit, counts, draw_ids, full_state,
    quality, rng, state_id, write_json,
)

DATASETS = {
    "PROTEINS": (load_proteins, "2da8de15284b88edabca2888ce5444d62f364ed41159260977088c4e53d4d848"),
    "IMDB-BINARY": (load_imdb, "b291ec8b26d85c70faa2ba0a2433e1f407ed2ef5d0fc072d36b9a95e49a1bb27"),
}
PARTITIONS = (27101, 27102, 27103)
INITS = (27201, 27202, 27203, 27204, 27205)
OUTER_SEED, INNER_SEED = 27001, 27100
ROUNDS, DESIGN_DRAWS, VERIFY_DRAWS = 30, 512, 8192
REGIMES = ("Clean", "Perturbed")
POLICIES = {
    "VCIA": "PCOC", "J-Rank-Feasible": "J-Rank-Feasible", "Unscreened": "Unscreened",
    "Census-Audit": "PCOC", "No-Age": "No-Age", "No-Count": "No-Count", "No-Verify": "PCOC",
    "Frozen": None,
}
DIAGNOSTICS = ("Local", "Centralized")
CONFIG = DecisionConfig.normalized(6, n_clusters=2, epsilon=0.10)
IDENTITY = ("dataset", "partition", "init", "regime", "arm")


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {path.name: digest(path) for path in [
        Path(__file__), *(PILOT / name for name in
        ("decision.py", "learner.py", "run_pilot.py", "run_imdb_pilot.py"))]}


def manifest(data):
    data_hashes = {name: digest(data / f"{name}.zip") for name in DATASETS}
    if any(data_hashes[name] != expected for name, (_, expected) in DATASETS.items()):
        raise ValueError(f"Unexpected TU archive content: {data_hashes}")
    return {
        "frozen_at_utc": now(), "data_sha256": data_hashes, "source_sha256": source_hashes(),
        "dataset_source": "TUDataset graph classification archive; pinned exploratory-pilot archives",
        "partitions": PARTITIONS, "initializations": INITS, "regimes": REGIMES,
        "policies": POLICIES, "diagnostics": DIAGNOSTICS, "rounds": ROUNDS,
        "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED,
        "split": "WL fingerprint grouped; outer 60/20/20 development/reserve/test; development 60/20/20 train/audit/validation",
        "learner": {"peers": 8, "alpha": .5, "min_peer_size": 8, "wl_depth": 3,
                    "wl_bins_per_depth": 192, "temperature": .15, "initial_lloyd_steps": 12,
                    "local_epochs_per_round": 1, "maximum_minibatch": 32, "step": .25,
                    "ring_neighbor_weight": .2},
        "local": "independent minibatch fits with no neighbor mixing; equal-weight aligned prediction diagnostic",
        "centralized": "same features and Lloyd updates in a pooled bank; same train-example budget per round",
        "perturbation": "zero-based peers 1 and 5 swap prototype coordinates and claim 1 during rounds 9–20; clean claim .25",
        "repeats": "unchanged proposal when (round+2*peer)%7==0, followed by registered perturbation",
        "design_draws": DESIGN_DRAWS, "verify_draws_per_sampled_attempt": VERIFY_DRAWS,
        "audit": "one candidate; uniform sampling with replacement; separate RNG for design and verification",
        "screen_families": "rank ablations and J-Rank use same screened family as VCIA; Unscreened omits candidate J eligibility",
        "verification_controls": "No-Verify skips independent verification; Census-Audit uses the complete finite audit pool",
        "enumeration": "all 28 masks per VCIA snapshot, no snapshot selection; exact finite audit tolerances and H>0",
        "round_cost": "wall-clock CPU learner + loopback application message transfers + design + verification/commit; no offline replay",
        "communication": "localhost TCP transmitted proposal and decision frames, application bytes; no remote-link/GPU measurements",
        "outer_evaluation": "round-30 frozen bank; NMI arithmetic and ARI primary; Hungarian ACC and macro-F1 descriptive",
        "primary_comparisons": ["VCIA minus J-Rank-Feasible", "VCIA minus Unscreened"],
        "useful_update": "committed canonical operator changes are reported independently of task quality",
        "environment": {"os": platform.platform(), "cpu": platform.processor(),
                        "python": platform.python_version(), "numpy": version("numpy"),
                        "scipy": version("scipy"), "sklearn": version("scikit-learn"),
                        "blas_threads": 1},
    }


class Rows:
    def __init__(self, path):
        self.stream = path.open("w", newline="", encoding="utf-8-sig")
        self.writer = None

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.stream.close()


def packet(sender, receiver, payload):
    parts = []
    for offset in range(0, len(payload), 4096):
        chunk = payload[offset:offset + 4096]
        sender.sendall(chunk)
        remaining = len(chunk)
        while remaining:
            part = receiver.recv(remaining)
            if not part:
                raise ConnectionError("Loopback connection closed during frame")
            parts.append(part)
            remaining -= len(part)
    return b"".join(parts)


def connection():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client = socket.create_connection(listener.getsockname(), timeout=15)
        server, _ = listener.accept()
        server.settimeout(15)
    for conn in (client, server):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return client, server


def gauge_round(pilot, state, key, seed, round_no, peer, coordinator):
    arm = key["arm"]
    before = state_id(state.maps, state.claims)
    start = perf_counter()
    local_seed = int(rng(seed, round_no, 1).integers(0, 2**32, dtype=np.uint32))
    maps = state.maps.copy() if arm == "Frozen" else canonicalize(propose(pilot, state.maps, local_seed))
    claims = np.full(len(maps), .25)
    repeat = np.flatnonzero((round_no + 2 * np.arange(len(maps))) % 7 == 0)
    maps[repeat], claims[repeat] = state.maps[repeat], state.claims[repeat]
    stressed = key["regime"] == "Perturbed" and 9 <= round_no <= 20
    if stressed:
        for identity in (1, 5):
            maps[identity] = maps[identity, ::-1].copy()
            claims[identity] = 1.0
    train_seconds = perf_counter() - start if arm != "Frozen" else 0.
    transfer_start = perf_counter()
    uplink = 0
    if arm != "Frozen":
        for identity, prototype in enumerate(maps):
            frame = bytes((round_no, identity)) + np.r_[prototype.ravel(), claims[identity]].astype("<f8").tobytes()
            received = packet(peer, coordinator, frame)
            if received[:2] != frame[:2]:
                raise ValueError("Proposal frame addressed incorrectly")
            values = np.frombuffer(received, offset=2, dtype="<f8")
            maps[identity] = values[:-1].reshape(prototype.shape)
            claims[identity] = values[-1]
            uplink += len(received)
    transfer_seconds = perf_counter() - transfer_start if uplink else 0.
    design_seconds = audit_seconds = 0.
    cache = PredictionCache(pilot)
    selection = result = exact_attempt = staged_hash = design_inputs = None
    design_size = tries = verify_size = census_size = n_feasible = 0
    status = reason = "frozen"
    updates = dict.fromkeys(("operator_only", "claim_only", "joint"), 0)
    next_state = state
    if arm != "Frozen":
        started = perf_counter()
        changed = np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
        _, ids, idx = draw_ids(pilot, seed, round_no, 2, DESIGN_DRAWS)
        design_inputs = (cache.get(maps, ids), cache.get(state.maps, ids), claims,
                         state.claims, changed, round_no - state.last_content, idx, CONFIG)
        outcome = design(*design_inputs, screen_only=arm != "Unscreened")
        choice = POLICIES[arm]
        selection = outcome.selections[choice]
        design_size = len(outcome.candidates)
        n_feasible = sum(c.rejection is None if arm != "Unscreened"
                         else c.unscreened_rejection is None for c in outcome.candidates)
        status, reason = "no_candidate", outcome.hold_reasons[choice]
        design_seconds = perf_counter() - started
        if selection is not None:
            started = perf_counter()
            tries = 1
            candidate_maps, candidate_claims = full_state(state, maps, claims, selection.selected)
            staged_hash = state_id(candidate_maps, candidate_claims)
            if arm == "Census-Audit":
                ids = pilot.audit_ids
                census_size = len(ids)
                result = exact(cache.get(candidate_maps, ids), cache.get(state.maps, ids),
                               claims, state.claims, selection, CONFIG)
                passed = result.within_tolerance
            elif arm == "No-Verify":
                passed = True
            else:
                _, ids, idx = draw_ids(pilot, seed, round_no, 4, VERIFY_DRAWS)
                verify_size = len(idx)
                result = verify(cache.get(candidate_maps, ids), cache.get(state.maps, ids),
                                claims, state.claims, selection, idx, CONFIG)
                passed = result.accepted
            status, reason = ("committed", "") if passed else ("rejected", "audit_threshold")
            if passed:
                updates, changed = counts(state, maps, claims, selection.selected)
                next_state = commit(state, maps, claims, selection.selected, changed, round_no, staged_hash)
            audit_seconds = perf_counter() - started
    after = state_id(next_state.maps, next_state.claims)
    downlink = 0
    if uplink:
        transfer_start = perf_counter()
        mask = sum(1 << i for i in selection.selected) if selection is not None else 0
        status_code = {"no_candidate": 0, "rejected": 1, "committed": 2}[status]
        for identity in range(len(maps)):
            frame = bytes((round_no, identity, status_code, mask)) + bytes.fromhex(after)
            if packet(coordinator, peer, frame) != frame:
                raise ValueError("Decision frame mismatch")
            downlink += len(frame)
        transfer_seconds += perf_counter() - transfer_start
    exact_inputs = None
    if tries or arm == "VCIA":
        offline = PredictionCache(pilot)
        ids = pilot.audit_ids
        exact_inputs = (offline.get(maps, ids), offline.get(state.maps, ids), claims, state.claims)
        if tries:
            exact_attempt = exact(*exact_inputs, selection, CONFIG)
    round_row = {
        **key, "round": round_no, "perturbed": int(stressed), "status": status,
        "hold_reason": reason, "candidate_mask": mask_text(selection), "attempted": tries,
        "committed": int(status == "committed"), "rejected": int(status == "rejected"),
        "no_candidate": int(status == "no_candidate"), "operator_updates": updates["operator_only"] + updates["joint"],
        "content_updates": sum(updates.values()), **updates,
        "composition_evaluations": design_size, "design_feasible_compositions": n_feasible,
        "design_draws": DESIGN_DRAWS if uplink else 0, "verify_draws": verify_size,
        "census_graphs": census_size, "graph_operator_evaluations": cache.evaluated,
        "audit_threshold_within": int(exact_attempt.within_tolerance) if exact_attempt else None,
        "exact_ell": exact_attempt.ell if exact_attempt else None,
        "exact_phi": exact_attempt.phi if exact_attempt else None,
        "audit_ell": result.ell if result else None, "audit_phi": result.phi if result else None,
        "audit_upper_ell": getattr(result, "upper_ell", None),
        "audit_upper_phi": getattr(result, "upper_phi", None),
        "train_seconds": train_seconds, "design_seconds": design_seconds,
        "audit_commit_seconds": audit_seconds, "loopback_seconds": transfer_seconds,
        "online_seconds": train_seconds + transfer_seconds + design_seconds + audit_seconds,
        "uplink_app_bytes": uplink, "downlink_app_bytes": downlink,
        "stored_hash": before, "candidate_hash": staged_hash, "committed_hash": after,
    }
    return next_state, round_row, maps, claims, design_inputs, exact_inputs


def mask_text(candidate):
    return "" if candidate is None else "P" + "-".join(str(i) for i in candidate.selected)


def enumerate_snapshot(key, round_row, pilot, seed, state, maps, claims, inputs, exact_inputs, masks):
    full = design(*inputs)
    candidates = full.candidates
    hit = full.selections["PCOC"]
    assert len(candidates) == 28 and mask_text(hit) == round_row["candidate_mask"]
    prefix = {**key, "round": round_row["round"]}
    sparse = [item for item in candidates if item.rejection is None]
    masked = 0
    summary = dict.fromkeys((
        "total_masks", "evaluated_after_screen", "design_feasible", "design_feasible_removed",
        "exact_feasible", "exact_feasible_removed", "joint_feasible", "joint_feasible_removed",
        "max_exact_operator_changes", "max_retained_exact_operator_changes",
    ), 0)
    summary["total_masks"] = len(candidates)
    for item in candidates:
        measured = exact(*exact_inputs, item, CONFIG)
        eligible = item.H > 0
        design_feasible = int(eligible and item.feasible)
        exact_feasible = int(eligible and measured.within_tolerance)
        joint = design_feasible * exact_feasible
        updates, _ = counts(state, maps, claims, item.selected)
        operator_changes = updates["operator_only"] + updates["joint"]
        screened_out = int(not item.screened)
        summary["evaluated_after_screen"] += int(item.screened)
        summary["design_feasible"] += design_feasible
        summary["design_feasible_removed"] += design_feasible * screened_out
        summary["exact_feasible"] += exact_feasible
        summary["exact_feasible_removed"] += exact_feasible * screened_out
        summary["joint_feasible"] += joint
        summary["joint_feasible_removed"] += joint * screened_out
        if exact_feasible:
            summary["max_exact_operator_changes"] = max(summary["max_exact_operator_changes"], operator_changes)
            if item.screened:
                summary["max_retained_exact_operator_changes"] = max(summary["max_retained_exact_operator_changes"], operator_changes)
        masks.add({
            **prefix, "mask": mask_text(item), "retained_by_screen": int(item.screened),
            "H": item.H, "A": item.A, "sum_upper_J": item.U, "design_ell": item.ell,
            "design_phi": item.phi, "exact_ell": measured.ell, "exact_phi": measured.phi,
            "design_feasible": design_feasible, "exact_feasible": exact_feasible,
            "actual_operator_changes_if_committed": operator_changes,
        })
    verification = {}
    if hit is not None:
        _, ids, idx = draw_ids(pilot, seed, round_row["round"], 4, VERIFY_DRAWS)
        audit_idx = np.searchsorted(pilot.audit_ids, ids)
        current, stored, cur_claims, stored_claims = exact_inputs
        replay_inputs = (current[:, audit_idx], stored[:, audit_idx], cur_claims, stored_claims)
        for name in ("PCOC", "No-Count", "No-Age", "J-Rank-Feasible", "Unscreened"):
            choice = full.selections[name]
            token = mask_text(choice)
            if choice is not None and token not in verification:
                verification[token] = verify(*replay_inputs, choice, idx, CONFIG).accepted
        assert int(verification[mask_text(hit)]) == round_row["committed"]
    effects = {}
    for tag, name in (("no_age", "No-Age"), ("no_count", "No-Count"),
                      ("j_rank", "J-Rank-Feasible"), ("unscreened", "Unscreened")):
        choice = full.selections[name]
        token = mask_text(choice)
        differs = token != mask_text(hit)
        accepted = verification.get(token, False)
        if hit is None and choice is not None:
            # Unscreened can still select when the screened family is empty.
            _, ids, idx = draw_ids(pilot, seed, round_row["round"], 4, VERIFY_DRAWS)
            audit_idx = np.searchsorted(pilot.audit_ids, ids)
            current, stored, cur_claims, stored_claims = exact_inputs
            accepted = verify(current[:, audit_idx], stored[:, audit_idx], cur_claims,
                              stored_claims, choice, idx, CONFIG).accepted
        updated = 0
        if accepted:
            split_counts, _ = counts(state, maps, claims, choice.selected)
            updated = split_counts["operator_only"] + split_counts["joint"]
        effects.update({f"{tag}_mask": token, f"{tag}_changes_mask": int(differs),
                        f"{tag}_changes_audit": int(accepted != bool(round_row["committed"])),
                        f"{tag}_committed_operators": updated})
    if hit is not None:
        update_parts, _ = counts(state, maps, claims, hit.selected)
        masked = update_parts["operator_only"] + update_parts["joint"]
    return {
        **prefix, **summary,
        "all_exact_feasible_removed": int(summary["exact_feasible"] > 0 and summary["exact_feasible"] == summary["exact_feasible_removed"]),
        "nonempty_screened_family": int(bool(sparse)),
        "count_active": int(full.count_active), "age_active": int(full.age_active),
        "vcia_mask": mask_text(hit), "vcia_passed": round_row["committed"],
        "vcia_committed_operators": round_row["operator_updates"],
        "no_verify_committed_operators": masked,
        "no_verify_changes_decision": int(hit is not None and not round_row["committed"]),
        **effects,
    }


def load_data(directory, name, partition):
    raw = DATASETS[name][0](directory / f"{name}.zip")
    return prepare_pilot(raw, INNER_SEED, partition_seed=partition, sealed_outer_seed=OUTER_SEED)


def roles(pilot):
    _, reserved, outer = _split_roles(pilot.wl_fingerprints, OUTER_SEED)
    return {"train": pilot.train_ids, "audit": pilot.audit_ids, "validation": pilot.test_ids,
            "unused_outer_reserve": reserved, "outer_test": outer}


def diagnostics(pilot, initial, seed, init, arm):
    if arm == "Local":
        bank = initial.copy()
    else:
        features = pilot.features[pilot.train_ids]
        bank = _initial_centers(features, rng(init, 0, 8))
        bank = canonicalize(_lloyd(features, bank, epochs=12, step=1.))[None, :, :]
    batch_size = sum(min(32, len(ids)) for ids in pilot.peer_train_ids)
    for i in range(1, ROUNDS + 1):
        start = perf_counter()
        if arm == "Local":
            round_seed = int(rng(seed, i, 1).integers(0, 2**32, dtype=np.uint32))
            bank = canonicalize(propose(pilot, bank, round_seed, neighbor_weight=0.))
        else:
            ids = rng(seed, i, 8).choice(pilot.train_ids, batch_size, replace=False)
            fitted = _lloyd(pilot.features[ids], bank[0], epochs=1, step=.25)
            bank = canonicalize(_align(fitted, bank[0]))[None, :, :]
        yield i, bank, batch_size, perf_counter() - start


def execute(data, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = manifest(data)
    write_json(out / "frozen_protocol.json", protocol)
    table_names = ("rounds", "enumeration_rounds", "enumeration_masks", "partitions", "splits", "diagnostics")
    tables = {name: Rows(out / f"{name}.csv") for name in table_names}
    banks = {}
    index = []
    peer, coordinator = connection()
    try:
        for name, partition in itertools.product(DATASETS, PARTITIONS):
            pilot = load_data(data, name, partition)
            role_groups = roles(pilot)
            for role, ids in role_groups.items():
                for gid in ids:
                    tables["splits"].add(dict(dataset=name, partition=partition, role=role,
                                              graph_id=int(gid), fingerprint=pilot.wl_fingerprints[gid]))
            for client, ids in enumerate(pilot.peer_train_ids):
                for gid in ids:
                    tables["partitions"].add(dict(dataset=name, partition=partition, client=client, graph_id=int(gid)))
            print(name, partition, {role: len(ids) for role, ids in role_groups.items()}, flush=True)
            for init in INITS:
                seed = int(np.random.SeedSequence([partition, init]).generate_state(1)[0])
                initial = canonicalize(initialize(pilot, init))
                claims = np.full(len(initial), .25)
                zeros = np.zeros(len(initial), dtype=int)
                base = {"dataset": name, "partition": partition, "init": init}
                for arm in DIAGNOSTICS:
                    key = {**base, "regime": "Clean", "arm": arm}
                    for r, bank, batch, seconds in diagnostics(pilot, initial, seed, init, arm):
                        tables["diagnostics"].add({**key, "round": r, "train_examples": batch, "train_seconds": seconds})
                    token = f"run_{len(index):04d}"
                    claimed = np.full(len(bank), .25)
                    banks[token], banks[token + "_claims"] = bank.copy(), claimed
                    index.append({**key, "bank": token, "final_hash": state_id(bank, claimed),
                                  "initial_hash": state_id(initial, claims) if arm == "Local" else None})
                for regime in REGIMES:
                    for arm in POLICIES:
                        key = {**base, "regime": regime, "arm": arm}
                        state = State(initial.copy(), claims.copy(), zeros.copy(), zeros.copy())
                        for r in range(1, ROUNDS + 1):
                            next_state, log, maps, proposals, design_inputs, exact_inputs = gauge_round(
                                pilot, state, key, seed, r, peer, coordinator)
                            tables["rounds"].add(log)
                            if arm == "VCIA":
                                tables["enumeration_rounds"].add(enumerate_snapshot(
                                    key, log, pilot, seed, state, maps, proposals, design_inputs,
                                    exact_inputs, tables["enumeration_masks"]))
                            state = next_state
                        token = f"run_{len(index):04d}"
                        banks[token], banks[token + "_claims"] = state.maps, state.claims
                        index.append({**key, "bank": token, "final_hash": state_id(state.maps, state.claims),
                                      "initial_hash": state_id(initial, claims)})
                print(name, partition, init, "completed", flush=True)
    finally:
        peer.close()
        coordinator.close()
        for table in tables.values():
            table.close()
    np.savez_compressed(out / "frozen_banks.npz", **banks)
    write_json(out / "index.json", index)
    files = ["frozen_protocol.json", "frozen_banks.npz", "index.json", *(name + ".csv" for name in table_names)]
    write_json(out / "trajectory_freeze.json", {"frozen_at_utc": now(), "sha256": {
        name: digest(out / name) for name in files}})


def score(data, out):
    if (out / "outer_scoring.json").exists():
        raise FileExistsError("Outer score already present")
    protocol = json.loads((out / "frozen_protocol.json").read_text(encoding="utf8"))
    frozen = json.loads((out / "trajectory_freeze.json").read_text(encoding="utf8"))
    assert all(digest(out / name) == sha for name, sha in frozen["sha256"].items())
    assert source_hashes() == protocol["source_sha256"]
    assert all(digest(data / f"{name}.zip") == sha for name, sha in protocol["data_sha256"].items())
    score_start = now()
    index = json.loads((out / "index.json").read_text(encoding="utf8"))
    predictions = Rows(out / "outer_predictions.csv")
    endpoints = Rows(out / "endpoints.csv")
    try:
        with np.load(out / "frozen_banks.npz") as banks:
            for (name, partition), group in itertools.groupby(index, lambda r: (r["dataset"], r["partition"])):
                pilot = load_data(data, name, partition)
                ids = roles(pilot)["outer_test"]
                for record in group:
                    key = {k: record[k] for k in IDENTITY}
                    bank = banks[record["bank"]]
                    claims = banks[record["bank"] + "_claims"]
                    assert state_id(bank, claims) == record["final_hash"]
                    scores = all_peer_predictions(pilot, bank, ids)
                    pooled = aggregate(scores, scores, claims, claims, (), CONFIG) if len(bank) > 1 else scores[0]
                    labels = pilot.labels[ids]
                    endpoints.add({**key, "outer_graphs": len(ids), **quality(pooled, labels)})
                    classes = pooled.argmax(axis=1)
                    for gid, label, cl, p in zip(ids, labels, classes, pooled):
                        predictions.add({**key, "graph_id": int(gid), "class": int(label),
                                         "cluster": int(cl), "p0": p[0], "p1": p[1]})
    finally:
        predictions.close()
        endpoints.close()
    write_json(out / "outer_scoring.json", {"started_at_utc": score_start, "finished_at_utc": now(),
                "sha256": {name: digest(out / name) for name in ("endpoints.csv", "outer_predictions.csv")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "score"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/binary-outer"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"execute": execute, "score": score}[args.mode](args.data, args.out)
