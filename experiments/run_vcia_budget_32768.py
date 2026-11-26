"""Real-graph closed-loop and matched probe-source checks on the locked study."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from itertools import product
from pathlib import Path
import csv
import json

import numpy as np
from threadpoolctl import threadpool_limits

import run_evidence as old
from decision import _full_key, clopper_pearson_upper

HERE = Path(__file__).resolve().parent

base = old.base
plan = old.PROTOCOL
round_fields = ("dataset", "outer_seed", "rep", "peers", "topology", "condition", "round")
arms = ("Full-8192", "No-Composition-8192", "Full-16384", "Full-32768")
matched_draws = (plan["design_draws"], 2048, plan["verification_draws"])


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, fields=round_fields):
    return tuple(str(row[name]) for name in fields)


def start_state(initial):
    peers = len(initial)
    return base.State(initial.copy(), np.full(peers, .25), np.zeros(peers, int), np.zeros(peers, int))


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
    if 11 <= round_no <= 35 and condition != "Clean":
        stressed = (1, 1 + peers // 2) if condition == "Moderate" else range(1, peers, 2)
        for i in stressed:
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
    pool, _ = design(pilot, state, seed, r, config, cache, operators, plan["design_draws"])
    available = [c for c in pool.candidates if c.H > 0]
    choice = (min(available, key=_full_key, default=None)
              if arm == arms[1] else pool.selections["PCOC"])
    record = dict(row, arm=arm, pre_sha=fingerprint(state), status="no_candidate",
                  mask=None, candidate_sha=None, finite_feasible=None, exact_ell=None, exact_phi=None,
                  possible_operator_updates=0, operator_updates=0, audit_draws=0,
                  no_candidate_reason=(("insufficient_eligible_peers" if
                                       sum(v.eligible for v in pool.version_stats) < config.q
                                       else "no_detectable_update") if arm == arms[1]
                                       else pool.hold_reasons["PCOC"]),
                  eligible_peers=sum(v.eligible for v in pool.version_stats),
                  screened_detectable_masks=len(available),
                  design_feasible_masks=sum(c.feasible for c in available),
                  upper_ell=None, upper_phi=None, new_state_sha=None)
    if choice is not None:
        draws = {"Full-16384": 16384, "Full-32768": 32768}.get(arm, plan["verification_draws"])
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


def close_enough(a, b):
    assert np.isclose(a, float(b), atol=1e-12, rtol=0), (a, b)


def compare_archive(new, archived):
    for actual, stored in (("status", "status"), ("mask", "mask"), ("candidate_sha", "candidate_sha"),
                           ("operator_updates", "operator_updates"), ("new_state_sha", "committed_sha"),
                           ("finite_feasible", "finite_feasible")):
        value = "" if new[actual] is None else str(new[actual])
        assert value == archived[stored], (actual, value, archived)


def replay_probe_sources(pilot, state, maps, claims, seed, row, config, archive_snapshot, archive_fixed):
    cache, operators = cached_predictions(pilot, state, maps, claims)
    r = row["round"]
    exact_args = evaluate(cache, operators, pilot.audit_ids)
    for count in matched_draws:
        pool, used = design(pilot, state, seed, r, config, cache, operators, count)
        choice = pool.selections["PCOC"]
        record = dict(row, history_sha=fingerprint(state), equal_draws=count, mask=old.mask(choice),
                      finite_feasible=None, exact_ell=None, exact_phi=None, pool_graphs=len(pilot.audit_ids),
                      chi=config.chi, gamma=config.gamma, delta_verify=config.delta_verify)
        fresh, _, _ = base.draw_ids(pilot, seed, r, 4, count)
        for label in ("reuse", "fresh"):
            for field in ("passed", "ell", "phi", "upper_ell", "upper_phi"):
                record[f"{label}_{field}"] = None
        historic = count == plan["design_draws"]
        if historic:
            assert str(old.mask(choice) or "") == archive_snapshot["vcia_mask"]
            assert (choice is not None) == (archive_fixed is not None)
        if choice is not None:
            exact = base.exact(*exact_args, choice, config)
            record.update(finite_feasible=int(exact.within_tolerance), exact_ell=exact.ell, exact_phi=exact.phi)
            # Compare equal-sized samples on the same frozen candidate.
            for label, probes in (("reuse", used), ("fresh", fresh)):
                ids, idx = np.unique(probes, return_inverse=True)
                audit = base.verify(*evaluate(cache, operators, ids), choice, idx, config)
                record.update({f"{label}_passed": int(audit.accepted), f"{label}_ell": audit.ell,
                               f"{label}_phi": audit.phi, f"{label}_upper_ell": audit.upper_ell,
                               f"{label}_upper_phi": audit.upper_phi})
            if historic:
                _, ids, idx = base.draw_ids(pilot, seed, r, 4, plan["verification_draws"])
                audit = base.verify(*evaluate(cache, operators, ids), choice, idx, config)
                assert str(int(audit.accepted)) == archive_fixed["passed"]
                assert str(int(exact.within_tolerance)) == archive_fixed["finite_feasible"]
                close_enough(exact.ell, archive_fixed["ell"])
                close_enough(exact.phi, archive_fixed["phi"])
        yield record


def advance_archive(state, maps, claims, archived):
    if archived["mask"]:
        selected = tuple(i for i in range(len(maps)) if int(archived["mask"]) & (1 << i))
        staged, n_updates, changed = stage(state, maps, claims, selected)
        assert staged == archived["candidate_sha"], archived
        if archived["status"] == "committed":
            assert n_updates == int(archived["operator_updates"])
            state = base.commit(state, maps, claims, selected, changed, int(archived["round"]), staged)
    assert fingerprint(state) == archived["committed_sha"], archived
    return state


def save_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(out, records, smoke):
    out.mkdir(parents=True, exist_ok=False)
    seal_path = records / "seal.json"
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))["sha256"]
    archived_names = ("rounds.csv", "snapshots.csv", "fixed_audit.csv", "splits.json",
                      "index.json", "sealed_outer_probabilities.npz")
    for name in archived_names:
        assert base.digest(records / name) == sealed[name], name
    source = [Path(__file__), Path(old.__file__), HERE / "evidence_protocol.json", Path(base.__file__),
              HERE / "multiclass.py", HERE / "snapshots.py", HERE / "decisions.py",
              *(base.PILOT / name for name in ("decision.py", "learner.py", "run_pilot.py", "run_imdb_pilot.py"))]
    base.write_json(out / "protocol.json", dict(
        started_utc=base.now(), smoke=smoke, closed_arms=arms, rounds=plan["rounds"],
        focal=dict(peers=8, q=6, topology="Ring", condition="Moderate"),
        datasets=plan["datasets"], outer_seeds=plan["outer_seeds"], repetitions=2,
        toggle="Drop only design ell and phi feasibility thresholds; retain individual screen, H > 0 and rank key",
        budget_check="Same full design, 8192, 16384 and 32768 audit draws per nomination with nested draw prefixes",
        probe_source="Reconstruct all archived VCIA pre-decision snapshots; nominate with the stated design budget; "
                     "compare exact reused batch against equally sized independent audit batch for the same mask",
        matched_probes=matched_draws, replay_rounds=plan["replay_rounds"],
        cp_min_upper_512=clopper_pearson_upper(0, 512, 1 / 2400),
        underlying_implementation="Actual local CPU state commits; transport omitted from this mechanism experiment",
        source_sha256={str(p.relative_to(HERE.parent)): base.digest(p) for p in source},
        archive_sha256={name: sealed[name] for name in archived_names},
        versions={p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}))
    archive_splits = json.loads((records / "splits.json").read_text(encoding="utf-8"))
    archive_rounds = {key(r): r for r in read_csv(records / "rounds.csv") if r["arm"] == "VCIA"}
    archive_snapshots = {key(r): r for r in read_csv(records / "snapshots.csv")}
    archive_fixed = {key(r): r for r in read_csv(records / "fixed_audit.csv") if r["policy"] == "IID 8192"}
    archive_index = {key(r, round_fields[:-1]): r["token"] for r in
                     json.loads((records / "index.json").read_text(encoding="utf-8")) if r["arm"] == "VCIA"}
    rounds, probes, indexes, holdout, label_sets = [], [], [], {}, {}
    with np.load(records / "sealed_outer_probabilities.npz", allow_pickle=False) as previous:
        for name in plan["datasets"][:1] if smoke else plan["datasets"]:
            graphs, labels, _, _ = old.dataset(name)
            label_sets[name] = labels
            wl, fingerprints = old._wl_features(graphs, 2, 512)
            splits = [s for s in archive_splits if s["dataset"] == name]
            if smoke:
                splits = splits[:1]
            for split in splits:
                outer, rep, peers = (split[k] for k in ("outer_seed", "rep", "peers"))
                seed = split["init_and_round_seed"]
                train, audit, test = (np.asarray(split[k]) for k in ("train", "audit", "test"))
                for restored, archived in zip(base._split_roles(fingerprints, outer), (train, audit, test), strict=True):
                    np.testing.assert_array_equal(restored, archived)
                pilot = old.PilotData(old.representation(wl, None, train, "WL-TFIDF"),
                                      np.full(len(labels), -1), train, audit, test,
                                      [np.asarray(ids) for ids in split["peer_train"]], fingerprints,
                                      outer, split["partition_seed"], split["partition_attempts"],
                                      max(Counter(fingerprints).values()), 6 if name == "ENZYMES" else 2,
                                      2, 512, .25)
                config = base.DecisionConfig(**split["decision"])
                initial = old.initial_maps(pilot, seed)
                cells = (("Ring", "Moderate"),) if smoke else product(plan["topologies"], plan["conditions"])
                for topology, condition in cells:
                    focal = peers == 8 and topology == "Ring" and condition == "Moderate"
                    scope = dict(dataset=name, outer_seed=outer, rep=rep, peers=peers, q=config.q,
                                 topology=topology, condition=condition)
                    history = start_state(initial)
                    current = {arm: start_state(initial) for arm in arms} if focal else {}
                    stop = plan["rounds"] if focal else max(plan["replay_rounds"])
                    for r in range(1, stop + 1):
                        row = dict(scope, round=r)
                        archive_key = key(row)
                        archived = archive_rounds[archive_key]
                        maps, claims = proposal(pilot, history, seed, row)
                        if archive_key in archive_snapshots:
                            probes.extend(replay_probe_sources(pilot, history, maps, claims, seed, row, config,
                                                               archive_snapshots[archive_key],
                                                               archive_fixed.get(archive_key)))
                        history = advance_archive(history, maps, claims, archived)
                        for arm, state in current.items():
                            current[arm], result = closed_step(pilot, state, seed, row, config, arm)
                            if arm == arms[0]:
                                compare_archive(result, archived)
                            rounds.append(result)
                    for arm, state in current.items():
                        token = f"p{len(indexes):03d}"
                        predictions = base.all_peer_predictions(pilot, state.maps, test)
                        holdout[token] = base.aggregate(predictions, predictions, state.claims, state.claims, (), config)
                        if arm == arms[0]:
                            historical = previous[archive_index[key(scope, round_fields[:-1])]][-1]
                            np.testing.assert_allclose(holdout[token], historical, rtol=0, atol=1e-12)
                        indexes.append(dict(scope, arm=arm, token=token, test_ids=test, final_sha=fingerprint(state)))
                print(f"{name} split={outer} rep={rep} peers={peers}: {len(rounds)} loop rounds, "
                      f"{len(probes)} probe comparisons", flush=True)
    assert (len(rounds), len(probes)) == ((240, 18) if smoke else (2880, 2592))
    if not smoke:
        assert sum(r["equal_draws"] == 512 and r["mask"] is not None for r in probes) == 297
    save_csv(out / "rounds.csv", rounds)
    save_csv(out / "probe_pairs.csv", probes)
    base.write_json(out / "test_index.json", indexes)
    np.savez_compressed(out / "sealed_final_predictions.npz", **holdout)
    names = ["rounds.csv", "probe_pairs.csv", "test_index.json", "sealed_final_predictions.npz", "protocol.json"]
    base.write_json(out / "raw_seal.json", dict(sealed_utc=base.now(),
                                              sha256={name: base.digest(out / name) for name in names}))
    by_trajectory = defaultdict(list)
    for row in read_csv(out / "rounds.csv"):
        by_trajectory[key(row, (*round_fields[:-1], "arm"))].append(row)
    scores = []
    with np.load(out / "sealed_final_predictions.npz", allow_pickle=False) as frozen:
        for idx in indexes:
            group = by_trajectory[key(idx, (*round_fields[:-1], "arm"))]
            assert [int(r["round"]) for r in group] == list(range(1, plan["rounds"] + 1))
            assert all(a["new_state_sha"] == b["pre_sha"] for a, b in zip(group, group[1:]))
            assert group[-1]["new_state_sha"] == idx["final_sha"]
            predicted = frozen[idx["token"]]
            metrics = base.quality(predicted, label_sets[idx["dataset"]][idx["test_ids"]])
            overlimit = sum(r["finite_feasible"] == "0" for r in group)
            holds = Counter(r["no_candidate_reason"] for r in group if r["status"] == "no_candidate")
            scores.append(dict({k: v for k, v in idx.items() if k != "test_ids"}, rounds=len(group),
                               nominations=sum(bool(r["mask"]) for r in group), overlimit_nominations=overlimit,
                               commits=sum(r["status"] == "committed" for r in group),
                               overlimit_commits=sum(r["status"] == "committed" and r["finite_feasible"] == "0" for r in group),
                               feasible_audit_holds=sum(r["status"] == "rejected" and r["finite_feasible"] == "1" for r in group),
                               no_nom_screen=holds["insufficient_eligible_peers"],
                               no_nom_composition=holds["no_feasible_composition"],
                               no_nom_undetectable=holds["no_detectable_update"],
                               commits_at_quota=sum(int(r["operator_updates"]) == idx["q"] for r in group),
                               updates=sum(int(r["operator_updates"]) for r in group),
                               feasible_updates=sum(int(r["operator_updates"]) for r in group if r["finite_feasible"] == "1"),
                               audit_draws=sum(int(r["audit_draws"]) for r in group), **metrics))
    save_csv(out / "trajectories.csv", scores)
    base.write_json(out / "score_seal.json", dict(scored_utc=base.now(), sha256={
        name: base.digest(out / name) for name in ("trajectories.csv", "raw_seal.json")}))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing the TUDataset ZIPs.")
    parser.add_argument("--closed-loop", required=True, type=Path, help="Sealed output of run_evidence.py.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    old.DATA = options.data
    with threadpool_limits(limits=1):
        execute(options.output, options.closed_loop, options.smoke)
