"""Run one WL trajectory, with distinct online, offline and evaluation ledgers."""

import json
import tracemalloc
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha256
from math import comb
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from .configuration import RANKERS
from .decision import Census, aggregate, design, verify
from .learner import initialize, predict, predict_features, propose


@dataclass(frozen=True)
class State:
    maps: np.ndarray
    claims: np.ndarray
    submitted: np.ndarray
    changed: np.ndarray


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    ).encode("utf-8")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        json.dumps(value, indent=2, allow_nan=False, default=json_default).encode(
            "utf-8"
        )
        + b"\n"
    )
    temporary.replace(path)


def canonical_maps(maps):
    maps = np.array(np.round(maps, 8), dtype="<f8", order="C", copy=True)
    if not np.isfinite(maps).all():
        raise ValueError("nonfinite prototype payload")
    maps[maps == 0] = 0
    return maps


def digest(value):
    return sha256(value).hexdigest()


def state_id(state):
    return digest(
        state.maps.tobytes() + np.asarray(state.claims, dtype="<f8").tobytes()
    )


def contents(state, maps, claims, selected):
    current = np.isin(np.arange(len(maps)), selected)
    op = current & np.any(maps != state.maps, axis=(1, 2))
    cl = current & (claims != state.claims)
    return {
        "operator_only": int((op & ~cl).sum()),
        "claim_only": int((cl & ~op).sum()),
        "joint": int((op & cl).sum()),
    }, op | cl


def staged_state(state, maps, claims, selected, round_no):
    ids = list(selected)
    _, changed = contents(state, maps, claims, selected)
    new = State(
        state.maps.copy(),
        state.claims.copy(),
        state.submitted.copy(),
        state.changed.copy(),
    )
    new.maps[ids], new.claims[ids] = maps[ids], claims[ids]
    new.submitted[ids] = round_no
    new.changed[changed] = round_no
    for value in (new.maps, new.claims, new.submitted, new.changed):
        value.setflags(write=False)
    return new


def binding(state, staged, selected, round_no, config_id):
    return digest(
        canonical_json(
            {
                "round": round_no,
                "quota": len(selected),
                "ids": selected,
                "configuration": config_id,
                "prior": state_id(state),
                "candidate": state_id(staged),
                "old_submit": state.submitted,
                "old_content": state.changed,
                "submit": staged.submitted,
                "content": staged.changed,
            }
        )
    )


class Predictions:
    def __init__(self, pilot):
        self.pilot = pilot
        self.outputs = {}
        self.cost = {
            "requested_graph_operator_pairs": 0,
            "new_graph_operator_pairs": 0,
            "vectorized_prediction_calls": 0,
            "prediction_seconds": 0.0,
        }

    def get(self, banks, ids):
        outputs = []
        for bank in banks:
            cache = self.outputs.setdefault(digest(bank.tobytes()), {})
            missing = np.asarray([i for i in ids if i not in cache], dtype=int)
            self.cost["requested_graph_operator_pairs"] += len(ids)
            if missing.size:
                start = perf_counter()
                cache.update(zip(missing, predict(self.pilot, bank, missing)))
                self.cost["prediction_seconds"] += perf_counter() - start
                self.cost["new_graph_operator_pairs"] += len(missing)
                self.cost["vectorized_prediction_calls"] += 1
            outputs.append(np.stack([cache[i] for i in ids]))
        return np.stack(outputs)


def quality(scores, labels):
    assignments = scores.argmax(axis=1)
    classes, encoded = np.unique(labels, return_inverse=True)
    counts = contingency_matrix(encoded, assignments)
    rows, cols = linear_sum_assignment(counts, maximize=True)
    matched = np.full(scores.shape[1], -1, dtype=int)
    matched[np.unique(assignments)[cols]] = rows
    return {
        "acc": float(counts[rows, cols].sum() / len(labels)),
        "macro_f1": float(
            f1_score(
                encoded,
                matched[assignments],
                labels=np.arange(len(classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "nmi": float(normalized_mutual_info_score(labels, assignments)),
        "ari": float(adjusted_rand_score(labels, assignments)),
    }


def stream(study, partition, seed, round_no, purpose):
    namespace = [seed, round_no, purpose]
    if not study.paired_inner_seeds:
        namespace.extend((partition, study.probe_seed))
    return np.random.default_rng(np.random.SeedSequence(namespace))


def proposal(study, pilot, state, partition, seed, regime, round_no):
    rng = stream(study, partition, seed, round_no, 1)
    local_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
    maps = canonical_maps(
        propose(
            pilot,
            state.maps,
            local_seed,
            local_epochs=study.local_steps,
            topology=study.topology,
            round_no=round_no,
        )
    )
    claims = np.full(study.peers, study.normal_claim, dtype="<f8")
    if study.repeat_every:
        repeated = (round_no + 2 * np.arange(study.peers)) % study.repeat_every == 0
        maps[repeated], claims[repeated] = state.maps[repeated], state.claims[repeated]
    if regime != "clean" and study.stress_start <= round_no <= study.stress_end:
        for index, peer in enumerate(study.compromised):
            if regime == "perturbed":
                maps[peer] = maps[peer, ::-1].copy()
            else:
                anchor = maps[peer] if regime == "displaced" else state.maps[0]
                direction = anchor[1] - anchor[0]
                sign = -1 if regime == "canceling" and index % 2 else 1
                maps[peer] += sign * study.displacement * direction
            claims[peer] = study.attacked_claim
        maps = canonical_maps(maps)
    missing = stream(
        study, partition, seed, (round_no - 1) // study.missing_block + 1, 6
    )
    available = missing.random(study.peers) >= study.missing_fraction
    maps[~available], claims[~available] = (
        state.maps[~available],
        state.claims[~available],
    )
    return maps, claims, available


def make_records(state):
    from .transport import Record

    return tuple(
        Record(
            identity=i,
            version=int(state.submitted[i]),
            operator=bank.tobytes(),
            claim=np.asarray(state.claims[i], dtype="<f8").tobytes(),
        )
        for i, bank in enumerate(state.maps)
    )


def from_records(records, shape):
    maps = np.stack(
        [
            np.frombuffer(record.operator, dtype="<f8").reshape(shape)
            for record in records
        ]
    )
    claims = np.asarray(
        [np.frombuffer(record.claim, dtype="<f8").item() for record in records]
    )
    if not np.isfinite(maps).all() or not np.isfinite(claims).all():
        raise ValueError("nonfinite bound record")
    return maps, claims


def census_audit(exact, config):
    return {
        **asdict(exact),
        "accepted": exact.within_tolerance,
        "draws": 0,
        "drift_failed": exact.ell > config.chi,
        "assignment_failed": exact.phi > config.gamma,
    }


def peer_audit(prior, candidate, configuration, task, *, features, shape, temperature):
    """Executed by the designated peer using only staged bytes and registered audit features."""
    from threadpoolctl import threadpool_limits

    from .decision import DecisionConfig

    if digest(features.tobytes()) != configuration["audit_features_sha256"]:
        raise ValueError("auditor feature map differs from the registration")
    route = task["route"]
    policy = configuration["policy"]
    expected_route = {
        "Unprotected": "No-Verify",
        "No-Verify": "No-Verify",
        "Census-Audit": "Census-Audit",
    }.get(policy, "sampled")
    if route != expected_route or any(type(i) is not int for i in task["indices"]):
        raise ValueError("audit route and sample indices must match the registration")
    if route == "No-Verify":
        if task["indices"]:
            raise ValueError("the no-verification control has no audit draws")
        return {"accepted": True, "route": route, "draws": 0}
    draws = np.asarray(task["indices"], dtype=int)
    if route == "sampled":
        study = SimpleNamespace(**configuration["study"])
        round_no = max(candidate.submission_clocks)
        generator = stream(
            study, configuration["partition"], configuration["seed"], round_no, 4
        )
        if not np.array_equal(
            draws, generator.integers(len(features), size=study.verify_draws)
        ):
            raise ValueError(
                "draws differ from the registered independent verification stream"
            )
    unique, inverse = np.unique(draws, return_inverse=True)
    maps, claims = from_records(candidate.records, shape)
    old_maps, old_claims = from_records(prior.records, shape)
    if route == "Census-Audit" and not np.array_equal(draws, np.arange(len(features))):
        raise ValueError("audit census must cover the registered finite target")
    if not len(unique) or unique[0] < 0 or unique[-1] >= len(features):
        raise ValueError("invalid bound audit indices")
    cache = {}
    calls = 0

    def predictions(banks):
        nonlocal calls
        result = []
        for bank in banks:
            key = bank.tobytes()
            if key not in cache:
                calls += 1
                cache[key] = predict_features(features[unique], bank, temperature)
            result.append(cache[key])
        return np.stack(result)

    config = DecisionConfig(**configuration["decision"])
    with threadpool_limits(limits=1):
        current, stored = predictions(maps), predictions(old_maps)
        if route == "Census-Audit":
            exact = Census(current, stored, claims, old_claims, config).score(
                task["selected"]
            )
            result = census_audit(exact, config)
        else:
            result = asdict(
                verify(
                    current,
                    stored,
                    claims,
                    old_claims,
                    task["selected"],
                    inverse,
                    config,
                )
            )
    return {
        **result,
        "route": route,
        "worker_forward_pairs": calls * len(unique),
        "worker_requested_pairs": 2 * len(maps) * len(unique),
        "worker_prediction_calls": calls,
    }


def pri_key(candidate):
    return (
        -candidate.H,
        -candidate.A,
        candidate.phi,
        candidate.ell,
        candidate.selected,
    )


def ranking_shadow(ranking, state, maps, claims, cache, audit_ids, positions, config):
    unique, indices = np.unique(positions, return_inverse=True)
    current = cache.get(maps, audit_ids[unique])
    stored = cache.get(state.maps, audit_ids[unique])
    comparison = {}
    for policy in RANKERS:
        nominee = ranking.selections[policy]
        if nominee is None:
            comparison[policy] = {
                "selected": None,
                "accepted": False,
                "operator_updates": 0,
                "claim_updates": 0,
            }
            continue
        evidence = verify(
            current, stored, claims, state.claims, nominee, indices, config
        )
        change, _ = contents(state, maps, claims, nominee.selected)
        comparison[policy] = {
            "selected": nominee.selected,
            "accepted": evidence.accepted,
            "operator_updates": change["operator_only"] + change["joint"],
            "claim_updates": change["claim_only"] + change["joint"],
        }
    return comparison


def run_round(
    pilot,
    study,
    config,
    state,
    partition,
    seed,
    regime,
    arm,
    round_no,
    registration,
    cluster,
):
    from .transport import DurableCommitError, ProtocolError

    started = perf_counter()
    logs = {
        "status": "frozen",
        "selected": None,
        "hold_reason": None,
        "nominated_exact": None,
        "verification": None,
        "fixed_evidence": None,
        "candidates": None,
        "selected_payload_hash": None,
        "prior_state_hash": state_id(state),
        "operator_updates": 0,
        "claim_updates": 0,
        "nominated_H": None,
        "content_updates": {"operator_only": 0, "claim_only": 0, "joint": 0},
    }
    cost = {
        key: 0
        for key in (
            "train_seconds",
            "lloyd_steps",
            "screen_seconds",
            "composition_seconds",
            "composition_evaluations",
            "design_seconds",
            "design_draws",
            "design_unique_graphs",
            "verify_draws",
            "verify_unique_graphs",
            "verification_seconds",
            "offline_seconds",
            "quality_seconds",
            "transport_seconds",
            "wire_bytes",
            "offline_pairs",
            "verification_attempts",
        )
    }
    online = Predictions(pilot)
    executed = state
    if arm != "Frozen":
        t = perf_counter()
        maps, claims, available = proposal(
            study, pilot, state, partition, seed, regime, round_no
        )
        logs["available"] = int(available.sum())
        offered = tuple(np.flatnonzero(available).tolist())
        cost.update(
            train_seconds=perf_counter() - t,
            lloyd_steps=study.peers * study.local_steps,
        )
        configuration_id = digest(canonical_json(registration))
        changed = (
            np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
        ) & available
        ages = round_no - state.changed
        audit_ids = pilot.audit_ids
        reference = None
        reference_cache = Predictions(pilot)
        screened_nomination = arm in {"PCOC", "Census-Audit", "No-Verify"}
        if arm == "Unprotected":
            route = "No-Verify"
            selected = offered[: config.q] if len(offered) >= max(2, config.q) else None
            hold_reason = "insufficient_available_peers"
        else:
            t = perf_counter()
            draw_positions = stream(study, partition, seed, round_no, 2).integers(
                len(audit_ids), size=study.design_draws
            )
            unique, idx = np.unique(draw_positions, return_inverse=True)
            design_set = audit_ids[unique]
            rng = stream(study, partition, seed, round_no, 3)
            rank_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
            design_args = (
                online.get(maps, design_set),
                online.get(state.maps, design_set),
                claims,
                state.claims,
                changed,
                ages,
                idx,
                config,
            )
            rank = design(
                *design_args,
                random_seed=rank_seed,
                screen_only=screened_nomination,
                rank_full=not screened_nomination,
                individual_only=arm == "J-Rank",
                available=available,
            )
            cost.update(
                design_seconds=perf_counter() - t,
                design_draws=len(draw_positions),
                design_unique_graphs=len(unique),
                screen_seconds=rank.screen_seconds,
                composition_seconds=rank.composition_seconds,
                composition_evaluations=len(rank.candidates),
            )
            logs.update(
                priority_count_active=rank.count_active,
                priority_age_active=rank.age_active,
            )
            logs["family_size"] = sum(
                (c.rejection if screened_nomination else c.unscreened_rejection) is None
                for c in rank.candidates
            )
            route = (
                arm
                if arm
                in {"No-Verify", "Census-Audit", "Census-Direct", "Simultaneous-family"}
                else "sampled"
            )
            selector = (
                "PCOC"
                if screened_nomination
                else arm
                if arm in RANKERS
                else "Unscreened"
            )
            choice = rank.selections[selector]
            selected = choice.selected if choice else None
            hold_reason = rank.hold_reasons[selector] or "no_admissible_composition"
        if arm == "J-Rank":
            eligible = sorted(
                (s for s in rank.version_stats if s.eligible),
                key=lambda s: (s.upper_J, s.identity),
            )
            selected = (
                tuple(sorted(s.identity for s in eligible[: config.q]))
                if (len(eligible) >= config.q and len(offered) >= max(2, config.q))
                else None
            )
            hold_reason = (
                "insufficient_available_peers"
                if len(offered) < max(2, config.q)
                else "insufficient_eligible_peers"
            )

        def finite(cache):
            return Census(
                cache.get(maps, audit_ids),
                cache.get(state.maps, audit_ids),
                claims,
                state.claims,
                config,
            )

        def sample(candidate, *, alpha_config=config, candidate_state=None):
            unique, indices = np.unique(verify_positions, return_inverse=True)
            ids = audit_ids[unique]
            if candidate_state is None:
                candidate_state = staged_state(state, maps, claims, candidate, round_no)
            return asdict(
                verify(
                    online.get(candidate_state.maps, ids),
                    online.get(state.maps, ids),
                    candidate_state.claims,
                    state.claims,
                    candidate,
                    indices,
                    alpha_config,
                )
            )

        evidence, transport = None, None
        if route == "Census-Direct":
            t = perf_counter()
            reference = finite(online)
            tested = [
                (candidate, reference.score(candidate))
                for candidate in rank.candidates
                if candidate.H > 0
            ]
            best = min(
                ((c, x) for c, x in tested if x.within_tolerance),
                key=lambda item: (
                    -item[0].H,
                    -item[0].A,
                    item[1].phi,
                    item[1].ell,
                    item[0].selected,
                ),
                default=None,
            )
            selected = best[0].selected if best else None
            if best:
                evidence = {
                    **asdict(best[1]),
                    "accepted": True,
                    "route": route,
                    "draws": 0,
                }
            elif tested:
                evidence = {"accepted": False, "route": route, "draws": 0}
            cost.update(
                verification_seconds=perf_counter() - t,
                verification_attempts=len(tested),
                verify_unique_graphs=len(audit_ids) if tested else 0,
            )
        if route == "Simultaneous-family" and rank.candidates:
            t = perf_counter()
            universe = tuple(c for c in rank.candidates if c.H > 0)
            freeze = [
                binding(
                    state,
                    staged_state(state, maps, claims, c.selected, round_no),
                    c.selected,
                    round_no,
                    configuration_id,
                )
                for c in universe
            ]
            verify_positions = stream(study, partition, seed, round_no, 4).integers(
                len(audit_ids), size=study.verify_draws
            )
            alpha_config = replace(
                config, delta_verify=config.delta_verify / comb(study.peers, config.q)
            )
            tested = [
                (c, sample(c.selected, alpha_config=alpha_config)) for c in universe
            ]
            best = min(
                (item for item in tested if item[1]["accepted"]),
                key=lambda item: pri_key(item[0]),
                default=None,
            )
            selected = best[0].selected if best else None
            evidence = {
                **(best[1] if best else {}),
                "accepted": bool(best),
                "route": route,
                "draws": len(verify_positions) if universe else 0,
            }
            logs["family_digests"] = freeze
            cost.update(
                verification_seconds=perf_counter() - t,
                verification_attempts=len(tested),
                verify_unique_graphs=len(np.unique(verify_positions))
                if universe
                else 0,
                verify_draws=len(verify_positions) if universe else 0,
            )

        logs["status"] = "no_candidate"
        logs["hold_reason"] = hold_reason
        if (
            route in {"Census-Direct", "Simultaneous-family"}
            and evidence
            and cost["verification_attempts"]
            and selected is None
        ):
            logs.update(
                status="verification_rejected",
                hold_reason="no_verified_family_member",
                verification=evidence,
            )
        if selected is not None:
            new = staged_state(state, maps, claims, selected, round_no)
            bound_id = binding(state, new, selected, round_no, configuration_id)
            change, _ = contents(state, maps, claims, selected)
            logs.update(
                selected=selected,
                selected_payload_hash=bound_id,
                nominated_state_hash=state_id(new),
                candidate_operator_updates=change["operator_only"] + change["joint"],
            )
            if arm != "Unprotected":
                logs["nominated_H"] = sum(
                    rank.version_stats[i].detectable for i in selected
                )
            t = perf_counter()
            try:
                if cluster is not None:
                    proposal_records = make_records(
                        State(
                            maps,
                            claims,
                            np.where(available, round_no, state.submitted),
                            state.changed,
                        )
                    )
                    transport = cluster.stage(round_no, proposal_records, selected)
                    if transport.snapshot.records != make_records(new):
                        raise ProtocolError(
                            "staged peer payload differs from the local nomination"
                        )
                    logs["selected_payload_hash"] = transport.digest
                if evidence is None:
                    if route == "Census-Audit":
                        verify_positions = np.arange(len(audit_ids))
                    elif route == "sampled":
                        verify_positions = stream(
                            study, partition, seed, round_no, 4
                        ).integers(len(audit_ids), size=study.verify_draws)
                    else:
                        verify_positions = np.empty(0, dtype=int)
                    cost["verification_attempts"] = int(route != "No-Verify")
                    cost["verify_draws"] = (
                        len(verify_positions) if route == "sampled" else 0
                    )
                    cost["verify_unique_graphs"] = len(np.unique(verify_positions))
                    if cluster is not None:
                        evidence = cluster.audit(
                            transport,
                            {
                                "route": route,
                                "indices": verify_positions.tolist(),
                                "selected": list(selected),
                            },
                        )
                        for local, peer in (
                            ("new_graph_operator_pairs", "worker_forward_pairs"),
                            (
                                "requested_graph_operator_pairs",
                                "worker_requested_pairs",
                            ),
                            ("vectorized_prediction_calls", "worker_prediction_calls"),
                        ):
                            online.cost[local] += evidence.get(peer, 0)
                    elif route == "Census-Audit":
                        exact = Census(
                            online.get(new.maps, audit_ids),
                            online.get(state.maps, audit_ids),
                            new.claims,
                            state.claims,
                            config,
                        ).score(selected)
                        evidence = census_audit(exact, config)
                    elif route == "sampled":
                        evidence = sample(selected, candidate_state=new)
                    else:
                        evidence = {"accepted": True, "draws": 0}
                    evidence = {**evidence, "route": route}
                if evidence["accepted"]:
                    if bound_id != binding(
                        state, new, selected, round_no, configuration_id
                    ):
                        raise ProtocolError("binding changed before local execution")
                    if cluster is not None:
                        # Commit the exact auditor receipt. The route is already part of that receipt.
                        cluster.commit(transport, evidence)
                        if cluster.snapshot.records != make_records(new):
                            raise DurableCommitError(
                                "durable peer state diverged from the expected candidate"
                            )
                    executed = new
                    accepted, _ = contents(state, maps, claims, selected)
                    logs.update(
                        status="accepted",
                        hold_reason=None,
                        content_updates=accepted,
                        operator_updates=accepted["operator_only"] + accepted["joint"],
                        claim_updates=accepted["claim_only"] + accepted["joint"],
                    )
                else:
                    if cluster is not None:
                        cluster.abort(transport)
                    logs.update(
                        status="verification_rejected", hold_reason="audit_rejected"
                    )
            except DurableCommitError:
                raise
            except ProtocolError as error:
                if cluster is not None:
                    cluster.recover()
                logs.update(status="transport_hold", hold_reason=str(error))
            cost["verification_seconds"] += perf_counter() - t
            logs["verification"] = evidence
            if cluster is not None:
                cost.update(
                    transport=asdict(cluster.last_cost),
                    transport_seconds=cluster.last_cost.elapsed_seconds,
                    wire_bytes=cluster.last_cost.tx_bytes + cluster.last_cost.rx_bytes,
                )

        if study.census == "all" or (
            selected is not None
            and (study.census == "nominated" or study.fixed_evidence)
        ):
            t = perf_counter()
            offline = reference if reference is not None else finite(reference_cache)
            if selected is not None:
                logs["nominated_exact"] = asdict(offline.score(selected))
                if study.fixed_evidence:
                    distances, flips = offline.evidence(selected)
                    logs["fixed_evidence"] = {
                        "distances": distances,
                        "flips": flips,
                        "chi": config.chi,
                        "alpha": config.delta_verify / 2,
                        "gamma": config.gamma,
                        "output_range": 2 / pilot.n_clusters,
                        "design_unique": len(design_set),
                        "distinct_operators": len(
                            {
                                x.tobytes()
                                for x in np.concatenate((state.maps, new.maps))
                            }
                        ),
                    }
                if logs["operator_updates"]:
                    target = pilot.audit_ids
                    before = reference_cache.get(state.maps, target)
                    after = reference_cache.get(executed.maps, target)
                    effects = np.abs(before - after).mean(axis=(1, 2))
                    logs["observable_operator_updates"] = int(
                        (effects > config.nu).sum()
                    )
            if study.census == "all" and arm != "Unprotected":
                universe = (
                    rank
                    if not screened_nomination and arm != "J-Rank"
                    else design(
                        *design_args,
                        random_seed=rank_seed,
                        rank_full=True,
                        available=available,
                    )
                )
                logs["candidates"] = []
                for c in universe.candidates:
                    change, _ = contents(state, maps, claims, c.selected)
                    logs["candidates"].append(
                        {
                            **asdict(c),
                            "exact": asdict(offline.score(c)),
                            "operator_updates": change["operator_only"]
                            + change["joint"],
                            "claim_updates": change["claim_only"] + change["joint"],
                        }
                    )
            cost.update(
                offline_seconds=perf_counter() - t,
                offline_pairs=reference_cache.cost["new_graph_operator_pairs"],
            )
        if (study.fixed_evidence and study.census == "all") or (
            study.fixed_ranking_shadow and arm == "Unscreened"
        ):
            t = perf_counter()
            positions = stream(study, partition, seed, round_no, 4).integers(
                len(audit_ids), size=study.verify_draws
            )
            shadow_rank = (
                universe if study.fixed_evidence and study.census == "all" else rank
            )
            logs["fixed_ranking"] = ranking_shadow(
                shadow_rank,
                state,
                maps,
                claims,
                reference_cache,
                audit_ids,
                positions,
                config,
            )
            cost["offline_seconds"] += perf_counter() - t
            cost["offline_pairs"] = reference_cache.cost["new_graph_operator_pairs"]
    if not logs["operator_updates"] and study.census != "none":
        logs["observable_operator_updates"] = 0
    cost.update(online.cost)
    cost["cache_hits"] = (
        cost["requested_graph_operator_pairs"] - cost["new_graph_operator_pairs"]
    )
    t = perf_counter()
    eval_cache = Predictions(pilot)
    outputs = eval_cache.get(executed.maps, pilot.test_ids)
    scores = aggregate(outputs, outputs, executed.claims, executed.claims, (), config)
    logs["quality"] = quality(scores, pilot.labels[pilot.test_ids])
    if study.log_predictions:
        logs.update(eval_assignments=scores.argmax(axis=1), eval_scores=scores)
    cost.update(
        quality_seconds=perf_counter() - t,
        evaluation_pairs=eval_cache.cost["new_graph_operator_pairs"],
        round_seconds=perf_counter() - started,
    )
    logs.update(
        cost=cost,
        executed_state_hash=state_id(executed),
        submission_clocks=executed.submitted,
        content_clocks=executed.changed,
        content_ages=round_no - executed.changed,
    )
    return executed, logs


def run_job(
    pilot, study, dataset, partition, seed, regime, arm, directory, data_sha256
):
    from .transport import PeerCluster, Snapshot

    config = study.thresholds(dataset)
    maps = canonical_maps(initialize(pilot, seed))
    if study.poisoned_memory:
        for peer in study.compromised:
            maps[peer] = maps[peer, ::-1].copy()
    state = State(
        maps,
        np.full(study.peers, study.normal_claim, dtype="<f8"),
        np.zeros(study.peers, dtype=int),
        np.zeros(study.peers, dtype=int),
    )
    features = np.asarray(pilot.features[pilot.audit_ids], dtype="<f8", order="C")
    registration = {
        "study": asdict(study),
        "dataset": dataset,
        "partition": partition,
        "seed": seed,
        "regime": regime,
        "policy": arm,
        "decision": asdict(config),
        "alignment": "min-square/matched-own",
        "encoding": "WL-histogram, little-endian f64, 8 decimals",
        "temperature": pilot.temperature,
        "data_sha256": data_sha256,
        "feature_shape": maps.shape[1:],
        "audit_features_sha256": digest(features.tobytes()),
        "audit_ids_sha256": digest(np.asarray(pilot.audit_ids, dtype="<i8").tobytes()),
        "evaluation_ids_sha256": digest(
            np.asarray(pilot.test_ids, dtype="<i8").tobytes()
        ),
        "probe_law": "uniform with replacement on registered audit IDs",
    }
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "registration.json", registration)
    write_json(
        directory / "roles.json",
        {
            "train_ids": pilot.train_ids,
            "audit_ids": pilot.audit_ids,
            "evaluation_ids": pilot.test_ids,
            "peer_train_ids": pilot.peer_train_ids,
            "split_seed": pilot.split_seed,
            "partition_seed": pilot.partition_seed,
            "partition_attempts": pilot.partition_attempts,
            "max_fingerprint_group": pilot.max_fingerprint_group,
            "features_sha256": digest(
                np.asarray(pilot.features, dtype="<f8", order="C").tobytes()
            ),
        },
    )
    context = nullcontext()
    if study.backend == "loopback" and arm != "Frozen":
        snapshot = Snapshot(
            records=make_records(state),
            submission_clocks=tuple(state.submitted.tolist()),
            content_clocks=tuple(state.changed.tolist()),
        )
        context = PeerCluster(
            directory / "protocol",
            snapshot,
            registration,
            quota=study.quota,
            audit_function=partial(
                peer_audit,
                features=features,
                shape=maps.shape[1:],
                temperature=pilot.temperature,
            ),
        )
    last = None
    accepted = 0
    operator_updates = 0
    transport_setup = None
    with (
        context as cluster,
        (directory / "rounds.jsonl").open("w", encoding="utf-8", newline="\n") as file,
    ):
        if cluster is not None:
            transport_setup = asdict(cluster.setup_cost)
        for round_no in range(1, study.rounds + 1):
            if study.profile_memory:
                tracemalloc.start()
            try:
                state, row = run_round(
                    pilot,
                    study,
                    config,
                    state,
                    partition,
                    seed,
                    regime,
                    arm,
                    round_no,
                    registration,
                    cluster,
                )
                if study.profile_memory:
                    row["cost"]["peak_traced_python_bytes"] = (
                        tracemalloc.get_traced_memory()[1]
                    )
            finally:
                if study.profile_memory:
                    tracemalloc.stop()
            segment = (
                "warmup"
                if round_no < study.stress_start
                else ("stress" if round_no <= study.stress_end else "recovery")
            )
            row.update(
                round=round_no,
                dataset=dataset,
                phase=study.phase,
                partition_seed=partition,
                seed=seed,
                regime=regime,
                arm=arm,
                segment=segment,
                scheduled_rounds=study.rounds,
            )
            file.write(canonical_json(row).decode("utf-8") + "\n")
            file.flush()
            accepted += row["status"] == "accepted"
            operator_updates += row["operator_updates"]
            last = row
    summary = {
        "status": "complete",
        "dataset": dataset,
        "phase": study.phase,
        "partition_seed": partition,
        "seed": seed,
        "regime": regime,
        "arm": arm,
        "scheduled_rounds": study.rounds,
        "accepted_rounds": accepted,
        "operator_updates": operator_updates,
        "endpoint": last["quality"],
        "end_state_hash": last["executed_state_hash"],
        "transport_setup": transport_setup,
    }
    with (directory / "checkpoint.npz").open("wb") as file:
        np.savez_compressed(
            file,
            operators=state.maps,
            claims=state.claims,
            submission_clocks=state.submitted,
            content_clocks=state.changed,
        )
    write_json(directory / "summary.json", summary)
    return summary
