"""Online round for the fixed 60-round protocol."""

from time import perf_counter

import numpy as np

from run import (
    CONFIG, DESIGN_DRAWS, POLICIES, VERIFY_DRAWS, PredictionCache, canonicalize,
    commit, counts, design, draw_ids, exact, full_state, mask_text, packet,
    propose, rng, state_id, verify,
)

def gauge_round(pilot, state, key, seed, round_no, peer, coordinator):
    arm = key["arm"]
    before = state_id(state.maps, state.claims)
    start = perf_counter()
    local_seed = int(rng(seed, round_no, 1).integers(0, 2**32, dtype=np.uint32))
    maps = state.maps.copy() if arm == "Frozen" else canonicalize(propose(pilot, state.maps, local_seed))
    claims = np.full(len(maps), .25)
    repeat = np.flatnonzero((round_no + 2 * np.arange(len(maps))) % 7 == 0)
    maps[repeat], claims[repeat] = state.maps[repeat], state.claims[repeat]
    stressed = key["regime"] == "Perturbed" and 11 <= round_no <= 35
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



