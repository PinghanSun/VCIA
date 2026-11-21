"""Replay all archived VCIA snapshots, freeze counterfactual predictions, then score."""

from collections import defaultdict
from csv import DictReader
from itertools import product
import json
from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

WORK = Path(__file__).resolve().parent
OUT = Path("runs/graph-snapshots")
import run as base
from decisions import search

ARCHIVE = Path("runs/binary-outer")
DATA = Path("data")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def identity(row):
    return tuple(str(row[k]) for k in ("dataset", "partition", "init", "regime", "round"))


def mask_text(mask):
    return "P" + "-".join(map(str, mask))


def choice_text(choice):
    return mask_text(choice.selected) if choice else ""


def freeze(paths, name):
    base.write_json(OUT / name, {"frozen_at_utc": base.now(),
                                "sha256": {p.name: base.digest(p) for p in paths}})


def execute():
    source_hash = read_json(ARCHIVE / "trajectory_freeze.json")["sha256"]
    assert all(base.digest(ARCHIVE / name) == source_hash[name] for name in
               ("rounds.csv", "enumeration_masks.csv", "frozen_protocol.json", "splits.csv"))
    assert base.source_hashes() == read_json(ARCHIVE / "frozen_protocol.json")["source_sha256"]
    assert all(base.digest(DATA / f"{name}.zip") == sha for name, (_, sha) in base.DATASETS.items())
    configs = [("VCIA", 512, None), ("J-only", 512, None), ("Unscreened", 512, None),
               ("J+D 0.75κ", 512, .75), ("J+D 0.50κ", 512, .50), ("J+D 0.90κ", 512, .90),
               ("J+D 1.00κ", 512, 1.), ("VCIA", 256, None), ("VCIA", 1024, None)]
    protocol = {
        "status": "recorded before new replay and scoring; historical contexts observed",
        "source_hashes": source_hash,
        "code_sha256": {name: base.digest(WORK / name) for name in ("snapshots.py", "decisions.py")},
        "comparisons": configs,
        "j_native": "delta_screen/N, eligibility based on J only",
        "j_component": "delta_screen/(2N), held fixed for J+D contrasts; J+D 1.00κ is a negative control",
        "d_bound": "U_D=min(U_J, empirical Bernstein D upper bound with width 2R_K)",
        "audit": "same chosen archived VCIA mask; independent of phase-2 design; phase-4 sequence prefixes paired across audit sizes; uniform IID with replacement from the same finite audit pool",
        "design_budget": "same stored bank and proposals; phase-2 sequence prefixes paired across 256/512/1024; identical verification at 8192",
        "per_round_delta": "screen=1/1200, verify=1/1200 shared equally by ell and phi; 30 rounds at most 0.05 by union bound for one fixed policy if assumptions hold; descriptive sweeps are not one simultaneous guarantee",
        "timings": "single-thread local wall time; search setup/screen/enumeration and audit sampling/check on cached outputs, excluding common prediction construction, remote communication and offline outer scoring",
        "outer_scoring": "stored and counterfactual scores for unique selected masks frozen first; selection and audit do not access outer outcomes",
        "probe_tolerances": {"ell": base.CONFIG.chi, "phi": base.CONFIG.gamma, "design_fraction": base.CONFIG.eta},
    }
    base.write_json(OUT / "snapshot_protocol.json", protocol)
    archived = {identity(row): row for row in read_csv(ARCHIVE / "rounds.csv") if row["arm"] == "VCIA"}
    enumeration = defaultdict(dict)
    for row in read_csv(ARCHIVE / "enumeration_masks.csv"):
        enumeration[identity(row)][row["mask"]] = row
    assert len(archived) == len(enumeration) == 1800
    names = ("screen_detail.csv", "audit_detail.csv", "snapshot_index.json", "snapshot_predictions.npz")
    screen, audit = (base.Rows(OUT / name) for name in names[:2])
    index, predictions = [], {}
    peer, coordinator = base.connection()
    try:
        for dataset, partition in product(base.DATASETS, base.PARTITIONS):
            pilot = base.load_data(DATA, dataset, partition)
            outer = base.roles(pilot)["outer_test"]
            for init, regime in product(base.INITS, base.REGIMES):
                initial = base.canonicalize(base.initialize(pilot, init))
                state = base.State(initial, np.full(8, .25), np.zeros(8, dtype=int), np.zeros(8, dtype=int))
                seed = int(np.random.SeedSequence([partition, init]).generate_state(1)[0])
                key = dict(dataset=dataset, partition=partition, init=init, regime=regime, arm="VCIA")
                for r in range(1, base.ROUNDS + 1):
                    next_state, replay, maps, claims, _, inputs = base.gauge_round(
                        pilot, state, key, seed, r, peer, coordinator)
                    prefix = {**key, "round": r, "perturbed_now": replay["perturbed"]}
                    old = archived[identity(prefix)]
                    for field in ("stored_hash", "candidate_hash", "committed_hash", "candidate_mask", "status"):
                        assert str(replay[field] or "") == str(old[field] or ""), (identity(prefix), field)
                    old_masks = enumeration[identity(prefix)]
                    assert len(old_masks) == 28
                    exact_count = sum(int(m["exact_feasible"]) for m in old_masks.values())
                    expected = sum(int(m["design_feasible"]) for m in old_masks.values())
                    changed = np.any(maps != state.maps, axis=(1, 2)) | (claims != state.claims)
                    draws = base.rng(seed, r, 2).choice(len(pilot.audit_ids), 1024, replace=True)
                    age = r - state.last_content
                    ordered = base.rng(seed, r, 42).permutation(len(configs))
                    searches = {}
                    for j in ordered:
                        policy, count, fraction = configs[j]
                        searches[j] = search(*inputs[:4], changed, age, draws[:count], base.CONFIG,
                                             policy=policy, beta_fraction=fraction)
                    registered = searches[0]
                    assert choice_text(registered.choice) == old["candidate_mask"]
                    assert searches[1].masks == searches[6].masks
                    assert choice_text(searches[1].choice) == choice_text(searches[6].choice)
                    audit_draws = base.rng(seed, r, 4).choice(len(pilot.audit_ids), 16384, replace=True)
                    cfg = base.CONFIG
                    token = f"s{len(index):04d}"
                    candidate_names = sorted({choice_text(item.choice) for item in searches.values()} - {""})
                    index.append(dict(**prefix, token=token, masks=["", *candidate_names],
                                      outer_graphs=len(outer), outer_ids=outer.tolist()))
                    start = perf_counter()
                    cur_outer = base.all_peer_predictions(pilot, maps, outer)
                    old_outer = base.all_peer_predictions(pilot, state.maps, outer)
                    predictions[token] = np.stack([
                        base.aggregate(cur_outer, old_outer, claims, state.claims,
                                       tuple(map(int, name[1:].split("-"))) if name else (), cfg)
                        for name in ("", *candidate_names)])
                    scoring_prep_ms = 1000 * (perf_counter() - start)
                    exact_cache = {}
                    for j, (policy, count, fraction) in enumerate(configs):
                        item = searches[j]
                        choice = item.choice
                        name = choice_text(choice)
                        kept = [old_masks[mask_text(mask)] for mask in item.masks]
                        measured = None
                        verification = None
                        audit_ms = 0.
                        if choice:
                            if name not in exact_cache:
                                exact_cache[name] = base.exact(*inputs, choice, cfg)
                            measured = exact_cache[name]
                            start = perf_counter()
                            verification = base.verify(*inputs, choice, audit_draws[:8192], cfg)
                            audit_ms = 1000 * (perf_counter() - start)
                        if j == 0:
                            assert len(kept) == int(old["composition_evaluations"])
                            assert all(int(m["retained_by_screen"]) for m in kept)
                            assert bool(verification and verification.accepted) == bool(int(old["committed"]))
                        passed = bool(verification and verification.accepted)
                        updates = int(old_masks[name]["actual_operator_changes_if_committed"]) if name else 0
                        screen.add(dict(**prefix, policy=policy, design_draws=count, design_unique=item.unique_probes,
                                        audit_pool=len(pilot.audit_ids), legal_masks=28, evaluations=len(kept),
                                        design_feasible_ref_512=expected,
                                        design_feasible_retained_ref_512=sum(int(m["design_feasible"]) for m in kept),
                                        finite_feasible=exact_count, finite_feasible_retained=sum(int(m["exact_feasible"]) for m in kept),
                                        mask=name, selected_J_detectable=choice.H if choice else None,
                                        selected_design_ell=choice.ell if choice else None,
                                        selected_design_phi=choice.phi if choice else None,
                                        selected_exact_ell=measured.ell if measured else None,
                                        selected_exact_phi=measured.phi if measured else None,
                                        selected_exact_feasible=int(measured.within_tolerance) if measured else None,
                                        accepted=int(passed), proposed_operators=updates,
                                        committed_operators=updates if passed else 0,
                                        setup_ms=1000 * item.setup_s, version_ms=1000 * item.screen_s,
                                        composition_ms=1000 * item.compose_s, audit_ms=audit_ms,
                                        decision_ms=1000 * (item.setup_s + item.screen_s + item.compose_s) + audit_ms,
                                        outer_inference_shared_ms=scoring_prep_ms, prediction_key=token))
                    choice = registered.choice
                    mask = choice_text(choice)
                    for label, count in (("No-Verify", 0), ("Independent", 2048), ("Independent", 8192),
                                         ("Independent", 16384), ("Finite census", 0)):
                        observed = upper_ell = upper_phi = accepted = unique = None
                        seconds = 0.
                        if choice:
                            observed = exact_cache[mask]
                            start = perf_counter()
                            if label == "Independent":
                                result = base.verify(*inputs, choice, audit_draws[:count], cfg)
                                accepted, unique = result.accepted, result.unique_graphs
                                upper_ell, upper_phi = result.upper_ell, result.upper_phi
                            elif label == "Finite census":
                                result = base.exact(*inputs, choice, cfg)
                                accepted, unique = result.within_tolerance, result.size
                            else:
                                accepted, unique = True, 0
                            seconds = perf_counter() - start
                        updates = int(old_masks[mask]["actual_operator_changes_if_committed"]) if mask else 0
                        ratio = max(observed.ell / cfg.chi, observed.phi / cfg.gamma) if observed else None
                        band = ("≤0.5" if ratio <= .5 else "0.5–0.9" if ratio <= .9 else
                                "0.9–1.0" if ratio <= 1. else ">1.0") if observed else "no candidate"
                        audit.add(dict(**prefix, policy=label, design_draws=512, audit_draws=count,
                                       candidate_mask=mask, attempted=int(bool(mask)), audit_pool=len(pilot.audit_ids),
                                       unique_graphs=unique, finite_ell=observed.ell if observed else None,
                                       finite_phi=observed.phi if observed else None,
                                       exact_feasible=int(observed.within_tolerance) if observed else None,
                                       upper_ell=upper_ell, upper_phi=upper_phi, band=band, max_threshold_ratio=ratio,
                                       accepted=int(bool(accepted)), committed_operators=updates if accepted else 0,
                                       feasible_committed_operators=updates if accepted and observed.within_tolerance else 0,
                                       audit_ms=1000 * seconds, prediction_key=token))
                    state = next_state
            print("replay checkpoints verified", dataset, partition, flush=True)
    finally:
        peer.close()
        coordinator.close()
        screen.close()
        audit.close()
    base.write_json(OUT / names[2], index)
    np.savez_compressed(OUT / names[3], **predictions)
    freeze([OUT / name for name in (*names, "snapshot_protocol.json")], "snapshot_freeze.json")


def score():
    manifest = read_json(OUT / "snapshot_freeze.json")
    assert all(base.digest(OUT / name) == sha for name, sha in manifest["sha256"].items())
    index = read_json(OUT / "snapshot_index.json")
    start = base.now()
    sheet = base.Rows(OUT / "snapshot_outer.csv")
    labels_by_data = {name: loader(DATA / f"{name}.zip").labels for name, (loader, _) in base.DATASETS.items()}
    try:
        within = ("nmi", "ari", "acc")
        with np.load(OUT / "snapshot_predictions.npz") as data:
            for rec in index:
                scores = data[rec["token"]]
                labels = labels_by_data[rec["dataset"]][rec["outer_ids"]]
                before = base.quality(scores[0], labels)
                for mask, output in zip(rec["masks"], scores, strict=True):
                    after = base.quality(output, labels) if mask else before
                    sheet.add(dict(prediction_key=rec["token"], mask=mask, outer_graphs=len(labels),
                                   **{f"stored_{metric}": before[metric] for metric in within},
                                   **{f"candidate_{metric}": after[metric] for metric in within},
                                   **{f"delta_{metric}": after[metric] - before[metric] for metric in within}))
    finally:
        sheet.close()
    base.write_json(OUT / "snapshot_scoring.json", {"started_at_utc": start, "finished_at_utc": base.now(),
                    "sha256": base.digest(OUT / "snapshot_outer.csv")})


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("mode", choices=("execute", "score"))
    cli.add_argument("--data", type=Path, default=DATA)
    cli.add_argument("--binary", type=Path, default=ARCHIVE)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    DATA, ARCHIVE, OUT = args.data, args.binary, args.output
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        {"execute": execute, "score": score}[args.mode]()
