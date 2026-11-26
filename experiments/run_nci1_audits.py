"""Benchmark census and independent IID audits on nested disjoint NCI1 pools."""

from argparse import ArgumentParser
import csv
from itertools import combinations, product
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
import followup

base = followup.base
ROUNDS = (5, 11, 20, 29, 40, 58)
SIZES = (256, 493, 1024, 1315)
METHODS = (("Census", 0), ("IID 2048", 2048), ("IID 8192", 8192))
CACHES = ("Cold", "Design-warm")
TIMING_REPS = 3
POOL_SEED = 6112401
AUDIT_SEED = 6112402


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pools_for(pilot, partition, split_rows):
    roles = base.roles(pilot)
    if split_rows is not None:
        for name, ids in roles.items():
            expected = split_rows[name]
            assert list(ids) == sorted(expected), name
            assert all(pilot.wl_fingerprints[i] == expected[i] for i in ids)
    groups = [{pilot.wl_fingerprints[i] for i in ids} for ids in roles.values()]
    assert all(not left & right for left, right in combinations(groups, 2))
    native, reserve = roles["audit"], roles["unused_outer_reserve"]
    assert len(native) == 493 and len(reserve) == 822 and len(roles["outer_test"]) == 822
    rng = np.random.default_rng([POOL_SEED, partition])
    short = np.sort(rng.permutation(native)[:256])
    extra = rng.permutation(reserve)
    pools = {256: short, 493: native, 1024: np.union1d(native, extra[:531]),
             1315: np.union1d(native, extra)}
    for size, ids in pools.items():
        assert len(ids) == size
        assert all(not np.intersect1d(roles[role], ids).size for role in ("outer_test", "train", "validation"))
    assert all(set(pools[left]) <= set(pools[right]) for left, right in zip(SIZES, SIZES[1:]))
    return pools


def replay(out, history, split_rows, raw, partitions, initializations, final_round):
    scheduled = base.Rows(out / "scheduled_snapshots.csv")
    sockets = base.connection()
    cases, memberships = [], []
    verified = 0
    try:
        for partition in partitions:
            pilot = base.prepare_pilot(raw, base.INNER_SEED, partition_seed=partition, sealed_outer_seed=base.OUTER_SEED)
            pools = pools_for(pilot, partition, split_rows.get(partition))
            memberships.extend(dict(partition=partition, pool=size, ids=ids) for size, ids in pools.items())
            for init, regime in product(initializations, base.REGIMES):
                seed = int(np.random.SeedSequence([partition, init]).generate_state(1)[0])
                initial = base.canonicalize(base.initialize(pilot, init))
                state = base.State(initial.copy(), np.full(8, .25), np.zeros(8, dtype=int), np.zeros(8, dtype=int))
                key = dict(dataset="NCI1", partition=partition, init=init, regime=regime, arm="VCIA")
                for rnd in range(1, final_round + 1):
                    previous = state
                    state, row, maps, claims, design_inputs, _ = followup.gauge_round(pilot, state, key, seed, rnd, *sockets)
                    if history:
                        ref = history[(partition, init, regime, rnd)]
                        for name in ("stored_hash", "candidate_hash", "committed_hash", "status", "candidate_mask", "operator_updates"):
                            assert ("" if row[name] is None else str(row[name])) == ref[name], (key, rnd, name, row[name], ref[name])
                        verified += 1
                    if rnd not in ROUNDS:
                        continue
                    token = f"NCI1-{partition}-{init}-{regime}-{rnd}"
                    scheduled.add(dict(snapshot_id=token, **key, round=rnd, status=row["status"],
                                       candidate_mask=row["candidate_mask"], stored_hash=row["stored_hash"],
                                       candidate_hash=row["candidate_hash"], committed_hash=row["committed_hash"]))
                    if row["attempted"]:
                        choice = base.design(*design_inputs, screen_only=True).selections["PCOC"]
                        assert base.mask_text(choice) == row["candidate_mask"]
                        candidate, candidate_claims = base.full_state(previous, maps, claims, choice.selected)
                        assert base.state_id(candidate, candidate_claims) == row["candidate_hash"]
                        _, design_ids, _ = base.draw_ids(pilot, seed, rnd, 2, base.DESIGN_DRAWS)
                        cases.append((token, row, pilot, previous, candidate, claims, choice, pools, design_ids))
            print("Frozen nominations reconstructed:", partition, len(cases), flush=True)
    finally:
        scheduled.close()
        for socket in sockets:
            socket.close()
    return cases, memberships, verified


def measure(pilot, state, candidate, claims, choice, pool, design_ids, random_key, cache_mode, method):
    name, draws = method
    cache = base.PredictionCache(pilot)
    if cache_mode == "Design-warm":
        cache.get(candidate, design_ids)
        cache.get(state.maps, design_ids)
    before = cache.evaluated
    start = perf_counter()
    if draws:
        sampled = np.random.default_rng(random_key).choice(pool, size=draws, replace=True)
        ids, inverse = np.unique(sampled, return_inverse=True)
    else:
        ids, inverse = pool.copy(), None
    prepared = perf_counter()
    cur, mem = cache.get(candidate, ids), cache.get(state.maps, ids)
    inferred = perf_counter()
    args = cur, mem, claims, state.claims, choice
    config = base.CONFIG
    if draws:
        report = base.verify(*args, inverse, config)
        passed, upper_ell, upper_phi = report.accepted, report.upper_ell, report.upper_phi
    else:
        report = base.exact(*args, config)
        passed, upper_ell, upper_phi = report.within_tolerance, None, None
    end = perf_counter()
    values = dict(method=name, audit_draws=draws, graph_visits=draws or len(ids), unique_graphs=len(ids),
                  new_graph_operator_evals=cache.evaluated - before, estimate_ell=report.ell,
                  estimate_phi=report.phi, audit_upper_ell=upper_ell, audit_upper_phi=upper_phi, passed=int(passed))
    times = dict(prep_ms=1000 * (prepared - start), inference_ms=1000 * (inferred - prepared),
                 decision_ms=1000 * (end - inferred), audit_ms=1000 * (end - start))
    return values, times


def audit(out, cases):
    results = base.Rows(out / "candidate_results.csv")
    timing = base.Rows(out / "timings.csv")
    warmups = set()
    try:
        for index, (token, row, pilot, state, candidate, claims, choice, pools, design_ids) in enumerate(cases):
            for size, pool in pools.items():
                case = dict(snapshot_id=token, dataset="NCI1", partition=row["partition"], init=row["init"],
                            regime=row["regime"], round=row["round"], pool_size=size,
                            mask=row["candidate_mask"], chi=base.CONFIG.chi, gamma=base.CONFIG.gamma)
                key = [AUDIT_SEED, row["partition"], row["init"], base.REGIMES.index(row["regime"]), row["round"], size]
                collected, measurements = {}, {}
                for cache_mode in CACHES:
                    common = (pilot, state, candidate, claims, choice, pool, design_ids, key, cache_mode)
                    warm = (size, cache_mode)
                    if warm not in warmups:
                        for method in METHODS:
                            measure(*common, method)
                        warmups.add(warm)
                    for rep in range(TIMING_REPS):
                        offset = (index + rep) % len(METHODS)
                        for order, method in enumerate(METHODS[offset:] + METHODS[:offset], 1):
                            values, times = measure(*common, method)
                            label = cache_mode, method
                            assert collected.setdefault(label, values) == values
                            measurements.setdefault(label, []).append(times["audit_ms"])
                            timing.add(dict(snapshot_id=token, pool_size=size, cache=cache_mode, method=method[0],
                                            repeat=rep + 1, order=order, **times))
                truth = collected[("Cold", METHODS[0])]
                if size == 493:
                    np.testing.assert_allclose([truth["estimate_ell"], truth["estimate_phi"]],
                                               [row["exact_ell"], row["exact_phi"]], rtol=0, atol=2e-13)
                    assert truth["passed"] == row["audit_threshold_within"]
                for (cache_mode, method), values in collected.items():
                    cold = collected[("Cold", method)]
                    for field in ("estimate_ell", "estimate_phi", "passed", "unique_graphs"):
                        assert cold[field] == values[field]
                    feasible = truth["passed"]
                    results.add(dict(**case, cache=cache_mode, finite_ell=truth["estimate_ell"],
                                     finite_phi=truth["estimate_phi"], finite_feasible=feasible,
                                     feasible_pass=feasible * values["passed"],
                                     excess_pass=(1 - feasible) * values["passed"],
                                     feasible_held=feasible * (1 - values["passed"]),
                                     audit_ms=float(np.median(measurements[(cache_mode, method)])), **values))
            if index % 15 == 14 or index + 1 == len(cases):
                print("Measured frozen nominations:", index + 1, "/", len(cases), flush=True)
    finally:
        results.close()
        timing.close()


def load_reference(directory):
    history = {
        (int(r["partition"]), int(r["init"]), r["regime"], int(r["round"])): r
        for r in read(directory / "rounds.csv") if r["dataset"] == "NCI1" and r["arm"] == "VCIA"
    }
    assert len(history) == 1800
    split_rows = {
        p: {role: {} for role in ("train", "audit", "validation", "unused_outer_reserve", "outer_test")}
        for p in base.PARTITIONS
    }
    for r in read(directory / "splits.csv"):
        if r["dataset"] == "NCI1":
            split_rows[int(r["partition"])][r["role"]][int(r["graph_id"])] = r["fingerprint"]
    return history, split_rows


def main(out, archive, reference, smoke):
    digest = base.digest(archive)
    if digest != followup.DATASETS["NCI1"][1]:
        raise ValueError(f"Unexpected SHA-256 for {archive.name}: {digest}")
    history, split_rows = load_reference(reference) if reference is not None else ({}, {})
    partitions = base.PARTITIONS[:1] if smoke else base.PARTITIONS
    initializations = base.INITS[:1] if smoke else base.INITS
    final_round = 20 if smoke else 60
    scheduled_rounds = [round_no for round_no in ROUNDS if round_no <= final_round]
    trajectory_count = len(partitions) * len(initializations) * len(base.REGIMES)
    sources = [Path(__file__), Path(followup.__file__), Path(base.__file__), HERE / "long_driver.py",
               base.PILOT / "decision.py", base.PILOT / "learner.py", base.PILOT / "run_pilot.py"]
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "execution.json",
                    dict(start_utc=base.now(), source_file_sha256={p.relative_to(HERE).as_posix(): base.digest(p)
                                                                 for p in sources},
                         archive_sha256=digest, partition=partitions, initializations=initializations,
                         regimes=base.REGIMES, rounds=scheduled_rounds, final_round=final_round,
                         pool_sizes=SIZES, pool_seed=POOL_SEED, audit_seed=AUDIT_SEED,
                         timing_reps=TIMING_REPS, audit_methods=METHODS, caches=CACHES,
                         cpu=platform.processor(), os=platform.platform(), numerical_threads=1,
                         smoke=smoke))
    cases, membership, checked = replay(out, history, split_rows, followup.load_nci1(archive),
                                        partitions, initializations, final_round)
    assert checked == (trajectory_count * final_round if history else 0)
    assert cases
    base.write_json(out / "pools.json", membership)
    audit(out, cases)
    base.write_json(out / "verification.json",
                    dict(archived_rounds_checked=checked, replayed_rounds=trajectory_count * final_round,
                         scheduled=trajectory_count * len(scheduled_rounds), nominated=len(cases),
                         finished_utc=base.now(), sha256={p.name: base.digest(p) for p in out.iterdir() if p.is_file()}))


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("data/NCI1.zip"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--reference", type=Path, help="Optional archived splits.csv and rounds.csv for state comparison.")
    cli.add_argument("--smoke", action="store_true")
    options = cli.parse_args()
    with threadpool_limits(limits=1):
        main(options.output, options.data, options.reference, options.smoke)
