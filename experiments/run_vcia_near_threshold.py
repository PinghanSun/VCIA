"""Paired audit replay of the original VCIA nominations at the finite-pool boundary."""

from argparse import ArgumentParser
from collections import Counter
from functools import lru_cache
from itertools import product
from pathlib import Path
import json

import numpy as np
from scipy.stats import binom
from threadpoolctl import threadpool_limits

import run_vcia_budget_32768 as study


base, old = study.base, study.old
design_draws = 512
budgets = (2048, 8192)
direct_reps = 256
conditional_reps = 1024
audit_seed = 187241939
scopes = ("Feasible", "Nearest over-limit", "Other over-limit")


@lru_cache(maxsize=None)
def flip_cutoff(draws, alpha, gamma):
    lower, upper = -1, draws
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if study.clopper_pearson_upper(mid, draws, alpha) <= gamma:
            lower = mid
        else:
            upper = mid
    return lower


def soft_gate(sums, squared_sums, draws, output_range, alpha, chi):
    mean = sums / draws
    variance = np.maximum((squared_sums - draws * np.square(mean)) / (draws - 1), 0)
    log_term = np.log(2 / alpha)
    upper = np.minimum(output_range, mean + np.sqrt(2 * variance * log_term / draws)
                       + 7 * output_range * log_term / (3 * (draws - 1)))
    return upper, upper <= chi


def direct(rng, draws, prefix, moments, args, candidate, config, cutoff):
    n = draws - prefix.sum()
    graph_count = len(prefix)
    counts = rng.multinomial(n, np.full(graph_count, 1 / graph_count), size=direct_reps) + prefix
    sums, squares, flips = (counts @ moments).T
    upper, soft = soft_gate(sums, squares, draws, 2 / args[0].shape[-1], config.delta_verify / 2, config.chi)
    passed = soft & (flips <= cutoff)
    check = base.verify(*args, candidate, np.repeat(np.arange(graph_count), counts[0]), config)
    assert (check.accepted, check.flips) == (passed[0], flips[0])
    np.testing.assert_allclose([check.ell, check.upper_ell], [sums[0] / draws, upper[0]], rtol=0, atol=1e-12)
    return passed, (float(np.mean(sums / draws)), float(np.mean(flips / draws)))


def drift_sums(rng, n, n_reps, distances):
    if n == 0:
        return np.zeros((n_reps, 2))
    counts = rng.multinomial(n, np.full(len(distances), 1 / len(distances)), size=n_reps)
    return counts @ np.column_stack((distances, np.square(distances)))


def conditional(rng, draws, prefix, moments, args, config, cutoff):
    distances, _, flip_mask = moments.T
    flip_mask = flip_mask.astype(bool)
    remaining = draws - int(prefix.sum())
    prior_sums = prefix @ moments
    remaining_cutoff = cutoff - int(prior_sums[2])
    p = float(flip_mask.mean())
    hard_probability = float(binom.cdf(remaining_cutoff, remaining, p))
    if hard_probability == 0:
        return 0., 0, 0, 0.
    # Sample flip counts conditional on passing the reassignment bound.
    k = np.arange(min(remaining, remaining_cutoff) + 1)
    conditional_pmf = binom.pmf(k, remaining, p)
    np.testing.assert_allclose(conditional_pmf.sum(), hard_probability, rtol=1e-10, atol=0)
    groups = np.bincount(rng.choice(k, size=conditional_reps, p=conditional_pmf / conditional_pmf.sum()))
    passed = 0
    for flips, size in enumerate(groups):
        if not size:
            continue
        sums, squares = (prior_sums[:2] + drift_sums(rng, flips, size, distances[flip_mask])
                         + drift_sums(rng, remaining - flips, size, distances[~flip_mask])).T
        _, soft = soft_gate(sums, squares, draws, 2 / args[0].shape[-1], config.delta_verify / 2, config.chi)
        passed += int(soft.sum())
    return hard_probability, conditional_reps, passed, hard_probability * passed / conditional_reps


def score(scope, pilot, state, maps, claims, seed, config, source, entry):
    cache, operators = study.cached_predictions(pilot, state, maps, claims)
    pool, used = study.design(pilot, state, seed, scope["round"], config, cache, operators, design_draws)
    choice = pool.selections["PCOC"]
    assert choice is not None and str(old.mask(choice)) == source["mask"]
    assert study.fingerprint(state) == source["history_sha"]
    args = study.evaluate(cache, operators, pilot.audit_ids)
    exact = base.exact(*args, choice, config)
    before = base.aggregate(*args, (), config)
    after = base.aggregate(*args, choice.selected, config)
    distances = np.abs(after - before).mean(axis=-1)
    flips = before.argmax(axis=-1) != after.argmax(axis=-1)
    moments = np.column_stack((distances, np.square(distances), flips))
    graph_count = len(distances)
    order = {graph: i for i, graph in enumerate(pilot.audit_ids)}
    prefix = np.bincount([order[graph] for graph in used], minlength=graph_count)
    no_prefix = np.zeros_like(prefix)
    for actual, archived in ((distances.mean(), source["exact_ell"]), (flips.mean(), source["exact_phi"]),
                             (prefix @ distances / design_draws, source["reuse_ell"]),
                             (prefix @ flips / design_draws, source["reuse_phi"]),
                             (config.chi, source["chi"]), (config.gamma, source["gamma"])):
        study.close_enough(actual, archived)
    assert str(int(exact.within_tolerance)) == source["finite_feasible"]
    boundary = int(np.floor(config.gamma * graph_count))
    if exact.within_tolerance:
        cohort = scopes[0]
    elif flips.sum() == boundary + 1 and exact.ell <= config.chi:
        cohort = scopes[1]
    else:
        assert flips.sum() > boundary and exact.ell <= config.chi
        cohort = scopes[2]
    for draws in budgets:
        cutoff = flip_cutoff(draws, config.delta_verify / 2, config.gamma)
        rng = [np.random.default_rng(s) for s in np.random.SeedSequence([audit_seed, entry, draws]).spawn(4)]
        reused, reuse_mean = direct(rng[0], draws, prefix, moments, args, choice, config, cutoff)
        fresh, fresh_mean = direct(rng[1], draws, no_prefix, moments, args, choice, config, cutoff)
        row = dict(scope, cohort=cohort, mask=source["mask"], pool_graphs=graph_count,
                   exact_ell=exact.ell, exact_phi=exact.phi, chi=config.chi, gamma=config.gamma,
                   exact_flips=int(flips.sum()), design_draws=design_draws, design_flips=int(prefix @ flips),
                   audit_draws=draws, audit_flip_cutoff=cutoff, repetitions=direct_reps,
                   reuse_pass=int(reused.sum()), fresh_pass=int(fresh.sum()), both_pass=int((reused & fresh).sum()),
                   reuse_only=int((reused & ~fresh).sum()), fresh_only=int((fresh & ~reused).sum()),
                   reuse_ell_bias=reuse_mean[0] - exact.ell, fresh_ell_bias=fresh_mean[0] - exact.ell,
                   reuse_phi_bias=reuse_mean[1] - exact.phi, fresh_phi_bias=fresh_mean[1] - exact.phi)
        for label, r, pfx in (("reuse", rng[2], prefix), ("fresh", rng[3], no_prefix)):
            result = (None, None, None, None) if exact.within_tolerance else conditional(
                r, draws, pfx, moments, args, config, cutoff)
            row.update({f"{label}_{k}": v for k, v in zip(
                ("hard_probability", "conditional_reps", "conditional_pass", "joint_probability"), result)})
        yield row


def run(out, records, source_dir, smoke):
    out.mkdir(parents=True, exist_ok=False)
    source_file = source_dir / "probe_pairs.csv"
    source_seal = json.loads((source_dir / "raw_seal.json").read_text(encoding="utf-8"))["sha256"]
    assert base.digest(source_file) == source_seal[source_file.name]
    sources = [r for r in study.read_csv(source_file) if r["equal_draws"] == str(design_draws) and r["mask"]]
    case_index = {study.key(r): (i, r) for i, r in enumerate(sources, 1)}
    assert len(sources) == len(case_index) > 0
    if not smoke:
        assert len(case_index) == 297
    seal = json.loads((records / "seal.json").read_text(encoding="utf-8"))["sha256"]
    for name in ("splits.json", "rounds.csv"):
        assert base.digest(records / name) == seal[name]
    base.write_json(out / "protocol.json", dict(
        started_utc=base.now(), smoke=smoke, design_draws=design_draws, audit_draws=budgets, audit_seed=audit_seed,
        direct_reps=direct_reps, conditional_reps=conditional_reps, cases=len(sources),
        reused="exact 512 nomination probes plus fresh draws from the same pool, with replacement",
        fresh="full fresh draws from the same pool, with replacement",
        boundary="first attainable hard-assignment flip count exceeding native gamma, with exact drift within chi",
        conditional="sample flip counts conditional on passing the hard gate, then draw from the corresponding "
                    "flip and nonflip empirical pools to estimate passing both gates; fixed prefix on reuse side",
        source_sha256=source_seal[source_file.name], script_sha256=base.digest(Path(__file__)),
        archive_sha256={name: seal[name] for name in ("splits.json", "rounds.csv")}))
    splits = json.loads((records / "splits.json").read_text(encoding="utf-8"))
    archive = {study.key(r): r for r in study.read_csv(records / "rounds.csv") if r["arm"] == "VCIA"}
    rows = []
    for dataset in study.plan["datasets"] if not smoke else study.plan["datasets"][:1]:
        graphs, labels, _, _ = old.dataset(dataset)
        wl, fingerprints = old._wl_features(graphs, 2, 512)
        active_splits = [s for s in splits if s["dataset"] == dataset]
        for split in active_splits[:1] if smoke else active_splits:
            outer, rep, peers = (split[k] for k in ("outer_seed", "rep", "peers"))
            seed = split["init_and_round_seed"]
            train, audit, test = (np.asarray(split[k]) for k in ("train", "audit", "test"))
            for restored, archived in zip(base._split_roles(fingerprints, outer), (train, audit, test), strict=True):
                np.testing.assert_array_equal(restored, archived)
            pilot = old.PilotData(old.representation(wl, None, train, "WL-TFIDF"),
                                  np.full(len(labels), -1), train, audit, test,
                                  [np.asarray(ids) for ids in split["peer_train"]], fingerprints,
                                  outer, split["partition_seed"], split["partition_attempts"],
                                  max(Counter(fingerprints).values()), 6 if dataset == "ENZYMES" else 2,
                                  2, 512, .25)
            config = base.DecisionConfig(**split["decision"])
            initial = old.initial_maps(pilot, seed)
            for topology, condition in product(study.plan["topologies"], study.plan["conditions"]):
                scope = dict(dataset=dataset, outer_seed=outer, rep=rep, peers=peers, q=config.q,
                             topology=topology, condition=condition)
                last = max((r for r in study.plan["replay_rounds"]
                            if study.key(dict(scope, round=r)) in case_index), default=0)
                history = study.start_state(initial)
                for r in range(1, last + 1):
                    row = dict(scope, round=r)
                    k = study.key(row)
                    maps, claims = study.proposal(pilot, history, seed, row)
                    if k in case_index:
                        entry, source = case_index[k]
                        rows.extend(score(row, pilot, history, maps, claims, seed, config, source, entry))
                    history = study.advance_archive(history, maps, claims, archive[k])
            print(f"{dataset} split={outer} rep={rep} peers={peers}: {len(rows)} candidate-budget rows", flush=True)
    assert len(rows) == len(case_index) * len(budgets)
    for n in budgets:
        group = [r for r in rows if r["audit_draws"] == n]
        counts = Counter(r["cohort"] for r in group)
        if not smoke:
            assert counts[scopes[0]] == 250 and counts[scopes[1]] + counts[scopes[2]] == 47
        for dataset in (*study.plan["datasets"], "All"):
            for cohort in (*scopes, "All over-limit"):
                panel = [r for r in group if (dataset == "All" or r["dataset"] == dataset)
                         and (r["cohort"] == cohort or cohort == "All over-limit" and r["cohort"] != scopes[0])]
                if panel:
                    print(n, dataset, cohort, len(panel), *(sum(r[f"{arm}_pass"] for r in panel)
                          for arm in ("reuse", "fresh")), *(np.mean([r[f"{arm}_joint_probability"] for r in panel])
                          for arm in ("reuse", "fresh") if cohort != scopes[0]), flush=True)
    study.save_csv(out / "candidate_audits.csv", rows)
    base.write_json(out / "seal.json", {p.name: base.digest(p) for p in out.iterdir() if p.is_file()})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing the TUDataset ZIPs.")
    parser.add_argument("--closed-loop", required=True, type=Path, help="Sealed output of run_evidence.py.")
    parser.add_argument("--budget-run", required=True, type=Path, help="Sealed output of run_vcia_budget_32768.py.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Use the budget runner's smoke output.")
    args = parser.parse_args()
    old.DATA = args.data
    with threadpool_limits(limits=1):
        run(args.output, args.closed_loop, args.budget_run, args.smoke)
