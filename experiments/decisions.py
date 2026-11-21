"""Matched composition search with the optional D gate specified in the supplement."""

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np

from decision import (_Pool, _composition_effect, _distance, _draw_counts, _full_key,
                      _moments, Candidate, empirical_bernstein_upper)


@dataclass
class Search:
    choice: Candidate | None
    masks: tuple
    unique_probes: int
    setup_s: float
    screen_s: float
    compose_s: float


def search(current, stored, claims, old_claims, changed, ages, draws, config,
           *, policy="VCIA", beta_fraction=None):
    start = perf_counter()
    weights = np.bincount(draws, minlength=current.shape[1])
    indices = np.flatnonzero(weights)
    weights = weights[indices]
    pool = _Pool(current, stored, claims, old_claims, indices, config)
    peers = tuple(range(pool.n_peers))
    setup_s = perf_counter() - start
    start = perf_counter()
    all_current = pool.composed(peers)
    alpha = config.delta_screen / (len(peers) * (1 if policy == "VCIA" else 2))
    detectable, eligible, upper_j = [], [], []
    for i in peers:
        others = tuple(j for j in peers if i != j)
        replaced = pool.composed(others)
        j = _distance(all_current, replaced)
        mean_j, var_j = _moments(j, weights, len(draws))
        detectable.append(bool(changed[i] and mean_j > config.nu))
        uj = empirical_bernstein_upper(mean_j, var_j, pool.range, len(draws), alpha)
        upper_j.append(uj)
        keep = policy == "Unscreened" or uj <= config.kappa
        if keep and beta_fraction is not None:
            keep_others = np.arange(pool.n_peers) != i
            background = pool.cur_weighted[keep_others].sum(axis=0) / pool.cur_weights[keep_others].sum()
            d = _distance(all_current, background) - _distance(replaced, background)
            mean_d, var_d = _moments(d, weights, len(draws))
            ud = min(uj, empirical_bernstein_upper(mean_d, var_d, 2 * pool.range, len(draws), alpha))
            keep = ud <= beta_fraction * config.kappa
        eligible.append(keep)
    screen_s = perf_counter() - start
    start = perf_counter()
    baseline = pool.composed(())
    labels = baseline.argmax(axis=-1)
    capped = np.minimum(ages, config.age_cap)
    masks = tuple(combinations([i for i in peers if eligible[i]], config.q))
    feasible = []
    for mask in masks:
        useful = [i for i in mask if detectable[i]]
        distances, flips = _composition_effect(pool, mask, baseline, labels, weights)
        phi = flips / len(draws)
        ell = float(np.dot(distances, weights) / len(draws))
        if useful and ell <= config.eta * config.chi and phi <= config.eta * config.gamma:
            feasible.append(Candidate(mask, len(useful), float(capped[useful].sum()),
                                      sum(upper_j[i] for i in mask), phi, ell, True, True, None, None, 0))
    choice = min(feasible, key=_full_key, default=None)
    return Search(choice, masks, len(indices), setup_s, screen_s, perf_counter() - start)
