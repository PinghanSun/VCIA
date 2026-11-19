"""Version screening, exact-quota composition, and one-shot audit for PCOC."""

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class DecisionConfig:
    q: int
    kappa: float = 0.10
    chi: float = 0.02
    gamma: float = 0.01
    nu: float = 1e-4
    eta: float = 0.8
    age_cap: int = 30
    epsilon: float = 1e-6
    claim_cap: float = 1.0
    delta_screen: float = 1 / 1200
    delta_verify: float = 1 / 1200

    @classmethod
    def normalized(cls, q, *, n_clusters, kappa=0.10, chi=0.02, nu=1e-4, **kwargs):
        """Construct a configuration from tolerances expressed as fractions of 2/K."""
        output_range = 2 / n_clusters
        return cls(
            q,
            kappa=kappa * output_range,
            chi=chi * output_range,
            nu=nu * output_range,
            **kwargs,
        )

    def validate(self, n_peers, n_clusters):
        output_range = 2 / n_clusters
        if not isinstance(self.q, (int, np.integer)) or not 1 <= self.q <= n_peers:
            raise ValueError("q must be an integer between 1 and the number of peers")
        if not (
            0 < self.chi <= output_range and 0 <= self.nu < self.kappa <= output_range
        ):
            raise ValueError("kappa, chi, and nu must be in the native d_K range")
        if not (
            0 < self.gamma < 1
            and 0 < self.eta <= 1
            and 0 < self.delta_screen < 1
            and 0 < self.delta_verify < 1
        ):
            raise ValueError(
                "gamma and the failure budgets must be in (0, 1), eta in (0, 1]"
            )
        if not (
            np.isfinite(self.epsilon)
            and self.epsilon > 0
            and np.isfinite(self.claim_cap)
            and self.claim_cap >= 0
            and np.isfinite(self.age_cap)
            and self.age_cap > 0
        ):
            raise ValueError(
                "epsilon and age cap must be positive and claim cap must be nonnegative"
            )


@dataclass(frozen=True)
class VersionStats:
    identity: int
    mean_J: float
    var_J: float
    upper_J: float
    mean_D: float
    var_D: float
    changed: bool
    detectable: bool
    eligible: bool


@dataclass(frozen=True)
class Candidate:
    selected: tuple[int, ...]
    H: int
    A: float
    U: float
    phi: float
    ell: float
    screened: bool
    feasible: bool
    rejection: str | None
    unscreened_rejection: str | None
    seconds: float


@dataclass(frozen=True)
class DesignResult:
    version_stats: tuple[VersionStats, ...]
    candidates: tuple[Candidate, ...]
    selections: dict[str, Candidate | None]
    hold_reasons: dict[str, str | None]
    draws: int
    unique_graphs: int
    count_active: bool
    age_active: bool
    screen_seconds: float
    composition_seconds: float
    screened_composition_seconds: float
    unscreened_composition_seconds: float
    seconds: float


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    ell: float
    phi: float
    upper_ell: float
    upper_phi: float
    flips: int
    draws: int
    unique_graphs: int
    drift_failed: bool
    assignment_failed: bool
    seconds: float


@dataclass(frozen=True)
class ExactResult:
    ell: float
    phi: float
    flips: int
    size: int
    within_tolerance: bool


class _Pool:
    def __init__(self, current, stored, current_claims, stored_claims, indices, config):
        current = np.asarray(current)
        stored = np.asarray(stored)
        if current.ndim != 3 or current.shape != stored.shape:
            raise ValueError(
                "current and stored must share the (peer, graph, cluster) shape"
            )
        self.n_peers, self.n_graphs, self.n_clusters = current.shape
        if self.n_peers < 2 or self.n_graphs < 1 or self.n_clusters < 2:
            raise ValueError(
                "the pool needs at least two peers, one graph, and two clusters"
            )
        config.validate(self.n_peers, self.n_clusters)
        if indices is not None:
            current, stored = current[:, indices], stored[:, indices]
        current = np.asarray(current, dtype=np.float64)
        stored = np.asarray(stored, dtype=np.float64)
        for outputs in (current, stored):
            if (
                not np.all(np.isfinite(outputs))
                or np.any(outputs < 0)
                or not np.allclose(outputs.sum(axis=-1), 1.0, rtol=0.0, atol=5e-7)
            ):
                raise ValueError(
                    "each accessed peer output must be a probability vector"
                )
        self.cur_weights = self._weights(current_claims, config)
        self.mem_weights = self._weights(stored_claims, config)
        self.cur_weighted = self.cur_weights[:, None, None] * current
        self.mem_weighted = self.mem_weights[:, None, None] * stored
        self.range = 2 / self.n_clusters

    def _weights(self, claims, config):
        claims = np.asarray(claims, dtype=np.float64)
        if (
            claims.shape != (self.n_peers,)
            or np.any(claims < 0)
            or not np.all(np.isfinite(claims))
        ):
            raise ValueError("each peer must have a finite nonnegative claim")
        return config.epsilon + np.minimum(claims, config.claim_cap)

    def composed(self, selected):
        take_current = np.zeros(self.n_peers, dtype=bool)
        take_current[list(selected)] = True
        weighted = np.where(
            take_current[:, None, None], self.cur_weighted, self.mem_weighted
        )
        weights = np.where(take_current, self.cur_weights, self.mem_weights)
        return weighted.sum(axis=0) / weights.sum()

    def version_effect(self, identity, all_current, available):
        other = available & (np.arange(self.n_peers) != identity)
        if not other.any():
            return _distance(
                all_current, self.mem_weighted[identity] / self.mem_weights[identity]
            ), np.zeros(len(all_current))
        numerator, denominator = (
            self.cur_weighted[other].sum(axis=0),
            self.cur_weights[other].sum(),
        )
        background = numerator / denominator
        counterfactual = (numerator + self.mem_weighted[identity]) / (
            denominator + self.mem_weights[identity]
        )
        J = _distance(all_current, counterfactual)
        D = _distance(all_current, background) - _distance(counterfactual, background)
        return J, D


def _selected(candidate, n_peers):
    values = (
        candidate.selected if isinstance(candidate, Candidate) else tuple(candidate)
    )
    if any(
        not isinstance(i, (int, np.integer)) or not 0 <= i < n_peers for i in values
    ) or len(set(values)) != len(values):
        raise ValueError("selected identities must be distinct zero-based peer indices")
    return values


def aggregate(current, stored, current_claims, stored_claims, selected, config):
    """Return the aggregate on every supplied graph for a fixed version mask."""
    pool = _Pool(current, stored, current_claims, stored_claims, None, config)
    return pool.composed(_selected(selected, pool.n_peers))


def _draw_counts(indices, n_graphs):
    draws = np.asarray(indices)
    if (
        draws.ndim != 1
        or len(draws) < 2
        or not np.issubdtype(draws.dtype, np.integer)
        or np.any(draws < 0)
        or np.any(draws >= n_graphs)
    ):
        raise ValueError("probe draws must be at least two valid graph indices")
    return np.unique(draws, return_counts=True)


def _distance(left, right):
    return np.abs(left - right).mean(axis=-1)


def _moments(values, counts, draws):
    mean = float(np.dot(values, counts) / draws)
    variance = float(np.dot((values - mean) ** 2, counts) / (draws - 1))
    return mean, variance


def empirical_bernstein_upper(mean, variance, output_range, draws, alpha):
    log_term = np.log(2 / alpha)
    radius = np.sqrt(2 * variance * log_term / draws) + 7 * output_range * log_term / (
        3 * (draws - 1)
    )
    return float(min(output_range, mean + radius))


def clopper_pearson_upper(flips, draws, alpha):
    if flips == draws:
        return 1.0
    if flips == 0:
        return float(-np.expm1(np.log(alpha) / draws))
    return float(beta.isf(alpha, flips + 1, draws - flips))


def _composition_effect(pool, selected, baseline, baseline_labels, counts):
    output = pool.composed(selected)
    distances = _distance(output, baseline)
    flips = int(np.dot(output.argmax(axis=-1) != baseline_labels, counts))
    return distances, flips


def _full_key(candidate):
    return (
        -candidate.H,
        -candidate.A,
        candidate.phi,
        candidate.ell,
        candidate.selected,
    )


def design(
    current,
    stored,
    current_claims,
    stored_claims,
    changed,
    ages,
    design_idx,
    config,
    *,
    random_seed=0,
    screen_only=False,
    rank_full=False,
    individual_only=False,
    available=None,
):
    """Screen and rank immutable masks using the design draws alone."""
    start = perf_counter()
    indices, counts = _draw_counts(design_idx, np.shape(current)[1])
    draws = int(counts.sum())
    pool = _Pool(current, stored, current_claims, stored_claims, indices, config)
    changed = np.asarray(changed)
    ages = np.asarray(ages, dtype=np.float64)
    if changed.shape != (pool.n_peers,) or changed.dtype != np.dtype(bool):
        raise ValueError("changed must be a boolean array indexed by peer")
    if (
        ages.shape != (pool.n_peers,)
        or not np.all(np.isfinite(ages))
        or np.any(ages < 0)
    ):
        raise ValueError("ages must be finite, nonnegative, and indexed by peer")
    peer_ids = tuple(range(pool.n_peers))
    available = (
        np.ones(pool.n_peers, dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if available.shape != (pool.n_peers,) or available.dtype != np.dtype(bool):
        raise ValueError("available must be a boolean array indexed by peer")
    if screen_only and rank_full:
        raise ValueError("a full-family ranker requires full enumeration")
    capped_ages = np.minimum(ages, config.age_cap)

    screen_start = perf_counter()
    offered = tuple(i for i in peer_ids if available[i])
    all_current = (
        pool.cur_weighted[available].sum(axis=0) / pool.cur_weights[available].sum()
        if offered
        else None
    )
    version_stats = []
    for identity in peer_ids:
        if not available[identity]:
            version_stats.append(
                VersionStats(
                    identity, 0.0, 0.0, pool.range, 0.0, 0.0, False, False, False
                )
            )
            continue
        J, D = pool.version_effect(identity, all_current, available)
        mean_J, var_J = _moments(J, counts, draws)
        mean_D, var_D = _moments(D, counts, draws)
        upper_J = empirical_bernstein_upper(
            mean_J,
            var_J,
            pool.range,
            draws,
            config.delta_screen / len(offered),
        )
        version_stats.append(
            VersionStats(
                identity,
                mean_J,
                var_J,
                upper_J,
                mean_D,
                var_D,
                bool(changed[identity]),
                bool(available[identity] and changed[identity] and mean_J > config.nu),
                bool(available[identity] and upper_J <= config.kappa),
            )
        )
    screen_seconds = perf_counter() - screen_start

    composition_start = perf_counter()
    baseline = pool.composed(())
    labels = baseline.argmax(axis=-1)
    candidates = []
    masks = (
        combinations(offered, config.q)
        if len(offered) >= max(2, config.q) and not individual_only
        else ()
    )
    for selected in masks:
        screened = all(version_stats[i].eligible for i in selected)
        if screen_only and not screened:
            continue
        candidate_start = perf_counter()
        detectable = [i for i in selected if version_stats[i].detectable]
        H = len(detectable)
        A = float(capped_ages[detectable].sum())
        U = sum(version_stats[i].upper_J for i in selected)
        distances, flips = _composition_effect(pool, selected, baseline, labels, counts)
        ell = float(np.dot(distances, counts) / draws)
        phi = flips / draws
        drift_failed = ell > config.eta * config.chi
        assignment_failed = phi > config.eta * config.gamma
        feasible = not (drift_failed or assignment_failed)
        if drift_failed and assignment_failed:
            unscreened_rejection = "design_drift_and_assignment"
        elif drift_failed:
            unscreened_rejection = "design_drift"
        elif assignment_failed:
            unscreened_rejection = "design_assignment"
        elif H == 0:
            unscreened_rejection = "no_detectable_update"
        else:
            unscreened_rejection = None
        rejection = unscreened_rejection if screened else "individual_screen"
        candidates.append(
            Candidate(
                selected,
                H,
                A,
                U,
                phi,
                ell,
                screened,
                feasible,
                rejection,
                unscreened_rejection,
                perf_counter() - candidate_start,
            )
        )
    composition_seconds = perf_counter() - composition_start

    screened = [item for item in candidates if item.rejection is None]
    unscreened = [item for item in candidates if item.unscreened_rejection is None]
    family = unscreened if rank_full else screened
    selectors = {
        "PCOC": _full_key,
        "Drift-First": lambda c: (c.ell, c.phi, -c.H, -c.A, c.selected),
        "J-Rank-Feasible": lambda c: (c.U, c.selected),
        "No-Count": lambda c: (-c.A, c.phi, c.ell, c.selected),
        "No-Age": lambda c: (-c.H, c.phi, c.ell, c.selected),
    }
    selections = {
        "PCOC": min(screened, key=_full_key, default=None),
        "Unscreened": min(unscreened, key=_full_key, default=None),
    }
    selections.update(
        {
            name: min(family, key=key, default=None)
            for name, key in selectors.items()
            if name != "PCOC"
        }
    )
    selections["Random-Feasible"] = (
        family[int(np.random.default_rng(random_seed).integers(len(family)))]
        if family
        else None
    )
    H_max = max((item.H for item in family), default=0)
    count_active = len({item.H for item in family}) > 1
    age_active = len({item.A for item in family if item.H == H_max}) > 1

    if len(offered) < max(2, config.q):
        screened_hold = "insufficient_available_peers"
    elif len([item for item in version_stats if item.eligible]) < config.q:
        screened_hold = "insufficient_eligible_peers"
    elif not any(item.screened and item.feasible for item in candidates):
        screened_hold = "no_feasible_composition"
    else:
        screened_hold = "no_detectable_update"
    unscreened_hold = (
        "insufficient_available_peers"
        if len(offered) < max(2, config.q)
        else (
            "no_detectable_update"
            if any(item.feasible for item in candidates)
            else "no_feasible_composition"
        )
    )
    hold_reasons = {
        name: None
        if choice is not None
        else (
            unscreened_hold
            if name == "Unscreened" or (rank_full and name != "PCOC")
            else screened_hold
        )
        for name, choice in selections.items()
    }
    return DesignResult(
        tuple(version_stats),
        tuple(candidates),
        selections,
        hold_reasons,
        draws,
        len(indices),
        count_active,
        age_active,
        screen_seconds,
        composition_seconds,
        sum(item.seconds for item in candidates if item.screened),
        sum(item.seconds for item in candidates),
        perf_counter() - start,
    )


def verify(
    current, stored, current_claims, stored_claims, candidate, verify_idx, config
):
    """Audit one already-fixed composition on an independently drawn batch."""
    start = perf_counter()
    indices, counts = _draw_counts(verify_idx, np.shape(current)[1])
    draws = int(counts.sum())
    pool = _Pool(current, stored, current_claims, stored_claims, indices, config)
    selected = _selected(candidate, pool.n_peers)
    if len(selected) != config.q or (
        isinstance(candidate, Candidate) and candidate.H < 1
    ):
        raise ValueError(
            "verification requires one selected, positive-update exact-quota candidate"
        )
    baseline = pool.composed(())
    distances, flips = _composition_effect(
        pool, selected, baseline, baseline.argmax(axis=-1), counts
    )
    ell, variance = _moments(distances, counts, draws)
    alpha = config.delta_verify / 2
    upper_ell = empirical_bernstein_upper(ell, variance, pool.range, draws, alpha)
    upper_phi = clopper_pearson_upper(flips, draws, alpha)
    drift_failed = upper_ell > config.chi
    assignment_failed = upper_phi > config.gamma
    return VerificationResult(
        not (drift_failed or assignment_failed),
        ell,
        flips / draws,
        upper_ell,
        upper_phi,
        flips,
        draws,
        len(indices),
        drift_failed,
        assignment_failed,
        perf_counter() - start,
    )


class Census:
    """Uniform finite-target scores for a fixed, independently evaluated pool."""

    def __init__(self, current, stored, current_claims, stored_claims, config):
        self.pool = _Pool(current, stored, current_claims, stored_claims, None, config)
        self.baseline = self.pool.composed(())
        self.labels = self.baseline.argmax(axis=-1)
        self.config = config

    def evidence(self, candidate):
        selected = _selected(candidate, self.pool.n_peers)
        output = self.pool.composed(selected)
        return _distance(output, self.baseline), output.argmax(axis=-1) != self.labels

    def score(self, candidate):
        distances, flips = self.evidence(candidate)
        ell, phi = float(distances.mean()), float(flips.mean())
        return ExactResult(
            ell,
            phi,
            int(flips.sum()),
            len(flips),
            ell <= self.config.chi and phi <= self.config.gamma,
        )


def exact(current, stored, current_claims, stored_claims, candidate, config):
    """Score a fixed mask under the uniform law on all supplied audit graphs."""
    return Census(current, stored, current_claims, stored_claims, config).score(
        candidate
    )
