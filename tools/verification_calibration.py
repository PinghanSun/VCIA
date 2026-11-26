"""Exact prospective verification calibration, using Python's standard library.

This computes numerical operating characteristics, not learner experiments or an
observed pilot.  The Clopper--Pearson (CP) gate accepts k flips when
Pr[Binomial(m, gamma) <= k] <= alpha.  Consequently its power at a specified
population flip probability p is Pr[Binomial(m, p) <= k_max].

The empirical-Bernstein (EB) floor is the normalized additive radius at zero
sample variance.  Passing the binary gate does not guarantee passing the EB
gate, so binary power is an upper bound on joint verification power.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


BUDGETS = (2048, 4096, 8192, 16384, 32768, 40960, 49152, 65536)
TOTAL_FAILURE = 0.05
ROUNDS = 30
STAGES = 2
VERIFICATION_COMPONENTS = 2
ALPHA = TOTAL_FAILURE / (ROUNDS * STAGES * VERIFICATION_COMPONENTS)
GAMMA = 0.01
NORMALIZED_CHI = 0.02
POWERS_AT = (0.005, 0.008, 0.01)


def binomial_cdf_forward(n: int, p: float, k: int) -> float:
    """Stable forward recurrence for the budgets and small p used here."""
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    term = math.exp(n * math.log1p(-p))
    if term == 0.0:
        raise ArithmeticError("PMF(0) underflow: use the log-tail implementation")
    terms = [term]
    odds = p / (1.0 - p)
    for j in range(k):
        term *= (n - j) / (j + 1) * odds
        terms.append(term)
    return min(1.0, math.fsum(terms))


def binomial_cdf_log_tail(n: int, p: float, k: int) -> float:
    """Independent QA via log-PMF at k and a downward left-tail recurrence."""
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    log_pmf = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    term = math.exp(log_pmf)
    terms = [term]
    inverse_odds = (1.0 - p) / p
    for j in range(k, 0, -1):
        term *= j / (n - j + 1) * inverse_odds
        terms.append(term)
    return min(1.0, math.fsum(terms))


def max_accepted_count(n: int, gamma: float, alpha: float) -> int:
    """Find largest count satisfying the exact CP upper-bound test."""
    if binomial_cdf_forward(n, gamma, 0) > alpha:
        return -1
    lower, upper = 0, n
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        # Accepted counts are left of the binomial mean for the present alpha.
        # The log-tail method avoids an unnecessarily long forward recurrence.
        if middle >= n * gamma:
            upper = middle
        elif binomial_cdf_log_tail(n, gamma, middle) <= alpha:
            lower = middle
        else:
            upper = middle
    return lower


def calibrate() -> dict:
    log_factor = math.log(2.0 / ALPHA)
    rows = []
    for n in BUDGETS:
        k = max_accepted_count(n, GAMMA, ALPHA)
        at_boundary = binomial_cdf_forward(n, GAMMA, k)
        above_boundary = binomial_cdf_forward(n, GAMMA, k + 1)
        if not at_boundary <= ALPHA < above_boundary:
            raise AssertionError(f"Incorrect CP boundary for m={n}")
        floor = 7.0 * log_factor / (3.0 * (n - 1))
        row = {
            "m": n,
            "k_max": k,
            "normalized_zero_variance_EB_floor": floor,
            "largest_constant_normalized_drift_accepted": max(0.0, NORMALIZED_CHI - floor),
            "cdf_at_gamma_k_max": at_boundary,
            "cdf_at_gamma_k_max_plus_one": above_boundary,
        }
        for p in POWERS_AT:
            forward = binomial_cdf_forward(n, p, k)
            log_tail = binomial_cdf_log_tail(n, p, k)
            # lgamma subtraction loses a few digits at the largest n.
            if not math.isclose(forward, log_tail, rel_tol=2e-9, abs_tol=2e-10):
                raise AssertionError(f"Independent CDF checks differ for m={n}, p={p}")
            row[f"binary_gate_power_p_{p:g}"] = forward
        rows.append(row)
    return {
        "status": "prospective_numerical_calculation_not_measured_pilot",
        "configuration": {
            "total_failure_probability": TOTAL_FAILURE,
            "rounds": ROUNDS,
            "stages_per_round": STAGES,
            "verification_components": VERIFICATION_COMPONENTS,
            "alpha_per_verification_component": ALPHA,
            "gamma": GAMMA,
            "normalized_chi": NORMALIZED_CHI,
            "log_2_over_alpha": log_factor,
            "powers_at_true_flip_probability": list(POWERS_AT),
        },
        "formulas": {
            "k_max": "max{k: BinomialCDF(m, gamma, k) <= alpha}",
            "binary_gate_power": "BinomialCDF(m, p, k_max)",
            "zero_variance_EB_floor": "7 * log(2 / alpha) / (3 * (m - 1))",
        },
        "interpretation": [
            "Scientific tolerances gamma and normalized chi are fixed across rows.",
            "Binary-gate power is an upper bound on joint binary-and-drift acceptance power.",
            "Power at p = gamma is at most alpha, so high power requires a stated interior target.",
            "Empirical design slack does not establish a population margin.",
            "No acceptance-power monotonicity in m is assumed because discrete count thresholds cause sawtooth behavior.",
            "IID sampling is required for the stated CP calculation. A finite-pool census is a separate audit route and target.",
            "Repeated graphs sampled with replacement do not add independent graph coverage.",
            "Joint acceptance power, candidate attrition, content updates and runtime require a separate measured pilot.",
        ],
        "verification": {
            "dependency_free": True,
            "independent_binomial_cdf_implementations": 2,
            "boundary_check": "CDF_gamma(k_max) <= alpha < CDF_gamma(k_max + 1) for every row",
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "qa")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = calibrate()
    json_path = args.output_dir / "verification_calibration.json"
    csv_path = args.output_dir / "verification_calibration.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(result["rows"]), "checks": "passed"}))


if __name__ == "__main__":
    main()
