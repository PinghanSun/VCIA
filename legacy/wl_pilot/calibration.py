"""Compute prospective operating characteristics for PCOC verification."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

try:
    from scipy.stats import binom
except ModuleNotFoundError as error:
    if error.name != "scipy":
        raise
    binom = None


VERIFY_DRAWS = (2048, 4096, 8192, 16384, 32768, 40960, 49152, 65536)
TRUE_FLIP_RATES = (0, 0.003, 0.005, 0.008, 0.01, 0.015)
AUDIT_TARGET_SIZES = {
    "MUTAG": 37,
    "PROTEINS": 222,
    "NCI1": 822,
    "ENZYMES": 120,
    "IMDB-BINARY": 200,
}
TOTAL_FAILURE_PROBABILITY = 0.05
SCHEDULED_ROUNDS = 30
STAGES_PER_ROUND = 2
VERIFICATION_COMPONENTS = 2
ASSIGNMENT_CHANGE_TOLERANCE = 0.01
NORMALIZED_DRIFT_TOLERANCE = 0.02


def _left_tail(k: int, draws: int, log_rate: float, log_failure_rate: float) -> float:
    """Sum a binomial left tail downwards from its largest term."""
    log_mass = (
        math.lgamma(draws + 1)
        - math.lgamma(k + 1)
        - math.lgamma(draws - k + 1)
        + k * log_rate
        + (draws - k) * log_failure_rate
    )
    if k == 0:
        return math.exp(log_mass)
    failure_odds = math.exp(log_failure_rate - log_rate)
    relative_mass = 1.0
    relative_terms = [relative_mass]
    for count in range(k, 0, -1):
        relative_mass *= count / (draws - count + 1) * failure_odds
        if relative_mass == 0:
            break
        relative_terms.append(relative_mass)
    return min(1.0, math.exp(log_mass + math.log(math.fsum(relative_terms))))


def _binomial_cdf(k: int, draws: int, rate: float) -> float:
    if k < 0:
        return 0.0
    if k >= draws or rate == 0:
        return 1.0
    if rate == 1:
        return 0.0
    if binom is not None:
        return float(binom.cdf(k, draws, rate))
    log_rate = math.log(rate)
    log_failure_rate = math.log1p(-rate)
    if k >= draws * rate:
        return max(
            0.0, 1.0 - _left_tail(draws - k - 1, draws, log_failure_rate, log_rate)
        )
    return _left_tail(k, draws, log_rate, log_failure_rate)


def binary_acceptance_cutoff(draws: int, gamma: float, alpha: float) -> int:
    """Return the greatest flip count passing the one-sided exact gate."""
    low, high = -1, draws
    while high - low > 1:
        count = (low + high) // 2
        if _binomial_cdf(count, draws, gamma) <= alpha:
            low = count
        else:
            high = count
    return low


def calibrate(
    budgets: Sequence[int] = VERIFY_DRAWS,
    rates: Sequence[float] = TRUE_FLIP_RATES,
    targets: dict[str, int] | None = None,
) -> dict:
    """Evaluate the registered tolerances across verification budgets."""
    if not budgets or any(draws < 2 for draws in budgets):
        raise ValueError("Empirical Bernstein verification requires at least two draws")
    if any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError("True assignment-change rates must belong to [0, 1]")
    if targets is None:
        targets = AUDIT_TARGET_SIZES
    if any(size < 1 for size in targets.values()):
        raise ValueError("Each finite audit target must contain at least one graph")

    stage_delta = TOTAL_FAILURE_PROBABILITY / (SCHEDULED_ROUNDS * STAGES_PER_ROUND)
    alpha = stage_delta / VERIFICATION_COMPONENTS
    gamma = ASSIGNMENT_CHANGE_TOLERANCE
    chi = NORMALIZED_DRIFT_TOLERANCE
    floor_numerator = (7 / 3) * math.log(2 / alpha)
    tolerance_ratio = Fraction(str(gamma))

    finite_targets = {}
    for name, size in targets.items():
        allowed = math.floor(size * tolerance_ratio)
        finite_targets[name] = {
            "graphs": size,
            "census_max_flips": allowed,
            "largest_admissible_rate": allowed / size,
            "first_inadmissible_rate": (allowed + 1) / size,
        }

    rows = []
    for draws in budgets:
        cutoff = binary_acceptance_cutoff(draws, gamma, alpha)
        cdf_at_cutoff = _binomial_cdf(cutoff, draws, gamma)
        cdf_after_cutoff = _binomial_cdf(cutoff + 1, draws, gamma)
        if not cdf_at_cutoff <= alpha < cdf_after_cutoff:
            raise ArithmeticError(
                f"Exact binary cutoff failed verification at m={draws}"
            )
        floor = floor_numerator / (draws - 1)
        rows.append(
            {
                "draws": draws,
                "max_flips": cutoff,
                "binary_acceptance": {
                    str(int(rate) if rate in (0, 1) else rate): _binomial_cdf(
                        cutoff, draws, rate
                    )
                    for rate in rates
                },
                "binary_acceptance_at_finite_boundary": {
                    name: _binomial_cdf(
                        cutoff, draws, target["largest_admissible_rate"]
                    )
                    for name, target in finite_targets.items()
                },
                "cdf_at_cutoff_under_gamma": cdf_at_cutoff,
                "cdf_after_cutoff_under_gamma": cdf_after_cutoff,
                "normalized_eb_floor": floor,
                "max_constant_normalized_drift_passing": chi - floor
                if floor <= chi
                else None,
            }
        )

    return {
        "status": "calculated_operating_characteristics",
        "cdf_backend": "scipy.stats.binom"
        if binom is not None
        else "stdlib log-PMF recurrence",
        "configuration": {
            "total_failure_probability": TOTAL_FAILURE_PROBABILITY,
            "scheduled_rounds": SCHEDULED_ROUNDS,
            "stages_per_round": STAGES_PER_ROUND,
            "verification_components": VERIFICATION_COMPONENTS,
            "stage_delta": stage_delta,
            "alpha_per_verification_component": alpha,
            "assignment_change_tolerance": gamma,
            "normalized_drift_tolerance": chi,
            "minimum_draws_to_pass_with_zero_flips": math.ceil(
                math.log(alpha) / math.log1p(-gamma)
            ),
        },
        "interpretation": [
            "The binary gate assumes independent draws from a fixed verification law after candidate selection.",
            "Binary acceptance probability upper-bounds joint gate acceptance probability.",
            "The largest passing constant drift assumes zero sample variance. Null indicates that zero drift fails.",
            "Finite census thresholds assume a uniform target of distinct graphs and deterministic predictions.",
            "Finite-boundary powers use IID sampling with replacement from that finite target. A census itself is exact on the target.",
        ],
        "finite_targets": finite_targets,
        "rows": rows,
    }


def _print_report(report: dict) -> None:
    def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        widths = [max(map(len, column)) for column in zip(headers, *rows, strict=True)]
        for row in (headers, *rows):
            print(
                "  ".join(
                    value.rjust(width) for value, width in zip(row, widths, strict=True)
                )
            )

    rows = report["rows"]
    rates = rows[0]["binary_acceptance"]
    print(f"CDF backend: {report['cdf_backend']}")
    print("Binary acceptance probabilities and normalized continuous gate limits")
    print_table(
        (
            "Draws",
            "Max flips",
            *(f"p={rate}" for rate in rates),
            "EB floor",
            "Max constant drift",
        ),
        [
            (
                str(row["draws"]),
                str(row["max_flips"]),
                *(
                    f"{probability:.4%}"
                    for probability in row["binary_acceptance"].values()
                ),
                f"{row['normalized_eb_floor']:.6f}",
                "fails at zero"
                if row["max_constant_normalized_drift_passing"] is None
                else f"{row['max_constant_normalized_drift_passing']:.6f}",
            )
            for row in rows
        ],
    )
    print("\nFinite census assignment-change thresholds")
    print_table(
        (
            "Dataset",
            "Graphs",
            "Max flips",
            "Largest passing rate",
            "First failing rate",
        ),
        [
            (
                name,
                str(target["graphs"]),
                str(target["census_max_flips"]),
                f"{target['largest_admissible_rate']:.4%}",
                f"{target['first_inadmissible_rate']:.4%}",
            )
            for name, target in report["finite_targets"].items()
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", nargs="+", type=int, default=VERIFY_DRAWS)
    parser.add_argument("--rates", nargs="+", type=float, default=TRUE_FLIP_RATES)
    parser.add_argument(
        "--target",
        action="append",
        metavar="NAME=SIZE",
        help="Override census target sizes",
    )
    parser.add_argument(
        "--json", type=Path, metavar="PATH", help="Save full calibration as JSON"
    )
    args = parser.parse_args()

    try:
        targets = None
        if args.target:
            targets = {}
            for specification in args.target:
                name, size = specification.split("=", 1)
                if not name.strip() or name in targets:
                    raise ValueError("Audit targets require unique, nonempty names")
                targets[name] = int(size)
        report = calibrate(args.draws, args.rates, targets)
    except ValueError as error:
        parser.error(str(error))

    _print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
