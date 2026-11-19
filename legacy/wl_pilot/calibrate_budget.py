"""Replay verification budgets on frozen candidates from a completed inner pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import binom
from threadpoolctl import threadpool_limits

from .calibration import VERIFY_DRAWS, binary_acceptance_cutoff
from .decision import clopper_pearson_upper, empirical_bernstein_upper

CHI_GRID = (0.01, 0.02, 0.05)
GAMMA_GRID = (0.005, 0.01, 0.02, 0.05)
OUTCOMES = ("drift", "flip", "joint")
COUNT_KEYS = (
    "requested_graph_operator_pairs",
    "new_graph_operator_pairs",
    "cache_hits",
    "vectorized_prediction_calls",
    "prediction_seconds",
)


@dataclass
class Context:
    metadata: dict
    drift: np.ndarray
    flips: np.ndarray
    unseen_graphs: int
    distinct_verify_operators: int


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def same_float(actual, expected, field):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"Archived {field} mismatch. Recomputed {actual}, logged {expected}"
        )


def locate(ids, audit_ids):
    ordered = np.argsort(audit_ids)
    offsets = np.searchsorted(audit_ids[ordered], ids)
    if np.any(offsets >= len(ordered)) or np.any(audit_ids[ordered[offsets]] != ids):
        raise ValueError("A recorded probe ID is outside the frozen audit pool")
    return ordered[offsets]


def validate_candidate(out, drift, flips, sampled, output_range, alpha, chi, gamma):
    finite = out["finite_audit_law"]
    same_float(drift.mean() * output_range, finite["ell"], "finite audit mean")
    count = int(flips.sum())
    if len(drift) != finite["size"] or count != finite["flips"]:
        raise ValueError("Frozen effects do not reproduce the logged audit law")
    audit_pass = bool(drift.mean() <= chi and count / len(drift) <= gamma)
    if audit_pass != finite["within_tolerance"]:
        raise ValueError("Frozen effects do not reproduce finite audit admissibility")

    verify = out["verification"]
    losses, observed_flips = drift[sampled], int(flips[sampled].sum())
    draws = len(losses)
    eb = empirical_bernstein_upper(losses.mean(), losses.var(ddof=1), 1.0, draws, alpha)
    cp = clopper_pearson_upper(observed_flips, draws, alpha)
    same_float(losses.mean() * output_range, verify["ell"], "sampled drift mean")
    same_float(eb * output_range, verify["upper_ell"], "sampled drift bound")
    same_float(cp, verify["upper_phi"], "sampled flip bound")
    if verify["draws"] != draws or verify["flips"] != observed_flips:
        raise ValueError("Frozen effects do not reproduce logged verification counts")
    if (
        verify["drift_failed"] != (eb > chi)
        or verify["assignment_failed"] != (cp > gamma)
        or verify["accepted"] != (eb <= chi and cp <= gamma)
        or (out["status"] == "accepted") != verify["accepted"]
    ):
        raise ValueError(
            "Frozen effects do not reproduce logged verification disposition"
        )


def load_run(root, policy_names):
    manifest_path = root / "manifest.json"
    index_path = root / "run_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        manifest["probe_law"]
        != "iid uniform with replacement on the reserved finite audit split"
    ):
        raise ValueError("Replay requires the pilot's IID uniform finite audit law")
    output_range = 2 / manifest["learner"]["clusters_known"]
    chi = manifest["decision"]["chi"] / output_range
    gamma = manifest["decision"]["gamma"]
    alpha = manifest["decision"]["delta_verify"] / 2
    contexts, costs, sources = [], [], {}
    inventory = {child["path"]: child for child in index}
    expected_runs = (
        len(manifest["seeds"]) * len(manifest["arms"]) * len(manifest["regimes"])
    )
    if len(inventory) != expected_runs:
        raise ValueError("Pilot run index is missing a scheduled trajectory")
    for trajectory, entry in inventory.items():
        directory = root / trajectory
        effects_path = directory / "finite_effects.npz"
        rounds_path = directory / "rounds.jsonl"
        draws_path = directory / "states_and_draws.npz"
        for path in (effects_path, rounds_path, draws_path):
            sources[path.relative_to(root).as_posix()] = digest(path)
        with np.load(effects_path, allow_pickle=False) as archive:
            audit_ids, rounds = archive["audit_ids"], archive["rounds"]
            names = archive["policy_names"].tolist()
            drifts, flips, present = (
                archive["drifts"] / output_range,
                archive["flips"],
                archive["present"],
            )
        with np.load(draws_path, allow_pickle=False) as archive:
            design_ids, verify_ids = (
                archive["design_probe_ids"],
                archive["verification_probe_ids"],
            )
            current_maps, stored_maps = archive["current_maps"], archive["stored_maps"]
        rows = [
            json.loads(line)
            for line in rounds_path.read_text(encoding="utf-8").splitlines()
        ]
        size, n_rounds = len(audit_ids), entry["rounds"]
        shape = (n_rounds, len(names), size)
        if (
            audit_ids.ndim != 1
            or len(np.unique(audit_ids)) != size
            or size < 1
            or not np.array_equal(rounds, np.arange(1, n_rounds + 1))
            or len(rows) != n_rounds
            or drifts.shape != shape
            or flips.shape != shape
            or present.shape != shape[:2]
            or not np.all(np.isfinite(drifts))
            or drifts.min() < 0
            or drifts.max() > 1 + 1e-12
            or flips.dtype != np.bool_
            or present.dtype != np.bool_
            or len(set(names)) != len(names)
            or set(policy_names).difference(names)
        ):
            raise ValueError(f"Malformed finite audit archive in {directory}")
        for r, row in enumerate(rows):
            if row["round"] != rounds[r] or any(
                row[k] != entry[k] for k in ("regime", "arm", "seed")
            ):
                raise ValueError(f"Mismatched round identity in {directory}")
            metadata = {
                "trajectory": trajectory,
                **{k: row[k] for k in ("seed", "regime", "arm", "round")},
            }
            design = locate(design_ids[r], audit_ids)
            sampled = (
                locate(verify_ids[r], audit_ids)
                if present[r].any()
                else np.empty(0, dtype=int)
            )
            design_unique = len(np.unique(design))
            sampled_unique = len(np.unique(sampled))
            sampled_new = len(np.setdiff1d(sampled, design))
            primary = row["policies"][row["arm"]]
            primary_verify = primary.get("verification") or {}
            costs.append(
                {
                    **metadata,
                    "status": primary["status"],
                    "design_draws": len(design),
                    "design_unique_graphs": design_unique,
                    "verification_draws": primary_verify.get("draws", 0),
                    "verification_unique_graphs": primary_verify.get(
                        "unique_graphs", 0
                    ),
                    "online_selection_compositions": row[
                        "online_selection_compositions"
                    ],
                    "online_design_wall_seconds": row["online_design_wall_seconds"],
                    "online_screen_seconds": row["online_screen_seconds"],
                    "online_verify_and_commit_wall_seconds": primary.get(
                        "verify_and_commit_wall_seconds", 0
                    ),
                    **{f"design_{k}": row["design_cache"][k] for k in COUNT_KEYS},
                    **{
                        f"verification_{k}": primary.get("verify_cache_delta", {}).get(
                            k, 0
                        )
                        for k in COUNT_KEYS
                    },
                }
            )
            for policy in policy_names:
                p = names.index(policy)
                out = row["policies"][policy]
                has_candidate = bool(present[r, p])
                if has_candidate != (out["candidate"] is not None):
                    raise ValueError(
                        f"Candidate presence mismatch in {directory}, round {r + 1}"
                    )
                drift, flip = drifts[r, p], flips[r, p]
                operator_count = 0
                if has_candidate:
                    validate_candidate(
                        out, drift, flip, sampled, output_range, alpha, chi, gamma
                    )
                    operator_count = len(
                        {
                            a.tobytes()
                            for a in (
                                *stored_maps[r],
                                *current_maps[r, out["candidate"]["selected"]],
                            )
                        }
                    )
                    if policy == row["arm"]:
                        if (
                            out["verify_cache_delta"]["new_graph_operator_pairs"]
                            != sampled_new * operator_count
                        ):
                            raise ValueError(
                                "Measured primary cache misses conflict with frozen operator/ID sets"
                            )
                        if out["verification"]["unique_graphs"] != sampled_unique:
                            raise ValueError(
                                "Measured primary verification unique graph count conflicts with draw IDs"
                            )
                contexts.append(
                    Context(
                        {
                            **metadata,
                            "policy": policy,
                            "audit_graphs": size,
                            "has_candidate": has_candidate,
                            "base_actual_status": out["status"],
                            "base_actual_draws": out.get("verification", {}).get(
                                "draws", 0
                            ),
                            "base_actual_closed_loop": policy == row["arm"],
                        },
                        drift,
                        flip,
                        size - design_unique,
                        operator_count,
                    )
                )
    sources[manifest_path.name] = digest(manifest_path)
    sources[index_path.name] = digest(index_path)
    return contexts, costs, (chi, gamma, alpha), sources


def settings_grid(budgets, manifest_draws, chi, gamma, alpha):
    settings = [(m, chi, gamma) for m in budgets]
    sensitivity_draws = manifest_draws if manifest_draws in budgets else budgets[0]
    settings += [(sensitivity_draws, v, gamma) for v in CHI_GRID if v != chi]
    settings += [(sensitivity_draws, chi, v) for v in GAMMA_GRID if v != gamma]
    return [
        {
            "draws": m,
            "chi_normalized": c,
            "gamma": g,
            "alpha": alpha,
            "flip_cutoff": binary_acceptance_cutoff(m, g, alpha),
            "eb_zero_variance_radius": 7 * math.log(2 / alpha) / (3 * (m - 1)),
        }
        for m, c, g in settings
    ]


def estimate(contexts, settings, batches, seed):
    """Use paired IID multinomial batches for a given size and budget."""
    estimates = np.full(
        (len(settings), len(contexts), len(OUTCOMES)), -1, dtype=np.int64
    )
    eligible_by_size, by_budget = defaultdict(list), defaultdict(list)
    for i, ctx in enumerate(contexts):
        if ctx.metadata["has_candidate"]:
            eligible_by_size[len(ctx.drift)].append(i)
    for s, setting in enumerate(settings):
        by_budget[setting["draws"]].append(s)
    for size, indices in eligible_by_size.items():
        # Identical frozen masks often occur under several ranking rules.
        laws = {}
        for i in indices:
            ctx = contexts[i]
            laws.setdefault((ctx.drift.tobytes(), ctx.flips.tobytes()), []).append(i)
        law_groups = list(laws.values())
        representatives = [contexts[group[0]] for group in law_groups]
        drift = np.column_stack([ctx.drift for ctx in representatives])
        flips = np.column_stack([ctx.flips for ctx in representatives]).astype(float)
        means = drift.mean(axis=0)
        centered = drift - means
        squared = centered**2
        probability = np.full(size, 1 / size)
        for draws, setting_indices in by_budget.items():
            rng = np.random.default_rng(np.random.SeedSequence([seed, size, draws]))
            totals = np.zeros(
                (len(setting_indices), len(representatives), len(OUTCOMES)),
                dtype=np.int64,
            )
            for start in range(0, batches, 256):
                sample = rng.multinomial(
                    draws, probability, size=min(batches - start, 256)
                ).astype(float)
                deviation = (sample @ centered) / draws
                variance = np.maximum(
                    0, ((sample @ squared) - draws * deviation**2) / (draws - 1)
                )
                flip_count = sample @ flips
                log_term = math.log(2 / settings[setting_indices[0]]["alpha"])
                upper = np.minimum(
                    1.0,
                    means
                    + deviation
                    + np.sqrt(2 * variance * log_term / draws)
                    + settings[setting_indices[0]]["eb_zero_variance_radius"],
                )
                for k, s in enumerate(setting_indices):
                    drift_accepted = upper <= settings[s]["chi_normalized"]
                    flip_accepted = flip_count <= settings[s]["flip_cutoff"]
                    totals[k, :, :] += np.column_stack(
                        (
                            drift_accepted.sum(axis=0),
                            flip_accepted.sum(axis=0),
                            (drift_accepted & flip_accepted).sum(axis=0),
                        )
                    )
            for k, s in enumerate(setting_indices):
                for law, group in enumerate(law_groups):
                    estimates[s, group, :] = totals[k, law]
    return estimates


def grid_rows(contexts, settings, estimates, batches):
    for s, setting in enumerate(settings):
        draws, chi, gamma = (setting[k] for k in ("draws", "chi_normalized", "gamma"))
        for i, ctx in enumerate(contexts):
            size = len(ctx.drift)
            allowed = math.floor(size * Fraction(str(gamma)))
            row = {
                **ctx.metadata,
                **setting,
                "census_max_flips": allowed,
                "candidate_true_normalized_drift": None,
                "candidate_true_flips": None,
                "candidate_true_flip_rate": None,
                "drift_margin": None,
                "flip_margin": None,
                "census_pass": None,
                "binary_exact_pass_probability": None,
                "mc_batches": 0,
                **{f"{key}_mc_passes": None for key in OUTCOMES},
                "expected_unique_graphs": 0,
                "expected_additional_graph_operator_evaluations": 0,
            }
            if ctx.metadata["has_candidate"]:
                mean, flips = float(ctx.drift.mean()), int(ctx.flips.sum())
                coverage = (
                    1 if size == 1 else -math.expm1(draws * math.log1p(-1 / size))
                )
                row.update(
                    {
                        "candidate_true_normalized_drift": mean,
                        "candidate_true_flips": flips,
                        "candidate_true_flip_rate": flips / size,
                        "drift_margin": chi - mean,
                        "flip_margin": gamma - flips / size,
                        "census_pass": mean <= chi and flips <= allowed,
                        "binary_exact_pass_probability": float(
                            binom.cdf(setting["flip_cutoff"], draws, flips / size)
                        ),
                        "mc_batches": batches,
                        **{
                            f"{key}_mc_passes": int(count)
                            for key, count in zip(
                                OUTCOMES, estimates[s, i], strict=True
                            )
                        },
                        "expected_unique_graphs": size * coverage,
                        "expected_additional_graph_operator_evaluations": (
                            ctx.unseen_graphs * ctx.distinct_verify_operators * coverage
                        ),
                    }
                )
            yield row


def summarize(all_rows, batches):
    dimensions = ("regime", "arm", "policy", "draws", "chi_normalized", "gamma")
    groups = defaultdict(list)
    for row in all_rows:
        groups[tuple(row[k] for k in dimensions)].append(row)
    for keys, rows in groups.items():
        candidates = [r for r in rows if r["has_candidate"]]
        admitted = [r for r in candidates if r["census_pass"]]
        outside = [r for r in candidates if not r["census_pass"]]
        joint_passes = sum(r["joint_mc_passes"] for r in candidates)
        outside_passes = sum(r["joint_mc_passes"] for r in outside)
        yield {
            **dict(zip(dimensions, keys, strict=True)),
            "scheduled_contexts": len(rows),
            "candidate_contexts": len(candidates),
            "census_admissible_contexts": len(admitted),
            "census_outside_contexts": len(outside),
            "base_actual_passed_contexts": sum(
                r["base_actual_status"] == "accepted" for r in rows
            ),
            "mean_conditional_binary_exact": (
                sum(r["binary_exact_pass_probability"] for r in candidates)
                / len(candidates)
                if candidates
                else None
            ),
            **{
                f"mean_conditional_{outcome}_mc": (
                    sum(r[f"{outcome}_mc_passes"] for r in candidates)
                    / (batches * len(candidates))
                    if candidates
                    else None
                )
                for outcome in OUTCOMES
            },
            "mean_joint_mc_given_census_admissible": (
                sum(r["joint_mc_passes"] for r in admitted) / (batches * len(admitted))
                if admitted
                else None
            ),
            "mean_joint_mc_given_census_outside": outside_passes
            / (batches * len(outside))
            if outside
            else None,
            "mean_joint_mc_all_scheduled": joint_passes / (batches * len(rows)),
            "mean_outside_joint_mc_all_scheduled": outside_passes
            / (batches * len(rows)),
            "mean_expected_unique_graphs_per_round": sum(
                r["expected_unique_graphs"] for r in rows
            )
            / len(rows),
            "mean_expected_additional_graph_operator_evaluations_per_round": (
                sum(r["expected_additional_graph_operator_evaluations"] for r in rows)
                / len(rows)
            ),
        }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--policies", nargs="+", default=["PCOC"])
    parser.add_argument("--draws", nargs="+", type=int, default=VERIFY_DRAWS)
    parser.add_argument("--batches", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=904311)
    args = parser.parse_args()
    if (
        min(args.draws) < 2
        or args.batches < 1
        or args.seed < 0
        or len(set(args.draws)) != len(args.draws)
        or len(set(args.policies)) != len(args.policies)
    ):
        parser.error(
            "Require distinct budgets >=2, distinct policies, positive batch count and nonnegative seed"
        )
    if args.out.exists():
        parser.error("Choose a new output directory")
    started = perf_counter()
    contexts, costs, (chi, gamma, alpha), hashes = load_run(
        args.run_root, args.policies
    )
    manifest = json.loads((args.run_root / "manifest.json").read_text(encoding="utf-8"))
    settings = settings_grid(args.draws, manifest["verify_draws"], chi, gamma, alpha)
    with threadpool_limits(limits=1):
        estimates = estimate(contexts, settings, args.batches, args.seed)
    rows = list(grid_rows(contexts, settings, estimates, args.batches))
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "fixed_candidate_grid.csv", rows)
    write_csv(args.out / "replay_summary.csv", list(summarize(rows, args.batches)))
    write_csv(args.out / "actual_online_cost.csv", costs)
    report = {
        "evidence_scope": "fixed-candidate, empirical finite audit law from completed exploratory inner learning",
        "run_root": args.run_root.name,
        "input_sha256": hashes,
        "numpy": np.__version__,
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("calibrate_budget.py", "calibration.py", "decision.py")
        },
        "policies": args.policies,
        "contexts": len(contexts),
        "candidate_contexts": int((estimates[0, :, 0] >= 0).sum()),
        "batches_per_context_setting": args.batches,
        "seed": args.seed,
        "settings": settings,
        "replay_wall_seconds": perf_counter() - started,
        "interpretation": [
            "Fixed masks and trajectories were learned at the recorded base setting. Replay changes only the verification gate. It does not change design eligibility, change which mask is selected, or generate alternative learning trajectories.",
            "A scheduled hold with no candidate makes no verification draws or inference requests. Per-candidate rates omit holds. Mean rates over scheduled contexts include holds.",
            "Census scores are exact for the uniform archived audit split. Binary acceptance probabilities are exact binomial values on that split. Drift and joint counts are Monte Carlo operating characteristics using complete paired per-graph effects, with independent batches within a fixed context and common batches across contexts of the same audit size and budget.",
            "Monte Carlo batches are not learner runs or independent across contexts. A zero observed Monte Carlo numerator is not a proof that the true acceptance probability is zero.",
            "Expected unique graphs and additional graph/operator evaluations use finite-pool coverage and the measured intra-round design cache footprint. Model-forward and decision timings in actual_online_cost.csv come only from the base online setting, exclude offline shadow work, and apply only to this CPU implementation.",
            "Tolerance sensitivity varies one verification tolerance at a time at the base draw count (or the first requested draw count if the base budget is unavailable). Neither design slack nor a threshold sweep is an inferred population margin or confirmatory retuning.",
        ],
    }
    (args.out / "replay_metadata.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("contexts", "candidate_contexts", "replay_wall_seconds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
