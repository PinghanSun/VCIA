"""Main 20-round generated-object numerical benchmark."""

import csv
import itertools

import numpy as np

from . import kernels as g
from .common import (
    CORE_PROFILES,
    MEASURED_FIELDS,
    POINT_FIELDS,
    REPEATS,
    check,
    condition_endpoints,
    configure,
    make_clean,
    observation,
    paired_interval,
    round_means,
    run_endpoints,
    save_stats,
    write_json,
    write_records,
)
from .costs import run_cost
from .recovery import recovery


def comparisons(dataset, by_condition):
    rows = []
    for attack, method, endpoint in itertools.product(
        g.ATK, g.METHODS[1:], ("round20", "tail5")
    ):
        pcoc = np.asarray([item[:, 1:] for item in by_condition["PCOC", attack]])
        comparator = np.asarray([item[:, 1:] for item in by_condition[method, attack]])
        differences = pcoc - comparator
        deltas = (
            differences[:, -1] if endpoint == "round20" else differences[:, -5:].mean(1)
        )
        center, lower, upper = paired_interval(deltas)
        for metric, mean, lo, hi in zip(g.METRICS, center, lower, upper):
            rows.append(
                {
                    "dataset": dataset,
                    "attack": attack,
                    "comparator": method,
                    "endpoint": endpoint,
                    "metric": metric,
                    "delta_PCOC_minus_comparator": float(mean),
                    "CI95_lower": float(lo),
                    "CI95_upper": float(hi),
                    "CI95_contains_zero": bool(lo <= 0 <= hi),
                    "independent_partitions": 3,
                    "inits_per_partition": 5,
                }
            )
    return rows


def run(output):
    configure(20)
    condition_rows, means, run_rows, parameters = [], [], [], []
    paired, recovery_rows, costs = [], [], []
    point_count = 0
    for di, dataset in enumerate(g.DS):
        g.N = g.OBJECTS[dataset]
        stats, by_condition = [], {}
        with (output / f"rounds_{dataset}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "method",
                    "attack",
                    "partition",
                    "init",
                    "round",
                    *POINT_FIELDS,
                    "toy_hold",
                    *MEASURED_FIELDS,
                ]
            )
            for mi, method in enumerate(g.METHODS):
                for partition, init in REPEATS:
                    key = (di, mi, partition, init)
                    ids = {
                        "dataset": dataset,
                        "method": method,
                        "partition": partition,
                        "init": init,
                    }
                    clean, holds, risk, heterogeneity, params = make_clean(*key)
                    parameters.append(dict(**ids, **params))
                    for ai, (attack, strength) in enumerate(g.ATK.items()):
                        ids_run = dict(**ids, attack=attack)
                        latent = g.attacked_curve(
                            clean, *key, strength, risk, heterogeneity
                        )
                        values, confusion, nll, probability = observation(
                            latent, key, params
                        )
                        stats.append(
                            ((mi, ai, partition, init), confusion, nll, values, latent)
                        )
                        combined = np.column_stack([latent, values])
                        by_condition.setdefault((method, attack), []).append(combined)
                        writer.writerows(
                            [
                                method,
                                attack,
                                partition,
                                init,
                                t + 1,
                                *combined[t],
                                int(holds[t]),
                                None,
                                None,
                                None,
                            ]
                            for t in range(20)
                        )
                        point_count += 20
                        if (
                            method in CORE_PROFILES
                            and (partition, init) == (1, 1)
                            and attack in {"A0", "A2"}
                        ):
                            np.savez_compressed(
                                output
                                / f"probabilities_{dataset}_{method}_{attack}.npz",
                                **probability,
                            )
                        if attack == "A0":
                            status = {
                                "attack_tail_loss_pp": None,
                                "recovery_start": None,
                                "recovery_confirm": None,
                                "recovery_status": "not_applicable",
                            }
                        else:
                            result = recovery(
                                clean, latent, method, "stop", "latent", 20
                            )
                            status = {
                                name: result[name]
                                for name in (
                                    "attack_tail_loss_pp",
                                    "recovery_start",
                                    "recovery_confirm",
                                    "recovery_status",
                                )
                            }
                        recovery_rows.append(dict(**ids_run, **status))
                        acc = values[:, 0]
                        run_rows.append(
                            dict(
                                **ids_run,
                                **run_endpoints(values),
                                ACC_decline_rounds=int((np.diff(acc) < 0).sum()),
                                ACC_max_drawdown_pp=float(
                                    (np.maximum.accumulate(acc) - acc).max()
                                ),
                                **status,
                                nominal_E=10,
                                **dict.fromkeys(MEASURED_FIELDS),
                            )
                        )
                        run_key = f"main:{dataset}:{method}:{attack}:{partition}:{init}"
                        costs.append(
                            run_cost(
                                dict(run_key=run_key, module="main", **ids_run), {}
                            )
                        )
        save_stats(output / f"stats_{dataset}.npz", stats)
        for (method, attack), combined in by_condition.items():
            identifiers = {"dataset": dataset, "method": method, "attack": attack}
            means.extend(round_means(combined, identifiers))
            condition_rows.append(
                dict(
                    **identifiers, **condition_endpoints(np.asarray(combined)[:, :, 1:])
                )
            )
        paired.extend(comparisons(dataset, by_condition))
    for filename, rows in (
        ("round_means", means),
        ("run_summary", run_rows),
        ("condition_summary", condition_rows),
        ("paired_comparisons", paired),
        ("run_parameters", parameters),
        ("latent_recovery", recovery_rows),
        ("analytic_run_costs", costs),
    ):
        write_records(output / f"{filename}.csv", rows)
    check(
        len(run_rows) == 8100 and point_count == 162000,
        "Main20 trajectory coverage changed",
    )
    write_json(
        output / "manifest.json",
        {
            "study": "main20",
            "evidence": "generated_probability_response",
            "actual_training": False,
            "trajectory_count": len(run_rows),
            "point_rows": point_count,
            "rounds": 20,
            "seed": g.SEED,
            "noise_seed": g.NOISE_SEED,
            "attack_rounds": [5, 12],
            "early_dynamics_seed": 2026111510,
            "late_dynamics_seed": 2026111512,
            "methods": list(g.METHODS),
            "scenarios": list(g.DS),
            "attack_strengths": g.ATK,
            "independent_partitions": 3,
            "initializations_per_partition": 5,
            "metrics": g.METRICS,
            "measured_fields": dict.fromkeys(MEASURED_FIELDS),
        },
    )
