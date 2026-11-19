"""Twenty-round sensitivity, learning-label, and mechanism response scenarios."""

import csv
import itertools
import json
import math
from contextlib import ExitStack
from functools import partial

import numpy as np

from . import early_dynamics as early
from . import kernels as g
from . import late_dynamics as late
from .common import (
    MEASURED_FIELDS,
    POINT_FIELDS,
    REPEATS,
    check,
    condition_endpoints,
    configure,
    load_main_stats,
    make_clean,
    observation,
    paired_interval,
    round_means,
    run_endpoints,
    save_stats,
    write_json,
    write_records,
)
from .costs import run_cost, topology_estimates

SUPPLEMENT_SEED = 2026111513
DEFAULTS = {
    "alpha": 0.5,
    "malicious_fraction": 0.2,
    "audit_size": 128,
    "candidate_budget": 6,
    "threshold": 0.01,
    "noise_scale": 1.0,
    "local_E": 10,
}
AXES = {
    "alpha": [0.1, 0.5, 1, 10],
    "malicious_fraction": [0, 0.1, 0.2, 0.3],
    "audit_size": [32, 64, 128, 256],
    "candidate_budget": [2, 6, 12, 24],
    "threshold": [0.005, 0.01, 0.02, 0.05],
    "noise_scale": [0.6, 1, 1.4],
    "local_E": [5, 10, 20],
}
REUSED = {
    "Full": "PCOC",
    "NoH": "NoH_priority",
    "NoAge": "NoAge_priority",
    "G_only": "G_only",
    "X_only": "X_only",
    "NoVerify": "NoVerify",
}
EXTRA = {
    "StaleNorm": [0.18, 0.20, 0.48, 0.76, 64, 1.12],
    "ReuseV": [0.10, 0.16, 0.42, 0.73, 67, 1.14],
    "NonAtomic": [0.15, 0.22, 0.35, 0.70, 72, 1.20],
    "RandomRank": [0.50, 0.55, 0.30, 0.60, 82, 1.30],
    "Neither_GX": [0, 0.12, 0.05, 0.42, 90, 1.25],
}


def conditions():
    """Preserve source condition order because IDs instantiate the RNG streams."""
    for dataset in g.DS:
        for key, values in AXES.items():
            for value in values:
                yield (
                    "sensitivity",
                    dataset,
                    f"{key}={value}",
                    key,
                    DEFAULTS | {key: value},
                    "A2",
                    "PCOC",
                )
        for alpha, fraction in itertools.product(
            AXES["alpha"], AXES["malicious_fraction"]
        ):
            yield (
                "sensitivity",
                dataset,
                f"alpha={alpha};f={fraction}",
                "alpha*malicious_fraction",
                DEFAULTS | {"alpha": alpha, "malicious_fraction": fraction},
                "A2",
                "PCOC",
            )
        for encoder, rate, count in itertools.product(
            ("GCN64", "GIN64"),
            (0.0001, 0.0003, 0.001, 0.003, 0.01),
            (40, 80, 160, 320),
        ):
            yield (
                "learning",
                dataset,
                f"{encoder};lr={rate};n={count}",
                "encoder*learning_rate*items",
                {
                    "encoder": encoder,
                    "learning_rate": rate,
                    "training_items": count,
                    "local_E": 10,
                },
                "A0",
                "PCOC",
            )
    for dataset, method, attack in itertools.product(g.DS, EXTRA, g.ATK):
        yield (
            "mechanism",
            dataset,
            method,
            "single_switch_or_neither_GX",
            {"local_E": 10},
            attack,
            method,
        )


def paired_rng(condition_id, *key):
    seed = (
        [g.NOISE_SEED, *key] if key[0] == 20 else [SUPPLEMENT_SEED, condition_id, *key]
    )
    return np.random.default_rng(np.random.SeedSequence(seed))


def adjust_scenario(scenario, module, settings, strength):
    task = dict(scenario)
    penalty = 0.0
    pace = noise = 1.0
    if module == "sensitivity":
        heterogeneity = (0.5 / settings["alpha"]) ** 0.20
        task["heterogeneity"] *= heterogeneity
        task["risk"] *= 1 + 0.4 * max(heterogeneity - 1, 0)
        epochs = settings["local_E"]
        pace = (epochs / 10) ** 0.55
        noise = settings["noise_scale"] * (10 / epochs) ** 0.15
        audit_size = settings["audit_size"]
        budget = settings["candidate_budget"]
        threshold = math.log2(settings["threshold"] / 0.01)
        penalty = (
            0.50 * math.log2(128 / audit_size)
            + 0.25 * math.log2(budget / 6) ** 2
            + 0.35 * threshold**2
        )
        noise *= (128 / audit_size) ** 0.10 * (1 + 0.04 * abs(threshold))
        strength *= settings["malicious_fraction"] / 0.2
    elif module == "learning":
        deviation = math.log10(settings["learning_rate"] / 0.001)
        count = settings["training_items"]
        penalty = 1.2 * deviation**2 + 1.4 * max(0, math.log2(160 / count))
        pace = float(np.clip(10 ** (0.3 * deviation), 0.4, 1.8)) * (count / 160) ** 0.08
        noise = (160 / count) ** 0.18 * (1 + 0.25 * max(deviation, 0))
        if settings["encoder"] == "GIN64":
            penalty += 0.20
            noise *= 1.05
            pace *= 1.08
    task["target"] -= penalty
    task["noise"] *= noise
    return task, penalty, pace, noise, strength


def interaction(dataset, attack, main, neither):
    index = list(g.ATK).index(attack)
    arrays = [
        [main[g.METHODS.index(method), index, p, i][0][-1, 0] for p, i in REPEATS]
        for method in ("PCOC", "G_only", "X_only")
    ]
    both, g_only, x_only = np.array(arrays)
    deltas = both - g_only - x_only + neither
    mean, lo, hi = paired_interval(deltas)
    return {
        "dataset": dataset,
        "attack": attack,
        "both_mean_ACC": float(both.mean()),
        "G_only_mean_ACC": float(g_only.mean()),
        "X_only_mean_ACC": float(x_only.mean()),
        "neither_mean_ACC": float(neither.mean()),
        "interaction_pp": float(mean[0]),
        "CI95_low": float(lo[0]),
        "CI95_high": float(hi[0]),
    }


def run(output, main_directory):
    configure(20)
    main_by_dataset = {
        dataset: load_main_stats(main_directory, dataset) for dataset in g.DS
    }
    g.PARAMS.update(EXTRA)
    g.METHODS.extend(EXTRA)
    generated_stats = {dataset: [] for dataset in g.DS}
    source_tasks = g.TASKS.copy()
    condition_rows, runs, params_rows, summaries, means, costs = [], [], [], [], [], []
    interactions = []
    with ExitStack() as stack:
        point_writers = {}
        for dataset in g.DS:
            path = output / f"rounds_{dataset}.csv"
            writer = csv.writer(
                stack.enter_context(path.open("w", encoding="utf-8", newline=""))
            )
            writer.writerow(
                [
                    "condition_id",
                    "module",
                    "dataset",
                    "condition",
                    "method",
                    "attack",
                    "partition",
                    "init",
                    "round",
                    "nominal_E",
                    *POINT_FIELDS,
                    *MEASURED_FIELDS,
                ]
            )
            point_writers[dataset] = writer
        for ci, (module, dataset, label, axis, settings, attack, method) in enumerate(
            conditions(), 1
        ):
            di, mi = g.DS.index(dataset), g.METHODS.index(method)
            g.N = g.OBJECTS[dataset]
            g.TASKS[di], penalty, pace, noise, strength = adjust_scenario(
                source_tasks[di], module, settings, g.ATK[attack]
            )
            g.rng = partial(paired_rng, ci)
            early.SEED, late.SEED = 2026111510 + 1000 + ci, 2026111512 + 1000 + ci
            ids = {
                "condition_id": ci,
                "module": module,
                "dataset": dataset,
                "condition": label,
                "method": method,
                "attack": attack,
            }
            condition_rows.append(
                dict(
                    **ids,
                    axis=axis,
                    settings_json=json.dumps(settings, sort_keys=True),
                    rounds=20,
                    nominal_E=settings["local_E"],
                    generated_evaluation_objects=g.N,
                    target_penalty_pp=penalty,
                    pace_multiplier=pace,
                    noise_multiplier=noise,
                    attack_strength=strength,
                )
            )
            matrix = []
            for partition, init in REPEATS:
                key = (di, mi, partition, init)
                ids_run = dict(**ids, partition=partition, init=init)
                clean, _, risk, heterogeneity, params = make_clean(
                    *key, pace=pace, noise=noise
                )
                latent = g.attacked_curve(clean, *key, strength, risk, heterogeneity)
                values, confusion, nll, probability = observation(latent, key, params)
                combined = np.column_stack([latent, values])
                matrix.append(combined)
                generated_stats[dataset].append(
                    ((ci, partition, init), confusion, nll, values, latent)
                )
                point_writers[dataset].writerows(
                    [
                        ci,
                        module,
                        dataset,
                        label,
                        method,
                        attack,
                        partition,
                        init,
                        t + 1,
                        settings["local_E"],
                        *combined[t],
                        None,
                        None,
                        None,
                    ]
                    for t in range(20)
                )
                if (partition, init) == (1, 1) and ci % 25 == 1:
                    np.savez_compressed(
                        output / f"probabilities_C{ci:03d}.npz", **probability
                    )
                runs.append(
                    dict(
                        **ids_run,
                        nominal_E=settings["local_E"],
                        **run_endpoints(values),
                        **dict.fromkeys(MEASURED_FIELDS),
                    )
                )
                params_rows.append(dict(**ids_run, **params))
                costs.append(
                    run_cost(
                        dict(run_key=f"supp:{ci}:{partition}:{init}", **ids_run),
                        settings,
                    )
                )
            combined = np.asarray(matrix)
            summaries.append(
                dict(
                    **ids,
                    source="generated_supplement",
                    **condition_endpoints(combined[:, :, 1:]),
                )
            )
            means.extend(
                round_means(combined, dict(**ids, source="generated_supplement"))
            )
            if method == "Neither_GX":
                row = interaction(
                    dataset, attack, main_by_dataset[dataset], combined[:, -1, 1]
                )
                interactions.append(row)
            g.TASKS[di] = source_tasks[di]
    for dataset, stats in generated_stats.items():
        save_stats(output / f"stats_{dataset}.npz", stats)
    for dataset, (label, method), (ai, attack) in itertools.product(
        g.DS, REUSED.items(), enumerate(g.ATK)
    ):
        mi = g.METHODS.index(method)
        entries = [main_by_dataset[dataset][mi, ai, p, i] for p, i in REPEATS]
        combined = np.asarray(
            [np.column_stack([latent, values]) for values, latent in entries]
        )
        ids = {
            "condition_id": None,
            "module": "mechanism",
            "dataset": dataset,
            "condition": label,
            "method": method,
            "attack": attack,
            "source": f"reused_main20_{method}",
        }
        summaries.append(dict(**ids, **condition_endpoints(combined[:, :, 1:])))
        means.extend(round_means(combined, ids))
    for filename, rows in (
        ("conditions", condition_rows),
        ("round_means", means),
        ("run_summary", runs),
        ("condition_summary", summaries),
        ("run_parameters", params_rows),
        ("GX_interaction", interactions),
        ("analytic_run_costs", costs),
        ("analytic_topology", topology_estimates()),
    ):
        write_records(output / f"{filename}.csv", rows)
    check(
        len(condition_rows) == 510 and len(runs) == 7650,
        "Supplement20 trajectory coverage changed",
    )
    write_json(
        output / "manifest.json",
        {
            "study": "supplement20",
            "evidence": "generated_probability_response",
            "actual_training": False,
            "new_conditions": len(condition_rows),
            "reused_main_conditions": len(summaries) - len(condition_rows),
            "trajectory_count": len(runs),
            "point_rows": len(runs) * 20,
            "rounds": 20,
            "condition_seed": SUPPLEMENT_SEED,
            "paired_item_seed": g.NOISE_SEED,
            "measured_fields": dict.fromkeys(MEASURED_FIELDS),
        },
    )
