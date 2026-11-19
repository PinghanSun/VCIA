"""Construct matched clean, stopped, and continuous generated trajectories."""

import csv
import itertools
from contextlib import ExitStack

import numpy as np

from . import kernels as g
from .common import (
    BRANCH_CODES,
    MEASURED_FIELDS,
    POINT_FIELDS,
    REPEATS,
    check,
    configure,
    load_main_stats,
    make_clean,
    observation,
    round_means,
    save_stats,
    write_json,
    write_records,
)
from .recovery import recovery, summarize

HORIZONS = (20, 30, 60)
THRESHOLDS = (0.5, 1.0)
WINDOWS = (3, 5)
RECOVERY_SETTINGS = tuple(itertools.product(HORIZONS, THRESHOLDS, WINDOWS)) + (
    (60, 0.5, 3),
)


def recovery_rows(ids, kind, clean, latent, v0, values):
    for scale, ref, outcome in (
        ("latent", clean, latent),
        ("ACC", v0[:, 0], values[:, 0]),
    ):
        for index, (horizon, threshold, window) in enumerate(RECOVERY_SETTINGS):
            corrected = index == len(RECOVERY_SETTINGS) - 1
            yield dict(
                **ids,
                **recovery(
                    ref,
                    outcome,
                    ids["method"],
                    kind,
                    scale,
                    horizon,
                    threshold=threshold,
                    window=window,
                    correct=corrected,
                ),
            )


def branches(clean, key, params, risk, heterogeneity):
    clean_values = observation(clean, key, params)
    yield "clean", 0, "A0", clean, clean_values, clean_values
    for attack_index, (attack, strength) in enumerate(g.ATK.items()):
        if attack == "A0":
            continue
        g.ATTACK_END = 12
        stopped = g.attacked_curve(clean, *key, strength, risk, heterogeneity)
        g.ATTACK_END = 60
        continuous = g.attacked_curve(clean, *key, strength, risk, heterogeneity)
        g.ATTACK_END = 12
        check(
            np.array_equal(stopped[:12], continuous[:12]),
            f"Stopped and active prefixes differ at {key} {attack}",
        )
        yield (
            "stop",
            attack_index,
            attack,
            stopped,
            observation(stopped, key, params),
            clean_values,
        )
        yield (
            "continuous",
            attack_index,
            attack,
            continuous,
            observation(continuous, key, params),
            clean_values,
        )


def write_probability_examples(output, dataset, method, examples):
    data = {}
    for kind, probability in examples.items():
        data[f"{kind}_probabilities"] = probability["probabilities"]
        data[f"{kind}_alignment"] = probability["alignment"]
        data["truth"] = probability["truth"]
    np.savez_compressed(output / f"probabilities_{dataset}_{method}.npz", **data)


def run(output, main_directory):
    configure(60)
    primary, means, parameters = [], [], []
    trajectory_count = 0
    prefix_error = 0.0
    sensitivity_count = 0
    with ExitStack() as stack:
        sensitivity_stream = stack.enter_context(
            (output / "recovery_sensitivity.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            )
        )
        sensitivity_writer = None
        for di, dataset in enumerate(g.DS):
            main = load_main_stats(main_directory, dataset)
            g.N = g.OBJECTS[dataset]
            statistics, by_condition = [], {}
            with (output / f"rounds_{dataset}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "method",
                        "branch",
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
                        save_example = method in {"PCOC", "CensusAudit"} and (
                            partition,
                            init,
                        ) == (1, 1)
                        examples = {}
                        for kind, ai, attack, latent, data, clean_data in branches(
                            clean,
                            key,
                            params,
                            risk,
                            heterogeneity,
                        ):
                            values, confusion, nll, probability = data
                            check(
                                np.array_equal(latent[:4], clean[:4]),
                                f"Pre-attack prefix changed at {key}",
                            )
                            if kind != "continuous":
                                main_values, main_latent = main[mi, ai, partition, init]
                                error = max(
                                    float(np.max(np.abs(latent[:20] - main_latent))),
                                    float(np.max(np.abs(values[:20] - main_values))),
                                )
                                check(
                                    error < 1e-10,
                                    f"Main20 prefix mismatch at {dataset} {method} {attack} {partition} {init}",
                                )
                                prefix_error = max(prefix_error, error)
                            combined = np.column_stack([latent, values])
                            by_condition.setdefault((method, kind, attack), []).append(
                                combined
                            )
                            writer.writerows(
                                [
                                    method,
                                    kind,
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
                                for t in range(60)
                            )
                            statistics.append(
                                (
                                    (mi, BRANCH_CODES[kind], ai, partition, init),
                                    confusion,
                                    nll,
                                    values,
                                    latent,
                                )
                            )
                            trajectory_count += 1
                            if save_example and attack in {"A0", "A2"}:
                                examples[kind] = probability
                            if kind == "clean":
                                continue
                            ids_run = dict(**ids, branch=kind, attack=attack)
                            for row in recovery_rows(
                                ids_run, kind, clean, latent, clean_data[0], values
                            ):
                                if sensitivity_writer is None:
                                    sensitivity_writer = csv.DictWriter(
                                        sensitivity_stream, fieldnames=list(row)
                                    )
                                    sensitivity_writer.writeheader()
                                sensitivity_writer.writerow(row)
                                sensitivity_count += 1
                                if (
                                    row["threshold_pp"] == 0.5
                                    and row["consecutive_rounds"] == 3
                                    and not row["preattack_corrected"]
                                ):
                                    primary.append(row)
                        if save_example:
                            write_probability_examples(
                                output, dataset, method, examples
                            )
            save_stats(output / f"stats_{dataset}.npz", statistics)
            for (method, kind, attack), matrix in by_condition.items():
                means.extend(
                    round_means(
                        matrix,
                        {
                            "dataset": dataset,
                            "method": method,
                            "branch": kind,
                            "attack": attack,
                        },
                    )
                )
    stopped = [row for row in primary if row["branch"] == "stop"]
    for filename, rows in (
        ("round_means", means),
        ("run_parameters", parameters),
        ("primary_recovery", primary),
        (
            "recovery_summary",
            summarize(primary, ("scale", "branch", "horizon", "method")),
        ),
        (
            "scenario_recovery",
            summarize(stopped, ("dataset", "attack", "scale", "method", "horizon")),
        ),
    ):
        write_records(output / f"{filename}.csv", rows)
    check(
        trajectory_count == 14175 and len(primary) == 72900,
        "Followup60 trajectory coverage changed",
    )
    check(
        sensitivity_count == 315900,
        "Recovery settings did not cover all follow-up trajectories",
    )
    write_json(
        output / "manifest.json",
        {
            "study": "followup60",
            "evidence": "generated_probability_response",
            "actual_training": False,
            "rounds": 60,
            "trajectory_count": trajectory_count,
            "point_rows": trajectory_count * 60,
            "primary_recovery_rows": len(primary),
            "sensitivity_recovery_rows": sensitivity_count,
            "maximum_main20_prefix_error": prefix_error,
            "stop_rounds": [5, 12],
            "continuous_rounds": [5, 60],
            "horizons": HORIZONS,
            "thresholds_pp": THRESHOLDS,
            "consecutive_windows": WINDOWS,
            "nominal_E": 10,
            "measured_fields": dict.fromkeys(MEASURED_FIELDS),
        },
    )
