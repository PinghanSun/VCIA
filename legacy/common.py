"""Shared I/O, dynamic response construction, and paired summaries."""

import csv
import itertools
import json
import math

import numpy as np

from . import early_dynamics as early
from . import kernels as g
from . import late_dynamics as late

REPEATS = tuple(itertools.product(range(1, 4), range(1, 6)))
T_CRITICAL = 4.30265273
BRANCH_CODES = {"clean": 0, "stop": 1, "continuous": 2}
MEASURED_FIELDS = ["actual_local_epochs", "actual_GPU_seconds", "actual_peak_GPU_bytes"]
POINT_FIELDS = ["latent_score_pct", *g.METRICS]
CORE_PROFILES = {"PCOC", "JRank", "FGCN-DKS", "CensusAudit"}


def write_records(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def check(condition, description):
    if not condition:
        raise RuntimeError(description)


def save_stats(path, stats):
    metadata, confusion, nll, metrics, latent = zip(*stats)
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        confusion=np.asarray(confusion),
        class_nll_sums=np.asarray(nll),
        metrics=np.asarray(metrics),
        latent=np.asarray(latent),
    )


def configure(horizon):
    g.T = horizon
    g.ATTACK_END = 12
    g.TASKS[:] = [dict(task) for task in g.CONF["tasks"].values()]
    g.PARAMS.clear()
    g.PARAMS.update(g.CONF["methods"])
    g.METHODS[:] = list(g.PARAMS)
    g.rng = g.base_rng
    early.SEED = 2026111510
    late.SEED = 2026111512


def extended_component(seed, rho, length):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    innovations = rng.standard_t(6, size=length + 100) * np.sqrt(4 / 6)
    response = np.zeros_like(innovations)
    for j in range(1, len(response)):
        response[j] = rho * response[j - 1] + np.sqrt(1 - rho * rho) * innovations[j]
    response = response[100:]
    return response - response[:11].mean()


def extend_late(clean, params, dataset_index, method_index, partition, init):
    length = len(clean) - 9
    key = [dataset_index, method_index]
    rho = params["late_rho"]
    shock = params["late_condition_amplitude_pp"] * extended_component(
        [late.SEED, 802, *key],
        rho,
        length,
    )
    shock += params["late_partition_amplitude_pp"] * extended_component(
        [late.SEED, 803, *key, partition],
        rho,
        length,
    )
    shock += params["late_run_amplitude_pp"] * extended_component(
        [late.SEED, 804, *key, partition, init],
        rho,
        length,
    )
    shock[:2] *= [0.45, 0.8]
    shock -= shock[:11].mean()
    result = clean.copy()
    result[9:] += shock
    reference = late.modify(clean[:20], params, *key, partition, init)
    check(
        np.isfinite(result).all() and ((result > 0) & (result < 100)).all(),
        "Invalid late response",
    )
    check(
        np.allclose(result[:20], reference, rtol=0, atol=1e-12),
        "Late 20-round prefix changed",
    )
    return result


def make_clean(di, mi, partition, init, pace=1.0, noise=1.0):
    method = g.METHODS[mi]
    key = (di, mi, partition, init)
    parameters = g.noise_parameters(*key)
    parameters["convergence_tau"] *= 0.65 / pace
    clean, holds, risk, heterogeneity = g.clean_curve(*key, parameters)
    early_params = early.parameters(*key, method)
    late_params = late.parameters(*key, method)
    for name in (
        "late_condition_amplitude_pp",
        "late_partition_amplitude_pp",
        "late_run_amplitude_pp",
    ):
        late_params[name] *= noise**0.35
    clean = early.modify(clean, early_params)
    apply_late = late.modify if g.T == 20 else extend_late
    clean = apply_late(clean, late_params, *key)
    return clean, holds, risk, heterogeneity, parameters | early_params | late_params


def observation(latent, key, params):
    values, confusion, nll, probability = g.joint_probabilities(latent, *key, params)
    check(np.isfinite(values).all(), f"Non-finite generated metric at {key}")
    check(
        (confusion.sum((1, 2)) == g.N).all(),
        f"Confusion totals do not cover generated items at {key}",
    )
    check(
        np.allclose(probability["probabilities"].sum(-1), 1),
        f"Probabilities do not sum to one at {key}",
    )
    return values, confusion, nll, probability


def run_endpoints(values):
    return {
        **{f"final_{key}": float(values[-1, j]) for j, key in enumerate(g.METRICS)},
        **{
            f"tail5_{key}": float(values[-5:, j].mean())
            for j, key in enumerate(g.METRICS)
        },
    }


def condition_endpoints(values):
    return {
        "nested_runs": len(values),
        **{
            f"final_mean_{key}": float(values[:, -1, j].mean())
            for j, key in enumerate(g.METRICS)
        },
        **{
            f"final_SD_{key}": float(values[:, -1, j].std(ddof=1))
            for j, key in enumerate(g.METRICS)
        },
        **{
            f"tail5_mean_{key}": float(values[:, -5:, j].mean())
            for j, key in enumerate(g.METRICS)
        },
    }


def round_means(arrays, identifiers):
    matrix = np.asarray(arrays)
    means = matrix.mean(0)
    sds = matrix.std(0, ddof=1)
    return [
        dict(
            **identifiers,
            round=t + 1,
            nested_runs=len(matrix),
            **{f"mean_{key}": float(means[t, j]) for j, key in enumerate(POINT_FIELDS)},
            **{f"SD_{key}": float(sds[t, j]) for j, key in enumerate(POINT_FIELDS)},
        )
        for t in range(len(means))
    ]


def paired_interval(deltas):
    partition_means = deltas.reshape(3, 5, -1).mean(1)
    center = deltas.reshape(15, -1).mean(0)
    radius = T_CRITICAL * partition_means.std(0, ddof=1) / math.sqrt(3)
    return center, center - radius, center + radius


def load_main_stats(directory, dataset):
    path = directory / f"stats_{dataset}.npz"
    with np.load(path, allow_pickle=False) as saved:
        metadata = saved["metadata"]
        values = saved["metrics"]
        latent = saved["latent"]
    expected = len(g.CONF["methods"]) * len(g.ATK) * len(REPEATS)
    check(
        values.shape == (expected, 20, len(g.METRICS)),
        f"Unexpected main20 metric shape in {path}",
    )
    check(latent.shape == (expected, 20), f"Unexpected main20 latent shape in {path}")
    check(
        metadata.shape == (expected, 4), f"Unexpected main20 metadata shape in {path}"
    )
    indexed = {
        tuple(map(int, ids)): (y, score)
        for ids, y, score in zip(metadata, values, latent)
    }
    keys = set(
        itertools.product(
            range(len(g.CONF["methods"])), range(len(g.ATK)), range(1, 4), range(1, 6)
        )
    )
    check(indexed.keys() == keys, f"Unexpected trajectory keys in {path}")
    return indexed
