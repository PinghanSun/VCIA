"""Declared late fluctuations; no fit to published sequences.

Condition shocks are shared within a method/scenario, with separate partition
and initialization components. This dependence is disclosed in the workbook.
"""

import numpy as np

SEED = 2026111512


def parameters(di, mi, partition, init, method):
    c = np.random.default_rng(np.random.SeedSequence([SEED, 801, di, mi]))
    return {
        "late_condition_amplitude_pp": float(c.uniform(0.65, 1.05))
        if method != "Frozen"
        else 0.0,
        "late_partition_amplitude_pp": 0.20 if method != "Frozen" else 0.0,
        "late_run_amplitude_pp": 0.45 if method != "Frozen" else 0.0,
        "late_rho": float(c.uniform(0.10, 0.45)),
    }


def stationary_component(seed, rho, length):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    # Burn-in for stationary AR. The Student-t variance is df/(df-2).
    innovations = rng.standard_t(6, size=length + 100) * np.sqrt(4 / 6)
    x = np.zeros_like(innovations)
    for j in range(1, len(x)):
        x[j] = rho * x[j - 1] + np.sqrt(1 - rho * rho) * innovations[j]
    x = x[-length:]
    return x - x.mean()  # No linear trend or chosen endpoint is imposed.


def modify(clean, params, di, mi, partition, init):
    result = clean.copy()
    n = len(clean) - 9
    shock = params["late_condition_amplitude_pp"] * stationary_component(
        [SEED, 802, di, mi], params["late_rho"], n
    )
    shock += params["late_partition_amplitude_pp"] * stationary_component(
        [SEED, 803, di, mi, partition], params["late_rho"], n
    )
    shock += params["late_run_amplitude_pp"] * stationary_component(
        [SEED, 804, di, mi, partition, init], params["late_rho"], n
    )
    ramp = np.ones(n)
    ramp[:2] = [0.45, 0.80]
    shock *= ramp
    shock -= shock.mean()  # Center after ramp too; no sustained mean shift.
    result[9:] += shock
    assert np.array_equal(result[:9], clean[:9])
    assert np.isfinite(result).all() and np.all((result > 0) & (result < 100)), (
        "Do not silently clip invalid simulation values."
    )
    return result
