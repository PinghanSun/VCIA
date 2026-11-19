"""Declared warm-up assumptions, never calibrated to published logs.

Finite-support transients change rounds 1..10 only. They do not enforce ranks,
terminal gaps, non-crossing, or significance. Each metric is recomputed from
the resulting generated probability model in prepare.py.
"""

import numpy as np

SEED = 2026111510
# start offset, early pulse, pulse center, reversal, reversal center (latent pp).
# These are pedagogical examples of different optimization paths, not estimates
# of the behavior of the named algorithms.
CORE_PROFILES = {
    "PCOC": [-11.5, 4.5, 3.4, -5.5, 6.2],
    "JRank": [-8.0, 1.5, 4.2, 3.3, 8.0],
    "FGCN-DKS": [-5.5, 4.5, 4.5, -7.5, 7.8],
    "CensusAudit": [-2.8, -2.0, 5.2, 3.0, 8.6],
}


def parameters(dataset_index, method_index, partition, init, method):
    c = np.random.default_rng(
        np.random.SeedSequence([SEED, 701, dataset_index, method_index])
    )
    r = np.random.default_rng(
        np.random.SeedSequence(
            [SEED, 702, dataset_index, method_index, partition, init]
        )
    )
    defaults = [
        c.uniform(-14, -3),
        c.uniform(-6, 9),
        c.uniform(2.4, 5.8),
        c.uniform(-8, 5),
        c.uniform(6.0, 9.0),
    ]
    profile = CORE_PROFILES.get(method, defaults)
    scale = c.uniform(0.80, 1.20) * r.uniform(0.78, 1.22)
    # Condition and run timing vary; no shared round-wise random increments.
    result = {
        "early_start_offset_pp": float(
            (profile[0] + c.normal(0, 0.7) + r.normal(0, 0.9)) * scale
        ),
        "early_pulse_pp": float((profile[1] + c.normal(0, 0.55)) * scale),
        "early_pulse_center": float(
            profile[2] + c.uniform(-0.5, 0.5) + r.uniform(-0.45, 0.45)
        ),
        "early_reversal_pp": float((profile[3] + c.normal(0, 0.55)) * scale),
        "early_reversal_center": float(
            profile[4] + c.uniform(-0.5, 0.5) + r.uniform(-0.45, 0.45)
        ),
        "early_width": float(r.uniform(0.9, 1.55)),
        "early_start_tau": float(r.uniform(2.2, 3.5)),
    }
    if method == "Frozen":
        for k in ["early_start_offset_pp", "early_pulse_pp", "early_reversal_pp"]:
            result[k] = 0.0
    return result


def modify(clean, params):
    t = np.arange(1, len(clean) + 1, dtype=float)
    taper = np.ones_like(t)
    taper[t >= 11] = 0
    middle = (t > 7) & (t < 11)
    taper[middle] = 0.5 * (1 + np.cos(np.pi * (t[middle] - 7) / 4))
    pulse = params["early_start_offset_pp"] * np.exp(
        -(t - 1) / params["early_start_tau"]
    )
    pulse += params["early_pulse_pp"] * np.exp(
        -0.5 * ((t - params["early_pulse_center"]) / params["early_width"]) ** 2
    )
    pulse += params["early_reversal_pp"] * np.exp(
        -0.5 * ((t - params["early_reversal_center"]) / params["early_width"]) ** 2
    )
    result = clean.copy()
    result[:10] += (pulse * taper)[:10]
    assert np.isfinite(result).all() and np.all((result > 0) & (result < 100)), (
        "Invalid simulation parameter range; do not clip silently."
    )
    assert np.array_equal(result[10:], clean[10:])
    return result
