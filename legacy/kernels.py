"""Frozen probability-response kernels from the original numerical study."""

import itertools
import json
import math
from pathlib import Path

import numpy as np

CONF = json.loads(Path(__file__).with_name("profiles.json").read_text(encoding="utf-8"))
DS = list(CONF["tasks"])
TASKS = [dict(task) for task in CONF["tasks"].values()]
PARAMS = dict(CONF["methods"])
METHODS = list(PARAMS)
ATK = CONF["attack_strength"]
OBJECTS = {
    "PROTEINS": 256,
    "NCI1": 384,
    "ENZYMES": 288,
    "IMDB-BINARY": 320,
    "MUTAG": 160,
}
METRICS = ["ACC_pct", "NMI_pct", "ARI_pct", "macro_F1_pct", "proxy_NLL_nats"]
SEED = 2026111303
NOISE_SEED = 2026111520
T = 20
N = 256
ATTACK_START = 5
ATTACK_END = 12


def base_rng(*key):
    return np.random.default_rng(np.random.SeedSequence([NOISE_SEED, *key]))


rng = base_rng


def noise_parameters(di, mi, p, i):
    r = rng(10, di, mi, p, i)
    alpha = float(r.uniform(0.05, 0.18))
    beta = float(r.uniform(0.55, 0.94 - alpha))
    return {
        "phi_fast": float(r.uniform(-0.15, 0.65)),
        "phi_slow": float(r.uniform(0.55, 0.95)),
        "slow_weight": float(r.uniform(0.2, 0.7)),
        "student_df": float(r.uniform(4, 12)),
        "garch_alpha": alpha,
        "garch_beta": beta,
        "shock_probability": float(r.uniform(0.006, 0.03)),
        "shock_retention": float(r.uniform(0.25, 0.75)),
        "noise_floor": float(r.uniform(0.30, 0.70)),
        "noise_decay_rounds": float(r.uniform(8, 28)),
        "transient_amplitude": float(r.uniform(0.25, 1.2)),
        "transient_tau": float(r.uniform(5, 16)),
        "transient_period": float(r.uniform(8, 24)),
        "transient_phase": float(r.uniform(-math.pi, math.pi)),
        "convergence_tau": float(r.uniform(3.5, 10)),
        "observation_ar": float(r.uniform(0.30, 0.85)),
        "observation_scale": float(r.uniform(0.025, 0.08)),
        "class_bias_sd": float(r.uniform(0.05, 0.30)),
        "class_dynamic_sd": float(r.uniform(0.035, 0.18)),
        "logit_sharpness": float(r.uniform(1.05, 1.85)),
        "temperature": float(r.uniform(0.8, 1.25)),
    }


def clean_curve(di, mi, p, i, pr):
    task = TASKS[di]
    bias, resid, over, _protect, _tau, nm = PARAMS[METHODS[mi]]
    rp = np.random.default_rng(np.random.SeedSequence([SEED, 1, di, p]))
    ri = np.random.default_rng(np.random.SeedSequence([SEED, 2, di, p, i]))
    rm = np.random.default_rng(np.random.SeedSequence([SEED, 3, di, mi, p, i]))
    offset = rp.normal(0, 0.55) + ri.normal(0, 0.4)
    h = task["heterogeneity"] * math.exp(rp.normal(0, 0.18) + ri.normal(0, 0.12))
    risk = task["risk"] * math.exp(ri.normal(0, 0.30))
    loss = bias + 3.2 * h * resid + over * (1 - 0.25 * risk)
    loss += rm.normal(0, 0.22 + 0.12 * h) * (0.5 + resid + over)
    target = task["target"] + offset - loss
    r = rng(11, di, mi, p, i)
    start = task["target"] - 9 + offset + ri.normal(0, 0.25) + r.normal(0, 0.7)
    if METHODS[mi] == "Frozen":
        return np.full(T, start), np.zeros(T, dtype=np.uint8), risk, h
    acceptance = 1 / (1 + math.exp(-2.9 + over * 0.65 + h * 0.12))
    rate = 1 - math.exp(-task["pace"] / pr["convergence_tau"])
    x = np.empty(T)
    holds = np.zeros(T, dtype=np.uint8)
    envelope = pr["noise_floor"] + (1 - pr["noise_floor"]) * np.exp(
        -np.arange(T) / pr["noise_decay_rounds"]
    )
    fast = slow = shock = previous_innovation = 0.0
    variance = 1.0
    prev = start
    # Arrays use separate substreams, making all prefixes invariant to total horizon.
    innovations = rng(111, di, mi, p, i).standard_t(
        pr["student_df"], (T, 3)
    ) * math.sqrt((pr["student_df"] - 2) / pr["student_df"])
    u = rng(112, di, mi, p, i).random((T, 2))
    shocks = rng(113, di, mi, p, i).normal(-0.15, 0.8, T)
    shared = rng(114, di, p, i).normal(0, task["noise"], T)
    for j in range(T):
        variance = (
            (1 - pr["garch_alpha"] - pr["garch_beta"])
            + pr["garch_alpha"] * previous_innovation**2
            + pr["garch_beta"] * variance
        )
        previous_innovation = innovations[j, 0] * math.sqrt(variance)
        fast = (
            pr["phi_fast"] * fast
            + math.sqrt(1 - pr["phi_fast"] ** 2) * previous_innovation
        )
        slow = (
            pr["phi_slow"] * slow
            + math.sqrt(1 - pr["phi_slow"] ** 2) * innovations[j, 1]
        )
        shock = pr["shock_retention"] * shock + (
            shocks[j] if u[j, 1] < pr["shock_probability"] else 0
        )
        transient = (
            pr["transient_amplitude"]
            * math.exp(-j / pr["transient_tau"])
            * math.sin(2 * math.pi * j / pr["transient_period"] + pr["transient_phase"])
        )
        colored = (
            0.45
            * nm
            * (pr["slow_weight"] * slow + (1 - pr["slow_weight"]) * fast + shock)
            * envelope[j]
        )
        if u[j, 0] < acceptance:
            prev += (
                rate * (target + transient + colored - prev)
                + (innovations[j, 2] * task["noise"] * 0.65 * nm + shared[j] * 0.15)
                * envelope[j]
            )
        else:
            holds[j] = 1
        x[j] = prev + shared[j] * 0.15 * envelope[j]
    assert np.isfinite(x).all() and ((x > 0) & (x < 100)).all()
    return x, holds, risk, h


def metrics_from_cm(cm, nll_sums):
    a = cm.astype(float)
    n = a.sum(axis=(-2, -1))
    row = a.sum(-1)
    col = a.sum(-2)
    diag = np.diagonal(a, axis1=-2, axis2=-1)
    acc = diag.sum(-1) / n
    den = row[..., :, None] * col[..., None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.where(
            a > 0,
            (a / n[..., None, None])
            * np.log(np.maximum(a, 1) * n[..., None, None] / np.maximum(den, 1)),
            0,
        ).sum((-2, -1))
        hr = -np.where(
            row > 0, row / n[..., None] * np.log(np.maximum(row, 1) / n[..., None]), 0
        ).sum(-1)
        hc = -np.where(
            col > 0, col / n[..., None] * np.log(np.maximum(col, 1) / n[..., None]), 0
        ).sum(-1)
    nmi = np.divide(2 * mi, hr + hc, out=np.ones_like(mi), where=(hr + hc) > 1e-15)
    pairs = (a * (a - 1) / 2).sum((-2, -1))
    rp = (row * (row - 1) / 2).sum(-1)
    cp = (col * (col - 1) / 2).sum(-1)
    expected = rp * cp / (n * (n - 1) / 2)
    denom = (rp + cp) / 2 - expected
    ari = np.divide(
        pairs - expected, denom, out=np.ones_like(pairs), where=np.abs(denom) > 1e-15
    )
    f1 = np.divide(
        2 * diag, row + col, out=np.zeros_like(diag), where=(row + col) > 0
    ).mean(-1)
    return np.stack(
        [acc * 100, nmi * 100, ari * 100, f1 * 100, nll_sums.sum(-1) / n], axis=-1
    )


def joint_probabilities(latent, di, mi, p, i, pr):
    k = 6 if DS[di] == "ENZYMES" else 2
    r = rng(20, di, p, i)
    # Scenario-specific sample sizes and imbalanced labels; not real graph inputs.
    prior = r.dirichlet(np.full(k, 8.0))
    truth = r.choice(k, N, p=prior)
    assert len(np.unique(truth)) == k
    difficulty = np.empty(N)
    for c in range(k):
        ix = np.flatnonzero(truth == c)
        u = (np.arange(len(ix)) + 0.5) / len(ix)
        r.shuffle(u)
        difficulty[ix] = np.log(u / (1 - u))
    loadings = r.normal(0, 1, (N, 2))
    wrong = -r.uniform(0.4, 2.4, (N, k))
    competitor = (truth + r.integers(1, k, N)) % k
    wrong[np.arange(N), competitor] = 0
    factors = np.zeros((T, 3 + k))
    innov = rng(21, di, mi, p, i).normal(0, 1, (T, 3 + k))
    for j in range(1, T):
        factors[j] = (
            pr["observation_ar"] * factors[j - 1]
            + math.sqrt(1 - pr["observation_ar"] ** 2) * innov[j]
        )
    class_static = rng(22, di, mi, p, i).normal(0, pr["class_bias_sd"], k)
    class_static -= np.dot(class_static, prior)
    if METHODS[mi] == "Frozen":
        factors[:] = 0
    drift = pr["observation_scale"] * (factors[:, :2] @ loadings.T)
    class_error = class_static[None, :] + pr["class_dynamic_sd"] * factors[:, 3:]
    threshold = np.log(latent / (100 - latent))
    z = (threshold[:, None] - difficulty[None, :] - drift + class_error[:, truth]) * pr[
        "logit_sharpness"
    ]
    logits = np.broadcast_to(wrong, (T, N, k)).copy()
    if k > 2:
        rival = rng(23, di, mi, p, i).normal(0, 0.10, (T, k))
        if METHODS[mi] == "Frozen":
            rival[:] = rival[0]
        logits += rival[:, None, :]
    logits[:, np.arange(N), truth] = z
    # Calibration affects confidence separately from the top-ranked class.
    temp = pr["temperature"] * np.exp(0.12 * factors[:, 2])
    logits /= temp[:, None, None]
    logits -= logits.max(-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    pred = probs.argmax(-1)
    cm = np.zeros((T, k, k), dtype=np.uint16)
    np.add.at(cm, (np.repeat(np.arange(T), N), np.tile(truth, T), pred.ravel()), 1)
    perm = np.broadcast_to(np.arange(k), (T, k)).copy()
    diag = np.diagonal(cm, axis1=1, axis2=2)
    bad = np.flatnonzero((diag < cm.max(2)).any(1))
    if len(bad):
        perms = np.array(list(itertools.permutations(range(k))))
        scores = cm[bad][:, np.arange(k)[None, :], perms].sum(-1)
        perm[bad] = perms[scores.argmax(-1)]
    aligned = np.take_along_axis(cm, perm[:, None, :], axis=2)
    correctprob = probs[np.arange(T)[:, None], np.arange(N)[None, :], perm[:, truth]]
    itemloss = -np.log(correctprob)
    nll = np.stack([itemloss[:, truth == c].sum(1) for c in range(k)], axis=1)
    metrics = metrics_from_cm(aligned, nll)
    return (
        metrics,
        aligned,
        nll,
        {"truth": truth, "probabilities": probs, "alignment": perm, "metrics": metrics},
    )


def attacked_curve(clean, di, mi, p, i, strength, risk, h):
    if strength == 0 or METHODS[mi] == "Frozen":
        return clean.copy()
    r = rng(30, di, mi, p, i)
    protect = PARAMS[METHODS[mi]][3]
    tau = PARAMS[METHODS[mi]][4] / 3 * math.exp(r.normal(0, 0.4)) * (0.9 + 0.2 * h)
    burden = math.exp(r.normal(-0.1, 0.48))
    stream = rng(31, di, mi, p, i).normal(size=(T, 5))
    uniform = rng(32, di, mi, p, i).random((T, 2))
    damage = scar = 0.0
    out = clean.copy()
    for j in range(ATTACK_START - 1, T):
        if j + 1 <= ATTACK_END:
            exposure = 0.2 + (
                math.exp(-0.14 + 0.55 * stream[j, 0]) if uniform[j, 0] < 0.6 else 0
            )
            target = 7 * strength * risk * (1 - protect) * burden * exposure
            damage += 0.19 * (target - damage) + 0.06 * stream[j, 1] * strength * (
                1.3 - protect
            )
            scar += 0.0033 * max(damage, 0) * (1 + 0.25 * h)
        else:
            progress = 0.15 if uniform[j, 1] < 0.25 else 1.25
            damage *= math.exp(-progress / (tau + 1.7 * max(damage, 0)))
            scar *= math.exp(-1 / (100 + 20 * h))
        base = clean[j] / 100
        logit = math.log(base / (1 - base)) - (damage + scar) / (
            100 * base * (1 - base)
        )
        out[j] = 100 / (1 + math.exp(-logit))
    return out
