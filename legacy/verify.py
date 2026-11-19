"""Compare regenerated sufficient statistics with the author's archived NPZ files."""

import json

import numpy as np

from . import kernels as g
from .common import check

REFERENCE_NAMES = {
    "main20": "{scenario}_充分统计量.npz",
    "supplement20": "{scenario}_补充充分统计量.npz",
    "followup60": "充分统计量_{scenario}.npz",
}


def run(output, reference):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    study = manifest.get("study")
    check(study in REFERENCE_NAMES, f"Unsupported numerical study in {output}")
    comparisons = {}
    for scenario in g.DS:
        original = reference / REFERENCE_NAMES[study].format(scenario=scenario)
        regenerated = output / f"stats_{scenario}.npz"
        errors = {}
        with (
            np.load(original, allow_pickle=False) as source,
            np.load(regenerated, allow_pickle=False) as target,
        ):
            for key in ("metadata", "confusion", "class_nll_sums", "metrics", "latent"):
                expected, observed = source[key], target[key]
                check(
                    observed.shape == expected.shape, f"{scenario} {key} shape mismatch"
                )
                if key in {"metadata", "confusion"}:
                    check(
                        np.array_equal(observed, expected), f"{scenario} {key} mismatch"
                    )
                    continue
                error = float(np.max(np.abs(observed - expected)))
                check(error < 1e-10, f"{scenario} {key} absolute error {error:.3g}")
                errors[key] = error
        comparisons[scenario] = errors
    print(
        json.dumps(
            {"study": study, "matches_reference": True, "absolute_errors": comparisons},
            indent=2,
        )
    )
