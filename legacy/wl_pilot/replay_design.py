"""Reselect and re-audit fixed real-learning contexts across one-factor design settings."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .decision import DecisionConfig, design, exact, verify
from .run_pilot import rng


def settings(config, draws):
    yield "base", draws, config
    for size in (128, 256, 1024):
        if size != draws:
            yield f"design_draws={size}", size, config
    for factor, options in {
        "eta": (0.6, 0.95),
        "kappa": (0.05, 0.20),
        "chi": (0.01, 0.05),
        "gamma": (0.005, 0.02, 0.05),
    }.items():
        for value in options:
            yield f"{factor}={value}", draws, replace(config, **{factor: value})


def add_measurement(
    bucket, selected, current, stored, claims, memory, verify_idx, config
):
    bucket["contexts"] += 1
    bucket["eligible_peers"] += sum(stats.eligible for stats in selected.version_stats)
    bucket["screened_masks"] += sum(mask.screened for mask in selected.candidates)
    bucket["full_masks"] += len(selected.candidates)
    bucket["excluded_positive_feasible_masks"] += sum(
        mask.unscreened_rejection is None and not mask.screened
        for mask in selected.candidates
    )
    candidate = selected.selections["PCOC"]
    if candidate is None:
        bucket[selected.hold_reasons["PCOC"]] += 1
        return
    bucket["candidates"] += 1
    validation = verify(current, stored, claims, memory, candidate, verify_idx, config)
    census = exact(current, stored, claims, memory, candidate, config)
    bucket["drift_component_pass"] += not validation.drift_failed
    bucket["flip_component_pass"] += not validation.assignment_failed
    bucket["joint_pass"] += validation.accepted
    bucket["exact_pass"] += census.within_tolerance
    bucket["sampled_accept_outside_exact"] += (
        validation.accepted and not census.within_tolerance
    )
    bucket["sampled_hold_inside_exact"] += (
        not validation.accepted and census.within_tolerance
    )
    bucket["detected_refreshes_on_pass"] += candidate.H * validation.accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--out", type=Path, required=True, help="New replay output directory"
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory")
    root = args.results
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    runs = json.loads((root / "run_index.json").read_text(encoding="utf-8"))
    base = DecisionConfig(**manifest["decision"])
    grid = tuple(settings(base, manifest["design_draws"]))
    totals = defaultdict(lambda: defaultdict(int))
    matched = 0
    started = perf_counter()
    with threadpool_limits(limits=1):
        for run in runs:
            if run["arm"] != "PCOC":
                continue
            folder = root / run["path"]
            records = [
                json.loads(line)
                for line in (folder / "rounds.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            with (
                np.load(folder / "finite_effects.npz") as effects,
                np.load(folder / "states_and_draws.npz") as states,
            ):
                current, memory = effects["audit_current"], effects["audit_stored"]
                claims, stored_claims = (
                    states["current_claims"],
                    states["stored_claims"],
                )
                design_ids = states["design_probe_ids"]
                audit_ids = effects["audit_ids"]
                for i, row in enumerate(records):
                    seed, round_no = row["seed"], row["round"]
                    selection_seed = int(
                        rng(seed, round_no, 3).integers(0, 2**32, dtype=np.uint32)
                    )
                    draws = rng(seed, round_no, 2).integers(
                        len(audit_ids), size=max(size for _, size, _ in grid)
                    )
                    verify_idx = rng(seed, round_no, 4).integers(
                        len(audit_ids), size=manifest["verify_draws"]
                    )
                    if not np.array_equal(
                        audit_ids[draws[: manifest["design_draws"]]], design_ids[i]
                    ):
                        raise RuntimeError("Pilot design draw regeneration failed")
                    ages = round_no - np.asarray(row["last_content_before"])
                    for variant, size, config in grid:
                        selected = design(
                            current[i],
                            memory[i],
                            claims[i],
                            stored_claims[i],
                            np.asarray(row["content_changed_in_proposal"]),
                            ages,
                            draws[:size],
                            config,
                            random_seed=selection_seed,
                        )
                        if variant == "base":
                            picked = selected.selections["PCOC"]
                            recorded = row["policies"]["PCOC"]["candidate"]
                            ids = list(picked.selected) if picked is not None else None
                            if ids != (
                                recorded["selected"] if recorded is not None else None
                            ):
                                raise RuntimeError(
                                    f"Base decision did not reproduce: {run['path']}, round {round_no}"
                                )
                            matched += 1
                        bucket = totals[variant, row["regime"]]
                        add_measurement(
                            bucket,
                            selected,
                            current[i],
                            memory[i],
                            claims[i],
                            stored_claims[i],
                            verify_idx,
                            config,
                        )

    fields = (
        "contexts",
        "eligible_peers",
        "screened_masks",
        "full_masks",
        "excluded_positive_feasible_masks",
        "insufficient_eligible_peers",
        "no_feasible_composition",
        "no_detectable_update",
        "candidates",
        "drift_component_pass",
        "flip_component_pass",
        "joint_pass",
        "exact_pass",
        "sampled_accept_outside_exact",
        "sampled_hold_inside_exact",
        "detected_refreshes_on_pass",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "design_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("variant", "regime", "design_draws", "verify_draws", *fields),
        )
        writer.writeheader()
        for variant, size, _ in grid:
            for regime in manifest["regimes"]:
                record = totals[variant, regime]
                writer.writerow(
                    {
                        "variant": variant,
                        "regime": regime,
                        "design_draws": size,
                        "verify_draws": manifest["verify_draws"],
                        **{field: record[field] for field in fields},
                    }
                )
    report = {
        "scope": "fixed PCOC trajectory contexts from the inner pilot; no new training or outer-test scoring",
        "dataset": manifest.get("dataset_name", "PROTEINS"),
        "design": "common prefixes of independent IID design streams; masks frozen before the matched fresh audit",
        "verification_draws": manifest["verify_draws"],
        "reproduced_base_masks": matched,
        "grid": [
            {"variant": name, "design_draws": size, "configuration": asdict(config)}
            for name, size, config in grid
        ],
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds_offline": perf_counter() - started,
    }
    (args.out / "design_sensitivity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reproduced {matched} masks over {len(grid)} fixed-context settings")


if __name__ == "__main__":
    main()
