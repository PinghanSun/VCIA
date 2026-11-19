"""Registered graph-study conditions and deterministic, side-effect-free planning."""

import json
from dataclasses import dataclass, field
from itertools import product
from math import comb, isfinite
from pathlib import Path

from .decision import DecisionConfig
from .learner import TU_DATASETS

RANKERS = (
    "Unscreened",
    "J-Rank-Feasible",
    "Drift-First",
    "Random-Feasible",
    "No-Count",
    "No-Age",
)
CONTROLS = (
    "PCOC",
    *RANKERS,
    "J-Rank",
    "No-Verify",
    "Census-Audit",
    "Census-Direct",
    "Simultaneous-family",
    "Frozen",
    "Unprotected",
)
REGIMES = ("clean", "perturbed", "displaced", "canceling", "reinforcing")


@dataclass(frozen=True)
class Study:
    phase: str
    datasets: tuple[str, ...]
    partitions: tuple[int, ...]
    initializations: tuple[int, ...]
    regimes: tuple[str, ...]
    arms: tuple[str, ...]
    split_seed: int = 27001
    paired_inner_seeds: bool = False
    rounds: int = 30
    stress_start: int = 9
    stress_end: int = 20
    probe_seed: int = 27301
    peers: int = 8
    quota: int = 6
    partition_alpha: float = 0.5
    min_peer_graphs: int = 8
    local_steps: int = 1
    design_draws: int = 512
    verify_draws: int = 8192
    compromised: tuple[int, ...] = (1, 5)
    normal_claim: float = 0.25
    attacked_claim: float = 1.0
    displacement: float = 0.2
    repeat_every: int = 7
    missing_fraction: float = 0.0
    missing_block: int = 1
    topology: str = "ring"
    census: str = "nominated"
    fixed_evidence: bool = False
    fixed_ranking_shadow: bool = False
    log_predictions: bool = True
    strict_dataset_hash: bool = True
    poisoned_memory: bool = False
    profile_memory: bool = False
    backend: str = "local"
    decision: dict = field(default_factory=dict)

    def thresholds(self, dataset):
        per_round = 0.025 / self.rounds
        return DecisionConfig.normalized(
            self.quota,
            n_clusters=TU_DATASETS[dataset][1],
            **(
                {"epsilon": 0.1, "delta_screen": per_round, "delta_verify": per_round}
                | self.decision
            ),
        )

    def validate(self):
        if self.phase not in {"inner", "outer"}:
            raise ValueError("phase must be inner or outer")
        for name, values, allowed in (
            ("datasets", self.datasets, TU_DATASETS),
            ("regimes", self.regimes, REGIMES),
            ("arms", self.arms, CONTROLS),
        ):
            if (
                not values
                or len(set(values)) != len(values)
                or not set(values).issubset(allowed)
            ):
                raise ValueError(f"{name} must contain unique registered values")
        for seeds in (
            self.partitions,
            self.initializations,
            (self.split_seed, self.probe_seed),
        ):
            if (
                not seeds
                or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or seed < 0 for seed in seeds)
            ):
                raise ValueError(
                    "seed lists must contain distinct nonnegative integers"
                )
        if self.paired_inner_seeds and (
            self.phase != "inner" or self.partitions != self.initializations
        ):
            raise ValueError(
                "paired_inner_seeds requires matching lists and inner phase"
            )
        if self.fixed_evidence and self.phase != "inner":
            raise ValueError("fixed-evidence calibration is confined to inner runs")
        if self.fixed_evidence and {"Frozen", "Unprotected"}.intersection(self.arms):
            raise ValueError(
                "fixed-evidence calibration requires sampled design nominations"
            )
        if self.fixed_ranking_shadow and "Unscreened" not in self.arms:
            raise ValueError(
                "the fixed ranking shadow requires an Unscreened trajectory"
            )
        if not 3 <= self.peers <= 16 or not 1 <= self.quota <= self.peers:
            raise ValueError("expected 3–16 peers and an exact quota in the pool")
        if not 1 <= self.stress_start <= self.stress_end <= self.rounds:
            raise ValueError("require 1 <= stress_start <= stress_end <= rounds")
        if (
            min(self.local_steps, self.missing_block) < 1
            or min(self.design_draws, self.verify_draws) < 2
        ):
            raise ValueError("invalid local step, availability, or evidence budget")
        if self.repeat_every < 0 or not 0 <= self.missing_fraction <= 1:
            raise ValueError("invalid repeat or missing-proposal setting")
        if any(
            not isfinite(value) or value < 0
            for value in (self.normal_claim, self.attacked_claim, self.displacement)
        ):
            raise ValueError("claims and displacement must be finite and non-negative")
        if not isfinite(self.partition_alpha) or self.partition_alpha <= 0:
            raise ValueError("partition_alpha must be positive")
        if len(set(self.compromised)) != len(self.compromised) or any(
            not 0 <= peer < self.peers for peer in self.compromised
        ):
            raise ValueError(
                "compromised must list distinct zero-based peer identities"
            )
        if self.topology not in {
            "ring",
            "exponential",
            "dynamic",
        } or self.census not in {"none", "nominated", "all"}:
            raise ValueError("unknown topology or census setting")
        if self.backend not in {"local", "loopback"}:
            raise ValueError("backend must be local or loopback")
        if self.backend == "loopback" and {
            "Simultaneous-family",
            "Census-Direct",
        }.intersection(self.arms):
            raise ValueError("loopback supports one frozen nomination per round")
        for dataset in self.datasets:
            k = TU_DATASETS[dataset][1]
            if self.min_peer_graphs < k:
                raise ValueError(f"{dataset} needs at least {k} graphs per peer")
            self.thresholds(dataset).validate(self.peers, k)

    def job_keys(self):
        pairs = (
            zip(self.partitions, self.initializations)
            if self.paired_inner_seeds
            else product(self.partitions, self.initializations)
        )
        return tuple(
            (dataset, partition, seed, regime, arm)
            for partition, seed in pairs
            for dataset in self.datasets
            for regime in self.regimes
            for arm in self.arms
        )

    def plan(self):
        jobs = self.job_keys()
        return {
            "phase": self.phase,
            "datasets": self.datasets,
            "backend": self.backend,
            "trajectories": len(jobs),
            "scheduled_rounds": len(jobs) * self.rounds,
            "max_masks_per_round": comb(self.peers, self.quota),
            "design_draws": self.design_draws,
            "verify_draws": self.verify_draws,
            "per_dataset": {d: sum(j[0] == d for j in jobs) for d in self.datasets},
        }


def read_study(path: Path) -> Study:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in (
        "datasets",
        "partitions",
        "initializations",
        "regimes",
        "arms",
        "compromised",
    ):
        if key in raw:
            if not isinstance(raw[key], list):
                raise ValueError(f"{key} must be a JSON array")
            raw[key] = tuple(raw[key])
    study = Study(**raw)
    study.validate()
    return study
