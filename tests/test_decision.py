"""Decision and state invariants on hand-authored probabilities and payloads."""

import numpy as np
import pytest

from pcoc.__main__ import resolve_jobs
from pcoc.configuration import read_study
from pcoc.decision import Census, DecisionConfig, design
from pcoc.experiment import State, binding, staged_state, state_id


def test_large_individual_effects_can_cancel_within_an_unscreened_mask():
    config = DecisionConfig(q=2, epsilon=0.1)
    stored = np.array([[[0.7, 0.3]]] * 3)
    current = np.array([[[1.0, 0.0]], [[0.4, 0.6]], [[0.7, 0.3]]])
    claims = np.ones(3)
    changed = np.array([True, True, False])
    result = design(
        current,
        stored,
        claims,
        claims,
        changed,
        np.ones(3),
        np.zeros(8192, dtype=int),
        config,
        rank_full=True,
    )
    assert result.selections["PCOC"] is None
    chosen = result.selections["Unscreened"]
    assert chosen.selected == (0, 1)
    assert (
        Census(current, stored, claims, claims, config).score(chosen).within_tolerance
    )


def test_unavailable_identity_is_absent_from_common_background_but_kept_in_full_state():
    config = DecisionConfig(q=2, epsilon=0.1)
    old = np.array([[[0.8, 0.2]], [[0.6, 0.4]], [[0.3, 0.7]]])
    current = old.copy()
    current[0, 0] = [1.0, 0.0]
    claims = np.ones(3)
    chosen = design(
        current,
        old,
        claims,
        claims,
        np.array([True, False, False]),
        np.ones(3),
        np.zeros(512, dtype=int),
        config,
        available=np.array([True, True, False]),
    )
    assert chosen.version_stats[0].mean_J == pytest.approx(0.1)
    assert not chosen.version_stats[2].eligible
    assert [c.selected for c in chosen.candidates] == [(0, 1)]
    assert Census(current, old, claims, claims, config).score(
        (0, 1)
    ).ell == pytest.approx(0.2 / 3)


def test_identical_resubmission_only_advances_submission_clock():
    maps = np.zeros((3, 2, 1), dtype="<f8")
    claims = np.ones(3, dtype="<f8")
    old = State(maps, claims, np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    staged = staged_state(old, maps, claims, (0, 2), 4)
    assert staged.submitted.tolist() == [4, 0, 4]
    assert staged.changed.tolist() == [0, 0, 0]
    assert state_id(staged) == state_id(old)
    assert binding(old, staged, (0, 2), 4, "arm-a") != binding(
        old, staged, (0, 2), 4, "arm-b"
    )


def test_rank_plan_and_job_ranges():
    from pathlib import Path

    base = Path(__file__).resolve().parents[1]
    study = read_study(base / "configs" / "outer_ranking.json")
    assert study.plan()["trajectories"] == 540
    assert study.plan()["max_masks_per_round"] == 28
    first_six = resolve_jobs(study, "1:6")
    assert len({key[:-1] for _, key in first_six}) == 1
    assert len({key[-1] for _, key in first_six}) == 6
    with pytest.raises(ValueError):
        resolve_jobs(study, "540:541")
