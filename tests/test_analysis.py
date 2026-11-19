"""Hand-authored fixtures for PCOC report arithmetic."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pcoc.analysis import (
    analytic,
    calibrate,
    load_runs,
    measured_summary,
    screening,
    summarize,
)


def row(
    *, arm="Unscreened", partition=1, seed=11, step=1, chosen=True, updates=2, nmi=0.4
):
    return {
        "dataset": "fixture",
        "phase": "inner",
        "partition_seed": partition,
        "seed": seed,
        "arm": arm,
        "regime": "clean",
        "round": step,
        "segment": "stress",
        "status": "accepted" if chosen else "no_candidate",
        "selected": [0, 1] if chosen else None,
        "nominated_state_hash": "selected" if chosen else None,
        "operator_updates": updates if chosen else 0,
        "claim_updates": 0,
        "quality": {"nmi": nmi},
        "executed_state_hash": "selected" if chosen else "baseline",
        "cost": {
            "screen_seconds": 1.0,
            "composition_seconds": 4.0,
            "new_graph_operator_pairs": 12,
        },
    }


class AnalysisTests(unittest.TestCase):
    def test_exact_safety_rates_use_scheduled_and_commit_denominators(self):
        good = dict(
            row(),
            content_updates={"operator_only": 1, "claim_only": 0, "joint": 1},
            nominated_H=1,
            nominated_exact={"within_tolerance": True},
        )
        bad = dict(
            row(step=2),
            content_updates={"operator_only": 2},
            nominated_H=2,
            nominated_exact={"within_tolerance": False},
        )
        held = row(step=3, chosen=False)
        report = measured_summary([good, bad, held])
        self.assertAlmostEqual(report["r_all"], 1 / 3)
        self.assertEqual(report["r_commit"], 0.5)
        self.assertEqual(report["U_adm"], 2)
        self.assertEqual(report["C_adm"], 1)
        self.assertEqual(report["nomination_H_distribution"], {1: 1, 2: 1})
        self.assertIsNone(measured_summary([good, row(step=2), held])["r_all"])

    def test_screening_reports_loss_and_online_cost_separately(self):
        screened = row(arm="PCOC", chosen=False)
        baseline = row()
        screened["candidates"] = [
            {
                "selected": [0, 1],
                "H": 2,
                "operator_updates": 2,
                "screened": False,
                "feasible": True,
                "exact": {"within_tolerance": True},
            },
            {
                "selected": [1, 2],
                "H": 1,
                "operator_updates": 1,
                "screened": True,
                "feasible": True,
                "exact": {"within_tolerance": False},
            },
            {
                "selected": [0, 2],
                "H": 0,
                "operator_updates": 0,
                "screened": True,
                "feasible": True,
                "exact": {"within_tolerance": True},
            },
        ]
        result = screening([screened, baseline], "PCOC", "Unscreened")
        exact = result["offline_rounds"][0]["families"]
        self.assertEqual(exact["design_feasible"]["reference_masks"], 2)
        self.assertEqual(exact["census_admissible"]["all_reference_masks_removed"], 1)
        self.assertEqual(exact["both"]["maximum_canonical_updates_lost"], 2)
        contrast = result["paired_execution"][0]
        self.assertEqual(
            contrast["endpoints"]["operator_updates"]["mean_difference"], -2
        )
        self.assertEqual(
            contrast["endpoints"]["screen_and_composition_seconds"]["mean_difference"],
            0,
        )
        self.assertEqual(contrast["different_nominated_states"], 1)

    def test_pair_compares_nominated_state_payload(self):
        screened = dict(row(arm="PCOC"), selected_payload_hash="screen-registration")
        baseline = dict(row(), selected_payload_hash="full-registration")
        shifted = dict(row(arm="PCOC", nmi=0.2), regime="perturbed")
        report = summarize([screened, baseline, shifted], "Unscreened")
        matched = report["paired_contrasts"][0]
        self.assertEqual(matched["comparable_nominated_states"], 1)
        self.assertEqual(matched["same_nominated_states"], 1)
        self.assertEqual(matched["different_nominated_states"], 0)
        delta = report["paired_regime_contrasts"][0]["endpoints"]["end_nmi"][
            "mean_difference"
        ]
        self.assertAlmostEqual(delta, -0.2)

    def test_partition_aggregation_and_frozen_contrast(self):
        rows = []
        for partition, seed, gain in [(1, 11, 2), (1, 12, 4), (2, 11, 7)]:
            rows.append(row(partition=partition, seed=seed, updates=0))
            rows.append(row(partition=partition, seed=seed, arm="No-Age", updates=gain))
        rows[0]["fixed_ranking"] = {
            "Unscreened": {"selected": [0, 1], "accepted": True, "operator_updates": 2},
            "No-Age": {"selected": [0, 2], "accepted": False, "operator_updates": 2},
        }
        report = summarize(rows, "Unscreened")
        result = report["paired_contrasts"][0]["endpoints"]["operator_updates"]
        self.assertEqual(result["partition_deltas"], {1: 3.0, 2: 7.0})
        self.assertEqual(result["mean_difference"], 5)
        snapshot = next(
            v for v in report["fixed_snapshot_ranking"] if v["ranking"] == "No-Age"
        )
        self.assertEqual(snapshot["changed_nomination_masks"], 1)
        self.assertEqual(snapshot["would_pass_operator_updates"], 0)

    def test_point_mass_calibration_keeps_no_candidate_denominator(self):
        selected = row()
        selected["nominated_exact"] = {"within_tolerance": True}
        selected["candidate_operator_updates"] = 2
        selected["fixed_evidence"] = {
            "distances": [0.0],
            "flips": [False],
            "output_range": 1.0,
            "gamma": 0.01,
            "chi": 0.02,
            "alpha": 0.05 / 120,
            "design_unique": 1,
            "distinct_operators": 2,
        }
        no_candidate = row(step=2, chosen=False)
        result = calibrate([selected, no_candidate], [8192], 1, 1)
        first = result["groups"][0]
        self.assertEqual(first["without_nomination"], 1)
        self.assertEqual(first["mean_joint_pass_probability_given_candidate"], 1)
        self.assertEqual(first["mean_joint_pass_probability_per_scheduled_round"], 0.5)
        self.assertEqual(
            first[
                "expected_operator_updates_per_scheduled_round_on_frozen_nominations"
            ],
            1,
        )

    def test_exact_binary_calculation(self):
        result = analytic([2048, 8192], [0.005, 0.008], 0.05 / 120, 0.02, 0.01)[
            "budgets"
        ]
        self.assertAlmostEqual(
            result[0]["exact_binary_pass_probability"]["0.008"], 0.003026, places=5
        )
        self.assertAlmostEqual(
            result[1]["exact_binary_pass_probability"]["0.005"], 0.971297, places=6
        )

    def test_refuse_incomplete_trajectory(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rounds.jsonl"
            path.write_text(
                json.dumps(dict(row(), scheduled_rounds=2)) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unexpected round count"):
                load_runs(path)

    def test_registered_missingness_preserves_scheduled_denominator(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                {
                    "job": index,
                    "dataset": "fixture",
                    "partition_seed": index,
                    "seed": 11,
                    "regime": "clean",
                    "arm": "Unscreened",
                }
                for index in range(1, 5)
            ]
            manifest = {
                "study": {"phase": "inner", "rounds": 2},
                "plan": {"jobs": jobs},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            index = [
                {
                    **jobs[0],
                    "phase": "inner",
                    "scheduled_rounds": 2,
                    "status": "complete",
                }
            ]
            (root / "run_index.json").write_text(json.dumps(index), encoding="utf-8")
            for partition, n_rounds in ((1, 2), (2, 1), (3, 1)):
                trajectory = (
                    root / "fixture" / f"p{partition}_i11" / "clean" / "Unscreened"
                )
                trajectory.mkdir(parents=True)
                lines = [
                    json.dumps(
                        dict(row(partition=partition, step=step), scheduled_rounds=2)
                    )
                    for step in range(1, n_rounds + 1)
                ]
                (trajectory / "rounds.jsonl").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                if partition == 2:
                    failure = {
                        **jobs[1],
                        "phase": "inner",
                        "scheduled_rounds": 2,
                        "failure": "Fixture error",
                    }
                    (trajectory / "failure.json").write_text(
                        json.dumps(failure), encoding="utf-8"
                    )
            records, sources, coverage = load_runs(root, phase="inner")
            self.assertEqual(len(records), 2)
            self.assertEqual(len(sources), 3)
            self.assertEqual(
                coverage["statuses"],
                {"complete": 1, "failed": 1, "incomplete": 1, "unstarted": 1},
            )
            self.assertEqual(coverage["scheduled_rounds_all_segments"], 8)
            self.assertEqual(coverage["recorded_rounds_all_segments"], 4)
            self.assertEqual(coverage["included_metric_rounds"], 2)


if __name__ == "__main__":
    unittest.main()
