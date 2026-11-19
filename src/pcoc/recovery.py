"""Measure stress and sustained recovery from paired WL trajectory traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, ulp
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile

import numpy as np

PAIR_FIELDS = ("dataset", "phase", "partition_seed", "seed", "arm")
STUDY_LISTS = {"datasets", "partitions", "initializations", "regimes", "arms"}
TRACE_OPTIONS = {
    "log_predictions",
    "profile_memory",
    "census",
    "fixed_evidence",
    "fixed_ranking_shadow",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class Trajectory:
    path: Path
    key: tuple
    regime: str
    design: str
    evaluation: str
    graphs: int
    clusters: int
    stress_start: int
    stress_end: int
    end: int

    def segment(self, step):
        if step < self.stress_start:
            return "warmup"
        return "stress" if step <= self.stress_end else "recovery"


def _registration(path):
    data = _json(path.with_name("registration.json"))
    study = data["study"]
    roles = _json(path.with_name("roles.json"))
    evaluation_ids = roles["evaluation_ids"]
    if (
        not isinstance(evaluation_ids, list)
        or not evaluation_ids
        or any(type(i) is not int or not 0 <= i < 2**63 for i in evaluation_ids)
        or len(set(evaluation_ids)) != len(evaluation_ids)
    ):
        raise ValueError("invalid evaluation identifiers")
    evaluation = hashlib.sha256(
        np.asarray(evaluation_ids, dtype="<i8").tobytes()
    ).hexdigest()
    if evaluation != data["evaluation_ids_sha256"]:
        raise ValueError("evaluation identifiers differ from the registration digest")
    identity = {
        "dataset": data["dataset"],
        "phase": study["phase"],
        "partition_seed": data["partition"],
        "seed": data["seed"],
        "arm": data["policy"],
    }
    key = tuple(identity[name] for name in PAIR_FIELDS)
    if (
        any(not isinstance(identity[name], str) for name in ("dataset", "phase", "arm"))
        or any(type(identity[name]) is not int for name in ("partition_seed", "seed"))
        or not isinstance(data["regime"], str)
    ):
        raise ValueError("invalid trajectory identity")
    design = {
        k: v
        for k, v in data.items()
        if k not in {"regime", "study", "evaluation_ids_sha256"}
    }
    design["study"] = {
        k: v for k, v in study.items() if k not in STUDY_LISTS | TRACE_OPTIONS
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            design, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    first, last, end = (
        study[name] for name in ("stress_start", "stress_end", "rounds")
    )
    clusters = data["feature_shape"][0]
    if any(type(value) is not int for value in (first, last, end, clusters)) or not (
        1 <= first <= last <= end and clusters >= 2
    ):
        raise ValueError("invalid stress schedule or cluster dimension")
    return Trajectory(
        path,
        key,
        data["regime"],
        fingerprint,
        evaluation,
        len(evaluation_ids),
        clusters,
        first,
        last,
        end,
    )


def _prediction(row, graphs, clusters):
    assignments = scores = None
    if row.get("eval_assignments") is not None:
        assignments = np.asarray(row["eval_assignments"])
        if (
            assignments.shape != (graphs,)
            or assignments.dtype.kind not in "iu"
            or np.any(assignments < 0)
            or np.any(assignments >= clusters)
        ):
            raise ValueError("invalid graph assignment vector")
    if row.get("eval_scores") is not None:
        scores = np.asarray(row["eval_scores"], dtype=float)
        if (
            scores.shape != (graphs, clusters)
            or not np.isfinite(scores).all()
            or np.any(scores < 0)
            or not np.allclose(scores.sum(axis=1), 1.0, rtol=0.0, atol=5e-7)
        ):
            raise ValueError("invalid evaluation score matrix")
        derived = scores.argmax(axis=1)
        if assignments is not None and not np.array_equal(assignments, derived):
            raise ValueError("graph assignments differ from saved score argmax")
        assignments = derived
    return assignments, scores


def _rounds(traj):
    with traj.path.open(encoding="utf-8-sig") as stream:
        count = 0
        for line in stream:
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            if (
                tuple(row[key] for key in PAIR_FIELDS) != traj.key
                or row["regime"] != traj.regime
                or row["round"] != count
                or row["segment"] != traj.segment(count)
                or row["scheduled_rounds"] != traj.end
            ):
                raise ValueError(f"invalid round identity or schedule at round {count}")
            nmi = float(row["quality"]["nmi"])
            if not isfinite(nmi) or not -1e-12 <= nmi <= 1 + 1e-12:
                raise ValueError(f"invalid NMI at round {count}")
            yield nmi, *_prediction(row, traj.graphs, traj.clusters)
        if count != traj.end:
            raise ValueError(f"incomplete trajectory ({count}/{traj.end} rounds)")


def _mean(values):
    values = [value for value in values if value is not None]
    return fmean(values) if values else None


def _segment(curve, name):
    rows = [row for row in curve if row["segment"] == name]
    return {
        "paired_rounds": len(rows),
        "mean_nmi_deficit": _mean(row["nmi_deficit"] for row in rows),
        "mean_assignment_flip_rate": _mean(row["assignment_flip_rate"] for row in rows),
        "assignment_paired_rounds": sum(
            row["assignment_flip_rate"] is not None for row in rows
        ),
        "mean_soft_output_drift": _mean(row["soft_output_drift"] for row in rows),
        "soft_output_paired_rounds": sum(
            row["soft_output_drift"] is not None for row in rows
        ),
    }


def _compare(clean, attack, gap, window):
    course = []
    streak = 0
    recovered = None
    limit = gap + 2 * ulp(1.0)
    for step, (left, right) in enumerate(
        zip(_rounds(clean), _rounds(attack), strict=True), start=1
    ):
        clean_nmi, clean_labels, clean_scores = left
        attack_nmi, attack_labels, attack_scores = right
        deficit = clean_nmi - attack_nmi
        segment = attack.segment(step)
        if segment == "recovery" and recovered is None:
            streak = streak + 1 if deficit <= limit else 0
            if streak == window:
                recovered = step - window + 1
        flips = (
            int(np.count_nonzero(clean_labels != attack_labels))
            if clean_labels is not None and attack_labels is not None
            else None
        )
        drift = (
            float(np.abs(clean_scores - attack_scores).mean())
            if clean_scores is not None and attack_scores is not None
            else None
        )
        course.append(
            {
                "round": step,
                "segment": segment,
                "clean_nmi": clean_nmi,
                "attacked_nmi": attack_nmi,
                "nmi_deficit": deficit,
                "assignment_flips": flips,
                "assignment_flip_rate": flips / attack.graphs
                if flips is not None
                else None,
                "soft_output_drift": drift,
            }
        )
    stress = [row["nmi_deficit"] for row in course if row["segment"] == "stress"]
    return {
        "evaluation_graphs": attack.graphs,
        "evaluation_ids_sha256": attack.evaluation,
        "stress_peak_nmi_deficit": max(stress),
        "stress_deficit_exceeded_gap": any(x > limit for x in stress),
        "sustained_recovery": {
            "event": recovered is not None,
            "first_round": recovered,
            "confirmation_round": recovered + window - 1
            if recovered is not None
            else None,
            "time_since_stress_end": (attack.end if recovered is None else recovered)
            - attack.stress_end,
            "right_censor_round": attack.end if recovered is None else None,
        },
        "segments": {
            name: _segment(course, name) for name in ("warmup", "stress", "recovery")
        },
        "rounds": course,
    }


def report(root: Path, *, gap: float, window: int):
    """Pair complete trajectories on their registered WL settings and graph IDs."""
    if not isfinite(gap) or not 0 <= gap <= 1 or type(window) is not int or window < 1:
        raise ValueError("gap must be in [0, 1] and window must be a positive integer")
    paths = sorted(
        set(root.rglob("rounds.jsonl"))
        | {p.with_name("rounds.jsonl") for p in root.rglob("failure.json")}
    )
    if not paths:
        raise ValueError(f"No trajectory traces under {root}")
    trajectories, excluded = [], []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            if path.with_name("failure.json").exists():
                failure = _json(path.with_name("failure.json"))
                excluded.append(
                    {
                        "path": relative,
                        "status": "failed_trajectory",
                        "reason": failure.get("failure"),
                        "regime": failure.get("regime"),
                    }
                )
            else:
                trajectories.append(_registration(path))
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            IndexError,
            AttributeError,
        ) as error:
            excluded.append(
                {
                    "path": relative,
                    "status": "invalid_registration",
                    "reason": str(error),
                }
            )
    clean = defaultdict(list)
    attacked = []
    for trace in trajectories:
        if trace.regime == "clean":
            clean[trace.key].append(trace)
        else:
            attacked.append(trace)
    copies = Counter((t.key, t.regime, t.design, t.evaluation) for t in attacked)
    pairs = []
    for attack in attacked:
        row = {
            **dict(zip(PAIR_FIELDS, attack.key, strict=True)),
            "regime": attack.regime,
            "attacked_path": attack.path.relative_to(root).as_posix(),
            "settings_sha256": attack.design,
        }
        controls = clean[attack.key]
        compatible = [c for c in controls if c.design == attack.design]
        aligned = [c for c in compatible if c.evaluation == attack.evaluation]
        if copies[(attack.key, attack.regime, attack.design, attack.evaluation)] > 1:
            row["status"] = "duplicate_attacked"
        elif not controls:
            row["status"] = "missing_clean"
        elif not compatible:
            row["status"] = "configuration_mismatch"
        elif not aligned:
            row["status"] = "evaluation_role_mismatch"
        elif len(aligned) > 1:
            row["status"] = "ambiguous_clean"
        else:
            row["clean_path"] = aligned[0].path.relative_to(root).as_posix()
            try:
                row.update(_compare(aligned[0], attack, gap, window))
                row["status"] = "paired"
            except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
                row.update(status="invalid_trace", reason=str(error))
        pairs.append(row)
    statuses = Counter(row["status"] for row in pairs)
    events = sum(
        row["sustained_recovery"]["event"] for row in pairs if row["status"] == "paired"
    )
    return {
        "gap": gap,
        "window": window,
        "definitions": {
            "nmi_deficit": "clean NMI minus attacked NMI",
            "recovery": "first post-stress round starting a full window of deficits at or below gap",
            "gap_comparison_roundoff": 2 * ulp(1.0),
            "assignment": "prediction-coordinate mismatch in verified identical evaluation ID order",
            "soft_drift": "mean absolute coordinate difference over evaluation graphs and clusters",
            "alignment": "inherited WL coordinates with no cross-run permutation or label matching",
        },
        "counts": {
            "trace_sources": len(paths),
            "excluded_sources": len(excluded),
            "registered_clean": sum(map(len, clean.values())),
            "registered_attacked": len(attacked),
            "paired": statuses["paired"],
            "unpaired": len(attacked) - statuses["paired"],
            "statuses": dict(statuses),
            "recovery_events": events,
            "right_censored": statuses["paired"] - events,
        },
        "excluded_sources": excluded,
        "trajectories": pairs,
    }


def _write_json(destination, result):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(destination)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Root containing saved WL trajectory directories",
    )
    parser.add_argument(
        "--gap",
        type=float,
        required=True,
        help="Largest acceptable clean-minus-attacked NMI deficit",
    )
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Consecutive post-stress rounds required",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination JSON report"
    )
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        run_artifacts = {
            "rounds.jsonl",
            "registration.json",
            "roles.json",
            "failure.json",
            "summary.json",
            "checkpoint.npz",
            "manifest.json",
            "run_index.json",
        }
        if (
            output.is_relative_to(args.runs.resolve())
            and output.name.lower() in run_artifacts
        ):
            raise ValueError("output path is reserved for saved WL run artifacts")
        result = report(args.runs, gap=args.gap, window=args.window)
        _write_json(output, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
