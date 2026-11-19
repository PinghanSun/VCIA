"""Plan and launch registered PCOC studies from the command line."""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path
from tempfile import NamedTemporaryFile
from traceback import format_exc
from urllib.request import urlopen

from threadpoolctl import threadpool_info, threadpool_limits

from .configuration import read_study
from .experiment import run_job, write_json
from .learner import TU_DATASETS, load_tu, prepare_pilot


def inspect_archive(path, dataset, strict_hash):
    _, _, tokens, expected = TU_DATASETS[dataset]
    with path.open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    if strict_hash and expected and fingerprint != expected:
        raise ValueError(
            f"{dataset}: expected registered SHA-256 {expected}, found {fingerprint}"
        )
    suffixes = ["A", "graph_indicator", "graph_labels"]
    if tokens == "labels":
        suffixes.append("node_labels")
    with zipfile.ZipFile(path) as data:
        missing = {f"{dataset}/{dataset}_{suffix}.txt" for suffix in suffixes} - set(
            data.namelist()
        )
        if missing:
            raise ValueError(f"{dataset}: missing ZIP members {sorted(missing)}")
        broken = data.testzip()
        if broken:
            raise ValueError(f"{dataset}: CRC check failed for {broken}")
    return {
        "sha256": fingerprint,
        "bytes": path.stat().st_size,
        "registered_hash": expected,
        "matches_registered_hash": fingerprint == expected if expected else None,
    }


def fetch(study, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for dataset in study.datasets:
        target = folder / f"{dataset}.zip"
        if target.exists():
            inspect_archive(target, dataset, study.strict_dataset_hash)
            continue
        with NamedTemporaryFile(
            dir=folder, prefix=f".{dataset}.", suffix=".part", delete=False
        ) as temp:
            staged = Path(temp.name)
        try:
            url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset}.zip"
            with urlopen(url, timeout=60) as download, staged.open("wb") as out:
                shutil.copyfileobj(download, out)
            inspect_archive(staged, dataset, study.strict_dataset_hash)
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)


def resolve_jobs(study, selection):
    jobs = study.job_keys()
    if selection is None:
        return list(enumerate(jobs, start=1))
    first, separator, last = selection.partition(":")
    lo = int(first)
    hi = int(last) if separator else lo
    if not 1 <= lo <= hi <= len(jobs):
        raise ValueError(
            f"--job must be a 1-based ID or inclusive range in 1..{len(jobs)}"
        )
    return [(index, jobs[index - 1]) for index in range(lo, hi + 1)]


def plan(study, jobs, show_jobs):
    result = {
        **study.plan(),
        "selected_trajectories": len(jobs),
        "selected_rounds": len(jobs) * study.rounds,
    }
    if show_jobs:
        fields = ("dataset", "partition_seed", "seed", "regime", "arm")
        result["jobs"] = [
            {"job": index, **dict(zip(fields, key))} for index, key in jobs
        ]
    return result


def run(study, folder, output, jobs, config_path):
    selected_datasets = dict.fromkeys(key[0] for _, key in jobs)
    archives = {
        name: inspect_archive(folder / f"{name}.zip", name, study.strict_dataset_hash)
        for name in selected_datasets
    }
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    versions = {
        package: metadata.version(package)
        for package in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
    }
    manifest = {
        "study": asdict(study),
        "plan": plan(study, jobs, True),
        "archives": archives,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "numerical_libraries": threadpool_info(),
    }
    write_json(output / "manifest.json", manifest)
    raw = {}
    prepared = {}
    failures = {}
    completed = []
    with threadpool_limits(limits=1):
        for index, (dataset, partition, seed, regime, arm) in jobs:
            entry = {
                "job": index,
                "dataset": dataset,
                "phase": study.phase,
                "partition_seed": partition,
                "seed": seed,
                "regime": regime,
                "arm": arm,
                "scheduled_rounds": study.rounds,
            }
            key = (dataset, partition)
            directory = output / dataset / f"p{partition}_i{seed}" / regime / arm
            try:
                if key in failures:
                    raise ValueError(failures[key])
                if key not in prepared:
                    try:
                        if dataset not in raw:
                            raw[dataset] = load_tu(
                                folder / f"{dataset}.zip",
                                dataset,
                                strict_hash=study.strict_dataset_hash,
                            )
                            if (
                                raw[dataset].source_sha256
                                != archives[dataset]["sha256"]
                            ):
                                raise ValueError("dataset ZIP changed after preflight")
                        inner = study.phase == "inner"
                        prepared[key] = prepare_pilot(
                            raw[dataset],
                            partition if inner else study.split_seed,
                            partition_seed=None
                            if study.paired_inner_seeds
                            else partition,
                            sealed_outer_seed=study.split_seed if inner else None,
                            peers=study.peers,
                            alpha=study.partition_alpha,
                            min_peer=study.min_peer_graphs,
                        )
                    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
                        failures[key] = f"data preparation: {error}"
                        raise
                run_job(
                    prepared[key],
                    study,
                    dataset,
                    partition,
                    seed,
                    regime,
                    arm,
                    directory,
                    archives[dataset]["sha256"],
                )
                entry["status"] = "complete"
            except Exception as error:  # noqa: BLE001 - record per-job failures and continue the matrix.
                entry.update(
                    status="failed", failure=f"{type(error).__name__}: {error}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                write_json(
                    directory / "failure.json", {**entry, "traceback": format_exc()}
                )
                print(f"Job {index} failed: {error}", file=sys.stderr)
            completed.append(entry)
            write_json(output / "run_index.json", completed)
    return int(any(row["status"] != "complete" for row in completed))


def matrix(path, output):
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    base = read_study(path.parent / spec["base"])
    variations = {}
    for group, variants in spec["groups"].items():
        if not group.replace("-", "").isalnum():
            raise ValueError(f"invalid group name {group}")
        for index, overrides in enumerate(variants, start=1):
            if "compromised" in overrides:
                overrides["compromised"] = tuple(overrides["compromised"])
            if "decision" in overrides:
                overrides["decision"] = base.decision | overrides["decision"]
            study = replace(base, **overrides)
            study.validate()
            variations[f"{group}-{index:02d}"] = study
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        for name, study in variations.items():
            write_json(output / f"{name}.json", asdict(study))
    return {
        "configurations": {name: study.plan() for name, study in variations.items()},
        "trajectories": sum(len(study.job_keys()) for study in variations.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pcoc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "check-data", "fetch", "run", "matrix"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"check-data", "fetch", "run"}:
            command.add_argument("--data-dir", type=Path, default=Path("data"))
        if name in {"plan", "run"}:
            command.add_argument(
                "--job", help="one 1-based job ID or inclusive FIRST:LAST range"
            )
        if name in {"run", "matrix"}:
            command.add_argument("--output", type=Path, required=name == "run")
    commands.choices["plan"].add_argument(
        "--list", action="store_true", help="include each selected job ID"
    )
    commands.choices["run"].add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="explicitly authorize learner execution",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "matrix":
            result = matrix(args.config, args.output)
        else:
            study = read_study(args.config)
            if args.command in {"plan", "run"}:
                jobs = resolve_jobs(study, args.job)
                if args.command == "run":
                    return run(study, args.data_dir, args.output, jobs, args.config)
                result = plan(study, jobs, args.list)
            else:
                if args.command == "fetch":
                    fetch(study, args.data_dir)
                result = {
                    dataset: inspect_archive(
                        args.data_dir / f"{dataset}.zip",
                        dataset,
                        study.strict_dataset_hash,
                    )
                    for dataset in study.datasets
                }
        print(json.dumps(result, indent=2))
    except (
        ArithmeticError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        parser.exit(2, f"pcoc: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
