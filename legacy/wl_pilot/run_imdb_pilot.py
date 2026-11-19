"""Repeat the inner WL pilot on canonical IMDB-BINARY using unlabeled degree tokens."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from .decision import DecisionConfig
from .learner import ProteinGraph, RawProteins, prepare_pilot
from .provenance import (
    IMDB_SOURCES,
    PROTEINS_SOURCES,
    source_hashes,
    validate_source_hashes,
)
from .run_pilot import run, runtime_metadata, write_json


def load_imdb(path):
    """Preserve TU record order and use each node's unweighted degree as its WL token."""
    path = path.resolve(strict=True)
    with ZipFile(path) as raw:

        def lines(which):
            return (
                raw.read(f"IMDB-BINARY/IMDB-BINARY_{which}.txt")
                .decode("ascii")
                .splitlines()
            )

        membership = np.array([int(line) - 1 for line in lines("graph_indicator")])
        classes, labels = np.unique(
            [int(line) for line in lines("graph_labels")], return_inverse=True
        )
        edges = [tuple(int(node) - 1 for node in row.split(",")) for row in lines("A")]
    if (
        len(classes) != 2
        or len(labels) != 1000
        or not np.array_equal(np.unique(membership), np.arange(len(labels)))
    ):
        raise ValueError("Expected the full two-class, 1,000-graph IMDB-BINARY archive")
    nodes_by_graph = [[] for _ in labels]
    for node, graph in enumerate(membership):
        nodes_by_graph[graph].append(node)
    adjacency = [set() for _ in membership]
    for u, v in edges:
        if not (
            0 <= u < len(membership)
            and 0 <= v < len(membership)
            and membership[u] == membership[v]
        ):
            raise ValueError("TU edge leaves its graph")
        adjacency[u].add(v)
        adjacency[v].add(u)
    graphs = []
    for nodes in nodes_by_graph:
        local = {node: index for index, node in enumerate(nodes)}
        neighbors = tuple(
            tuple(sorted(local[adjacent] for adjacent in adjacency[node]))
            for node in nodes
        )
        graphs.append(ProteinGraph(neighbors, tuple(map(len, neighbors))))
    return RawProteins(
        tuple(graphs), labels, path, sha256(path.read_bytes()).hexdigest()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ref_bytes = args.reference_manifest.read_bytes()
    manifest = json.loads(ref_bytes)
    validate_source_hashes(manifest["source_sha256"], PROTEINS_SOURCES)
    raw = load_imdb(args.dataset)
    if (
        raw.source_sha256
        != "b291ec8b26d85c70faa2ba0a2433e1f407ed2ef5d0fc072d36b9a95e49a1bb27"
    ):
        raise ValueError(
            "Dataset bytes differ from the canonical archive used for this extension"
        )
    manifest["dataset"] = raw.source_path.name
    manifest["dataset_sha256"] = raw.source_sha256
    manifest["dataset_name"] = "IMDB-BINARY"
    manifest["learner"]["node_tokens"] = "unweighted degree"
    manifest["reference_manifest_sha256"] = sha256(ref_bytes).hexdigest()
    manifest["source_sha256"] = source_hashes(IMDB_SOURCES)
    manifest.update(runtime_metadata())
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "manifest.json", manifest)
    config = DecisionConfig(**manifest["decision"])
    summaries = []
    with threadpool_limits(limits=1):
        for seed in manifest["seeds"]:
            started = perf_counter()
            pilot = prepare_pilot(
                raw, seed, sealed_outer_seed=manifest["sealed_outer_seed"]
            )
            write_json(
                args.out / f"split_{seed}.json",
                {
                    "train_ids": pilot.train_ids,
                    "audit_ids": pilot.audit_ids,
                    "inner_validation_ids": pilot.test_ids,
                    "peer_train_ids": pilot.peer_train_ids,
                    "partition_seed": pilot.partition_seed,
                    "partition_attempts": pilot.partition_attempts,
                    "largest_wl_fingerprint_group": pilot.max_fingerprint_group,
                    "feature_dimension": pilot.features.shape[1],
                    "preparation_seconds": perf_counter() - started,
                },
            )
            for regime in manifest["regimes"]:
                for arm in manifest["arms"]:
                    directory = args.out / f"seed_{seed}_{regime.lower()}_{arm.lower()}"
                    summary = run(
                        pilot,
                        seed,
                        regime,
                        arm,
                        config,
                        directory,
                        manifest["rounds"],
                        manifest["design_draws"],
                        manifest["verify_draws"],
                    )
                    summaries.append(summary)
                    print(
                        summary["path"],
                        summary["accepted_rounds"],
                        summary["endpoint_inner_validation"]["nmi"],
                        flush=True,
                    )
    write_json(args.out / "run_index.json", summaries)


if __name__ == "__main__":
    main()
