"""Real six-cluster graph pilot, with train-fitted representations and sealed scoring."""

from collections import Counter
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import snapshots
from snapshots import base, freeze, read_json
from learner import PilotData, ProteinGraph, _distances, _partition_peers, _wl_features

ARCHIVE = Path("data/ENZYMES.zip")
OUT = Path("runs/enzymes")
ARCHIVE_SHA = "13d832eb6ffa084192daf6e5750250028a18437ee692c38d29a10cd60e18aaf4"
SOURCE_URL = "https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip"
DATASET_PAPER = "https://www.aiml.informatik.tu-darmstadt.de/papers/morris2020grlb.pdf"
SPLITS = (30201, 30202, 30203)
PARTITIONS = (30301, 30302, 30303)
ARMS = ("VCIA", "J-Rank-Feasible", "Unscreened")
FEATURES = (("WL-TFIDF", .1), ("WL-TFIDF", .5), ("WL-TFIDF", 1.), ("Spectral-PCA", .5))
K = 6


def data(path=ARCHIVE):
    assert base.digest(path) == ARCHIVE_SHA
    with ZipFile(path) as archive:
        def lines(part):
            return archive.read(f"ENZYMES/ENZYMES_{part}.txt").decode().splitlines()
        groups = np.asarray([int(line) - 1 for line in lines("graph_indicator")])
        labels = np.asarray([int(line) for line in lines("graph_labels")])
        node_labels = np.asarray([int(line) for line in lines("node_labels")])
        edges = [tuple(int(field) - 1 for field in line.split(",")) for line in lines("A")]
    assert len(labels) == 600 and len(np.unique(labels)) == K
    neighbors = [set() for _ in groups]
    for u, v in edges:
        assert groups[u] == groups[v]
        neighbors[u].add(v)
        neighbors[v].add(u)
    graphs = []
    for graph_id in range(len(labels)):
        nodes = np.flatnonzero(groups == graph_id)
        positions = {u: i for i, u in enumerate(nodes)}
        graphs.append(ProteinGraph(tuple(tuple(sorted(positions[v] for v in neighbors[u])) for u in nodes),
                                   tuple(node_labels[nodes])))
    return tuple(graphs), labels


def topology_features(graphs):
    rows = []
    for graph in graphs:
        n = len(graph.neighbors)
        adjacency = np.zeros((n, n))
        for i, neighbors in enumerate(graph.neighbors):
            adjacency[i, list(neighbors)] = 1
        degree = adjacency.sum(axis=1)
        scale = np.divide(1., np.sqrt(degree), out=np.zeros(n), where=degree > 0)
        spectrum = np.linalg.eigvalsh(np.diag((degree > 0).astype(float)) - scale[:, None] * adjacency * scale)
        heat = np.exp(-spectrum[:, None] * 2. ** np.linspace(-2, 4, 12)).mean(axis=0)
        node_type = [np.mean(np.asarray(graph.node_labels) == i) for i in (1, 2, 3)]
        triangle_walks = np.sum((adjacency @ adjacency) * adjacency, axis=1)
        denom = degree * (degree - 1)
        clustering = np.divide(triangle_walks, denom, out=np.zeros(n), where=denom > 0).mean()
        rows.append([np.log1p(n), np.log1p(degree.mean()), np.log1p(degree.std()),
                     np.log1p(degree.max()), clustering, *np.quantile(degree, [.25, .5, .75]),
                     *node_type, *heat])
    return np.asarray(rows)


def representation(wl, spectral, train, which):
    if which == "WL-TFIDF":
        frequency = np.count_nonzero(wl[train], axis=0)
        kept = frequency > 0
        x = wl[:, kept] * (1 + np.log((len(train) + 1) / (frequency[kept] + 1)))
    else:
        scaler = StandardScaler().fit(spectral[train])
        x = scaler.transform(spectral)
        projection = PCA(n_components=12, svd_solver="full").fit(x[train])
        x = projection.transform(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, where=norms > 0, out=np.zeros_like(x))


def align(centers, reference):
    row, col = linear_sum_assignment(_distances(reference, centers))
    assert np.array_equal(row, np.arange(K))
    return centers[col]


def starting(features, seed):
    centers, _ = kmeans_plusplus(features, n_clusters=K, random_state=seed)
    return base._lloyd(features, centers, epochs=12, step=1.)


def initialize(pilot, seed):
    bank = [starting(pilot.features[ids], int(np.random.SeedSequence([seed, client]).generate_state(1)[0]))
            for client, ids in enumerate(pilot.peer_train_ids)]
    return base.canonicalize(np.stack([align(centers, bank[0]) for centers in bank]))


def propose(pilot, bank, round_seed, neighbor_weight=.2):
    maps = np.empty_like(bank)
    for client, ids in enumerate(pilot.peer_train_ids):
        rng = np.random.default_rng(np.random.SeedSequence([round_seed, client]))
        near = (bank[client - 1] + bank[(client + 1) % len(bank)]) / 2.
        start = (1 - neighbor_weight) * bank[client] + neighbor_weight * near
        chosen = rng.choice(ids, size=min(32, len(ids)), replace=False)
        fitted = base._lloyd(pilot.features[chosen], start, epochs=1, step=.25)
        maps[client] = align(fitted, bank[client])
    return maps


def diagnostic(pilot, initial, seed, init, arm):
    start = perf_counter()
    bank = initial.copy() if arm == "Local" else starting(pilot.features[pilot.train_ids], init)[None]
    batch = sum(min(32, len(ids)) for ids in pilot.peer_train_ids)
    for r in range(1, base.ROUNDS + 1):
        if arm == "Local":
            local_seed = int(base.rng(seed, r, 1).integers(0, 2**32, dtype=np.uint32))
            bank = base.canonicalize(propose(pilot, bank, local_seed, neighbor_weight=0.))
        else:
            ids = base.rng(seed, r, 8).choice(pilot.train_ids, batch, replace=False)
            fitted = base._lloyd(pilot.features[ids], bank[0], epochs=1, step=.25)
            bank = base.canonicalize(align(fitted, bank[0]))[None]
    return bank, perf_counter() - start


def save_predictions(pilot, outer, index, outputs, key, bank, claims, cost, config):
    scores = base.all_peer_predictions(pilot, bank, outer)
    combined = base.aggregate(scores, scores, claims, claims, (), config) if len(bank) != 1 else scores[0]
    token = f"m{len(index):04d}"
    outputs[token] = combined
    sizes = np.bincount(combined.argmax(axis=1), minlength=K)
    index.append(dict(**key, token=token, outer_ids=outer.tolist(), end_cpu_wall_s=cost,
                      final_bank_hash=base.state_id(bank, claims),
                      cluster_counts=sizes.tolist()))


def execute():
    graphs, labels = data(ARCHIVE)
    wl, fingerprints = _wl_features(graphs, 3, 192)
    spectral = topology_features(graphs)
    cfg = base.DecisionConfig.normalized(6, n_clusters=K, epsilon=.1)
    base.write_json(OUT / "enzymes_protocol.json", {
        "frozen_at_utc": base.now(), "archive_sha256": ARCHIVE_SHA, "archive_reference": SOURCE_URL,
        "dataset_paper": DATASET_PAPER, "code_sha256": {Path(__file__).name: base.digest(Path(__file__)),
                                                        **base.source_hashes()},
        "outer_seeds": SPLITS, "client_partition_seeds": PARTITIONS, "features_alpha": FEATURES,
        "init": "one matched initialization per 3 client splits per outer split; seed derived from client+outer seeds",
        "features": "WL depth 3 hashed TF-IDF vs 23 fixed graph spectral and structure descriptors, train StandardScaler + PCA 12; per-graph L2; ENZYMES node types used as node attributes, no graph class used in feature fitting",
        "split": "group on pre-hashing WL fingerprints; outer 60/20/20 development/reserved/test then development 60/20/20 train/audit/validation",
        "partition": "class-wise Dirichlet + multinomial; reject until every client has at least 8 graphs; maximum 10000 attempts; graph labels used only to construct heterogeneity and score sealed outputs",
        "topology": "logical ring degree 2 for proposal mixing, same host stores eight operator versions and emulates central round coordinator and independent audit. One loopback TCP connection. Entire eight-client network is the candidate roster, exact quota 6. No distributed consensus implementation.",
        "rounds": base.ROUNDS, "regimes": base.REGIMES, "policies": ARMS,
        "config": asdict(cfg), "design_draws": base.DESIGN_DRAWS, "audit_draws": base.VERIFY_DRAWS,
        "perturbation": "matched peer 1 & 5 rows reversed in cluster coordinates with claim 1, rounds 9–20",
        "local": "one minibatch Lloyd update per round, 32 cap, step .25, two neighbors .2; Hungarian multiclass prototype alignment",
        "planned_rounds": 6480, "independent_outer_splits": 3, "devices": "Windows CPU, single thread; actual application bytes over local TCP",
        "scoring": "paired outer NMI/ARI, Hungarian ACC, sizes and empty clusters; check per-split directions; descriptive only, practical noninferiority margins not specified",
    })
    names = ("enzymes_rounds.csv", "enzymes_splits.csv", "enzymes_allocations.csv",
             "enzymes_index.json", "enzymes_outer_probabilities.npz")
    rounds, splits, allocations = (base.Rows(OUT / name) for name in names[:3])
    base.CONFIG = cfg
    base.propose = propose
    index, outputs = [], {}
    peer, coordinator = base.connection()
    try:
        for outer_seed in SPLITS:
            development, reserve, outer = base._split_roles(fingerprints, outer_seed)
            inner_seed = outer_seed + 200
            train, audit, validation = base._split_roles(fingerprints, inner_seed, development)
            owner = {}
            for role, ids in zip(("train", "audit", "validation", "reserve", "outer_test"),
                                 (train, audit, validation, reserve, outer), strict=True):
                for graph_id in ids:
                    sig = fingerprints[graph_id]
                    assert owner.setdefault(sig, role) == role
                    splits.add(dict(outer_seed=outer_seed, role=role, graph_id=int(graph_id), fingerprint=sig))
            assert len(set(np.concatenate((train, audit, validation, reserve, outer)))) == len(graphs)
            feature_bank = {name: representation(wl, spectral, train, name) for name in dict(FEATURES)}
            clients = {}
            for alpha, partition in product(sorted({alpha for _, alpha in FEATURES}), PARTITIONS):
                allocation, attempts = _partition_peers(train, labels, partition, 8, alpha, 8, max_attempts=10000)
                assert np.array_equal(np.sort(np.concatenate(allocation)), train)
                clients[(alpha, partition)] = (allocation, attempts)
                for client, ids in enumerate(allocation):
                    distribution = Counter(labels[ids])
                    allocations.add(dict(outer_seed=outer_seed, alpha=alpha, client_seed=partition, client=client,
                                         allocation_attempts=attempts, train_graphs=len(ids),
                                         **{f"class_{i}": distribution[i] for i in range(1, K + 1)},
                                         graph_ids=" ".join(map(str, ids))))
            for (which, alpha), partition in product(FEATURES, PARTITIONS):
                allocation, attempts = clients[(alpha, partition)]
                pilot = PilotData(feature_bank[which], labels, train, audit, validation, allocation,
                                  fingerprints, inner_seed, partition, attempts, max(Counter(fingerprints).values()),
                                  K, 3, 192, .15)
                init = int(np.random.SeedSequence([outer_seed, partition, 17]).generate_state(1)[0])
                seed = int(np.random.SeedSequence([outer_seed, partition, init]).generate_state(1)[0])
                initial = initialize(pilot, init)
                claims = np.full(8, .25)
                zeros = np.zeros(8, dtype=int)
                key = dict(dataset="ENZYMES", representation=which, alpha=alpha, outer_seed=outer_seed,
                           partition=partition, init=init)
                for arm in ("Local", "Centralized"):
                    bank, cost = diagnostic(pilot, initial, seed, init, arm)
                    save_predictions(pilot, outer, index, outputs, dict(**key, arm=arm, regime="Clean"),
                                     bank, np.full(len(bank), .25), cost, cfg)
                for regime, arm in product(base.REGIMES, ARMS):
                    state = base.State(initial.copy(), claims.copy(), zeros.copy(), zeros.copy())
                    cost = 0.
                    for r in range(1, base.ROUNDS + 1):
                        state, log, *_ = base.gauge_round(pilot, state, dict(**key, regime=regime, arm=arm),
                                                         seed, r, peer, coordinator)
                        assert log["attempted"] == log["committed"] + log["rejected"] == 1 - log["no_candidate"]
                        rounds.add(log)
                        cost += log["online_seconds"]
                    save_predictions(pilot, outer, index, outputs, dict(**key, arm=arm, regime=regime),
                                     state.maps, state.claims, cost, cfg)
                print("ENZYMES", outer_seed, which, alpha, partition, flush=True)
    finally:
        for stream in (rounds, splits, allocations):
            stream.close()
        peer.close()
        coordinator.close()
    base.write_json(OUT / names[3], index)
    np.savez_compressed(OUT / names[4], **outputs)
    assert len(index) == 288
    freeze([OUT / name for name in (*names, "enzymes_protocol.json")], "enzymes_freeze.json")


def score():
    frozen = read_json(OUT / "enzymes_freeze.json")
    assert all(base.digest(OUT / name) == sha for name, sha in frozen["sha256"].items())
    start = base.now()
    _, labels = data(ARCHIVE)
    rows = base.Rows(OUT / "enzymes_outer.csv")
    try:
        with np.load(OUT / "enzymes_outer_probabilities.npz") as predictions:
            for record in read_json(OUT / "enzymes_index.json"):
                probabilities = predictions[record["token"]]
                ids = record["outer_ids"]
                counts = np.bincount(probabilities.argmax(axis=1), minlength=K)
                assert counts.tolist() == record["cluster_counts"]
                key = {k: v for k, v in record.items() if k not in ("cluster_counts", "outer_ids")}
                rows.add(dict(**key, outer_graphs=len(ids), effective_clusters=int(np.count_nonzero(counts)),
                              empty_clusters=int((counts == 0).sum()), min_cluster_share=float(counts.min()/len(ids)),
                              cluster_sizes=" / ".join(map(str, counts)), **base.quality(probabilities, labels[ids])))
    finally:
        rows.close()
    base.write_json(OUT / "enzymes_scoring.json", {"started_at_utc": start, "finished_at_utc": base.now(),
                    "sha256": base.digest(OUT / "enzymes_outer.csv")})


if __name__ == "__main__":
    cli = ArgumentParser()
    cli.add_argument("mode", choices=("execute", "score"))
    cli.add_argument("--data", type=Path, default=ARCHIVE.parent)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    ARCHIVE = args.data / "ENZYMES.zip"
    OUT = snapshots.OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        {"execute": execute, "score": score}[args.mode]()
