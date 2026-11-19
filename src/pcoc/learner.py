"""TU graph data and the WL prototype learner used by PCOC.

The TU parser and WL histogram rules follow the historical run_bscm_pcoc.py, with
only its graph inputs and hashed WL features carried over. The pilot fits one
shared inverse-frequency transform on the training split and learns local NumPy
prototype operators. The shared transform is an offline benchmark preprocessing
step. Each peer updates from the previous executed maps of itself and its two
ring neighbors.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

TU_DATASETS = {
    "PROTEINS": (
        1113,
        2,
        "labels",
        "2da8de15284b88edabca2888ce5444d62f364ed41159260977088c4e53d4d848",
    ),
    "IMDB-BINARY": (
        1000,
        2,
        "degree",
        "b291ec8b26d85c70faa2ba0a2433e1f407ed2ef5d0fc072d36b9a95e49a1bb27",
    ),
    "MUTAG": (188, 2, "labels", None),
    "NCI1": (4110, 2, "labels", None),
    "DD": (1178, 2, "labels", None),
    "ENZYMES": (600, 6, "labels", None),
    "IMDB-MULTI": (1500, 3, "degree", None),
}


@dataclass(frozen=True)
class ProteinGraph:
    neighbors: tuple[tuple[int, ...], ...]
    node_labels: tuple[int, ...]


@dataclass(frozen=True)
class RawProteins:
    graphs: tuple[ProteinGraph, ...]
    labels: np.ndarray
    source_path: Path
    source_sha256: str
    n_clusters: int


@dataclass(frozen=True)
class PilotData:
    features: np.ndarray
    labels: np.ndarray
    train_ids: np.ndarray
    audit_ids: np.ndarray
    test_ids: np.ndarray
    peer_train_ids: tuple[np.ndarray, ...]
    wl_fingerprints: tuple[str, ...]
    split_seed: int
    partition_seed: int
    partition_attempts: int
    max_fingerprint_group: int
    n_clusters: int
    wl_depth: int
    wl_bins: int
    temperature: float


def _tu_lines(archive: zipfile.ZipFile, name: str, file: str) -> list[str]:
    return archive.read(f"{name}/{name}_{file}.txt").decode("utf-8").splitlines()


def load_tu(
    zip_path: str | Path, name: str, *, strict_hash: bool = True
) -> RawProteins:
    """Read the named TU dataset in original graph order without extracting the archive."""
    path = Path(zip_path).resolve(strict=True)
    expected_size, expected_classes, tokens, expected_digest = TU_DATASETS[name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if strict_hash and expected_digest and digest != expected_digest:
        raise ValueError(
            f"Dataset SHA-256 for {name} differs from the registered archive: {digest}"
        )
    with zipfile.ZipFile(path) as archive:
        indicator = [
            int(line) - 1 for line in _tu_lines(archive, name, "graph_indicator")
        ]
        raw_labels = np.asarray(
            [int(line) for line in _tu_lines(archive, name, "graph_labels")],
            dtype=np.int64,
        )
        node_labels = (
            [int(line) for line in _tu_lines(archive, name, "node_labels")]
            if tokens == "labels"
            else None
        )
        edges = _tu_lines(archive, name, "A")
    if len(raw_labels) != expected_size:
        raise ValueError(
            f"Expected the complete {expected_size}-graph {name} archive at {path}"
        )
    classes, labels = np.unique(raw_labels, return_inverse=True)
    if len(classes) != expected_classes or set(indicator) != set(range(len(labels))):
        raise ValueError("Unexpected class count or invalid graph indicator")
    if node_labels is not None and len(indicator) != len(node_labels):
        raise ValueError("TU graph indicator and node label counts differ")

    nodes_by_graph = [[] for _ in labels]
    for node, graph in enumerate(indicator):
        nodes_by_graph[graph].append(node)
    adjacency = [set() for _ in indicator]
    for edge in edges:
        first, second = (int(item) - 1 for item in edge.replace(",", " ").split())
        if not (
            0 <= first < len(indicator)
            and 0 <= second < len(indicator)
            and indicator[first] == indicator[second]
        ):
            raise ValueError("TU edge leaves its graph")
        adjacency[first].add(second)
        adjacency[second].add(first)

    graphs = []
    for nodes in nodes_by_graph:
        local_index = {node: index for index, node in enumerate(nodes)}
        neighbors = tuple(
            tuple(sorted(local_index[n] for n in adjacency[node])) for node in nodes
        )
        labels_at_nodes = (
            tuple(node_labels[node] for node in nodes)
            if node_labels is not None
            else tuple(map(len, neighbors))
        )
        graphs.append(ProteinGraph(neighbors, labels_at_nodes))
    return RawProteins(tuple(graphs), labels, path, digest, expected_classes)


def _wl_features(
    graphs: tuple[ProteinGraph, ...], depth: int, bins: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compute the historical hashed histograms and a pre-binning split fingerprint."""
    features = np.zeros((len(graphs), (depth + 1) * bins), dtype=float)
    fingerprints = []
    for graph, hist in zip(graphs, features):
        tokens = [f"l{label}" for label in graph.node_labels]
        signature = hashlib.sha256()
        for level in range(depth + 1):
            signature.update((",".join(sorted(tokens)) + "\n").encode("utf-8"))
            for token in tokens:
                bucket = (
                    int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16)
                    % bins
                )
                hist[level * bins + bucket] += 1.0
            if level == depth:
                break
            updated = []
            for token, neighbors in zip(tokens, graph.neighbors):
                payload = token + "|" + "|".join(sorted(tokens[n] for n in neighbors))
                updated.append(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20])
            tokens = updated
        hist /= max(len(tokens), 1)
        fingerprints.append(signature.hexdigest())
    return np.log1p(features), tuple(fingerprints)


def _split_roles(
    fingerprints: tuple[str, ...], seed: int, ids=None
) -> tuple[np.ndarray, ...]:
    groups: dict[str, list[int]] = {}
    if ids is None:
        ids = np.arange(len(fingerprints))
    for index in ids:
        groups.setdefault(fingerprints[index], []).append(int(index))

    # Place the boundaries at the nearest whole-group cuts of a seeded ordering.
    batches = list(groups.values())
    randomized = [
        batches[i] for i in np.random.default_rng(seed).permutation(len(batches))
    ]
    stops = np.cumsum([0, *map(len, randomized)])
    n = len(ids)
    train_target = int(0.6 * n)
    audit_target = int(0.2 * n)
    first = int(np.argmin(np.abs(stops - train_target)))
    second = int(np.argmin(np.abs(stops - train_target - audit_target)))
    return tuple(
        np.asarray(sorted(i for group in role for i in group), dtype=np.int64)
        for role in (randomized[:first], randomized[first:second], randomized[second:])
    )


def _partition_peers(
    train_ids: np.ndarray,
    labels: np.ndarray,
    seed: int,
    peers: int,
    alpha: float,
    min_peer: int,
    max_attempts: int = 100,
) -> tuple[tuple[np.ndarray, ...], int]:
    rng = np.random.default_rng(seed)
    by_class = [
        train_ids[labels[train_ids] == label] for label in np.unique(labels[train_ids])
    ]
    for attempt in range(1, max_attempts + 1):
        assignments = [[] for _ in range(peers)]
        for ids in by_class:
            proportions = rng.dirichlet(np.full(peers, alpha))
            counts = rng.multinomial(len(ids), proportions)
            batches = np.split(rng.permutation(ids), np.cumsum(counts)[:-1])
            for peer, batch in enumerate(batches):
                assignments[peer].extend(batch)
        if min(map(len, assignments)) >= min_peer:
            return tuple(
                np.asarray(sorted(ids), dtype=np.int64) for ids in assignments
            ), attempt
    raise ValueError(
        f"Could not allocate {min_peer} graphs per peer in {max_attempts} Dirichlet draws"
    )


def prepare_pilot(
    raw: RawProteins,
    seed: int,
    *,
    partition_seed: int | None = None,
    sealed_outer_seed: int | None = None,
    peers: int = 8,
    alpha: float = 0.5,
    min_peer: int = 8,
    wl_depth: int = 3,
    wl_bins: int = 192,
    temperature: float = 0.15,
) -> PilotData:
    """Build fingerprint-grouped role splits and a training-fitted feature map.

    Graph classes determine the training partition and are available for external
    evaluation. Neither WL extraction, transform fitting, nor prototype fitting
    consumes graph classes. The class count comes from the dataset specification.
    """
    if (
        peers < 3
        or min_peer < raw.n_clusters
        or alpha <= 0
        or wl_depth < 0
        or wl_bins < 1
        or temperature <= 0
    ):
        raise ValueError("Invalid peer, partition, WL, or temperature configuration")
    if partition_seed is None:
        partition_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    features, fingerprints = _wl_features(raw.graphs, wl_depth, wl_bins)
    inner_universe = None
    if sealed_outer_seed is not None:
        inner_universe, _, _ = _split_roles(fingerprints, sealed_outer_seed)
    train_ids, audit_ids, test_ids = _split_roles(fingerprints, seed, inner_universe)
    if min(len(audit_ids), len(test_ids)) == 0 or len(train_ids) < peers * min_peer:
        raise ValueError(
            "Fingerprint-grouped split cannot supply the configured training and evaluation roles"
        )
    peer_ids, attempts = _partition_peers(
        train_ids, raw.labels, partition_seed, peers, alpha, min_peer
    )

    document_frequency = np.count_nonzero(features[train_ids], axis=0)
    keep = document_frequency > 0
    inverse_frequency = 1.0 + np.log(
        (len(train_ids) + 1) / (document_frequency[keep] + 1)
    )
    features = features[:, keep] * inverse_frequency
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)

    _, fingerprint_counts = np.unique(fingerprints, return_counts=True)
    return PilotData(
        features,
        raw.labels.copy(),
        train_ids,
        audit_ids,
        test_ids,
        peer_ids,
        fingerprints,
        seed,
        partition_seed,
        attempts,
        int(fingerprint_counts.max()),
        raw.n_clusters,
        wl_depth,
        wl_bins,
        temperature,
    )


def _distances(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    squared = (
        np.einsum("ij,ij->i", features, features)[:, None]
        + np.einsum("ij,ij->i", centers, centers)[None, :]
        - 2.0 * (features @ centers.T)
    )
    return np.maximum(squared, 0.0)


def _initial_centers(
    features: np.ndarray, rng: np.random.Generator, k: int
) -> np.ndarray:
    first = int(rng.integers(len(features)))
    chosen = [first]
    while len(chosen) < k:
        weights = _distances(features, features[chosen]).min(axis=1)
        weights[chosen] = 0
        total = weights.sum()
        if total > 0:
            index = int(rng.choice(len(features), p=weights / total))
        else:
            index = next(
                (first + offset) % len(features)
                for offset in range(1, len(features))
                if (first + offset) % len(features) not in chosen
            )
        chosen.append(index)
    return features[chosen].copy()


def _lloyd(
    features: np.ndarray, start: np.ndarray, epochs: int, step: float
) -> np.ndarray:
    centers = start.copy()
    for _ in range(epochs):
        groups = np.argmin(_distances(features, centers), axis=1)
        for cluster in range(len(centers)):
            members = features[groups == cluster]
            if len(members):
                centers[cluster] += step * (members.mean(axis=0) - centers[cluster])
    return centers


def _align(centers: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(centers) != 2:
        _, order = linear_sum_assignment(_distances(reference, centers))
        return centers[order]
    direct = np.square(centers - reference).sum()
    swapped = np.square(centers[::-1] - reference).sum()
    return centers[::-1] if swapped < direct else centers


def initialize(pilot: PilotData, seed: int) -> np.ndarray:
    """Fit independent local centers and align the bank to peer 0's center order."""
    centers = []
    for peer, ids in enumerate(pilot.peer_train_ids):
        rng = np.random.default_rng(np.random.SeedSequence([seed, peer]))
        features = pilot.features[ids]
        start = _initial_centers(features, rng, pilot.n_clusters)
        centers.append(_lloyd(features, start, epochs=12, step=1.0))
    anchor = centers[0]
    return np.asarray([_align(bank, anchor) for bank in centers])


def propose(
    pilot: PilotData,
    committed: np.ndarray,
    round_seed: int,
    *,
    local_epochs: int = 1,
    local_step: float = 0.25,
    batch_size: int = 32,
    neighbor_weight: float = 0.2,
    topology: str = "ring",
    round_no: int = 1,
) -> np.ndarray:
    """Fit proposals in the fixed coordinate system inherited from committed maps."""
    peers = len(pilot.peer_train_ids)
    expected = (peers, pilot.n_clusters, pilot.features.shape[1])
    if committed.shape != expected:
        raise ValueError(
            f"Expected committed prototype shape {expected}, got {committed.shape}"
        )
    if (
        local_epochs < 1
        or batch_size < 2
        or not 0 < local_step <= 1
        or not 0 <= neighbor_weight <= 1
    ):
        raise ValueError("Invalid local proposal configuration")
    if topology not in {"ring", "exponential", "dynamic"}:
        raise ValueError(f"Unknown topology: {topology}")
    offsets = (1, peers - 1)
    if topology == "exponential":
        offsets = tuple(1 << i for i in range((peers - 1).bit_length()))
    elif topology == "dynamic":
        step = 1 << ((round_no - 1) % (peers - 1).bit_length())
        offsets = tuple(sorted({step, peers - step}))

    proposals = np.empty_like(committed)
    for peer, ids in enumerate(pilot.peer_train_ids):
        rng = np.random.default_rng(np.random.SeedSequence([round_seed, peer]))
        if topology == "ring":
            neighbors = (
                committed[(peer - 1) % peers] + committed[(peer + 1) % peers]
            ) / 2.0
        else:
            neighbors = committed[[(peer + offset) % peers for offset in offsets]].mean(
                axis=0
            )
        centers = (1.0 - neighbor_weight) * committed[
            peer
        ] + neighbor_weight * neighbors
        for _ in range(local_epochs):
            batch = rng.choice(ids, size=min(batch_size, len(ids)), replace=False)
            centers = _lloyd(pilot.features[batch], centers, epochs=1, step=local_step)
        proposals[peer] = _align(centers, committed[peer])
    return proposals


def predict(pilot: PilotData, centers: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return one peer's soft cluster assignments in its aligned coordinates."""
    if centers.shape != (pilot.n_clusters, pilot.features.shape[1]):
        raise ValueError("Expected one aligned prototype bank")
    return predict_features(
        pilot.features[np.asarray(ids, dtype=np.int64)], centers, pilot.temperature
    )


def predict_features(features, centers, temperature):
    logits = -_distances(features, centers) / temperature
    numerators = np.exp(logits - logits.max(axis=1, keepdims=True))
    return numerators / numerators.sum(axis=1, keepdims=True)


def all_peer_predictions(
    pilot: PilotData, centers: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Return aligned soft assignments indexed by peer, graph, and cluster."""
    return np.asarray([predict(pilot, bank, ids) for bank in centers])


def perturb(
    centers: np.ndarray, *, peers: tuple[int, ...] = (0, 4), strength: float = 0.2
) -> np.ndarray:
    """Translate affected banks in their inter-cluster direction without swapping indices."""
    stressed = centers.copy()
    for peer in peers:
        direction = stressed[peer, 1] - stressed[peer, 0]
        stressed[peer] += strength * direction
    return stressed
