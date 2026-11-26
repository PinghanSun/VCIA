"""Longitudinal graph runs and independent scoring with a frozen recovery rule."""

import argparse
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

import run as base
from learner import ProteinGraph, RawProteins
from long_driver import gauge_round

DATASETS = {
    **base.DATASETS,
    "NCI1": (None, "10e1458f3bd9224f14e6d7627e74dcfd13e48d376d73935e7bd2900590ef1d82"),
}
ARMS = ("VCIA", "J-Rank-Feasible", "Unscreened", "Census-Audit")
T = 60
THRESHOLD, STREAK = .05, 3
RECOVERY = "Mean outer graph binary TV after a whole-bank optimal swap. Disrupted if >0.05 during rounds 11–35. Recovered when <=0.05 for 3 consecutive rounds starting at 36 or later; record the confirmation round and any post-confirmation relapse. Follow-up censored at 60."


def load_nci1(path):
    with zipfile.ZipFile(path) as archive:
        def lines(field):
            return archive.read(f"NCI1/NCI1_{field}.txt").decode("utf-8").splitlines()

        labels = np.unique(np.asarray(lines("graph_labels"), dtype=int), return_inverse=True)[1]
        graph_ids = np.asarray(lines("graph_indicator"), dtype=int) - 1
        node_labels = np.asarray(lines("node_labels"), dtype=int)
        neighbors = [set() for _ in graph_ids]
        for line in lines("A"):
            u, v = (int(item) - 1 for item in line.replace(",", " ").split())
            assert graph_ids[u] == graph_ids[v]
            neighbors[u].add(v)
            neighbors[v].add(u)
    nodes = [[] for _ in labels]
    for index, graph_id in enumerate(graph_ids):
        nodes[graph_id].append(index)
    graphs = []
    for ids in nodes:
        local = {index: position for position, index in enumerate(ids)}
        graphs.append(ProteinGraph(tuple(tuple(sorted(local[j] for j in neighbors[i])) for i in ids),
                                   tuple(int(node_labels[i]) for i in ids)))
    assert len(graphs) == 4110 and len(set(labels)) == 2
    return RawProteins(tuple(graphs), labels, path, base.digest(path))


def data_hashes(data):
    result = {name: base.digest(data / f"{name}.zip") for name in DATASETS}
    assert result == {name: sha for name, (_, sha) in DATASETS.items()}
    return result


def source_hashes():
    return {**base.source_hashes(), **{
        name: base.digest(base.ROOT / name) for name in ("followup.py", "long_driver.py")}}


def pilot_data(data, name, partition):
    path = data / f"{name}.zip"
    loader = DATASETS[name][0] or load_nci1
    return base.prepare_pilot(loader(path), base.INNER_SEED, partition_seed=partition,
                              sealed_outer_seed=base.OUTER_SEED)


def probs(pilot, bank, claims, ids):
    scores = base.all_peer_predictions(pilot, bank, ids)
    return base.aggregate(scores, scores, claims, claims, (), base.CONFIG) if len(bank) > 1 else scores[0]


def execute(data, out):
    contract = base.manifest(data)
    contract.update(frozen_at_utc=base.now(), data_sha256=data_hashes(data), source_sha256=source_hashes(),
                    policies={arm: base.POLICIES[arm] for arm in ARMS}, rounds=T, recovery=RECOVERY,
                    perturbation="Zero-based peers 1,5 swap prototype coordinates and claim 1 in rounds 11–35.",
                    enumeration="See the earlier 30-round experiment for the complete mask enumeration.",
                    outer_evaluation="All sealed per-round scores frozen before external class label scoring.",
                    diagnostics={arm: 30 for arm in base.DIAGNOSTICS},
                    network="Local single-host TCP only.")
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "frozen_protocol.json", contract)
    online = base.Rows(out / "rounds.csv")
    local = base.Rows(out / "diagnostic_rounds.csv")
    splits = base.Rows(out / "splits.csv")
    allocations = base.Rows(out / "partitions.csv")
    index, banks = [], {}
    peer, coordinator = base.connection()
    try:
        for name, partition in itertools.product(DATASETS, base.PARTITIONS):
            pilot = pilot_data(data, name, partition)
            roles = base.roles(pilot)
            ids = roles["outer_test"]
            assert sum(map(len, roles.values())) == len(pilot.labels)
            owners = {}
            for role, gids in roles.items():
                for gid in gids:
                    fp = pilot.wl_fingerprints[gid]
                    assert owners.setdefault(fp, role) == role
                    splits.add(dict(dataset=name, partition=partition, role=role, graph_id=int(gid), fingerprint=fp))
            for client, gids in enumerate(pilot.peer_train_ids):
                for gid in gids:
                    allocations.add(dict(dataset=name, partition=partition, client=client, graph_id=int(gid)))
            print(name, partition, {role: len(gids) for role, gids in roles.items()}, flush=True)
            for init in base.INITS:
                seed = int(np.random.SeedSequence([partition, init]).generate_state(1)[0])
                initial = base.canonicalize(base.initialize(pilot, init))
                claims = np.full(len(initial), .25)
                zeros = np.zeros(len(initial), dtype=int)
                prefix = dict(dataset=name, partition=partition, init=init)
                for arm in base.DIAGNOSTICS:
                    key = dict(**prefix, regime="Clean", arm=arm)
                    for r, bank, examples, seconds in base.diagnostics(pilot, initial, seed, init, arm):
                        local.add(dict(**key, round=r, examples=examples, seconds=seconds))
                    token = f"run_{len(index):04d}"
                    banks[token] = probs(pilot, bank, np.full(len(bank), .25), ids)[None, :, :]
                    index.append(dict(**key, bank=token, rounds=[30]))
                for regime, arm in itertools.product(base.REGIMES, ARMS):
                    key = dict(**prefix, regime=regime, arm=arm)
                    state = base.State(initial.copy(), claims.copy(), zeros.copy(), zeros.copy())
                    sequence = []
                    for r in range(1, T + 1):
                        previous = base.state_id(state.maps, state.claims)
                        state, row, *_ = gauge_round(pilot, state, key, seed, r, peer, coordinator)
                        assert previous == row["stored_hash"]
                        assert row["committed_hash"] == base.state_id(state.maps, state.claims)
                        assert (row["committed_hash"] != previous) == bool(row["committed"])
                        assert row["attempted"] == row["committed"] + row["rejected"] == 1 - row["no_candidate"]
                        online.add(row)
                        sequence.append(probs(pilot, state.maps, state.claims, ids))
                    token = f"run_{len(index):04d}"
                    banks[token] = np.stack(sequence)
                    index.append(dict(**key, bank=token, rounds=list(range(1, T + 1))))
                print(name, partition, init, "completed", flush=True)
    finally:
        peer.close()
        coordinator.close()
        for stream in (online, local, splits, allocations):
            stream.close()
    np.savez_compressed(out / "sealed_outer_probabilities.npz", **banks)
    base.write_json(out / "index.json", index)
    frozen = {p.name: base.digest(p) for p in out.iterdir() if p.is_file()}
    base.write_json(out / "trajectory_freeze.json", dict(frozen_at_utc=base.now(), sha256=frozen))


def coverage(gap):
    peak = float(gap[10:35].max())
    confirmation = next((i for i in range(35 + STREAK, T + 1)
                         if (gap[i - STREAK:i] <= THRESHOLD).all()), None)
    if peak <= THRESHOLD:
        return "not_disrupted", None, None, peak
    if confirmation is None:
        return "censored", None, None, peak
    return "recovered", confirmation, int((gap[confirmation:] > THRESHOLD).any()), peak


def score(data, out):
    def read(name):
        return json.loads((out / name).read_text(encoding="utf8"))

    expected = ("outer_scoring.json", "outer_round_metrics.csv", "recovery.csv", "paired_round_gaps.csv")
    assert not any((out / name).exists() for name in expected)
    protocol, frozen = read("frozen_protocol.json"), read("trajectory_freeze.json")
    assert source_hashes() == protocol["source_sha256"] and data_hashes(data) == protocol["data_sha256"]
    assert all(base.digest(out / name) == sha for name, sha in frozen["sha256"].items())
    started = base.now()
    assert protocol["frozen_at_utc"] < frozen["frozen_at_utc"] < started
    metrics, recovery, gaps = (base.Rows(out / name) for name in expected[1:])
    try:
        with np.load(out / "sealed_outer_probabilities.npz") as banks:
            for (name, partition), records in itertools.groupby(read("index.json"),
                                                               lambda row: (row["dataset"], row["partition"])):
                pilot = pilot_data(data, name, partition)
                ids = base.roles(pilot)["outer_test"]
                y = pilot.labels[ids]
                clean = {}
                for row in records:
                    key = {name: row[name] for name in base.IDENTITY}
                    arm, regime, init = key["arm"], key["regime"], key["init"]
                    seq = banks[row["bank"]]
                    assert seq.shape == (len(row["rounds"]), len(ids), 2)
                    assert np.isfinite(seq).all() and np.allclose(seq.sum(axis=2), 1)
                    values = [base.quality(p, y) for p in seq]
                    for r, values_r, p in zip(row["rounds"], values, seq, strict=True):
                        metrics.add(dict(**key, round=r, outer_graphs=len(ids),
                                         cluster0=int((p.argmax(axis=1) == 0).sum()), **values_r))
                    if arm in base.DIAGNOSTICS:
                        continue
                    if regime == "Clean":
                        clean[(init, arm)] = (seq, values)
                        continue
                    reference, ref_values = clean[(init, arm)]
                    assert np.array_equal(reference[:10], seq[:10])
                    direct = np.mean(np.abs(reference[:, :, 0] - seq[:, :, 0]), axis=1)
                    swapped = np.mean(np.abs(reference[:, :, 0] - seq[:, :, 1]), axis=1)
                    gap = np.minimum(direct, swapped)
                    status, confirmation, relapse, peak = coverage(gap)
                    for r, (distance, current, control) in enumerate(zip(gap, values, ref_values, strict=True), 1):
                        gaps.add(dict(**key, round=r, gap=float(distance),
                                      delta_nmi=current["nmi"] - control["nmi"],
                                      delta_ari=current["ari"] - control["ari"]))
                    recovery.add(dict(**key, status=status, confirmation_round=confirmation,
                                      rounds_after_perturbation=confirmation - 35 if confirmation else None,
                                      relapse=relapse, peak_during_perturbation=peak, final_gap=float(gap[-1]),
                                      final_delta_nmi=values[-1]["nmi"] - ref_values[-1]["nmi"],
                                      final_delta_ari=values[-1]["ari"] - ref_values[-1]["ari"]))
    finally:
        for stream in (metrics, recovery, gaps):
            stream.close()
    base.write_json(out / expected[0], dict(started_at_utc=started, finished_at_utc=base.now(),
                   sha256={name: base.digest(out / name) for name in expected[1:]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "score"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=base.ROOT / "followup_records")
    options = parser.parse_args()
    with threadpool_limits(limits=1):
        {"execute": execute, "score": score}[options.mode](options.data, options.out)
