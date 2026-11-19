"""Analytic resource proxies used in the generated 20-round study."""

import itertools

GRAPH_SCENARIOS = {
    "PROTEINS": (40, 80, 3),
    "NCI1": (30, 35, 37),
    "ENZYMES": (33, 62, 3),
    "IMDB-BINARY": (20, 96, 16),
    "MUTAG": (18, 20, 7),
}
NO_AUDIT = {"NoVerify", "Unscreened", "LocalOnly"}


def estimate(dataset, method, epochs=10, items=320, audit_size=128, candidates=6):
    nodes, edges, features = GRAPH_SCENARIOS[dataset]
    hidden = 64
    parameters = features * hidden + hidden + hidden * hidden + hidden
    flops = 2 * (
        nodes * features * hidden
        + nodes * hidden * hidden
        + 2 * edges * (features + hidden)
    )
    evaluations = 0 if method in NO_AUDIT else audit_size * candidates
    messages = 0 if method == "LocalOnly" else 8 * 7
    network = messages * (parameters * 4 + (64 * 64 * 4 if evaluations else 0))
    train = items * epochs * 3 * flops / 2e10
    audit = evaluations * flops / 2e10
    communication = network / 125000000 + messages * 0.0005
    memory = (
        0.35
        + (parameters * 4 * 4 + 32 * nodes * hidden * 2 * 4 * 3 + evaluations * 64 * 4)
        / 2**30
    )
    return {
        "parameters": parameters,
        "forward_FLOPs_per_graph": flops,
        "validation_evaluations_per_round": evaluations,
        "directed_messages_per_round": messages,
        "network_bytes_per_round_est": network,
        "train_seconds_per_round_est": train,
        "audit_seconds_per_round_est": audit,
        "communication_seconds_per_round_est": communication,
        "train20_seconds_per_client_est": train * 20,
        "GPU_compute20_seconds_per_client_est": (train + audit) * 20,
        "GPU_compute20_device_seconds_all_clients_est": (train + audit) * 20 * 8,
        "network20_seconds_est": communication * 20,
        "serialized_wall20_seconds_est": (train + audit + communication) * 20,
        "network20_bytes_est": network * 20,
        "peak_allocated_GiB_per_client_est": memory,
        "peak_allocated_MiB_per_client_est": memory * 1024,
    }


def run_cost(identifiers, settings):
    epochs = settings.get("local_E", 10)
    items = settings.get("training_items", 320)
    audit_size = settings.get("audit_size", 128)
    candidates = settings.get("candidate_budget", 6)
    return dict(
        **identifiers,
        rounds=20,
        nominal_E=epochs,
        training_items_assumed=items,
        audit_size_assumed=audit_size,
        candidate_budget_assumed=candidates,
        encoder_label=settings.get("encoder", "GCN64"),
        assumed_cost_backbone="two-layer GCN64 proxy",
        **estimate(
            identifiers["dataset"],
            identifiers["method"],
            epochs,
            items,
            audit_size,
            candidates,
        ),
        actual_GPU_seconds=None,
        actual_peak_GPU_bytes=None,
        measurement_status="ANALYTIC_ESTIMATE_NOT_MEASURED",
    )


def topology_estimates():
    """Independent 4 MB per message communication model."""
    rows = []
    for clients, topology, budget in itertools.product(
        (8, 12, 16, 24, 32),
        ("ring", "full"),
        (6, 16, 28),
    ):
        messages = 2 * clients if topology == "ring" else clients * (clients - 1)
        total_bytes = 4_000_000 * messages
        rows.append(
            {
                "clients": clients,
                "topology": topology,
                "candidate_budget": budget,
                "rounds": 20,
                "nominal_E": 10,
                "directed_messages_per_round": messages,
                "bytes_per_message_assumed": 4_000_000,
                "bytes_per_round_est": total_bytes,
                "total20_bytes_est": 20 * total_bytes,
                "validation_evaluations_per_round": 128 * budget,
                "total20_communication_seconds_est": 20 * total_bytes / 125_000_000
                + 0.01 * messages,
                "actual_GPU_seconds": None,
                "actual_peak_GPU_bytes": None,
                "model_scope": "COMMUNICATION_ONLY_MODEL",
            }
        )
    return rows
