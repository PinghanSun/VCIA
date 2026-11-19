"""Paired threshold-window recovery definitions and summaries."""

from collections import defaultdict

import numpy as np

STATUSES = (
    "not_applicable",
    "not_disrupted",
    "recovered",
    "censored",
    "ongoing_disrupted",
)


def recovery(
    ref, branch, method, kind, scale, horizon, threshold=0.5, window=3, correct=False
):
    signed = ref - branch
    offset = float(signed[:4].mean()) if correct else 0.0
    signed = signed - offset
    disruption = float(signed[8:12].mean())
    eligible = disruption > 1 and method != "Frozen"
    start = confirm = None
    gap = np.abs(signed) if scale == "latent" else np.maximum(signed, 0)
    if eligible:
        for index in range(12, horizon - window + 1):
            if (gap[index : index + window] <= threshold).all():
                start, confirm = index + 1, index + window
                break
    apparent = None
    if method == "Frozen":
        status = "not_applicable"
    elif not eligible:
        status = "not_disrupted"
    elif kind == "continuous":
        status = "ongoing_disrupted"
        apparent = confirm
        start = confirm = None
    else:
        status = "recovered" if confirm is not None else "censored"
    return {
        "scale": scale,
        "horizon": horizon,
        "threshold_pp": threshold,
        "consecutive_rounds": window,
        "preattack_corrected": correct,
        "preattack_offset_pp": offset,
        "attack_tail_loss_pp": disruption,
        "eligible": eligible,
        "recovery_status": status,
        "recovery_start": start,
        "recovery_confirm": confirm,
        "wait_to_confirm": None if confirm is None else confirm - 12,
        "first_within_threshold_while_attack_active": apparent,
    }


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in fields)
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items()):
        identifiers = dict(zip(fields, key))
        counts = {
            name: sum(row["recovery_status"] == name for row in group)
            for name in STATUSES
        }
        eligible = sum(row["eligible"] for row in group)
        waits = [
            row["wait_to_confirm"]
            for row in group
            if row["recovery_status"] == "recovered"
        ]
        fraction = (
            counts["recovered"] / eligible
            if eligible and group[0]["branch"] == "stop"
            else None
        )
        result.append(
            dict(
                **identifiers,
                total=len(group),
                eligible=eligible,
                **counts,
                recovery_fraction=fraction,
                recovered_only_median_wait=float(np.median(waits)) if waits else None,
            )
        )
    return result
