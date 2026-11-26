"""Walk through two VCIA decisions using fixed, hand-authored predictions."""

import argparse
from pathlib import Path

import numpy as np

from .decision import DecisionConfig, design, verify


def _probabilities(first_cluster):
    first = np.asarray(first_cluster, dtype=float)
    return np.stack((first, 1 - first), axis=-1)


def _policy_rows(current, stored, claims, result, policies, verify_indices, config):
    for policy in policies:
        label = "VCIA" if policy == "PCOC" else policy
        candidate = result.selections[policy]
        if candidate is None:
            reason = result.hold_reasons[policy].replace("_", " ")
            yield label, "None", "n/a", "n/a", "HOLD", "None", reason
            continue

        audit = verify(
            current, stored, claims, claims, candidate, verify_indices, config
        )
        mask = str(list(candidate.selected))
        failed = []
        if audit.drift_failed:
            failed.append("drift")
        if audit.assignment_failed:
            failed.append("assignment changes")
        yield (
            label,
            mask,
            f"{audit.upper_ell:.5f}",
            f"{audit.upper_phi:.5f}",
            "COMMIT" if audit.accepted else "HOLD",
            mask if audit.accepted else "None",
            "both bounds pass"
            if audit.accepted
            else "audit fails on " + " and ".join(failed),
        )


def _case(title, summary, old, new, ages, policies):
    stored, current = _probabilities(old), _probabilities(new)
    claims = np.ones(len(old))
    config = DecisionConfig(q=2)
    design_indices = np.zeros(512, dtype=int)
    verify_indices = np.ones(8192, dtype=int)
    result = design(
        current,
        stored,
        claims,
        claims,
        np.any(current != stored, axis=(1, 2)),
        ages,
        design_indices,
        config,
    )
    return (
        title,
        summary,
        (
            (
                f"Policy outcomes (audit limits: drift {config.chi:.2f}, flip rate {config.gamma:.2f})",
                (
                    "Policy",
                    "Proposed",
                    "Drift upper bound",
                    "Flip rate upper bound",
                    "Decision",
                    "Accepted current peers",
                    "Reason",
                ),
                tuple(
                    _policy_rows(
                        current,
                        stored,
                        claims,
                        result,
                        policies,
                        verify_indices,
                        config,
                    )
                ),
            ),
            (
                "Hand-authored inputs (probability of cluster 0)",
                (
                    "Peer",
                    "Age",
                    "Design stored",
                    "Design current",
                    "Audit stored",
                    "Audit current",
                ),
                tuple(
                    (
                        str(peer),
                        str(age),
                        f"{s[0]:.2f}",
                        f"{c[0]:.2f}",
                        f"{s[1]:.2f}",
                        f"{c[1]:.2f}",
                    )
                    for peer, (age, s, c) in enumerate(zip(ages, old, new, strict=True))
                ),
            ),
            (
                f"Individual screening (J upper bound <= {config.kappa:.2f})",
                ("Peer", "Mean J", "J upper bound", "Eligible"),
                tuple(
                    (
                        str(row.identity),
                        f"{row.mean_J:.5f}",
                        f"{row.upper_J:.5f}",
                        "Yes" if row.eligible else "No",
                    )
                    for row in result.version_stats
                ),
            ),
            (
                "All quota masks on the design batch",
                (
                    "Current peers",
                    "Updates H",
                    "Age sum A",
                    "Drift",
                    "Flip rate",
                    "Screened in",
                    "Design tolerances met",
                ),
                tuple(
                    (
                        str(list(c.selected)),
                        str(c.H),
                        f"{c.A:.0f}",
                        f"{c.ell:.5f}",
                        f"{c.phi:.5f}",
                        "Yes" if c.screened else "No",
                        "Yes" if c.feasible else "No",
                    )
                    for c in result.candidates
                ),
            ),
        ),
    )


def walkthrough():
    """Return display tables based on the production screening and audit functions."""
    return (
        _case(
            "Screening can exclude a cancelling pair",
            "Peers 0 and 1 fail the individual screen. Together their changes cancel, "
            "so the Unscreened policy can commit both after audit.",
            [(0.70, 0.70)] * 3,
            [(1.00, 1.00), (0.40, 0.40), (0.70, 0.70)],
            [1, 1, 1],
            ("PCOC", "Unscreened"),
        ),
        _case(
            "Ranking chooses the proposal and audit controls the commit",
            "All three masks pass design and have two detectable updates. VCIA proposes the "
            "largest total age and Drift-First proposes the smallest drift. Each policy audits "
            "its own fixed proposal once. VCIA commits, and Drift-First holds after its audit fails.",
            [(0.65, 0.55)] * 3,
            [(0.68, 0.25), (0.64, 0.85), (0.62, 0.50)],
            [9, 6, 1],
            ("PCOC", "Drift-First"),
        ),
    )


def _print_table(headers, rows):
    widths = tuple(max(map(len, column)) for column in zip(headers, *rows, strict=True))
    for row in (headers, *rows):
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            ).rstrip()
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html", type=Path, help="write a standalone walkthrough HTML file"
    )
    args = parser.parse_args(argv)
    sections = walkthrough()
    if args.html is not None:
        from .demo_html import render_html

        args.html.write_text(render_html(sections), encoding="utf-8")
        return

    print("VCIA decision walkthrough | hand-authored inputs")
    print(
        "Three peers, quota 2, equal claims. Each mask lists the peers using current versions."
    )
    print(
        "Fixed probes: 512 repeats of the design graph and 8,192 repeats of the audit graph."
    )
    print("HOLD retains all stored versions.")
    print(
        "COMMIT denotes acceptance in this in-memory walkthrough."
    )
    print(
        "Each batch repeats one probe to illustrate the calculation. The statistical guarantee requires independent draws."
    )
    print(
        "The walkthrough uses hand-authored probe inputs."
    )
    for title, summary, tables in sections:
        print(f"\n{title}\n{summary}")
        for caption, headers, rows in tables:
            print(f"\n{caption}")
            _print_table(headers, rows)


if __name__ == "__main__":
    main()
