"""Check the two walkthrough decisions and the checked-in offline view."""

from pathlib import Path

from pcoc.demo import walkthrough
from pcoc.demo_html import render_html


def test_walkthrough_selection_and_commit():
    decisions = [
        [(row[0], row[1], row[4], row[5]) for row in tables[0][2]]
        for _, _, tables in walkthrough()
    ]
    assert decisions == [
        [
            ("VCIA", "None", "HOLD", "None"),
            ("Unscreened", "[0, 1]", "COMMIT", "[0, 1]"),
        ],
        [
            ("VCIA", "[0, 1]", "COMMIT", "[0, 1]"),
            ("Drift-First", "[0, 2]", "HOLD", "None"),
        ],
    ]


def test_offline_walkthrough_matches_generated_view():
    example = (
        Path(__file__).resolve().parents[1] / "examples/admission_walkthrough.html"
    )
    assert example.read_text(encoding="utf-8") == render_html(walkthrough())
