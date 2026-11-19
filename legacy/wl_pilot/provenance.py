"""Check recorded hashes for either the donor scripts or this standalone port."""

import json
from hashlib import sha256
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
PROTEINS_SOURCES = ("run_pilot.py", "learner.py", "decision.py")
IMDB_SOURCES = (*PROTEINS_SOURCES, "run_imdb_pilot.py")


def source_hashes(names):
    return {
        name: sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in names
    }


def validate_source_hashes(recorded, names):
    local = source_hashes(names)
    donor = json.loads((SOURCE_ROOT / "provenance.json").read_text(encoding="utf-8"))
    original = {name: donor["source_sha256"][name] for name in names}
    if recorded == local:
        return "standalone_port"
    if recorded == original:
        return "archived_donor"
    raise ValueError(
        "Source fingerprints differ from the original pilot and the local port"
    )
