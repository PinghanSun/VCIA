"""Pure protocol contract checks using hand-authored miniature payloads."""

import pytest

from pcoc.transport import ProtocolError, Record, Snapshot, _binding, _check_receipt


def initial_state():
    return Snapshot(
        (Record(0, 0, b"first", b"a"), Record(1, 0, b"other", b"b")), (0, 0), (0, 0)
    )


def test_binding_fixes_prior_configuration_complete_state_and_both_clocks():
    initial = initial_state()
    config = {"preprocessing": "fingerprint-sha256", "alignment": [0, 1]}
    unchanged = (Record(0, 1, b"first", b"a"),)
    changed = (Record(0, 1, b"first", b"b"),)
    repeated = _binding(initial, unchanged, 1, config, 1)
    updated = _binding(initial, changed, 1, config, 1)
    assert repeated.snapshot.submission_clocks == (1, 0)
    assert repeated.snapshot.content_clocks == (0, 0)
    assert updated.snapshot.content_clocks == (1, 0)
    assert updated.snapshot.records[1] == initial.records[1]
    assert Snapshot.from_wire(updated.snapshot.wire()) == updated.snapshot
    assert repeated.digest != updated.digest
    assert (
        _binding(initial, changed, 1, {**config, "alignment": [1, 0]}, 1).digest
        != updated.digest
    )
    other_prior = Snapshot(
        (initial.records[0], Record(1, 0, b"other", b"c")), (0, 0), (0, 0)
    )
    assert _binding(other_prior, changed, 1, config, 1).digest != updated.digest


@pytest.mark.parametrize(
    "proposal",
    [
        Record(0, 0, b"first", b"a"),
        Record(2, 1, b"first", b"a"),
        Record(0, 1, b"unshaped", b"a"),
    ],
)
def test_binding_rejects_invalid_proposal_before_any_audit(proposal):
    with pytest.raises(ValueError):
        _binding(initial_state(), (proposal,), 1, {}, 1)


def test_receipt_must_accept_and_bind_the_candidate():
    prepared = _binding(initial_state(), (Record(0, 1, b"first", b"b"),), 1, {}, 1)
    _check_receipt(prepared, {"accepted": True, "digest": prepared.digest})
    for receipt in (
        {"accepted": False, "digest": prepared.digest},
        {"accepted": 1, "digest": prepared.digest},
        {"accepted": True, "digest": initial_state().digest},
    ):
        with pytest.raises(ProtocolError):
            _check_receipt(prepared, receipt)
