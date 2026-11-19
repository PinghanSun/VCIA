"""Authenticated loopback transport and trusted finality for PCOC rounds.

The coordinator is the trusted relay and sole durable decision writer. Each
replica runs in a spawned process, re-derives the complete state, and verifies
the audit receipt's binding. One replica can also recompute audit evidence.
This module provides a single-host permissioned protocol, with no quorum or
Byzantine fault-tolerance claim. The journal directory has one writer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import multiprocessing as mp
import os
import socket
import struct
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any, Self


class ProtocolError(RuntimeError):
    """A transport, payload, or journal operation failed."""


class DurableCommitError(ProtocolError):
    """The decision is durable and must be recovered before learning resumes."""


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _signature(secret: bytes, value: Any) -> str:
    return hmac.digest(secret, _encoded(value), "sha256").hex()


@dataclass(frozen=True)
class Record:
    identity: int
    version: int
    operator: bytes
    claim: bytes

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not int
            or type(self.version) is not int
            or min(self.identity, self.version) < 0
        ):
            raise ValueError("Record identity and version must be nonnegative integers")
        if (
            not isinstance(self.operator, bytes)
            or not self.operator
            or not isinstance(self.claim, bytes)
            or not self.claim
        ):
            raise ValueError(
                "Operator and claim must be nonempty canonical byte strings"
            )

    def wire(self) -> dict:
        return {
            "identity": self.identity,
            "version": self.version,
            "operator": base64.b64encode(self.operator).decode("ascii"),
            "claim": base64.b64encode(self.claim).decode("ascii"),
        }

    @classmethod
    def from_wire(cls, value: dict) -> Record:
        return cls(
            value["identity"],
            value["version"],
            base64.b64decode(value["operator"], validate=True),
            base64.b64decode(value["claim"], validate=True),
        )


@dataclass(frozen=True)
class Snapshot:
    records: tuple[Record, ...]
    submission_clocks: tuple[int, ...]
    content_clocks: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.records)
        if not isinstance(self.records, tuple) or size < 2:
            raise ValueError(
                "A snapshot requires an ordered tuple of at least two records"
            )
        if tuple(record.identity for record in self.records) != tuple(range(size)):
            raise ValueError("Snapshot records must appear in identity order")
        for clocks in (self.submission_clocks, self.content_clocks):
            if (
                not isinstance(clocks, tuple)
                or len(clocks) != size
                or any(type(clock) is not int or clock < 0 for clock in clocks)
            ):
                raise ValueError(
                    "Snapshot clocks must contain one nonnegative integer per identity"
                )
        for record, submitted, changed in zip(
            self.records, self.submission_clocks, self.content_clocks
        ):
            if record.version != submitted or changed > submitted:
                raise ValueError("Submission/version and content clocks disagree")

    def wire(self) -> dict:
        return {
            "records": [record.wire() for record in self.records],
            "submission_clocks": list(self.submission_clocks),
            "content_clocks": list(self.content_clocks),
        }

    @classmethod
    def from_wire(cls, value: dict) -> Snapshot:
        return cls(
            tuple(Record.from_wire(record) for record in value["records"]),
            tuple(value["submission_clocks"]),
            tuple(value["content_clocks"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.wire())


@dataclass(frozen=True)
class Prepared:
    round_no: int
    digest: str
    snapshot: Snapshot

    @property
    def selected(self) -> tuple[int, ...]:
        return tuple(
            record.identity
            for record in self.snapshot.records
            if record.version == self.round_no
        )


AuditFunction = Callable[[Snapshot, Snapshot, dict, dict], dict]


def _check_proposal(prior: Snapshot, record: Record, round_no: int) -> None:
    if (
        type(round_no) is not int
        or record.version != round_no
        or record.identity >= len(prior.records)
    ):
        raise ValueError(
            "Proposal identity or version is out of scope for the current round"
        )
    old = prior.records[record.identity]
    if (
        old.version >= record.version
        or len(record.operator) != len(old.operator)
        or len(record.claim) != len(old.claim)
    ):
        raise ValueError("Proposal is stale or differs from the canonical record shape")


def _binding(
    prior: Snapshot,
    proposals: tuple[Record, ...],
    round_no: int,
    configuration: dict,
    quota: int,
) -> Prepared:
    identities = tuple(record.identity for record in proposals)
    if len(identities) != quota or tuple(sorted(set(identities))) != identities:
        raise ValueError(
            "Staged records must be a quota-sized ordered set of identities"
        )
    records, submitted, changed = (
        list(prior.records),
        list(prior.submission_clocks),
        list(prior.content_clocks),
    )
    for record in proposals:
        _check_proposal(prior, record, round_no)
        i = record.identity
        if (record.operator, record.claim) != (records[i].operator, records[i].claim):
            changed[i] = round_no
        records[i] = record
        submitted[i] = round_no
    snapshot = Snapshot(tuple(records), tuple(submitted), tuple(changed))
    candidate_digest = _digest(
        {
            "schema": 1,
            "round": round_no,
            "quota": quota,
            "configuration": configuration,
            "prior": prior.digest,
            "candidate": snapshot.digest,
        }
    )
    return Prepared(round_no, candidate_digest, snapshot)


def _check_receipt(prepared: Prepared, receipt: dict) -> None:
    if (
        type(receipt) is not dict
        or receipt.get("accepted") is not True
        or receipt.get("digest") != prepared.digest
    ):
        raise ProtocolError(
            "Finality requires an accepted audit receipt for the fully bound candidate"
        )
    _encoded(receipt)


_HEADER = struct.Struct("!I")
_TAG_BYTES = hashlib.sha256().digest_size
_MAX_FRAME = 128 * 1024 * 1024


@dataclass
class TransportCost:
    tx_bytes: int = 0
    rx_bytes: int = 0
    tx_messages: int = 0
    rx_messages: int = 0
    elapsed_seconds: float = 0.0
    restarts: int = 0


class _Link:
    """Length-framed, replay-resistant, authenticated JSON over a loopback TCP stream."""

    def __init__(
        self,
        stream: socket.socket,
        secret: bytes,
        *,
        coordinator: bool,
        cost: TransportCost,
    ):
        self.stream = stream
        self.secret = secret
        self.role = b"c" if coordinator else b"p"
        self.rx_role = b"p" if coordinator else b"c"
        self.cost = cost
        self.sent = 0
        self.received = 0
        self.stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _tag(self, role: bytes, index: int, header: bytes, body: bytes) -> bytes:
        key = hmac.new(self.secret, role + index.to_bytes(8, "big") + header, "sha256")
        key.update(body)
        return key.digest()

    def _collect(self, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = self.stream.recv(min(size - len(result), 1 << 20))
            if not chunk:
                raise ProtocolError("Peer disconnected mid-frame")
            self.cost.rx_bytes += len(chunk)
            result.extend(chunk)
        return bytes(result)

    def send(self, value: dict) -> None:
        body = _encoded(value)
        if len(body) > _MAX_FRAME:
            raise ProtocolError("Transport message exceeds the configured frame limit")
        header = _HEADER.pack(len(body))
        tag = self._tag(self.role, self.sent, header, body)
        for segment in (header, body, tag):
            view = memoryview(segment)
            while view:
                count = self.stream.send(view)
                if not count:
                    raise ProtocolError("Peer disconnected mid-frame")
                self.cost.tx_bytes += count
                view = view[count:]
        self.sent += 1
        self.cost.tx_messages += 1

    def read(self) -> dict:
        header = self._collect(_HEADER.size)
        (size,) = _HEADER.unpack(header)
        if not 0 < size <= _MAX_FRAME:
            raise ProtocolError("Invalid transport frame size")
        body, tag = self._collect(size), self._collect(_TAG_BYTES)
        if not hmac.compare_digest(
            tag, self._tag(self.rx_role, self.received, header, body)
        ):
            raise ProtocolError("Transport frame authentication failed")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ProtocolError("Expected a transport message object")
        self.received += 1
        self.cost.rx_messages += 1
        return value


def _run_peer(
    address: tuple[str, int],
    identity: int,
    secret: bytes,
    timeout: float,
    audit_function: AuditFunction | None,
) -> None:
    """Replica handler. Committed state remains unchanged until finality arrives."""
    try:
        with socket.create_connection(address, timeout) as stream:
            stream.settimeout(None)
            link = _Link(stream, secret, coordinator=False, cost=TransportCost())
            start = link.read()
            if start["op"] != "sync" or start["identity"] != identity:
                raise ProtocolError("Invalid replica initialization message")
            configuration, quota = start["configuration"], start["quota"]
            prior = Snapshot.from_wire(start["snapshot"])
            link.send({"ok": True, "state": prior.digest})
            pending = None
            receipt = None
            voted = None
            signed_round = max(prior.submission_clocks)
            while True:
                request = link.read()
                op = request["op"]
                if op == "close":
                    return
                try:
                    if op == "open":
                        round_no = request["round"]
                        record = Record.from_wire(request["record"])
                        if (
                            record.identity != identity
                            or request["prior"] != prior.digest
                            or pending is not None
                            or round_no <= signed_round
                        ):
                            raise ProtocolError(
                                "Proposal opening does not extend this replica's current state"
                            )
                        _check_proposal(prior, record, round_no)
                        signed_round = round_no
                        result = {"signature": _signature(secret, request)}
                    elif op == "stage":
                        if pending is not None or request["prior"] != prior.digest:
                            raise ProtocolError(
                                "Stage does not extend the replica's current state"
                            )
                        records = tuple(
                            Record.from_wire(record) for record in request["proposals"]
                        )
                        pending = _binding(
                            prior, records, request["round"], configuration, quota
                        )
                        if pending.digest != request["digest"]:
                            raise ProtocolError(
                                "Candidate digest differs from the independently staged state"
                            )
                        receipt, voted = None, None
                        result = {"digest": pending.digest}
                    elif op == "audit":
                        if (
                            audit_function is None
                            or pending is None
                            or request["digest"] != pending.digest
                            or receipt is not None
                        ):
                            raise ProtocolError(
                                "Unexpected repeat, unbound candidate, or unavailable auditor"
                            )
                        task = request["task"]
                        if tuple(task["selected"]) != pending.selected:
                            raise ProtocolError(
                                "Audit task differs from the identities whose versions were bound"
                            )
                        evidence = audit_function(
                            prior,
                            pending.snapshot,
                            json.loads(_encoded(configuration)),
                            task,
                        )
                        if (
                            type(evidence) is not dict
                            or type(evidence.get("accepted")) is not bool
                        ):
                            raise ProtocolError(
                                "Auditor must return an explicit Boolean acceptance decision"
                            )
                        receipt = {
                            **evidence,
                            "digest": pending.digest,
                            "task_digest": _digest(task),
                        }
                        result = {"receipt": receipt}
                    elif op in ("vote", "finalize"):
                        if (
                            pending is None
                            or request["digest"] != pending.digest
                            or request["round"] != pending.round_no
                        ):
                            raise ProtocolError(
                                "Finality references a candidate other than the staged snapshot"
                            )
                        audited = request["receipt"]
                        _check_receipt(pending, audited)
                        if audit_function is not None and _encoded(audited) != _encoded(
                            receipt
                        ):
                            raise ProtocolError(
                                "Finality receipt differs from the designated auditor's result"
                            )
                        receipt_digest = _digest(audited)
                        if op == "finalize":
                            if voted != receipt_digest:
                                raise ProtocolError(
                                    "Finality differs from the replica's vote"
                                )
                            signed_round = max(signed_round, pending.round_no)
                            prior, pending, receipt, voted = (
                                pending.snapshot,
                                None,
                                None,
                                None,
                            )
                        else:
                            voted = receipt_digest
                        result = (
                            {"state": prior.digest}
                            if op == "finalize"
                            else {"digest": pending.digest}
                        )
                    elif op == "abort":
                        if pending is None or request["digest"] != pending.digest:
                            raise ProtocolError(
                                "Abort references a candidate other than the staged snapshot"
                            )
                        pending, receipt, voted = None, None, None
                        result = {"state": prior.digest}
                    else:
                        raise ProtocolError("Unknown protocol operation")
                except (
                    ValueError,
                    TypeError,
                    RuntimeError,
                    KeyError,
                    ArithmeticError,
                ) as error:
                    link.send({"ok": False, "error": str(error)})
                else:
                    link.send({"ok": True, **result})
    except (OSError, ValueError, ProtocolError, KeyError):
        return


def _write_atomic(path: Path, value: dict) -> None:
    raw = _encoded(value) + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".decision-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _lock_journal(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".writer.lock").open("a+b")
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise ProtocolError(
            f"Cannot acquire exclusive journal writer lock: {root}"
        ) from error
    return stream


class PeerCluster:
    """One replica per peer, with a coordinator-owned durable decision journal.

    Stage reserves one candidate for the round before an audit is requested.
    Call abort when the audit fails. A prepared candidate cannot be retried.
    The next round must read snapshot after commit or resolve DurableCommitError.
    The same root reopens only when genesis and configuration match.
    """

    def __init__(
        self,
        root: str | Path,
        initial: Snapshot,
        configuration: dict,
        *,
        quota: int,
        timeout: float = 15.0,
        audit_function: AuditFunction | None = None,
        auditor: int = 0,
    ):
        if (
            type(configuration) is not dict
            or type(quota) is not int
            or not 1 <= quota <= len(initial.records)
        ):
            raise ValueError("Invalid configuration or replacement quota")
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not 0 <= auditor < len(initial.records)
        ):
            raise ValueError("Invalid replica timeout or auditor identity")
        self.root = Path(root)
        self.initial = initial
        self._configuration = json.loads(_encoded(configuration))
        self.quota = quota
        self.timeout = timeout
        self.audit_function = audit_function
        self.auditor = auditor
        self.snapshot = initial
        self._pending: Prepared | None = None
        self._audit_started = False
        self._receipt: dict | None = None
        self._last_reserved = max(initial.submission_clocks)
        self._links: list[_Link] = []
        self._processes: list[Any] = []
        self._journal_lock = None
        self._entered = False
        self.last_cost = TransportCost()
        self.setup_cost = TransportCost()

    def __enter__(self) -> Self:
        if self._entered:
            raise ProtocolError("Replica cluster context is already active")
        self._journal_lock = _lock_journal(self.root)
        try:
            self._load_journal()
            self._entered = True
            self.recover()
        except BaseException:
            self.__exit__()
            raise
        self.setup_cost = replace(self.last_cost)
        self.last_cost = TransportCost()
        return self

    def __exit__(self, *_: object) -> None:
        try:
            self._close_links()
        finally:
            if self._journal_lock is not None:
                self._journal_lock.close()
                self._journal_lock = None
            self._pending = None
            self._entered = False

    def _load_journal(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for directory in ("commits", "rounds"):
                (self.root / directory).mkdir(exist_ok=True)
            genesis = {
                "schema": 1,
                "initial": self.initial.wire(),
                "configuration": self._configuration,
                "quota": self.quota,
                "auditor": self.auditor if self.audit_function is not None else None,
            }
            path = self.root / "genesis.json"
            if not path.exists():
                _write_atomic(path, genesis)
            if json.loads(path.read_bytes()) != genesis:
                raise ProtocolError(
                    "Existing journal genesis differs from the requested experiment"
                )
            self.snapshot = self.initial
            self._last_reserved = max(self.initial.submission_clocks)
            reservations = {}
            for path in sorted(
                (self.root / "rounds").glob("*.json"), key=lambda p: int(p.stem)
            ):
                record = json.loads(path.read_bytes())
                number = int(path.stem)
                if number != record["round"] or number <= self._last_reserved:
                    raise ProtocolError("Invalid journal round reservation")
                reservations[number] = record
                self._last_reserved = number
            for path in sorted(
                (self.root / "commits").glob("*.json"), key=lambda p: int(p.stem)
            ):
                decision = json.loads(path.read_bytes())
                number = int(path.stem)
                openings = tuple(
                    Record.from_wire(item) for item in decision["proposals"]
                )
                prepared = _binding(
                    self.snapshot, openings, number, self._configuration, self.quota
                )
                expected = {
                    "round": number,
                    "digest": prepared.digest,
                    "prior": self.snapshot.digest,
                }
                if (
                    reservations.get(number) != expected
                    or decision["round"] != number
                    or decision["digest"] != prepared.digest
                    or Snapshot.from_wire(decision["snapshot"]) != prepared.snapshot
                ):
                    raise ProtocolError(
                        "Journal contains an unbound or inconsistent commit"
                    )
                _check_receipt(prepared, decision["receipt"])
                self.snapshot = prepared.snapshot
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ProtocolError(f"Cannot load replica journal: {error}") from error

    def _round_path(self, directory: str, round_no: int) -> Path:
        return self.root / directory / f"{round_no:08d}.json"

    def _fresh_connections(self, deadline: float) -> None:
        context = mp.get_context("spawn")
        for identity in range(len(self.initial.records)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                secret = os.urandom(32)
                worker = context.Process(
                    target=_run_peer,
                    args=(
                        listener.getsockname(),
                        identity,
                        secret,
                        self.timeout,
                        self.audit_function if identity == self.auditor else None,
                    ),
                    name=f"pcoc-peer-{identity}",
                    daemon=True,
                )
                worker.start()
                self._processes.append(worker)
                listener.settimeout(self._remaining(deadline))
                stream, _ = listener.accept()
            link = _Link(stream, secret, coordinator=True, cost=self.last_cost)
            self._links.append(link)
            response = self._request(
                identity,
                {
                    "op": "sync",
                    "identity": identity,
                    "quota": self.quota,
                    "configuration": self._configuration,
                    "snapshot": self.snapshot.wire(),
                },
                deadline,
            )
            if response["state"] != self.snapshot.digest:
                raise ProtocolError(
                    "Replica synchronized to a different journal snapshot"
                )

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Replica protocol deadline expired")
        return remaining

    def _request(self, identity: int, value: dict, deadline: float) -> dict:
        link = self._links[identity]
        link.stream.settimeout(self._remaining(deadline))
        link.send(value)
        link.stream.settimeout(self._remaining(deadline))
        result = link.read()
        if result.get("ok") is not True:
            raise ProtocolError(
                f"Peer {identity} refused {value['op']}: {result.get('error', 'missing status')}"
            )
        return result

    def _all(self, value: dict, field: str, expected: str, deadline: float) -> None:
        for identity in range(len(self.initial.records)):
            if self._request(identity, value, deadline).get(field) != expected:
                raise ProtocolError(f"Peer {identity} disagreed about {field}")

    def _close_links(self) -> None:
        for link in self._links:
            link.stream.close()
        self._links.clear()
        for process in self._processes:
            process.join(timeout=0.05)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
        self._processes.clear()

    def recover(self) -> None:
        """Reconnect fresh replicas to the coordinator's durable snapshot."""
        if not self._entered:
            raise ProtocolError("Use PeerCluster in a context manager")
        self._pending = None
        self._receipt = None
        start = monotonic()
        try:
            self._close_links()
            self._fresh_connections(start + self.timeout)
            self.last_cost.restarts += 1
        except (OSError, ProtocolError, KeyError, ValueError, TypeError) as error:
            self._close_links()
            raise ProtocolError(f"Cannot synchronize replicas: {error}") from error
        finally:
            self.last_cost.elapsed_seconds += monotonic() - start

    def stage(
        self,
        round_no: int,
        current_records: tuple[Record, ...],
        selected: tuple[int, ...],
    ) -> Prepared:
        """Authenticate selected proposals and reserve their complete state for one audit."""
        if (
            not self._entered
            or self._pending is not None
            or type(round_no) is not int
            or round_no <= self._last_reserved
        ):
            raise ProtocolError(
                "The round is stale, or another stage is already active"
            )
        if tuple(record.identity for record in current_records) != tuple(
            range(len(self.initial.records))
        ):
            raise ValueError(
                "Provide one current or unavailable record per identity in order"
            )
        if any(
            type(i) is not int or not 0 <= i < len(current_records) for i in selected
        ):
            raise ValueError("Selected identities are outside the current record pool")
        proposals = tuple(current_records[i] for i in sorted(selected))
        prepared = _binding(
            self.snapshot, proposals, round_no, self._configuration, self.quota
        )
        self.last_cost = TransportCost()
        for link in self._links:
            link.cost = self.last_cost
        if not self._links:
            self.recover()
        start = monotonic()
        prior = self.snapshot.digest
        reservation = {"round": round_no, "digest": prepared.digest, "prior": prior}
        try:
            _write_atomic(self._round_path("rounds", round_no), reservation)
            self._last_reserved = round_no
            for record in proposals:
                opening = {
                    "op": "open",
                    "round": round_no,
                    "prior": prior,
                    "record": record.wire(),
                }
                reply = self._request(record.identity, opening, start + self.timeout)
                owner = self._links[record.identity]
                if not hmac.compare_digest(
                    reply.get("signature", ""), _signature(owner.secret, opening)
                ):
                    raise ProtocolError(
                        f"Peer {record.identity} did not authenticate its proposal"
                    )
            self._all(
                {
                    "op": "stage",
                    **reservation,
                    "proposals": [record.wire() for record in proposals],
                },
                "digest",
                prepared.digest,
                start + self.timeout,
            )
        except (OSError, ProtocolError, KeyError, ValueError, TypeError) as error:
            self._close_links()
            raise ProtocolError(f"Cannot stage round {round_no}: {error}") from error
        finally:
            self.last_cost.elapsed_seconds += monotonic() - start
        self._pending = prepared
        self._audit_started = False
        self._receipt = None
        return prepared

    def _check_pending(self, prepared: Prepared) -> None:
        if self._pending != prepared or not self._links:
            raise ProtocolError("Candidate is no longer staged on the current replicas")

    def audit(self, prepared: Prepared, task: dict) -> dict:
        """Recompute the specified audit in the designated replica on staged bytes."""
        self._check_pending(prepared)
        if self.audit_function is None or self._audit_started or type(task) is not dict:
            raise ProtocolError("Audit is unavailable, repeated, or malformed")
        if "selected" in task and tuple(task["selected"]) != prepared.selected:
            raise ProtocolError("Audit task does not refer to the staged identities")
        task = {**task, "selected": list(prepared.selected)}
        self._audit_started = True
        start = monotonic()
        try:
            reply = self._request(
                self.auditor,
                {"op": "audit", "digest": prepared.digest, "task": task},
                start + self.timeout,
            )
            receipt = reply["receipt"]
            if (
                type(receipt) is not dict
                or receipt.get("digest") != prepared.digest
                or receipt.get("task_digest") != _digest(task)
                or type(receipt.get("accepted")) is not bool
            ):
                raise ProtocolError(
                    "Replica auditor returned an unbound or malformed receipt"
                )
            self._receipt = receipt
            return json.loads(_encoded(receipt))
        except (OSError, ProtocolError, ValueError, KeyError, TypeError) as error:
            self._pending = None
            self._close_links()
            raise ProtocolError(
                f"Auditor could not evaluate round {prepared.round_no}: {error}"
            ) from error
        finally:
            self.last_cost.elapsed_seconds += monotonic() - start

    def abort(self, prepared: Prepared) -> None:
        """Record no state change, discarding staged bytes from every replica."""
        self._check_pending(prepared)
        start = monotonic()
        try:
            self._all(
                {"op": "abort", "digest": prepared.digest},
                "state",
                self.snapshot.digest,
                start + self.timeout,
            )
        except (OSError, ProtocolError, ValueError, KeyError, TypeError) as error:
            self._close_links()
            raise ProtocolError(
                f"Aborted by closing the affected replica sessions: {error}"
            ) from error
        finally:
            self._pending = None
            self._receipt = None
            self.last_cost.elapsed_seconds += monotonic() - start

    def commit(self, prepared: Prepared, audit_transcript: dict) -> Snapshot:
        """Write the final decision after all peers vote, then install it everywhere."""
        self._check_pending(prepared)
        start = monotonic()
        number = prepared.round_no
        message = {
            "round": number,
            "digest": prepared.digest,
            "receipt": audit_transcript,
        }
        try:
            _check_receipt(prepared, audit_transcript)
            if self.audit_function is not None and _encoded(
                audit_transcript
            ) != _encoded(self._receipt):
                raise ProtocolError(
                    "Receipt does not match the independent replica auditor's result"
                )
            self._all(
                {"op": "vote", **message},
                "digest",
                prepared.digest,
                start + self.timeout,
            )
            proposals = [
                record.wire()
                for record in prepared.snapshot.records
                if record.version == number
            ]
            _write_atomic(
                self._round_path("commits", number),
                {
                    **message,
                    "snapshot": prepared.snapshot.wire(),
                    "proposals": proposals,
                },
            )
        except (OSError, ProtocolError, ValueError, KeyError, TypeError) as error:
            self._close_links()
            raise ProtocolError(
                f"No durable commit for round {number}: {error}"
            ) from error
        finally:
            self._pending = None
            self._receipt = None
            self.last_cost.elapsed_seconds += monotonic() - start
        self.snapshot = prepared.snapshot
        start = monotonic()
        try:
            self._all(
                {"op": "finalize", **message},
                "state",
                self.snapshot.digest,
                start + self.timeout,
            )
        except (OSError, ProtocolError, ValueError, KeyError, TypeError) as error:
            self.last_cost.elapsed_seconds += monotonic() - start
            try:
                self.recover()
            except ProtocolError as recovery_error:
                raise DurableCommitError(
                    f"Round {number} committed durably but replicas need recovery: {error}"
                ) from recovery_error
        else:
            self.last_cost.elapsed_seconds += monotonic() - start
        return self.snapshot
