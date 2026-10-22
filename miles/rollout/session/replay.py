"""Session-owned request identity and immutable replay outcomes.

All mutations happen under the containing session's existing lock. Entries are
never evicted while a session is alive, including failures after dispatch.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Protocol

from starlette.responses import JSONResponse, Response

_IGNORED_HEADERS = frozenset(
    {"idempotency-key", "host", "connection", "keep-alive", "content-length", "transfer-encoding"}
)


def request_key(headers: list[tuple[bytes, bytes]]) -> str | None:
    values = [value for name, value in headers if name.lower() == b"idempotency-key"]
    if not values:
        return None
    if len(values) != 1 or not 1 <= len(values[0]) <= 128 or any(c < 33 or c > 126 for c in values[0]):
        raise ValueError(
            "Idempotency-Key requires one header with 1-128 printable ASCII characters without whitespace"
        )
    return values[0].decode("ascii")


def request_fingerprint(
    *, method: str, route: str, query: str, body: bytes, headers: list[tuple[bytes, bytes]]
) -> str:
    # Sort header names, but preserve order and multiplicity within each name.
    normalized: dict[str, list[str]] = {}
    for name, value in headers:
        name_text = name.decode("latin-1").lower()
        if name_text not in _IGNORED_HEADERS:
            normalized.setdefault(name_text, []).append(value.decode("latin-1").strip(" \t"))
    envelope = json.dumps([method.upper(), route, query, sorted(normalized.items())], separators=(",", ":")).encode()
    digest = hashlib.sha256()
    digest.update(len(envelope).to_bytes(8, "big"))
    digest.update(envelope)
    digest.update(body)
    return digest.hexdigest()


@dataclass(frozen=True)
class CompletedOutcome:
    fingerprint: str
    status: int
    body: bytes
    headers: tuple[tuple[bytes, bytes], ...]

    @classmethod
    def capture(cls, fingerprint: str, response: Response) -> "CompletedOutcome":
        return cls(fingerprint, response.status_code, bytes(response.body), tuple(response.raw_headers))

    @property
    def retained_bytes(self) -> int:
        return len(self.body) + sum(len(name) + len(value) for name, value in self.headers)

    def response(self) -> Response:
        response = Response(content=self.body, status_code=self.status)
        response.raw_headers = list(self.headers)
        return response


@dataclass(frozen=True)
class TerminalFailure:
    fingerprint: str
    description: Literal["idempotency_capacity_exceeded", "internal_error", "session_deleted"]

    def response(self) -> Response:
        status = {"idempotency_capacity_exceeded": 503, "internal_error": 500, "session_deleted": 404}[
            self.description
        ]
        return JSONResponse(status_code=status, content={"error": self.description})


ReplayOutcome = CompletedOutcome | TerminalFailure


@dataclass(frozen=True)
class PendingOperation:
    fingerprint: str
    task: asyncio.Task[ReplayOutcome]


@dataclass
class ReplayState:
    entries: dict[str, PendingOperation | ReplayOutcome] = field(default_factory=dict)
    retained_bytes: int = 0

    def pending_tasks(self) -> tuple[asyncio.Task[ReplayOutcome], ...]:
        return tuple(entry.task for entry in self.entries.values() if isinstance(entry, PendingOperation))

    def clear(self) -> None:
        self.entries.clear()
        self.retained_bytes = 0


class ReplaySession(Protocol):
    lock: asyncio.Lock
    closing: bool
    replay: ReplayState
