"""Idempotent chat-completion recovery over the real session HTTP routes."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from typing import Any

import httpx
import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.core import _chat_client_response
from miles.rollout.session.replay import CompletedOutcome, PendingOperation
from miles.rollout.session.request_args import ClientResponseIntent
from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


class _NoMismatchComparator:
    def compare_sequences(self, _expected: list[int], _actual: list[int]) -> list:
        return []


class _DeterministicTitoTokenizer(TITOTokenizer):
    """A tiny CPU-only renderer that avoids loading a checkpoint tokenizer."""

    def __init__(self) -> None:
        super().__init__(tokenizer=object())

    def create_comparator(self) -> _NoMismatchComparator:
        return _NoMismatchComparator()

    def apply_chat_template(
        self,
        _messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
        template_args: dict[str, Any] | None = None,
    ) -> str | list[int]:
        return [1] if tokenize else "prompt"

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        return list(pretokenized_token_ids)


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[tuple[int, bytes, dict[str, str]]] = []
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.entered_target = 1
        self.entered_count = 0
        self.active = 0
        self.max_active = 0
        self.cancelled = asyncio.Event()
        self.raise_error: Exception | None = None
        self.weight_version = "weights-1"
        self.server: SessionServer | None = None
        self.shared_client: httpx.AsyncClient | None = None
        self.client_open_when_cancelled: bool | None = None

    async def do_proxy(self, request, path: str, *, body: bytes, headers: dict[str, str]) -> dict:
        self.calls.append({"method": request.method, "path": path, "body": json.loads(body), "headers": headers})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.entered is not None:
            self.entered_count += 1
            if self.entered_count >= self.entered_target:
                self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
        except asyncio.CancelledError:
            if self.shared_client is not None:
                self.client_open_when_cancelled = not self.shared_client.is_closed
            self.cancelled.set()
            raise
        finally:
            self.active -= 1
        if self.raise_error is not None:
            raise self.raise_error
        if self.responses:
            status, response_body, response_headers = self.responses.pop(0)
            return {
                "request_body": body,
                "response_body": response_body,
                "status_code": status,
                "headers": response_headers,
            }
        call_number = len(self.calls)
        response = self.completion_response(call_number, self.weight_version)
        return {
            "request_body": body,
            "response_body": json.dumps(response).encode(),
            "status_code": 200,
            "headers": {"content-type": "application/json", "x-upstream-call": str(call_number)},
        }

    @staticmethod
    def completion_response(call_number: int, weight_version: str) -> dict[str, Any]:
        return {
            "id": f"completion-{call_number}",
            "object": "chat.completion",
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"answer-{call_number}"},
                    "finish_reason": "stop",
                    "meta_info": {
                        "weight_version": weight_version,
                        "completion_tokens": 1,
                        "output_token_logprobs": [[-0.25, 100 + call_number]],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@asynccontextmanager
async def _router(monkeypatch: pytest.MonkeyPatch, version: str, **config_overrides: Any):
    from miles.rollout.session import sessions

    monkeypatch.setattr(sessions, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sessions, "get_tito_tokenizer", lambda *_args, **_kwargs: _DeterministicTitoTokenizer())
    monkeypatch.setattr(sessions, "detect_inline_system_support", lambda _template: True)
    v2_hooks = (
        {
            "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
            "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        }
        if version == "v2"
        else {}
    )
    session_config_overrides = {
        "use_session_server": version,
        "hf_checkpoint": "offline-fake-checkpoint",
        "replay_max_entries": 256,
        "replay_max_bytes": 1 << 20,
        **v2_hooks,
    }
    session_config_overrides.update(config_overrides)
    config = make_session_server_config(**session_config_overrides)
    backend = _FakeBackend()
    server = SessionServer(config)
    server.do_proxy = backend.do_proxy
    backend.server = server
    backend.shared_client = server.client
    app = server.app
    core = next(callback.__self__ for callback in app.router.on_shutdown if getattr(callback, "__self__", None))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://session.test") as client:
        yield client, backend, core
    await _shutdown_app(app)


async def _shutdown_app(app):
    async with app.router.lifespan_context(app):
        pass


def _expected_cached_outcome_bytes(call_number: int = 1) -> int:
    upstream_response = _FakeBackend.completion_response(call_number, "weights-1")
    client_response = _chat_client_response(
        {"status_code": 200, "headers": {"content-type": "application/json", "x-upstream-call": str(call_number)}},
        upstream_response,
        ClientResponseIntent(stream=False, logprobs=False),
    )
    return CompletedOutcome.capture("request", client_response).retained_bytes


async def _new_session(client: httpx.AsyncClient) -> str:
    response = await client.post("/sessions", json={})
    assert response.status_code == 200
    return response.json()["session_id"]


def _chat_path(session_id: str) -> str:
    return f"/sessions/{session_id}/v1/chat/completions"


def _initial_payload(content: str = "question") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}]}


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_concurrent_and_completed_duplicates_dispatch_and_commit_once(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, core):
        session_id = await _new_session(client)
        backend.entered = asyncio.Event()
        backend.release = asyncio.Event()
        joined = asyncio.Event()
        replay_event = core._replay_event

        def observe_join(event: str, *args: Any) -> None:
            if event == "joined":
                joined.set()
            replay_event(event, *args)

        core._replay_event = observe_join
        path = _chat_path(session_id)
        headers = {"Idempotency-Key": "request-1"}
        payload = _initial_payload()

        first = asyncio.create_task(client.post(path, json=payload, headers=headers))
        await backend.entered.wait()
        second = asyncio.create_task(client.post(path, json=payload, headers=headers))
        await joined.wait()
        assert len(backend.calls) == 1
        backend.release.set()
        first_response, second_response = await asyncio.gather(first, second)

        assert first_response.status_code == second_response.status_code == 200
        assert first_response.content == second_response.content
        assert {first_response.headers["x-miles-replayed"], second_response.headers["x-miles-replayed"]} == {
            "false",
            "true",
        }
        replay = await client.post(path, json=payload, headers=headers)
        assert replay.status_code == 200
        assert replay.content == first_response.content
        assert replay.headers["x-miles-replayed"] == "true"
        state = core.registry.get_session(session_id)
        assert len(backend.calls) == 1
        assert len(state.records if version == "v1" else state.tree.nodes) == 1
        assert state.replay.pending_tasks() == ()
        cached = state.replay.entries["request-1"]
        assert isinstance(cached, CompletedOutcome)
        with pytest.raises(FrozenInstanceError):
            cached.body = b"mutated"
        fresh_response = cached.response()
        original_body = cached.body
        fresh_response.body = b"mutated response"
        fresh_response.raw_headers.clear()
        assert cached.body == original_body
        assert cached.response().body == original_body


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_independent_sessions_can_overlap_backend_dispatch(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, _core):
        first_session = await _new_session(client)
        second_session = await _new_session(client)
        backend.entered = asyncio.Event()
        backend.entered_target = 2
        backend.release = asyncio.Event()

        first = asyncio.create_task(
            client.post(_chat_path(first_session), json=_initial_payload("one"), headers={"Idempotency-Key": "one"})
        )
        second = asyncio.create_task(
            client.post(_chat_path(second_session), json=_initial_payload("two"), headers={"Idempotency-Key": "two"})
        )
        await backend.entered.wait()
        assert backend.max_active == 2
        backend.release.set()
        responses = await asyncio.gather(first, second)

        assert [response.status_code for response in responses] == [200, 200]
        assert len(backend.calls) == 2


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_pending_entries_consume_entry_capacity_before_dispatch(monkeypatch, version):
    async with _router(monkeypatch, version, replay_max_entries=1) as (client, backend, core):
        session_id = await _new_session(client)
        backend.entered = asyncio.Event()
        backend.release = asyncio.Event()
        path = _chat_path(session_id)
        first = asyncio.create_task(client.post(path, json=_initial_payload(), headers={"Idempotency-Key": "pending"}))
        await backend.entered.wait()

        rejected = await client.post(path, json=_initial_payload("second"), headers={"Idempotency-Key": "other"})
        replay_rejection = await client.post(
            path, json=_initial_payload("second"), headers={"Idempotency-Key": "other"}
        )
        assert rejected.status_code == replay_rejection.status_code == 503
        assert rejected.content == replay_rejection.content
        assert len(backend.calls) == 1
        state = core.registry.get_session(session_id)
        assert isinstance(state.replay.entries["pending"], PendingOperation)
        assert state.replay.pending_tasks() == (state.replay.entries["pending"].task,)

        backend.release.set()
        assert (await first).status_code == 200
        assert state.replay.pending_tasks() == ()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_replay_conflicts_on_query_or_body_and_does_not_rewind_later_state(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        first_payload = _initial_payload()
        first = await client.post(f"{path}?mode=first", json=first_payload, headers={"Idempotency-Key": "old"})
        first_message = first.json()["choices"][0]["message"]
        conflict_query = await client.post(
            f"{path}?mode=second", json=first_payload, headers={"Idempotency-Key": "old"}
        )
        conflict_body = await client.post(path, json=_initial_payload("different"), headers={"Idempotency-Key": "old"})
        assert conflict_query.status_code == conflict_body.status_code == 409
        assert conflict_query.json() == conflict_body.json() == {"error": "idempotency_conflict"}

        later_payload = {"messages": [*first_payload["messages"], first_message, {"role": "user", "content": "later"}]}
        later = await client.post(path, json=later_payload, headers={"Idempotency-Key": "later"})
        assert later.status_code == 200
        assert len(backend.calls) == 2

        old_replay = await client.post(f"{path}?mode=first", json=first_payload, headers={"Idempotency-Key": "old"})
        assert old_replay.status_code == 200
        assert old_replay.content == first.content
        state = core.registry.get_session(session_id)
        assert len(state.records if version == "v1" else state.tree.nodes) == 2
        assert len(backend.calls) == 2


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_fingerprint_preserves_duplicate_header_values_and_ignores_hop_headers(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, _core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        payload = _initial_payload()
        initial_headers = [
            ("Idempotency-Key", "headers"),
            ("X-Trace", "first"),
            ("X-Trace", "second"),
            ("Connection", "keep-alive"),
            ("Keep-Alive", "timeout=5"),
        ]
        original = await client.post(path, json=payload, headers=initial_headers)
        same_fingerprint = await client.post(
            path,
            json=payload,
            headers=[
                ("Idempotency-Key", "headers"),
                ("X-Trace", "first"),
                ("X-Trace", "second"),
                ("Connection", "close"),
                ("Keep-Alive", "timeout=20"),
            ],
        )
        reordered_values = await client.post(
            path,
            json=payload,
            headers=[("Idempotency-Key", "headers"), ("X-Trace", "second"), ("X-Trace", "first")],
        )
        new_key = await client.post(
            path,
            json=payload,
            headers=[("Idempotency-Key", "new-key"), ("X-Trace", "first"), ("X-Trace", "second")],
        )

        assert original.status_code == same_fingerprint.status_code == 200
        assert same_fingerprint.headers["x-miles-replayed"] == "true"
        assert reordered_values.status_code == 409
        assert new_key.status_code == 200
        assert new_key.headers["x-miles-replayed"] == "false"
        assert len(backend.calls) == 2


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_replay_keeps_original_weight_version_after_backend_advances(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        payload = _initial_payload()
        backend.weight_version = "weights-before-update"
        original = await client.post(path, json=payload, headers={"Idempotency-Key": "stable"})
        backend.weight_version = "weights-after-update"
        replay = await client.post(path, json=payload, headers={"Idempotency-Key": "stable"})

        assert original.status_code == replay.status_code == 200
        state = core.registry.get_session(session_id)
        records = state.records if version == "v1" else [node.record for node in state.tree.nodes]
        assert records[0].response["choices"][0]["meta_info"]["weight_version"] == "weights-before-update"
        assert backend.weight_version == "weights-after-update"
        assert replay.content == original.content
        assert len(backend.calls) == len(records) == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize(
    ("kind", "expected_status", "upstream_status"),
    [
        ("session", 400, None),
        ("unexpected", 500, None),
        ("upstream_429", 429, 429),
        ("upstream_502", 502, 502),
        ("upstream_503", 503, 503),
    ],
)
@pytest.mark.asyncio
async def test_errors_are_terminal_and_replayed_without_redispatch(
    monkeypatch, version, kind, expected_status, upstream_status
):
    async with _router(monkeypatch, version) as (client, backend, _core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        payload = _initial_payload()
        if kind == "session":
            payload["input_ids"] = [1]
        elif kind == "unexpected":
            backend.raise_error = RuntimeError("backend bug")
        else:
            backend.responses.append(
                (
                    upstream_status,
                    b'{"error":"busy"}',
                    {"content-type": "application/json", "retry-after": "3"},
                )
            )

        first = await client.post(path, json=payload, headers={"Idempotency-Key": kind})
        second = await client.post(path, json=payload, headers={"Idempotency-Key": kind})

        assert first.status_code == second.status_code == expected_status
        assert first.content == second.content
        assert first.headers.get("retry-after") == second.headers.get("retry-after")
        assert len(backend.calls) == (0 if kind == "session" else 1)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_cancelling_http_waiter_does_not_cancel_owned_generation(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, core):
        session_id = await _new_session(client)
        backend.entered = asyncio.Event()
        backend.release = asyncio.Event()
        request = asyncio.create_task(
            client.post(_chat_path(session_id), json=_initial_payload(), headers={"Idempotency-Key": "waiter"})
        )
        await backend.entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        backend.release.set()
        for _ in range(100):
            state = core.registry.get_session(session_id)
            entries = state.replay.entries
            if entries and not state.replay.pending_tasks():
                break
            await asyncio.sleep(0.01)

        replay = await client.post(
            _chat_path(session_id), json=_initial_payload(), headers={"Idempotency-Key": "waiter"}
        )
        assert replay.status_code == 200
        assert replay.headers["x-miles-replayed"] == "true"
        assert len(backend.calls) == 1
        assert len(state.records if version == "v1" else state.tree.nodes) == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("shutdown_mode", ["delete", "shutdown"])
@pytest.mark.asyncio
async def test_delete_and_shutdown_drain_pending_operations_idempotently(monkeypatch, version, shutdown_mode):
    async with _router(monkeypatch, version) as (client, backend, core):
        session_id = await _new_session(client)
        state = core.registry.get_session(session_id)
        backend.entered = asyncio.Event()
        backend.release = asyncio.Event()
        request = asyncio.create_task(
            client.post(_chat_path(session_id), json=_initial_payload(), headers={"Idempotency-Key": "pending"})
        )
        await backend.entered.wait()

        if shutdown_mode == "delete":
            deleted = await client.delete(f"/sessions/{session_id}")
            assert deleted.status_code == 204
            missing = await client.post(
                _chat_path(session_id), json=_initial_payload(), headers={"Idempotency-Key": "after-delete"}
            )
            assert missing.status_code == 404
            await _shutdown_app(backend.server.app)
            await core.shutdown()
            await core.shutdown()
        else:
            await _shutdown_app(backend.server.app)
            await _shutdown_app(backend.server.app)

        response = await request
        assert response.status_code == 404
        assert len(backend.calls) == 1
        assert session_id not in core.registry.sessions
        assert state.closing is True
        assert state.replay.entries == {}
        assert state.replay.retained_bytes == 0
        assert state.replay.pending_tasks() == ()
        if version == "v1":
            assert state.records == []
        else:
            assert state.tree.nodes == []
        assert backend.cancelled.is_set()
        assert backend.client_open_when_cancelled is True
        assert backend.server.client.is_closed


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_missing_or_closing_session_rejects_keyed_request(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, core):
        missing = await client.post(
            _chat_path("missing-session"), json=_initial_payload(), headers={"Idempotency-Key": "missing"}
        )
        assert missing.status_code == 404
        assert backend.calls == []

        session_id = await _new_session(client)
        state = core.registry.get_session(session_id)
        state.closing = True
        closing = await client.post(
            _chat_path(session_id), json=_initial_payload(), headers={"Idempotency-Key": "closing"}
        )
        assert closing.status_code == 404
        assert backend.calls == []


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ([("Idempotency-Key", "")], 400),
        ([("Idempotency-Key", "has whitespace")], 400),
        ([("Idempotency-Key", "a" * 129)], 400),
        ([("Idempotency-Key", "one"), ("idempotency-key", "two")], 400),
    ],
)
@pytest.mark.asyncio
async def test_http_key_validation_and_stream_and_anthropic_routes(monkeypatch, version, headers, expected_status):
    async with _router(monkeypatch, version) as (client, backend, _core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        invalid = await client.post(path, json=_initial_payload(), headers=headers)
        assert invalid.status_code == expected_status
        assert backend.calls == []

        stream = await client.post(
            path,
            json={**_initial_payload(), "stream": True},
            headers={"Idempotency-Key": "no-stream"},
        )
        assert stream.status_code == 400
        assert backend.calls == []

        anthropic = await client.post(
            f"/sessions/{session_id}/v1/messages",
            json={"model": "fake-model", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Idempotency-Key": "unsupported"},
        )
        assert anthropic.status_code == 400
        assert "unsupported" in anthropic.text.lower()
        assert backend.calls == []


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("limit_kind", ["entries", "bytes"])
@pytest.mark.asyncio
async def test_replay_capacity_failures_are_retained(monkeypatch, version, limit_kind):
    limits = {"replay_max_entries": 1 << 20, "replay_max_bytes": 1 << 20}
    limits["replay_max_entries" if limit_kind == "entries" else "replay_max_bytes"] = 1
    async with _router(monkeypatch, version, **limits) as (client, backend, core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        first = await client.post(path, json=_initial_payload(), headers={"Idempotency-Key": "first"})
        if limit_kind == "entries":
            second = await client.post(path, json=_initial_payload("second"), headers={"Idempotency-Key": "second"})
            replay_second = await client.post(
                path, json=_initial_payload("second"), headers={"Idempotency-Key": "second"}
            )
            assert first.status_code == 200
            assert second.status_code == replay_second.status_code == 503
            assert second.content == replay_second.content
            assert len(backend.calls) == 1
        else:
            replay = await client.post(path, json=_initial_payload(), headers={"Idempotency-Key": "first"})
            assert first.status_code == replay.status_code == 503
            assert first.content == replay.content
            assert first.json() == {"error": "idempotency_capacity_exceeded"}
            assert len(backend.calls) == 1

        state = core.registry.get_session(session_id)
        assert len(state.replay.entries) == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_remaining_byte_budget_counts_exact_cached_responses(monkeypatch, version):
    one_response_bytes = _expected_cached_outcome_bytes(1)
    async with _router(monkeypatch, version, replay_max_bytes=2 * one_response_bytes) as (client, backend, core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        first = await client.post(path, json=_initial_payload("first"), headers={"Idempotency-Key": "first"})
        assert first.status_code == 200
        state = core.registry.get_session(session_id)
        first_outcome = state.replay.entries["first"]
        assert isinstance(first_outcome, CompletedOutcome)
        assert first_outcome.retained_bytes == one_response_bytes
        assert state.replay.retained_bytes == one_response_bytes

        second = await client.post(path, json=_initial_payload("second"), headers={"Idempotency-Key": "second"})
        assert second.status_code == 200
        assert state.replay.retained_bytes == 2 * one_response_bytes

        third = await client.post(path, json=_initial_payload("third"), headers={"Idempotency-Key": "third"})
        third_replay = await client.post(path, json=_initial_payload("third"), headers={"Idempotency-Key": "third"})
        assert third.status_code == third_replay.status_code == 503
        assert third.content == third_replay.content
        assert len(backend.calls) == 2
        assert len(state.replay.entries) == 2
        assert state.replay.pending_tasks() == ()
        assert all(not isinstance(entry, PendingOperation) for entry in state.replay.entries.values())


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.asyncio
async def test_unkeyed_requests_keep_legacy_dispatch_behavior(monkeypatch, version):
    async with _router(monkeypatch, version) as (client, backend, _core):
        session_id = await _new_session(client)
        path = _chat_path(session_id)
        payload = _initial_payload()
        first = await client.post(path, json=payload)
        second = await client.post(path, json=payload)

        assert first.status_code == second.status_code == 200
        assert len(backend.calls) == 2
        assert "x-miles-replayed" not in first.headers
        assert "x-miles-replayed" not in second.headers
