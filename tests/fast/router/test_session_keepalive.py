"""Session server idle-timeout configuration reaches the HTTP transport."""

import asyncio
import argparse
import socket
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from pydantic import ValidationError
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import server
from miles.rollout.session.config import compute_session_server_config
from miles.utils.arguments import get_miles_extra_args_provider


@pytest.mark.parametrize("timeout", [5, 1831])
def test_launch_flag_is_parsed(monkeypatch, timeout):
    monkeypatch.setattr(sys, "argv", ["test", "--rollout-batch-size", "1", "--session-server-keepalive-timeout", str(timeout)])
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args, _ = parser.parse_known_args()
    assert args.session_server_keepalive_timeout == timeout


def test_keepalive_timeout_survives_worker_config_serialization():
    config = make_session_server_config(keepalive_timeout=1831)
    args = SimpleNamespace(**config.model_dump(), miles_router_timeout=config.timeout, session_server_keepalive_timeout=config.keepalive_timeout)
    worker = compute_session_server_config(
        args,
        host=config.host,
        port=config.port,
        instance_id=config.instance_id,
        backend_url=config.backend_url,
    )
    restored = type(config).model_validate_json(worker.model_dump_json())
    assert restored.keepalive_timeout == 1831


@pytest.mark.parametrize("timeout", [0, -1])
def test_invalid_keepalive_timeout_is_rejected(timeout):
    with pytest.raises(ValidationError):
        make_session_server_config(keepalive_timeout=timeout)


def test_server_entrypoint_applies_configured_keepalive(monkeypatch):
    config = make_session_server_config(keepalive_timeout=1831)
    app = object()
    monkeypatch.setattr(server, "SessionServer", lambda _: SimpleNamespace(app=app))
    monkeypatch.setattr(server, "configure_logger_raw", lambda _: None)
    monkeypatch.setattr(server.setproctitle, "setproctitle", lambda _: None)
    run = Mock()
    monkeypatch.setattr(server.uvicorn, "run", run)
    server.run_session_server(config)
    assert run.call_args.args == (app,)
    assert run.call_args.kwargs["timeout_keep_alive"] == 1831


@pytest.mark.asyncio
@pytest.mark.parametrize("keepalive_timeout", [5, 1831])
async def test_pooled_connection_survives_the_old_timeout_boundary(keepalive_timeout):
    config = make_session_server_config(keepalive_timeout=keepalive_timeout)
    app = FastAPI()

    @app.post("/")
    async def reply(request: Request):
        await request.body()
        return {"peer": list(request.client)}

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    http_server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            lifespan="off",
            log_level="warning",
            timeout_keep_alive=config.keepalive_timeout,
        )
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(http_server.serve(sockets=[listener])),
        daemon=True,
    )
    thread.start()
    try:
        async with asyncio.timeout(10):
            while not http_server.started:
                await asyncio.sleep(0.01)

        async def delayed_send(name, info):
            if name == "http11.send_request_headers.started":
                # Recreate the scheduling gap AFTER httpcore chooses the old
                # socket, so it crosses the original server's five-second cap.
                time.sleep(0.6)

        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            url = f"http://127.0.0.1:{port}/"
            first = await client.post(url, json={"turn": 0})
            await asyncio.sleep(4.7)
            if keepalive_timeout == 5:
                with pytest.raises(httpx.TransportError):
                    await client.post(url, json={"turn": 1}, extensions={"trace": delayed_send})
            else:
                second = await client.post(url, json={"turn": 1}, extensions={"trace": delayed_send})
                assert second.status_code == 200
                assert second.json()["peer"] == first.json()["peer"]
    finally:
        http_server.should_exit = True
        await asyncio.to_thread(thread.join, 3)
        listener.close()
        assert not thread.is_alive()
