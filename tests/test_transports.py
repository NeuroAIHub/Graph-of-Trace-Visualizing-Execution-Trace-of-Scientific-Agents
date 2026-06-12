"""End-to-end checks for HTTP (streamable-http) and SSE transports.

Boots server.py as a real subprocess on a free port for each transport, then
connects a matching MCP client and verifies the build_trace tool is exposed.
No LLM API key needed (initialize + list_tools only).
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

REPO = Path(__file__).resolve().parent.parent


def _local_httpx_factory(headers=None, timeout=None, auth=None):
    """Build an httpx client that ignores ambient proxy env vars.

    The CI/dev box may export ALL_PROXY/HTTP_PROXY (e.g. a SOCKS proxy); we are
    talking to a localhost server, so trust_env must be off.
    """
    kwargs = {"trust_env": False, "follow_redirects": True}
    if headers is not None:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(transport: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(REPO / "server.py"),
         "--transport", transport, "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO),
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _wait_until_listening(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"server did not start listening on :{port}")


async def _list_tools_http(port: int) -> list[str]:
    url = f"http://127.0.0.1:{port}/mcp"
    async with streamablehttp_client(url, httpx_client_factory=_local_httpx_factory) as (read, write, _get_sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]


async def _list_tools_sse(port: int) -> list[str]:
    url = f"http://127.0.0.1:{port}/sse"
    async with sse_client(url, httpx_client_factory=_local_httpx_factory) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]


def _run_transport(transport: str, lister) -> list[str]:
    port = _free_port()
    proc = _spawn(transport, port)
    try:
        _wait_until_listening(port)
        return asyncio.run(lister(port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_streamable_http_transport():
    names = _run_transport("streamable-http", _list_tools_http)
    assert "build_trace" in names, names


def test_sse_transport():
    names = _run_transport("sse", _list_tools_sse)
    assert "build_trace" in names, names


if __name__ == "__main__":
    print("HTTP :", _run_transport("streamable-http", _list_tools_http))
    print("SSE  :", _run_transport("sse", _list_tools_sse))
    print("OK HTTP + SSE transports passed")
