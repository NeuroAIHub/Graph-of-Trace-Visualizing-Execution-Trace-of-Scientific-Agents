"""End-to-end MCP check: launch server.py as a real stdio MCP server and
connect a client to it. Proves the repo runs as a standalone MCP server that
any MCP host (Claude Desktop, agent CLIs/SDKs) can integrate.

No LLM API key needed: we only initialize the session and list/inspect tools.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent


async def _check():
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO / "server.py")],
        cwd=str(REPO),
        env={**os.environ},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            server_name = init.serverInfo.name

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert "build_trace" in names, f"build_trace not exposed; got {names}"

            bt = next(t for t in tools.tools if t.name == "build_trace")
            props = list((bt.inputSchema or {}).get("properties", {}).keys())

            print(f"server name      : {server_name}")
            print(f"tools exposed    : {names}")
            print(f"build_trace args : {props}")
            return server_name, names, props


def test_mcp_server_lists_tool():
    server_name, names, props = asyncio.run(_check())
    assert server_name == "Graph_of_Trace"
    assert "build_trace" in names
    assert "payload" in props


if __name__ == "__main__":
    asyncio.run(_check())
    print("OK MCP server check passed")
