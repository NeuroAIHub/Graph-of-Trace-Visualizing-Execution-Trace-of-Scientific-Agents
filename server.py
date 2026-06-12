import asyncio
import os
from mcp.server.fastmcp import FastMCP
import tool
import logging
import time
import functools
from logging.handlers import RotatingFileHandler

# ------------------ Log config------------------
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)

# auto rotate log file to prevent it from getting too large
file_handler = RotatingFileHandler(
    "mcp_server.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-5s | %(message)s  [pid:%(process)d]'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# console log (info level and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def logged_tool(tool_func):
    """Automatically add detailed logging for each tool"""
    tool_name = tool_func.__name__
    is_async = asyncio.iscoroutinefunction(tool_func)

    @functools.wraps(tool_func)
    async def wrapper(**kwargs):
        start_time = time.time()


        # Safe preview of parameters (avoid printing overly large or sensitive content)
        params_preview = {
            k: (repr(v)[:150] + "..." if len(repr(v)) > 150 else repr(v))
            for k, v in kwargs.items()
        }

        logger.info(f"Tool call starting → {tool_name} | Args: {params_preview} | Category: {'async' if is_async else 'sync'}")

        try:
            if is_async:
                result = await tool_func(**kwargs)
            else:
                result = tool_func(**kwargs)
            
            duration = time.time() - start_time

            
            result_str = repr(result)
            result_preview = result_str[:300] + "..." if len(result_str) > 300 else result_str

            logger.info(
                f"Success ← {tool_name} | Category: {'async' if is_async else 'sync'} | "
                f"Time Comsume: {duration:.3f}s | Result Type: {type(result).__name__} | Result Preview: {result_preview}")
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed ← {tool_name} | Category: {'async' if is_async else 'sync'} | "
                f"Time Comsume: {duration:.3f}s | Error: {str(e)}",
                exc_info=True  # Print full stack trace
            )
            raise

    return wrapper


_VALID_TRANSPORTS = {"stdio", "sse", "streamable-http"}


def _build_server(host: str, port: int) -> FastMCP:
    mcp = FastMCP(name="Graph_of_Trace", host=host, port=port)
    mcp.add_tool(logged_tool(tool.build_trace))
    return mcp


# Module-level instance for hosts that import the app object directly
# (e.g. `server.mcp`). The CLI in main() builds its own with chosen host/port.
mcp = _build_server(os.getenv("GOT_HOST", "127.0.0.1"), int(os.getenv("GOT_PORT", "8000")))


def main() -> None:
    """Run the MCP server.

    Transport is selected via CLI or env (CLI wins):
      - --transport {stdio|sse|streamable-http}   (env: GOT_TRANSPORT, default: stdio)
      - --host HOST                               (env: GOT_HOST, default: 127.0.0.1)
      - --port PORT                               (env: GOT_PORT, default: 8000)

    HTTP and SSE bind to host:port; stdio ignores them.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Graph of Trace MCP server")
    parser.add_argument(
        "--transport",
        choices=sorted(_VALID_TRANSPORTS),
        default=os.getenv("GOT_TRANSPORT", "stdio"),
        help="Transport protocol (default: stdio).",
    )
    parser.add_argument("--host", default=os.getenv("GOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GOT_PORT", "8000")))
    args = parser.parse_args()

    mcp = _build_server(args.host, args.port)

    if args.transport == "stdio":
        logger.info("MCP Server Starting... transport=stdio")
    else:
        logger.info(
            "MCP Server Starting... transport=%s host=%s port=%d",
            args.transport,
            args.host,
            args.port,
        )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
