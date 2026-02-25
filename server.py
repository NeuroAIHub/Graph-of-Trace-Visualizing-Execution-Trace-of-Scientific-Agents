import asyncio
import uvicorn
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


mcp = FastMCP(name="TraceGraph")

mcp.add_tool(logged_tool(tool.build_trace))


def main() -> None:
    logger.info("MCP Server Starting...")
    mcp.run()                    
                                 # Create a new tmux section：tmux new -s <section_name>
if __name__ == "__main__":       ## conda activate mcp_env ;Run server.py by: fastmcp run server.py:mcp --transport http --host 0.0.0.0 --port 8000
    main()                       ### http://host.docker.internal:8000/mcp   URL configured in openhands
                                 